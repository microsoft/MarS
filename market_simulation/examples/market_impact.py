from __future__ import annotations

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import dates
from pandas import Timestamp

from market_simulation.agents.background_agent import BackgroundAgent
from market_simulation.agents.noise_agent import NoiseAgent
from market_simulation.agents.trading_agent import TradingAgent
from market_simulation.conf import C
from market_simulation.rollout.model_client import ModelClient
from market_simulation.states.order_state import Converter, OrderState
from market_simulation.states.trade_info_state import TradeInfoState
from market_simulation.utils import pkl_utils
from mlib.core.env import Env
from mlib.core.event import create_exchange_events
from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction

if TYPE_CHECKING:
    from mlib.core.base_agent import BaseAgent
    from mlib.core.trade_info import TradeInfo


def get_agent_for_init_state(
    symbol: str,
    seed: int,
    start_time: Timestamp,
    end_time: Timestamp,
) -> BaseAgent:
    """Get agent for init state."""
    init_agent = NoiseAgent(
        symbol=symbol,
        init_price=100000,
        interval_seconds=1,
        start_time=start_time,
        end_time=end_time,
        seed=seed,
    )
    return init_agent


class RolloutTask(NamedTuple):
    """Rollout task."""

    rollout_index: int
    symbol: str
    start_time: Timestamp
    init_end_time: Timestamp
    end_time: Timestamp
    seed_for_init_state: int
    include_twap_agent: bool
    twap_agent_target_volume: int


def run_rollout_task(task: RolloutTask) -> list[TradeInfo]:
    """Run a rollout task."""
    exchange_config = create_exchange_config_without_call_auction(
        market_open=task.start_time,
        market_close=task.end_time,
        symbols=[task.symbol],
    )
    exchange = Exchange(exchange_config)

    converter_dir = Path(C.directory.input_root_dir) / C.order_model.converter_dir
    converter = Converter(converter_dir)
    model_client = ModelClient(model_name=C.model_serving.model_name, ip=C.model_serving.ip, port=C.model_serving.port)
    init_agent = get_agent_for_init_state(
        symbol=task.symbol,
        seed=task.seed_for_init_state,
        start_time=task.start_time,
        end_time=task.init_end_time,
    )
    bg_agent = BackgroundAgent(
        symbol=task.symbol,
        converter=converter,
        start_time=task.init_end_time,
        end_time=task.end_time,
        model_client=model_client,
        init_agent=init_agent,
    )
    exchange.register_state(
        OrderState(
            num_max_orders=C.order_model.seq_len,
            num_bins_price_level=converter.price_level.num_bins,
            num_bins_pred_order_volume=converter.pred_order_volume.num_bins,
            num_bins_order_interval=converter.order_interval.num_bins,
            converter=converter,
        )
    )

    exchange.register_state(TradeInfoState())
    env = Env(exchange=exchange, description=f"{task.rollout_index}th rollout task")
    env.register_agent(init_agent)
    env.register_agent(bg_agent)
    if task.include_twap_agent:
        twap_agent = TradingAgent(
            symbol=task.symbol,
            start_time=task.init_end_time,
            target_volume=task.twap_agent_target_volume,
            direction="B",
            max_passive_volume_ratio=0.9,
            aggressive_price_change=0,
            passive_seconds=10,
            idle_seconds=5,
            total_seconds=int((task.end_time - task.init_end_time).total_seconds()),
        )
        env.register_agent(twap_agent)
    env.push_events(create_exchange_events(exchange_config))
    for observation in env.env():
        action = observation.agent.get_action(observation)
        env.step(action)
    trade_infos: list[TradeInfo] = get_trade_infos(exchange, task.symbol, task.start_time, task.end_time)
    logging.info(f"Get {len(trade_infos)} trade infos from {task.rollout_index}th simulation.")
    return trade_infos


def try_run_rollout_task(task: RolloutTask) -> list[TradeInfo]:
    """Try to run a rollout task."""
    try:
        return run_rollout_task(task)
    except Exception as _:
        logging.exception(f"Error in {task.rollout_index}th rollout task.")
        return []


def run_simulation(num_rollouts: int, rollouts_path: Path, seed_for_init_state: int, volume_ratio: float) -> None:
    """Run simulation with noise agent."""
    symbol: str = "000000"
    start_time = Timestamp("2024-01-01 09:30:00")
    init_end_time = Timestamp("2024-01-01 10:00:00")
    end_time = Timestamp("2024-01-01 10:05:00")
    tasks = [
        RolloutTask(
            rollout_index=i,
            symbol=symbol,
            start_time=start_time,
            init_end_time=init_end_time,
            end_time=end_time,
            seed_for_init_state=seed_for_init_state,
            include_twap_agent=False,
            twap_agent_target_volume=0,
        )
        for i in range(num_rollouts)
    ]
    if C.debug.enable:
        results = [run_rollout_task(task) for task in tasks]
    else:
        with Pool(processes=16) as pool:
            results = pool.map(try_run_rollout_task, tasks)

    avg_volume = get_avg_volume(results, init_end_time, end_time)
    target_volume = int(avg_volume * volume_ratio) // 100 * 100
    logging.info(f"Avg volume: {avg_volume}, target volume: {target_volume}")
    trading_tasks = [
        RolloutTask(
            rollout_index=i,
            symbol=symbol,
            start_time=start_time,
            init_end_time=init_end_time,
            end_time=end_time,
            seed_for_init_state=seed_for_init_state,
            include_twap_agent=True,
            twap_agent_target_volume=target_volume,
        )
        for i in range(num_rollouts)
    ]
    if C.debug.enable:
        trading_results = [run_rollout_task(task) for task in trading_tasks]
    else:
        with Pool(processes=16) as pool:
            trading_results = pool.map(try_run_rollout_task, trading_tasks)

    pkl_utils.save_pkl_zstd((tasks + trading_tasks, results + trading_results), rollouts_path)
    logging.info(f"Saved {len(results) * 2} rollouts to {rollouts_path}")


def get_avg_volume(rollouts: list[list[TradeInfo]], start_time: Timestamp, end_time: Timestamp) -> int:
    """Get avg volume from rollouts."""
    volumes: list[int] = []
    for trade_infos in rollouts:
        last_trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]
        transaction_volume: int = 0
        for trade_info in last_trade_infos:
            if trade_info.order.type in ["B", "S"] and trade_info.transactions:
                transaction_volume += sum([t.volume for t in trade_info.transactions])
        volumes.append(transaction_volume)
    avg_volume = int(sum(volumes) / len(volumes))
    logging.info(f"Avg volume: {avg_volume} from {len(volumes)} rollouts.")
    return avg_volume


def get_trade_infos(exchange: Exchange, symbol: str, start_time: Timestamp, end_time: Timestamp) -> list[TradeInfo]:
    """Get trade infos from TradeInfoState."""
    state = exchange.states()[symbol][TradeInfoState.__name__]
    assert isinstance(state, TradeInfoState)
    trade_infos = state.trade_infos
    trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]
    return trade_infos


def plot_price_curves(tasks: list[RolloutTask], rollouts: list[list[TradeInfo]], path: Path) -> None:
    """Plot price curves."""
    if not rollouts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    prices = []
    for i, (task, trade_infos) in enumerate(zip(tasks, rollouts, strict=True)):
        trading_agent_orders: set[int] = set()
        finished_volume: int = 0
        for trade_info in trade_infos:
            if trade_info.lob_snapshot.last_price < 0:
                continue
            if trade_info.order.agent_id == 2:
                assert task.include_twap_agent
                trading_agent_orders.add(trade_info.order.order_id)
            if trade_info.transactions:
                for trans in trade_info.transactions:
                    if trans.type not in ["B", "S"]:
                        continue
                    assert len(trans.buy_id) == 1 and len(trans.sell_id) == 1
                    if trans.buy_id[0] in trading_agent_orders or trans.sell_id[0] in trading_agent_orders:
                        finished_volume += trans.volume
            prices.append(
                {
                    "Time": trade_info.order.time,
                    "Price": trade_info.lob_snapshot.last_price,
                    "Rollout": i,
                    "TradingAgent": "w/ Twap-Buy" if task.include_twap_agent else "w/o Twap-Buy",
                    "IsSimulation": trade_info.order.time >= task.init_end_time,
                    "TargetVolume": task.twap_agent_target_volume,
                    "FinishedVolume": finished_volume,
                }
            )
    # group by 1 minute
    price_data = pd.DataFrame(prices)
    price_data["Minute"] = price_data["Time"].dt.floor("T")
    price_data = price_data.drop(columns=["Time"])
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax = axes[0]
    finished_volume_data = (
        price_data[price_data["TradingAgent"] == "w/ Twap-Buy"]
        .groupby("Minute")
        .agg(
            FinishedVolume=("FinishedVolume", "mean"),
        )
        .reset_index()
    )
    sns.lineplot(x="Minute", y="FinishedVolume", data=finished_volume_data, ax=ax)

    ax.axhline(y=price_data["TargetVolume"].max(), color="orange", linestyle="--")
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_title("Fulfillment")

    ax = axes[1]
    # calculate the median and standard deviation for each minute
    price_data = price_data[(price_data["TradingAgent"] == "w/o Twap-Buy") | (price_data["IsSimulation"])]
    price_data = price_data.groupby(["Minute", "TradingAgent", "Rollout"]).mean().reset_index()
    price_data = (
        price_data.groupby(["Minute", "TradingAgent"])
        .agg(
            median_price=("Price", "median"),
            std_price=("Price", "std"),
        )
        .reset_index()
    )
    price_data = price_data.sort_values(by=["Minute"])
    logging.info(f"Price data: \n{price_data}")

    sns.lineplot(x="Minute", y="median_price", data=price_data, ax=ax, hue="TradingAgent", style="TradingAgent", markers=True)
    data_no_trading = price_data[price_data["TradingAgent"] == "w/o Twap-Buy"]
    data_trading = price_data[price_data["TradingAgent"] == "w/ Twap-Buy"]

    ax.fill_between(
        data_trading["Minute"],
        data_trading["median_price"] - data_trading["std_price"],  # type: ignore
        data_trading["median_price"] + data_trading["std_price"],  # type: ignore
        alpha=0.2,
        color="orange",
    )

    ax.fill_between(
        data_no_trading["Minute"],
        data_no_trading["median_price"] - data_no_trading["std_price"],  # type: ignore
        data_no_trading["median_price"] + data_no_trading["std_price"],  # type: ignore
        alpha=0.2,
        color="blue",
    )

    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_title("Simulated Rollouts")
    ax.legend(title="Agent Type")
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)
    logging.info(f"Saved price curves to {path}")


def visualize_rollouts(rollouts_path: Path) -> None:
    """Visualize rollouts."""
    if not rollouts_path.exists():
        return
    tasks, rollouts = pkl_utils.load_pkl_zstd(rollouts_path)
    plot_price_curves(tasks, rollouts, rollouts_path.with_suffix(".png"))


if __name__ == "__main__":
    output_dir = Path(C.directory.output_root_dir) / "market-impact-example"
    output_dir.mkdir(parents=True, exist_ok=True)
    num_rollouts = 16
    for seed in range(10):
        for volume_ratio in [0.1, 0.3, 0.5]:
            rollouts_path = output_dir / f"rollouts-seed{seed}-volume_ratio{volume_ratio}.zstd"
            run_simulation(
                num_rollouts=num_rollouts,
                rollouts_path=rollouts_path,
                seed_for_init_state=seed,
                volume_ratio=volume_ratio,
            )
            visualize_rollouts(rollouts_path)
