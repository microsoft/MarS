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


def run_simulation(num_rollouts: int, rollouts_path: Path, seed_for_init_state: int) -> None:
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
        )
        for i in range(num_rollouts)
    ]
    if C.debug.enable:
        results = [run_rollout_task(task) for task in tasks]
    else:
        with Pool(processes=16) as pool:
            results = pool.map(try_run_rollout_task, tasks)

    pkl_utils.save_pkl_zstd(results, rollouts_path)
    logging.info(f"Saved {len(results)} rollouts to {rollouts_path}")


def get_trade_infos(exchange: Exchange, symbol: str, start_time: Timestamp, end_time: Timestamp) -> list[TradeInfo]:
    """Get trade infos from TradeInfoState."""
    state = exchange.states()[symbol][TradeInfoState.__name__]
    assert isinstance(state, TradeInfoState)
    trade_infos = state.trade_infos
    trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]
    return trade_infos


def plot_price_curves(rollouts: list[list[TradeInfo]], path: Path) -> None:
    """Plot price curves."""
    if not rollouts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    prices = []
    for i, trade_infos in enumerate(rollouts):
        prices.extend(
            [
                {
                    "Time": x.order.time,
                    "Price": x.lob_snapshot.last_price,
                    "Agent": "Init-Agent" if x.order.agent_id == 0 else "BG-Agent",
                    "Rollout": i,
                }
                for x in trade_infos
                if x.lob_snapshot.last_price > 0
            ]
        )
    # group by 1 minute
    price_data = pd.DataFrame(prices)
    price_data["Minute"] = price_data["Time"].dt.floor("min")
    price_data = price_data.drop(columns=["Time"])
    price_data = price_data.groupby(["Minute", "Rollout", "Agent"]).mean().reset_index()
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # calculate the median and standard deviation for each minute
    price_data = (
        price_data.groupby(["Minute", "Agent"])
        .agg(
            median_price=("Price", "median"),
            std_price=("Price", "std"),
        )
        .reset_index()
    )
    sns.lineplot(
        x="Minute",
        y="median_price",
        data=price_data,
        ax=ax,
    )
    ax.fill_between(
        price_data["Minute"],
        price_data["median_price"] - price_data["std_price"],  # type: ignore
        price_data["median_price"] + price_data["std_price"],  # type: ignore
        alpha=0.2,
        color="orange",
    )

    sns.scatterplot(x="Minute", y="median_price", data=price_data, hue="Agent", ax=ax)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax.set_title("Simulated Rollouts")
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)
    logging.info(f"Saved price curves to {path}")


def visualize_rollouts(rollouts_path: Path) -> None:
    """Visualize rollouts."""
    if not rollouts_path.exists():
        return
    rollouts = pkl_utils.load_pkl_zstd(rollouts_path)
    plot_price_curves(rollouts, rollouts_path.with_suffix(".png"))


if __name__ == "__main__":
    output_dir = Path(C.directory.output_root_dir) / "forecasting-example"
    output_dir.mkdir(parents=True, exist_ok=True)
    num_rollouts = 2
    for seed in range(10):
        for run in range(2):
            rollouts_path = output_dir / f"rollouts-seed{seed}-run{run}.zstd"
            run_simulation(
                num_rollouts=num_rollouts,
                rollouts_path=rollouts_path,
                seed_for_init_state=seed,
            )
            visualize_rollouts(rollouts_path)
