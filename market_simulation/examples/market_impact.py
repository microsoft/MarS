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


def create_initialization_agent(
    symbol: str,
    seed: int,
    start_time: Timestamp,
    end_time: Timestamp,
) -> BaseAgent:
    """Create an initialization agent for the market simulation.

    Creates a noise agent that establishes initial market conditions with
    specified parameters before other agents take over.

    Args:
        symbol: Market symbol identifier
        seed: Random seed for reproducibility
        start_time: Timestamp when the agent starts operating
        end_time: Timestamp when the agent stops operating

    Returns:
        BaseAgent: Configured noise agent for initialization phase
    """
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
    """Configuration parameters for a single market simulation rollout.

    Attributes:
        rollout_index: Unique identifier for this simulation run
        symbol: Market symbol identifier
        start_time: When the simulation begins
        init_end_time: When initialization phase ends and trading begins
        end_time: When the simulation ends
        seed_for_init_state: Random seed for initialization phase
        include_twap_agent: Whether to include TWAP trading agent in simulation
        twap_agent_target_volume: Target trading volume for TWAP agent
    """

    rollout_index: int
    symbol: str
    start_time: Timestamp
    init_end_time: Timestamp
    end_time: Timestamp
    seed_for_init_state: int
    include_twap_agent: bool
    twap_agent_target_volume: int


def execute_single_simulation(task: RolloutTask) -> list[TradeInfo]:
    """Execute a single market simulation according to the provided task parameters.

    Sets up the market exchange, agents (including optional TWAP agent), and simulation
    environment, then runs the simulation from start to end time.

    Args:
        task: Configuration parameters for the simulation run

    Returns:
        list[TradeInfo]: Collection of trade information generated during simulation
    """
    exchange_config = create_exchange_config_without_call_auction(
        market_open=task.start_time,
        market_close=task.end_time,
        symbols=[task.symbol],
    )
    exchange = Exchange(exchange_config)

    # Set up the converter and model client for the background agent
    converter_dir = Path(C.directory.input_root_dir) / C.order_model.converter_dir
    converter = Converter(converter_dir)
    model_client = ModelClient(model_name=C.model_serving.model_name, ip=C.model_serving.ip, port=C.model_serving.port)

    # Create agents for different simulation phases
    init_agent = create_initialization_agent(
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

    # Register simulation states to track orders and trades
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

    # Configure simulation environment
    env = Env(exchange=exchange, description=f"{task.rollout_index}th rollout task")
    env.register_agent(init_agent)
    env.register_agent(bg_agent)

    # Add TWAP agent if specified in task configuration
    if task.include_twap_agent:
        twap_agent = TradingAgent(
            symbol=task.symbol,
            start_time=task.init_end_time,
            target_volume=task.twap_agent_target_volume,
            direction="B",  # Buy direction
            max_passive_volume_ratio=0.9,
            aggressive_price_change=0,
            passive_seconds=10,
            idle_seconds=5,
            total_seconds=int((task.end_time - task.init_end_time).total_seconds()),
        )
        env.register_agent(twap_agent)

    # Initialize and run the simulation
    env.push_events(create_exchange_events(exchange_config))
    for observation in env.env():
        action = observation.agent.get_action(observation)
        env.step(action)

    # Extract and return trade information
    trade_infos: list[TradeInfo] = extract_trade_information(exchange, task.symbol, task.start_time, task.end_time)
    logging.info(f"Got {len(trade_infos)} trade infos from {task.rollout_index}th simulation.")
    return trade_infos


def execute_simulation_with_error_handling(task: RolloutTask) -> list[TradeInfo]:
    """Execute a simulation with error handling to prevent process crashes.

    Wraps the main simulation execution function with exception handling to ensure
    the overall batch process continues even if individual simulations fail.

    Args:
        task: Configuration parameters for the simulation run

    Returns:
        list[TradeInfo]: Collection of trade information or empty list on error
    """
    try:
        return execute_single_simulation(task)
    except Exception as _:
        logging.exception(f"Error in {task.rollout_index}th rollout task.")
        return []


def run_simulation(
    symbol: str,
    start_time: Timestamp,
    init_end_time: Timestamp,
    end_time: Timestamp,
    num_rollouts: int,
    rollouts_path: Path,
    seed_for_init_state: int,
    volume_ratio: float,
) -> None:
    """Run market impact simulations with and without trading agents.

    This function executes two sets of simulations:
    1. Baseline simulations without TWAP agent
    2. Market impact simulations with TWAP agent targeting a volume
       determined by the volume_ratio parameter

    Args:
        symbol: Market symbol identifier
        start_time: When the simulation begins
        init_end_time: When initialization phase ends
        end_time: When the simulation ends
        num_rollouts: Number of simulation runs in each set
        rollouts_path: Path where simulation results will be saved
        seed_for_init_state: Random seed for initialization phase
        volume_ratio: Target volume for TWAP agent as ratio of average volume

    Returns:
        None
    """
    # Create and run baseline simulations without TWAP agent
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

    # Execute tasks sequentially in debug mode or in parallel otherwise
    if C.debug.enable:
        results = [execute_single_simulation(task) for task in tasks]
    else:
        with Pool(processes=16) as pool:
            results = pool.map(execute_simulation_with_error_handling, tasks)

    # Calculate target volume for TWAP agent based on average volume
    avg_volume = calculate_average_volume(results, init_end_time, end_time)
    target_volume = int(avg_volume * volume_ratio) // 100 * 100  # Round to nearest 100
    logging.info(f"Average volume: {avg_volume}, target volume: {target_volume}")

    # Create and run simulations with TWAP agent
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

    # Execute TWAP agent simulations
    if C.debug.enable:
        trading_results = [execute_single_simulation(task) for task in trading_tasks]
    else:
        with Pool(processes=16) as pool:
            trading_results = pool.map(execute_simulation_with_error_handling, trading_tasks)

    # Save all simulation results
    pkl_utils.save_pkl_zstd((tasks + trading_tasks, results + trading_results), rollouts_path)
    logging.info(f"Saved {len(results) * 2} rollouts to {rollouts_path}")


def calculate_average_volume(rollouts: list[list[TradeInfo]], start_time: Timestamp, end_time: Timestamp) -> int:
    """Calculate the average trading volume from simulation rollouts.

    This is used to determine the target volume for TWAP agent in market impact simulations.

    Args:
        rollouts: List of lists containing TradeInfo objects from multiple simulation runs
        start_time: Beginning of time range for calculation
        end_time: End of time range for calculation

    Returns:
        int: Average trading volume across all rollouts
    """
    volumes: list[int] = []

    for trade_infos in rollouts:
        # Filter trades to the specified time range
        filtered_trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]

        # Calculate total transaction volume for this rollout
        transaction_volume: int = 0
        for trade_info in filtered_trade_infos:
            if trade_info.order.type in ["B", "S"] and trade_info.transactions:
                transaction_volume += sum([t.volume for t in trade_info.transactions])

        volumes.append(transaction_volume)

    # Calculate and return the average volume across all rollouts
    avg_volume = int(sum(volumes) / len(volumes)) if volumes else 0
    logging.info(f"Average volume: {avg_volume} from {len(volumes)} rollouts.")
    return avg_volume


def extract_trade_information(exchange: Exchange, symbol: str, start_time: Timestamp, end_time: Timestamp) -> list[TradeInfo]:
    """Extract trade information from a completed simulation.

    Retrieves trade data from the exchange's TradeInfoState and filters it
    to the specified time range.

    Args:
        exchange: The exchange instance containing simulation states
        symbol: Market symbol to extract data for
        start_time: Beginning of time range for filtering
        end_time: End of time range for filtering

    Returns:
        list[TradeInfo]: Filtered trade information records
    """
    state = exchange.states()[symbol][TradeInfoState.__name__]
    assert isinstance(state, TradeInfoState)
    trade_infos = state.trade_infos
    trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]
    return trade_infos


def visualize_market_impact(tasks: list[RolloutTask], rollouts: list[list[TradeInfo]], path: Path) -> None:
    """Visualize market impact of trading agent on price and volume execution.

    Creates a two-panel figure showing:
    1. TWAP agent's volume execution over time
    2. Price comparison between simulations with and without TWAP agent

    Args:
        tasks: List of task configurations for each simulation run
        rollouts: List of lists containing TradeInfo objects from simulations
        path: Location to save the generated visualization

    Returns:
        None
    """
    if not rollouts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract and process price and volume data from simulation results
    prices = []
    for i, (task, trade_infos) in enumerate(zip(tasks, rollouts, strict=True)):
        # Track orders from trading agent and their fulfilled volume
        trading_agent_orders: set[int] = set()
        finished_volume: int = 0

        for trade_info in trade_infos:
            # Skip invalid price entries
            if trade_info.lob_snapshot.last_price < 0:
                continue

            # Identify trading agent orders (agent_id 2)
            if trade_info.order.agent_id == 2:
                assert task.include_twap_agent
                trading_agent_orders.add(trade_info.order.order_id)

            # Calculate fulfilled volume for trading agent
            if trade_info.transactions:
                for trans in trade_info.transactions:
                    if trans.type not in ["B", "S"]:
                        continue
                    assert len(trans.buy_id) == 1 and len(trans.sell_id) == 1
                    # Count volume if trading agent is involved
                    if trans.buy_id[0] in trading_agent_orders or trans.sell_id[0] in trading_agent_orders:
                        finished_volume += trans.volume

            # Record price data with metadata
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

    # Prepare data for visualization
    price_data = pd.DataFrame(prices)
    price_data["Minute"] = price_data["Time"].dt.floor("T")
    price_data = price_data.drop(columns=["Time"])

    # Configure visualization style and create figure
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Panel 1: Plot volume execution for TWAP agent
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

    # Add target volume line reference
    ax.axhline(y=price_data["TargetVolume"].max(), color="orange", linestyle="--")
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_title("TWAP Volume Fulfillment")

    # Panel 2: Plot price impact comparison
    ax = axes[1]

    # Filter data to relevant simulation period and calculate statistics
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

    # Create line plot with confidence bands for both scenarios
    sns.lineplot(x="Minute", y="median_price", data=price_data, ax=ax, hue="TradingAgent", style="TradingAgent", markers=True)

    # Add confidence bands (standard deviation) for both scenarios
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

    # Format time axis and add labels
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_title("Price Impact of Trading")
    ax.legend(title="Scenario")

    # Save the visualization
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)
    logging.info(f"Saved market impact visualization to {path}")


def visualize_rollouts(rollouts_path: Path) -> None:
    """Load simulation results and create market impact visualizations.

    This function loads serialized rollout data from a file and passes it
    to the visualization function. It ensures the file exists before proceeding.

    Args:
        rollouts_path: Path to the serialized rollout data file

    Returns:
        None
    """
    if not rollouts_path.exists():
        return
    tasks, rollouts = pkl_utils.load_pkl_zstd(rollouts_path)
    visualize_market_impact(tasks, rollouts, rollouts_path.with_suffix(".png"))


if __name__ == "__main__":
    # Set up output directory for simulation results
    output_dir = Path(C.directory.output_root_dir) / "market-impact-example"
    output_dir.mkdir(parents=True, exist_ok=True)
    num_rollouts = 16

    # Run multiple simulations with different seeds and volume ratios
    for seed in range(10):
        for volume_ratio in [0.1, 0.3, 0.5]:
            rollouts_path = output_dir / f"rollouts-seed{seed}-volume_ratio{volume_ratio}.zstd"
            run_simulation(
                symbol="000000",
                start_time=Timestamp("2024-01-01 09:30:00"),
                init_end_time=Timestamp("2024-01-01 10:00:00"),
                end_time=Timestamp("2024-01-01 10:05:00"),
                num_rollouts=num_rollouts,
                rollouts_path=rollouts_path,
                seed_for_init_state=seed,
                volume_ratio=volume_ratio,
            )
            visualize_rollouts(rollouts_path)
