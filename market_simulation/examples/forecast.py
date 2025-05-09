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


def create_initialization_agent(
    symbol: str,
    seed: int,
    start_time: Timestamp,
    end_time: Timestamp,
) -> BaseAgent:
    """Create an initialization agent for market simulation.

    Creates a noise agent that establishes initial market conditions before
    the background agent takes over.

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
        init_end_time: When initialization phase ends and background agent takes over
        end_time: When the simulation ends
        seed_for_init_state: Random seed for initialization phase
    """

    rollout_index: int
    symbol: str
    start_time: Timestamp
    init_end_time: Timestamp
    end_time: Timestamp
    seed_for_init_state: int


def execute_single_simulation(task: RolloutTask) -> list[TradeInfo]:
    """Execute a single market simulation according to the provided task parameters.

    Sets up the market exchange, agents, and simulation environment, then runs
    the simulation from start to end time.

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

    # Register simulation states
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

    # Configure and run the simulation environment
    env = Env(exchange=exchange, description=f"{task.rollout_index}th rollout task")
    env.register_agent(init_agent)
    env.register_agent(bg_agent)
    env.push_events(create_exchange_events(exchange_config))

    # Execute simulation steps
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
) -> None:
    """Run multiple market simulations and save the results.

    Creates and executes multiple simulation tasks, potentially in parallel,
    and saves the aggregated results to a compressed file.

    Args:
        symbol: Market symbol identifier
        start_time: When the simulation begins
        init_end_time: When initialization phase ends
        end_time: When the simulation ends
        num_rollouts: Number of simulation repetitions to run
        rollouts_path: Path where simulation results will be saved
        seed_for_init_state: Random seed for initialization phase

    Returns:
        None
    """
    # Create task configurations for each rollout
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

    # Execute tasks sequentially in debug mode or in parallel otherwise
    if C.debug.enable:
        results = [execute_single_simulation(task) for task in tasks]
    else:
        with Pool(processes=16) as pool:
            results = pool.map(execute_simulation_with_error_handling, tasks)

    # Save simulation results
    pkl_utils.save_pkl_zstd(results, rollouts_path)
    logging.info(f"Saved {len(results)} rollouts to {rollouts_path}")


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


def visualize_price_data(rollouts: list[list[TradeInfo]], path: Path) -> None:
    """Visualize price data from simulation rollouts.

    Creates a plot showing price trends across multiple simulation runs,
    with median values and uncertainty bands. The visualization includes
    information about agent types and saves the result as an image.

    Args:
        rollouts: List of lists containing TradeInfo objects from multiple simulation runs
        path: Location to save the generated visualization

    Returns:
        None
    """
    if not rollouts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract and format price data from simulation results
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

    # Aggregate data by minute for temporal analysis
    price_data = pd.DataFrame(prices)
    price_data["Minute"] = price_data["Time"].dt.floor("min")
    price_data = price_data.drop(columns=["Time"])
    price_data = price_data.groupby(["Minute", "Rollout", "Agent"]).mean().reset_index()

    # Configure visualization style
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Calculate statistical aggregates for visualization
    price_data = (
        price_data.groupby(["Minute", "Agent"])
        .agg(
            median_price=("Price", "median"),
            std_price=("Price", "std"),
        )
        .reset_index()
    )

    # Create line plot of median prices
    sns.lineplot(
        x="Minute",
        y="median_price",
        data=price_data,
        ax=ax,
    )

    # Add standard deviation bands for uncertainty visualization
    ax.fill_between(
        price_data["Minute"],
        price_data["median_price"] - price_data["std_price"],  # type: ignore
        price_data["median_price"] + price_data["std_price"],  # type: ignore
        alpha=0.2,
        color="orange",
    )

    # Add scatter points to show agent-specific data
    sns.scatterplot(x="Minute", y="median_price", data=price_data, hue="Agent", ax=ax)

    # Format x-axis to show time properly
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax.set_title("Simulated Rollouts")
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)
    logging.info(f"Saved price curves to {path}")


def visualize_rollouts(rollouts_path: Path) -> None:
    """Load simulation results and create visualizations.

    This function loads serialized rollout data from a file and passes it
    to the visualization function. It ensures the file exists before proceeding.

    Args:
        rollouts_path: Path to the serialized rollout data file

    Returns:
        None
    """
    if not rollouts_path.exists():
        return
    rollouts = pkl_utils.load_pkl_zstd(rollouts_path)
    visualize_price_data(rollouts, rollouts_path.with_suffix(".png"))


if __name__ == "__main__":
    # Set up output directory for simulation results
    output_dir = Path(C.directory.output_root_dir) / "forecasting-example"
    output_dir.mkdir(parents=True, exist_ok=True)
    num_rollouts = 2

    # Run multiple simulations with different seeds and configurations
    for seed in range(10):
        for run in range(2):
            rollouts_path = output_dir / f"rollouts-seed{seed}-run{run}.zstd"
            run_simulation(
                symbol="000000",
                start_time=Timestamp("2024-01-01 09:30:00"),
                init_end_time=Timestamp("2024-01-01 10:00:00"),
                end_time=Timestamp("2024-01-01 10:05:00"),
                num_rollouts=num_rollouts,
                rollouts_path=rollouts_path,
                seed_for_init_state=seed,
            )
            visualize_rollouts(rollouts_path)
