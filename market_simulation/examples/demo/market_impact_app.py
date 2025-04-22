import time as py_time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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
from mlib.core.trade_info import TradeInfo


@dataclass
class RolloutTask:
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


def create_initialization_agent(
    symbol: str,
    seed: int,
    start_time: Timestamp,
    end_time: Timestamp,
) -> NoiseAgent:
    """Create an initialization agent for the market simulation.

    Creates a noise agent that establishes initial market conditions with
    specified parameters before other agents take over.

    Args:
        symbol: Market symbol identifier
        seed: Random seed for reproducibility
        start_time: Timestamp when the agent starts operating
        end_time: Timestamp when the agent stops operating

    Returns:
        NoiseAgent: Configured noise agent for initialization phase
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

    converter_dir = Path(C.directory.input_root_dir) / C.order_model.converter_dir
    converter = Converter(converter_dir)
    model_client = ModelClient(model_name=C.model_serving.model_name, ip=C.model_serving.ip, port=C.model_serving.port)

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

    trade_infos: list[TradeInfo] = extract_trade_information(exchange, task.symbol, task.start_time, task.end_time)
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
    except Exception as e:
        st.error(f"Error in {task.rollout_index}th rollout task: {e!s}")
        return []


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
        filtered_trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]

        transaction_volume: int = 0
        for trade_info in filtered_trade_infos:
            if trade_info.order.type in ["B", "S"] and trade_info.transactions:
                transaction_volume += sum([t.volume for t in trade_info.transactions])

        volumes.append(transaction_volume)

    avg_volume = int(sum(volumes) / len(volumes)) if volumes else 0
    return avg_volume


def create_price_chart(  # noqa: PLR0915
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame | None = None,
    stage_title: str = "Market Impact Simulation",
) -> matplotlib.figure.Figure:
    """Create a price chart from price data, using Seaborn for the price plot."""
    plt.style.use("dark_background")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax1, ax2 = axes

    fig.patch.set_facecolor("#0E1117")
    ax1.set_facecolor("#262730")
    ax2.set_facecolor("#262730")

    if volume_data is not None and not volume_data.empty:
        init_data = volume_data[volume_data["FinishedVolume"] == 0]
        if not init_data.empty:
            ax1.plot(init_data["Minute"], init_data["FinishedVolume"], "o-", color="#2ecc71", markersize=6, linewidth=2.5, label="Volume")

        non_zero_data = volume_data[volume_data["FinishedVolume"] > 0]
        if not non_zero_data.empty:
            ax1.step(non_zero_data["Minute"], non_zero_data["FinishedVolume"], where="post", color="#2ecc71", linewidth=2.5, marker="o", markersize=6)

        if volume_data is not None and not volume_data.empty and len(volume_data) >= 2:
            ax1.fill_between(volume_data["Minute"], 0, volume_data["FinishedVolume"], color="#2ecc71", alpha=0.15, step="post")

        if not volume_data.empty:
            target_volume = volume_data["TargetVolume"].iloc[-1]
            ax1.axhline(y=target_volume, color="orange", linestyle="--", linewidth=2, label=f"Target: {target_volume:,}")

        handles, labels = ax1.get_legend_handles_labels()
        if "Volume" not in labels and len(handles) > 0:
            handles.insert(0, plt.Line2D([0], [0], color="#2ecc71", lw=2, marker="o", markersize=6))
            labels.insert(0, "Volume")
        ax1.legend(handles, labels, loc="upper left", frameon=True, fontsize=9)

    ax1.grid(color="#444444", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax1.set_xlabel("Time", fontsize=11, color="white", fontweight="bold")
    ax1.set_ylabel("Trading Volume", fontsize=11, color="white", fontweight="bold")
    ax1.tick_params(colors="white", labelsize=10)
    ax1.set_title("Trading Progress", fontsize=12, color="white", fontweight="bold", pad=10)

    if volume_data is not None and not volume_data.empty:
        y_max = int(max(volume_data["TargetVolume"].max() * 1.05, volume_data["FinishedVolume"].max() * 1.05))
        ax1.set_ylim(0, y_max)

    if not price_data.empty:
        x_min = price_data["Minute"].min()
        x_max = price_data["Minute"].max()
        ax1.set_xlim(x_min, x_max)

    if not price_data.empty:
        for agent_type in price_data["TradingAgent"].unique():
            agent_data = price_data[price_data["TradingAgent"] == agent_type]

            if agent_type == "w/o Twap-Buy":
                color = "#3498db"  # Blue
                style = "-"
                marker = "."
                marker_size = 8
                zorder = 5
                line_width = 2.0
                label = "w/o Twap-Buy"
            else:  # "w/ Twap-Buy"
                color = "#e74c3c"  # Red
                style = "--"
                marker = "o"
                marker_size = 6
                zorder = 10
                line_width = 2.5
                label = "w/ Twap-Buy"

            sns.lineplot(
                data=agent_data,
                x="Minute",
                y="median_price",
                ax=ax2,
                label=label,
                color=color,
                linestyle=style,
                marker=marker,
                markersize=marker_size,
                linewidth=line_width,
                zorder=zorder,
            )

            simulation_data = agent_data[agent_data["IsSimulation"]]
            if not simulation_data.empty:
                ax2.fill_between(
                    simulation_data["Minute"],
                    simulation_data["median_price"] - simulation_data["std_price"],
                    simulation_data["median_price"] + simulation_data["std_price"],
                    alpha=0.15,
                    color=color,
                    zorder=zorder - 1,
                )

        ax2.grid(color="#444444", linestyle="--", linewidth=0.5, alpha=0.7)
        ax2.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
        ax2.set_title("Price Impact", pad=10, fontsize=12, color="white", fontweight="bold")
        ax2.set_xlabel("Time", fontsize=11, color="white", fontweight="bold")
        ax2.set_ylabel("Price", fontsize=11, color="white", fontweight="bold")
        ax2.tick_params(colors="white", labelsize=10)

        ax2.set_ylim(
            price_data["median_price"].min() * 0.995,
            price_data["median_price"].max() * 1.005,
        )

        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

        plt.xticks(rotation=45)

        legend = ax2.legend(title="Agent Type", loc="upper left", frameon=True, fontsize=9)
        legend.get_frame().set_facecolor("#262730")
        legend.get_frame().set_edgecolor("#555555")
        plt.setp(legend.get_texts(), color="white")
        plt.setp(legend.get_title(), color="white", fontweight="bold")

    if stage_title:
        fig.suptitle(stage_title, fontsize=14, color="white", fontweight="bold", y=0.98)

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(wspace=0.2)

    return fig


def process_simulation_data(
    tasks: list[RolloutTask],
    results: list[list[TradeInfo]],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Process simulation data into DataFrames for visualization."""
    prices = []

    for i, (task, trade_infos) in enumerate(zip(tasks, results, strict=True)):
        trading_agent_orders = set()
        finished_volume = 0

        for trade_info in trade_infos:
            if trade_info.lob_snapshot.last_price < 0:
                continue

            if trade_info.order.agent_id == 2:  # TWAP agent ID
                trading_agent_orders.add(trade_info.order.order_id)

            if trade_info.transactions:
                for trans in trade_info.transactions:
                    if trans.type not in ["B", "S"]:
                        continue
                    if (
                        len(trans.buy_id) == 1
                        and len(trans.sell_id) == 1
                        and (trans.buy_id[0] in trading_agent_orders or trans.sell_id[0] in trading_agent_orders)
                    ):
                        finished_volume += trans.volume

            agent_type = "w/ Twap-Buy" if task.twap_agent_target_volume > 0 else "w/o Twap-Buy"
            is_simulation = trade_info.order.time >= task.init_end_time

            if task.twap_agent_target_volume == 0 or is_simulation:
                prices.append(
                    {
                        "Time": trade_info.order.time,
                        "Price": float(trade_info.lob_snapshot.last_price),
                        "Rollout": int(i),
                        "TradingAgent": str(agent_type),
                        "IsSimulation": bool(is_simulation),
                        "TargetVolume": int(task.twap_agent_target_volume),
                        "FinishedVolume": int(finished_volume),
                    }
                )

    price_data = pd.DataFrame(prices)
    if price_data.empty:
        return pd.DataFrame(), None

    price_data["Minute"] = price_data["Time"].dt.floor("min")
    price_data = price_data.drop(columns=["Time"])

    price_stats = (
        price_data.groupby(["Minute", "TradingAgent", "IsSimulation"])
        .agg(
            median_price=("Price", "median"),
            std_price=("Price", "std"),
        )
        .reset_index()
    )

    volume_data = None
    if "w/ Twap-Buy" in price_data["TradingAgent"].unique():
        twap_data = price_data[price_data["TradingAgent"] == "w/ Twap-Buy"]

        sim_volume_data = (
            twap_data.groupby("Minute")
            .agg(
                FinishedVolume=("FinishedVolume", "mean"),
                TargetVolume=("TargetVolume", "max"),
            )
            .reset_index()
        )

        all_time_points = sorted(price_data["Minute"].unique())
        target_volume = int(twap_data["TargetVolume"].iloc[0]) if not twap_data.empty else 0

        volume_data_list = []
        for minute in all_time_points:
            matching_rows = sim_volume_data[sim_volume_data["Minute"] == minute]
            if not matching_rows.empty:
                volume_data_list.append(
                    {
                        "Minute": minute,
                        "FinishedVolume": float(matching_rows["FinishedVolume"].iloc[0]),
                        "TargetVolume": int(matching_rows["TargetVolume"].iloc[0]),
                    }
                )
            else:
                volume_data_list.append({"Minute": minute, "FinishedVolume": 0.0, "TargetVolume": target_volume})

        volume_data = pd.DataFrame(volume_data_list)
        volume_data = volume_data.sort_values("Minute")

    return price_stats, volume_data


def run_baseline_simulation(
    num_rollouts: int,
    rollouts_path: Path,
    seed_for_init_state: int,
    start_time: pd.Timestamp | None = None,
    init_end_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> tuple[list[RolloutTask], list[list[TradeInfo]]]:
    """Run baseline market simulations without trading agents.

    Args:
        num_rollouts: Number of simulation runs
        rollouts_path: Path where simulation results will be saved
        seed_for_init_state: Random seed for initialization phase
        start_time: When the simulation begins
        init_end_time: When initialization phase ends
        end_time: When the simulation ends

    Returns:
        Tuple of tasks and results from baseline simulation
    """
    # Handle default times
    if start_time is None:
        start_time = pd.Timestamp("2024-01-01 09:30:00")
    if init_end_time is None:
        init_end_time = pd.Timestamp("2024-01-01 10:00:00")
    if end_time is None:
        end_time = pd.Timestamp("2024-01-01 10:05:00")

    symbol = "000000"

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
        results = [execute_single_simulation(task) for task in tasks]
    else:
        with Pool(processes=16) as pool:
            results = pool.map(execute_simulation_with_error_handling, tasks)

    pkl_utils.save_pkl_zstd((tasks, results), rollouts_path)

    return tasks, results


def run_twap_simulation(
    num_rollouts: int,
    rollouts_path: Path,
    seed_for_init_state: int,
    baseline_results: list[list[TradeInfo]],
    volume_ratio: float = 0.3,
    start_time: pd.Timestamp | None = None,
    init_end_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> tuple[list[RolloutTask], list[list[TradeInfo]]]:
    """Run market impact simulations with TWAP trading agents.

    Args:
        num_rollouts: Number of simulation runs
        rollouts_path: Path where simulation results will be saved
        seed_for_init_state: Random seed for initialization phase
        baseline_results: Results from baseline simulation to calculate target volume
        volume_ratio: Target volume for TWAP agent as ratio of average volume
        start_time: When the simulation begins
        init_end_time: When initialization phase ends
        end_time: When the simulation ends

    Returns:
        Tuple of tasks and results from TWAP simulation
    """
    # Handle default times
    if start_time is None:
        start_time = pd.Timestamp("2024-01-01 09:30:00")
    if init_end_time is None:
        init_end_time = pd.Timestamp("2024-01-01 10:00:00")
    if end_time is None:
        end_time = pd.Timestamp("2024-01-01 10:05:00")

    symbol = "000000"

    avg_volume = calculate_average_volume(baseline_results, init_end_time, end_time)
    target_volume = int(avg_volume * volume_ratio) // 100 * 100

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
        trading_results = [execute_single_simulation(task) for task in trading_tasks]
    else:
        with Pool(processes=16) as pool:
            trading_results = pool.map(execute_simulation_with_error_handling, trading_tasks)

    pkl_utils.save_pkl_zstd((trading_tasks, trading_results), rollouts_path)

    return trading_tasks, trading_results


def run_simulation(
    num_rollouts: int,
    rollouts_path: Path,
    seed_for_init_state: int,
    volume_ratio: float = 0.3,
    start_time: pd.Timestamp | None = None,
    init_end_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> None:
    """Run market impact simulations with and without trading agents.

    This function executes two sets of simulations:
    1. Baseline simulations without TWAP agent
    2. Market impact simulations with TWAP agent targeting a volume
       determined by the volume_ratio parameter

    Args:
        num_rollouts: Number of simulation runs in each set
        rollouts_path: Path where simulation results will be saved
        seed_for_init_state: Random seed for initialization phase
        volume_ratio: Target volume for TWAP agent as ratio of average volume
        start_time: When the simulation begins
        init_end_time: When initialization phase ends
        end_time: When the simulation ends

    Returns:
        None
    """
    # Handle default times
    if start_time is None:
        start_time = pd.Timestamp("2024-01-01 09:30:00")
    if init_end_time is None:
        init_end_time = pd.Timestamp("2024-01-01 10:00:00")
    if end_time is None:
        end_time = pd.Timestamp("2024-01-01 10:05:00")

    baseline_path = rollouts_path.with_name(f"{rollouts_path.stem}_baseline{rollouts_path.suffix}")
    twap_path = rollouts_path.with_name(f"{rollouts_path.stem}_twap{rollouts_path.suffix}")

    tasks, results = run_baseline_simulation(
        num_rollouts=num_rollouts,
        rollouts_path=baseline_path,
        seed_for_init_state=seed_for_init_state,
        start_time=start_time,
        init_end_time=init_end_time,
        end_time=end_time,
    )

    trading_tasks, trading_results = run_twap_simulation(
        num_rollouts=num_rollouts,
        rollouts_path=twap_path,
        seed_for_init_state=seed_for_init_state,
        baseline_results=results,
        volume_ratio=volume_ratio,
        start_time=start_time,
        init_end_time=init_end_time,
        end_time=end_time,
    )

    pkl_utils.save_pkl_zstd((tasks + trading_tasks, results + trading_results), rollouts_path)


def display_market_impact_app() -> None:  # noqa: PLR0915
    """Display the market impact app content when called from the main app."""
    st.markdown(
        """
        <style>
        /* Download button styling */
        .stDownloadButton>button {
            width: 100%;
            background-color: #1f6feb;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            font-weight: 600;
            margin-top: 0px !important;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stDownloadButton>button:hover {
            background-color: #388bfd;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        /* Info box styling */
        .info-box {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        /* Progress bar styling */
        [data-testid="stProgressBar"] > div {
            background-color: #238636 !important;
        }
        /* Slider styling */
        .stSlider [data-baseweb="slider"] {
            margin-top: 10px !important;
        }
        .stSlider [data-baseweb="thumb"] {
            background-color: #238636 !important;
            border-color: #238636 !important;
        }
        .stSlider [data-baseweb="track"] {
            background-color: #30363d !important;
        }
        .stSlider [data-baseweb="tick"] {
            background-color: #8b949e !important;
        }
        /* Success message styling */
        .stSuccess {
            background-color: rgba(35, 134, 54, 0.2) !important;
            color: #f0f6fc !important;
            border: 1px solid rgba(35, 134, 54, 0.3) !important;
        }
        /* Info message styling */
        .stInfo {
            background-color: rgba(31, 111, 235, 0.2) !important;
            color: #f0f6fc !important;
            border: 1px solid rgba(31, 111, 235, 0.3) !important;
        }
        /* Error message styling */
        .stError {
            background-color: rgba(248, 81, 73, 0.2) !important;
            color: #f0f6fc !important;
            border: 1px solid rgba(248, 81, 73, 0.3) !important;
        }
        .stNumberInput {
            position: relative;
        }
        .stNumberInput div[data-baseweb="input"] {
            background-color: #161b22 !important;
            border-color: #30363d !important;
            border-radius: 6px !important;
            box-shadow: none !important;
        }
        .stNumberInput input {
            color: #f0f6fc !important;
            font-size: 14px !important;
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }
        .stNumberInput [data-testid="stWidgetLabel"] {
            color: #c9d1d9 !important;
        }
        button[data-testid="baseButton-secondary"] {
            color: #79c0ff !important;
        }
        button[data-testid="decrease"],
        button[data-testid="increase"] {
            border-radius: 4px !important;
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
            color: #f0f6fc !important;
            font-weight: bold !important;
            transition: all 0.2s ease !important;
        }
        button[data-testid="decrease"]:hover,
        button[data-testid="increase"]:hover {
            background-color: #30363d !important;
            border-color: #58a6ff !important;
        }
        /* Icon container styling for header */
        .header-icon-container {
            background: linear-gradient(135deg, #238636, #58a6ff);
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .header-icon {
            width: 30px;
            height: 30px;
            fill: white;
        }
        .stApp {
            background-color: #0d1117;
            color: #f0f6fc;
        }
        /* Similar styles as in home_app.py */

        [data-theme="light"] input[type="number"],
        [data-theme="light"] .stNumberInput input {
            color: #000000 !important;
        }

        [data-theme="light"] .stNumberInput div[data-baseweb="input"],
        [data-theme="light"] .stDateInput div[data-baseweb="input"],
        [data-theme="light"] input[type="time"] {
            background-color: #ffffff !important;
            border-color: #cccccc !important;
        }

        /* Make all number inputs black text on white background regardless of theme */
        input[type="number"],
        .stNumberInput input {
            color: #000000 !important;
            background-color: #ffffff !important;
        }

        .stNumberInput div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border-color: #cccccc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Increase top margin in title area
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)

    col_back, col_title = st.columns([0.8, 5])
    with col_back:
        st.markdown('<div class="back-button">', unsafe_allow_html=True)
        back = st.button("← Back to Home")
        st.markdown("</div>", unsafe_allow_html=True)
        if back:
            st.session_state["current_app"] = "home"
            st.rerun()

    with col_title:
        st.markdown(
            """
            <div class="header-container">
                <div class="header-icon-container">
                    <svg class="header-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 3V18C3 19.1 3.9 20 5 20H21V18H5V3H3ZM21 12L16 7L12 11L8 7L6 9V14H21V12Z" fill="white"/>
                    </svg>
                </div>
                <h1>Market Impact Simulation</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-box">
            <p style="margin: 0; font-size: 0.9rem;">
                <strong>Note:</strong> This simulation demonstrates the market impact of TWAP trading strategies.
                The simulation runs in two phases: first without TWAP agent to establish baseline,
                then with TWAP agent to show market impact. Due to commercial licensing restrictions on real order flow data,
                we use noise agents to generate initial phase data. For better results,
                you can replace this with real order data if available.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<h2 style='margin-top: 0; padding-top: 0;'>Simulation Parameters</h2>", unsafe_allow_html=True)

        st.subheader("Time Settings")
        default_time = dt_time(9, 30, 0)
        st.markdown("<small>Trading period: 09:30-11:30</small>", unsafe_allow_html=True)

        start_date_input = st.date_input("Start Date", datetime(2024, 1, 1))
        if not isinstance(start_date_input, date):
            st.error("Invalid start date selected.")
            return
        start_date: date = start_date_input  # Assert type using imported date

        start_time = st.time_input("Start Time", default_time)

        start_datetime = datetime.combine(start_date, start_time)
        max_end_time = datetime.combine(start_date, dt_time(11, 30, 0))
        max_available_minutes = int((max_end_time - start_datetime).total_seconds() / 60)

        if max_available_minutes <= 0:
            st.error("Start time must be before 11:30")
            max_available_minutes = 30

        init_min = 1
        init_max = min(60, max_available_minutes)
        init_value = min(30, init_max)

        init_duration = st.number_input(
            "Initial Phase Duration (min)",
            min_value=init_min,
            max_value=init_max,
            value=init_value,
        )

        total_min = init_duration
        total_max = max_available_minutes
        total_value = min(init_duration + 5, total_max)

        total_duration = st.number_input(
            "Total Duration (min)",
            min_value=total_min,
            max_value=total_max,
            value=total_value,
        )

        end_datetime = start_datetime + timedelta(minutes=total_duration)
        st.info(f"Simulation starts at: {start_datetime.strftime('%H:%M')}")
        st.info(f"Simulation ends at: {end_datetime.strftime('%H:%M')}")

        st.subheader("Simulation Settings")
        num_rollouts = st.number_input("Number of Rollouts", min_value=2, value=8)
        seed = st.number_input("Random Seed", min_value=0, value=0)
        volume_ratio = st.slider("Volume Ratio", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

        run_simulation_button = st.button("Run Simulation", type="primary")

    if run_simulation_button:
        col1, col2 = st.columns([3, 1])

        with col1:
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            with progress_placeholder.container():
                progress_bar = st.progress(0)

            status_placeholder.info("Preparing to start simulation...")

        chart_placeholder = st.empty()

        stages = {
            "Preparing environment": 0.05,
            "Running baseline phase": 0.40,
            "Processing baseline results": 0.10,
            "Running TWAP phase": 0.35,
            "Processing final results": 0.10,
        }

        try:
            if not isinstance(start_date_input, date):
                st.error("Invalid start date selected.")
                return
            start_date = start_date_input

            start_datetime = datetime.combine(start_date, start_time)
            init_end_datetime = start_datetime + timedelta(minutes=init_duration)
            end_datetime = start_datetime + timedelta(minutes=total_duration)

            with chart_placeholder.container():
                st.markdown(
                    """
                    <div style="
                        height: 200px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #0E1117;
                        border-radius: 10px;
                        margin: 10px 0;
                        font-size: 18px;
                        color: #8B949E;
                    ">
                        Running baseline simulation...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            current_progress = 0.0
            progress_bar.progress(current_progress)
            status_placeholder.info("Preparing simulation environment...")
            py_time.sleep(0.3)

            current_progress += stages["Preparing environment"]
            progress_bar.progress(current_progress)

            output_dir = Path(C.directory.output_root_dir) / "market-impact-example"
            output_dir.mkdir(parents=True, exist_ok=True)

            status_placeholder.info("Running baseline simulation...")
            baseline_rollouts_path = output_dir / f"baseline_rollouts-seed{seed}-volume_ratio{volume_ratio}.zstd"

            baseline_tasks, baseline_results = run_baseline_simulation(
                num_rollouts=num_rollouts,
                rollouts_path=baseline_rollouts_path,
                seed_for_init_state=seed,
                start_time=pd.Timestamp(start_datetime),
                init_end_time=pd.Timestamp(init_end_datetime),
                end_time=pd.Timestamp(end_datetime),
            )

            current_progress += stages["Running baseline phase"]
            progress_bar.progress(current_progress)
            status_placeholder.info("Processing baseline results...")

            baseline_price_stats, _ = process_simulation_data(baseline_tasks, baseline_results)

            chart_placeholder.empty()
            with chart_placeholder.container():
                st.subheader("Baseline Market Simulation (without TWAP)")
                st.pyplot(create_price_chart(baseline_price_stats, None, "Baseline Market Simulation"))

            current_progress += stages["Processing baseline results"]
            progress_bar.progress(current_progress)

            status_placeholder.info("Running TWAP simulation...")

            chart_placeholder.empty()
            with chart_placeholder.container():
                st.subheader("Baseline Market Simulation (without TWAP)")
                st.pyplot(create_price_chart(baseline_price_stats, None, "Baseline Market Simulation"))
                st.markdown(
                    """
                    <div style="
                        height: 80px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #161b22;
                        border-radius: 10px;
                        margin: 10px 0;
                        font-size: 16px;
                        color: #58a6ff;
                    ">
                        Running TWAP simulation now... Please wait.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            twap_rollouts_path = output_dir / f"twap_rollouts-seed{seed}-volume_ratio{volume_ratio}.zstd"

            twap_tasks, twap_results = run_twap_simulation(
                num_rollouts=num_rollouts,
                rollouts_path=twap_rollouts_path,
                seed_for_init_state=seed,
                baseline_results=baseline_results,
                volume_ratio=volume_ratio,
                start_time=pd.Timestamp(start_datetime),
                init_end_time=pd.Timestamp(init_end_datetime),
                end_time=pd.Timestamp(end_datetime),
            )

            twap_price_stats, volume_data = process_simulation_data(twap_tasks, twap_results)

            combined_price_stats = pd.concat([baseline_price_stats, twap_price_stats])

            current_progress += stages["Running TWAP phase"]
            progress_bar.progress(current_progress)
            status_placeholder.info("Generating final visualizations...")

            chart_placeholder.empty()

            tab2, tab1, tab3 = st.tabs(["    Impact Analysis    ", "    Baseline Simulation    ", "    Parameters    "])

            with tab2:
                st.subheader("Market Impact Analysis")
                st.pyplot(create_price_chart(combined_price_stats, volume_data, "Complete Market Impact Analysis"))

            with tab1:
                st.subheader("Baseline Market Simulation")
                baseline_only_stats = baseline_price_stats[baseline_price_stats["TradingAgent"] == "w/o Twap-Buy"]
                st.pyplot(create_price_chart(baseline_only_stats, None, "Market Simulation without TWAP Agent"))

            with tab3:
                col1, col2 = st.columns([3, 1])
                with col1:
                    params_df = pd.DataFrame(
                        {
                            "Parameter": [
                                "Start Time",
                                "Initial Phase End",
                                "Total End Time",
                                "Rollouts",
                                "Seed",
                                "Volume Ratio",
                            ],
                            "Value": [
                                start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                init_end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                str(num_rollouts),
                                str(seed),
                                f"{volume_ratio:.1f}",
                            ],
                        }
                    )
                    st.dataframe(params_df, use_container_width=True, hide_index=True)

                with col2:
                    st.download_button(
                        label="Download Baseline Data",
                        data=baseline_rollouts_path.read_bytes(),
                        file_name=f"baseline_seed{seed}_ratio{volume_ratio}.zstd",
                        mime="application/octet-stream",
                    )
                    st.download_button(
                        label="Download TWAP Data",
                        data=twap_rollouts_path.read_bytes(),
                        file_name=f"twap_seed{seed}_ratio{volume_ratio}.zstd",
                        mime="application/octet-stream",
                    )

                    combined_rollouts_path = output_dir / f"combined_rollouts-seed{seed}-volume_ratio{volume_ratio}.zstd"
                    pkl_utils.save_pkl_zstd((baseline_tasks + twap_tasks, baseline_results + twap_results), combined_rollouts_path)
                    st.download_button(
                        label="Download Combined Data",
                        data=combined_rollouts_path.read_bytes(),
                        file_name=f"combined_seed{seed}_ratio{volume_ratio}.zstd",
                        mime="application/octet-stream",
                    )

            current_progress = 1.0
            progress_bar.progress(current_progress)
            progress_placeholder.empty()
            status_placeholder.success("Simulation complete!")

        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"An error occurred during simulation: {e!s}")
            st.exception(e)


if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(
        page_title="Market Impact Simulation",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0d1117;
            color: #f0f6fc;
        }
        /* Similar styles as in home_app.py */
        </style>
        """,
        unsafe_allow_html=True,
    )

    display_market_impact_app()
