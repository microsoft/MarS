import time
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import dates

from market_simulation.conf import C
from market_simulation.examples.forecast import run_simulation
from market_simulation.utils import pkl_utils


def plot_price_curves(rollouts: list, output_path: Path, end_time: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Plot price curves."""
    if not rollouts:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices = []
    for i, trade_infos in enumerate(rollouts):
        filtered_infos = trade_infos
        if end_time:
            filtered_infos = [x for x in trade_infos if x.order.time <= end_time]

        prices.extend(
            [
                {
                    "Time": x.order.time,
                    "Price": x.lob_snapshot.last_price,
                    "Agent": "Init-Agent" if x.order.agent_id == 0 else "BG-Agent",
                    "Rollout": i,
                }
                for x in filtered_infos
                if x.lob_snapshot.last_price > 0
            ]
        )

    if not prices:
        return None

    price_data = pd.DataFrame(prices)
    price_data["Minute"] = price_data["Time"].dt.floor("min")
    price_data = price_data.drop(columns=["Time"])
    price_data = price_data.groupby(["Minute", "Rollout", "Agent"]).mean().reset_index()

    price_data = (
        price_data.groupby(["Minute", "Agent"])
        .agg(
            median_price=("Price", "median"),
            std_price=("Price", "std"),
        )
        .reset_index()
    )

    return price_data


def create_price_chart(price_data: pd.DataFrame, stage_title: str = "Market Price Simulation") -> plt.Figure:
    """Create a price chart from price data."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 3.5))

    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#262730")

    line_color = "#3498db"  # Bright blue
    area_color = "#2980b9"  # Deep blue
    scatter_colors = {"Init-Agent": "#e74c3c", "BG-Agent": "#2ecc71"}  # Red and green for specific agents

    sns.lineplot(x="Minute", y="median_price", data=price_data, ax=ax, linewidth=2.2, color=line_color)

    ax.fill_between(
        price_data["Minute"],
        price_data["median_price"] - price_data["std_price"],
        price_data["median_price"] + price_data["std_price"],
        alpha=0.25,
        color=area_color,
        label="Standard Deviation",
    )

    sns.scatterplot(
        x="Minute",
        y="median_price",
        data=price_data,
        hue="Agent",
        ax=ax,
        s=30,
        alpha=0.8,
        palette=scatter_colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.grid(color="#444444", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax.set_title(f"{stage_title}", pad=10, fontsize=12, color="white", fontweight="bold")
    ax.set_xlabel("Time", fontsize=10, color="white", fontweight="bold")
    ax.set_ylabel("Price", fontsize=10, color="white", fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    plt.xticks(rotation=45)

    legend = ax.legend(title="Agent Type", bbox_to_anchor=(1.0, 1), loc="upper left", frameon=True, fontsize=9, title_fontsize=10)
    legend.get_frame().set_facecolor("#262730")
    legend.get_frame().set_edgecolor("#555555")
    plt.setp(legend.get_texts(), color="white")
    plt.setp(legend.get_title(), color="white", fontweight="bold")

    fig.tight_layout()

    return fig


def split_simulation(
    num_rollouts: int, rollouts_path: Path, seed_for_init_state: int, start_time: pd.Timestamp, init_end_time: pd.Timestamp, end_time: pd.Timestamp
) -> tuple[list, list]:
    """Split the simulation into two phases and show intermediate results."""
    interim_rollouts_path = rollouts_path.with_name(f"interim_rollouts-seed{seed_for_init_state}.zstd")

    run_simulation(
        symbol="000000",
        num_rollouts=num_rollouts,
        rollouts_path=interim_rollouts_path,
        seed_for_init_state=seed_for_init_state,
        start_time=start_time,
        init_end_time=init_end_time,
        end_time=init_end_time,
    )

    interim_rollouts = pkl_utils.load_pkl_zstd(interim_rollouts_path)

    run_simulation(
        symbol="000000",
        num_rollouts=num_rollouts,
        rollouts_path=rollouts_path,
        seed_for_init_state=seed_for_init_state,
        start_time=start_time,
        init_end_time=init_end_time,
        end_time=end_time,
    )

    full_rollouts = pkl_utils.load_pkl_zstd(rollouts_path)

    return interim_rollouts, full_rollouts


def display_forecast_app() -> None:  # noqa: PLR0915
    """Display the forecast app content when called from the main app."""
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

        /* Sidebar styling */
        .css-1vq4p4l.e1fqkh3o4 {  /* Sidebar container */
            background-color: #161b22 !important;
            border-right: 1px solid #30363d !important;
            padding: 2rem 1rem !important;
        }

        /* Sidebar section headers */
        .sidebar .stMarkdown h2 {
            color: #e6edf3 !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 1.5rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid #30363d !important;
        }

        /* Sidebar subheaders */
        .sidebar .stMarkdown h3 {
            color: #58a6ff !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            margin: 1.5rem 0 1rem 0 !important;
        }

        /* Input containers in sidebar */
        .sidebar .stNumberInput,
        .sidebar .stDateInput,
        .sidebar .stTimeInput {
            background: #21262d !important;
            border-radius: 8px !important;
            padding: 0.5rem !important;
            margin-bottom: 1rem !important;
            border: 1px solid #30363d !important;
            transition: all 0.2s ease !important;
        }

        .sidebar .stNumberInput:hover,
        .sidebar .stDateInput:hover,
        .sidebar .stTimeInput:hover {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 1px #58a6ff !important;
        }

        /* Input labels in sidebar */
        .sidebar [data-testid="stWidgetLabel"] {
            color: #8b949e !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            margin-bottom: 0.3rem !important;
        }

        /* Small text and hints in sidebar */
        .sidebar small {
            color: #8b949e !important;
            font-size: 0.8rem !important;
            opacity: 0.8 !important;
        }

        /* Info messages in sidebar */
        .sidebar .stInfo {
            background-color: rgba(56, 139, 253, 0.1) !important;
            color: #58a6ff !important;
            padding: 0.5rem 1rem !important;
            border-radius: 6px !important;
            margin: 0.5rem 0 !important;
            font-size: 0.9rem !important;
        }

        /* Run simulation button in sidebar */
        .sidebar .stButton button {
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
            color: white !important;
            width: 100% !important;
            padding: 0.75rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 6px !important;
            margin-top: 1.5rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }

        .sidebar .stButton button:hover {
            background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
        }

        /* Number input spinners */
        .sidebar .stNumberInput [data-testid="stSpinner"] {
            background-color: #30363d !important;
            border-radius: 4px !important;
        }

        .sidebar .stNumberInput [data-testid="stSpinner"]:hover {
            background-color: #444c56 !important;
        }

        /* Date and time inputs */
        .sidebar input[type="date"],
        .sidebar input[type="time"] {
            background-color: #21262d !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            color: #e6edf3 !important;
            padding: 0.5rem !important;
            width: 100% !important;
        }

        .sidebar input[type="date"]:focus,
        .sidebar input[type="time"]:focus {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 1px #58a6ff !important;
        }

        /* Divider styling */
        .sidebar hr {
            border-color: #30363d !important;
            margin: 1.5rem 0 !important;
        }

        /* Sidebar headers and labels */
        .sidebar .stMarkdown h1,
        .sidebar .stMarkdown h2,
        .sidebar .stMarkdown h3,
        .sidebar [data-testid="stWidgetLabel"] {
            color: #e6edf3 !important;
            font-weight: 500;
        }

        /* Sidebar small text and hints */
        .sidebar small,
        .sidebar .stMarkdown small {
            color: #8b949e !important;
        }

        /* Number input styling */
        .stNumberInput {
            position: relative;
        }
        .stNumberInput div[data-baseweb="input"] {
            background-color: #21262d !important;
            border-color: #30363d !important;
            border-radius: 6px !important;
            box-shadow: none !important;
        }
        .stNumberInput input {
            color: #e6edf3 !important;
            font-size: 14px !important;
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }

        /* Date and time input styling */
        .stDateInput div[data-baseweb="input"],
        input[type="time"] {
            background-color: #21262d !important;
            border-color: #30363d !important;
            color: #e6edf3 !important;
        }

        /* Input labels */
        .stNumberInput [data-testid="stWidgetLabel"],
        .stDateInput [data-testid="stWidgetLabel"],
        label {
            color: #e6edf3 !important;
            font-weight: 500 !important;
        }

        /* Custom increase/decrease buttons */
        button[data-testid="decrease"],
        button[data-testid="increase"] {
            border-radius: 4px !important;
            background-color: #30363d !important;
            border: 1px solid #484f58 !important;
            color: #e6edf3 !important;
            font-weight: bold !important;
            transition: all 0.2s ease !important;
        }
        button[data-testid="decrease"]:hover,
        button[data-testid="increase"]:hover {
            background-color: #3c444d !important;
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
        /* Help icon styling */
        button[data-testid="baseButton-secondary"] {
            color: #58a6ff !important;
        }
        button[data-testid="baseButton-secondary"]:hover {
            color: #79c0ff !important;
        }
        /* Override any light theme styles */
        .stApp {
            background-color: #0d1117;
            color: #f0f6fc;
        }
        [data-theme="light"] .stApp {
            background-color: #0d1117;
            color: #f0f6fc;
        }
        [data-theme="light"] .stMarkdown,
        [data-theme="light"] .stText,
        [data-theme="light"] .stSubheader,
        [data-theme="light"] .stHeader {
            color: #f0f6fc;
        }
        [data-theme="light"] .stDataFrame {
            background-color: #161b22;
            color: #f0f6fc;
        }
        [data-theme="light"] .stDataFrame td,
        [data-theme="light"] .stDataFrame th {
            color: #f0f6fc;
            border-color: #30363d;
        }
        [data-theme="light"] .stTabs [data-baseweb="tab-list"] {
            background-color: #161b22;
        }
        [data-theme="light"] .stTabs [data-baseweb="tab"] {
            color: #f0f6fc;
        }
        [data-theme="light"] .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #58a6ff;
            background-color: #21262d;
        }

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
                        <path d="M3 3V21H21V3H3ZM19 19H5V5H19V19ZM7 13H9V17H7V13ZM11 10H13V17H11V10ZM15 7H17V17H15V7Z" fill="white"/>
                    </svg>
                </div>
                <h1>Market Simulation Forecast</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-box">
            <p style="margin: 0; font-size: 0.9rem;">
                <strong>Note:</strong> Due to commercial licensing restrictions on real order flow data,
                we use noise agents to generate initial phase data. For better results,
                you can replace this with real order data if available.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .stButton>button {
            margin-top: 0px !important;
        }
        .stDownloadButton>button {
            margin-top: 0px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<h2 style='margin-top: 0; padding-top: 0;'>Simulation Parameters</h2>", unsafe_allow_html=True)

        st.subheader("Time Settings")

        default_time = datetime.strptime("09:30:00", "%H:%M:%S").time()  # noqa: DTZ007

        st.markdown("<small>Trading period: 09:30-11:30</small>", unsafe_allow_html=True)

        start_date = st.date_input("Start Date", datetime(2024, 1, 1))

        if start_date is None:
            start_date = datetime(2024, 1, 1).date()
        elif isinstance(start_date, (datetime, pd.Timestamp)):
            start_date = start_date.date()

        start_time = st.time_input("Start Time", default_time)

        if isinstance(start_date, date):
            date_to_use = start_date
        else:
            date_to_use = date(2024, 1, 1)

        start_datetime = datetime.combine(date_to_use, start_time)
        # NOTE: Using naive datetime is acceptable as we're only interested in time deltas
        max_end_time = datetime.combine(date_to_use, datetime.strptime("11:30:00", "%H:%M:%S").time())  # noqa: DTZ007
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

        run_simulation_button = st.button(
            "Run Simulation",
            type="primary",
        )

    if run_simulation_button:
        col1, col2 = st.columns([4, 1])
        with col1:
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

        chart_placeholder = st.empty()

        with progress_placeholder.container():
            progress_bar = st.progress(0)

        status_placeholder.info("Preparing to start simulation...")

        stages = {
            "Preparing environment": 0.05,
            "Running initial phase": 0.3,
            "Generating interim visualization": 0.1,
            "Running main phase": 0.4,
            "Processing final results": 0.05,
            "Generating final visualization": 0.1,
        }

        try:
            if isinstance(start_date, date):
                date_to_use = start_date
            else:
                date_to_use = date(2024, 1, 1)

            start_datetime = datetime.combine(date_to_use, start_time)
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
                        Simulation running, please wait...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            current_progress = 0.0
            progress_bar.progress(current_progress)
            status_placeholder.info("Preparing simulation environment...")
            time.sleep(0.3)

            current_progress += stages["Preparing environment"]
            progress_bar.progress(current_progress)

            output_dir = Path(C.directory.output_root_dir) / "forecasting-example"
            output_dir.mkdir(parents=True, exist_ok=True)
            rollouts_path = output_dir / f"rollouts-seed{seed}.zstd"

            status_placeholder.info(f"Running initial phase simulation ({init_duration} minutes)...")

            interim_rollouts_path = output_dir / f"interim_rollouts-seed{seed}.zstd"

            run_simulation(
                symbol="000000",
                num_rollouts=num_rollouts,
                rollouts_path=interim_rollouts_path,
                seed_for_init_state=seed,
                start_time=pd.Timestamp(start_datetime),
                init_end_time=pd.Timestamp(init_end_datetime),
                end_time=pd.Timestamp(init_end_datetime),
            )

            current_progress += stages["Running initial phase"]
            progress_bar.progress(current_progress)
            status_placeholder.info("Processing initial phase results...")

            interim_rollouts = pkl_utils.load_pkl_zstd(interim_rollouts_path)
            interim_price_data = plot_price_curves(interim_rollouts, interim_rollouts_path.with_suffix(".png"))

            if interim_price_data is not None:
                chart_placeholder.empty()
                with chart_placeholder.container():
                    st.pyplot(create_price_chart(interim_price_data, stage_title="Initial Phase Results"))

            status_placeholder.info("Running main phase simulation...")

            current_progress += stages["Generating interim visualization"]
            progress_bar.progress(current_progress)

            run_simulation(
                symbol="000000",
                num_rollouts=num_rollouts,
                rollouts_path=rollouts_path,
                seed_for_init_state=seed,
                start_time=pd.Timestamp(start_datetime),
                init_end_time=pd.Timestamp(init_end_datetime),
                end_time=pd.Timestamp(end_datetime),
            )

            current_progress += stages["Running main phase"]
            progress_bar.progress(current_progress)
            status_placeholder.info("Processing final simulation results...")

            rollouts = pkl_utils.load_pkl_zstd(rollouts_path)
            price_data = plot_price_curves(rollouts, rollouts_path.with_suffix(".png"))

            current_progress += stages["Processing final results"]
            progress_bar.progress(current_progress)
            status_placeholder.info("Generating final visualization...")

            chart_placeholder.empty()

            tab1, tab2, tab3 = st.tabs(["    Full Simulation    ", "    Initial Phase    ", "    Parameters    "])

            with tab1:
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    if price_data is not None:
                        st.pyplot(create_price_chart(price_data, stage_title="Complete Simulation Results"))
                    else:
                        st.error("No data available for visualization")

            with tab2:
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    if interim_price_data is not None:
                        st.pyplot(create_price_chart(interim_price_data, stage_title="Initial Phase Results"))
                    else:
                        st.error("No data available for initial phase visualization")

            with tab3:
                col1, col2 = st.columns([3, 1])
                with col1:
                    params_df = pd.DataFrame(
                        {
                            "Parameter": ["Start Time", "Initial Phase End", "Total End Time", "Rollouts", "Seed"],
                            "Value": [
                                start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                init_end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                str(num_rollouts),
                                str(seed),
                            ],
                        }
                    )
                    st.dataframe(params_df, use_container_width=True, hide_index=True)

                with col2:
                    with Path(rollouts_path).open("rb") as f:
                        st.download_button(
                            label="Download Full Data",
                            data=f,
                            file_name=f"simulation_seed{seed}.zstd",
                            mime="application/octet-stream",
                        )

                    with Path(interim_rollouts_path).open("rb") as f:
                        st.download_button(
                            label="Download Initial Data",
                            data=f,
                            file_name=f"initial_seed{seed}.zstd",
                            mime="application/octet-stream",
                        )

            current_progress = 1.0
            progress_bar.progress(current_progress)
            progress_placeholder.empty()
            status_placeholder.success("Simulation complete!")

        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"An error occurred during simulation: {e}")


if __name__ == "__main__":
    from pathlib import Path

    import streamlit as st

    st.set_page_config(
        page_title="Market Simulation Forecast",
        page_icon="📈",
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
        /* Override any light theme styles */
        [data-theme="light"] .stApp {
            background-color: #0d1117;
            color: #f0f6fc;
        }
        [data-theme="light"] .stMarkdown,
        [data-theme="light"] .stText,
        [data-theme="light"] .stSubheader,
        [data-theme="light"] .stHeader {
            color: #f0f6fc;
        }
        [data-theme="light"] .stDataFrame {
            background-color: #161b22;
            color: #f0f6fc;
        }
        [data-theme="light"] .stDataFrame td,
        [data-theme="light"] .stDataFrame th {
            color: #f0f6fc;
            border-color: #30363d;
        }
        [data-theme="light"] .stTabs [data-baseweb="tab-list"] {
            background-color: #161b22;
        }
        [data-theme="light"] .stTabs [data-baseweb="tab"] {
            color: #f0f6fc;
        }
        [data-theme="light"] .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #58a6ff;
            background-color: #21262d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the forecast app
    display_forecast_app()
