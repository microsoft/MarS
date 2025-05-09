import streamlit as st


def main() -> None:
    """Main application entry point for the Market Simulation Dashboard."""
    st.set_page_config(
        page_title="Market Simulation Dashboard",
        page_icon="doc/img/MarS_icon.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* Force dark theme regardless of system settings */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stSidebar"], [data-testid="stDecoration"], [data-testid="baseButton-headerNoPadding"],
        [data-testid="stVerticalBlock"] {
            background-color: #0d1117 !important;
            color: #f0f6fc !important;
        }

        /* Fix main content background */
        [data-testid="stAppViewBlockContainer"] {
            background-color: #0d1117 !important;
        }

        /* Fix tooltips and other floating elements */
        div.stTooltipIcon, span.st-emotion-cache-36jtlm, div[data-baseweb="tooltip"], div[data-baseweb="popover"] {
            background-color: #161b22 !important;
            color: #f0f6fc !important;
            border-color: #30363d !important;
        }

        /* Fix for tab containers */
        [data-testid="stTabContent"] {
            background-color: #0d1117 !important;
        }

        /* Make sure all sections maintain dark theme */
        section[data-testid="stSidebar"] > div, div[data-testid="collapsedControl"] {
            background-color: #0d1117 !important;
            border-color: #30363d !important;
        }

        /* Fix iframe backgrounds */
        iframe {
            background-color: #0d1117 !important;
        }

        /* Fix tooltips and menu items */
        div[role="tooltip"], div[role="menu"], div[role="menuitem"] {
            background-color: #161b22 !important;
            color: #f0f6fc !important;
            border-color: #30363d !important;
        }

        /* Fix any white backgrounds in light mode */
        div:not([class]):not([id]) {
            background-color: transparent !important;
        }

        .stApp {
            background-color: #0d1117;
            color: #f0f6fc;
        }
        /* Button styling */
        .stButton>button {
            width: 100%;
            background-color: #238636;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            font-weight: 600;
            margin-top: 0px !important;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #2ea043;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        /* Card styling */
        .card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
            border-color: #58a6ff;
        }
        .card-title {
            color: #f0f6fc;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .card-description {
            color: #8b949e;
            flex-grow: 1;
            margin-bottom: 1.2rem;
            line-height: 1.6;
        }
        .card-icon {
            font-size: 3rem;
            margin-bottom: 1.2rem;
            color: #238636;
            text-shadow: 0 0 10px rgba(35, 134, 54, 0.3);
        }
        /* Feature card styling */
        .feature-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            height: 100%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            border-color: #58a6ff;
        }
        .feature-card h3 {
            color: #58a6ff;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .feature-card p {
            color: #8b949e;
            line-height: 1.5;
        }
        /* Header styling */
        h1 {
            color: #f0f6fc;
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        h2 {
            color: #f0f6fc;
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 0.8rem;
            margin-bottom: 1.2rem;
            position: relative;
            padding-bottom: 0.5rem;
        }
        h2:after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 80px;
            height: 3px;
            background: linear-gradient(90deg, #238636, #58a6ff);
            border-radius: 3px;
        }
        h3 {
            color: #f0f6fc;
            font-size: 1.4rem;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #30363d;
        }
        .logo {
            font-size: 3.2rem;
            margin-right: 1rem;
            color: #238636;
            text-shadow: 0 0 15px rgba(35, 134, 54, 0.4);
        }
        .intro-box {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            line-height: 1.6;
        }
        .intro-box p {
            margin: 0;
            color: #8b949e;
            font-size: 1.05rem;
        }
        /* Adjust container padding */
        div.block-container {
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }
        /* Footer styling */
        .footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #30363d;
            text-align: center;
            color: #8b949e;
            font-size: 0.9rem;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid #30363d;
        }
        /* Make streamlit elements look better */
        .stSelectbox [data-baseweb="select"] {
            background-color: #161b22 !important;
            border-color: #30363d !important;
        }
        .stNumberInput [data-baseweb="input"] {
            background-color: #161b22 !important;
            border-color: #30363d !important;
            color: #f0f6fc !important;
        }
        .logo-img {
            height: 3.5rem;
            margin-right: 1rem;
        }
        /* Back button styling */
        .back-button>button {
            background-color: #21262d !important;
            color: #c9d1d9 !important;
            width: auto !important;
            padding: 0.4rem 0.8rem !important;
            font-size: 0.9rem !important;
            float: left !important;
            border: 1px solid #30363d !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
        }
        .back-button>button:hover {
            background-color: #30363d !important;
            border-color: #8b949e !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        /* Enhanced Card Styling */
        .enhanced-card {
            background: linear-gradient(145deg, #161b22, #1a2029);
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.4s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25);
        }
        .enhanced-card:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.35);
            border-color: #58a6ff;
            background: linear-gradient(145deg, #1a2029, #21262d);
        }
        .enhanced-icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
            text-shadow: 0 0 15px rgba(35, 134, 54, 0.5);
            display: inline-block;
            background: linear-gradient(45deg, #238636, #58a6ff);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .enhanced-title {
            color: #f0f6fc;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 0 2px 3px rgba(0, 0, 0, 0.3);
            position: relative;
            padding-bottom: 10px;
        }
        .enhanced-title:after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, #238636, #58a6ff);
            border-radius: 3px;
        }
        .enhanced-description {
            color: #8b949e;
            flex-grow: 1;
            margin-bottom: 1.5rem;
            line-height: 1.7;
            font-size: 1.1rem;
        }
        .card-features {
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }
        .card-feature-item {
            background-color: rgba(35, 134, 54, 0.15);
            border-radius: 20px;
            padding: 5px 12px;
            margin-right: 8px;
            margin-bottom: 8px;
            color: #8bc34a;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .forecast-icon {
            width: 64px;
            height: 64px;
            margin-bottom: 1.5rem;
        }
        .impact-icon {
            width: 64px;
            height: 64px;
            margin-bottom: 1.5rem;
        }
        .icon-container {
            background: linear-gradient(135deg, #238636, #58a6ff);
            width: 80px;
            height: 80px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
        }

        .icon {
            width: 48px;
            height: 48px;
            fill: white;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    if "current_app" not in st.session_state:
        st.session_state["current_app"] = "home"

    if st.session_state["current_app"] == "home":
        display_home_app()
    elif st.session_state["current_app"] == "forecast":
        from market_simulation.examples.demo.forecast_app import display_forecast_app

        display_forecast_app()
    elif st.session_state["current_app"] == "impact":
        from market_simulation.examples.demo.market_impact_app import (
            display_market_impact_app,
        )

        display_market_impact_app()


def display_home_app() -> None:
    """Display the home page content with app selection cards."""
    st.markdown(
        """
        <div class="header-container">
            <h1>Market Simulation Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-box">
            <p>Welcome to the <strong style="color:#f0f6fc">Market Simulation Dashboard</strong>,
            which provides advanced tools for simulating market behavior under various conditions,
            analyzing market impact, and forecasting price movements.
            <br>
            Select one of the simulation tools below to begin your analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .icon-container {
            background: linear-gradient(135deg, #238636, #58a6ff);
            width: 80px;
            height: 80px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
        }

        .icon {
            width: 48px;
            height: 48px;
            fill: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="enhanced-card">
                <div class="icon-container">
                    <svg class="icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 3V21H21V3H3ZM19 19H5V5H19V19ZM7 13H9V17H7V13ZM11 10H13V17H11V10ZM15 7H17V17H15V7Z" fill="white"/>
                    </svg>
                </div>
                <div class="enhanced-title">Market Forecast Simulation</div>
                <div class="enhanced-description">
                    Simulate future market prices with advanced techniques.
                    Analyze price movements over time and visualize potential market scenarios with
                    confidence intervals and statistical analyses.
                </div>
                <div class="card-features">
                    <span class="card-feature-item">Price Prediction</span>
                    <span class="card-feature-item">Trend Analysis</span>
                    <span class="card-feature-item">Risk Assessment</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Forecast Tool", key="forecast_btn"):
            st.session_state["current_app"] = "forecast"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="enhanced-card">
                <div class="icon-container">
                    <svg class="icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 3V18C3 19.1 3.9 20 5 20H21V18H5V3H3ZM21 12L16 7L12 11L8 7L6 9V14H21V12Z" fill="white"/>
                    </svg>
                </div>
                <div class="enhanced-title">Market Impact Analysis</div>
                <div class="enhanced-description">
                    Evaluate how trading strategies like TWAP affect market prices and liquidity.
                    Analyze the market impact of various trading activities and optimize execution
                    parameters to minimize costs and maximize efficiency.
                </div>
                <div class="card-features">
                    <span class="card-feature-item">Impact Modeling</span>
                    <span class="card-feature-item">TWAP Strategy</span>
                    <span class="card-feature-item">Execution Analysis</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Impact Analysis Tool", key="impact_btn"):
            st.session_state["current_app"] = "impact"
            st.rerun()

    st.markdown(
        """
        <div class="footer">
            <p>Market Simulation Dashboard © 2024 | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
