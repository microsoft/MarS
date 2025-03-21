"""Stylized facts of Cont.

- 1. Absence of autocorrelations: corr(r(t, delta_t), r(t+tau, delta_t))
- 2. Heavy tails: kurtosis of returns.
- 3. Gain/loss asymmetry: skews of returns.
- 4. Aggregational Gaussianity: kurtosis of returns.
- 5. Intermittency: Fano factor.
- 6. Volatility clustering: corr(|r(t, delta_t)|, |r(t+tau, delta_t)|)
- 7. Conditional heavy tails: kurtosis of returns, need normalization.
- 8. Slow decay of autocorrelation in absolute returns: similar to volatility clustering.
- 9. Leverage effect: corr(r(t, delta_t), |r(t+tau, delta_t)|)
- 10. Volume/volatility correlation: corr(v(t, delta_t), r(t, delta_t))
- 11. Asymmetry in timescales: coarse grain and fine grain returnn correlation.

Information needed to generate these stylized facts:
- r(t, delta_t), delta_t = 1, ..., 20,
    - for above return information, we keep both return of last trade and return of average trade.
    - keep symbol information.
    - simulation start time
- volume for delta_t = 1, ..., 20

Reference paper:
- http://rama.cont.perso.math.cnrs.fr/pdf/empirical.pdf
- https://arxiv.org/abs/2311.07738
"""

import logging
from pathlib import Path
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# np.seterr(all="raise")
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas import Timestamp
from tqdm import tqdm

from market_simulation.utils import pkl_utils
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.trade_info import TradeInfo

ERROR_BAR = ("ci", 95)


RED = "#FF3737"
LIGHT_RED = "#FF8B8B"
BLUE = "#0058DA"
LIGHT_BLUE = "#85B6FF"
GREEN = "#359741"
LIGHT_GREEN = "#A5DFAC"
HUE_ORDER = ["Replay", "Simulation"]

DPI = 300

plt.rcParams["axes.labelsize"] = 8
plt.rcParams["font.size"] = 8
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "DejaVu Sans, Arial, Helvetica, Lucida Grande, Verdana, Geneva, Lucid, Avant Garde, sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavusans"
# plt.rcParams["legend.frameon"] = False

plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 1.0

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.05

sns.set_context("paper")  # poster, paper, talk, notebook

PriceType = Literal["last", "mean"]

AX_TICK_20 = [0, *list(range(2, 21, 2))]


class MinuteInfo(NamedTuple):
    """Minute Info."""

    time: Timestamp
    price_last: float
    price_mean: float
    price_mid: float
    price_mid_last: float
    volume: float
    num_orders: int


class RolloutInfo(NamedTuple):
    """Rollout Info."""

    symbol: str
    start_time: Timestamp
    simulation_minutes: list[MinuteInfo]
    replay_minutes: list[MinuteInfo]


def save_and_close_fig(fig: Figure, output_path: Path, dpi: int = 300) -> None:
    """Save and close figure."""
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)
    logging.info(f"Saved to {output_path}.")


def cal_log_return(minutes: list[MinuteInfo], start_minute: int, end_minute: int, price_type: PriceType) -> float:
    """Calculate log return."""
    assert 0 <= start_minute < end_minute < len(minutes)
    assert price_type in ["last", "mean"]
    if price_type == "mean":
        p0 = minutes[start_minute].price_mean
        p1 = minutes[end_minute].price_mean
    else:
        p0 = minutes[start_minute].price_last
        p1 = minutes[end_minute].price_last
    r = np.log(p1 / p0)
    return r


def cal_coarse_return(minutes: list[MinuteInfo], start_minute: int, end_minute: int, price_type: PriceType) -> float:
    """Calculate coarse return: abs(sum(return))."""
    assert 0 <= start_minute < end_minute < len(minutes)
    returns = []
    for i in range(start_minute, end_minute):
        j = i + 1
        ret = cal_log_return(minutes, i, j, price_type)
        returns.append(ret)

    ret = np.abs(np.sum(returns))
    return ret


def cal_fine_return(minutes: list[MinuteInfo], start_minute: int, end_minute: int, price_type: PriceType) -> float:
    """Calculate fine return: sum(abs(return))."""
    assert 0 <= start_minute < end_minute < len(minutes)
    returns = []
    for i in range(start_minute, end_minute):
        j = i + 1
        ret = cal_log_return(minutes, i, j, price_type)
        returns.append(np.abs(ret))

    ret = np.sum(returns)
    return ret


def get_minute_info(trade_infos: list[TradeInfo], start_lob: LobSnapshot) -> list[MinuteInfo]:
    """Get minute info."""
    infos = []
    last_price = start_lob.last_price
    for trade_info in trade_infos:
        trans_volume = 0
        if trade_info.transactions and trade_info.transactions[0].type != "C":
            last_price = trade_info.transactions[0].price
            trans_volume = sum([x.volume for x in trade_info.transactions])
        infos.append(
            {
                "time": trade_info.order.time.round("60s"),
                "price": last_price,
                "volume": trans_volume,
                "mid_price": trade_info.lob_snapshot.float_mid_price,
            }
        )
    data = pd.DataFrame(infos)
    minutes_data = (
        data.groupby("time")
        .agg(
            price_mean=("price", "mean"),
            price_mid=("mid_price", "mean"),
            price_mid_last=("mid_price", "last"),
            price_last=("price", "last"),
            volume=("volume", "sum"),
            num_orders=("price", "count"),
        )
        .reset_index()
    )
    minutes: list[MinuteInfo] = []
    row: MinuteInfo
    for row in minutes_data.itertuples():  # type: ignore
        minutes.append(
            MinuteInfo(
                time=row.time,
                price_last=row.price_last,
                price_mean=row.price_mean,
                price_mid=row.price_mid,
                price_mid_last=row.price_mid_last,
                volume=row.volume,
                num_orders=row.num_orders,
            )
        )
    return minutes


def get_rollout_info(path: Path) -> RolloutInfo | None:
    """Get RolloutInfo from trade infos for both replay and simulation."""
    rollouts: list[tuple[list[TradeInfo], LobSnapshot]] = pkl_utils.load_pkl_zstd(path)
    assert rollouts[0] is not None
    replay_trade_infos, _ = rollouts[0]  # real replay
    rollouts = [x for x in rollouts if x is not None]
    if len(rollouts) != 2:
        logging.warning(f"Got {len(rollouts)} rollouts")
        return None
    simulated_trade_infos, start_lob = rollouts[1]
    replay_stylized_fact = get_minute_info(replay_trade_infos, start_lob)
    rollouts_stylized_facts = get_minute_info(simulated_trade_infos, start_lob)
    rollout_info = RolloutInfo(
        symbol=replay_trade_infos[0].order.symbol,
        start_time=start_lob.time,
        simulation_minutes=rollouts_stylized_facts,
        replay_minutes=replay_stylized_fact,
    )
    if len(rollout_info.replay_minutes) != 26:
        logging.warning(f"Got {len(rollout_info.replay_minutes)} replay minutes")
        return None
    if len(rollout_info.simulation_minutes) != 26:
        logging.warning(f"Got {len(rollout_info.simulation_minutes)} simulation minutes")
        return None
    return rollout_info


def get_return_info(
    rollout_infos: list[RolloutInfo],
    delta_ts: list[int],
    taus: list[int],
    price_type: PriceType = "last",
    *,
    need_coarse_fine_info: bool = False,
) -> pd.DataFrame:
    """Get return info."""
    infos = []
    for rollout_info in tqdm(rollout_infos, desc="Extracting return info."):
        symbol = rollout_info.symbol
        replay_minutes = rollout_info.replay_minutes
        simulation_minutes = rollout_info.simulation_minutes
        assert len(replay_minutes) == len(simulation_minutes)
        num_minutes = len(replay_minutes)
        for source, minutes in [("Replay", replay_minutes), ("Simulation", simulation_minutes)]:
            for delta_t in delta_ts:
                for tau in taus:
                    for i_start in range(1, num_minutes - tau):
                        # for i_start in range(1, 2):
                        if i_start + tau + delta_t >= num_minutes:
                            continue
                        if i_start + delta_t >= num_minutes:
                            continue
                        if i_start + tau < 0:
                            continue
                        assert delta_t > 0
                        r1 = cal_log_return(minutes, i_start, i_start + delta_t, price_type)
                        r2 = cal_log_return(minutes, i_start + tau, i_start + tau + delta_t, price_type)

                        info = {
                            "symbol": symbol,
                            "source": source,
                            "lag": tau,
                            "r1": r1,
                            "r2": r2,
                            "delta_t": delta_t,
                            "time": minutes[i_start].time,
                            "volume": minutes[i_start].volume,
                            "start_time": rollout_info.start_time,
                        }
                        if need_coarse_fine_info:
                            info.update(
                                {
                                    "r1_fine": cal_fine_return(minutes, i_start, i_start + delta_t, price_type),
                                    "r2_coarse": cal_coarse_return(minutes, i_start + tau, i_start + tau + delta_t, price_type),
                                }
                            )
                        infos.append(info)

    data = pd.DataFrame(infos)
    return data


def plot_cont1(rollout_infos: list[RolloutInfo], output_dir: Path, price_type: PriceType) -> None:
    """Plot cont1. Absence of autocorrelations: corr(r(t, delta_t), r(t+tau, delta_t))."""
    max_tau = 10
    data = get_return_info(rollout_infos, delta_ts=[1], taus=list(range(1, max_tau + 1)), price_type=price_type)
    groups = data.groupby(["source", "lag", "symbol"])

    def r_corr(x, col1, col2) -> float:  # noqa: ANN001
        return np.corrcoef(x[col1], x[col2])[0, 1]

    corr = groups.apply(lambda x: r_corr(x, "r1", "r2")).reset_index().rename(columns={0: "corr"})
    # group by symbol
    # corr = corr.groupby(["source", "lag"]).agg(corr=("corr", "mean")).reset_index()
    logging.info(f"Correlation: {corr}")
    corr.to_csv(output_dir / "cont1.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(
        data=corr,
        x="lag",
        y="corr",
        hue="source",
        markers=True,
        style="source",
        ax=ax,
        errorbar=ERROR_BAR,
        hue_order=HUE_ORDER,
    )
    # set xaixs labelticks
    ax.set_xticks(range(1, max_tau + 1))
    # ax.set_title(f"Absence of Autocorrelations ({price_type.title()})")
    ax.set_title(f"Auto-Correlation of Returns ({price_type.title()})")
    # ax.axhline(0, color="gray", linestyle="--")
    fig_path = output_dir / f"cont1-{price_type}-absence-of-auto-correlations.pdf"
    save_and_close_fig(fig, fig_path)


def plot_cont2(rollout_infos: list[RolloutInfo], output_dir: Path, price_type: PriceType) -> None:
    """Plot cont2. Heavy tails: kurtosis of returns.

    NOTE: why not use last price: because it's not a unique peak -> there are actually 3 peaks.
    """
    max_delta_t = 20
    data = get_return_info(rollout_infos, delta_ts=list(range(1, max_delta_t + 1)), taus=[0], price_type=price_type)
    data = data[(data["r1"] > -0.5) & (data["r1"] < 0.5)]
    groups = data.groupby(["source", "delta_t", "symbol"])
    kurtosis = groups.apply(lambda x: x["r1"].kurtosis()).reset_index().rename(columns={0: "kurtosis"})
    logging.info(f"Kurtosis: {kurtosis}")
    kurtosis.to_csv(output_dir / "cont2.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(
        data=kurtosis,
        x="delta_t",
        y="kurtosis",
        hue="source",
        markers=True,
        style="source",
        ax=ax,
        errorbar=ERROR_BAR,
        hue_order=HUE_ORDER,
    )
    ax.set_xlabel("Period (Minutes)")
    # ax.set_xticks(range(1, max_delta_t + 1))
    ax.set_xticks(AX_TICK_20)
    # ax.set_title("Heavy Tails and Aggregational Gaussianity")
    ax.set_title("Kurtosis of Returns")
    ax.set_ylim(1, 9)
    fig_path = output_dir / f"cont2-{price_type}-heavy-tails-aggregational-gaussianity.pdf"
    save_and_close_fig(fig, fig_path)

    cont2_detail_dir = output_dir / "cont2-detail"
    cont2_detail_dir.mkdir(parents=True, exist_ok=True)
    # filter outlier

    for delta_t in range(1, max_delta_t + 1):
        sub_data = data[data["delta_t"] == delta_t]
        sub_data = sub_data[(sub_data["r1"] > -0.01) & (sub_data["r1"] < 0.01)]
        logging.info(f"Get {len(sub_data)} data for delta_t = {delta_t} from {len(data)} data.")
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.histplot(data=sub_data, x="r1", hue="source", kde=False, alpha=0.4, stat="probability", bins=129, ax=ax)
        ax.set_xlabel("Return")
        ax.set_title(f"Delta_t = {delta_t}")
        ax.set_xlim(-0.01, 0.01)
        fig_path = cont2_detail_dir / f"cont2-delta_t-{delta_t}-{price_type}.pdf"
        save_and_close_fig(fig, fig_path)


def plot_cont3(rollout_infos: list[RolloutInfo], output_dir: Path) -> None:
    """Plot cont3. Gain/loss asymmetry: skews of returns."""
    max_delta_t = 20
    data = get_return_info(rollout_infos, delta_ts=list(range(1, max_delta_t + 1)), taus=[0])
    data = data[(data["r1"] > -0.05) & (data["r1"] < 0.05)]  # filter outlier
    groups = data.groupby(["source", "delta_t", "symbol"])
    skews = groups.apply(lambda x: x["r1"].skew()).reset_index().rename(columns={0: "skew"})
    logging.info(f"skews: {skews}")
    skews.to_csv(output_dir / "cont3.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(data=skews, x="delta_t", y="skew", hue="source", markers=True, style="source", ax=ax, errorbar=ERROR_BAR)
    ax.set_xlabel("Period (Minutes)")
    ax.set_xticks(AX_TICK_20)
    # ax.set_title("Gain/loss Asymmetry")
    ax.set_title("Skewness of Returns")

    fig_path = output_dir / "cont3-gain-loss-asymmetry.pdf"
    save_and_close_fig(fig, fig_path)


def plot_cont5(rollout_infos: list[RolloutInfo], output_dir: Path) -> None:
    """Plot cont5. Intermittency: Fano factor."""
    data = get_return_info(rollout_infos, delta_ts=[1], taus=[0])
    data["r1_abs"] = np.abs(data["r1"])
    groups = data.groupby(["source", "symbol"])
    # calculate 90% quantile for each group using transform
    data["r1_abs_90"] = groups["r1_abs"].transform(lambda x: x.quantile(0.99))
    data["is_extreme"] = data["r1_abs"] > data["r1_abs_90"]
    # every 60 minutes
    data["time"] = data["time"].dt.floor("60T")
    extreme_counts = data.groupby(["source", "symbol", "time"]).agg(extreme_count=("is_extreme", "sum")).reset_index()
    # calculate mean and variance
    fano_factor_data = (
        extreme_counts.groupby(["source", "symbol"])
        .agg(mean_extreme_count=("extreme_count", "mean"), var_extreme_count=("extreme_count", "var"))
        .reset_index()
    )
    fano_factor_data["fano_factor"] = fano_factor_data["var_extreme_count"] / fano_factor_data["mean_extreme_count"]
    logging.info(f"Fano factor: {fano_factor_data}")
    fano_factor_data.to_csv(output_dir / "cont5.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.boxplot(data=fano_factor_data, x="fano_factor", hue="source", ax=ax, showmeans=True)
    ax.axvline(1, color="gray", linestyle="--")
    # ax.set_title("Intermittency: Fano Factor")
    ax.set_title('Fano Factor of "Burst" Returns')
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    fig_path = output_dir / "cont5-intermittency.pdf"
    save_and_close_fig(fig, fig_path)


def plot_cont6(rollout_infos: list[RolloutInfo], output_dir: Path) -> None:
    """Plot cont5. Volatility clustering: corr(|r(t, delta_t)|, |r(t+tau, delta_t)|)."""
    max_tau = 20
    data = get_return_info(rollout_infos, delta_ts=[1], taus=list(range(1, max_tau + 1)))
    data["r1_abs"] = np.abs(data["r1"])
    data["r2_abs"] = np.abs(data["r2"])
    groups = data.groupby(["source", "lag", "symbol"])

    def r_corr(x, col1, col2) -> float:  # noqa: ANN001
        return np.corrcoef(x[col1], x[col2])[0, 1]

    corr = groups.apply(lambda x: r_corr(x, "r1_abs", "r2_abs")).reset_index().rename(columns={0: "corr"})
    # group by symbol
    # corr = corr.groupby(["source", "lag"]).agg(corr=("corr", "mean")).reset_index()
    logging.info(f"Correlation: {corr}")
    corr.to_csv(output_dir / "cont6.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(data=corr, x="lag", y="corr", hue="source", markers=True, style="source", ax=ax, errorbar=ERROR_BAR)
    # set xaixs labelticks
    ax.set_xticks(AX_TICK_20)
    # ax.set_title("Volatility Clustering and Slow Decay of Absolute Return", fontsize=8)
    ax.set_title("Auto-Correaltion of Absolute Returns")
    fig_path = output_dir / "cont6-volatility-clustering-slow-decay.pdf"
    save_and_close_fig(fig, fig_path)


def plot_cont7(rollout_infos: list[RolloutInfo], output_dir: Path) -> None:
    """Plot cont7. Conditional heavy tails: kurtosis of returns."""
    max_delta_t = 20
    data = get_return_info(rollout_infos, delta_ts=list(range(1, max_delta_t + 1)), taus=[0])
    data = data[(data["r1"] > -0.5) & (data["r1"] < 0.5)]
    data["time"] = data["time"].dt.time
    data["minute_vol"] = data.groupby(["source", "delta_t", "time"])["r1"].transform(lambda x: (x.std()))
    data["r1"] = data["r1"] / data["minute_vol"]
    groups = data.groupby(["source", "delta_t", "symbol"])
    kurtosis = groups.apply(lambda x: x["r1"].kurtosis()).reset_index().rename(columns={0: "kurtosis"})
    logging.info(f"Kurtosis: {kurtosis}")
    kurtosis.to_csv(output_dir / "cont7.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(
        data=kurtosis,
        x="delta_t",
        y="kurtosis",
        hue="source",
        markers=True,
        style="source",
        ax=ax,
        errorbar=ERROR_BAR,
        hue_order=HUE_ORDER,
    )
    ax.set_xlabel("Period (Minutes)")
    ax.set_xticks(AX_TICK_20)
    ax.set_ylim(1, 9)
    # ax.set_title("Conditional Heavy Tails")
    ax.set_title("Kurtosis of Returns (Normalized)")
    fig_path = output_dir / "cont7-conditional-heavy-tails.pdf"
    save_and_close_fig(fig, fig_path)

    # output details for cont7
    cont7_detail_dir = output_dir / "cont7-detail"
    cont7_detail_dir.mkdir(parents=True, exist_ok=True)
    for delta_t in range(1, max_delta_t + 1):
        sub_data = data[data["delta_t"] == delta_t]
        logging.info(f"Get {len(sub_data)} data for delta_t = {delta_t} from {len(data)} data.")
        if len(sub_data) == 0:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        # clip sub_data with quantile 0.05 and 0.95
        quantile_5 = sub_data["r1"].quantile(0.05)
        quantile_95 = sub_data["r1"].quantile(0.95)
        sub_data = sub_data[(sub_data["r1"] > quantile_5) & (sub_data["r1"] < quantile_95)]
        sns.histplot(data=sub_data, x="r1", hue="source", kde=False, alpha=0.4, stat="probability", bins=129, ax=ax)
        ax.set_xlabel("Return")
        ax.set_title(f"Delta_t = {delta_t}")
        fig_path = cont7_detail_dir / f"cont7-delta_t-{delta_t}.pdf"
        save_and_close_fig(fig, fig_path)


def plot_cont9(rollout_infos: list[RolloutInfo], output_dir: Path) -> None:
    """Plot cont9. Leverage effect: corr(r(t, delta_t), |r(t+tau, delta_t)|)."""
    max_tau = 20
    data = get_return_info(rollout_infos, delta_ts=[1], taus=list(range(1, max_tau + 1)))
    data["r2_abs"] = np.abs(data["r2"])
    groups = data.groupby(["source", "lag", "symbol"])

    def r_corr(x, col1, col2) -> float:  # noqa: ANN001
        return np.corrcoef(x[col1], x[col2])[0, 1]

    corr = groups.apply(lambda x: r_corr(x, "r1", "r2_abs")).reset_index().rename(columns={0: "corr"})
    # group by symbol
    # corr = corr.groupby(["source", "lag"]).agg(corr=("corr", "mean")).reset_index()
    logging.info(f"Correlation: {corr}")
    corr.to_csv(output_dir / "cont9.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(data=corr, x="lag", y="corr", hue="source", markers=True, style="source", ax=ax, errorbar=ERROR_BAR)
    # set xaixs labelticks
    ax.set_xticks(AX_TICK_20)
    # ax.set_title("Leverage Effect")
    ax.set_title("Correlation between Return and Lagged Volatility", fontsize=8)

    fig_path = output_dir / "cont9-leverage-effect.pdf"
    save_and_close_fig(fig, fig_path)


def plot_cont10(rollout_infos: list[RolloutInfo], output_dir: Path) -> None:
    """Plot cont10. Volume/volatility correlation: corr(v(t, delta_t), |r(t, delta_t)|)."""
    max_tau = 20
    data = get_return_info(rollout_infos, delta_ts=[1], taus=list(range(1, max_tau + 1)))
    data["r2_abs"] = np.abs(data["r2"])
    groups = data.groupby(["source", "lag", "symbol"])

    def r_corr(x, col1, col2) -> float:  # noqa: ANN001
        return np.corrcoef(x[col1], x[col2])[0, 1]

    corr = groups.apply(lambda x: r_corr(x, "volume", "r2_abs")).reset_index().rename(columns={0: "corr"})
    logging.info(f"Correlation: {corr}")
    corr.to_csv(output_dir / "cont10.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(data=corr, x="lag", y="corr", hue="source", markers=True, style="source", ax=ax, errorbar=ERROR_BAR)
    # set xaixs labelticks
    ax.set_xticks(AX_TICK_20)
    # ax.set_title("Volume/Volatility Correlation")
    ax.set_title("Correlation between Volume and Lagged Volatility", fontsize=8)
    fig_path = output_dir / "cont10-volume-volatility-corr.pdf"
    save_and_close_fig(fig, fig_path)


def plot_cont11(rollout_infos: list[RolloutInfo], output_dir: Path) -> None:
    """Plot cont11. Asymmetry in timescales: coarse grain and fine grain returnn correlation."""
    data = get_return_info(rollout_infos, delta_ts=[5], taus=list(range(-10, 10 + 1)), need_coarse_fine_info=True)
    groups = data.groupby(["source", "lag", "symbol"])
    corr = groups.apply(lambda x: x["r2_coarse"].corr(x["r1_fine"])).reset_index().rename(columns={0: "cf_corr"})
    logging.info(f"Correlation: {corr}")
    corr.to_csv(output_dir / "cont11.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.5), nrows=1, ncols=1)
    sns.lineplot(
        data=corr,
        x="lag",
        y="cf_corr",
        hue="source",
        markers=True,
        style="source",
        ax=ax,
        errorbar=ERROR_BAR,
        hue_order=HUE_ORDER,
    )
    # set xaixs labelticks
    ax.set_xticks(list(range(-10, 11, 2)))
    # ax.set_title("Asymmetry in Timescales")
    ax.set_title("Correlation between Coarse and Fine Grain Volatilities", fontsize=7)

    corr = corr.set_index(["source", "lag", "symbol"])
    diff_info = []
    for idx, row_data in corr.iterrows():
        source, lag, symbol = idx  # Unpack the multi-index tuple
        if lag < 0:
            continue
        diff_info.append(
            {
                "source": source,
                "lag": lag,
                "symbol": symbol,
                "diff": row_data.cf_corr - corr.loc[(source, -lag, symbol)].cf_corr,  # type: ignore
            }
        )
    diff_data = pd.DataFrame(diff_info)
    sns.lineplot(
        data=diff_data,
        x="lag",
        y="diff",
        hue="source",
        markers=True,
        style="source",
        ax=ax,
        errorbar=ERROR_BAR,
        hue_order=HUE_ORDER,
        legend=False,
    )
    ax.set_ylabel("Corr")
    ax.axvline(0, color="gray", linestyle="--")
    ax.axhline(0, color="gray", linestyle="solid")
    fig_path = output_dir / "cont11-asymmetry-in-timescales.pdf"
    save_and_close_fig(fig, fig_path)


def plot_data_distribution(rollouts: list[RolloutInfo], output_dir: Path) -> None:
    """Plot date distribution."""
    total_count = len(rollouts)
    dates = [x.start_time.date() for x in rollouts]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.histplot(dates, bins=30, ax=ax)
    ax.set_title(f"Date Distribution (Total Simulation: {total_count})")
    # show the xticks with interval for every 3 months, from first month to last month in dates, rotate the xtickslabel with 45 degree
    min_date = min(dates)
    max_date = max(dates)
    xticks = [min_date + pd.DateOffset(months=i) for i in range(0, 50, 1)]
    xticks = [x for x in xticks if x.date() <= max_date]  # type: ignore
    ax.set_xticks(xticks)  # type: ignore
    ax.set_xticklabels([x.strftime("%Y-%m-%d") for x in xticks], rotation=45)
    fig_path = output_dir / "date-distribution.pdf"
    logging.info(f"First date: {min_date}, Last date: {max_date}")
    save_and_close_fig(fig, fig_path)

    symbols = [x.symbol for x in rollouts]
    total_symbols = len(set(symbols))
    symbol_counts = pd.Series(symbols).value_counts()
    # plot count distribution
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.histplot(symbol_counts.values, bins=30, ax=ax)  # type: ignore
    ax.set_title(f"Instrument Distribution (Total Symbols: {total_symbols})")
    ax.set_xlabel("# of Simulaton")
    ax.set_ylabel("# of Instruments")
    fig_path = output_dir / "symbol-distribution.pdf"
    save_and_close_fig(fig, fig_path)


def generate_stylized_facts(path: Path, output_dir: Path) -> None:
    """Generate stylized facts."""
    rollouts: list[RolloutInfo] = pkl_utils.load_pkl_zstd(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Found {len(rollouts)} rollouts in {path}")
    plot_cont1(rollouts, output_dir, price_type="last")
    plot_cont1(rollouts, output_dir, price_type="mean")
    plot_cont2(rollouts, output_dir, "last")
    plot_cont3(rollouts, output_dir)
    # # cont4: Aggregational Gaussianity: kurtosis of returns.
    plot_cont5(rollouts, output_dir)
    plot_cont6(rollouts, output_dir)
    plot_cont7(rollouts, output_dir)
    # # cont8: Slow decay of autocorrelation in absolute returns
    plot_cont9(rollouts, output_dir)
    plot_cont10(rollouts, output_dir)
    plot_cont11(rollouts, output_dir)
    plot_data_distribution(rollouts, output_dir)


if __name__ == "__main__":
    # Report Cont's 11 stylized facts from RolloutInfos.
    # Note: this function turns RolloutInfos into stylized facts,
    # you probably need to convert your trade infos to RolloutInfos first,
    # which includes real historical data and we can not provide them due to data license.
    # To get the RolloutInfos from trade infos, please check out the function `get_rollout_info` in this file.

    rollout_info_path: Path = Path("/data/blob_root/mars-open/stylized-facts/rollout_info_25_minutes.zstd")
    output_dir: Path = Path("./tmp/stylized-facts")

    generate_stylized_facts(rollout_info_path, output_dir)
