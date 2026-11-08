from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "backtest_predictions_v1.csv"
FORWARD_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "forward_scores_matches.csv"
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_closing_odds_v1.csv"
FORWARD_MODEL_VERSION = "v1"
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
PROBABILITY_COLUMNS = ["p_home", "p_draw", "p_away"]
MARKET_COLUMNS = ["bookie_home", "bookie_draw", "bookie_away"]
LEAGUE_NAMES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "B1": "Pro League",
    "SC0": "Premiership",
    "T1": "Süper Lig",
}
LOG_LOSS_EPSILON = 1e-15
PLOT_BACKGROUND = "#0e141b"
PLOT_GRID = "rgba(255, 255, 255, 0.05)"
ACCENT = "#43d9b5"
PLOT_FONT_FAMILY = "sans-serif"


def proportional_devig(values: Sequence[float]) -> np.ndarray:
    """Normalize exported market probabilities or convert decimal odds to probabilities."""
    numbers = np.asarray(values, dtype=float)
    if numbers.shape != (3,) or not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError("Market values must be three finite, non-negative numbers.")

    if (numbers > 1).all():
        implied_probabilities = 1 / numbers
    elif (numbers <= 1).all():
        implied_probabilities = numbers
    else:
        raise ValueError("Market values must be all probabilities or all decimal odds.")

    booksum = float(implied_probabilities.sum())
    if booksum <= 0:
        raise ValueError("Market probabilities must have a positive sum.")
    return implied_probabilities / booksum


def calculate_log_loss(probabilities: Sequence[float], actual_result: str) -> float:
    """Return the negative log probability assigned to the observed 1X2 outcome."""
    outcome_index = OUTCOME_INDEX.get(actual_result)
    if outcome_index is None:
        raise ValueError(f"Unknown actual result: {actual_result!r}")

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be three finite, non-negative values.")

    probability_sum = float(values.sum())
    if probability_sum <= 0:
        raise ValueError("Probabilities must have a positive sum.")
    values = values / probability_sum
    return -float(np.log(np.clip(values[outcome_index], LOG_LOSS_EPSILON, 1.0)))


def _market_values_are_valid(values: np.ndarray) -> np.ndarray:
    probabilities = np.all((values >= 0) & (values <= 1), axis=1)
    decimal_odds = np.all(values > 1, axis=1)
    probabilities &= values.sum(axis=1) > 0
    return np.isfinite(values).all(axis=1) & (probabilities | decimal_odds)


def _prediction_values_are_valid(values: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(values).all(axis=1)
        & np.all((values >= 0) & (values <= 1), axis=1)
        & (values.sum(axis=1) > 0)
    )


@st.cache_data(show_spinner=False)
def bootstrap_cumulative_band(
    edges: np.ndarray,
    n_checkpoints: int = 60,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate paired_logloss_diff_ci for cumulative sums of edge prefixes."""
    edges = np.asarray(edges, dtype=float)
    n = len(edges)
    if n < 10:
        return np.full(n, np.nan), np.full(n, np.nan)

    checkpoints = np.unique(np.rint(np.linspace(10, n, n_checkpoints)).astype(int))
    rng = np.random.default_rng(seed)
    lower_checkpoints = []
    upper_checkpoints = []
    for checkpoint in checkpoints:
        boot_sums = []
        for start in range(0, n_boot, 250):
            resamples = min(250, n_boot - start)
            indices = rng.integers(0, checkpoint, size=(resamples, checkpoint))
            boot_sums.extend(edges[:checkpoint][indices].sum(axis=1))
        lower, upper = np.percentile(boot_sums, [2.5, 97.5])
        lower_checkpoints.append(lower)
        upper_checkpoints.append(upper)

    positions = np.arange(1, n + 1)
    lower = np.interp(positions, checkpoints, lower_checkpoints, left=np.nan)
    upper = np.interp(positions, checkpoints, upper_checkpoints, left=np.nan)
    return lower, upper

@st.cache_data(show_spinner=False)
def bootstrap_mean_interval(
    edges: np.ndarray, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float]:
    numbers = np.asarray(edges, dtype=float)
    if len(numbers) < 10:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(numbers)
    for start in range(0, n_boot, 500):
        batch = min(500, n_boot - start)
        indices = rng.integers(0, n, size=(batch, n))
        boot_means.extend(numbers[indices].mean(axis=1))
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(lower), float(upper)


@st.cache_data(show_spinner=False)
def scoring_rule_edges(
    betty: np.ndarray, market: np.ndarray, outcome_idx: np.ndarray
) -> dict[str, np.ndarray]:
    betty_probs = np.asarray(betty, dtype=float)
    market_probs = np.asarray(market, dtype=float)
    outcomes = np.asarray(outcome_idx, dtype=int)
    n = len(outcomes)

    one_hot = np.zeros((n, 3), dtype=float)
    one_hot[np.arange(n), outcomes] = 1.0

    brier_betty = np.sum((betty_probs - one_hot) ** 2, axis=1)
    brier_market = np.sum((market_probs - one_hot) ** 2, axis=1)

    cum_betty = np.cumsum(betty_probs[:, :2], axis=1)
    cum_market = np.cumsum(market_probs[:, :2], axis=1)
    cum_one_hot = np.cumsum(one_hot[:, :2], axis=1)

    rps_betty = 0.5 * np.sum((cum_betty - cum_one_hot) ** 2, axis=1)
    rps_market = 0.5 * np.sum((cum_market - cum_one_hot) ** 2, axis=1)

    return {
        "brier_betty": brier_betty,
        "brier_market": brier_market,
        "rps_betty": rps_betty,
        "rps_market": rps_market,
    }


@st.cache_data(show_spinner=False)
def calibration_bins(
    predicted: np.ndarray, occurred: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = np.asarray(predicted, dtype=float)
    occ = np.asarray(occurred, dtype=int)
    if len(pred) == 0:
        return (
            pd.DataFrame(columns=["mean_predicted", "actual_rate", "n", "lower", "upper"]),
            pd.DataFrame(columns=["predicted", "occurred"]),
        )

    data = pd.DataFrame({"predicted": pred, "occurred": occ})
    data["probability_bin"] = pd.cut(
        data["predicted"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    agg = (
        data.groupby("probability_bin", observed=True)
        .agg(
            mean_predicted=("predicted", "mean"),
            actual_rate=("occurred", "mean"),
            n=("occurred", "size"),
        )
        .reset_index(drop=True)
    )
    if agg.empty:
        agg["lower"] = []
        agg["upper"] = []
        return agg, data

    p = agg["actual_rate"].to_numpy(dtype=float)
    n = agg["n"].to_numpy(dtype=float)
    z = 1.96
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * np.sqrt(np.maximum(0.0, p * (1.0 - p) / n + z2 / (4.0 * (n ** 2)))) / denom
    agg["lower"] = center - half
    agg["upper"] = center + half
    return agg, data


@st.cache_data(show_spinner=False)
def calibration_slope_intercept(
    predicted: np.ndarray,
    observed: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float | tuple[float, float]]:
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if len(pred) == 0:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "slope_ci": (0.0, 0.0),
            "intercept_ci": (0.0, 0.0),
            "slope_se": 0.0,
            "intercept_se": 0.0,
        }

    x = pred
    y = obs
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev = x - x_mean
    var_x = np.dot(x_dev, x_dev)
    if var_x == 0.0:
        slope = 0.0
        intercept = float(y_mean)
    else:
        slope = float(np.dot(x_dev, y - y_mean) / var_x)
        intercept = float(y_mean - slope * x_mean)

    rng = np.random.default_rng(seed)
    boot_slopes = np.empty(n_boot, dtype=float)
    boot_intercepts = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi = x[idx]
        yi = y[idx]
        xi_mean = xi.mean()
        yi_mean = yi.mean()
        xi_dev = xi - xi_mean
        var_xi = np.dot(xi_dev, xi_dev)
        if var_xi == 0.0:
            s = 0.0
            inc = yi_mean
        else:
            s = np.dot(xi_dev, yi - yi_mean) / var_xi
            inc = yi_mean - s * xi_mean
        boot_slopes[i] = s
        boot_intercepts[i] = inc

    s_ci = np.percentile(boot_slopes, [2.5, 97.5])
    i_ci = np.percentile(boot_intercepts, [2.5, 97.5])

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_ci": (float(s_ci[0]), float(s_ci[1])),
        "intercept_ci": (float(i_ci[0]), float(i_ci[1])),
        "slope_se": float(np.std(boot_slopes, ddof=1)),
        "intercept_se": float(np.std(boot_intercepts, ddof=1)),
    }


@st.cache_data(show_spinner=False)
def compute_edge_candles(edges: np.ndarray, window: int = 50) -> pd.DataFrame:
    edge_vals = np.asarray(edges, dtype=float)
    if len(edge_vals) == 0:
        return pd.DataFrame(
            columns=["window_index", "open", "high", "low", "close", "is_complete"]
        )

    cum_edge = np.cumsum(edge_vals)
    n = len(cum_edge)
    n_windows = int(np.ceil(n / window))

    records = []
    prev_close = 0.0

    for i in range(n_windows):
        start = i * window
        end = min((i + 1) * window, n)
        w_cum = cum_edge[start:end]
        is_complete = bool(len(w_cum) == window)

        w_open = prev_close
        w_close = float(w_cum[-1])
        all_vals = np.append(w_cum, w_open)
        w_high = float(np.max(all_vals))
        w_low = float(np.min(all_vals))

        records.append(
            {
                "window_index": i + 1,
                "open": w_open,
                "high": w_high,
                "low": w_low,
                "close": w_close,
                "is_complete": is_complete,
            }
        )
        prev_close = w_close

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def compute_rolling_edge(edges: np.ndarray, window: int = 100) -> pd.DataFrame:
    series = pd.Series(np.asarray(edges, dtype=float))
    if len(series) < window:
        return pd.DataFrame(
            columns=["match_index", "rolling_edge", "lower_band", "upper_band"]
        )

    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    margin = 1.96 * (rolling_std / np.sqrt(window))

    df = pd.DataFrame(
        {
            "match_index": np.arange(1, len(series) + 1),
            "rolling_edge": rolling_mean,
            "lower_band": rolling_mean - margin,
            "upper_band": rolling_mean + margin,
        }
    )
    return df.dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def prepare_betty_picks(df: pd.DataFrame) -> pd.DataFrame:
    """Add Betty top pick, devigged closing odds, Kelly fractions, and actual outcome odds bracket."""
    res = df.copy()
    betty_probs = res[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    market_probs = res[MARKET_COLUMNS].to_numpy(dtype=float)
    top_pick_indices = np.argmax(betty_probs, axis=1)
    actual_indices = res["actual_result"].map(OUTCOME_INDEX).to_numpy(dtype=int)

    outcome_labels = np.array(["H", "D", "A"])
    res["betty_top_pick"] = outcome_labels[top_pick_indices]
    betty_prob = betty_probs[np.arange(len(res)), top_pick_indices]
    res["betty_prob"] = betty_prob
    market_odds = 1.0 / market_probs[np.arange(len(res)), top_pick_indices]
    res["top_market_odds"] = market_odds
    res["pick_won"] = (top_pick_indices == actual_indices)

    raw_cols = ["AvgCH", "AvgCD", "AvgCA"]
    if all(col in res.columns for col in raw_cols):
        raw_matrix = res[raw_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        raw_odds = raw_matrix[np.arange(len(res)), top_pick_indices]
        res["raw_market_odds"] = pd.Series(raw_odds, index=res.index).fillna(res["top_market_odds"])
    else:
        res["raw_market_odds"] = res["top_market_odds"]

    full_kelly = np.where(
        market_odds > 1.0,
        ((betty_prob * market_odds) - 1.0) / (market_odds - 1.0),
        0.0,
    )
    res["full_kelly"] = full_kelly
    res["quarter_kelly"] = np.where(full_kelly > 0.0, full_kelly * 0.25, 0.0)

    actual_outcome_prob = market_probs[np.arange(len(res)), actual_indices]
    res["market_odds"] = 1.0 / actual_outcome_prob
    res["odds_bracket"] = pd.cut(
        res["market_odds"],
        bins=[1.0, 1.5, 2.0, 3.0, 100.0],
        labels=["< 1.5 (Heavy Fav)", "1.5-2.0 (Fav)", "2.0-3.0 (Dog)", "> 3.0 (Longshot)"],
        include_lowest=True,
    )
    res["pick_odds_bracket"] = pd.cut(
        res["top_market_odds"],
        bins=[1.0, 1.5, 2.0, 3.0, 100.0],
        labels=["< 1.5 (Heavy Fav)", "1.5-2.0 (Fav)", "2.0-3.0 (Dog)", "> 3.0 (Longshot)"],
        include_lowest=True,
    )
    return res


@st.cache_data(show_spinner=False)
def simulate_bankroll_and_drawdown(
    filtered_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Quarter Kelly bankroll, flat bankroll, and drawdown curve for filtered matches."""
    if filtered_df.empty:
        return np.array([]), np.array([]), np.array([])

    cum_edge = filtered_df["edge"].cumsum()
    drawdown = (cum_edge - cum_edge.cummax()).to_numpy(dtype=float)

    qk_bankroll = []
    flat_bankroll = []
    current_qk = 100.0
    current_flat = 100.0

    raw_odds_series = (
        filtered_df["raw_market_odds"]
        if "raw_market_odds" in filtered_df.columns
        else filtered_df["top_market_odds"]
    )

    for has_edge, qk_frac, raw_odds, won in zip(
        filtered_df["quarter_kelly"] > 0,
        filtered_df["quarter_kelly"],
        raw_odds_series,
        filtered_df["pick_won"],
    ):
        if has_edge:
            if won:
                current_qk += current_qk * qk_frac * (raw_odds - 1.0)
                current_flat += (raw_odds - 1.0)
            else:
                current_qk -= current_qk * qk_frac
                current_flat -= 1.0
        qk_bankroll.append(current_qk)
        flat_bankroll.append(current_flat)

    return np.array(qk_bankroll), np.array(flat_bankroll), drawdown


@st.cache_data(show_spinner=False, max_entries=1)
def load_and_process_data(
    data_path: str,
    data_version: tuple[int, int],
    forward_data_path: str,
    forward_data_version: tuple[int, int],
) -> tuple[pd.DataFrame, int]:
    """Load and score the backtest and Core (v1) forward rows to match, caching by file metadata."""
    del data_version, forward_data_version
    data = pd.read_csv(data_path)
    forward = pd.read_csv(forward_data_path)
    required_columns = {
        *PROBABILITY_COLUMNS,
        *MARKET_COLUMNS,
        "actual_result",
        "league",
        "date",
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "The backtest export is missing required columns: "
            + ", ".join(missing_columns)
        )

    forward_required_columns = {
        *PROBABILITY_COLUMNS,
        *MARKET_COLUMNS,
        "FTR",
        "league",
        "match_date",
        "model_version",
    }
    missing_forward_columns = sorted(forward_required_columns.difference(forward.columns))
    if missing_forward_columns:
        raise ValueError(
            "The forward export is missing required columns: "
            + ", ".join(missing_forward_columns)
        )

    data["phase"] = "Backtest"
    forward = forward.loc[
        forward["model_version"].astype("string").str.lower() == FORWARD_MODEL_VERSION
    ]
    forward = forward.rename(
        columns={"FTR": "actual_result", "match_date": "date"}
    ).copy()
    forward["phase"] = "Forward"
    forward["bookie_valid_1x2"] = True
    data = pd.concat([data, forward], ignore_index=True, sort=False)

    for column in PROBABILITY_COLUMNS + MARKET_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["actual_result"] = data["actual_result"].astype("string").str.strip().str.upper()
    data["league"] = data["league"].astype("string").fillna("Unknown")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    market_values = data[MARKET_COLUMNS].to_numpy(dtype=float)
    betty_values = data[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    valid_market = _market_values_are_valid(market_values)
    if "bookie_valid_1x2" in data.columns:
        valid_market &= (
            data["bookie_valid_1x2"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
            .to_numpy()
        )
    valid_predictions = _prediction_values_are_valid(betty_values)
    valid_results = data["actual_result"].isin(OUTCOME_INDEX).to_numpy()
    valid_dates = data["date"].notna().to_numpy()
    valid_rows = valid_market & valid_predictions & valid_results & valid_dates
    scored = data.loc[valid_rows].copy().reset_index(drop=True)

    if scored.empty:
        raise ValueError("No valid prediction rows are available for scoring.")

    scored = scored.sort_values("date", kind="stable").reset_index(drop=True)
    backtest_count = int((scored["phase"] == "Backtest").sum())
    if not scored.iloc[:backtest_count]["phase"].eq("Backtest").all():
        raise ValueError("Backtest rows must precede forward rows for the edge curve band.")
    scored[MARKET_COLUMNS] = np.vstack(
        [proportional_devig(row) for row in scored[MARKET_COLUMNS].to_numpy(dtype=float)]
    )
    betty_arr = scored[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    scored[PROBABILITY_COLUMNS] = betty_arr / betty_arr.sum(axis=1, keepdims=True)

    actual_indices = scored["actual_result"].map(OUTCOME_INDEX).to_numpy(dtype=int)
    rows_range = np.arange(len(scored))

    betty_clipped = np.clip(
        scored[PROBABILITY_COLUMNS].to_numpy(dtype=float),
        LOG_LOSS_EPSILON,
        1.0 - LOG_LOSS_EPSILON,
    )
    betty_clipped = betty_clipped / betty_clipped.sum(axis=1, keepdims=True)
    scored["betty_log_loss"] = -np.log(betty_clipped[rows_range, actual_indices])

    market_clipped = np.clip(
        scored[MARKET_COLUMNS].to_numpy(dtype=float),
        LOG_LOSS_EPSILON,
        1.0 - LOG_LOSS_EPSILON,
    )
    market_clipped = market_clipped / market_clipped.sum(axis=1, keepdims=True)
    scored["market_log_loss"] = -np.log(market_clipped[rows_range, actual_indices])
    scored["edge"] = scored["market_log_loss"] - scored["betty_log_loss"]
    scored["cumulative_edge"] = scored["edge"].cumsum()
    rolling_columns = ["roll_home", "roll_draw", "roll_away"]
    scored[rolling_columns] = (
        scored[PROBABILITY_COLUMNS]
        .rolling(window=100, min_periods=1)
        .mean()
        .set_axis(rolling_columns, axis="columns")
    )
    raw_odds_path = Path(data_path).parent / "raw_closing_odds_v1.csv"
    if not raw_odds_path.exists():
        raw_odds_path = RAW_DATA_PATH
    if not raw_odds_path.exists():
        raw_odds_path = Path("data/raw_closing_odds_v1.csv")
    if raw_odds_path.exists():
        raw_odds = pd.read_csv(raw_odds_path)
        scored = scored.merge(raw_odds, on="match_id", how="left")
    return scored, len(data) - len(scored)


st.set_page_config(page_title="Betty Terminal", layout="wide")
st.title("Betty Terminal")
st.caption("Match outcome analytics · read-only backtest export")

try:
    data_stat = DATA_PATH.stat()
    forward_data_stat = FORWARD_DATA_PATH.stat()
    scored, dropped_rows = load_and_process_data(
        str(DATA_PATH),
        (data_stat.st_mtime_ns, data_stat.st_size),
        str(FORWARD_DATA_PATH),
        (forward_data_stat.st_mtime_ns, forward_data_stat.st_size),
    )
except FileNotFoundError:
    st.error(
        f"Backtest or forward export not found: {DATA_PATH} or {FORWARD_DATA_PATH}",
        icon=":material/error:",
    )
    st.stop()
except ValueError as error:
    st.error(str(error), icon=":material/error:")
    st.stop()

if dropped_rows:
    st.warning(
        f"Excluded {dropped_rows:,} rows with invalid market values, Betty probabilities, "
        "or actual results.",
        icon=":material/warning:",
    )

forward_matches = scored.loc[scored["phase"] == "Forward"]
forward_edge = float(forward_matches["edge"].sum())
backtest_count = int((scored["phase"] == "Backtest").sum())
backtest_edges = scored.loc[scored["phase"] == "Backtest", "edge"].to_numpy(dtype=float)
band_lower, band_upper = bootstrap_cumulative_band(backtest_edges)
st.caption(f"Betty Core (v1) | Backtest: {backtest_count:,} matches | Forward: live scored matches | Market: de-vigged closing odds")
if not forward_matches.empty:
    latest_forward_date = forward_matches["date"].dropna().max()
    if pd.notna(latest_forward_date):
        st.caption(f"Data through {latest_forward_date:%Y-%m-%d}")

metric_columns = st.columns(5)
with metric_columns[0]:
    st.metric("Total matches", f"{len(scored):,}", border=True)
with metric_columns[1]:
    st.metric("Betty log loss", f"{scored['betty_log_loss'].mean():.4f}", border=True)
with metric_columns[2]:
    st.metric("Market log loss", f"{scored['market_log_loss'].mean():.4f}", border=True)
with metric_columns[3]:
    st.metric(
        "Cumulative edge",
        f"{scored['cumulative_edge'].iloc[-1]:+.2f}",
        help="Cumulative market log loss minus Betty log loss; positive means Betty outperforms the market.",
        border=True,
    )
with metric_columns[4]:
    st.metric(
        "Forward Test Matches",
        f"{len(forward_matches):,}",
        delta=f"{forward_edge:+.2f} forward edge",
        help="Forward edge is market log loss minus Betty log loss; positive favors Betty.",
        border=True,
    )

forward_edges = forward_matches["edge"].to_numpy(dtype=float)
forward_n = len(forward_edges)
if forward_n >= 10:
    mean_edge = float(forward_edges.mean())
    lower, upper = bootstrap_mean_interval(forward_edges)

with st.container(border=True):
    st.subheader("Forward test progress")
    st.progress(
        min(forward_n / 300, 1.0),
        text=f"{forward_n:,} of 300 forward matches (first review). Second review at 1,000.",
    )
    if forward_n >= 10:
        st.caption(
            f"Forward edge per match: {mean_edge:+.4f} "
            f"(95% paired-bootstrap interval {lower:+.4f} to {upper:+.4f}, n={forward_n:,})"
        )
        if lower > 0:
            st.caption("Interval entirely above zero.")
        elif upper < 0:
            st.caption("Interval entirely below zero.")
        else:
            st.caption("Interval includes zero: inconclusive, not evidence for or against an edge.")
    else:
        st.caption("Too few forward matches for an interval.")
    with st.expander("Method and pre-registered rule"):
        st.markdown(
            "1. Frozen until the first review: model (Betty Core v1), benchmark "
            "(proportional devig of the Football-Data average closing odds, AvgC; Shin devig was tested as a sensitivity, see point 6) "
            "and filters (none; all matches).\n"
            "2. Edge = market log loss minus Betty log loss per match; positive favours Betty.\n"
            "3. Review at 300 forward matches: an interval entirely above zero counts as evidence of an edge. "
            "Anything else is inconclusive, not proof of no edge. If inconclusive, collection continues to a second review at 1,000 matches.\n"
            "4. Sensitivity: with the backtest spread of about 0.17 log loss per match, 300 matches can reliably detect only a true edge of about +0.028 per match or more, and 1,000 matches about +0.015. Smaller edges will usually look inconclusive.\n"
            "5. Backtest reference: Betty trails the market by about 0.022 log loss per match over 6,534 matches, with a 95% interval of about -0.026 to -0.018.\n"
            "6. Shin sensitivity (backtest only): with a Shin devig of the raw Football-Data average closing odds, the edge is -0.0227 per match (95% interval -0.0270 to -0.0184), against -0.0219 with the proportional devig used above. The difference is 0.0008 per match, no league changes sign, and 2 of 6,534 closing-odds rows with an implausible margin (below 1.0) do not change either result. Shin is not applied to the forward test, which has no raw odds in this export.\n"
            "7. Blend and time split (backtest only): the best log-loss mix of Betty and the market puts weight 0.000 on Betty (fitted on the whole backtest and bounded at 0, so the interval [0.000, 0.000] reflects the boundary, not perfect certainty). Betty's edge is -0.0232 on the older half (n=3,267) and -0.0206 on the newer half (n=3,267); the difference is within noise.\n"
            "8. Timing: forward predictions are made 3.8 to 13.8 hours before kickoff. The capture time of the closing odds is not independently verified.\n"
            "9. This is a statistical research display, not betting advice."
        )

with st.container(border=True):
    st.subheader("Scoring rules")
    st.caption(
        "Log loss is the headline metric and the metric of the pre-registered rule. "
        "Brier score and RPS are supporting views of the same predictions. "
        "Lower scores are better; edge = market score minus Betty score, positive favours Betty. "
        "Intervals: 95% paired bootstrap, seed 0."
    )
    scoring_rows = []
    for phase_name in ["Backtest", "Forward"]:
        phase_subset = scored[scored["phase"] == phase_name]
        n_phase = len(phase_subset)
        if n_phase == 0:
            continue

        b_probs = phase_subset[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        b_probs = b_probs / b_probs.sum(axis=1, keepdims=True)
        m_probs = phase_subset[MARKET_COLUMNS].to_numpy(dtype=float)
        o_indices = np.array([OUTCOME_INDEX[r] for r in phase_subset["actual_result"]], dtype=int)

        rules = scoring_rule_edges(b_probs, m_probs, o_indices)

        metric_specs = [
            (
                "Log loss",
                phase_subset["betty_log_loss"].to_numpy(dtype=float),
                phase_subset["market_log_loss"].to_numpy(dtype=float),
            ),
            ("Brier score", rules["brier_betty"], rules["brier_market"]),
            ("RPS", rules["rps_betty"], rules["rps_market"]),
        ]

        for metric_name, b_vals, m_vals in metric_specs:
            b_mean = float(np.mean(b_vals))
            m_mean = float(np.mean(m_vals))
            e_vals = m_vals - b_vals
            e_mean = float(np.mean(e_vals))

            if n_phase >= 10:
                low_ci, high_ci = bootstrap_mean_interval(e_vals)
                ci_text = f"{low_ci:+.4f} to {high_ci:+.4f}"
            else:
                ci_text = "n/a"

            scoring_rows.append({
                "Sample": phase_name,
                "Metric": metric_name,
                "Betty": f"{b_mean:.4f}",
                "Market": f"{m_mean:.4f}",
                "Edge per match": f"{e_mean:+.4f}",
                "95% interval": ci_text,
                "n": f"{n_phase:,}",
            })

    scoring_df = pd.DataFrame(scoring_rows)
    st.dataframe(scoring_df, hide_index=True, width="stretch")

with st.container(border=True):
    st.subheader("Edge curve")
    st.caption(
        "Cumulative market log loss minus Betty log loss; positive values favor Betty. "
        "Band: pointwise 95% paired bootstrap of the backtest, assuming independent matches; "
        "not shown for the forward test."
    )
    edge_figure = go.Figure()
    edge_x = np.arange(1, len(scored) + 1)
    edge_figure.add_trace(
        go.Scatter(
            x=edge_x[:backtest_count],
            y=band_upper,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    edge_figure.add_trace(
        go.Scatter(
            x=edge_x[:backtest_count],
            y=band_lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(255, 255, 255, 0.05)",
            name="95% paired-bootstrap band",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    edge_figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=scored["cumulative_edge"],
            mode="lines",
            line={"color": ACCENT, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(67, 217, 181, 0.10)",
            name="Cumulative edge",
            hovertemplate="Match %{x}<br>Cumulative edge %{y:.3f}<extra></extra>",
        )
    )
    edge_figure.add_hline(y=0, line_color="#72808c", line_dash="dot")
    edge_figure.add_vline(
        x=backtest_count + 0.5,
        line_color="#f0b858",
        line_dash="dash",
        annotation_text="Forward Test Started",
        annotation_position="top",
    )
    edge_figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
        hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        xaxis_title="Match index",
        yaxis_title="Cumulative edge (log-loss units)",
        xaxis={
            "gridcolor": PLOT_GRID,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(255, 255, 255, 0.3)",
        },
        yaxis={
            "gridcolor": PLOT_GRID,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(255, 255, 255, 0.3)",
            "zeroline": False,
        },
    )
    tab1, tab2, tab3 = st.tabs(["Cumulative Edge", "Edge Candles (50-match)", "Rolling Edge (100-match)"])
    with tab1:
        st.plotly_chart(
            edge_figure,
            width="stretch",
            config={"displayModeBar": False},
            alt="Cumulative edge across backtest matches; positive values mean Betty has lower log loss.",
        )

    with tab2:
        candle_df = compute_edge_candles(scored["edge"].to_numpy(dtype=float), window=50)
        candle_fig = go.Figure()
        complete_candles = candle_df[candle_df["is_complete"]]
        if not complete_candles.empty:
            candle_fig.add_trace(
                go.Candlestick(
                    x=complete_candles["window_index"],
                    open=complete_candles["open"],
                    high=complete_candles["high"],
                    low=complete_candles["low"],
                    close=complete_candles["close"],
                    name="Complete windows",
                    increasing_line_color=ACCENT,
                    decreasing_line_color="#f0b858",
                    increasing_fillcolor="rgba(67, 217, 181, 0.45)",
                    decreasing_fillcolor="rgba(240, 184, 88, 0.45)",
                )
            )
        incomplete_candles = candle_df[~candle_df["is_complete"]]
        if not incomplete_candles.empty:
            candle_fig.add_trace(
                go.Candlestick(
                    x=incomplete_candles["window_index"],
                    open=incomplete_candles["open"],
                    high=incomplete_candles["high"],
                    low=incomplete_candles["low"],
                    close=incomplete_candles["close"],
                    name="Current window (in progress)",
                    increasing_line_color=ACCENT,
                    decreasing_line_color="#f0b858",
                    increasing_fillcolor="rgba(0,0,0,0)",
                    decreasing_fillcolor="rgba(0,0,0,0)",
                )
            )
        candle_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BACKGROUND,
            plot_bgcolor=PLOT_BACKGROUND,
            font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
            hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            xaxis_title="Window index (50 matches)",
            yaxis_title="Cumulative edge (log-loss units)",
            xaxis_rangeslider_visible=False,
            xaxis={
                "gridcolor": PLOT_GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
            },
            yaxis={
                "gridcolor": PLOT_GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
                "zeroline": False,
            },
            showlegend=False,
        )
        st.plotly_chart(
            candle_fig,
            width="stretch",
            config={"displayModeBar": False},
            alt="50-match edge candlesticks of cumulative log-loss edge.",
        )
        st.caption(
            "Edge candles, not prices. Each candle represents the OHLC of the cumulative edge over a 50-match window."
        )

    with tab3:
        rolling_df = compute_rolling_edge(scored["edge"].to_numpy(dtype=float), window=100)
        rolling_fig = go.Figure()
        rolling_fig.add_trace(
            go.Scatter(
                x=rolling_df["match_index"],
                y=rolling_df["lower_band"],
                mode="lines",
                line_color="rgba(255,255,255,0)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        rolling_fig.add_trace(
            go.Scatter(
                x=rolling_df["match_index"],
                y=rolling_df["upper_band"],
                mode="lines",
                line_color="rgba(255,255,255,0)",
                fill="tonexty",
                fillcolor="rgba(255, 255, 255, 0.1)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        rolling_fig.add_trace(
            go.Scatter(
                x=rolling_df["match_index"],
                y=rolling_df["rolling_edge"],
                mode="lines",
                line_color="#dce5ec",
                name="Rolling Edge",
                hovertemplate="Match %{x}<br>Rolling edge %{y:.4f}<extra></extra>",
            )
        )
        rolling_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        rolling_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BACKGROUND,
            plot_bgcolor=PLOT_BACKGROUND,
            font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
            hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            xaxis_title="Match index",
            yaxis_title="Rolling edge (log-loss units)",
            xaxis_rangeslider_visible=False,
            xaxis={
                "gridcolor": PLOT_GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
            },
            yaxis={
                "gridcolor": PLOT_GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
                "zeroline": False,
            },
        )
        st.plotly_chart(
            rolling_fig,
            width="stretch",
            config={"displayModeBar": False},
            alt="Rolling 100-match average edge with 95% confidence bands.",
        )
        st.caption(
            "Rolling 100-match average edge (Betty log loss vs Market log loss) with 95% confidence bands. "
            "Positive values indicate Betty outperformed the market."
        )

with st.container(border=True):
    st.subheader("Calibration")
    st.caption(
        "Predicted probability bins of width 10 points. "
        "Bars show 95% Wilson intervals for Betty's observed rate. "
        "Hollow markers are bins with fewer than 100 predictions (low sample). "
        "The market line uses the same bins on the same matches, but each bin holds different matches for Betty and for the market, so the comparison is indicative only."
    )
    selection = st.radio(
        "Outcome",
        ["All", "Home", "Draw", "Away"],
        horizontal=True,
        key="calibration_outcome",
    )
    outcomes = scored["actual_result"].to_numpy()
    if selection == "All":
        betty_pred = scored[PROBABILITY_COLUMNS].to_numpy(dtype=float).ravel()
        market_pred = scored[MARKET_COLUMNS].to_numpy(dtype=float).ravel()
        occ = np.column_stack(
            [outcomes == result for result in ("H", "D", "A")]
        ).astype(int).ravel()
    else:
        col_map = {
            "Home": ("p_home", "bookie_home", "H"),
            "Draw": ("p_draw", "bookie_draw", "D"),
            "Away": ("p_away", "bookie_away", "A"),
        }
        b_col, m_col, res_code = col_map[selection]
        betty_pred = scored[b_col].to_numpy(dtype=float)
        market_pred = scored[m_col].to_numpy(dtype=float)
        occ = (outcomes == res_code).astype(int)

    betty_cal, betty_raw = calibration_bins(betty_pred, occ)
    market_cal, market_raw = calibration_bins(market_pred, occ)

    if betty_cal.empty and market_cal.empty:
        st.caption("No predictions in this selection.")
    else:
        calibration_figure = go.Figure()
        calibration_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line={"color": "#72808c", "dash": "dash"},
            )
        )
        betty_mask = (betty_cal["n"] >= 100).to_numpy()
        betty_symbols = np.where(betty_mask, "circle", "circle-open").tolist()
        betty_array = np.where(
            betty_mask,
            betty_cal["upper"] - betty_cal["actual_rate"],
            0.0,
        ).tolist()
        betty_arrayminus = np.where(
            betty_mask,
            betty_cal["actual_rate"] - betty_cal["lower"],
            0.0,
        ).tolist()

        calibration_figure.add_trace(
            go.Scatter(
                x=betty_cal["mean_predicted"],
                y=betty_cal["actual_rate"],
                mode="lines+markers",
                name="Betty",
                line={"color": ACCENT, "width": 2},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": betty_array,
                    "arrayminus": betty_arrayminus,
                    "thickness": 1,
                    "color": ACCENT,
                },
                customdata=betty_cal[["n"]],
                marker={
                    "color": ACCENT,
                    "size": 8,
                    "symbol": betty_symbols,
                },
                hovertemplate=(
                    "Betty<br>Mean predicted %{x:.1%}<br>Actual rate %{y:.1%}"
                    "<br>n=%{customdata[0]:,}<extra></extra>"
                ),
            )
        )

        market_symbols = ["circle" if count >= 100 else "circle-open" for count in market_cal["n"]]
        calibration_figure.add_trace(
            go.Scatter(
                x=market_cal["mean_predicted"],
                y=market_cal["actual_rate"],
                mode="lines+markers",
                name="Market",
                line={"color": "#f0b858", "dash": "dot", "width": 2},
                customdata=market_cal[["n"]],
                marker={
                    "color": "#f0b858",
                    "size": 8,
                    "symbol": market_symbols,
                },
                hovertemplate=(
                    "Market<br>Mean predicted %{x:.1%}<br>Actual rate %{y:.1%}"
                    "<br>n=%{customdata[0]:,}<extra></extra>"
                ),
            )
        )

        calibration_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BACKGROUND,
            plot_bgcolor=PLOT_BACKGROUND,
            font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
            hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed win rate",
            xaxis={
                "gridcolor": PLOT_GRID,
                "range": [0, 1],
                "tickformat": ".0%",
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
            },
            yaxis={
                "gridcolor": PLOT_GRID,
                "range": [0, 1],
                "tickformat": ".0%",
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
            },
            legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        )
        st.plotly_chart(
            calibration_figure,
            width="stretch",
            config={"displayModeBar": False},
            alt=f"Betty and market observed outcome rates by predicted-probability bin for {selection.lower()} outcomes compared with perfect calibration.",
        )
        cal_stats = calibration_slope_intercept(
            betty_raw["predicted"].to_numpy(dtype=float),
            betty_raw["occurred"].to_numpy(dtype=float),
        )
        st.caption(
            f"Slope: {cal_stats['slope']:.2f} [{cal_stats['slope_ci'][0]:.2f}, {cal_stats['slope_ci'][1]:.2f}], "
            f"Intercept: {cal_stats['intercept']:.2f} [{cal_stats['intercept_ci'][0]:.2f}, {cal_stats['intercept_ci'][1]:.2f}]"
        )

with st.container(border=True):
    st.subheader("Open-to-Close Market Movement")
    opening_cols = ["opening_avg_h", "opening_avg_d", "opening_avg_a"]
    has_opening = (
        all(c in scored.columns for c in opening_cols)
        and scored[opening_cols].notna().all(axis=1)
    )
    if isinstance(has_opening, pd.Series) and has_opening.any():
        opening_data = scored.loc[has_opening]
        open_odds = opening_data[opening_cols].to_numpy(dtype=float)
        open_probs = np.array([proportional_devig(row) for row in open_odds])
        open_home = open_probs[:, 0]
        close_home = opening_data["bookie_home"].to_numpy(dtype=float)

        movement_fig = go.Figure()
        movement_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="No movement (y = x)",
                hoverinfo="skip",
            )
        )
        movement_fig.add_trace(
            go.Scattergl(
                x=open_home,
                y=close_home,
                mode="markers",
                marker=dict(opacity=0.3, size=4, color=ACCENT),
                name="Home probability",
                hovertemplate="Opening: %{x:.1%}<br>Closing: %{y:.1%}<extra></extra>",
            )
        )
        movement_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BACKGROUND,
            plot_bgcolor=PLOT_BACKGROUND,
            font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
            hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            xaxis_title="Opening Market Probability",
            yaxis_title="Closing Market Probability",
            xaxis_rangeslider_visible=False,
            xaxis={
                "gridcolor": PLOT_GRID,
                "range": [0, 1],
                "tickformat": ".0%",
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
                "zeroline": False,
            },
            yaxis={
                "gridcolor": PLOT_GRID,
                "range": [0, 1],
                "tickformat": ".0%",
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
                "zeroline": False,
            },
            showlegend=False,
        )
        st.plotly_chart(
            movement_fig,
            width="stretch",
            config={"displayModeBar": False},
            alt="Scatter plot of devigged market opening vs closing probabilities for home teams.",
        )
    else:
        st.caption("No opening odds available.")
    st.caption(
        "Scatter plot of devigged market opening vs closing probabilities. "
        "Points above the dashed line indicate the market steamed in (probability increased); "
        "points below indicate drift."
    )

with st.container(border=True):
    st.subheader("League performance")
    st.caption(
        "Edge per match = market log loss minus Betty log loss; positive favors Betty. "
        "Backtest matches only. Leagues with n below 200 are low sample."
    )
    backtest = scored.loc[scored["phase"] == "Backtest"]
    league_summary = (
        backtest.groupby("league", as_index=False)
        .agg(
            matches=("betty_log_loss", "size"),
            betty_log_loss=("betty_log_loss", "mean"),
            market_log_loss=("market_log_loss", "mean"),
        )
    )
    league_summary["edge_per_match"] = (
        league_summary["market_log_loss"] - league_summary["betty_log_loss"]
    )
    league_summary["league"] = league_summary["league"].map(LEAGUE_NAMES).fillna(
        league_summary["league"]
    )
    league_summary["label"] = league_summary.apply(
        lambda row: f"{row['league']} (n={row['matches']:,})",
        axis=1,
    )
    league_summary = league_summary.sort_values("edge_per_match", ascending=True)
    league_figure = go.Figure(
        go.Bar(
            x=league_summary["edge_per_match"],
            y=league_summary["label"],
            orientation="h",
            marker_color=[
                ACCENT if edge >= 0 else "#c9575f"
                for edge in league_summary["edge_per_match"]
            ],
            customdata=league_summary[
                ["matches", "betty_log_loss", "market_log_loss"]
            ],
            hovertemplate=(
                "League %{y}<br>Edge per match %{x:+.4f}"
                "<br>Betty %{customdata[1]:.4f} | Market %{customdata[2]:.4f}"
                "<br>n=%{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    league_figure.add_vline(x=0, line_color="#72808c", line_dash="dot")
    league_figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
        hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        xaxis_title="Mean market log loss minus Betty log loss (per match)",
        yaxis_title="League",
        xaxis={
            "gridcolor": PLOT_GRID,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(255, 255, 255, 0.3)",
        },
        yaxis={
            "gridcolor": PLOT_GRID,
            "categoryorder": "array",
            "categoryarray": league_summary["label"],
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(255, 255, 255, 0.3)",
        },
    )
    st.plotly_chart(
        league_figure,
        width="stretch",
        config={"displayModeBar": False},
        alt="Edge per match by league, market log loss minus Betty log loss; negative values mean the market was better.",
    )

with st.container(border=True):
    st.subheader("Rolling Outcome Mix (100-match window)")
    st.caption("Rolling average of Betty's home, draw, and away probabilities.")
    outcome_mix_figure = go.Figure()
    outcome_mix_x = np.arange(1, len(scored) + 1)
    outcome_mix_series = [
        ("Home", "roll_home", "#43b89f", "rgba(67, 184, 159, 0.55)", "tozeroy"),
        ("Draw", "roll_draw", "#778493", "rgba(119, 132, 147, 0.55)", "tonexty"),
        ("Away", "roll_away", "#718dd0", "rgba(113, 141, 208, 0.55)", "tonexty"),
    ]
    for label, column, color, fillcolor, fill in outcome_mix_series:
        outcome_mix_figure.add_trace(
            go.Scatter(
                x=outcome_mix_x,
                y=scored[column],
                mode="lines",
                name=label,
                line={"color": color, "width": 1},
                fill=fill,
                fillcolor=fillcolor,
                hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
            )
        )
    outcome_mix_figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
        hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        xaxis_title="Match index",
        yaxis_title="Probability mix",
        xaxis={
            "gridcolor": PLOT_GRID,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(255, 255, 255, 0.3)",
        },
        yaxis={
            "gridcolor": PLOT_GRID,
            "range": [0, 1],
            "tickformat": ".0%",
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(255, 255, 255, 0.3)",
        },
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
    )
    outcome_mix_figure.add_vline(
        x=backtest_count + 0.5,
        line_color="#f0b858",
        line_dash="dash",
        annotation_text="Forward Test Started",
        annotation_position="top",
    )
    st.plotly_chart(
        outcome_mix_figure,
        width="stretch",
        config={"displayModeBar": False},
        alt="Betty's rolling 100-match average home, draw, and away probability mix.",
    )

st.caption(
    "Market values in this export are already proportional-devigged probabilities. "
    "Decimal odds are converted to implied probabilities if supplied in their place."
)

st.divider()
st.header("Experimental: hypothetical backtest simulations (in-sample)")
st.warning(
    "Hypothetical, in-sample. Filters were chosen by looking at this same backtest. "
    "Stake sizes (Kelly) are computed from devigged closing odds, which no bookmaker offers. "
    "Payouts on the 6,534 backtest matches use raw, margin-inclusive closing odds; "
    "payouts on the forward matches fall back to devigged odds, so the curve mixes two price types. "
    "The blend test in the Method panel found no weight on Betty, so Kelly staking assumes an advantage the evidence does not show. "
    "Not evidence of an edge; judge only on forward results."
)

sim_base = prepare_betty_picks(scored)

filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
with filter_col1:
    min_kelly = st.slider(
        "Min Kelly Fraction (Edge)",
        min_value=0.0,
        max_value=0.10,
        value=0.0,
        step=0.005,
        key="exp_min_kelly",
    )
with filter_col2:
    max_odds = st.number_input(
        "Max Odds Cap",
        value=10.0,
        min_value=1.0,
        step=0.5,
        key="exp_max_odds",
    )
with filter_col3:
    all_leagues = sorted(sim_base["league"].dropna().unique().tolist())
    selected_leagues = st.multiselect(
        "Leagues",
        options=all_leagues,
        default=all_leagues,
        key="exp_leagues",
    )

sim_mask = (
    sim_base["league"].isin(selected_leagues)
    & (sim_base["top_market_odds"] <= max_odds)
    & (sim_base["full_kelly"] >= min_kelly)
)
filtered_sim = sim_base.loc[sim_mask].copy().reset_index(drop=True)

st.caption(
    f"Filtered match count: **n = {len(filtered_sim):,}** "
    f"({len(filtered_sim) / len(sim_base):.1%} of total {len(sim_base):,} matches)"
)

qk_series, flat_series, drawdown_series = simulate_bankroll_and_drawdown(filtered_sim)
sim_backtest_count = int((filtered_sim["phase"] == "Backtest").sum()) if not filtered_sim.empty else 0
sim_forward_count = len(filtered_sim) - sim_backtest_count

with st.container(border=True):
    st.subheader("Recovery Tracker (Drawdown Curve)")
    st.caption(
        "Cumulative edge relative to its historical peak; values are zero at all-time highs "
        "and negative during drawdowns."
    )
    if filtered_sim.empty:
        st.info("No matches match the selected strategy filters.", icon=":material/info:")
    else:
        drawdown_figure = go.Figure()
        drawdown_x = np.arange(1, len(filtered_sim) + 1)
        drawdown_figure.add_trace(
            go.Scatter(
                x=drawdown_x,
                y=drawdown_series,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(240, 184, 88, 0.2)",
                line_color="rgba(240, 184, 88, 0.8)",
                name="Drawdown",
                hovertemplate="Match %{x}<br>Drawdown %{y:.3f}<extra></extra>",
            )
        )
        drawdown_figure.add_hline(y=0, line_color="#72808c", line_dash="dot")
        if sim_backtest_count > 0 and sim_forward_count > 0:
            drawdown_figure.add_vline(
                x=sim_backtest_count + 0.5,
                line_color="#f0b858",
                line_dash="dash",
                annotation_text="Forward Test Started",
                annotation_position="top",
            )
        max_drawdown = float(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0
        drawdown_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BACKGROUND,
            plot_bgcolor=PLOT_BACKGROUND,
            font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
            hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            xaxis_title="Match index",
            yaxis_title="Drawdown (log-loss units)",
            xaxis={
                "gridcolor": PLOT_GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
            },
            yaxis={
                "gridcolor": PLOT_GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
                "zeroline": False,
                "range": [max_drawdown * 1.05 if max_drawdown < 0 else -1.0, 0],
                "maxallowed": 0,
            },
        )
        st.plotly_chart(
            drawdown_figure,
            width="stretch",
            config={"displayModeBar": False},
            alt="Drawdown curve of cumulative edge from all-time peak; values are zero or negative.",
        )

with st.container(border=True):
    st.subheader("Performance Matrix (Mean Edge by League & Odds)")
    st.caption(
        "Mean edge per match (market log loss minus Betty log loss) by league and by the devigged closing odds of Betty's top pick, which are known before kickoff. Cells with few matches are noisy; this is descriptive, not a place to look for an edge."
    )
    if filtered_sim.empty:
        st.info("No matches match the selected strategy filters.", icon=":material/info:")
    else:
        performance_matrix = (
            filtered_sim.groupby(["league", "pick_odds_bracket"], observed=False)["edge"]
            .mean()
            .unstack("pick_odds_bracket")
        )
        matrix_figure = go.Figure(
            go.Heatmap(
                z=performance_matrix.values,
                x=list(performance_matrix.columns),
                y=list(performance_matrix.index),
                colorscale=[
                    [0, "rgba(240, 184, 88, 0.8)"],
                    [0.5, "rgba(14, 20, 27, 1)"],
                    [1, "rgba(67, 217, 181, 0.8)"],
                ],
                zmid=0,
                colorbar={"title": "Mean edge"},
                hovertemplate=(
                    "League: %{y}<br>"
                    "Odds: %{x}<br>"
                    "Mean edge: %{z:+.4f}<extra></extra>"
                ),
            )
        )
        matrix_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BACKGROUND,
            plot_bgcolor=PLOT_BACKGROUND,
            font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
            hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            xaxis_title="Odds bracket (Betty's top pick, devigged closing odds)",
            yaxis_title="League",
            xaxis={"gridcolor": PLOT_GRID},
            yaxis={
                "gridcolor": PLOT_GRID,
                "autorange": "reversed",
                "tickmode": "array",
                "tickvals": list(performance_matrix.index),
                "ticktext": [
                    f"{LEAGUE_NAMES.get(code, code)} ({code})"
                    for code in performance_matrix.index
                ],
            },
        )
        st.plotly_chart(
            matrix_figure,
            width="stretch",
            config={"displayModeBar": False},
            alt="Performance matrix heatmap showing mean edge by league and odds bracket.",
        )

with st.container(border=True):
    st.subheader("Bankroll Simulation (Quarter Kelly vs. 1-Unit Flat Staking)")
    st.caption(
        "Simulated 100-unit starting bankroll betting on Betty's highest-probability outcome "
        "when Betty has a positive edge against devigged closing odds. "
        "Quarter Kelly compounds stake sizing by edge, while 1-Unit Flat Staking wagers a constant 1 unit."
    )
    if filtered_sim.empty:
        st.info("No matches match the selected strategy filters.", icon=":material/info:")
    else:
        bankroll_figure = go.Figure()
        bankroll_x = np.arange(1, len(filtered_sim) + 1)
        bankroll_figure.add_trace(
            go.Scatter(
                x=bankroll_x,
                y=qk_series,
                mode="lines",
                line={"color": ACCENT, "width": 2},
                name="Quarter Kelly Bankroll",
                hovertemplate="Match %{x}<br>Quarter Kelly: %{y:.2f} units<extra></extra>",
            )
        )
        bankroll_figure.add_trace(
            go.Scatter(
                x=bankroll_x,
                y=flat_series,
                mode="lines",
                line={"color": "#72808c", "width": 1.5},
                name="Flat Staking Bankroll",
                hovertemplate="Match %{x}<br>Flat Staking: %{y:.2f} units<extra></extra>",
            )
        )
        bankroll_figure.add_hline(y=100, line_color="#72808c", line_dash="dot")
        if sim_backtest_count > 0 and sim_forward_count > 0:
            bankroll_figure.add_vline(
                x=sim_backtest_count + 0.5,
                line_color="#f0b858",
                line_dash="dash",
                annotation_text="Forward Test Started",
                annotation_position="top",
            )
        max_bankroll = max(
            float(qk_series.max()) if len(qk_series) > 0 else 100.0,
            float(flat_series.max()) if len(flat_series) > 0 else 100.0,
            100.0,
        )
        bankroll_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BACKGROUND,
            plot_bgcolor=PLOT_BACKGROUND,
            font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
            hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
            margin={"l": 12, "r": 12, "t": 24, "b": 12},
            xaxis_title="Match index",
            yaxis_title="Bankroll (units)",
            xaxis={
                "gridcolor": PLOT_GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
            },
            yaxis={
                "gridcolor": PLOT_GRID,
                "range": [0, max_bankroll * 1.05],
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikethickness": 1,
                "spikedash": "dot",
                "spikecolor": "rgba(255, 255, 255, 0.3)",
            },
            legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        )
        st.plotly_chart(
            bankroll_figure,
            width="stretch",
            config={"displayModeBar": False},
            alt="Bankroll simulation comparing Quarter Kelly compounding vs 1-unit flat staking.",
        )

