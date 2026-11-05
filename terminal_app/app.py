from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "backtest_predictions_v1.csv"
FORWARD_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "forward_scores_matches.csv"
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
    scored[MARKET_COLUMNS] = np.vstack(
        [proportional_devig(row) for row in scored[MARKET_COLUMNS].to_numpy(dtype=float)]
    )
    scored[PROBABILITY_COLUMNS] = np.vstack(
        [row / row.sum() for row in scored[PROBABILITY_COLUMNS].to_numpy(dtype=float)]
    )
    scored["betty_log_loss"] = [
        calculate_log_loss(probabilities, result)
        for probabilities, result in zip(
            scored[PROBABILITY_COLUMNS].to_numpy(dtype=float),
            scored["actual_result"],
        )
    ]
    scored["market_log_loss"] = [
        calculate_log_loss(probabilities, result)
        for probabilities, result in zip(
            scored[MARKET_COLUMNS].to_numpy(dtype=float),
            scored["actual_result"],
        )
    ]
    scored["edge"] = scored["market_log_loss"] - scored["betty_log_loss"]
    scored["cumulative_edge"] = scored["edge"].cumsum()
    rolling_columns = ["roll_home", "roll_draw", "roll_away"]
    scored[rolling_columns] = (
        scored[PROBABILITY_COLUMNS]
        .rolling(window=100, min_periods=1)
        .mean()
        .set_axis(rolling_columns, axis="columns")
    )
    backtest_edges = scored.loc[scored["phase"] == "Backtest", "edge"].to_numpy(dtype=float)
    backtest_std = float(np.std(backtest_edges, ddof=1)) if len(backtest_edges) > 1 else 0.0
    cumulative_std = backtest_std * np.sqrt(np.arange(1, len(scored) + 1))
    scored["upper_band"] = scored["cumulative_edge"] + (1.96 * cumulative_std)
    scored["lower_band"] = scored["cumulative_edge"] - (1.96 * cumulative_std)
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
st.caption(f"Betty Core (v1) | Backtest: {backtest_count:,} matches | Forward: live scored matches | Market: de-vigged closing odds")

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

with st.container(border=True):
    st.subheader("Edge curve")
    st.caption("Cumulative market log loss minus Betty log loss; positive values favor Betty.")
    edge_figure = go.Figure()
    edge_x = np.arange(1, len(scored) + 1)
    edge_figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=scored["upper_band"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    edge_figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=scored["lower_band"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(255, 255, 255, 0.05)",
            name="95% band, approximate",
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
    st.plotly_chart(
        edge_figure,
        width="stretch",
        config={"displayModeBar": False},
        alt="Cumulative edge across backtest matches; positive values mean Betty has lower log loss.",
    )

with st.container(border=True):
    st.subheader("Calibration")
    st.caption("Home, draw, and away predictions are grouped into ten probability bins. Bins with fewer than 100 predictions are low sample.")
    outcomes = scored["actual_result"].to_numpy()
    calibration_data = pd.DataFrame(
        {
            "predicted_probability": scored[PROBABILITY_COLUMNS].to_numpy(dtype=float).ravel(),
            "occurred": np.column_stack(
                [outcomes == result for result in ("H", "D", "A")]
            ).astype(int).ravel(),
        }
    )
    calibration_data["probability_bin"] = pd.cut(
        calibration_data["predicted_probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    calibration = (
        calibration_data.groupby("probability_bin", observed=True)
        .agg(
            mean_predicted=("predicted_probability", "mean"),
            actual_rate=("occurred", "mean"),
            predictions=("occurred", "size"),
        )
        .reset_index(drop=True)
    )
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
    calibration_figure.add_trace(
        go.Scatter(
            x=calibration["mean_predicted"],
            y=calibration["actual_rate"],
            mode="lines+markers",
            name="Betty",
            line={"color": ACCENT, "width": 2},
            customdata=calibration[["predictions"]],
            marker={
                "color": ACCENT,
                "size": 8,
            },
            hovertemplate=(
                "Mean predicted %{x:.1%}<br>Actual rate %{y:.1%}"
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
        alt="Betty's observed outcome rates by predicted-probability bin compared with perfect calibration.",
    )

with st.container(border=True):
    st.subheader("League performance")
    st.caption("Mean Betty log loss by league; lower values indicate better probabilistic accuracy.")
    league_summary = (
        scored.groupby("league", as_index=False)["betty_log_loss"]
        .mean()
        .sort_values("betty_log_loss", ascending=True)
    )
    league_summary["league"] = league_summary["league"].map(LEAGUE_NAMES).fillna(
        league_summary["league"]
    )
    league_figure = go.Figure(
        go.Bar(
            x=league_summary["betty_log_loss"],
            y=league_summary["league"],
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="League %{y}<br>Mean log loss %{x:.4f}<extra></extra>",
        )
    )
    league_figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": PLOT_FONT_FAMILY, "color": "#dce5ec"},
        hoverlabel={"font": {"family": PLOT_FONT_FAMILY}},
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        xaxis_title="Mean Betty log loss",
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
            "categoryarray": league_summary["league"],
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
        alt="Mean Betty log loss for each league, sorted from lowest to highest.",
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
