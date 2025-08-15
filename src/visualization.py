"""
Enhanced Visualization Module (Pylance-clean)

Comprehensive visualization utilities for climate data analysis with
robust typing, safe numeric conversions, and clear statistical overlays.
Icons/emojis have been removed per project requirements.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

from typing import Iterable, Sequence, cast
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Plotting defaults
plt.style.use("default")
sns.set_palette("husl")

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

COLOR_SCHEMES: Dict[str, Sequence[str]] = {
    "temperature": ["#FF6B6B", "#FF8E53", "#FF6B35", "#C44569"],
    "deforestation": ["#2ECC71", "#27AE60", "#16A085", "#1ABC9C"],
    "heatwave": ["#E74C3C", "#F39C12", "#F1C40F", "#E67E22"],
    "climate": ["#3498DB", "#2980B9", "#1ABC9C", "#16A085", "#27AE60", "#2ECC71"],
    "statistical": ["#9B59B6", "#8E44AD", "#673AB7", "#3F51B5"],
}

DEFAULT_FIGSIZE: Tuple[float, float] = (15.0, 10.0)
DPI: int = 300
FONT_SIZES: Dict[str, int] = {
    "title": 16,
    "subtitle": 14,
    "label": 12,
    "tick": 10,
    "legend": 11,
}

# canonical column names used throughout
TEMP_COLUMN: str = "Dhaka Temperature [2 m elevation corrected]"


# ------------------------------------------------------------------------------
# Helpers & utilities (type-safe, Pylance-friendly)
# ------------------------------------------------------------------------------


def _has_columns(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    """Return True if all columns exist in the DataFrame."""
    return all(c in df.columns for c in cols)


def _as_float_array(a: Iterable[Any]) -> NDArray[np.float64]:
    """
    Convert an iterable/array-like (Series/Index/list/ndarray) to a 1D float ndarray.
    Ensures finite dtype for matplotlib & scipy.
    """
    # Prefer numpy conversion; fallback to pandas-to-numeric for odd cases
    try:
        arr = np.asarray(a, dtype=float).reshape(-1)
    except Exception:
        arr = (
            pd.to_numeric(pd.Series(list(a)), errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1)
        )
    # Replace inf/-inf with NaN to avoid downstream errors
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def _as_index_float_array(idx: Iterable[Any]) -> NDArray[np.float64]:
    """
    Convert an index-like (e.g., pd.Index of years or datetimes) to float ndarray.
    Uses pandas dtype check (safe for ExtensionDtype).
    """
    if isinstance(idx, (pd.Index, pd.Series)):
        vals = idx.to_numpy()
    else:
        vals = np.asarray(list(idx))

    # For datetime-like values, use year component
    if is_datetime64_any_dtype(vals):
        years = (
            pd.to_datetime(vals, errors="coerce")
            .astype("datetime64[ns]")
            .astype("datetime64[Y]")
        )
        years = pd.to_datetime(years).year.to_numpy()
        return years.astype(float)

    # Otherwise, coerce to float safely
    return _as_float_array(vals)


def _as_bool_sequence(mask: np.ndarray) -> Sequence[bool]:
    """
    Convert a NumPy boolean array to a Sequence[bool] acceptable to
    matplotlib's 'where=' parameter (satisfies Pylance typing).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return cast(Sequence[bool], mask.tolist())


def _fmt_thousands(x: float, _pos: int) -> str:
    """Format numbers with thousands separators for axis tick labels."""
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)


def _safe_savefig(fig: Figure, save_path: Optional[str]) -> None:
    """Save a figure if a path is provided; always catch and log errors."""
    if not save_path:
        return
    try:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        logger.info("Figure saved to %s", save_path)
    except Exception as e:
        logger.error("Failed to save figure to %s: %s", save_path, e)


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

__all__ = [
    "plot_temperature_trends",
    "plot_deforestation_analysis",
    "plot_heatwave_analysis",
    "plot_correlation_matrix",
    "create_summary_dashboard",
]
# ------------------------------------------------------------------------------
# Plot 1: Temperature trends
# ------------------------------------------------------------------------------


def plot_temperature_trends(
    data: pd.DataFrame,
    annual_temp_stats: pd.DataFrame,
    save_path: Optional[str] = None,
    show_confidence: bool = True,
    highlight_extremes: bool = True,
) -> Figure:
    """
    Enhanced temperature trends visualization with statistical overlays.
    Icons/emojis removed.
    """
    logger.info("Creating enhanced temperature trends visualization")

    try:
        # Defensive copies and derived columns
        df = data.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "Year" not in df.columns and "timestamp" in df.columns:
            df["Year"] = df["timestamp"].dt.year

        fig = plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DPI)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        colors = COLOR_SCHEMES["temperature"]

        # 1) Daily temperature plot (recent years)
        ax1 = fig.add_subplot(gs[0, 0])
        if _has_columns(df, [TEMP_COLUMN, "timestamp", "Year"]):
            recent_data = df.loc[df["Year"] >= 2020].dropna(subset=[TEMP_COLUMN]).copy()
            if not recent_data.empty:
                x_time = _as_datetime_array(recent_data["timestamp"])
                y_temp = _as_float_array(recent_data[TEMP_COLUMN])

                ax1.plot(
                    x_time,
                    y_temp,
                    alpha=0.8,
                    linewidth=1.2,
                    color=colors[0],
                    label="Daily Temperature",
                )

                # Heatwave threshold & highlights
                if "Heatwave" in recent_data.columns:
                    heatwave_threshold: float = 36.0
                    ax1.axhline(
                        y=float(heatwave_threshold),
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=2.0,
                        label=f"Heatwave Threshold ({heatwave_threshold:.0f}°C)",
                    )

                    if highlight_extremes:
                        hw_mask = recent_data["Heatwave"] == True  # noqa: E712
                        if hw_mask.any():
                            ax1.scatter(
                                _as_datetime_array(
                                    recent_data.loc[hw_mask, "timestamp"]
                                ),
                                _as_float_array(recent_data.loc[hw_mask, TEMP_COLUMN]),
                                color="red",
                                alpha=0.6,
                                s=30.0,
                                zorder=5,
                                label="Heatwave Days",
                            )

                # Rolling average (30-day)
                if len(recent_data) > 30:
                    roll = (
                        recent_data[TEMP_COLUMN].rolling(window=30, center=True).mean()
                    )
                    ax1.plot(
                        x_time,
                        _as_float_array(roll.bfill().ffill()),
                        color=colors[1],
                        linewidth=2.0,
                        alpha=0.8,
                        label="30-day Average",
                    )

                ax1.set_title(
                    "Daily Temperature Trends (Recent Years)",
                    fontsize=FONT_SIZES["title"],
                    fontweight="bold",
                    pad=20.0,
                )
                ax1.set_xlabel("Date", fontsize=FONT_SIZES["label"])
                ax1.set_ylabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=FONT_SIZES["legend"])
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2) Annual temperature trends with statistical analysis
        ax2 = fig.add_subplot(gs[0, 1])
        temp_mean_col: Optional[str] = None
        temp_max_col: Optional[str] = None
        for col in annual_temp_stats.columns:
            low = col.lower()
            if "temperature" in low and "mean" in low:
                temp_mean_col = col
            if "temperature" in low and "max" in low:
                temp_max_col = col

        if "Year" in annual_temp_stats.columns:
            years = _as_float_array(annual_temp_stats["Year"])

            if temp_mean_col is not None and temp_mean_col in annual_temp_stats.columns:
                mean_vals = _as_float_array(annual_temp_stats[temp_mean_col])
                ax2.plot(
                    years,
                    mean_vals,
                    marker="o",
                    linewidth=2.5,
                    markersize=5.0,
                    color=colors[0],
                    label="Mean Temperature",
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                )

                if show_confidence and len(mean_vals) > 3:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        years, mean_vals
                    )
                    trend = slope * years + intercept
                    ax2.plot(
                        years,
                        trend,
                        "--",
                        color=colors[1],
                        linewidth=2.0,
                        alpha=0.8,
                        label=f"Trend (R²={float(r_value) ** 2:.3f})",
                    )
                    n = float(len(years))
                    sxx = float(np.sum((years - float(np.mean(years))) ** 2))
                    if n > 2 and sxx > 0.0:
                        t_val = float(stats.t.ppf(0.975, int(n - 2)))
                        residuals = mean_vals - trend
                        mse = float(np.sum(residuals**2) / (n - 2.0))
                        x_bar = float(np.mean(years))
                        ci = t_val * np.sqrt(
                            mse * (1.0 / n + ((years - x_bar) ** 2) / sxx)
                        )
                        ax2.fill_between(
                            years,
                            (trend - ci),
                            (trend + ci),
                            color=colors[1],
                            alpha=0.2,
                            label="95% Confidence",
                        )

            if temp_max_col is not None and temp_max_col in annual_temp_stats.columns:
                max_vals = _as_float_array(annual_temp_stats[temp_max_col])
                ax2.plot(
                    years,
                    max_vals,
                    marker="s",
                    linewidth=1.5,
                    markersize=4.0,
                    color=colors[2],
                    label="Max Temperature",
                    alpha=0.8,
                )

            ax2.set_title(
                "Annual Temperature Trends with Statistical Analysis",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=20.0,
            )
            ax2.set_xlabel("Year", fontsize=FONT_SIZES["label"])
            ax2.set_ylabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
            ax2.legend(fontsize=FONT_SIZES["legend"])
            ax2.grid(True, alpha=0.3, linestyle=":")

            if len(years) > 0:
                yr_max = int(np.nanmax(years))
                for decade in range(1970, 2031, 10):
                    if decade <= yr_max:
                        ax2.axvline(
                            x=float(decade),
                            color="gray",
                            alpha=0.3,
                            linestyle="-",
                            linewidth=0.8,
                        )

        # 3) Temperature distribution with overlays
        ax3 = fig.add_subplot(gs[1, 0])
        if TEMP_COLUMN in df.columns:
            temp_data = pd.to_numeric(df[TEMP_COLUMN], errors="coerce").dropna()
            if not temp_data.empty:
                n_bins = int(min(50, max(5, int(np.sqrt(len(temp_data))))))
                counts, bins, patches = ax3.hist(
                    _as_float_array(temp_data),
                    bins=n_bins,
                    alpha=0.7,
                    color=colors[0],
                    edgecolor="black",
                    linewidth=0.5,
                )
                mean_temp = float(temp_data.mean())
                median_temp = float(temp_data.median())
                ax3.axvline(
                    mean_temp,
                    color="red",
                    linestyle="--",
                    linewidth=2.0,
                    label=f"Mean: {mean_temp:.1f}°C",
                )
                ax3.axvline(
                    median_temp,
                    color="blue",
                    linestyle="--",
                    linewidth=2.0,
                    label=f"Median: {median_temp:.1f}°C",
                )
                p95 = float(temp_data.quantile(0.95))
                p05 = float(temp_data.quantile(0.05))
                ax3.axvline(
                    p95,
                    color="orange",
                    linestyle=":",
                    linewidth=2.0,
                    label=f"95th percentile: {p95:.1f}°C",
                )
                ax3.axvline(
                    p05,
                    color="cyan",
                    linestyle=":",
                    linewidth=2.0,
                    label=f"5th percentile: {p05:.1f}°C",
                )

                if len(temp_data) > 100 and len(bins) > 1:
                    x_norm = np.linspace(
                        float(temp_data.min()), float(temp_data.max()), 200
                    )
                    y_norm = stats.norm.pdf(
                        x_norm, mean_temp, float(temp_data.std(ddof=0))
                    )
                    y_scaled = (
                        y_norm * len(temp_data) * (float(bins[1]) - float(bins[0]))
                    )
                    ax3.plot(
                        x_norm,
                        y_scaled,
                        "r-",
                        linewidth=2.0,
                        alpha=0.8,
                        label="Normal Distribution",
                    )

                ax3.set_title(
                    "Temperature Distribution with Statistical Markers",
                    fontsize=FONT_SIZES["title"],
                    fontweight="bold",
                    pad=20.0,
                )
                ax3.set_xlabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
                ax3.set_ylabel("Frequency", fontsize=FONT_SIZES["label"])
                ax3.legend(fontsize=FONT_SIZES["legend"], loc="upper right")
                ax3.grid(True, alpha=0.3, linestyle=":")

        # 4) Seasonal temperature patterns with error bars
        ax4 = fig.add_subplot(gs[1, 1])
        if _has_columns(df, [TEMP_COLUMN, "Season"]):
            seasonal_stats = (
                df.dropna(subset=[TEMP_COLUMN])
                .groupby("Season")[TEMP_COLUMN]
                .agg(["mean", "std", "count"])
            )
            if not seasonal_stats.empty:
                season_names = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
                seasons_labels = [
                    season_names.get(int(i), f"Season {int(i)}")
                    for i in seasonal_stats.index
                ]
                means = np.asarray(seasonal_stats["mean"].to_numpy(dtype=float))
                stds = np.asarray(seasonal_stats["std"].to_numpy(dtype=float))
                counts = np.asarray(seasonal_stats["count"].to_numpy(dtype=float))

                seasonal_colors = [
                    colors[i % len(colors)] for i in range(len(seasons_labels))
                ]
                bars = ax4.bar(
                    seasons_labels,
                    means,
                    yerr=stds,
                    capsize=5.0,
                    color=seasonal_colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.0,
                    error_kw={"elinewidth": 2.0},
                )

                for i, (bar, mean, count, stdv) in enumerate(
                    zip(bars, means, counts, stds)
                ):
                    height = float(bar.get_height())
                    ax4.text(
                        float(bar.get_x() + bar.get_width() / 2.0),
                        height + float(stdv) + 0.5,
                        f"{mean:.1f}°C\n(n={int(count)})",
                        ha="center",
                        va="bottom",
                        fontsize=FONT_SIZES["tick"],
                        fontweight="bold",
                    )

                ax4.set_title(
                    "Seasonal Temperature Patterns with Variability",
                    fontsize=FONT_SIZES["title"],
                    fontweight="bold",
                    pad=20.0,
                )
                ax4.set_xlabel("Season", fontsize=FONT_SIZES["label"])
                ax4.set_ylabel("Average Temperature (°C)", fontsize=FONT_SIZES["label"])
                ax4.grid(True, alpha=0.3, linestyle=":", axis="y")

                overall_mean = float(
                    pd.to_numeric(df[TEMP_COLUMN], errors="coerce").mean()
                )
                ax4.axhline(
                    y=overall_mean,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2.0,
                    label=f"Annual Mean: {overall_mean:.1f}°C",
                )
                ax4.legend(fontsize=FONT_SIZES["legend"])

        plt.tight_layout()
        _safe_savefig(fig, save_path)
        return fig

    except Exception as e:
        logger.error("Error creating temperature trends plot: %s", e)
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        ax.text(
            0.5,
            0.5,
            f"Error creating plot: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig


# ------------------------------------------------------------------------------
# Plot 2: Deforestation analysis
# ------------------------------------------------------------------------------


def plot_deforestation_analysis(
    tree_loss_by_year: pd.DataFrame,
    combined_data: pd.DataFrame,
    save_path: Optional[str] = None,
    show_correlations: bool = True,
) -> Figure:
    """
    Deforestation analysis with trendlines, moving averages,
    cumulative milestones, and climate correlation. Icons/emojis removed.
    """
    logger.info("Creating enhanced deforestation analysis visualization")

    try:
        fig = plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DPI)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        colors = COLOR_SCHEMES["deforestation"]

        # 1) Annual tree cover loss with trend analysis
        ax1 = fig.add_subplot(gs[0, 0])
        loss_col: Optional[str] = None
        for col in tree_loss_by_year.columns:
            low = col.lower()
            if "loss" in low and "ha" in low:
                loss_col = col
                break

        if loss_col and "Year" in tree_loss_by_year.columns:
            years_sr = tree_loss_by_year["Year"]
            years = _as_index_float_array(years_sr)
            losses = pd.to_numeric(tree_loss_by_year[loss_col], errors="coerce")
            losses_np = _as_float_array(losses)

            ax1.plot(
                years,
                losses_np,
                marker="o",
                linewidth=3.0,
                markersize=7.0,
                color=colors[0],
                markerfacecolor="white",
                markeredgewidth=2.0,
                markeredgecolor=colors[0],
                label="Annual Tree Loss",
            )

            if len(years) > 3:
                z = np.polyfit(years, losses_np, 1)
                trend_line = np.poly1d(z)
                ax1.plot(
                    years,
                    trend_line(years),
                    "--",
                    color=colors[1],
                    linewidth=2.0,
                    alpha=0.8,
                    label=f"Trend (slope: {z[0]:.1f} ha/year)",
                )

            if len(losses_np) > 5:
                moving_avg = (
                    pd.Series(losses_np)
                    .rolling(window=3, center=True)
                    .mean()
                    .to_numpy()
                )
                ax1.plot(
                    years,
                    moving_avg,
                    color=colors[2],
                    linewidth=2.0,
                    alpha=0.7,
                    label="3-year Moving Average",
                )

            if len(losses_np) > 0:
                idx_max = int(np.nanargmax(losses_np))
                max_year = float(years[idx_max])
                max_loss = float(losses_np[idx_max])
                ax1.annotate(
                    f"Peak Loss\n{int(max_year)}: {max_loss:.0f} ha",
                    xy=(max_year, max_loss),
                    xytext=(max_year, max_loss * 1.2),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
                    fontsize=FONT_SIZES["tick"],
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

            ax1.set_title(
                "Annual Tree Cover Loss in Dhaka with Trend Analysis",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=20.0,
            )
            ax1.set_xlabel("Year", fontsize=FONT_SIZES["label"])
            ax1.set_ylabel("Tree Cover Loss (hectares)", fontsize=FONT_SIZES["label"])
            ax1.legend(fontsize=FONT_SIZES["legend"])
            ax1.grid(True, alpha=0.3, linestyle=":")
            ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

        # 2) Deforestation vs mean temperature correlation
        ax2 = fig.add_subplot(gs[0, 1])
        temp_mean_col: Optional[str] = None
        deforest_col: Optional[str] = None
        for col in combined_data.columns:
            low = col.lower()
            if "temperature" in low and "mean" in low:
                temp_mean_col = col
            if ("loss" in low or "deforest" in low) and "ha" in low:
                deforest_col = col

        if temp_mean_col and deforest_col:
            valid = combined_data[
                [
                    temp_mean_col,
                    deforest_col,
                    *(["Year"] if "Year" in combined_data.columns else []),
                ]
            ].copy()
            valid[temp_mean_col] = pd.to_numeric(valid[temp_mean_col], errors="coerce")
            valid[deforest_col] = pd.to_numeric(valid[deforest_col], errors="coerce")
            valid = valid.dropna(subset=[temp_mean_col, deforest_col])
            valid = valid[valid[deforest_col] > 0]

            if len(valid) > 3:
                x_vals = _as_float_array(valid[deforest_col])
                y_vals = _as_float_array(valid[temp_mean_col])

                if "Year" in valid.columns:
                    scatter = ax2.scatter(
                        x_vals,
                        y_vals,
                        c=_as_float_array(valid["Year"]),
                        cmap="viridis",
                        alpha=0.8,
                        s=100.0,
                        edgecolors="black",
                        linewidth=1.0,
                    )
                    cbar = plt.colorbar(scatter, ax=ax2)
                    cbar.set_label("Year", fontsize=FONT_SIZES["label"])
                else:
                    ax2.scatter(
                        x_vals,
                        y_vals,
                        alpha=0.8,
                        s=100.0,
                        color=colors[3],
                        edgecolors="black",
                        linewidth=1.0,
                    )

                # correlations
                correlation_pearson = float(
                    pd.Series(x_vals).corr(pd.Series(y_vals), method="pearson")
                )
                correlation_spearman = float(
                    pd.Series(x_vals).corr(pd.Series(y_vals), method="spearman")
                )

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_vals, y_vals
                )
                line_x = np.linspace(
                    float(np.nanmin(x_vals)), float(np.nanmax(x_vals)), 200
                )
                line_y = slope * line_x + intercept
                ax2.plot(
                    line_x,
                    line_y,
                    "r-",
                    linewidth=2.0,
                    alpha=0.8,
                    label=f"Trend (R²={r_value**2:.3f})",
                )

                if show_correlations:
                    stats_text = (
                        f"Pearson r: {correlation_pearson:.3f}\n"
                        f"Spearman ρ: {correlation_spearman:.3f}\n"
                        f"p-value: {p_value:.3f}"
                    )
                    ax2.text(
                        0.05,
                        0.95,
                        stats_text,
                        transform=ax2.transAxes,
                        fontsize=FONT_SIZES["tick"],
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="white", alpha=0.8
                        ),
                    )

                ax2.set_title(
                    "Deforestation–Temperature Correlation Analysis",
                    fontsize=FONT_SIZES["title"],
                    fontweight="bold",
                    pad=20.0,
                )
                ax2.set_xlabel(
                    "Tree Cover Loss (hectares)", fontsize=FONT_SIZES["label"]
                )
                ax2.set_ylabel("Mean Temperature (°C)", fontsize=FONT_SIZES["label"])
                ax2.legend(fontsize=FONT_SIZES["legend"])
                ax2.grid(True, alpha=0.3, linestyle=":")
                ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

        # 3) Cumulative deforestation with milestones
        ax3 = fig.add_subplot(gs[1, 0])
        if loss_col and "Year" in tree_loss_by_year.columns:
            years_sr = tree_loss_by_year["Year"]
            years = _as_index_float_array(years_sr)
            annual_loss = pd.to_numeric(
                tree_loss_by_year[loss_col], errors="coerce"
            ).fillna(0.0)
            annual_loss_np = _as_float_array(annual_loss)
            cumulative_loss = np.cumsum(annual_loss_np)

            ax3.plot(
                years,
                cumulative_loss,
                marker="s",
                linewidth=3.0,
                markersize=6.0,
                color=colors[0],
                markerfacecolor="white",
                markeredgewidth=2.0,
                markeredgecolor=colors[0],
                label="Cumulative Loss",
            )
            ax3.fill_between(
                years,
                np.zeros_like(cumulative_loss),
                cumulative_loss,
                alpha=0.3,
                color=colors[0],
            )

            milestones = [1000.0, 5000.0, 10000.0, 20000.0]
            for m in milestones:
                if float(np.nanmax(cumulative_loss)) > m:
                    idx = int(np.argmax(cumulative_loss >= m))
                    milestone_year = float(years[idx])
                    ax3.axhline(y=m, color="red", linestyle=":", alpha=0.7)
                    ax3.text(
                        milestone_year,
                        m,
                        f"{int(m):,} ha",
                        fontsize=FONT_SIZES["tick"],
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7
                        ),
                    )

            year_span = (
                float(np.nanmax(years) - np.nanmin(years)) if len(years) > 1 else 0.0
            )
            avg_annual_rate = (
                float(cumulative_loss[-1] / year_span) if year_span > 0 else 0.0
            )
            ax3.text(
                0.98,
                0.02,
                f"Total Loss: {cumulative_loss[-1]:,.0f} ha\nAvg Rate: {avg_annual_rate:.0f} ha/year",
                transform=ax3.transAxes,
                fontsize=FONT_SIZES["tick"],
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            )

            ax3.set_title(
                "Cumulative Tree Cover Loss with Milestones",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=20.0,
            )
            ax3.set_xlabel("Year", fontsize=FONT_SIZES["label"])
            ax3.set_ylabel("Cumulative Loss (hectares)", fontsize=FONT_SIZES["label"])
            ax3.legend(fontsize=FONT_SIZES["legend"])
            ax3.grid(True, alpha=0.3, linestyle=":")
            ax3.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

        # 4) Normalized comparison with cross-correlation
        ax4 = fig.add_subplot(gs[1, 1])
        if temp_mean_col and deforest_col and "Year" in combined_data.columns:
            norm_data = combined_data.loc[
                combined_data["Year"] >= 2001, ["Year", temp_mean_col, deforest_col]
            ].copy()
            norm_data[temp_mean_col] = pd.to_numeric(
                norm_data[temp_mean_col], errors="coerce"
            )
            norm_data[deforest_col] = pd.to_numeric(
                norm_data[deforest_col], errors="coerce"
            )
            norm_data = norm_data.dropna(subset=[temp_mean_col, deforest_col])

            if not norm_data.empty:
                scaler = MinMaxScaler()
                temp_values = (
                    norm_data[temp_mean_col].to_numpy(dtype=float).reshape(-1, 1)
                )
                def_values = (
                    norm_data[deforest_col].to_numpy(dtype=float).reshape(-1, 1)
                )

                temp_norm = scaler.fit_transform(temp_values).reshape(-1)
                def_norm = scaler.fit_transform(def_values).reshape(-1)
                years = _as_index_float_array(norm_data["Year"])

                ax4.plot(
                    years,
                    temp_norm,
                    marker="o",
                    linewidth=3.0,
                    markersize=6.0,
                    label="Temperature (normalized)",
                    color=COLOR_SCHEMES["temperature"][0],
                    markerfacecolor="white",
                    markeredgewidth=2.0,
                )
                ax4.plot(
                    years,
                    def_norm,
                    marker="s",
                    linewidth=3.0,
                    markersize=6.0,
                    label="Deforestation (normalized)",
                    color=colors[0],
                    markerfacecolor="white",
                    markeredgewidth=2.0,
                )

                corr = float(np.corrcoef(temp_norm, def_norm)[0, 1])
                diff = np.abs(temp_norm - def_norm)
                similar_periods = diff < 0.2
                if similar_periods.any():
                    ax4.fill_between(
                        years,
                        np.zeros_like(years),
                        np.ones_like(years),
                        where=similar_periods.tolist(),  # appease Pylance stubs
                        alpha=0.2,
                        color="yellow",
                        label="Similar Trend Periods",
                    )

                ax4.text(
                    0.02,
                    0.98,
                    f"Cross-correlation: {corr:.3f}",
                    transform=ax4.transAxes,
                    fontsize=FONT_SIZES["tick"],
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                )

                ax4.set_title(
                    "Normalized Climate–Deforestation Trend Comparison",
                    fontsize=FONT_SIZES["title"],
                    fontweight="bold",
                    pad=20.0,
                )
                ax4.set_xlabel("Year", fontsize=FONT_SIZES["label"])
                ax4.set_ylabel("Normalized Values (0–1)", fontsize=FONT_SIZES["label"])
                ax4.legend(fontsize=FONT_SIZES["legend"])
                ax4.grid(True, alpha=0.3, linestyle=":")
                ax4.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        _safe_savefig(fig, save_path)
        return fig

    except Exception as e:
        logger.error("Error creating deforestation analysis plot: %s", e)
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        ax.text(
            0.5,
            0.5,
            f"Error creating plot: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig


# ------------------------------------------------------------------------------
# Plot 3: Heatwave analysis
# ------------------------------------------------------------------------------


def plot_heatwave_analysis(
    data: pd.DataFrame,
    heatwave_summary: pd.DataFrame,
    threshold: float = 36.0,
    save_path: Optional[str] = None,
    show_trends: bool = True,
) -> Figure:
    """
    Heatwave analysis with frequency trends, seasonal distribution,
    rolling averages, and duration/intensity distributions.
    Icons/emojis removed.
    """
    logger.info("Creating enhanced heatwave analysis visualization")

    try:
        df = data.copy()
        if "Year" not in df.columns and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["Year"] = df["timestamp"].dt.year

        fig = plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DPI)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        colors = COLOR_SCHEMES["heatwave"]

        # Pre-compute annual heatwave counts if possible (used by panels 1 & 3)
        heatwave_days_per_year = pd.Series(dtype=float)
        if _has_columns(df, ["Heatwave", "Year"]):
            heatwave_days_per_year = (
                df.loc[df["Heatwave"] == True].groupby("Year").size()
            )  # noqa: E712
            if "Year" in df.columns:
                all_years = pd.Index(sorted(df["Year"].dropna().unique()))
                heatwave_days_per_year = heatwave_days_per_year.reindex(
                    all_years, fill_value=0
                ).astype(float)

        # 1) Annual heatwave days with trend and highlights
        ax1 = fig.add_subplot(gs[0, 0])
        if not heatwave_days_per_year.empty:
            years = _as_index_float_array(heatwave_days_per_year.index)
            counts = _as_float_array(heatwave_days_per_year.values)

            bars = ax1.bar(
                years,
                counts,
                alpha=0.8,
                color=colors[0],
                edgecolor="black",
                linewidth=0.5,
            )

            # Color by relative intensity
            max_days = float(np.nanmax(counts)) if len(counts) else 0.0
            if max_days > 0.0:
                for bar, cnt in zip(bars, counts):
                    intensity = float(cnt) / max_days
                    color_intensity = 0.3 + 0.7 * intensity
                    bar.set_color((1.0, 1.0 - color_intensity, 1.0 - color_intensity))

            # Trends
            if show_trends and len(years) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    years, counts
                )
                trend = slope * years + intercept
                ax1.plot(
                    years,
                    trend,
                    "--",
                    color="red",
                    linewidth=3.0,
                    alpha=0.8,
                    label=f"Trend: {slope:.2f} days/year (R²={r_value**2:.3f})",
                )
                if len(counts) > 5:
                    roll = (
                        pd.Series(counts)
                        .rolling(window=5, center=True)
                        .mean()
                        .to_numpy()
                    )
                    ax1.plot(
                        years,
                        roll,
                        color=colors[1],
                        linewidth=3.0,
                        alpha=0.8,
                        label="5-year Moving Average",
                    )

            # Highlight top 3 years
            if len(counts) > 0:
                top_idx = np.argsort(counts)[-3:][::-1]
                for k in top_idx:
                    year_k = float(years[k])
                    cnt_k = float(counts[k])
                    ax1.annotate(
                        f"{int(year_k)}\n{int(cnt_k)} days",
                        xy=(year_k, cnt_k),
                        xytext=(
                            year_k,
                            cnt_k + (max_days * 0.1 if max_days > 0 else 1.0),
                        ),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                        fontsize=FONT_SIZES["tick"],
                        ha="center",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7
                        ),
                    )

            ax1.set_title(
                f"Annual Heatwave Days (>{float(threshold):.0f}°C) with Trend Analysis",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=20.0,
            )
            ax1.set_xlabel("Year", fontsize=FONT_SIZES["label"])
            ax1.set_ylabel("Number of Heatwave Days", fontsize=FONT_SIZES["label"])
            ax1.legend(fontsize=FONT_SIZES["legend"])
            ax1.grid(True, alpha=0.3, linestyle=":")

            if len(years) > 0:
                yr_max = int(np.nanmax(years))
                for decade in range(1970, 2031, 10):
                    if decade <= yr_max:
                        ax1.axvline(
                            x=float(decade),
                            color="gray",
                            alpha=0.3,
                            linestyle="-",
                            linewidth=0.8,
                        )

        # 2) Heatwave days by month (seasonal analysis)
        ax2 = fig.add_subplot(gs[0, 1])
        if _has_columns(df, ["Heatwave", "Month"]):
            monthly_counts = (
                df.loc[df["Heatwave"] == True].groupby("Month").size().to_dict()
            )  # noqa: E712
            x_pos = np.arange(1, 13, dtype=float)
            counts = np.array(
                [float(monthly_counts.get(m, 0)) for m in range(1, 13)], dtype=float
            )
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            # Seasonal colors
            seasonal_colors = []
            for month in range(1, 13):
                if month in (12, 1, 2):
                    seasonal_colors.append("#3498DB")
                elif month in (3, 4, 5):
                    seasonal_colors.append("#E74C3C")
                elif month in (6, 7, 8, 9):
                    seasonal_colors.append("#27AE60")
                else:
                    seasonal_colors.append("#F39C12")

            bars = ax2.bar(
                x_pos,
                counts,
                color=seasonal_colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.0,
            )

            if counts.size and float(np.nanmax(counts)) > 0.0:
                for bar, cnt in zip(bars, counts):
                    if cnt > 0.0:
                        ax2.text(
                            float(bar.get_x() + bar.get_width() / 2.0),
                            float(bar.get_height() + (np.nanmax(counts) * 0.01)),
                            f"{int(cnt)}",
                            ha="center",
                            va="bottom",
                            fontsize=FONT_SIZES["tick"],
                            fontweight="bold",
                        )

            # Seasonal dividers and labels
            season_boundaries = [2.5, 5.5, 9.5]
            season_labels = [
                "Winter",
                "Pre-monsoon\n(Peak Heat)",
                "Monsoon",
                "Post-monsoon",
            ]
            season_positions = [1.5, 4.0, 7.5, 11.0]
            for boundary in season_boundaries:
                ax2.axvline(
                    x=float(boundary),
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=2.0,
                )
            if counts.size:
                ymax = float(np.nanmax(counts)) * 0.9
                for pos, label in zip(season_positions, season_labels):
                    ax2.text(
                        float(pos),
                        ymax,
                        label,
                        ha="center",
                        va="center",
                        fontsize=FONT_SIZES["tick"],
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                        ),
                    )

            ax2.set_title(
                "Monthly Heatwave Distribution with Seasonal Patterns",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=20.0,
            )
            ax2.set_xlabel("Month", fontsize=FONT_SIZES["label"])
            ax2.set_ylabel("Total Heatwave Days", fontsize=FONT_SIZES["label"])
            ax2.set_xticks(x_pos, labels=month_names, rotation=45)
            ax2.grid(True, alpha=0.3, linestyle=":", axis="y")

        # 3) Rolling averages and variability/change-points
        ax3 = fig.add_subplot(gs[1, 0])
        if not heatwave_days_per_year.empty:
            years = _as_index_float_array(heatwave_days_per_year.index)
            counts = _as_float_array(heatwave_days_per_year.values)  # type: ignore

            ax3.plot(
                years,
                counts,
                "o-",
                label="Annual Heatwave Days",
                alpha=0.4,
                color=colors[0],
                markersize=4.0,
            )

            for window, color in zip([3, 5, 10], [colors[1], colors[2], colors[3]]):
                if len(counts) >= window:
                    rolling_avg = (
                        pd.Series(counts)
                        .rolling(window=window, center=True)
                        .mean()
                        .to_numpy()
                    )
                    ax3.plot(
                        years,
                        rolling_avg,
                        label=f"{window}-Year Moving Average",
                        color=color,
                        linewidth=2.5,
                        alpha=0.8,
                    )

            if len(counts) > 10:
                window_var = pd.Series(counts).rolling(window=10).var()
                if window_var.notna().any():
                    thresh = float(window_var.quantile(0.8))
                    high_var_years = years[window_var.to_numpy() > thresh]
                    if high_var_years.size:
                        first = True
                        for y in high_var_years:
                            ax3.axvline(
                                x=float(y),
                                color="orange",
                                alpha=0.6,
                                linestyle=":",
                                linewidth=2.0,
                                label="High Variability Period" if first else "",
                            )
                            first = False

            # Period highlights
            left = float(np.nanmin(years)) if len(years) else 1970.0
            right = float(np.nanmax(years)) if len(years) else 2024.0
            ax3.axvspan(
                1970.0,
                min(1990.0, right),
                alpha=0.1,
                color="blue",
                label="Early Period",
            )
            ax3.axvspan(
                max(2000.0, left), right, alpha=0.1, color="red", label="Recent Period"
            )

            total_years = int(len(counts))
            total_days = float(np.nansum(counts))
            avg_per_year = float(np.nanmean(counts)) if total_years > 0 else 0.0

            stats_text = (
                f"Total Years: {total_years}\n"
                f"Total Heatwave Days: {int(total_days)}\n"
                f"Average per Year: {avg_per_year:.1f}"
            )
            ax3.text(
                0.02,
                0.98,
                stats_text,
                transform=ax3.transAxes,
                fontsize=FONT_SIZES["tick"],
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            )

            ax3.set_title(
                "Heatwave Frequency Trends with Multi-Scale Analysis",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=20.0,
            )
            ax3.set_xlabel("Year", fontsize=FONT_SIZES["label"])
            ax3.set_ylabel("Number of Heatwave Days", fontsize=FONT_SIZES["label"])
            ax3.legend(fontsize=FONT_SIZES["legend"], loc="upper left")
            ax3.grid(True, alpha=0.3, linestyle=":")

        # 4) Duration (preferred) or intensity distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if (
            isinstance(heatwave_summary, pd.DataFrame)
            and not heatwave_summary.empty
            and "Duration" in heatwave_summary.columns
        ):
            durations = pd.to_numeric(
                heatwave_summary["Duration"], errors="coerce"
            ).dropna()
            if not durations.empty:
                unique_bins = int(min(20, max(5, durations.nunique())))
                counts, bins, patches = ax4.hist(
                    durations.to_numpy(dtype=float),
                    bins=unique_bins,
                    alpha=0.7,
                    color=colors[0],
                    edgecolor="black",
                    linewidth=1.0,
                    density=False,
                )
                max_count = float(max(counts)) if len(counts) else 0.0
                if max_count > 0.0:
                    for c, patch in zip(counts, patches):
                        intensity = float(c) / max_count
                        patch.set_color((1.0, 0.6 - 0.3 * intensity, 0.0))

                mean_d = float(durations.mean())
                median_d = float(durations.median())
                p75 = float(durations.quantile(0.75))
                p95 = float(durations.quantile(0.95))

                ax4.axvline(
                    mean_d,
                    color="red",
                    linestyle="--",
                    linewidth=2.0,
                    label=f"Mean: {mean_d:.1f} days",
                )
                ax4.axvline(
                    median_d,
                    color="blue",
                    linestyle="--",
                    linewidth=2.0,
                    label=f"Median: {median_d:.1f} days",
                )
                ax4.axvline(
                    p75,
                    color="orange",
                    linestyle=":",
                    linewidth=2.0,
                    label=f"75th percentile: {p75:.1f} days",
                )
                ax4.axvline(
                    p95,
                    color="purple",
                    linestyle=":",
                    linewidth=2.0,
                    label=f"95th percentile: {p95:.1f} days",
                )

                stats_text = (
                    f"Total Events: {int(durations.size)}\n"
                    f"Max Duration: {int(durations.max())} days\n"
                    f"Std Dev: {float(durations.std(ddof=0)):.1f} days"
                )
                ax4.text(
                    0.98,
                    0.98,
                    stats_text,
                    transform=ax4.transAxes,
                    fontsize=FONT_SIZES["tick"],
                    va="top",
                    ha="right",
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
                    ),
                )

                ax4.set_title(
                    "Heatwave Duration Distribution with Statistics",
                    fontsize=FONT_SIZES["title"],
                    fontweight="bold",
                    pad=20.0,
                )
                ax4.set_xlabel("Duration (days)", fontsize=FONT_SIZES["label"])
                ax4.set_ylabel("Frequency", fontsize=FONT_SIZES["label"])
                ax4.legend(fontsize=FONT_SIZES["legend"], loc="upper right")
                ax4.grid(True, alpha=0.3, linestyle=":")
        else:
            # Fallback: intensity distribution
            if _has_columns(df, ["Heatwave", TEMP_COLUMN]):
                heatwave_temps = pd.to_numeric(
                    df.loc[df["Heatwave"], TEMP_COLUMN], errors="coerce"
                ).dropna()  # noqa: E712
                if not heatwave_temps.empty:
                    ax4.hist(
                        heatwave_temps.to_numpy(dtype=float),
                        bins=30,
                        alpha=0.7,
                        color=colors[0],
                        edgecolor="black",
                        linewidth=1.0,
                    )
                    ax4.axvline(
                        float(threshold),
                        color="red",
                        linestyle="--",
                        linewidth=2.0,
                        label=f"Threshold: {float(threshold):.0f}°C",
                    )
                    ax4.set_title(
                        "Heatwave Temperature Intensity Distribution",
                        fontsize=FONT_SIZES["title"],
                        fontweight="bold",
                        pad=20.0,
                    )
                    ax4.set_xlabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
                    ax4.set_ylabel("Frequency", fontsize=FONT_SIZES["label"])
                    ax4.legend(fontsize=FONT_SIZES["legend"])
                    ax4.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()
        _safe_savefig(fig, save_path)
        return fig

    except Exception as e:
        logger.error("Error creating heatwave analysis plot: %s", e)
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        ax.text(
            0.5,
            0.5,
            f"Error creating plot: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig


# ------------------------------------------------------------------------------
# Plot 4: Correlation matrix (significance-aware)
# ------------------------------------------------------------------------------


def plot_correlation_matrix(
    data: pd.DataFrame,
    method: str = "pearson",
    save_path: Optional[str] = None,
    include_derived: bool = True,
    significance_test: bool = True,
) -> Figure:
    """
    Correlation matrix with optional significance stars.
    Uses literal strings when calling pandas.DataFrame.corr to satisfy Pylance.
    """
    logger.info("Creating correlation matrix (method=%s)", method)

    try:
        # Select columns
        base_columns = [
            "Dhaka Temperature [2 m elevation corrected]",
            "Dhaka Precipitation Total",
            "Dhaka Relative Humidity [2 m]",
            "Dhaka Wind Speed [10 m]",
            "Dhaka Cloud Cover Total",
            "Dhaka Sunshine Duration",
            "Dhaka Mean Sea Level Pressure [MSL]",
        ]
        if include_derived:
            base_columns += [
                "Heat_Index",
                "Temp_MA_7",
                "Temp_MA_30",
                "Seasonal_Temp_Deviation",
                "Temp_Percentile",
            ]

        available = [c for c in base_columns if c in data.columns]
        if len(available) < 2:
            logger.warning("Insufficient variables for correlation analysis")
            fig, ax = plt.subplots(figsize=(8.0, 6.0))
            ax.text(
                0.5,
                0.5,
                "Insufficient variables for correlation analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        df_num = (
            data[available]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(axis=1, how="all")
        )
        if df_num.shape[1] < 2:
            fig, ax = plt.subplots(figsize=(8.0, 6.0))
            ax.text(
                0.5,
                0.5,
                "Not enough numeric columns for correlation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Compute correlation with literal method to appease Pylance stubs
        m = method.lower().strip()
        if m not in ("pearson", "spearman", "kendall"):
            logger.warning("Unknown method '%s'; falling back to 'pearson'", method)
            m = "pearson"

        if m == "pearson":
            corr = df_num.corr(method="pearson")  # literal
        elif m == "spearman":
            corr = df_num.corr(method="spearman")  # literal
        else:
            corr = df_num.corr(method="kendall")  # literal

        # Figure size scales with number of variables
        n_vars = int(len(corr.columns))
        fig = plt.figure(
            figsize=(max(12.0, n_vars * 0.8), max(10.0, n_vars * 0.8)),
            dpi=DPI,
        )

        # Optional significance mask
        significance_mask: Optional[np.ndarray] = None
        if significance_test and len(df_num) > 10:
            p_values = np.zeros_like(corr.to_numpy(dtype=float))
            cols = list(corr.columns)
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue
                    col_i = cols[i]
                    col_j = cols[j]
                    pair = df_num[[col_i, col_j]].dropna()
                    if len(pair) > 3:
                        x = pair[col_i].to_numpy(dtype=float)
                        y = pair[col_j].to_numpy(dtype=float)
                        # Remove non-finite values defensively
                        mask_finite = np.isfinite(x) & np.isfinite(y)
                        x = x[mask_finite]
                        y = y[mask_finite]
                        if x.size > 3 and y.size > 3:
                            if m == "pearson":
                                _, p = stats.pearsonr(x, y)
                            elif m == "spearman":
                                _, p = stats.spearmanr(x, y)
                            else:
                                _, p = stats.kendalltau(x, y)
                            p_values[i, j] = float(p)
                        else:
                            p_values[i, j] = np.nan
                    else:
                        p_values[i, j] = np.nan
            significance_mask = (p_values > 0.05) | ~np.isfinite(p_values)
            np.fill_diagonal(significance_mask, False)  # diagonal not masked

        # Plot heatmap (lower triangle)
        mask_upper = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(
            corr.astype(float),
            mask=mask_upper,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            linewidths=0.8,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": FONT_SIZES["tick"]},
            alpha=0.8 if significance_mask is None else 1.0,
        )

        # Overlay significance stars on significant cells (lower triangle only)
        if significance_mask is not None:
            cols = list(corr.columns)
            for i in range(n_vars):
                for j in range(i):  # j < i ensures lower triangle
                    if not significance_mask[i, j]:
                        plt.text(
                            float(j) + 0.5,
                            float(i) + 0.5,
                            "*",
                            fontsize=18.0,
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                        )

        # Titles & labels
        title_method = m.capitalize()
        plt.title(
            f"{title_method} Correlation Matrix of Climate Variables\n"
            f"({df_num.shape[1]} variables, {len(df_num):,} observations)",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            pad=20.0,
        )
        plt.xticks(rotation=45, ha="right", fontsize=FONT_SIZES["tick"])
        plt.yticks(rotation=0, fontsize=FONT_SIZES["tick"])

        # Legend note
        if significance_test:
            plt.figtext(
                0.02,
                0.02,
                "* Statistically significant (p < 0.05)",
                fontsize=FONT_SIZES["tick"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Summary box (counts of strong/moderate)
        corr_vals = corr.to_numpy(dtype=float)
        strong = int((np.abs(corr_vals) > 0.7).sum() - n_vars) // 2
        moderate = (
            int(((np.abs(corr_vals) > 0.3) & (np.abs(corr_vals) <= 0.7)).sum()) // 2
        )
        plt.figtext(
            0.98,
            0.02,
            f"Strong (|r| > 0.7): {strong}\nModerate (0.3 < |r| ≤ 0.7): {moderate}",
            fontsize=FONT_SIZES["tick"],
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        _safe_savefig(fig, save_path)
        return fig

    except Exception as e:
        logger.error("Error creating correlation matrix: %s", e)
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        ax.text(
            0.5,
            0.5,
            f"Error creating correlation matrix: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig


# ------------------------------------------------------------------------------
# Plot 5: Summary dashboard
# ------------------------------------------------------------------------------


def create_summary_dashboard(
    data: pd.DataFrame,
    statistical_results: Dict[str, Any],
    tree_loss_by_year: pd.DataFrame,
    save_path: Optional[str] = None,
    include_projections: bool = True,
) -> Figure:
    """
    Multi-panel dashboard summarizing climate, heatwaves, and deforestation.
    Icons/emojis removed. Numeric conversions and bounds checks included.
    """
    logger.info("Creating summary dashboard")

    try:
        df = data.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "Year" not in df.columns and "timestamp" in df.columns:
            df["Year"] = df["timestamp"].dt.year

        fig = plt.figure(figsize=(20.0, 14.0), dpi=DPI)
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

        colors = COLOR_SCHEMES["climate"]
        temp_col = TEMP_COLUMN

        # Dataset time span
        total_years = int(df["Year"].nunique()) if "Year" in df.columns else 0
        start_year = int(df["Year"].min()) if "Year" in df.columns else 0
        end_year = int(df["Year"].max()) if "Year" in df.columns else 0

        # Basic temperature stats
        if temp_col in df.columns:
            temp_series = pd.to_numeric(df[temp_col], errors="coerce")
            avg_temp = float(temp_series.mean())
            max_temp = float(temp_series.max())
            min_temp = float(temp_series.min())
        else:
            avg_temp = max_temp = min_temp = float("nan")

        # Heatwave stats
        if "Heatwave" in df.columns:
            total_heatwave_days = int(
                pd.to_numeric(df["Heatwave"], errors="coerce")
                .fillna(0)
                .astype(bool)
                .sum()
            )
        else:
            total_heatwave_days = 0
        avg_heatwave_per_year = (
            float(total_heatwave_days / total_years) if total_years > 0 else 0.0
        )

        # Deforestation metrics
        deforest_col: Optional[str] = None
        for c in tree_loss_by_year.columns:
            low = c.lower()
            if "loss" in low and "ha" in low:
                deforest_col = c
                break

        total_deforestation = 0.0
        deforest_years = 0
        if deforest_col and "Year" in tree_loss_by_year.columns:
            dser = pd.to_numeric(
                tree_loss_by_year[deforest_col], errors="coerce"
            ).fillna(0.0)
            total_deforestation = float(dser.sum())
            deforest_years = int((dser > 0).sum())

        # ------------------------------------------------------------------
        # (1) Temperature trend with simple projection
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        if _has_columns(df, [temp_col, "Year"]):
            annual_mean = (
                pd.to_numeric(df[temp_col], errors="coerce").groupby(df["Year"]).mean()
            )
            x_years = _as_index_float_array(annual_mean.index)
            y_means = _as_float_array(annual_mean.values)

            ax1.plot(
                x_years,
                y_means,
                linewidth=3.0,
                color=colors[0],
                label="Annual Mean",
                marker="o",
                markersize=4.0,
            )

            if len(x_years) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_years, y_means
                )
                trend = slope * x_years + intercept
                ax1.plot(
                    x_years,
                    trend,
                    "--",
                    color="red",
                    linewidth=2.0,
                    alpha=0.8,
                    label=f"Trend: {slope:.4f}°C/year",
                )
                if include_projections:
                    if len(x_years) > 0:
                        last = int(np.nanmax(x_years))
                        future_years = np.arange(last + 1, last + 11, dtype=float)
                        future_vals = slope * future_years + intercept
                        ax1.plot(
                            future_years,
                            future_vals,
                            ":",
                            color="orange",
                            linewidth=2.0,
                            alpha=0.8,
                            label="10-year Projection",
                        )

            if y_means.size:
                idx_max = int(np.nanargmax(y_means))
                idx_min = int(np.nanargmin(y_means))
                ax1.scatter(
                    [x_years[idx_max], x_years[idx_min]],
                    [y_means[idx_max], y_means[idx_min]],
                    color=["red", "blue"],
                    s=100.0,
                    zorder=5,
                )

            ax1.set_title(
                f"Temperature Trend Analysis ({start_year}-{end_year})",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=15.0,
            )
            ax1.set_ylabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
            ax1.legend(fontsize=FONT_SIZES["legend"])
            ax1.grid(True, alpha=0.3, linestyle=":")

        # ------------------------------------------------------------------
        # (2) Heatwave frequency
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        if _has_columns(df, ["Heatwave", "Year"]):
            hw_yearly = (
                df.loc[df["Heatwave"] == True].groupby("Year").size()
            )  # noqa: E712
            all_years = pd.Index(sorted(df["Year"].dropna().unique()))
            hw_yearly = hw_yearly.reindex(all_years, fill_value=0).astype(float)

            yrs = _as_index_float_array(hw_yearly.index)
            vals = _as_float_array(hw_yearly.values)
            if vals.size == 0:
                vals = np.zeros_like(yrs)

            max_days = float(np.nanmax(vals)) if vals.size else 1.0
            bar_colors = [
                plt.cm.get_cmap("Reds")(
                    0.3 + 0.7 * (v / max_days if max_days > 0 else 0.0)
                )
                for v in vals
            ]

            ax2.bar(
                yrs,
                vals,
                color=bar_colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

            if len(yrs) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    yrs, vals
                )
                trend = slope * yrs + intercept
                ax2.plot(
                    yrs,
                    trend,
                    "--",
                    color="darkred",
                    linewidth=3.0,
                    alpha=0.8,
                    label=f"Trend: {slope:.2f} days/year",
                )

            if vals.size and float(np.nanmax(vals)) > 0.0:
                idx = int(np.nanargmax(vals))
                ax2.annotate(
                    f"{int(yrs[idx])}\n{int(vals[idx])} days",
                    xy=(float(yrs[idx]), float(vals[idx])),
                    xytext=(
                        float(yrs[idx]),
                        float(vals[idx] + (0.1 * np.nanmax(vals))),
                    ),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
                    fontsize=FONT_SIZES["tick"],
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8),
                )

            ax2.set_title(
                f"Heatwave Frequency Trends — Total: {total_heatwave_days:,} days",
                fontsize=FONT_SIZES["title"],
                fontweight="bold",
                pad=15.0,
            )
            ax2.set_ylabel("Annual Heatwave Days", fontsize=FONT_SIZES["label"])
            ax2.legend(fontsize=FONT_SIZES["legend"])
            ax2.grid(True, alpha=0.3, linestyle=":")

        # ------------------------------------------------------------------
        # (3) Deforestation analysis
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        if deforest_col and "Year" in tree_loss_by_year.columns:
            ysr = tree_loss_by_year["Year"]
            y = _as_index_float_array(ysr)
            losses = (
                pd.to_numeric(tree_loss_by_year[deforest_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            max_loss = float(np.nanmax(losses)) if losses.size else 1.0
            colormap = plt.cm.get_cmap("Greens_r")
            bar_cols = [
                colormap(0.3 + 0.7 * (v / max_loss if max_loss > 0 else 0.0))
                for v in losses
            ]

            ax3.bar(
                y,
                losses,
                color=bar_cols,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

            cum = np.cumsum(losses)
            ax3_t = ax3.twinx()
            ax3_t.plot(
                y,
                cum,
                color="darkred",
                linewidth=3.0,
                alpha=0.8,
                label="Cumulative Loss",
            )
            ax3_t.set_ylabel(
                "Cumulative Loss (ha)", fontsize=FONT_SIZES["label"], color="darkred"
            )
            ax3_t.tick_params(axis="y", labelcolor="darkred")

            ax3.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
            ax3_t.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

        ax3.set_title(
            f"Deforestation Impact — {total_deforestation:,.0f} ha lost ({deforest_years} years)",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            pad=15.0,
        )
        ax3.set_ylabel("Annual Tree Loss (ha)", fontsize=FONT_SIZES["label"])
        ax3.grid(True, alpha=0.3, linestyle=":")

        # ------------------------------------------------------------------
        # (4) Decadal temperature analysis
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 0])
        if _has_columns(df, [temp_col, "Year"]):
            work = df.copy()
            work["Decade"] = (work["Year"] // 10) * 10
            dec = (
                pd.to_numeric(work[temp_col], errors="coerce")
                .groupby(work["Decade"])
                .agg(["mean", "std", "count"])
                .dropna()
            )

            if not dec.empty:
                labels = [f"{int(d)}s" for d in dec.index]
                means = dec["mean"].to_numpy(dtype=float)
                stds = dec["std"].to_numpy(dtype=float)
                counts = dec["count"].to_numpy(dtype=float)

                if means.size:
                    rng = (
                        float(np.nanmax(means) - np.nanmin(means))
                        if means.size > 1
                        else 1.0
                    )
                    cmap = plt.cm.get_cmap("coolwarm")
                    colors_dec = [
                        cmap((m - float(np.nanmin(means))) / rng if rng > 0 else 0.5)
                        for m in means
                    ]

                    bars = ax4.bar(
                        labels,
                        means,
                        yerr=stds,
                        capsize=5.0,
                        color=colors_dec,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=1.0,
                        error_kw={"elinewidth": 2.0},
                    )

                    for bar, mean, stdv, cnt in zip(bars, means, stds, counts):
                        height = float(bar.get_height())
                        ax4.text(
                            float(bar.get_x() + bar.get_width() / 2.0),
                            height + float(stdv) + 0.2,
                            f"{mean:.1f}°C\n(n={int(cnt):,})",
                            ha="center",
                            va="bottom",
                            fontsize=FONT_SIZES["tick"],
                            fontweight="bold",
                        )

        ax4.set_title(
            "Decadal Temperature Analysis",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            pad=15.0,
        )
        ax4.set_ylabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3, linestyle=":", axis="y")

        # ------------------------------------------------------------------
        # (5) Seasonal climate patterns
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(gs[1, 1])
        if _has_columns(df, [temp_col, "Season"]):
            seas = (
                pd.to_numeric(df[temp_col], errors="coerce")
                .groupby(df["Season"])
                .agg(["mean", "std", "count"])
                .dropna()
            )
            if not seas.empty:
                season_names = {
                    1: "Winter\n(Dec–Feb)",
                    2: "Spring\n(Mar–May)",
                    3: "Summer\n(Jun–Aug)",
                    4: "Autumn\n(Sep–Nov)",
                }
                labels = [
                    season_names.get(int(s), f"Season {int(s)}") for s in seas.index
                ]
                means = seas["mean"].to_numpy(dtype=float)
                stds = seas["std"].to_numpy(dtype=float)
                counts = seas["count"].to_numpy(dtype=float)

                seasonal_cols = ["#3498DB", "#E74C3C", "#27AE60", "#F39C12"]
                bars = ax5.bar(
                    labels,
                    means,
                    yerr=stds,
                    capsize=5.0,
                    color=seasonal_cols[: len(labels)],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.0,
                    error_kw={"elinewidth": 2.0},
                )

                # Overlay heatwave counts per season if available
                if "Heatwave" in df.columns:
                    hw_by_season = (
                        df.loc[df["Heatwave"] == True].groupby("Season").size()
                    )  # noqa: E712
                    for i, s in enumerate(seas.index):
                        count_hw = int(hw_by_season.get(int(s), 0))
                        ax5.text(
                            float(i),
                            float(means[i] + stds[i] + 1.0),
                            f"{count_hw} HW days",
                            ha="center",
                            va="bottom",
                            fontsize=FONT_SIZES["tick"],
                            color="red",
                            fontweight="bold",
                        )

                overall_mean = float(
                    pd.to_numeric(df[temp_col], errors="coerce").mean()
                )
                ax5.axhline(
                    y=overall_mean,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2.0,
                    label=f"Annual Mean: {overall_mean:.1f}°C",
                )

        ax5.set_title(
            "Seasonal Climate Patterns",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            pad=15.0,
        )
        ax5.set_ylabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
        ax5.legend(fontsize=FONT_SIZES["legend"])
        ax5.grid(True, alpha=0.3, linestyle=":", axis="y")

        # ------------------------------------------------------------------
        # (6) Statistics panel
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")

        completeness = float(len(df.dropna()) / len(df) * 100.0) if len(df) else 0.0
        stats_text = (
            "CLIMATE ANALYSIS SUMMARY\n\n"
            "Dataset Overview:\n"
            f"• Period: {start_year} - {end_year} ({total_years} years)\n"
            f"• Total Records: {len(df):,}\n"
            f"• Data Completeness: {completeness:.1f}%\n\n"
            "Temperature Statistics:\n"
            f"• Average: {avg_temp:.2f}°C\n"
            f"• Maximum: {max_temp:.1f}°C\n"
            f"• Minimum: {min_temp:.1f}°C\n"
            f"• Range: {float(max_temp - min_temp):.1f}°C\n\n"
            "Heatwave Analysis:\n"
            f"• Total Days: {total_heatwave_days:,}\n"
            f"• Average per Year: {avg_heatwave_per_year:.1f}\n"
            f"• Frequency: {float(total_heatwave_days / len(df) * 100.0 if len(df) else 0.0):.2f}% of days\n\n"
            "Deforestation Impact:\n"
            f"• Total Forest Loss: {total_deforestation:,.0f} ha\n"
            f"• Active Loss Years: {deforest_years}\n"
            f"• Average Annual Loss: {float(total_deforestation / deforest_years) if deforest_years else 0.0:,.0f} ha/year"
        )

        # Trend summary from statistical_results if present
        tr = statistical_results.get("temperature_trend", {})
        lin = tr.get("linear_regression", {}) if isinstance(tr, dict) else {}
        slope = float(lin.get("slope", 0.0))
        r2 = float(lin.get("r_squared", 0.0))
        if slope != 0.0 or r2 != 0.0:
            total_increase = slope * float(total_years)
            stats_text += (
                "\n\nClimate Trends:\n"
                f"• Warming Rate: {slope:.4f}°C/year\n"
                f"• Total Increase: {total_increase:.2f}°C over {total_years} years\n"
                f"• Trend Strength: R² = {r2:.3f}"
            )

        # Extreme events summary
        if temp_col in df.columns:
            p95 = float(pd.to_numeric(df[temp_col], errors="coerce").quantile(0.95))
            extreme_days = int(
                (pd.to_numeric(df[temp_col], errors="coerce") > p95).sum()
            )
            stats_text += (
                "\n\nExtreme Events:\n"
                f"• 95th Percentile: {p95:.1f}°C\n"
                f"• Extreme Heat Days: {extreme_days:,}\n"
                f"• Extreme Frequency: {float(extreme_days / len(df) * 100.0 if len(df) else 0.0):.2f}%"
            )

        ax6.text(
            0.05,
            0.95,
            stats_text,
            fontsize=FONT_SIZES["tick"],
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            transform=ax6.transAxes,
            family="monospace",
        )

        # ------------------------------------------------------------------
        # (7) Climate change indicators: anomalies timeline
        # ------------------------------------------------------------------
        ax7 = fig.add_subplot(gs[2, :])
        if _has_columns(df, [temp_col, "Year"]):
            annual = (
                pd.to_numeric(df[temp_col], errors="coerce").groupby(df["Year"]).mean()
            )
            if not annual.empty:
                base = float(annual.mean())
                anomalies = annual - base
                x = _as_index_float_array(anomalies.index)
                y = _as_float_array(anomalies.values)

                # color by sign of anomaly
                colors_scatter = ["red" if val > 0 else "blue" for val in y]
                intensities = np.abs(y) / (
                    np.nanmax(np.abs(y)) if np.nanmax(np.abs(y)) > 0 else 1.0
                )
                sizes = 50.0 + 100.0 * intensities

                ax7.scatter(
                    x,
                    y,
                    c=colors_scatter,
                    s=sizes,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                ax7.axhline(
                    y=0.0, color="gray", linestyle="-", alpha=0.5, linewidth=2.0
                )

                if len(x) > 3:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    trend = slope * x + intercept
                    ax7.plot(
                        x,
                        trend,
                        "--",
                        color="darkred",
                        linewidth=3.0,
                        alpha=0.8,
                        label=f"Trend: {slope:.4f}°C/year",
                    )

                # Period highlights
                if x.size:
                    left = float(np.nanmin(x))
                    right = float(np.nanmax(x))
                    ax7.axvspan(
                        max(1970.0, left),
                        min(1990.0, right),
                        alpha=0.1,
                        color="blue",
                        label="Early Period",
                    )
                    ax7.axvspan(
                        max(2000.0, left),
                        right,
                        alpha=0.1,
                        color="red",
                        label="Recent Warming",
                    )

                ax7.set_title(
                    "Climate Change Timeline: Temperature Anomalies from Long-term Average",
                    fontsize=FONT_SIZES["title"],
                    fontweight="bold",
                    pad=20.0,
                )
                ax7.set_xlabel("Year", fontsize=FONT_SIZES["label"])
                ax7.set_ylabel("Temperature Anomaly (°C)", fontsize=FONT_SIZES["label"])
                ax7.legend(fontsize=FONT_SIZES["legend"])
                ax7.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()
        _safe_savefig(fig, save_path)
        return fig

    except Exception as e:
        logger.error("Error creating summary dashboard: %s", e)
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        ax.text(
            0.5,
            0.5,
            f"Error creating dashboard: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig
