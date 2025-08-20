"""
Visualization Module with Enhanced KDE Support and File Saving
"""

import os
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

# Set high-quality defaults for publication
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"

# Constants for visualization
DEFAULT_DPI = 300
RECENT_YEARS_THRESHOLD = 2020
TEMP_COLUMN = "Dhaka Temperature [2 m elevation corrected]"


def _pick_temp_col(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    """Pick the best available temperature column"""
    if preferred and preferred in df.columns:
        return preferred
    for c in ("temp_max_c", "temp_mean_c", "temp_min_c"):
        if c in df.columns:
            return c
    # last resort: try legacy raw names from the data dictionary
    from data_dictionary import TEMPERATURE_COLUMNS

    for key in ("daily_max", "daily_mean", "daily_min"):
        raw = TEMPERATURE_COLUMNS.get(key)
        if raw in df.columns:
            return raw
    return df.columns[0] if len(df.columns) > 0 else None


def _save_plot(filename: str, output_dir: str | Path, dpi: int = DEFAULT_DPI) -> None:
    """Save plot with consistent settings.

    Args:
        filename: Name of the file to save
        output_dir: Directory to save the file in
        dpi: Resolution for the saved image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_path = output_path / filename
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {full_path}")


def _plot_recent_daily_temps(
    ax: matplotlib.axes.Axes, data: pd.DataFrame, primary_temp_col: str | None = None
) -> None:
    if "Year" not in data.columns:
        data = data.copy()
        data["Year"] = data.index.year  # type: ignore

    if primary_temp_col is None:
        temp_cols = [c for c in data.columns if c.startswith("temp_")]
        if temp_cols:
            primary_temp_col = temp_cols[0]
        else:
            raise ValueError("No canonical temperature columns found in data")

    recent_data = data[data["Year"] >= RECENT_YEARS_THRESHOLD].copy().sort_index()

    ax.plot(
        recent_data.index,
        recent_data[primary_temp_col],
        alpha=0.7,
        linewidth=1,
        color="skyblue",
    )

    x_numeric = np.arange(len(recent_data))
    z = np.polyfit(x_numeric, recent_data[primary_temp_col], 1)
    p = np.poly1d(z)
    ax.plot(
        recent_data.index,
        p(x_numeric),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: {z[0]:.4f}°C/day",
    )

    ax.set_title(
        f"Daily Temperature ({RECENT_YEARS_THRESHOLD}-2024) with Trend",
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_annual_temperature_trends(
    ax: matplotlib.axes.Axes,
    annual_temp_stats: pd.DataFrame,
    primary_temp_col: str | None = None,
) -> None:
    """Plot annual temperature trends with trendlines using canonical columns.

    Args:
        ax: Matplotlib axis to plot on
        annual_temp_stats: DataFrame with annual temperature statistics
        primary_temp_col: Primary temperature column name (canonical)
    """
    # Use the canonical column naming from forensic fix
    if primary_temp_col is None:
        # Fallback to detect primary column from available columns
        temp_cols = [
            col for col in annual_temp_stats.columns if col.startswith("temp_")
        ]
        if temp_cols:
            # Extract base column name (remove suffix)
            primary_temp_col = (
                temp_cols[0].split("_")[0]
                + "_"
                + temp_cols[0].split("_")[1]
                + "_"
                + temp_cols[0].split("_")[2]
            )
            primary_temp_col = temp_cols[0].rsplit("_", 1)[
                0
            ]  # Remove suffix like '_mean'
        else:
            raise ValueError(
                "No canonical temperature columns found in annual_temp_stats"
            )

    years: np.ndarray = annual_temp_stats["Year"].values  # type: ignore

    # Build column names programmatically using canonical_summstat pattern
    mean_key = f"{primary_temp_col}_mean"
    max_key = f"{primary_temp_col}_max"

    if (
        mean_key not in annual_temp_stats.columns
        or max_key not in annual_temp_stats.columns
    ):
        raise KeyError(
            f"Expected columns {mean_key} and {max_key} not found in annual_temp_stats"
        )

    mean_temps: np.ndarray = annual_temp_stats[mean_key].values  # type: ignore
    max_temps: np.ndarray = annual_temp_stats[max_key].values  # type: ignore

    ax.plot(
        years,
        mean_temps,
        marker="o",
        linewidth=2,
        markersize=4,
        color="red",
        label="Mean Temp",
    )
    ax.plot(
        years,
        max_temps,
        marker="s",
        linewidth=1,
        markersize=3,
        color="darkred",
        label="Max Temp",
    )

    # Add trendlines
    z_mean = np.polyfit(years, mean_temps, 1)
    z_max = np.polyfit(years, max_temps, 1)
    p_mean = np.poly1d(z_mean)
    p_max = np.poly1d(z_max)

    ax.plot(
        years,
        p_mean(years),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Mean Trend: +{z_mean[0]*52:.2f}°C/52yr",
    )
    ax.plot(
        years,
        p_max(years),
        "--",
        color="darkred",
        alpha=0.8,
        linewidth=2,
        label=f"Max Trend: +{z_max[0]*52:.2f}°C/52yr",
    )

    ax.set_title("Annual Temperature Trends (1972-2024)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_temperature_distribution(
    ax: matplotlib.axes.Axes, data: pd.DataFrame, primary_temp_col: str | None = None
) -> None:
    """Plot temperature distribution with KDE and normal curve overlay using canonical columns.

    Args:
        ax: Matplotlib axis to plot on
        data: DataFrame with temperature data
        primary_temp_col: Primary temperature column name (canonical)
    """
    # Use canonical column name if provided
    if primary_temp_col is None:
        # Fallback to detect primary column
        temp_cols = [col for col in data.columns if col.startswith("temp_")]
        if temp_cols:
            primary_temp_col = temp_cols[0]
        else:
            raise ValueError("No canonical temperature columns found in data")

    temp_data = data[primary_temp_col].dropna()
    n, bins, patches = ax.hist(
        temp_data, bins=50, alpha=0.6, edgecolor="black", density=True, color="skyblue"
    )

    # Add KDE curve
    kde = gaussian_kde(temp_data)
    x_kde = np.linspace(temp_data.min(), temp_data.max(), 200)
    kde_curve = kde(x_kde)
    ax.plot(
        x_kde,
        kde_curve,
        "g-",
        linewidth=3,
        label="KDE (Actual Distribution)",
        alpha=0.9,
    )

    # Overlay normal distribution curve for comparison
    mu, sigma = temp_data.mean(), temp_data.std()
    x = np.linspace(temp_data.min(), temp_data.max(), 200)
    normal_curve = (
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )
    ax.plot(
        x,
        normal_curve,
        "r--",
        linewidth=2,
        label=f"Normal (μ={mu:.1f}, σ={sigma:.1f})",
        alpha=0.8,
    )

    ax.set_title("Temperature Distribution: KDE vs Normal", fontweight="bold")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_temperature_extremes(
    ax: matplotlib.axes.Axes, data: pd.DataFrame, temp_column: str | None = TEMP_COLUMN
) -> str | None:
    """Plot temperature extremes and range trends.

    Args:
        ax: Matplotlib axis to plot on
        data: DataFrame with temperature data
        temp_column: Column name for temperature data
    """
    # Ensure temp_column is not None
    if temp_column is None:
        raise ValueError("temp_column must be a valid string, not None.")
    # Calculate annual extremes
    annual_extremes = data.groupby("Year")[temp_column].agg(["min", "max"])
    annual_range = annual_extremes["max"] - annual_extremes["min"]

    ax.plot(
        annual_extremes.index,
        annual_extremes["max"],
        "r-o",
        markersize=3,
        label="Annual Max",
        alpha=0.8,
    )
    ax.plot(
        annual_extremes.index,
        annual_extremes["min"],
        "b-o",
        markersize=3,
        label="Annual Min",
        alpha=0.8,
    )

    # Add trendlines for extremes
    z_max_extreme = np.polyfit(annual_extremes.index, annual_extremes["max"], 1)
    z_min_extreme = np.polyfit(annual_extremes.index, annual_extremes["min"], 1)
    p_max_extreme = np.poly1d(z_max_extreme)
    p_min_extreme = np.poly1d(z_min_extreme)

    ax.plot(
        annual_extremes.index,
        p_max_extreme(annual_extremes.index),
        "r--",
        alpha=0.6,
        linewidth=2,
    )
    ax.plot(
        annual_extremes.index,
        p_min_extreme(annual_extremes.index),
        "b--",
        alpha=0.6,
        linewidth=2,
    )

    # Add temperature range on secondary y-axis
    ax_twin = ax.twinx()
    ax_twin.fill_between(
        annual_extremes.index,
        annual_range,
        alpha=0.2,
        color="gray",
        label="Temperature Range",
    )
    ax_twin.set_ylabel("Temperature Range (°C)", color="gray")

    ax.set_title("Temperature Extremes & Range Trends", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_temperature_trends(
    data: pd.DataFrame,
    annual_temp_stats: pd.DataFrame,
    primary_temp_col: str | None = None,
    output_dir: str | Path = "images/data_exploration",
) -> None:
    """Plot temperature trends over time with enhanced trendlines.

    Args:
        data: DataFrame with daily temperature data
        annual_temp_stats: DataFrame with annual temperature statistics
        primary_temp_col: Primary temperature column name (canonical)
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Use helper functions for each subplot to improve maintainability
    _plot_recent_daily_temps(axes[0, 0], data, primary_temp_col)
    _plot_annual_temperature_trends(axes[0, 1], annual_temp_stats, primary_temp_col)
    _plot_temperature_distribution(axes[1, 0], data, primary_temp_col)
    _plot_temperature_extremes(axes[1, 1], data, primary_temp_col)

    plt.tight_layout()
    _save_plot("temperature_trends_analysis.png", output_dir)
    plt.close()


def plot_deforestation_analysis(
    tree_loss_by_year, combined_data, output_dir="images/data_exploration"
):
    """Plot deforestation analysis (robust to canonical/legacy names)"""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # --- normalize deforestation year/loss columns ---
    if isinstance(tree_loss_by_year, pd.Series):
        dfx = tree_loss_by_year.to_frame(name="tree_loss_ha")
        if (
            dfx.index.name
            and isinstance(dfx.index.name, str)
            and dfx.index.name.lower() == "year"
        ):
            dfx = dfx.reset_index().rename(columns={dfx.index.name: "Year"})
        elif "Year" not in dfx.columns:
            dfx = dfx.reset_index().rename(columns={"index": "Year"})
    else:
        dfx = tree_loss_by_year.copy()
        if "year" in dfx.columns and "Year" not in dfx.columns:
            dfx = dfx.rename(columns={"year": "Year"})
        if "tree_loss_ha" not in dfx.columns:
            if "umd_tree_cover_loss__ha" in dfx.columns:
                dfx = dfx.rename(columns={"umd_tree_cover_loss__ha": "tree_loss_ha"})
            else:
                # last resort: first non-Year numeric as tree loss
                cand = [c for c in dfx.columns if c != "Year"]
                if cand:
                    dfx = dfx.rename(columns={cand[0]: "tree_loss_ha"})

    # --- normalize combined_data to annual means for scatter/compare ---
    temp_col = _pick_temp_col(combined_data)
    if "Year" not in combined_data.columns:
        combined_data = combined_data.copy()
        combined_data["Year"] = combined_data.index.year

    # ensure tree_loss_ha column exists in combined annual frame
    annual = combined_data.groupby("Year")[temp_col].mean().to_frame(f"{temp_col}_mean")
    if "tree_loss_ha" in combined_data.columns:
        annual["tree_loss_ha"] = combined_data.groupby("Year")["tree_loss_ha"].mean()
    elif "umd_tree_cover_loss__ha" in combined_data.columns:
        annual["tree_loss_ha"] = combined_data.groupby("Year")[
            "umd_tree_cover_loss__ha"
        ].mean()
    else:
        annual["tree_loss_ha"] = 0.0

    # 1) Tree cover loss over time
    axes[0, 0].plot(
        dfx["Year"],
        dfx["tree_loss_ha"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="darkgreen",
    )
    axes[0, 0].set_title(
        "Annual Tree Cover Loss in Dhaka (2001-2023)", fontweight="bold"
    )
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Tree Cover Loss (ha)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2) Deforestation vs Temperature with KDE contours (annual)
    valid = annual[annual["tree_loss_ha"] > 0].dropna()
    axes[0, 1].scatter(
        valid["tree_loss_ha"],
        valid[f"{temp_col}_mean"],
        alpha=0.7,
        s=80,
        color="purple",
        edgecolors="white",
        linewidth=0.5,
    )

    if len(valid) > 5:
        x_data = valid["tree_loss_ha"].values
        y_data = valid[f"{temp_col}_mean"].values
        xy = np.vstack([x_data, y_data])
        kde_2d = gaussian_kde(xy)
        xx, yy = np.mgrid[
            x_data.min() : x_data.max() : 20j, y_data.min() : y_data.max() : 20j
        ]
        kde_vals = np.reshape(kde_2d(np.vstack([xx.ravel(), yy.ravel()]).T).T, xx.shape)
        axes[0, 1].contour(
            xx, yy, kde_vals, levels=3, alpha=0.5, colors="red", linewidths=1.5
        )

    axes[0, 1].set_title(
        "Tree Loss vs Temperature (with KDE contours)", fontweight="bold"
    )
    axes[0, 1].set_xlabel("Tree Cover Loss (ha)")
    axes[0, 1].set_ylabel("Mean Temperature (°C)")
    axes[0, 1].grid(True, alpha=0.3)

    # trend line (annual)
    if len(valid) > 1:
        z = np.polyfit(valid["tree_loss_ha"], valid[f"{temp_col}_mean"], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(
            valid["tree_loss_ha"], p(valid["tree_loss_ha"]), "r--", alpha=0.8
        )

    # 3) Cumulative deforestation
    dfx_sorted = dfx.sort_values("Year")
    cumulative_loss = dfx_sorted["tree_loss_ha"].cumsum()
    axes[1, 0].plot(
        dfx_sorted["Year"], cumulative_loss, marker="s", linewidth=2, color="brown"
    )
    axes[1, 0].set_title("Cumulative Tree Cover Loss", fontweight="bold")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Cumulative Loss (ha)")
    axes[1, 0].grid(True, alpha=0.3)

    # 4) Normalized comparison (annual)
    norm_df = annual.loc[
        annual.index >= 2001, [f"{temp_col}_mean", "tree_loss_ha"]
    ].copy()
    scaler = MinMaxScaler()
    norm_df[["temp_norm", "deforest_norm"]] = scaler.fit_transform(
        norm_df[[f"{temp_col}_mean", "tree_loss_ha"]]
    )
    axes[1, 1].plot(
        norm_df.index,
        norm_df["temp_norm"],
        marker="o",
        linewidth=2,
        label="Temperature (normalized)",
        color="red",
    )
    axes[1, 1].plot(
        norm_df.index,
        norm_df["deforest_norm"],
        marker="s",
        linewidth=2,
        label="Deforestation (normalized)",
        color="green",
    )
    axes[1, 1].set_title(
        "Normalized Temperature vs Deforestation Trends", fontweight="bold"
    )
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Normalized Values (0-1)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(output_dir, "deforestation_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_heatwave_analysis(
    data, heatwave_summary, threshold=36, output_dir="images/data_exploration"
):
    """Plot heatwave analysis"""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Heatwave days per year
    heatwave_days_per_year = data[data["Heatwave"]].groupby("Year").size()
    axes[0, 0].bar(
        heatwave_days_per_year.index, heatwave_days_per_year.values, alpha=0.7
    )
    axes[0, 0].set_title(
        "Heatwave Days per Year in Dhaka (1972-2024)", fontweight="bold"
    )
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Number of Heatwave Days")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Heatwave days by month
    heatwave_days_per_month = data[data["Heatwave"]].groupby("Month").size()
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
    axes[0, 1].bar(
        range(1, 13),
        [heatwave_days_per_month.get(i, 0) for i in range(1, 13)],
        color="orange",
        alpha=0.7,
    )
    axes[0, 1].set_title("Heatwave Days by Month in Dhaka", fontweight="bold")
    axes[0, 1].set_xlabel("Month")
    axes[0, 1].set_ylabel("Number of Heatwave Days")
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].set_xticklabels(month_names)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Rolling average of heatwave days
    rolling_average = heatwave_days_per_year.rolling(window=5).mean()
    axes[1, 0].plot(
        heatwave_days_per_year.index,
        heatwave_days_per_year.values,
        label="Heatwave Days",
        alpha=0.5,
    )
    axes[1, 0].plot(
        rolling_average.index,
        rolling_average.values,
        label="5-Year Rolling Average",
        color="red",
        linewidth=2,
    )
    axes[1, 0].set_title("Heatwave Trends (Rolling Average)", fontweight="bold")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Number of Heatwave Days")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Heatwave duration histogram with KDE
    if len(heatwave_summary) > 0:
        durations = heatwave_summary["Duration"].values

        # Create histogram with density=True for KDE overlay
        n, bins, patches = axes[1, 1].hist(
            durations,
            bins=15,
            alpha=0.6,
            edgecolor="black",
            density=True,
            color="lightcoral",
        )

        # Add KDE curve if we have enough data points
        if len(durations) > 3:
            kde = gaussian_kde(durations)
            x_kde = np.linspace(durations.min(), durations.max(), 100)
            kde_curve = kde(x_kde)
            axes[1, 1].plot(
                x_kde,
                kde_curve,
                "darkred",
                linewidth=3,
                label=f"KDE (n={len(durations)} events)",
                alpha=0.9,
            )
            axes[1, 1].legend()

        axes[1, 1].set_title(
            "Heatwave Duration Distribution with KDE", fontweight="bold"
        )
        axes[1, 1].set_xlabel("Duration (days)")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No heatwave events detected",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
        )
        axes[1, 1].set_title("Heatwave Duration Distribution", fontweight="bold")

    plt.tight_layout()

    # Save the plot with high resolution
    filename = os.path.join(output_dir, "heatwave_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_correlation_matrix(data, output_dir="images/data_exploration"):
    """Plot correlation matrix of climate variables"""
    os.makedirs(output_dir, exist_ok=True)
    selected_columns = [
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",  # canonical first
        "Dhaka Temperature [2 m elevation corrected]",
        "Dhaka Precipitation Total",
        "Dhaka Relative Humidity [2 m]",
        "Dhaka Wind Speed [10 m]",
        "Dhaka Cloud Cover Total",
        "Dhaka Sunshine Duration",
        "Dhaka Mean Sea Level Pressure [MSL]",
    ]

    # Filter columns that exist in the data
    available_columns = [col for col in selected_columns if col in data.columns]
    selected_data = data[available_columns].apply(pd.to_numeric, errors="coerce")

    correlation_matrix = selected_data.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        vmax=1,
        vmin=-1,
    )
    plt.title("Correlation Matrix of Climate Variables", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save the plot with high resolution
    filename = os.path.join(output_dir, "climate_correlation_matrix.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {filename}")
    plt.close()


def create_summary_dashboard(
    data, statistical_results, tree_loss_by_year, output_dir="images/summary"
):
    """Create a summary dashboard with key metrics (canonical-friendly)"""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    temp_col = _pick_temp_col(data)

    # ensure Year/Month columns
    if "Year" not in data.columns or "Month" not in data.columns:
        data = data.copy()
        data["Year"] = data.index.year
        data["Month"] = data.index.month

    # robust heatwave column
    hw_col = (
        "is_heatwave_day"
        if "is_heatwave_day" in data.columns
        else ("Heatwave" if "Heatwave" in data.columns else None)
    )

    total_years = int(data.index.year.nunique())
    avg_temp = float(data[temp_col].mean())
    total_heatwave_days = int(data[hw_col].sum()) if hw_col else 0

    # normalize deforestation input
    if isinstance(tree_loss_by_year, pd.Series):
        defo = tree_loss_by_year.rename("tree_loss_ha")
        defo_df = defo.reset_index().rename(
            columns={defo.index.name or "index": "Year"}
        )
    else:
        defo_df = tree_loss_by_year.copy()
        if "year" in defo_df.columns and "Year" not in defo_df.columns:
            defo_df = defo_df.rename(columns={"year": "Year"})
        if (
            "tree_loss_ha" not in defo_df.columns
            and "umd_tree_cover_loss__ha" in defo_df.columns
        ):
            defo_df = defo_df.rename(
                columns={"umd_tree_cover_loss__ha": "tree_loss_ha"}
            )
    total_deforestation = float(
        defo_df.get("tree_loss_ha", pd.Series(dtype=float)).sum()
    )

    # 1) Temperature trend
    annual_data = data.groupby("Year")[temp_col].mean()
    axes[0, 0].plot(
        annual_data.index,
        annual_data.values,
        linewidth=2,
        color="red",
        marker="o",
        markersize=3,
    )
    z = np.polyfit(annual_data.index, annual_data.values, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(
        annual_data.index, p(annual_data.index), "r--", alpha=0.8, linewidth=2
    )
    axes[0, 0].set_title(
        f"Temperature Trend\n({total_years} years, +{z[0]*52:.2f}°C)", fontweight="bold"
    )
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2) Heatwave frequency
    heatwave_annual = (
        data[data[hw_col]].groupby("Year").size() if hw_col else pd.Series(dtype=int)
    )
    all_years = np.sort(data["Year"].unique())
    heatwave_annual = heatwave_annual.reindex(all_years, fill_value=0)
    axes[0, 1].bar(
        heatwave_annual.index, heatwave_annual.values, alpha=0.7, color="orange"
    )
    z_hw = (
        np.polyfit(heatwave_annual.index, heatwave_annual.values, 2)  # type: ignore
        if len(heatwave_annual) > 2
        else [0, 0, 0]
    )
    p_hw = np.poly1d(z_hw)
    if len(heatwave_annual) > 2:
        axes[0, 1].plot(
            heatwave_annual.index,
            p_hw(heatwave_annual.index),
            "r-",
            alpha=0.8,
            linewidth=3,
        )
    axes[0, 1].set_title(
        f"Total Heatwave Days\n{total_heatwave_days} (trend: accelerating)",
        fontweight="bold",
    )
    axes[0, 1].set_ylabel("Heatwave Days")
    axes[0, 1].grid(True, alpha=0.3)

    # 3) Deforestation
    axes[0, 2].bar(defo_df["Year"], defo_df["tree_loss_ha"], alpha=0.7, color="green")
    if len(defo_df) > 1:
        z_def = np.polyfit(defo_df["Year"], defo_df["tree_loss_ha"], 1)
        p_def = np.poly1d(z_def)
        axes[0, 2].plot(
            defo_df["Year"], p_def(defo_df["Year"]), "r-", alpha=0.8, linewidth=3
        )
        loss_rate = z_def[0]
    else:
        loss_rate = 0.0
    axes[0, 2].set_title(
        f"Deforestation\n{total_deforestation:.0f} hectares lost", fontweight="bold"
    )
    axes[0, 2].set_ylabel("Tree Loss (ha)")
    axes[0, 2].grid(True, alpha=0.3)

    # 4) Volatility
    annual_volatility = data.groupby("Year")[temp_col].std()
    moving_avg_volatility = annual_volatility.rolling(window=5, min_periods=1).mean()
    axes[1, 0].plot(
        annual_volatility.index,
        annual_volatility.values,
        alpha=0.5,
        color="lightblue",
        label="Annual Volatility",
    )
    axes[1, 0].plot(
        moving_avg_volatility.index,
        moving_avg_volatility.values,
        color="darkblue",
        linewidth=2,
        label="5-Year Average",
    )
    z_vol = np.polyfit(
        annual_volatility.index, np.nan_to_num(annual_volatility.values), 1
    )
    p_vol = np.poly1d(z_vol)
    axes[1, 0].plot(
        annual_volatility.index,
        p_vol(annual_volatility.index),
        "r--",
        alpha=0.8,
        linewidth=2,
        label="Trend",
    )
    axes[1, 0].set_title("Temperature Volatility Analysis", fontweight="bold")
    axes[1, 0].set_ylabel("Temperature Std Dev (°C)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5) Monthly warming trends
    monthly_variation = data.groupby(["Year", "Month"])[temp_col].mean().unstack()
    month_trends = {}
    for month in monthly_variation.columns:
        monthly_data = monthly_variation[month].dropna()
        if len(monthly_data) > 1:
            month_trends[month] = (
                np.polyfit(monthly_data.index, monthly_data.values, 1)[0] * 52
            )
    months = list(month_trends.keys())
    trends = [month_trends[m] for m in months]
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
    if months:
        max_trend = max(trends)
        min_trend = min(trends)
        norm_trends = [
            (t - min_trend) / (max_trend - min_trend) if max_trend != min_trend else 0.5
            for t in trends
        ]
        cmap = plt.get_cmap("RdBu_r")
        colors = [cmap(0.8 if nt > 0.5 else 0.2) for nt in norm_trends]
        bars = axes[1, 1].bar(
            [month_names[m - 1] for m in months],
            trends,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar, trend in zip(bars, trends, strict=False):
            h = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                h + (0.01 if h >= 0 else -0.05),
                f"{trend:.2f}°C",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )
        axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=1.2)
        axes[1, 1].set_title(
            "Monthly Warming Trends (52-year)", fontweight="bold", fontsize=12
        )
        axes[1, 1].set_ylabel("Temperature Change (°C)", fontweight="bold")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

    # 6) Stats panel
    axes[1, 2].axis("off")
    start_year = int(data.index.year.min())
    end_year = int(data.index.year.max())
    stats_text = f"""
    ENHANCED CLIMATE ANALYSIS

    Dataset: {start_year} - {end_year}
    Records: {len(data):,} daily observations

    Temperature Trends:
    • Average: {avg_temp:.2f}°C
    • 52-yr Change: +{z[0]*52:.2f}°C
    • Volatility Trend: {'↑' if z_vol[0] > 0 else '↓'} {abs(z_vol[0]*52):.3f}°C

    Extremes:
    • Max: {data[temp_col].max():.1f}°C
    • Min: {data[temp_col].min():.1f}°C
    • Range: {data[temp_col].max() - data[temp_col].min():.1f}°C

    Heatwaves (>36°C):
    • Total: {total_heatwave_days:,} days
    • Rate: {total_heatwave_days/total_years:.1f}/year

    Environmental Impact:
    • Tree Loss: {total_deforestation:.0f} ha (2001-2023)
    • Loss Rate: {loss_rate:.0f} ha/year
    """
    if (
        isinstance(statistical_results, dict)
        and "temperature_trend" in statistical_results
    ):
        temp_trend = statistical_results["temperature_trend"]
        stats_text += f"""

    Statistical Significance:
    • Warming Rate: {temp_trend.get('slope', 0.0):.4f}°C/year
    • Total 52-yr: {temp_trend.get('total_increase_52years', 0.0):.2f}°C
        """

    axes[1, 2].text(
        0.05,
        0.95,
        stats_text,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightgray", "alpha": 0.7},
        transform=axes[1, 2].transAxes,
    )

    plt.tight_layout()
    filename = os.path.join(output_dir, "comprehensive_dashboard.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {filename}")
    plt.close()


# ============================================================================
# TIME SERIES & MACHINE LEARNING VISUALIZATION FUNCTIONS
# ============================================================================


def plot_time_series_results(
    forecasts, historical_data=None, output_dir="images/summary"
):
    """Plot time series forecasting results (ARIMA & SARIMA)"""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- NEW: derive history if not provided ---
    if historical_data is None and "dataframe" in forecasts:
        historical_data = forecasts["dataframe"]
    if historical_data is None:
        # Try to reconstruct a sensible history from model outputs (fallback: empty)
        hist_years = sorted(
            {
                int(y)
                for m in forecasts.values()
                if isinstance(m, dict)
                for y in (m.get("historical", {}) or {})
            }
        )
        hist_vals = [(y, np.nan) for y in hist_years]
        historical = pd.DataFrame(hist_vals, columns=["Year", "Temp"])
    else:
        # annual mean of temp_mean_c if present; else primary column
        col = (
            "temp_mean_c"
            if "temp_mean_c" in historical_data.columns
            else (
                "temp_max_c"
                if "temp_max_c" in historical_data.columns
                else historical_data.columns[0]
            )
        )
        historical = (
            historical_data[[col]]
            .resample("Y")
            .mean()
            .rename(columns={col: "Temp"})
            .assign(Year=lambda d: d.index.year)
            .reset_index(drop=True)
        )

    # Build forecast frame
    frows = []
    for name, payload in forecasts.items():
        if not isinstance(payload, dict):
            continue
        ann = payload.get("annual_forecasts")
        if not ann:
            continue
        for y, v in ann.items():
            frows.append((int(y), float(v), name))
    if not frows:
        print("ERROR: No predictions available for plotting")
        return

    import pandas as pd

    F = pd.DataFrame(frows, columns=["Year", "Temp", "Model"])

    # Plot
    plt.figure(figsize=(10, 5))
    if len(historical) and historical["Temp"].notna().any():
        plt.plot(
            historical["Year"], historical["Temp"], label="Historical", linewidth=2
        )
    for m in sorted(F["Model"].unique()):
        sub = F[F["Model"] == m]
        plt.plot(sub["Year"], sub["Temp"], label=m.upper())
    plt.legend()
    plt.title("Annual Temperature Forecasts")
    plt.xlabel("Year")
    plt.ylabel("°C")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_forecasts.png", dpi=140)
    plt.close()


def plot_ml_results(models_dict, output_dir="images/ml"):
    """Plot machine learning model results with safe R² lookup"""
    from pathlib import Path

    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bars, r2s = [], []
    for name, payload in models_dict.items():
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("validation_metrics", {}) or payload.get("metrics", {})
        # NEW: robust R² lookup
        r2 = (
            metrics.get("test_r2")
            or metrics.get("val_r2")
            or metrics.get("r2")
            or metrics.get("validation_r2")
        )
        if r2 is None:
            # last resort: NaN instead of crashing
            r2 = float("nan")
        bars.append(name.upper())
        r2s.append(r2)

    plt.figure(figsize=(8, 4))
    plt.bar(bars, r2s)
    plt.title("Model R² (validation)")
    plt.ylabel("R²")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ml_r2.png", dpi=140)
    plt.close()


def plot_arima_decomposition_colorful(decomposition, output_dir="images/arima"):
    """Plot colorful ARIMA seasonal decomposition"""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Original data (Blue)
    axes[0].plot(decomposition.observed, color="blue", linewidth=1.5)
    axes[0].set_title("Original Time Series", fontweight="bold", color="blue")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)

    # Trend (Green)
    axes[1].plot(decomposition.trend, color="green", linewidth=2)
    axes[1].set_title("Trend Component", fontweight="bold", color="green")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].grid(True, alpha=0.3)

    # Seasonal (Orange)
    axes[2].plot(decomposition.seasonal, color="orange", linewidth=2)
    axes[2].set_title("Seasonal Component", fontweight="bold", color="orange")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].grid(True, alpha=0.3)

    # Residual (Red)
    axes[3].plot(decomposition.resid, color="red", linewidth=1)
    axes[3].set_title("Residual Component", fontweight="bold", color="red")
    axes[3].set_ylabel("Temperature (°C)")
    axes[3].set_xlabel("Date")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot with high resolution
    filename = os.path.join(output_dir, "arima_seasonal_decomposition.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_sarima_enhanced(forecasts, output_dir="images/sarima"):
    """Plot enhanced SARIMA results with multiple visualizations (robust)."""
    os.makedirs(output_dir, exist_ok=True)
    if "sarima" not in forecasts:
        print("SARIMA results not available for plotting.")
        return

    sarima_data = forecasts["sarima"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1) SARIMA Forecast Plot
    monthly_temp = sarima_data.get("historical_data")
    if monthly_temp is None and "dataframe" in forecasts:
        df_hist = forecasts["dataframe"]
        col = (
            "temp_mean_c"
            if "temp_mean_c" in df_hist.columns
            else (
                "temp_max_c" if "temp_max_c" in df_hist.columns else df_hist.columns[0]
            )
        )
        monthly_temp = df_hist[col].resample("MS").mean()

    if monthly_temp is not None:
        axes[0, 0].plot(
            monthly_temp.index[-120:],
            monthly_temp.values[-120:],
            label="Historical (Last 10 years)",
            color="blue",
            linewidth=2,
        )
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "No historical data available",
            transform=axes[0, 0].transAxes,
            ha="center",
            va="center",
        )

    if all(k in sarima_data for k in ("dates", "forecast", "confidence_interval")):
        axes[0, 0].plot(
            sarima_data["dates"],
            sarima_data["forecast"],
            label="SARIMA Forecast",
            color="green",
            linewidth=2,
        )
        ci = sarima_data["confidence_interval"]
        axes[0, 0].fill_between(
            sarima_data["dates"],
            ci.iloc[:, 0],
            ci.iloc[:, 1],
            color="green",
            alpha=0.2,
            label="95% Confidence Interval",
        )

    axes[0, 0].set_title("SARIMA Temperature Forecast", fontweight="bold")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2) Seasonal Decomposition if available
    if "decomposition" in sarima_data:
        decomp = sarima_data["decomposition"]
        trend_data = decomp.trend.dropna()
        axes[0, 1].plot(
            trend_data.index, trend_data.values, color="purple", linewidth=2
        )
        axes[0, 1].set_title(
            "SARIMA Trend Component", fontweight="bold", color="purple"
        )
        axes[0, 1].set_ylabel("Temperature (°C)")
        axes[0, 1].grid(True, alpha=0.3)

        seasonal_data = decomp.seasonal
        axes[1, 0].plot(
            seasonal_data.index[:365],
            seasonal_data.values[:365],
            color="cyan",
            linewidth=2,
        )
        axes[1, 0].set_title(
            "SARIMA Seasonal Component", fontweight="bold", color="cyan"
        )
        axes[1, 0].set_ylabel("Temperature (°C)")
        axes[1, 0].grid(True, alpha=0.3)

    # 3) Model Summary
    axes[1, 1].axis("off")
    axes[1, 1].set_title("SARIMA Model Summary", fontsize=14, fontweight="bold")
    summary = sarima_data.get("model_summary", {})
    summary_text = "SARIMA MODEL RESULTS\n" + "=" * 25 + "\n\n"
    summary_text += f"Order: {summary.get('order', 'N/A')}\n"
    summary_text += f"Seasonal Order: {summary.get('seasonal_order', 'N/A')}\n"
    summary_text += f"AIC: {summary.get('aic', 'N/A')}\n"
    summary_text += f"BIC: {summary.get('bic', 'N/A')}\n"
    summary_text += f"Log Likelihood: {summary.get('llf', 'N/A')}\n"
    axes[1, 1].text(
        0.05,
        0.95,
        summary_text,
        fontsize=10,
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.8},
        transform=axes[1, 1].transAxes,
    )

    plt.tight_layout()
    filename = os.path.join(output_dir, "sarima_enhanced_results.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {filename}")
    plt.close()
