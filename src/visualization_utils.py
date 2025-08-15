"""
Visualization Utilities Module

Provides additional utility functions and advanced visualizations
for the climate analysis project.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Visualization constants
DEFAULT_FIGSIZE = (15, 10)
DPI = 300
FONT_SIZES = {"title": 16, "subtitle": 14, "label": 12, "tick": 10, "legend": 11}
TEMP_COLUMN = "Dhaka Temperature [2 m elevation corrected]"


def plot_model_predictions(
    predictions: Dict[str, Any],
    actual_data: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    confidence_intervals: bool = True,
) -> Figure:
    """
    Create a multi-panel visualization of model predictions.

    Includes:
    - Model performance comparison.
    - Future climate projections.
    - Residuals analysis.
    - Feature importance.
    """
    logger.info("Creating model predictions visualization")

    try:
        fig, axes = plt.subplots(2, 2, figsize=DEFAULT_FIGSIZE, dpi=DPI)
        fig.suptitle(
            "Climate Model Predictions Analysis",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )

        # Model performance comparison
        ax1 = axes[0, 0]
        if "model_comparison" in predictions:
            models = list(predictions["model_comparison"].keys())
            metrics = ["rmse", "r2", "mae"]
            x = np.arange(len(models))
            width = 0.25

            for i, metric in enumerate(metrics):
                values = [
                    predictions["model_comparison"][model].get(metric, 0)
                    for model in models
                ]
                ax1.bar(x + i * width, values, width, label=metric.upper(), alpha=0.8)

            ax1.set_xlabel("Models")
            ax1.set_ylabel("Metric Value")
            ax1.set_title("Model Performance Comparison")
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Future predictions
        ax2 = axes[0, 1]
        if "future_predictions" in predictions:
            future_data = predictions["future_predictions"]
            if isinstance(future_data, dict) and "years" in future_data:
                years = future_data["years"]
                predictions_vals = future_data.get("predictions", [])

                ax2.plot(
                    years,
                    predictions_vals,
                    "o-",
                    linewidth=2,
                    color="red",
                    label="Predictions",
                )

                if confidence_intervals and "confidence_lower" in future_data:
                    lower = future_data["confidence_lower"]
                    upper = future_data["confidence_upper"]
                    ax2.fill_between(
                        years,
                        lower,
                        upper,
                        alpha=0.3,
                        color="red",
                        label="95% Confidence Interval",
                    )

                ax2.set_xlabel("Year")
                ax2.set_ylabel("Predicted Temperature (°C)")
                ax2.set_title("Future Climate Projections")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # Residuals analysis
        ax3 = axes[1, 0]
        if "residuals" in predictions:
            residuals = predictions["residuals"]
            ax3.scatter(range(len(residuals)), residuals, alpha=0.6, color="blue")
            ax3.axhline(y=0, color="red", linestyle="--", alpha=0.8)
            ax3.set_xlabel("Observation Index")
            ax3.set_ylabel("Residuals")
            ax3.set_title("Model Residuals Analysis")
            ax3.grid(True, alpha=0.3)

        # Feature importance
        ax4 = axes[1, 1]
        if "feature_importance" in predictions:
            features = list(predictions["feature_importance"].keys())
            importance = list(predictions["feature_importance"].values())

            sorted_pairs = sorted(
                zip(features, importance), key=lambda x: x[1], reverse=True
            )
            features, importance = zip(*sorted_pairs[:10])

            ax4.barh(features, importance, color="green", alpha=0.8)
            ax4.set_xlabel("Importance Score")
            ax4.set_title("Top 10 Feature Importance")
            ax4.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
            logger.info(f"Model predictions plot saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating model predictions plot: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Error creating plot: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig


def create_interactive_timeline(
    data: pd.DataFrame,
    events: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a timeline visualization of climate events and trends.

    Plots temperature values over time with optional annotations for
    significant events and a moving average overlay.
    """
    logger.info("Creating interactive climate timeline")

    try:
        fig, ax = plt.subplots(figsize=(18, 10), dpi=DPI)

        if "timestamp" not in data.columns or TEMP_COLUMN not in data.columns:
            ax.text(
                0.5,
                0.5,
                "Required columns not found for timeline",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        dates = data["timestamp"]
        temps = data[TEMP_COLUMN]

        temp_norm = (temps - temps.min()) / (temps.max() - temps.min())
        colors = plt.get_cmap('coolwarm')(temp_norm)

        scatter = ax.scatter(dates, temps, c=colors, s=1, alpha=0.6)

        if len(data) > 365:
            rolling_temp = temps.rolling(window=365, center=True).mean()
            ax.plot(
                dates,
                rolling_temp,
                color="red",
                linewidth=2,
                alpha=0.8,
                label="1-Year Moving Average",
            )

        if "Heatwave" in data.columns:
            threshold = 36
            ax.axhline(
                y=threshold,
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Heatwave Threshold ({threshold}°C)",
            )

        if events:
            for event in events:
                if "date" in event and "description" in event:
                    event_date = pd.to_datetime(event["date"])
                    if dates.min() <= event_date <= dates.max():
                        closest_temp = temps[dates.dt.date == event_date.date()]
                        if not closest_temp.empty:
                            temp_val = closest_temp.iloc[0]
                            ax.annotate(
                                event["description"],
                                xy=(event_date, temp_val),
                                xytext=(event_date, temp_val + 5),
                                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                                fontsize=FONT_SIZES["tick"],
                                ha="center",
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    facecolor="yellow",
                                    alpha=0.8,
                                ),
                            )

        ax.set_title(
            "Climate Timeline: Temperature Evolution Over Time",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("Temperature (°C)", fontsize=FONT_SIZES["label"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Temperature (°C)", fontsize=FONT_SIZES["label"])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
            logger.info(f"Interactive timeline saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating interactive timeline: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Error creating timeline: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig


def set_publication_style():
    """Configure matplotlib parameters for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["tick"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "figure.titlesize": FONT_SIZES["title"],
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI,
        }
    )
    logger.info("Publication style set for visualizations")


def save_all_plots(
    data: pd.DataFrame,
    statistical_results: Dict[str, Any],
    tree_loss_by_year: pd.DataFrame,
    output_dir: str = "plots",
) -> Dict[str, str]:
    """
    Generate and save all climate analysis plots.

    Returns a dictionary mapping plot names to their file paths.
    """
    logger.info(f"Generating all plots and saving to {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved_plots = {}

    try:
        from . import visualization as viz

        set_publication_style()

        annual_temp_stats = (
            data.groupby("Year")[TEMP_COLUMN]
            .agg(["mean", "max", "min", "std"])
            .reset_index()
        )
        annual_temp_stats.columns = ["Year"] + [
            f"{TEMP_COLUMN}_{col}" for col in ["mean", "max", "min", "std"]
        ]

        fig1 = viz.plot_temperature_trends(
            data, annual_temp_stats, save_path=f"{output_dir}/temperature_trends.png"
        )
        saved_plots["temperature_trends"] = f"{output_dir}/temperature_trends.png"
        plt.close(fig1)

        combined_data = pd.merge(
            annual_temp_stats, tree_loss_by_year, on="Year", how="left"
        )
        fig2 = viz.plot_deforestation_analysis(
            tree_loss_by_year,
            combined_data,
            save_path=f"{output_dir}/deforestation_analysis.png",
        )
        saved_plots["deforestation_analysis"] = (
            f"{output_dir}/deforestation_analysis.png"
        )
        plt.close(fig2)

        heatwave_summary = pd.DataFrame()
        fig3 = viz.plot_heatwave_analysis(
            data, heatwave_summary, save_path=f"{output_dir}/heatwave_analysis.png"
        )
        saved_plots["heatwave_analysis"] = f"{output_dir}/heatwave_analysis.png"
        plt.close(fig3)

        fig4 = viz.plot_correlation_matrix(
            data, save_path=f"{output_dir}/correlation_matrix.png"
        )
        saved_plots["correlation_matrix"] = f"{output_dir}/correlation_matrix.png"
        plt.close(fig4)

        fig5 = viz.create_summary_dashboard(
            data,
            statistical_results,
            tree_loss_by_year,
            save_path=f"{output_dir}/summary_dashboard.png",
        )
        saved_plots["summary_dashboard"] = f"{output_dir}/summary_dashboard.png"
        plt.close(fig5)

        fig6 = create_interactive_timeline(
            data, save_path=f"{output_dir}/climate_timeline.png"
        )
        saved_plots["climate_timeline"] = f"{output_dir}/climate_timeline.png"
        plt.close(fig6)

        logger.info(f"Successfully saved {len(saved_plots)} plots to {output_dir}")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")

    return saved_plots


def create_plot_summary_report(saved_plots: Dict[str, str]) -> str:
    """
    Create a markdown summary report of all generated plots.
    """
    report = "# Climate Visualization Report\n\n"
    report += f"Generated {len(saved_plots)} visualization plots:\n\n"

    plot_descriptions = {
        "temperature_trends": "Daily and annual temperature trends with statistical analysis",
        "deforestation_analysis": "Deforestation impact and correlation with climate",
        "heatwave_analysis": "Heatwave frequency, duration, and seasonal patterns",
        "correlation_matrix": "Climate variables correlation analysis",
        "summary_dashboard": "Comprehensive climate analysis dashboard",
        "climate_timeline": "Interactive timeline of climate evolution",
    }

    for plot_name, file_path in saved_plots.items():
        description = plot_descriptions.get(plot_name, "Climate visualization")
        report += f"- **{plot_name.replace('_', ' ').title()}**: {description}\n"
        report += f"  - File: `{file_path}`\n\n"

    return report
