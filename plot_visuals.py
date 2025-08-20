# visualization.py
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DPI = 140


def _save_plot(filename: str, output_dir: str | Path, dpi: int = DEFAULT_DPI):
    """Save plot with consistent settings."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {outdir / filename}")
    plt.close()


def _monthly_flat(df: pd.DataFrame, primary_col: str) -> pd.DataFrame:
    """Return monthly mean/max/min with flat canonical names like temp_max_c_mean."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex for monthly aggregation")
    m = df[[primary_col]].resample("MS").agg(["mean", "max", "min"])
    m.columns = [f"{primary_col}_{c}" for c in m.columns]
    return m


def plot_monthly_trends(
    df_daily: pd.DataFrame, primary_col: str, output_dir: str | Path
):
    """Plot monthly trends for a canonical column."""
    import pandas as pd, numpy as np, matplotlib.pyplot as plt
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # NEW: compute monthly agg on-the-fly
    m = df_daily[[primary_col]].resample("MS").agg(["mean","max","min"])
    m.columns = [f"{primary_col}_{c}" for c in m.columns]
    need = [f"{primary_col}_mean", f"{primary_col}_max", f"{primary_col}_min"]
    if not all(c in m.columns for c in need):
        print(f"INFO: Missing {need}; skipping monthly trend plot.")
        return

    plt.figure(figsize=(10,5))
    plt.plot(m.index, m[f"{primary_col}_mean"], label="Monthly mean")
    plt.plot(m.index, m[f"{primary_col}_max"],  label="Monthly max", alpha=0.6)
    plt.plot(m.index, m[f"{primary_col}_min"],  label="Monthly min", alpha=0.6)
    plt.legend(); plt.title("Monthly Temperature (mean / max / min)")
    plt.tight_layout()
    _save_plot("monthly_trends.png", output_dir)


def plot_annual_heatwaves(annual_stats: pd.DataFrame, output_dir: str | Path):
    """Plot annual heatwave days."""
    # Pick year vector robustly
    if "Year" in annual_stats.columns:
        years = annual_stats["Year"].to_numpy()
    else:
        # Assume DatetimeIndex
        years = pd.DatetimeIndex(annual_stats.index).year.to_numpy()  # type: ignore

    if "heatwave_days" not in annual_stats.columns:
        print("INFO: No 'heatwave_days' in annual stats; skipping plot.")
        return

    vals = annual_stats["heatwave_days"].to_numpy()
    plt.figure(figsize=(10, 4))
    plt.bar(years, vals)
    plt.title("Annual heatwave days")
    plt.xlabel("Year")
    plt.ylabel("Days")
    _save_plot("annual_heatwaves.png", output_dir)
