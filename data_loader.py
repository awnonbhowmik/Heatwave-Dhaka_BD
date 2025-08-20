"""
Data Loading and Preprocessing Module
Updated with proper column definitions and data validation
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_dictionary import (
    HEATWAVE_MIN_CONSECUTIVE_DAYS,
    HEATWAVE_PERCENTILE,
    HEATWAVE_THRESHOLD,
    PRIMARY_TEMP_COLUMN,
    TEMPERATURE_COLUMNS,
    validate_temperature_data,
)

warnings.filterwarnings("ignore")

# Constants for better maintainability
DEFAULT_HEATWAVE_DATA_PATH = "data/1972_2024_Heatwave_Daily.xlsx"
DEFAULT_DEFORESTATION_DATA_PATH = "data/GFW_Dhaka.csv"
SEASONS_PER_YEAR = 4
MONTHS_PER_SEASON = 3


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to handle spaces and special characters.

    Args:
        df: DataFrame with potentially problematic column names

    Returns:
        DataFrame with normalized column names
    """
    # Define raw → canonical mapping
    colmap = {
        "Dhaka Temperature [2 m elevation corrected].2": "temp_mean_c",
        "Dhaka Temperature [2 m elevation corrected].1": "temp_min_c",
        "Dhaka Temperature [2 m elevation corrected]": "temp_max_c",
        "TMIN": "temp_min_c",
        "TMEAN": "temp_mean_c",
        "TMAX": "temp_max_c",
    }
    # Normalize surface: lower, underscores, strip symbols
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
        .str.lower()
    )
    # Build a normalized lookup for the raw keys
    norm_map = {
        k.replace(" ", "_")
        .replace(".", "")
        .replace("[", "")
        .replace("]", "")
        .lower(): v
        for k, v in colmap.items()
    }
    # Apply mapping when present
    df = df.rename(columns={c: norm_map.get(c, c) for c in df.columns})
    return df


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures a DatetimeIndex exists and is sorted.

    Args:
        df: DataFrame that may need datetime index

    Returns:
        DataFrame with proper DatetimeIndex
    """
    # Prefer explicit 'timestamp' when available
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.set_index("timestamp", drop=False)
    # If index already datetime, drop tz info for stability
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.tz_convert(None) if df.index.tz is not None else df
    # Try to infer from first column if it looks like a date
    else:
        first = df.columns[0]
        if "date" in first or "time" in first:
            df[first] = pd.to_datetime(df[first], errors="coerce")
            df = df.set_index(first, drop=False)
            df.index.name = "timestamp"
        else:
            raise ValueError("No DatetimeIndex or timestamp column found.")
    # Sort for resampling correctness
    return df.sort_index()


def _detect_heatwaves_percentile_based(df: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    """Detects heatwaves using rolling day-of-year percentiles.

    Args:
        df: DataFrame with temperature data and DatetimeIndex
        temp_col: Name of temperature column to use

    Returns:
        DataFrame with heatwave indicators added
    """
    # Work on a copy to preserve original structure
    dfw = df.copy()

    # Ensure datetime index is present
    dfw = _ensure_datetime_index(dfw)

    # Choose a temperature column if requested one is missing
    if temp_col not in dfw.columns:
        cand = [c for c in dfw.columns if c.startswith("temp_")]
        if not cand:
            raise ValueError("No temperature column available.")
        temp_col = cand[0]
        print(f"Info: falling back to {temp_col}")

    # Extract temperature values
    s = dfw[temp_col].astype(float).to_numpy()

    # Day-of-year as numpy array (1..366)
    if not isinstance(dfw.index, pd.DatetimeIndex):
        dfw.index = pd.to_datetime(dfw.index)
    doy = dfw.index.dayofyear.to_numpy()

    # Prepare threshold buffer for 1..366 (index with day directly)
    thr = np.full(367, np.nan, dtype=float)

    # Build ±7-day neighborhood percentile per day-of-year
    for d in range(1, 367):
        # Distance to target day, wrapped at 365/366
        delta = np.abs(doy - d)
        wrap = np.minimum(delta, np.abs(365 - delta))
        mask = wrap <= 7
        if mask.any():
            thr[d] = np.nanpercentile(s[mask], HEATWAVE_PERCENTILE)

    # Map each day to its threshold
    daily_thr = thr[doy]

    # Flag "hot" days where both value and threshold exist
    valid = np.isfinite(s) & np.isfinite(daily_thr)
    is_hot = np.zeros_like(valid, dtype=bool)
    is_hot[valid] = s[valid] >= daily_thr[valid]

    # Require ≥ HEATWAVE_MIN_CONSECUTIVE_DAYS consecutive hot days
    consec = np.zeros_like(is_hot, dtype=bool)
    run = 0
    for i, flag in enumerate(is_hot):
        run = run + 1 if flag else 0
        if run >= HEATWAVE_MIN_CONSECUTIVE_DAYS:
            start = i - HEATWAVE_MIN_CONSECUTIVE_DAYS + 1
            consec[start : i + 1] = True

    # Write back to original frame shape/index
    out = df.copy()
    out["is_heatwave_day"] = consec
    out["heatwave_threshold"] = daily_thr
    out["Heatwave"] = consec  # legacy compatibility

    # Log summary
    total = int(consec.sum())
    ratio = float(consec.mean() * 100.0)
    print("Heatwave detection:")
    print(f"  • Percentile: {HEATWAVE_PERCENTILE}")
    print(f"  • Min run: {HEATWAVE_MIN_CONSECUTIVE_DAYS} days")
    print(f"  • Heatwave days: {total} ({ratio:.2f}%)")

    return out


def _handle_validation_results(
    validation_results: dict[str, Any], data_type: str = "data"
) -> None:
    """Handle validation results with appropriate error/warning reporting.

    Args:
        validation_results: Dictionary containing errors, warnings, and stats
        data_type: Type of data being validated (for error messages)

    Raises:
        ValueError: If critical validation errors are found
    """
    if validation_results["errors"]:
        print(f"CRITICAL {data_type.upper()} ERRORS:")
        for error in validation_results["errors"]:
            print(f"   • {error}")
        raise ValueError(f"Critical {data_type} validation errors found")

    if validation_results["warnings"]:
        print(f"WARNING - {data_type.title()} validation warnings:")
        for warning in validation_results["warnings"]:
            print(f"   • {warning}")


def _add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to the dataset.

    Args:
        data: DataFrame with timestamp column

    Returns:
        DataFrame with added time features
    """
    data["Year"] = data["timestamp"].dt.year
    data["Month"] = data["timestamp"].dt.month
    data["DayOfYear"] = data["timestamp"].dt.dayofyear
    data["Season"] = data["timestamp"].dt.month % 12 // MONTHS_PER_SEASON + 1
    return data


def _add_temperature_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add temperature-based features to the dataset.

    Args:
        data: DataFrame with temperature columns

    Returns:
        DataFrame with added temperature features
    """
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]
    temp_min_col = TEMPERATURE_COLUMNS["daily_min"]
    temp_max_col = TEMPERATURE_COLUMNS["daily_max"]

    # Add heatwave indicator
    data["Heatwave"] = data[temp_col] > HEATWAVE_THRESHOLD

    # Calculate temperature range
    data["Temperature_Range"] = data[temp_max_col] - data[temp_min_col]

    return data


def load_heatwave_data(
    file_path: str | Path = DEFAULT_HEATWAVE_DATA_PATH,
) -> tuple[pd.DataFrame, str]:
    """Load and preprocess heatwave data with validation.

    Args:
        file_path: Path to the heatwave data file

    Returns:
        Tuple of (processed DataFrame, primary temperature column name)

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If critical data validation errors are found
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Heatwave data file not found: {file_path}")

    print("Loading heatwave data...")
    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load heatwave data from {file_path}: {e}") from e

    # Apply forensic fixes: normalize columns + enforce DatetimeIndex
    data = _normalize_column_names(data)
    data = _ensure_datetime_index(data)

    # Map PRIMARY_TEMP_COLUMN → canonical name
    # daily_max → temp_max_c ; daily_mean → temp_mean_c
    primary_map = {
        "daily_max": "temp_max_c",
        "daily_mean": "temp_mean_c",
        "daily_min": "temp_min_c",
    }
    primary_col = primary_map.get(PRIMARY_TEMP_COLUMN, "temp_max_c")

    # Fail fast if missing the chosen canonical column
    if primary_col not in data.columns:
        raise ValueError(f"Missing primary column: {primary_col}")

    print(f"Primary temperature column: {primary_col}")

    # Validate using a shim that recreates raw column aliases
    print("Validating temperature data...")

    # Build a copy for validation that includes the raw names
    df_for_validation = data.copy()
    raw_to_canon = {
        "Dhaka Temperature [2 m elevation corrected]": "temp_max_c",
        "Dhaka Temperature [2 m elevation corrected].1": "temp_min_c",
        "Dhaka Temperature [2 m elevation corrected].2": "temp_mean_c",
    }
    for raw, canon in raw_to_canon.items():
        if raw not in df_for_validation.columns and canon in df_for_validation.columns:
            # Create an alias column with the raw name pointing to the canonical series
            df_for_validation[raw] = df_for_validation[canon]

    validation_results = validate_temperature_data(df_for_validation)
    _handle_validation_results(validation_results, "temperature data")

    # Print validation statistics
    stats = validation_results["stats"]
    print("Data validation completed:")
    print(f"   - Records: {stats['records_validated']:,}")
    print(
        f"   - Mean temperature: {stats['mean_temperature_avg']:.2f}°C ± {stats['mean_temperature_std']:.2f}"
    )
    print(
        f"   - Daily temperature range: {stats['daily_range_avg']:.2f}°C ± {stats['daily_range_std']:.2f}"
    )

    # Add derived features
    data = _add_time_features(data)
    data = _add_temperature_features(data)

    # Apply robust percentile-based heatwave detection
    data = _detect_heatwaves_percentile_based(data, primary_col)

    print(
        f"Loaded {len(data):,} heatwave records ({data.index.min().year}-{data.index.max().year})"
    )
    print(f"   - Using primary column: {primary_col}")
    print(f"   - Percentile threshold: {HEATWAVE_PERCENTILE}th percentile")
    print(
        f"   - Heatwave days found: {data['Heatwave'].sum():,} ({data['Heatwave'].mean() * 100:.1f}% of days)"
    )

    return data, primary_col


def load_deforestation_data(
    file_path: str | Path = DEFAULT_DEFORESTATION_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Loads and validates deforestation, returns annual series + stats.

    Args:
        file_path: Path to deforestation data file

    Returns:
        Tuple of (raw DataFrame, annual series)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Deforestation data file not found: {file_path}")

    print("Loading deforestation data...")

    # Read the source table
    df = pd.read_csv(file_path)

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower()

    # Map raw column names to expected names
    column_mapping = {
        "tree_cover_loss_year": "year",
        "umd_tree_cover_loss__ha": "tree_loss_ha",
    }
    df = df.rename(columns=column_mapping)

    # Coerce year and tree loss
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["tree_loss_ha"] = pd.to_numeric(df["tree_loss_ha"], errors="coerce")

    # Drop rows without usable year
    df = df.dropna(subset=["year"]).copy()

    # Aggregate to annual totals if needed
    annual = df.groupby("year", dropna=True)["tree_loss_ha"].sum(min_count=1)

    # Compute completeness over the climate period
    # (assumes main climate data already loaded globally)
    # Guard if climate span is unknown
    try:
        yr_min = int(globals().get("GLOBAL_CLIMATE_START_YEAR", 1972))
        yr_max = int(globals().get("GLOBAL_CLIMATE_END_YEAR", 2024))
        expected = np.arange(yr_min, yr_max + 1)
        present = annual.index.to_numpy(dtype=int)
        # Intersection over expected years
        comp = 100.0 * (np.intersect1d(present, expected).size / expected.size)
        print(f"Deforestation completeness: {comp:.1f}%")
    except Exception:
        # If unknown, report basic presence
        print(f"Deforestation years present: {len(annual.index)}")

    # Do not fill missing years with zeros
    print(f"Loaded {len(annual)} years of deforestation data")
    print(f"   - Period: {annual.index.min()}-{annual.index.max()}")
    print(f"   - Total tree loss: {annual.sum():.0f} ha")

    # Return both the raw frame and the annual series
    return df, annual


def combine_datasets(
    df_daily: pd.DataFrame, defo_annual: pd.Series, primary_temp_col: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine climate daily and deforestation annual by aligning on year.

    Args:
        df_daily: Climate data DataFrame with normalized column names
        defo_annual: Deforestation annual series (no DataFrame wrapper)
        primary_temp_col: Primary temperature column name (canonical)

    Returns:
        Tuple of (combined daily DataFrame, annual temperature stats)
    """
    print("Combining datasets...")

    # Ensure datetime index on daily data
    df_daily = _ensure_datetime_index(df_daily)

    # Ensure index is DatetimeIndex before extracting year
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        df_daily.index = pd.to_datetime(df_daily.index)
    daily_year = df_daily.index.year
    # Map annual tree loss to daily rows by year
    year_to_loss = defo_annual.to_dict()
    # Build aligned column with NaN for missing years
    df_daily["tree_loss_ha"] = [year_to_loss.get(int(y), np.nan) for y in daily_year]

    # Build annual temperature stats for ALL temp_* columns present
    temp_cols = [c for c in df_daily.columns if c.startswith("temp_")]

    agg_map = {c: ["mean", "max", "min"] for c in temp_cols}
    agg_map["is_heatwave_day"] = ["sum"]

    annual_temp_stats = df_daily.resample("Y").agg(agg_map)  # type: ignore

    # Flatten MultiIndex columns
    flat_cols = []
    for top, sub in annual_temp_stats.columns:
        if top == "is_heatwave_day" and sub == "sum":
            flat_cols.append("heatwave_days")
        else:
            flat_cols.append(f"{top}_{sub}")
    annual_temp_stats.columns = flat_cols

    # Add Year column for compatibility
    annual_temp_stats = annual_temp_stats.reset_index()
    annual_temp_stats["Year"] = annual_temp_stats["timestamp"].dt.year
    annual_temp_stats = annual_temp_stats.drop("timestamp", axis=1)

    print("Datasets combined successfully!")
    print(f"   - Combined daily data shape: {df_daily.shape}")
    print(f"   - Annual stats shape: {annual_temp_stats.shape}")
    print(
        f"   - Years with deforestation data: {df_daily['tree_loss_ha'].notna().sum()}"
    )
    print(
        f"   - Years without deforestation data: {df_daily['tree_loss_ha'].isna().sum()}"
    )
    if primary_temp_col:
        print(f"   - Using primary column: {primary_temp_col}")

    # Return combined daily and annual stats
    return df_daily, annual_temp_stats


def get_dataset_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Get comprehensive dataset summary statistics using proper columns.

    Args:
        data: DataFrame to summarize

    Returns:
        Dictionary containing comprehensive dataset statistics

    Raises:
        ValueError: If required temperature column is not found
    """
    temp_col = TEMPERATURE_COLUMNS["daily_mean"]

    # Validate that required columns exist
    if temp_col not in data.columns:
        raise ValueError(f"Temperature column {temp_col} not found in data")

    summary = {
        "shape": data.shape,
        "time_range": (data["timestamp"].min(), data["timestamp"].max()),
        "total_days": len(data),
        "total_years": data["timestamp"].dt.year.nunique(),
        "missing_values": data.isnull().sum().sum(),
        "heatwave_days": data["Heatwave"].sum() if "Heatwave" in data.columns else 0,
        "heatwave_percentage": (
            data["Heatwave"].mean() * 100 if "Heatwave" in data.columns else 0
        ),
        "temperature_stats": {
            "mean": data[temp_col].mean(),
            "median": data[temp_col].median(),
            "std": data[temp_col].std(),
            "min": data[temp_col].min(),
            "max": data[temp_col].max(),
            "p95": data[temp_col].quantile(0.95),
            "p99": data[temp_col].quantile(0.99),
        },
        "temperature_column_used": temp_col,
        "heatwave_threshold": HEATWAVE_THRESHOLD,
    }
    return summary


def print_data_summary(summary: dict[str, Any]) -> None:
    """Print formatted dataset summary.

    Args:
        summary: Dictionary containing dataset summary statistics
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(
        f"Time Range: {summary['time_range'][0].strftime('%Y-%m-%d')} to {summary['time_range'][1].strftime('%Y-%m-%d')}"
    )
    print(f"Total Days: {summary['total_days']:,}")
    print(f"Total Years: {summary['total_years']}")
    print(f"Missing Values: {summary['missing_values']:,}")
    print(f"\nTEMPERATURE STATISTICS ({summary['temperature_column_used']}):")
    temp_stats = summary["temperature_stats"]
    print(f"   - Mean: {temp_stats['mean']:.2f}°C")
    print(f"   - Median: {temp_stats['median']:.2f}°C")
    print(f"   - Std Dev: {temp_stats['std']:.2f}°C")
    print(f"   - Range: {temp_stats['min']:.2f}°C to {temp_stats['max']:.2f}°C")
    print(f"   - 95th percentile: {temp_stats['p95']:.2f}°C")
    print(f"   - 99th percentile: {temp_stats['p99']:.2f}°C")
    print("\nHEATWAVE STATISTICS:")
    print(f"   - Threshold: {summary['heatwave_threshold']}°C")
    print(f"   - Heatwave Days: {summary['heatwave_days']:,}")
    print(f"   - Heatwave Percentage: {summary['heatwave_percentage']:.2f}%")
    print("=" * 60)
