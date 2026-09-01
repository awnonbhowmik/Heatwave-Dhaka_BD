"""Raw-data loading, column mapping, and immutable-source hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


COLUMN_MAP = {
    "timestamp": "date",
    "Dhaka Temperature [2 m elevation corrected]": "tmax",
    "Dhaka Temperature [2 m elevation corrected].1": "tmin",
    "Dhaka Temperature [2 m elevation corrected].2": "tmean",
    "Dhaka Precipitation Total": "precipitation",
    "Dhaka Relative Humidity [2 m]": "rh_max",
    "Dhaka Relative Humidity [2 m].1": "rh_min",
    "Dhaka Relative Humidity [2 m].2": "rh_mean",
    "Dhaka Wind Gust": "wind_gust_max",
    "Dhaka Wind Gust.1": "wind_gust_min",
    "Dhaka Wind Gust.2": "wind_gust_mean",
    "Dhaka Wind Speed [10 m]": "wind_speed_max",
    "Dhaka Wind Speed [10 m].1": "wind_speed_mean",
    "Dhaka Cloud Cover Total": "cloud_cover",
    "Dhaka Sunshine Duration": "sunshine_duration",
    "Dhaka Shortwave Radiation": "shortwave_radiation",
    "Dhaka Longwave Radiation": "longwave_radiation",
    "Dhaka UV Radiation": "uv_radiation",
    "Dhaka Direct Shortwave Radiation": "direct_shortwave_radiation",
    "Dhaka Mean Sea Level Pressure [MSL]": "pressure_max",
    "Dhaka Mean Sea Level Pressure [MSL].1": "pressure_min",
    "Dhaka Mean Sea Level Pressure [MSL].2": "pressure_mean",
    "Dhaka Evapotranspiration": "evapotranspiration",
    "Dhaka Vapor Pressure Deficit [2 m]": "vpd_max",
    "Dhaka Vapor Pressure Deficit [2 m].1": "vpd_min",
    "Dhaka Vapor Pressure Deficit [2 m].2": "vpd_mean",
    "Dhaka Soil Temperature [0-7 cm down]": "soil_temperature_max",
    "Dhaka Soil Temperature [0-7 cm down].1": "soil_temperature_min",
    "Dhaka Soil Temperature [0-7 cm down].2": "soil_temperature_mean",
    "Dhaka Soil Moisture [0-7 cm down]": "soil_moisture_max",
    "Dhaka Soil Moisture [0-7 cm down].1": "soil_moisture_min",
    "Dhaka Soil Moisture [0-7 cm down].2": "soil_moisture_mean",
}

UNITS = {
    "tmax": "degC", "tmin": "degC", "tmean": "degC",
    "precipitation": "mm/day", "rh_max": "%", "rh_min": "%", "rh_mean": "%",
    "wind_gust_max": "km/h", "wind_gust_min": "km/h", "wind_gust_mean": "km/h",
    "wind_speed_max": "km/h", "wind_speed_mean": "km/h", "cloud_cover": "%",
    "sunshine_duration": "min/day", "shortwave_radiation": "kJ/m2/day",
    "longwave_radiation": "kJ/m2/day", "uv_radiation": "kJ/m2/day",
    "direct_shortwave_radiation": "kJ/m2/day", "pressure_max": "hPa",
    "pressure_min": "hPa", "pressure_mean": "hPa", "evapotranspiration": "mm/day",
    "vpd_max": "hPa", "vpd_min": "hPa", "vpd_mean": "hPa",
    "soil_temperature_max": "degC", "soil_temperature_min": "degC",
    "soil_temperature_mean": "degC", "soil_moisture_max": "m3/m3",
    "soil_moisture_min": "m3/m3", "soil_moisture_mean": "m3/m3",
}

PHYSICAL_RANGES = {
    "tmax": (-10, 55), "tmin": (-10, 45), "tmean": (-10, 50),
    "precipitation": (0, 1000), "rh_max": (0, 100), "rh_min": (0, 100),
    "rh_mean": (0, 100), "cloud_cover": (0, 100),
    "pressure_max": (850, 1100), "pressure_min": (850, 1100),
    "pressure_mean": (850, 1100), "soil_moisture_max": (0, 1),
    "soil_moisture_min": (0, 1), "soil_moisture_mean": (0, 1),
}

TARGET_DERIVED = {"tmax", "tmin", "tmean", "vpd_max", "vpd_min", "vpd_mean"}


def load_daily(path: str | Path) -> pd.DataFrame:
    """Load raw CSV, validate exact schema, parse dates, and sort chronologically."""
    raw = pd.read_csv(path)
    missing = set(COLUMN_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected raw columns: {sorted(missing)}")
    out = raw.rename(columns=COLUMN_MAP).copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    out["year"] = out.date.dt.year
    out["month"] = out.date.dt.month
    out["day_of_year"] = out.date.dt.dayofyear
    out["day_night_range"] = out.tmax - out.tmin
    return out


def source_hashes(data_dir: str | Path) -> dict[str, str]:
    """SHA-256 all source files under data/, including shapefile components."""
    root = Path(data_dir)
    result: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result


def complete_years(df: pd.DataFrame, end_year: int = 2023) -> pd.DataFrame:
    """Return complete calendar years through the prespecified endpoint."""
    counts = df.groupby("year").date.nunique()
    expected = pd.Series({y: 366 if pd.Timestamp(y, 12, 31).is_leap_year else 365 for y in counts.index})
    years = counts.index[(counts == expected) & (counts.index <= end_year)]
    return df[df.year.isin(years)].copy()


def hot_season(df: pd.DataFrame, months: tuple[int, ...] = (3, 4, 5, 6)) -> pd.DataFrame:
    return df[df.month.isin(months)].copy()
