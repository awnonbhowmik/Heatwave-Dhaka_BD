"""Quality-control summaries and data dictionary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_io import COLUMN_MAP, PHYSICAL_RANGES, TARGET_DERIVED, UNITS


def completeness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in df.groupby("year"):
        expected = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
        hot = group[group.month.isin([3, 4, 5, 6])]
        rows.append({
            "year": year, "observed_days": group.date.nunique(), "expected_days": expected,
            "calendar_complete": group.date.nunique() == expected,
            "hot_season_days": hot.date.nunique(), "hot_season_complete": hot.date.nunique() == 122,
            "missing_values": int(group.isna().sum().sum()),
        })
    return pd.DataFrame(rows)


def data_dictionary(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for original, name in COLUMN_MAP.items():
        s = clean[name]
        numeric = pd.api.types.is_numeric_dtype(s)
        lo, hi = PHYSICAL_RANGES.get(name, (np.nan, np.nan))
        target = name in TARGET_DERIVED
        retained = name != "date"
        rows.append({
            "original_column_name": original, "cleaned_analysis_name": name,
            "inferred_data_type": str(s.dtype), "unit": "date" if name == "date" else UNITS.get(name, "unknown"),
            "temporal_aggregation": "daily", "expected_physical_min": lo,
            "expected_physical_max": hi, "missing_count": int(s.isna().sum()),
            "missing_percentage": float(100 * s.isna().mean()),
            "minimum": float(s.min()) if numeric else str(s.min()),
            "maximum": float(s.max()) if numeric else str(s.max()),
            "retained": retained, "target_derived": target,
            "intended_analytical_role": "outcome/sensitivity only" if target else ("time index" if name == "date" else "candidate predictor"),
            "reason_for_exclusion": "not a numeric analytic variable" if not retained else ("excluded from primary association model to prevent leakage" if target else ""),
        })
    return pd.DataFrame(rows)


def quality_findings(df: pd.DataFrame) -> dict:
    full_index = pd.date_range(df.date.min(), df.date.max(), freq="D")
    suspicious = {}
    for name, (lo, hi) in PHYSICAL_RANGES.items():
        suspicious[name] = int(((df[name] < lo) | (df[name] > hi)).sum())
    jumps = {}
    for name in ["tmax", "tmin", "tmean", "pressure_mean", "soil_moisture_mean"]:
        threshold = 6 * df[name].diff().mad() if hasattr(df[name].diff(), "mad") else 6 * np.nanmedian(np.abs(df[name].diff() - np.nanmedian(df[name].diff())))
        jumps[name] = int((df[name].diff().abs() > threshold).sum()) if threshold > 0 else 0
    return {
        "start": str(df.date.min().date()), "end": str(df.date.max().date()),
        "rows": len(df), "duplicate_dates": int(df.date.duplicated().sum()),
        "missing_dates": len(full_index.difference(df.date)), "total_missing_values": int(df.isna().sum().sum()),
        "feb29_rows": int(((df.month == 2) & (df.date.dt.day == 29)).sum()),
        "incomplete_years": completeness_table(df).loc[lambda x: ~x.calendar_complete, "year"].tolist(),
        "hot_season_incomplete_years": completeness_table(df).loc[lambda x: ~x.hot_season_complete, "year"].tolist(),
        "physical_range_flags": suspicious, "abrupt_change_flags": jumps,
    }
