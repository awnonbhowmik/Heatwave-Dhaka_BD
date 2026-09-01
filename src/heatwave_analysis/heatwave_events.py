"""Heatwave exceedance definitions, persistence grouping, and event metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .climatology import calendar_day_threshold


DEFINITIONS = {
    "operational_36c_1d": {"minimum_duration": 1, "kind": "fixed"},
    "persistent_36c_2d": {"minimum_duration": 2, "kind": "fixed"},
    "persistent_36c_3d": {"minimum_duration": 3, "kind": "fixed"},
    "relative_90p_3d": {"minimum_duration": 3, "kind": "relative", "percentile": 0.90},
    "relative_95p_3d": {"minimum_duration": 3, "kind": "relative", "percentile": 0.95},
    "compound_90p_2d": {"minimum_duration": 2, "kind": "compound", "percentile": 0.90},
}


def group_persistent_events(
    dates: pd.Series, exceeds: pd.Series, minimum_duration: int,
    threshold: pd.Series | float, tmax: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Group exceedances only when dates are consecutive; missing dates break events."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    exceeds = exceeds.fillna(False).astype(bool).reset_index(drop=True)
    tmax = tmax.reset_index(drop=True)
    thr = pd.Series(threshold, index=range(len(dates)), dtype=float) if np.isscalar(threshold) else threshold.reset_index(drop=True).astype(float)
    gap = dates.diff().dt.days.ne(1)
    new_run = exceeds & ((~exceeds.shift(1, fill_value=False)) | gap)
    run_id = new_run.cumsum()
    status = pd.Series(False, index=range(len(dates)), dtype=bool)
    rows = []
    exceed_indices = pd.Series(range(len(dates)))[exceeds.to_numpy()]
    for _, idx in exceed_indices.groupby(run_id[exceeds].to_numpy()).groups.items():
        idx = list(idx)
        if len(idx) < minimum_duration:
            continue
        status.iloc[idx] = True
        excess = tmax.iloc[idx].to_numpy() - thr.iloc[idx].to_numpy()
        rows.append({
            "start_date": dates.iloc[idx[0]], "end_date": dates.iloc[idx[-1]], "duration": len(idx),
            "peak_tmax": float(tmax.iloc[idx].max()), "mean_tmax": float(tmax.iloc[idx].mean()),
            "cumulative_excess": float(np.nansum(excess)), "maximum_excess": float(np.nanmax(excess)),
            "onset_day_of_year": int(dates.iloc[idx[0]].dayofyear),
            "cessation_day_of_year": int(dates.iloc[idx[-1]].dayofyear),
        })
    return status, pd.DataFrame(rows)


def construct_definition(
    df: pd.DataFrame, name: str, reference: tuple[int, int] = (1981, 2010),
    training_end_year: int | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    spec = DEFINITIONS[name]
    if spec["kind"] == "fixed":
        threshold = pd.Series(36.0, index=df.index)
        exceeds = df.tmax >= threshold
    else:
        threshold = calendar_day_threshold(df, "tmax", spec["percentile"], reference, training_end_year=training_end_year)
        exceeds = df.tmax > threshold
        if spec["kind"] == "compound":
            tmin_threshold = calendar_day_threshold(df, "tmin", spec["percentile"], reference, training_end_year=training_end_year)
            exceeds &= df.tmin > tmin_threshold
    status, events = group_persistent_events(df.date, exceeds, spec["minimum_duration"], threshold, df.tmax)
    if not events.empty:
        events.insert(0, "definition", name)
        events.insert(1, "event_id", [f"{name}_{i+1:04d}" for i in range(len(events))])
        events["start_year"] = pd.to_datetime(events.start_date).dt.year
        events["start_month"] = pd.to_datetime(events.start_date).dt.month
        events["decade"] = (events.start_year // 10) * 10
    return status.set_axis(df.index), events, threshold.set_axis(df.index)


def construct_all_definitions(df: pd.DataFrame, reference: tuple[int, int] = (1981, 2010)):
    daily = df[["date", "year", "month", "tmax", "tmin"]].copy()
    event_frames = []
    thresholds = {}
    for name in DEFINITIONS:
        status, events, threshold = construct_definition(df, name, reference)
        daily[name] = status
        thresholds[name] = threshold
        event_frames.append(events)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return daily, events, thresholds


def aggregate_metrics(daily: pd.DataFrame, events: pd.DataFrame, frequency: str = "annual") -> pd.DataFrame:
    keys = ["year"] if frequency == "annual" else ["year", "month"]
    rows = []
    for name in DEFINITIONS:
        counts = daily.groupby(keys)[name].sum()
        for key, heatwave_days in counts.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            mask = pd.Series(True, index=events.index)
            if events.empty:
                ev = events
            else:
                mask &= events.definition.eq(name)
                mask &= events.start_year.eq(key_tuple[0])
                if frequency == "monthly":
                    mask &= events.start_month.eq(key_tuple[1])
                ev = events[mask]
            row = {"definition": name, "heatwave_days": int(heatwave_days), "event_count": len(ev),
                   "total_intensity": float(ev.cumulative_excess.sum()) if len(ev) else 0.0,
                   "mean_duration": float(ev.duration.mean()) if len(ev) else 0.0,
                   "max_duration": int(ev.duration.max()) if len(ev) else 0}
            row.update(dict(zip(keys, key_tuple)))
            rows.append(row)
    return pd.DataFrame(rows)
