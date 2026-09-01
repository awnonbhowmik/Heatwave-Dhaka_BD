"""Leap-safe calendar-day climatologies and percentile thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def climatology_day(dates: pd.Series) -> pd.Series:
    """Map dates to a 365-day climatological calendar; Feb 29 shares Feb 28's index."""
    doy = dates.dt.dayofyear.astype(int)
    after_feb = dates.dt.is_leap_year & (dates.dt.month > 2)
    return doy - after_feb.astype(int)


def calendar_day_threshold(
    df: pd.DataFrame, variable: str, percentile: float, reference: tuple[int, int],
    window: int = 7, method: str = "linear", training_end_year: int | None = None,
) -> pd.Series:
    """Estimate circular +/- window calendar-day quantiles using reference years only."""
    start, end = reference
    if training_end_year is not None:
        end = min(end, training_end_year)
    ref = df[(df.year >= start) & (df.year <= end)].copy()
    if ref.empty:
        raise ValueError("No observations in climatology reference period")
    ref["clim_day"] = climatology_day(ref.date)
    thresholds = {}
    for day in range(1, 366):
        delta = np.abs(ref.clim_day.to_numpy() - day)
        circular = np.minimum(delta, 365 - delta)
        values = ref.loc[circular <= window, variable].dropna().to_numpy()
        thresholds[day] = float(np.quantile(values, percentile, method=method))
    mapped = climatology_day(df.date).map(thresholds).astype(float)
    feb29 = (df.date.dt.month == 2) & (df.date.dt.day == 29)
    if feb29.any():
        mapped.loc[feb29] = (thresholds[59] + thresholds[60]) / 2
    return mapped


def deseasonalize(df: pd.DataFrame, variables: list[str], reference: tuple[int, int] = (1981, 2010)) -> pd.DataFrame:
    out = df[["date", "year", "month"]].copy()
    clim_day = climatology_day(df.date)
    ref_mask = df.year.between(*reference)
    for variable in variables:
        means = df.loc[ref_mask].assign(clim_day=clim_day[ref_mask]).groupby("clim_day")[variable].mean()
        smooth = pd.Series({d: means.reindex([((d + k - 1) % 365) + 1 for k in range(-7, 8)]).mean() for d in range(1, 366)})
        out[variable] = df[variable] - clim_day.map(smooth)
    return out
