"""Continuous temperature trend models and nonparametric sensitivity estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def annual_temperature_series(df: pd.DataFrame, complete_year_end: int = 2023) -> pd.DataFrame:
    complete = df[df.year <= complete_year_end].copy()
    counts = complete.groupby("year").date.nunique()
    expected = pd.Series({y: 366 if pd.Timestamp(y, 12, 31).is_leap_year else 365 for y in counts.index})
    complete = complete[complete.year.isin(counts.index[counts == expected])]
    annual = complete.groupby("year").agg(
        annual_mean_tmax=("tmax", "mean"), annual_mean_tmin=("tmin", "mean"),
        annual_mean_day_night_range=("day_night_range", "mean"),
    ).reset_index()
    hot = df[df.month.isin([3, 4, 5, 6])].groupby("year").filter(lambda x: x.date.nunique() == 122)
    hot = hot.groupby("year").agg(march_june_mean_tmax=("tmax", "mean"), march_june_mean_tmin=("tmin", "mean")).reset_index()
    return annual.merge(hot, on="year", how="outer").sort_values("year")


def _mk_with_prewhitening(year: np.ndarray, values: np.ndarray):
    residual = values - stats.linregress(year, values).intercept - stats.linregress(year, values).slope * year
    lag1 = np.corrcoef(residual[:-1], residual[1:])[0, 1]
    if np.isfinite(lag1) and abs(lag1) > 1.96 / np.sqrt(len(values)):
        transformed = values[1:] - lag1 * values[:-1]
        tau, p = stats.kendalltau(year[1:], transformed)
        return tau, p, lag1, "trend-free prewhitened sensitivity"
    tau, p = stats.kendalltau(year, values)
    return tau, p, lag1, "unmodified (serial correlation not detected)"


def fit_temperature_trends(series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in [c for c in series if c != "year"]:
        d = series[["year", outcome]].dropna()
        x = sm.add_constant((d.year - d.year.mean()) / 10)
        ols = sm.OLS(d[outcome], x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        slope = ols.params["year"]
        ci = ols.conf_int().loc["year"]
        sen, _, sen_lo, sen_hi = stats.theilslopes(d[outcome], d.year, alpha=.95)
        tau, mkp, lag1, method = _mk_with_prewhitening(d.year.to_numpy(), d[outcome].to_numpy())
        rows.append({
            "outcome": outcome, "n_years": len(d), "start_year": d.year.min(), "end_year": d.year.max(),
            "ols_hac_slope_per_decade": slope, "ols_hac_se": ols.bse["year"],
            "ols_hac_ci_lower": ci[0], "ols_hac_ci_upper": ci[1], "ols_hac_p_value": ols.pvalues["year"],
            "r_squared": ols.rsquared,
            "sen_slope_per_decade": sen * 10, "sen_ci_lower_per_decade": sen_lo * 10,
            "sen_ci_upper_per_decade": sen_hi * 10, "mann_kendall_tau": tau,
            "mann_kendall_p_value": mkp, "residual_lag1_correlation": lag1, "mann_kendall_method": method,
        })
    return pd.DataFrame(rows)


def hot_season_endpoint_sensitivity(series: pd.DataFrame) -> pd.DataFrame:
    """Refit hot-season trends through 2023 and 2024 using the same HAC model."""
    rows = []
    for endpoint in (2023, 2024):
        subset = series[series.year <= endpoint]
        fitted = fit_temperature_trends(
            subset[["year", "march_june_mean_tmax", "march_june_mean_tmin"]]
        )
        fitted.insert(1, "endpoint", endpoint)
        rows.append(fitted)
    return pd.concat(rows, ignore_index=True)


def test_tmax_tmin_slope_difference(series: pd.DataFrame) -> dict:
    long = series.melt("year", ["annual_mean_tmax", "annual_mean_tmin"], var_name="temperature_type", value_name="temperature").dropna()
    long["decade"] = (long.year - long.year.mean()) / 10
    model = smf.ols("temperature ~ decade * C(temperature_type)", long).fit(cov_type="cluster", cov_kwds={"groups": long.year})
    term = "decade:C(temperature_type)[T.annual_mean_tmin]"
    ci = model.conf_int().loc[term]
    return {"contrast": "Tmin minus Tmax trend", "difference_per_decade": model.params[term],
            "ci_lower": ci[0], "ci_upper": ci[1], "p_value": model.pvalues[term]}
