"""Descriptive statistics, correlations, and collinearity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .climatology import deseasonalize


def descriptive_statistics(df: pd.DataFrame, variables: list[str], stratum: str = "all") -> pd.DataFrame:
    rows = []
    for variable in variables:
        x = df[variable].dropna()
        rows.append({
            "stratum": stratum, "variable": variable, "valid_n": len(x),
            "missing_n": int(df[variable].isna().sum()), "mean": x.mean(), "standard_deviation": x.std(ddof=1),
            "median": x.median(), "first_quartile": x.quantile(.25), "third_quartile": x.quantile(.75),
            "interquartile_range": x.quantile(.75) - x.quantile(.25), "minimum": x.min(), "maximum": x.max(),
            "skewness": x.skew(), "kurtosis": x.kurt(),
        })
    return pd.DataFrame(rows)


def grouped_descriptives(df: pd.DataFrame, variables: list[str], group_columns: list[str]) -> pd.DataFrame:
    frames = []
    for group in group_columns:
        if group not in df:
            continue
        for value, part in df.groupby(group, observed=True):
            stats = descriptive_statistics(part, variables, f"{group}={value}")
            stats["comparison"] = group
            stats["group"] = value
            frames.append(stats)
        a = df[df[group].astype(bool)]
        b = df[~df[group].astype(bool)]
        pooled = np.sqrt((a[variables].var() + b[variables].var()) / 2)
        smd = (a[variables].mean() - b[variables].mean()) / pooled
        for frame in frames[-2:]:
            frame["standardized_mean_difference"] = frame.variable.map(smd)
    return pd.concat(frames, ignore_index=True)


def correlation_outputs(df: pd.DataFrame, variables: list[str]):
    x = df[variables]
    raw = x.corr(method="spearman")
    pair_n = x.notna().astype(int).T.dot(x.notna().astype(int))
    anomalies = deseasonalize(df, variables)[variables]
    anomaly_corr = anomalies.corr(method="spearman")
    pearson = x.corr(method="pearson")
    return raw, anomaly_corr, pair_n, pearson, anomalies


def collinearity_assessment(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    x = df[variables].dropna()
    z = (x - x.mean()) / x.std(ddof=0)
    corr = z.corr().fillna(0)
    np.fill_diagonal(corr.values, 1)
    distance = 1 - corr.abs()
    hierarchy = linkage(squareform(distance.values, checks=False), method="average")
    order = leaves_list(hierarchy)
    singular_values = np.linalg.svd(z.to_numpy(), compute_uv=False)
    condition_index = singular_values.max() / singular_values.min()
    rows = []
    for j, variable in enumerate(variables):
        vif = variance_inflation_factor(z.to_numpy(), j)
        max_peer = corr[variable].drop(variable).abs().idxmax()
        rows.append({
            "variable": variable, "vif": vif, "maximum_absolute_correlation": corr.loc[variable, max_peer],
            "most_correlated_variable": max_peer, "global_condition_index": condition_index,
            "correlation_cluster_order": int(np.where(order == j)[0][0]),
            "primary_association_eligible": variable in {"rh_mean", "precipitation", "wind_speed_mean", "pressure_mean"},
        })
    return pd.DataFrame(rows)
