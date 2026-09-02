"""Leakage-safe antecedent meteorological association models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve,
)

from .variable_dictionary import PRIMARY_ASSOCIATION_PREDICTORS, TARGET_DERIVED_PREDICTORS


def construct_antecedent_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for variable in ["rh_mean", "wind_speed_mean", "pressure_mean"]:
        out[f"{variable}_lag1"] = out[variable].shift(1)
        out[f"{variable}_lag3_mean"] = out[variable].shift(1).rolling(3, min_periods=3).mean()
        out[f"{variable}_lag7_mean"] = out[variable].shift(1).rolling(7, min_periods=7).mean()
    out["precipitation_lag1"] = out.precipitation.shift(1)
    out["precipitation_lag3_mean"] = out.precipitation.shift(1).rolling(3, min_periods=3).mean()
    out["precipitation_lag7_sum"] = out.precipitation.shift(1).rolling(7, min_periods=7).sum()
    for k in (1, 2, 3):
        out[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * out.day_of_year / 365.25)
        out[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * out.day_of_year / 365.25)
    out["decade"] = (out.year - out.year.mean()) / 10
    return out


def assert_no_target_leakage(predictors: list[str]) -> None:
    bad = [p for p in predictors if p in TARGET_DERIVED_PREDICTORS or p.startswith(("tmax", "tmin", "tmean", "vpd", "heat_index"))]
    if bad:
        raise ValueError(f"Target-derived predictors are prohibited: {bad}")


def _design(data: pd.DataFrame, predictors: list[str], means=None, scales=None):
    seasonal = [f"{kind}_doy_{k}" for k in (1, 2, 3) for kind in ("sin", "cos")]
    columns = ["decade"] + seasonal + predictors
    x = data[columns].copy()
    if means is None:
        means = x.mean()
        scales = x.std(ddof=0).replace(0, 1)
        scales[seasonal] = 1.0
        means[seasonal] = 0.0
    x = (x - means) / scales
    return sm.add_constant(x, has_constant="add"), means, scales


def fit_association_models(
    data: pd.DataFrame, outcome: str = "persistent_36c_3d",
    predictors: list[str] | None = None, excluded_years: list[int] | None = None,
):
    predictors = list(PRIMARY_ASSOCIATION_PREDICTORS if predictors is None else predictors)
    assert_no_target_leakage(predictors)
    hot = data[data.month.isin([3, 4, 5, 6])].dropna(subset=[outcome] + predictors).copy()
    if excluded_years:
        hot = hot[~hot.year.isin(excluded_years)].copy()
    X_base, means_base, scales_base = _design(hot, [])
    X_full, means_full, scales_full = _design(hot, predictors)
    # GEE handles repeated daily observations and serial correlation within year.
    # NumPy arrays avoid a statsmodels 0.14.5 / pandas 2.3 multidimensional-indexing incompatibility.
    def gee(x):
        result=sm.GEE(hot[outcome].astype(int).to_numpy(),x.to_numpy(),groups=hot.year.to_numpy(),
                      time=hot.day_of_year.to_numpy(),family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Autoregressive(grid=True)).fit(maxiter=200)
        result._analysis_terms=list(x.columns)
        return result
    base=gee(X_base); full=gee(X_full)
    return hot, base, full, (means_base, scales_base), (means_full, scales_full)


def association_sensitivity(data: pd.DataFrame, influential_years: list[int]) -> pd.DataFrame:
    """Evaluate event onset, lag windows, and influential-season exclusions."""
    data = data.copy()
    data["persistent_onset"] = (
        data.persistent_36c_3d.astype(bool)
        & ~data.persistent_36c_3d.shift(1, fill_value=False).astype(bool)
    )
    variants = [
        ("primary_persistent_days", "persistent_36c_3d", list(PRIMARY_ASSOCIATION_PREDICTORS), []),
        ("event_onset", "persistent_onset", list(PRIMARY_ASSOCIATION_PREDICTORS), []),
        ("one_day_lags", "persistent_36c_3d", [
            "rh_mean_lag1", "precipitation_lag1", "wind_speed_mean_lag1", "pressure_mean_lag1",
        ], []),
        ("seven_day_lags", "persistent_36c_3d", [
            "rh_mean_lag7_mean", "precipitation_lag7_sum",
            "wind_speed_mean_lag7_mean", "pressure_mean_lag7_mean",
        ], []),
    ]
    variants.extend([
        (f"exclude_{year}", "persistent_36c_3d", list(PRIMARY_ASSOCIATION_PREDICTORS), [year])
        for year in influential_years
    ])
    frames = []
    for variant, outcome, predictors, excluded in variants:
        hot, _, full, _, _ = fit_association_models(
            data, outcome=outcome, predictors=predictors, excluded_years=excluded,
        )
        estimates = association_estimates(full, full)
        estimates = estimates[estimates.model.eq("antecedent_full")].copy()
        estimates.insert(0, "variant", variant)
        estimates.insert(1, "outcome", outcome)
        estimates.insert(2, "positive_days", int(hot[outcome].sum()))
        frames.append(estimates)
    return pd.concat(frames, ignore_index=True)


def association_estimates(base, full) -> pd.DataFrame:
    rows = []
    for model_name, model in [("seasonal_trend_base", base), ("antecedent_full", full)]:
        terms=model._analysis_terms
        params=pd.Series(np.asarray(model.params),index=terms); bse=pd.Series(np.asarray(model.bse),index=terms)
        cis=pd.DataFrame(np.asarray(model.conf_int()),index=terms)
        pvalues=pd.Series(np.asarray(model.pvalues),index=terms)
        for term in terms:
            if term == "const" or "doy" in term:
                continue
            ci = cis.loc[term]
            rows.append({"model": model_name, "term": term, "coefficient": params[term],
                         "standard_error": bse[term], "adjusted_odds_ratio": np.exp(params[term]),
                         "or_ci_lower": np.exp(ci[0]), "or_ci_upper": np.exp(ci[1]), "p_value": pvalues[term],
                         "aic": np.nan, "log_likelihood": np.nan,
                         "working_correlation":"AR(1)","dependence_parameter":float(np.asarray(model.cov_struct.dep_params))})
    return pd.DataFrame(rows)


def classification_metrics(y, probability, threshold: float) -> dict:
    y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
    pred = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    if len(np.unique(y)) > 1:
        calibration = sm.GLM(y, sm.add_constant(np.log(np.clip(probability, 1e-6, 1-1e-6) / np.clip(1-probability, 1e-6, 1))), family=sm.families.Binomial()).fit()
        cal_intercept,cal_slope=calibration.params
    else:
        cal_intercept,cal_slope=np.nan,np.nan
    return {
        "n": len(y), "positive_n": int(y.sum()), "brier_score": brier_score_loss(y, probability),
        "roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) > 1 else np.nan,
        "precision_recall_auc": average_precision_score(y, probability) if y.sum() else np.nan,
        "threshold": threshold, "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "calibration_intercept": cal_intercept, "calibration_slope": cal_slope,
    }


def rolling_binary_validation(data: pd.DataFrame, origins: list[int], outcome: str = "persistent_36c_3d") -> pd.DataFrame:
    predictors = list(PRIMARY_ASSOCIATION_PREDICTORS)
    rows = []; pooled = {"seasonal_trend_base": [], "antecedent_full": []}
    for year in origins:
        train = data[(data.year < year) & data.month.isin([3, 4, 5, 6])].dropna(subset=[outcome] + predictors)
        test = data[(data.year == year) & data.month.isin([3, 4, 5, 6])].dropna(subset=[outcome] + predictors)
        if test.empty or train[outcome].sum() < 10:
            continue
        for model_name, model_predictors in [("seasonal_trend_base", []), ("antecedent_full", predictors)]:
            Xtr, means, scales = _design(train, model_predictors)
            Xte, _, _ = _design(test, model_predictors, means, scales)
            model = sm.GLM(train[outcome].astype(int), Xtr, family=sm.families.Binomial()).fit()
            p_train = model.predict(Xtr)
            # Youden threshold is selected exclusively on training observations.
            fpr,tpr,candidates=roc_curve(train[outcome].astype(int),p_train)
            finite=np.isfinite(candidates); scores=tpr[finite]-fpr[finite]
            threshold=float(candidates[finite][int(np.nanargmax(scores))])
            metrics = classification_metrics(test[outcome], model.predict(Xte), threshold)
            probability = model.predict(Xte)
            metrics = classification_metrics(test[outcome], probability, threshold)
            rows.append({"validation_scope": "held_out_season", "validation_year": year,
                         "model": model_name, "train_end_year": year - 1, **metrics})
            pooled[model_name].append((test[outcome].to_numpy(dtype=int), np.asarray(probability), threshold))
    for model_name, pieces in pooled.items():
        if not pieces:
            continue
        y = np.concatenate([piece[0] for piece in pieces])
        probability = np.concatenate([piece[1] for piece in pieces])
        # Thresholds remain training-origin specific for classification metrics.
        prediction = np.concatenate([piece[1] >= piece[2] for piece in pieces])
        metrics = classification_metrics(y, probability, .5)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        metrics["threshold"] = np.nan
        metrics["sensitivity"] = tp / (tp + fn) if tp + fn else np.nan
        metrics["specificity"] = tn / (tn + fp) if tn + fp else np.nan
        rows.append({"validation_scope": "pooled_strictly_out_of_sample", "validation_year": np.nan,
                     "model": model_name, "train_end_year": max(origins) - 1, **metrics})
    return pd.DataFrame(rows)
