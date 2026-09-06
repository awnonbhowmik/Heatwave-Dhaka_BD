"""Leakage-controlled future persistent-hot-window classification benchmark."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


S0_FEATURES = ["target_sin1", "target_cos1", "target_sin2", "target_cos2", "time_decades"]
S1_HISTORY_FEATURES = [
    "tmax_latest", "tmin_latest", "tmax_mean3", "tmin_mean3", "tmax_mean7",
    "tmin_mean7", "tmax_sd7", "tmin_sd7", "tmax_trend7", "tmin_trend7",
    "observed_hot_run_length",
]
S2_HISTORY_VARIABLES = [
    ("rh_mean", "relative humidity", "%"),
    ("precipitation", "precipitation", "mm/day"),
    ("wind_speed_mean", "wind speed", "km/h"),
    ("pressure_mean", "mean sea-level pressure", "hPa"),
    ("cloud_cover", "cloud cover", "%"),
    ("shortwave_radiation", "shortwave radiation", "kJ/m2/day"),
    ("soil_moisture_mean", "shallow soil moisture", "m3/m3"),
]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Training-only deterministic redundancy filter preserving input priority."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(getattr(X, "columns", [f"x{i}" for i in range(frame.shape[1])]))
        frame.columns = self.feature_names_in_
        corr = frame.corr(method="spearman").abs()
        keep: list[str] = []
        drop: list[str] = []
        for col in self.feature_names_in_:
            if any(pd.notna(corr.loc[col, old]) and corr.loc[col, old] > self.threshold for old in keep):
                drop.append(str(col))
            else:
                keep.append(str(col))
        self.keep_ = keep
        self.drop_ = drop
        self.indices_ = [int(np.where(self.feature_names_in_ == col)[0][0]) for col in keep]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.keep_].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


def _linear_trend(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(values))[mask], values[mask], 1)[0])


def _hot_run_length(values: Iterable[float], threshold: float = 36.0) -> int:
    run = 0
    for value in reversed(list(values)):
        if pd.notna(value) and value >= threshold:
            run += 1
        else:
            break
    return run


def fixed_event_ids(daily: pd.DataFrame, threshold: float = 36.0) -> pd.Series:
    """Identifier for every fixed-threshold consecutive run lasting >=3 days."""
    hot = daily.tmax.ge(threshold) & daily.tmax.notna()
    gap = daily.date.diff().dt.days.ne(1)
    run = (hot & ((~hot.shift(fill_value=False)) | gap)).cumsum()
    ids = pd.Series(pd.NA, index=daily.index, dtype="string")
    counter = 0
    for _, idx in daily.index[hot].to_series().groupby(run[hot].to_numpy()):
        idx = list(idx)
        if len(idx) >= 3:
            counter += 1
            ids.iloc[idx] = f"fixed36_{counter:04d}"
    return ids


def feature_dictionary() -> pd.DataFrame:
    rows = []
    for name in S0_FEATURES:
        rows.append({"feature": name, "information_set": "S0", "source_variable": "target_start calendar" if name != "time_decades" else "issue_date", "unit": "unitless", "aggregation": "harmonic/calendar trend", "availability": "known at issue time"})
    mapping = {
        "tmax": ("daily maximum temperature", "degC"),
        "tmin": ("daily minimum temperature", "degC"),
    }
    for name in S1_HISTORY_FEATURES:
        source = "tmax" if name.startswith("tmax") or name == "observed_hot_run_length" else "tmin"
        aggregation = name.rsplit("_", 1)[-1] if name != "observed_hot_run_length" else "consecutive observed days through issue"
        rows.append({"feature": name, "information_set": "S1", "source_variable": mapping[source][0], "unit": "days" if name == "observed_hot_run_length" else mapping[source][1], "aggregation": aggregation, "availability": "t-6 through t only"})
    for variable, label, unit in S2_HISTORY_VARIABLES:
        suffixes = ["latest", "mean7"] if variable != "precipitation" else ["latest", "sum7"]
        for suffix in suffixes:
            rows.append({"feature": f"{variable}_{suffix}", "information_set": "S2", "source_variable": label, "unit": unit, "aggregation": suffix, "availability": "t-6 through t only"})
    return pd.DataFrame(rows)


def feature_sets() -> dict[str, list[str]]:
    s2 = []
    for variable, _, _ in S2_HISTORY_VARIABLES:
        s2.extend([f"{variable}_latest", f"{variable}_{'sum7' if variable == 'precipitation' else 'mean7'}"])
    return {"S0": S0_FEATURES, "S1": S0_FEATURES + S1_HISTORY_FEATURES, "S2": S0_FEATURES + S1_HISTORY_FEATURES + s2}


def construct_prediction_samples(
    daily: pd.DataFrame,
    leads: Iterable[int] = (1, 3, 7),
    lookback_days: int = 7,
    threshold: float = 36.0,
    months: Iterable[int] = (3, 4, 5, 6),
) -> pd.DataFrame:
    """Construct issue-time features and exact future-window labels on a continuous calendar."""
    daily = daily.sort_values("date").reset_index(drop=True).copy()
    if daily.date.duplicated().any():
        raise ValueError("Duplicate dates are not permitted")
    calendar = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
    work = daily.set_index("date").reindex(calendar)
    work.index.name = "date"
    event_id = fixed_event_ids(daily, threshold=threshold)
    event_by_date = pd.Series(event_id.to_numpy(), index=daily.date)
    months = set(int(m) for m in months)
    rows: list[dict[str, Any]] = []
    for lead in leads:
        for issue_i in range(lookback_days - 1, len(work)):
            issue = work.index[issue_i]
            target_start = issue + timedelta(days=int(lead))
            target_end = target_start + timedelta(days=2)
            if target_end > work.index[-1] or any(d.month not in months for d in pd.date_range(target_start, target_end)):
                continue
            target = work.loc[target_start:target_end, "tmax"]
            if len(target) != 3 or target.isna().any():
                continue
            history = work.iloc[issue_i - lookback_days + 1:issue_i + 1]
            if len(history) != lookback_days:
                continue
            doy = target_start.dayofyear
            outcome = int(target.ge(threshold).all())
            row: dict[str, Any] = {
                "issue_date": issue,
                "feature_start": history.index[0],
                "feature_end": history.index[-1],
                "target_start": target_start,
                "target_end": target_end,
                "label_available_date": target_end,
                "lead": int(lead),
                "outcome": outcome,
                "associated_event_id": event_by_date.get(target_start, pd.NA) if outcome else pd.NA,
                "issue_year": issue.year,
                "target_month": target_start.month,
                "issue_tmax": work.iloc[issue_i].tmax,
                "target_tmax_day1": float(target.iloc[0]),
                "target_tmax_day2": float(target.iloc[1]),
                "target_tmax_day3": float(target.iloc[2]),
                "target_sin1": math.sin(2 * math.pi * doy / 365.25),
                "target_cos1": math.cos(2 * math.pi * doy / 365.25),
                "target_sin2": math.sin(4 * math.pi * doy / 365.25),
                "target_cos2": math.cos(4 * math.pi * doy / 365.25),
                "time_decades": (issue.year - 1972) / 10,
            }
            for variable in ("tmax", "tmin"):
                values = history[variable].to_numpy(dtype=float)
                row[f"{variable}_latest"] = values[-1]
                row[f"{variable}_mean3"] = np.nanmean(values[-3:])
                row[f"{variable}_mean7"] = np.nanmean(values)
                row[f"{variable}_sd7"] = np.nanstd(values, ddof=1)
                row[f"{variable}_trend7"] = _linear_trend(values)
            row["observed_hot_run_length"] = _hot_run_length(history.tmax, threshold)
            for variable, _, _ in S2_HISTORY_VARIABLES:
                values = history[variable].to_numpy(dtype=float)
                row[f"{variable}_latest"] = values[-1]
                row[f"{variable}_{'sum7' if variable == 'precipitation' else 'mean7'}"] = np.nansum(values) if variable == "precipitation" else np.nanmean(values)
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(["lead", "issue_date"]).reset_index(drop=True)
    assert_prediction_provenance(result)
    return result


def assert_prediction_provenance(samples: pd.DataFrame) -> None:
    if not (samples.feature_end == samples.issue_date).all():
        raise AssertionError("Feature windows must end at issue time")
    if not (samples.feature_start <= samples.feature_end).all():
        raise AssertionError("Invalid feature interval")
    if not (samples.target_start > samples.issue_date).all():
        raise AssertionError("Future target must begin after issue time")
    if not (samples.label_available_date == samples.target_end).all():
        raise AssertionError("Label availability must equal target-window end")
    if not (samples.target_end - samples.target_start == timedelta(days=2)).all():
        raise AssertionError("Target window must span exactly three dates")


def inner_year_splits(train: pd.DataFrame, folds: int = 3, block_years: int = 4, gap_days: int = 0) -> list[dict[str, Any]]:
    years = sorted(train.issue_year.unique())
    needed = folds * block_years
    if len(years) < needed + 5:
        raise ValueError("Insufficient years for the prespecified expanding inner schedule")
    validation_years = years[-needed:]
    splits = []
    for fold in range(folds):
        val_years = validation_years[fold * block_years:(fold + 1) * block_years]
        val_start = train.loc[train.issue_year.eq(val_years[0]), "issue_date"].min()
        fit_cutoff = val_start - timedelta(days=gap_days)
        tr_idx = train.index[(train.issue_year < val_years[0]) & (train.label_available_date < fit_cutoff)].to_numpy()
        va_idx = train.index[train.issue_year.isin(val_years)].to_numpy()
        splits.append({"inner_fold": fold + 1, "train_index": tr_idx, "validation_index": va_idx, "train_end": train.loc[tr_idx, "issue_date"].max(), "validation_start": train.loc[va_idx, "issue_date"].min(), "validation_end": train.loc[va_idx, "issue_date"].max(), "validation_years": val_years})
    return splits


def _candidate_space(family: str) -> dict[str, list[Any]]:
    spaces = {
        "logistic": {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], "model__penalty": ["l1", "l2"]},
        "weighted_rf": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 3, 7, 12], "model__max_features": ["sqrt", 0.7, 1.0]},
        "balanced_rf": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 3, 7, 12], "model__max_features": ["sqrt", 0.7, 1.0]},
        "xgboost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.07, 0.15], "model__subsample": [0.7, 0.9, 1.0], "model__colsample_bytree": [0.7, 1.0], "model__reg_lambda": [1.0, 5.0, 10.0]},
        "weighted_svc": {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0], "model__gamma": ["scale", 0.01, 0.03, 0.1, 0.3]},
    }
    return spaces[family]


def candidate_parameters(family: str, n_candidates: int, seed: int) -> list[dict[str, Any]]:
    return list(ParameterSampler(_candidate_space(family), n_iter=n_candidates, random_state=seed))


def make_pipeline(family: str, seed: int, n_estimators: int = 120, positive_weight: float = 1.0) -> Pipeline:
    if family == "logistic":
        model = LogisticRegression(solver="liblinear", penalty="l2", class_weight="balanced", max_iter=3000, random_state=seed)
        scale = StandardScaler()
    elif family == "weighted_rf":
        model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample", random_state=seed, n_jobs=1)
        scale = "passthrough"
    elif family == "balanced_rf":
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, replacement=True, bootstrap=False)
        scale = "passthrough"
    elif family == "xgboost":
        model = XGBClassifier(n_estimators=n_estimators, objective="binary:logistic", eval_metric="logloss", tree_method="hist", scale_pos_weight=positive_weight, random_state=seed, n_jobs=1, verbosity=0)
        scale = "passthrough"
    elif family == "weighted_svc":
        model = SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=seed, cache_size=512)
        scale = StandardScaler()
    else:
        raise KeyError(family)
    return Pipeline([("correlation", CorrelationFilter(0.95)), ("imputer", SimpleImputer(strategy="median")), ("scale", scale), ("model", model)])


def _raw_scores(model: Pipeline, X: pd.DataFrame, family: str) -> np.ndarray:
    if family == "weighted_svc":
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _score_average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.sum(y) > 0 else np.nan


@dataclass
class FittedCalibration:
    method: str
    model: LogisticRegression | None
    family: str
    fallback_probability: float

    def predict(self, raw_score: np.ndarray) -> np.ndarray:
        raw_score = np.asarray(raw_score, dtype=float)
        if self.model is not None:
            x = raw_score if self.family == "weighted_svc" else logit(np.clip(raw_score, 1e-6, 1 - 1e-6))
            return self.model.predict_proba(x.reshape(-1, 1))[:, 1]
        if self.family != "weighted_svc":
            return np.clip(raw_score, 1e-6, 1 - 1e-6)
        return np.repeat(self.fallback_probability, len(raw_score))


def fit_sigmoid_calibration(raw_score: np.ndarray, y: np.ndarray, family: str) -> FittedCalibration:
    raw_score = np.asarray(raw_score, dtype=float); y = np.asarray(y, dtype=int)
    prevalence = float((y.sum() + 1) / (len(y) + 2))
    if len(y) < 30 or y.sum() < 5 or (len(y) - y.sum()) < 5 or np.nanstd(raw_score) < 1e-10:
        return FittedCalibration("unavailable_identity_or_prevalence", None, family, prevalence)
    x = raw_score if family == "weighted_svc" else logit(np.clip(raw_score, 1e-6, 1 - 1e-6))
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(x.reshape(-1, 1), y)
    return FittedCalibration("temporal_sigmoid", cal, family, prevalence)


def select_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.r_[0.5, np.quantile(probability, np.linspace(0.02, 0.98, 97))])
    scores = [balanced_accuracy_score(y, probability >= cut) for cut in candidates]
    best = np.flatnonzero(np.isclose(scores, np.nanmax(scores)))
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def tune_outer_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    family: str,
    seed: int,
    candidates: int = 12,
    inner_folds: int = 3,
    n_estimators: int = 120,
    gap_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Pipeline | None]:
    splits = inner_year_splits(train, folds=inner_folds, gap_days=gap_days)
    params_list = candidate_parameters(family, candidates, seed)
    tuning_rows: list[dict[str, Any]] = []
    best_score = -np.inf; best_complexity = np.inf; best_params: dict[str, Any] | None = None
    for candidate_id, params in enumerate(params_list, 1):
        fold_scores = []
        for split in splits:
            tr = train.loc[split["train_index"]]; va = train.loc[split["validation_index"]]
            failure = ""; captured = []
            score = np.nan
            if tr.outcome.nunique() < 2:
                failure = "one_class_training"
            else:
                weight = float((len(tr) - tr.outcome.sum()) / max(tr.outcome.sum(), 1))
                pipeline = make_pipeline(family, seed + candidate_id + split["inner_fold"], n_estimators, weight).set_params(**params)
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        pipeline.fit(tr[features], tr.outcome.astype(int))
                    captured = [f"{w.category.__name__}: {w.message}" for w in caught]
                    score = _score_average_precision(va.outcome.to_numpy(int), _raw_scores(pipeline, va[features], family))
                    if np.isnan(score):
                        failure = "no_positive_validation"
                except Exception as exc:  # recorded and allowed to lose candidate
                    failure = f"{type(exc).__name__}: {exc}"
            fold_scores.append(score)
            tuning_rows.append({"candidate_id": candidate_id, "inner_fold": split["inner_fold"], "family": family, "parameters": json.dumps(params, sort_keys=True), "train_end": split["train_end"], "validation_start": split["validation_start"], "validation_end": split["validation_end"], "validation_years": ",".join(map(str, split["validation_years"])), "average_precision": score, "warning": " | ".join(captured), "failure": failure})
        mean_score = float(np.nanmean(fold_scores)) if np.isfinite(fold_scores).any() else -np.inf
        complexity = sum(1 for value in params.values() if value not in (None, "sqrt", "scale"))
        if mean_score > best_score or (np.isclose(mean_score, best_score) and complexity < best_complexity):
            best_score, best_complexity, best_params = mean_score, complexity, params
    if best_params is None:
        prevalence = float((train.outcome.sum() + 1) / (len(train) + 2))
        pred = test.copy()
        pred["raw_score"] = prevalence; pred["probability"] = prevalence; pred["threshold"] = 0.5
        meta = {"status": "constant_fallback", "selected_parameters": {}, "calibration": "unavailable", "selected_threshold": 0.5, "selected_inner_average_precision": np.nan, "retained_features": features}
        return pd.DataFrame(tuning_rows), pred, meta, None

    oof_rows = []
    for split in splits:
        tr = train.loc[split["train_index"]]; va = train.loc[split["validation_index"]]
        if tr.outcome.nunique() < 2:
            raw = np.repeat((tr.outcome.sum() + 1) / (len(tr) + 2), len(va))
        else:
            weight = float((len(tr) - tr.outcome.sum()) / max(tr.outcome.sum(), 1))
            pipeline = make_pipeline(family, seed + 1000 + split["inner_fold"], n_estimators, weight).set_params(**best_params)
            pipeline.fit(tr[features], tr.outcome.astype(int))
            raw = _raw_scores(pipeline, va[features], family)
        oof_rows.append(pd.DataFrame({"index": va.index, "outcome": va.outcome.to_numpy(int), "raw_score": raw, "inner_fold": split["inner_fold"]}))
    oof = pd.concat(oof_rows, ignore_index=True).sort_values("index")
    calibration = fit_sigmoid_calibration(oof.raw_score.to_numpy(), oof.outcome.to_numpy(), family)
    oof["probability"] = calibration.predict(oof.raw_score.to_numpy())
    threshold = select_threshold(oof.outcome.to_numpy(), oof.probability.to_numpy())
    weight = float((len(train) - train.outcome.sum()) / max(train.outcome.sum(), 1))
    final_model = make_pipeline(family, seed + 9999, n_estimators, weight).set_params(**best_params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_model.fit(train[features], train.outcome.astype(int))
    raw_test = _raw_scores(final_model, test[features], family)
    pred = test.copy()
    pred["raw_score"] = raw_test
    pred["probability"] = calibration.predict(raw_test)
    pred["threshold"] = threshold
    retained = list(final_model.named_steps["correlation"].get_feature_names_out())
    meta = {"status": "fit", "selected_parameters": best_params, "calibration": calibration.method, "selected_threshold": threshold, "selected_inner_average_precision": best_score, "retained_features": retained, "dropped_redundant_features": final_model.named_steps["correlation"].drop_, "final_warnings": [f"{w.category.__name__}: {w.message}" for w in caught], "calibrator": calibration}
    return pd.DataFrame(tuning_rows), pred, meta, final_model


def classification_metrics(y: Iterable[int], probability: Iterable[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int); p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    reasons = []
    two_class = len(np.unique(y)) == 2
    roc = float(roc_auc_score(y, p)) if two_class else np.nan
    ap = float(average_precision_score(y, p)) if y.sum() else np.nan
    if not two_class: reasons.append("roc_auc/calibration undefined: one observed class")
    if not y.sum(): reasons.append("average_precision/recall/precision/f1 undefined: no positives")
    cal_intercept = cal_slope = np.nan
    if two_class and np.nanstd(p) > 1e-12:
        try:
            x = logit(p).reshape(-1, 1)
            cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(x, y)
            cal_intercept = float(cal.intercept_[0]); cal_slope = float(cal.coef_[0, 0])
        except Exception as exc:
            reasons.append(f"calibration regression failed: {type(exc).__name__}")
    return {
        "n": len(y), "positive_n": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "recall": float(recall_score(y, pred, zero_division=np.nan)),
        "precision": float(precision_score(y, pred, zero_division=np.nan)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y, pred, zero_division=np.nan)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two_class else np.nan,
        "average_precision": ap, "roc_auc": roc,
        "brier_score": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": cal_intercept, "calibration_slope": cal_slope,
        "accuracy": float(np.mean(pred == y)), "threshold": float(threshold),
        "undefined_reason": "; ".join(reasons),
    }


def baseline_probabilities(train: pd.DataFrame, test: pd.DataFrame, kind: str) -> np.ndarray:
    prevalence = float((train.outcome.sum() + 1) / (len(train) + 2))
    if kind == "always_negative":
        return np.repeat(1e-6, len(test))
    if kind == "seasonal_probability":
        counts = train.groupby("target_month").outcome.agg(["sum", "count"])
        return np.asarray([(counts.loc[m, "sum"] + 1) / (counts.loc[m, "count"] + 2) if m in counts.index else prevalence for m in test.target_month], dtype=float)
    if kind == "temperature_transition":
        tr = train.copy(); te = test.copy()
        tr["state"] = np.minimum(tr.observed_hot_run_length.fillna(0), 3).astype(int)
        te["state"] = np.minimum(te.observed_hot_run_length.fillna(0), 3).astype(int)
        grouped = tr.groupby(["target_month", "state"]).outcome.agg(["sum", "count"])
        state_group = tr.groupby("state").outcome.agg(["sum", "count"])
        values = []
        for month, state in zip(te.target_month, te.state):
            if (month, state) in grouped.index and grouped.loc[(month, state), "count"] >= 10:
                row = grouped.loc[(month, state)]
            elif state in state_group.index:
                row = state_group.loc[state]
            else:
                values.append(prevalence); continue
            values.append((row["sum"] + 1) / (row["count"] + 2))
        return np.asarray(values, dtype=float)
    raise KeyError(kind)


def prediction_rows(pred: pd.DataFrame, model: str, feature_set: str, outer_year: int, calibration: str) -> pd.DataFrame:
    columns = ["issue_date", "feature_start", "feature_end", "target_start", "target_end", "label_available_date", "lead", "outcome", "associated_event_id", "raw_score", "probability", "threshold"]
    out = pred[columns].copy()
    out.insert(0, "model", model); out.insert(1, "feature_set", feature_set)
    out.insert(2, "outer_fold", outer_year); out["predicted_class"] = (out.probability >= out.threshold).astype(int)
    out["calibration"] = calibration
    return out


def metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "feature_set", "lead", "outer_fold"]
    for key, frame in predictions.groupby(keys, dropna=False):
        metrics = classification_metrics(frame.outcome, frame.probability, float(frame.threshold.iloc[0]))
        rows.append(dict(zip(keys, key)) | {"scope": "held_out_season"} | metrics)
        sens = classification_metrics(frame.outcome, frame.probability, 0.5)
        rows.append(dict(zip(keys, key)) | {"scope": "held_out_season_threshold_0.5"} | sens)
    for key, frame in predictions.groupby(["model", "feature_set", "lead"], dropna=False):
        # Classification uses each fold's training-selected threshold.
        pclass = frame.predicted_class.to_numpy(int); y = frame.outcome.to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(y, pclass, labels=[0, 1]).ravel()
        metrics = classification_metrics(y, frame.probability, 0.5)
        metrics.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "recall": tp / (tp + fn) if tp + fn else np.nan, "precision": tp / (tp + fp) if tp + fp else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan, "balanced_accuracy": np.nanmean([tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp) if tn + fp else np.nan]), "accuracy": float(np.mean(pclass == y)), "threshold": np.nan})
        rows.append(dict(zip(["model", "feature_set", "lead"], key)) | {"outer_fold": np.nan, "scope": "pooled_strictly_out_of_sample"} | metrics)
    return pd.DataFrame(rows)


def year_block_bootstrap(predictions: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    base = predictions[predictions.feature_set.isin(["S1", "S2"])].copy()
    for (model, lead), frame in base.groupby(["model", "lead"]):
        if set(frame.feature_set.unique()) != {"S1", "S2"}: continue
        years = sorted(frame.outer_fold.unique())
        observed = {}
        for feature_set in ["S1", "S2"]:
            f = frame[frame.feature_set.eq(feature_set)]
            observed[feature_set] = {"average_precision": average_precision_score(f.outcome, f.probability) if f.outcome.sum() else np.nan, "brier_score": brier_score_loss(f.outcome, f.probability)}
        for metric in ["average_precision", "brier_score"]:
            values = [] ; undefined = 0
            for _ in range(draws):
                sampled = rng.choice(years, len(years), replace=True)
                pieces = []
                for draw_id, year in enumerate(sampled):
                    piece = frame[frame.outer_fold.eq(year)].copy(); piece["draw_id"] = draw_id; pieces.append(piece)
                boot = pd.concat(pieces, ignore_index=True)
                scores = {}
                for fs in ["S1", "S2"]:
                    f = boot[boot.feature_set.eq(fs)]
                    if metric == "average_precision":
                        scores[fs] = average_precision_score(f.outcome, f.probability) if f.outcome.sum() else np.nan
                    else: scores[fs] = brier_score_loss(f.outcome, f.probability)
                delta = scores["S2"] - scores["S1"]
                if np.isfinite(delta): values.append(delta)
                else: undefined += 1
            values = np.asarray(values)
            rows.append({"model": model, "lead": lead, "metric": metric, "contrast": "S2_minus_S1", "observed_difference": observed["S2"][metric] - observed["S1"][metric], "ci_lower": np.quantile(values, .025) if len(values) else np.nan, "ci_upper": np.quantile(values, .975) if len(values) else np.nan, "bootstrap_draws": draws, "defined_draws": len(values), "undefined_draws": undefined, "resampling_unit": "complete_held_out_year"})
    return pd.DataFrame(rows)
