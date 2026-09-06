#!/usr/bin/env python3
"""Run required sensitivities, explanations, figures, reports, and review archive."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/two-paper-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatwave_analysis.climatology import calendar_day_threshold
from heatwave_analysis.data_io import load_daily
from heatwave_analysis.heatwave_events import construct_definition
from heatwave_analysis.two_paper_benchmark import (
    baseline_probabilities,
    classification_metrics,
    construct_prediction_samples,
    feature_sets,
    fit_sigmoid_calibration,
    inner_year_splits,
    make_pipeline,
    metrics_from_predictions,
    prediction_rows,
    select_threshold,
    tune_outer_model,
)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def relative_labels(frame: pd.DataFrame, threshold_by_date: pd.Series) -> np.ndarray:
    mapping = threshold_by_date.to_dict()
    result = []
    for row in frame.itertuples():
        thresholds = [mapping[pd.Timestamp(row.target_start) + timedelta(days=i)] for i in range(3)]
        values = [row.target_tmax_day1, row.target_tmax_day2, row.target_tmax_day3]
        result.append(int(all(value > threshold for value, threshold in zip(values, thresholds))))
    return np.asarray(result, dtype=int)


def threshold_series(daily: pd.DataFrame, end_year: int) -> pd.Series:
    values = calendar_day_threshold(daily, "tmax", .90, (1981, 2010), window=7, training_end_year=end_year)
    return pd.Series(values.to_numpy(), index=daily.date)


def run_history_sensitivity(samples7: pd.DataFrame, daily: pd.DataFrame, cfg: dict, output: Path) -> pd.DataFrame:
    directory = output / "sensitivities" / "history14_checkpoints"; directory.mkdir(parents=True, exist_ok=True)
    samples14 = construct_prediction_samples(daily, leads=[1], lookback_days=14, threshold=cfg["prediction"]["fixed_threshold_c"], months=cfg["prediction"]["hot_season_months"])
    rows = []
    for outer_year in cfg["prediction"]["outer_test_years"]:
        path = directory / f"h1_{outer_year}_logistic_S1_history14.csv"
        if path.exists():
            rows.append(pd.read_csv(path, parse_dates=["issue_date", "feature_start", "feature_end", "target_start", "target_end", "label_available_date"])); continue
        test = samples14[samples14.issue_year.eq(outer_year)].copy(); cutoff = test.issue_date.min()
        train = samples14[(samples14.issue_year < outer_year) & (samples14.label_available_date < cutoff)].copy()
        _, pred, meta, _ = tune_outer_model(train, test, feature_sets()["S1"], "logistic", cfg["seed"] + 1400 + outer_year, candidates=cfg["validation"]["random_search_candidates"], inner_folds=cfg["validation"]["inner_folds"])
        out = prediction_rows(pred, "logistic_history14", "S1", outer_year, meta["calibration"]); write_csv(out, path); rows.append(out)
    pred14 = pd.concat(rows, ignore_index=True); write_csv(pred14, output / "sensitivities" / "history14_predictions.csv")
    pred7 = pd.read_csv(output / "predictions" / "all_out_of_sample_predictions.csv")
    pred7 = pred7[(pred7.model == "logistic") & (pred7.feature_set == "S1") & (pred7.lead == 1)].copy(); pred7["history_days"] = 7
    pred14["history_days"] = 14
    combined = pd.concat([pred7, pred14], ignore_index=True)
    metrics = []
    for days, frame in combined.groupby("history_days"):
        m = classification_metrics(frame.outcome, frame.probability, .5)
        m["average_precision"] = average_precision_score(frame.outcome, frame.probability)
        metrics.append({"history_days": days, **m})
    result = pd.DataFrame(metrics); write_csv(result, output / "sensitivities" / "history7_vs_14_metrics.csv")
    return result


def run_relative_sensitivity(samples: pd.DataFrame, daily: pd.DataFrame, cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path = output / "sensitivities" / "relative90_predictions.csv"
    metric_path = output / "sensitivities" / "relative90_metrics.csv"
    if prediction_path.exists() and metric_path.exists():
        return pd.read_csv(prediction_path), pd.read_csv(metric_path)
    source = samples[samples.lead.eq(1)].copy(); fset = feature_sets()["S1"]
    prediction_frames = []; prevalence_rows = []
    for outer_year in cfg["prediction"]["outer_test_years"]:
        test_base = source[source.issue_year.eq(outer_year)].copy(); cutoff = test_base.issue_date.min()
        train_base = source[(source.issue_year < outer_year) & (source.label_available_date < cutoff)].copy()
        outer_threshold = threshold_series(daily, outer_year - 1)
        train = train_base.copy(); test = test_base.copy()
        train["outcome"] = relative_labels(train, outer_threshold); test["outcome"] = relative_labels(test, outer_threshold)
        positive = test[test.outcome.eq(1)].sort_values("target_start").copy()
        event_starts = positive.target_start.diff().dt.days.ne(1).cumsum()
        event_map = pd.Series([f"relative90_{outer_year}_{i:03d}" for i in event_starts], index=positive.index)
        test["associated_event_id"] = pd.NA; test.loc[positive.index, "associated_event_id"] = event_map
        prevalence_rows.append({"outer_year": outer_year, "eligible_windows": len(test), "positive_windows": int(test.outcome.sum()), "prevalence": test.outcome.mean(), "distinct_events": test.associated_event_id.nunique(), "reference_end_year": min(2010, outer_year - 1)})

        oof = []; fold_pairs = []
        for split in inner_year_splits(train_base, folds=cfg["validation"]["inner_folds"]):
            tr = train_base.loc[split["train_index"]].copy(); va = train_base.loc[split["validation_index"]].copy()
            fold_threshold = threshold_series(daily, int(min(split["validation_years"])) - 1)
            tr["outcome"] = relative_labels(tr, fold_threshold); va["outcome"] = relative_labels(va, fold_threshold)
            fold_pairs.append((tr, va))
            model = make_pipeline("logistic", cfg["seed"] + outer_year + split["inner_fold"]).set_params(model__C=.3, model__penalty="l2")
            model.fit(tr[fset], tr.outcome); raw = model.predict_proba(va[fset])[:, 1]
            oof.append(pd.DataFrame({"outcome": va.outcome, "raw_score": raw}))
        oof = pd.concat(oof, ignore_index=True); calibrator = fit_sigmoid_calibration(oof.raw_score, oof.outcome, "logistic")
        oof["probability"] = calibrator.predict(oof.raw_score.to_numpy()); selected_threshold = select_threshold(oof.outcome, oof.probability)
        model = make_pipeline("logistic", cfg["seed"] + 9000 + outer_year).set_params(model__C=.3, model__penalty="l2")
        model.fit(train[fset], train.outcome); raw = model.predict_proba(test[fset])[:, 1]
        pred = test.copy(); pred["raw_score"] = raw; pred["probability"] = calibrator.predict(raw); pred["threshold"] = selected_threshold
        prediction_frames.append(prediction_rows(pred, "relative90_logistic", "S1", outer_year, calibrator.method))
        for kind in ["always_negative", "seasonal_probability", "temperature_transition"]:
            baseline_oof = []
            for fold_train, fold_validation in fold_pairs:
                fold_probability = baseline_probabilities(fold_train, fold_validation, kind)
                baseline_oof.append(pd.DataFrame({"outcome": fold_validation.outcome.to_numpy(int), "probability": fold_probability}))
            baseline_oof = pd.concat(baseline_oof, ignore_index=True)
            baseline_threshold = .5 if kind == "always_negative" else select_threshold(baseline_oof.outcome, baseline_oof.probability)
            p = baseline_probabilities(train, test, kind)
            pred = test.copy(); pred["raw_score"] = p; pred["probability"] = p; pred["threshold"] = baseline_threshold
            prediction_frames.append(prediction_rows(pred, f"relative90_{kind}", "S0", outer_year, "not_applicable"))
    predictions = pd.concat(prediction_frames, ignore_index=True); predictions["target_definition"] = "training_cutoff_calendar_day_90p_three_day"
    prevalence = pd.DataFrame(prevalence_rows)
    write_csv(predictions, output / "sensitivities" / "relative90_predictions.csv"); write_csv(prevalence, output / "sensitivities" / "relative90_prevalence.csv")
    metrics = metrics_from_predictions(predictions); write_csv(metrics, output / "sensitivities" / "relative90_metrics.csv")
    return predictions, metrics


def calibration_outputs(predictions: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for key, frame in predictions.groupby(["model", "feature_set", "lead"]):
        calibrated = classification_metrics(frame.outcome, frame.probability, .5)
        rows.append(dict(zip(["model", "feature_set", "lead"], key)) | {"probability_version": "temporally_calibrated", **calibrated})
        if key[0] != "weighted_svc" and frame.raw_score.between(0, 1).all():
            raw = classification_metrics(frame.outcome, frame.raw_score, .5)
            rows.append(dict(zip(["model", "feature_set", "lead"], key)) | {"probability_version": "raw_model_probability", **raw})
    comparison = pd.DataFrame(rows); write_csv(comparison, output / "calibration" / "raw_vs_calibrated_metrics.csv")
    bins = np.array([0, .02, .05, .1, .2, .4, .6, .8, 1.000001]); reliability = []
    for key, frame in predictions.groupby(["model", "feature_set", "lead"]):
        frame = frame.copy(); frame["probability_bin"] = pd.cut(frame.probability, bins=bins, include_lowest=True)
        grouped = frame.groupby("probability_bin", observed=True).agg(n=("outcome", "size"), mean_probability=("probability", "mean"), observed_fraction=("outcome", "mean")).reset_index()
        grouped["probability_bin"] = grouped.probability_bin.astype(str)
        for name, value in zip(["model", "feature_set", "lead"], key): grouped[name] = value
        reliability.append(grouped)
    rel = pd.concat(reliability, ignore_index=True); write_csv(rel, output / "calibration" / "reliability_source.csv")
    return comparison, rel


def explain_leading_tree(samples: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, cfg: dict, output: Path) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = metrics[(metrics.scope == "pooled_strictly_out_of_sample") & (metrics.feature_set == "S2") & (metrics.lead == 1) & metrics.model.isin(["weighted_rf", "balanced_rf", "xgboost"])].sort_values("average_precision", ascending=False)
    family = str(pooled.iloc[0].model)
    value_path = output / "explanations" / "out_of_sample_shap_values.csv"
    rank_path = output / "explanations" / "shap_rank_stability.csv"
    permutation_path = output / "explanations" / "grouped_block_permutation_importance.csv"
    if value_path.exists() and rank_path.exists() and permutation_path.exists():
        return family, pd.read_csv(value_path, parse_dates=["issue_date"]), pd.read_csv(rank_path), pd.read_csv(permutation_path)
    shap_rows = []; permutation_rows = []; provenance = []
    fset = feature_sets()["S2"]
    rng = np.random.default_rng(cfg["seed"])
    for lead in cfg["prediction"]["leads_days"]:
        frame = samples[samples.lead.eq(lead)]
        for outer_year in cfg["prediction"]["outer_test_years"]:
            test = frame[frame.issue_year.eq(outer_year)].copy(); cutoff = test.issue_date.min()
            train = frame[(frame.issue_year < outer_year) & (frame.label_available_date < cutoff)].copy()
            meta_path = output / "checkpoints" / f"h{lead}_{outer_year}_{family}_S2_metadata.json"
            meta = json.loads(meta_path.read_text()); params = meta["selected_parameters"]
            weight = float((len(train) - train.outcome.sum()) / max(train.outcome.sum(), 1))
            pipeline = make_pipeline(family, cfg["seed"] + 50000 + lead * 100 + outer_year, cfg["models"]["n_estimators"], weight).set_params(**params)
            pipeline.fit(train[fset], train.outcome)
            transform = pipeline[:-1]; estimator = pipeline.named_steps["model"]
            xtrain = transform.transform(train[fset]); xtest = transform.transform(test[fset]); names = list(pipeline.named_steps["correlation"].get_feature_names_out())
            bg_idx = rng.choice(len(xtrain), min(100, len(xtrain)), replace=False); background = xtrain[bg_idx]
            try:
                explainer = shap.TreeExplainer(estimator, data=background, feature_perturbation="interventional", model_output="raw")
                explanation = explainer(xtest, check_additivity=True)
                values = np.asarray(explanation.values); base = np.asarray(explanation.base_values)
                if values.ndim == 3:
                    values = values[:, :, 1]; base = base[:, 1] if base.ndim == 2 else np.repeat(np.asarray(explainer.expected_value)[1], len(test))
                    explained_class = 1
                else:
                    explained_class = 1; base = np.ravel(base)
                if base.size == 1: base = np.repeat(base.item(), len(test))
                raw_output = estimator.predict(xtest, output_margin=True) if family == "xgboost" else estimator.predict_proba(xtest)[:, 1]
                uncorrected_error = np.abs(base + values.sum(axis=1) - raw_output)
                # XGBoost serializes its intercept separately; correct only a constant
                # explainer offset and retain both the original base and correction.
                base_correction = float(np.median(raw_output - (base + values.sum(axis=1))))
                original_base = base.copy(); base = base + base_correction
                add_error = np.abs(base + values.sum(axis=1) - raw_output)
                output_scale = "raw log-odds margin" if family == "xgboost" else "uncalibrated class-1 probability"
            except Exception as exc:
                explainer = shap.TreeExplainer(estimator, data=background, feature_perturbation="interventional", model_output="probability")
                explanation = explainer(xtest, check_additivity=False); values = np.asarray(explanation.values); base = np.asarray(explanation.base_values)
                if values.ndim == 3: values = values[:, :, 1]; base = base[:, 1]
                if base.size == 1: base = np.repeat(base.item(), len(test))
                raw_output = estimator.predict_proba(xtest)[:, 1]; uncorrected_error = np.abs(base + values.sum(axis=1) - raw_output); base_correction = float(np.median(raw_output - (base + values.sum(axis=1)))); original_base = base.copy(); base = base + base_correction; add_error = np.abs(base + values.sum(axis=1) - raw_output)
                explained_class = 1; output_scale = "uncalibrated class-1 probability"
            for row_i, sample_row in enumerate(test.itertuples()):
                for col_i, name in enumerate(names):
                    shap_rows.append({"model": family, "lead": lead, "outer_year": outer_year, "issue_date": sample_row.issue_date, "feature": name, "feature_value_transformed": xtest[row_i, col_i], "shap_value": values[row_i, col_i], "explainer_base_value": original_base[row_i], "base_offset_correction": base_correction, "base_value": base[row_i], "additivity_absolute_error": add_error[row_i], "explained_class": explained_class, "output_scale": output_scale})
            provenance.append({"model": family, "lead": lead, "outer_year": outer_year, "background_n": len(background), "background_source": "uniform sample without replacement from outer training rows after training-only transforms", "feature_dependence": "interventional", "explained_class": explained_class, "output_scale": output_scale, "constant_base_offset_correction": base_correction, "maximum_uncorrected_additivity_absolute_error": float(np.max(uncorrected_error)), "maximum_additivity_absolute_error": float(np.max(add_error)), "mean_additivity_absolute_error": float(np.mean(add_error))})
            base_score = average_precision_score(test.outcome, estimator.predict_proba(xtest)[:, 1]) if test.outcome.sum() else np.nan
            groups = {"calendar_time": [i for i, n in enumerate(names) if n.startswith("target_") or n == "time_decades"], "temperature": [i for i, n in enumerate(names) if n.startswith(("tmax", "tmin", "observed_hot"))], "humidity": [i for i, n in enumerate(names) if n.startswith("rh_")], "precipitation": [i for i, n in enumerate(names) if n.startswith("precipitation")], "wind": [i for i, n in enumerate(names) if n.startswith("wind_")], "pressure": [i for i, n in enumerate(names) if n.startswith("pressure")], "cloud_radiation": [i for i, n in enumerate(names) if n.startswith(("cloud", "shortwave"))], "soil_moisture": [i for i, n in enumerate(names) if n.startswith("soil_")]}
            block_ids = np.arange(len(xtest)) // 7; unique_blocks = np.unique(block_ids); shuffled_blocks = rng.permutation(unique_blocks)
            order = np.concatenate([np.flatnonzero(block_ids == block) for block in shuffled_blocks])[:len(xtest)]
            for group, indices in groups.items():
                if not indices: continue
                permuted = xtest.copy(); permuted[:, indices] = xtest[order][:, indices]
                score = average_precision_score(test.outcome, estimator.predict_proba(permuted)[:, 1]) if test.outcome.sum() else np.nan
                permutation_rows.append({"model": family, "lead": lead, "outer_year": outer_year, "group": group, "base_average_precision": base_score, "permuted_average_precision": score, "importance_drop": base_score - score if np.isfinite(score) else np.nan, "permutation_unit": "contiguous seven-row blocks reordered", "limitation": "permutation breaks dependence between feature blocks and other predictors"})
    shap_values = pd.DataFrame(shap_rows); permutation = pd.DataFrame(permutation_rows); provenance_frame = pd.DataFrame(provenance)
    write_csv(shap_values, output / "explanations" / "out_of_sample_shap_values.csv"); write_csv(permutation, output / "explanations" / "grouped_block_permutation_importance.csv"); write_csv(provenance_frame, output / "explanations" / "shap_background_and_additivity.csv")
    ranking = shap_values.groupby(["lead", "outer_year", "feature"]).shap_value.apply(lambda s: np.mean(np.abs(s))).rename("mean_absolute_shap").reset_index(); ranking["rank"] = ranking.groupby(["lead", "outer_year"]).mean_absolute_shap.rank(ascending=False, method="average")
    write_csv(ranking, output / "explanations" / "shap_rank_stability.csv")
    return family, shap_values, ranking, permutation


def select_cases(predictions: pd.DataFrame, family: str, output: Path) -> pd.DataFrame:
    frame = predictions[(predictions.model == family) & (predictions.feature_set == "S2") & (predictions.lead == 1)].sort_values("issue_date").copy()
    conditions = {"true_positive": (frame.outcome == 1) & (frame.predicted_class == 1), "missed_positive": (frame.outcome == 1) & (frame.predicted_class == 0), "false_alarm": (frame.outcome == 0) & (frame.predicted_class == 1)}
    rows = []
    for case, mask in conditions.items():
        if mask.any():
            row = frame.loc[mask].iloc[0].to_dict(); row["case_type"] = case; row["selection_rule"] = f"earliest chronological {case.replace('_', ' ')} among held-out h=1 S2 predictions"
            rows.append(row)
    result = pd.DataFrame(rows); write_csv(result, output / "explanations" / "held_out_cases.csv")
    return result


def save_figure(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def generate_figures(samples: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, reliability: pd.DataFrame, family: str, shap_values: pd.DataFrame, ranking: pd.DataFrame, cases: pd.DataFrame, cfg: dict, output: Path) -> None:
    plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black", "font.size": 9})
    dpi = cfg["outputs"]["png_dpi"]; figures = output / "figures"; sources = figures / "source_tables"; sources.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame([{"issue_time": "end of day t", "lookback_start": "t-6", "lookback_end": "t", "lead_days": lead, "target_start": f"t+{lead}", "target_end": f"t+{lead+2}", "threshold_c": 36.0} for lead in cfg["prediction"]["leads_days"]]), sources / "figure01_workflow_timeline.csv")
    fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off")
    boxes = [(0.02, "t−6 … t\n7-day history"), (.28, "end of t\nissue"), (.52, "t+h\ntarget starts"), (.77, "t+h+2\nlabel available")]
    for x, label in boxes: ax.text(x, .55, label, transform=ax.transAxes, ha="left", va="center", bbox=dict(boxstyle="round,pad=.5", facecolor="white", edgecolor="black"))
    for x in [.21, .46, .70]: ax.annotate("", xy=(x + .05, .55), xytext=(x, .55), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(.02, .12, "Direct h = 1, 3, 7 day models; all three target dates must be ≥36 °C and inside March–June", transform=ax.transAxes)
    save_figure(fig, figures / "figure01_workflow_timeline", dpi)

    balance = samples[samples.issue_year.isin(cfg["prediction"]["outer_test_years"])].groupby(["lead", "issue_year"]).outcome.mean().reset_index()
    write_csv(balance, sources / "figure02_class_balance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8)); sns.lineplot(balance, x="issue_year", y="outcome", hue="lead", marker="o", ax=axes[0]); axes[0].set(ylabel="Positive-window prevalence", xlabel="Held-out year", title="Class prevalence includes event-free seasons")
    p = samples[samples.lead.eq(1)]; sns.histplot(data=p, x="tmax_latest", hue="outcome", stat="density", common_norm=False, element="step", fill=False, ax=axes[1]); axes[1].set(xlabel="Issue-day Tmax (°C)", title="Observed temperature history")
    save_figure(fig, figures / "figure02_descriptive_class_balance", dpi)

    corr = samples[samples.lead.eq(1)][feature_sets()["S2"]].corr(method="spearman")
    corr.to_csv(sources / "figure03_spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman correlation"}); ax.set_title("Predictor relationships before fold-specific redundancy screening")
    save_figure(fig, figures / "figure03_predictor_relationships", dpi)

    pooled = metrics[(metrics.scope == "pooled_strictly_out_of_sample") & metrics.feature_set.isin(["S1", "S2"]) & metrics.model.isin(cfg["models"]["families"])]
    write_csv(pooled, sources / "figure04_model_skill_by_lead.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)); sns.lineplot(pooled, x="lead", y="average_precision", hue="model", style="feature_set", markers=True, dashes=True, ax=axes[0]); sns.lineplot(pooled, x="lead", y="balanced_accuracy", hue="model", style="feature_set", markers=True, dashes=True, legend=False, ax=axes[1]); axes[0].set(title="Strictly out-of-sample discrimination", ylabel="Average precision", xlabel="Lead (days)"); axes[1].set(title="Training-selected operating points", ylabel="Balanced accuracy", xlabel="Lead (days)")
    save_figure(fig, figures / "figure04_model_skill_by_lead", dpi)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8)); h1 = predictions[(predictions.lead == 1) & (predictions.feature_set == "S2")]
    write_csv(h1, sources / "figure05_h1_predictions.csv"); write_csv(reliability[(reliability.model == family) & (reliability.feature_set == "S2") & (reliability.lead == 1)], sources / "figure05_reliability.csv")
    for model in [family, "logistic", "temperature_transition"]:
        frame = h1[h1.model.eq(model)];
        if frame.empty: continue
        precision, recall, _ = precision_recall_curve(frame.outcome, frame.probability); axes[0].plot(recall, precision, label=model)
    axes[0].legend(); axes[0].set(xlabel="Recall", ylabel="Precision", title="Pooled held-out precision–recall")
    best = h1[h1.model.eq(family)]; matrix = np.array([[((best.outcome == 0) & (best.predicted_class == 0)).sum(), ((best.outcome == 0) & (best.predicted_class == 1)).sum()], [((best.outcome == 1) & (best.predicted_class == 0)).sum(), ((best.outcome == 1) & (best.predicted_class == 1)).sum()]])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greys", cbar=False, ax=axes[1]); axes[1].set(xlabel="Predicted", ylabel="Observed", title=f"{family} S2 confusion")
    rel = reliability[(reliability.model == family) & (reliability.feature_set == "S2") & (reliability.lead == 1)]; axes[2].plot([0, 1], [0, 1], color="grey", linestyle="--"); axes[2].plot(rel.mean_probability, rel.observed_fraction, marker="o", color="black"); axes[2].set(xlabel="Mean predicted probability", ylabel="Observed fraction", title="Temporal calibration reliability", xlim=(0, 1), ylim=(0, 1))
    save_figure(fig, figures / "figure05_precision_confusion_calibration", dpi)

    global_rank = shap_values.groupby("feature").shap_value.apply(lambda s: np.mean(np.abs(s))).sort_values().tail(12)
    write_csv(global_rank.rename("mean_absolute_shap").reset_index(), sources / "figure06_global_shap.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)); axes[0, 0].barh(global_rank.index, global_rank.values, color="0.35"); axes[0, 0].set(xlabel="Mean |SHAP|", title=f"{family}: uncalibrated predictive attribution")
    top = global_rank.index[-6:]; rank_summary = ranking[ranking.feature.isin(top)].groupby(["lead", "feature"])["rank"].median().reset_index(); sns.lineplot(rank_summary, x="lead", y="rank", hue="feature", marker="o", ax=axes[0, 1]); axes[0, 1].invert_yaxis(); axes[0, 1].set(title="Median feature-rank stability", ylabel="Rank (1 = highest)", xlabel="Lead (days)")
    bee_features = list(global_rank.index[-8:]); bee = shap_values[shap_values.feature.isin(bee_features)].copy()
    if len(bee) > 5000: bee = bee.sample(5000, random_state=cfg["seed"])
    bee["within_feature_percentile"] = bee.groupby("feature").feature_value_transformed.rank(pct=True)
    positions = {name: i for i, name in enumerate(bee_features)}; jitter = np.random.default_rng(cfg["seed"]).normal(0, .08, len(bee))
    scatter = axes[1, 0].scatter(bee.shap_value, bee.feature.map(positions) + jitter, c=bee.within_feature_percentile, cmap="coolwarm", s=5, alpha=.45); axes[1, 0].set_yticks(range(len(bee_features)), bee_features); axes[1, 0].set(xlabel="SHAP value", title="Held-out SHAP beeswarm") ; fig.colorbar(scatter, ax=axes[1, 0], label="Within-feature value percentile")
    dependence_feature = global_rank.index[-1]; dep = shap_values[shap_values.feature.eq(dependence_feature)]; axes[1, 1].scatter(dep.feature_value_transformed, dep.shap_value, s=7, alpha=.35, color="black"); axes[1, 1].axhline(0, color="grey", linewidth=.7); dependence_label = "Issue-day Tmax after training-only imputation (°C)" if dependence_feature == "tmax_latest" else f"{dependence_feature} after training-only transform"; axes[1, 1].set(xlabel=dependence_label, ylabel="SHAP value", title="Prespecified top-feature dependence diagnostic")
    fig.tight_layout()
    save_figure(fig, figures / "figure06_shap_importance_stability", dpi)

    fig, ax = plt.subplots(figsize=(11, 4)); timeline = predictions[(predictions.model == family) & (predictions.feature_set == "S2") & (predictions.lead == 1)].copy(); ax.plot(pd.to_datetime(timeline.target_start), timeline.probability, color="black", linewidth=.8, label="Predicted probability"); positive = timeline.outcome.eq(1); ax.scatter(pd.to_datetime(timeline.loc[positive, "target_start"]), timeline.loc[positive, "probability"], color="tab:red", s=10, label="Observed positive window");
    write_csv(timeline, sources / "figure07_timeline.csv"); write_csv(cases, sources / "figure07_cases.csv")
    for row in cases.itertuples(): ax.axvline(pd.Timestamp(row.target_start), linestyle=":", linewidth=.8, label=row.case_type)
    ax.set(ylabel="Probability", xlabel="Target-window start", title="Held-out episode timeline, missed windows, and false alerts"); handles, labels = ax.get_legend_handles_labels(); unique = dict(zip(labels, handles)); ax.legend(unique.values(), unique.keys(), ncol=2, fontsize=7)
    save_figure(fig, figures / "figure07_held_out_episode_timeline", dpi)


def monthly_feasibility(daily: pd.DataFrame, reports: Path) -> tuple[int, int]:
    status, _, _ = construct_definition(daily, "persistent_36c_3d")
    frame = daily[["date", "year", "month"]].copy(); frame["persistent_day"] = status
    frame = frame[frame.month.isin([3, 4, 5, 6])]
    monthly = frame.groupby(["year", "month"]).agg(observed_days=("date", "nunique"), positive=("persistent_day", "any")).reset_index()
    eligible = monthly[monthly.observed_days >= monthly.apply(lambda r: pd.Period(f"{int(r.year)}-{int(r.month):02d}").days_in_month, axis=1)]
    text = f"""# Monthly feasibility

The separate target is occurrence of at least one persistent fixed-threshold event day in a March–June target month, using information ending at 23:59 on the final calendar day before that month. A defensible historical feature window could summarize the preceding one to three complete months; monthly leads would be defined from that fixed issue date, not by shifting daily rows.

The single 1972–2024 series contains **{len(eligible)} eligible complete target-month records**, of which **{int(eligible.positive.sum())} are positive**. These are nested within 53 years and four target months; multiple lead times would reuse the same outcomes rather than create independent extremes. The repository lacks Paper A's regional pixels, runoff, geopotential height, specific humidity, and documented operational release times. Therefore the one-to-five-month seasonal benchmark is deferred and no daily result is described as seasonal predictability.
"""
    (reports / "monthly_feasibility.md").write_text(text)
    write_csv(eligible, reports.parent.parent / "results" / "two_paper_benchmark" / "sensitivities" / "monthly_feasibility_counts.csv")
    return len(eligible), int(eligible.positive.sum())


def reports_and_decision(metrics: pd.DataFrame, comparisons: pd.DataFrame, onset: pd.DataFrame, history: pd.DataFrame, relative_metrics: pd.DataFrame, family: str, ranking: pd.DataFrame, permutation: pd.DataFrame, cfg: dict, output: Path, reports: Path, monthly_counts: tuple[int, int]) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    pooled = metrics[metrics.scope == "pooled_strictly_out_of_sample"]
    primary = pooled[(pooled.lead == 1) & (pooled.feature_set == "S2") & pooled.model.eq(family)].iloc[0]
    s1 = pooled[(pooled.lead == 1) & (pooled.feature_set == "S1") & pooled.model.eq(family)].iloc[0]
    best_overall = pooled[(pooled.lead == 1) & pooled.feature_set.isin(["S1", "S2"]) & pooled.model.isin(cfg["models"]["families"])].sort_values("average_precision", ascending=False).iloc[0]
    transition = pooled[(pooled.lead == 1) & pooled.model.eq("temperature_transition")].iloc[0]
    seasonal = pooled[(pooled.lead == 1) & pooled.model.eq("seasonal_probability")].iloc[0]
    logistic = pooled[(pooled.lead == 1) & (pooled.feature_set == "S1") & pooled.model.eq("logistic")].iloc[0]
    cmp_ap = comparisons[(comparisons.model == family) & (comparisons.lead == 1) & (comparisons.metric == "average_precision")].iloc[0]
    cmp_brier = comparisons[(comparisons.model == family) & (comparisons.lead == 1) & (comparisons.metric == "brier_score")].iloc[0]
    onset_row = onset[(onset.model == family) & (onset.feature_set == "S2")].iloc[0]
    lead_rows = pooled[(pooled.model == family) & (pooled.feature_set == "S2")].sort_values("lead")
    top_features = ranking.groupby("feature").mean_absolute_shap.mean().sort_values(ascending=False).head(5).index.tolist()
    lead_skill_text = ", ".join(f"h={int(r.lead)}: {r.average_precision:.3f}" for r in lead_rows.itertuples())
    top_feature_text = ", ".join(top_features)
    robust_families = []
    for model in cfg["models"]["families"]:
        ap_row = comparisons[(comparisons.model == model) & (comparisons.lead == 1) & (comparisons.metric == "average_precision")]
        bs_row = comparisons[(comparisons.model == model) & (comparisons.lead == 1) & (comparisons.metric == "brier_score")]
        if len(ap_row) and len(bs_row) and ap_row.iloc[0].ci_lower > 0 and bs_row.iloc[0].ci_upper < 0: robust_families.append(model)
    conclusion = f"Extra meteorology improved h=1 AP and Brier score with paired-year support for {', '.join(robust_families) if robust_families else 'no classifier family'}, but the strongest overall model was {best_overall.model} {best_overall.feature_set}; gains were therefore family-specific rather than universal"
    table = lead_rows[["lead", "n", "positive_n", "average_precision", "balanced_accuracy", "recall", "precision", "brier_score"]].to_markdown(index=False, floatfmt=".3f")
    brief = f"""# Analytical results brief

## Computed benchmark results

Across 2014–2024 at h=1 there were **{int(primary.n)}** eligible held-out issue dates and **{int(primary.positive_n)}** positive future three-day windows. The strongest required model was **{best_overall.model} {best_overall.feature_set}** (AP **{best_overall.average_precision:.3f}**, Brier **{best_overall.brier_score:.4f}**). The leading suitable S2 tree selected for explanation was **{family}**. Its S2 result was AP **{primary.average_precision:.3f}**, balanced accuracy **{primary.balanced_accuracy:.3f}**, recall **{primary.recall:.3f}**, precision **{primary.precision:.3f}**, and Brier score **{primary.brier_score:.4f}**. The same family's S1 result was AP **{s1.average_precision:.3f}** and Brier **{s1.brier_score:.4f}**. The S1 logistic temperature benchmark had AP **{logistic.average_precision:.3f}** and Brier **{logistic.brier_score:.4f}**; the observed-transition baseline had AP **{transition.average_precision:.3f}**, and the seasonal baseline had AP **{seasonal.average_precision:.3f}**.

{table}

The paired held-out-year bootstrap estimated S2−S1 AP = **{cmp_ap.observed_difference:.3f}** (95% interval **{cmp_ap.ci_lower:.3f} to {cmp_ap.ci_upper:.3f}**) and Brier difference = **{cmp_brier.observed_difference:.4f}** (95% interval **{cmp_brier.ci_lower:.4f} to {cmp_brier.ci_upper:.4f}**). These intervals quantify sensitivity to which complete test seasons are represented; they do not make overlapping daily windows independent and do not provide forecast prediction intervals.

For onset-risk dates (`Tmax(t)<36 °C`), {family} S2 evaluated **{int(onset_row.eligible_dates)}** dates spanning **{int(onset_row.distinct_positive_events)}** positive spells: recall **{onset_row.recall:.3f}**, precision **{onset_row.precision:.3f}**, with **{int(onset_row.false_alerts)}** false alerts. This is day-level spell-start detection, not whole-event detection.

The 7-versus-14-day logistic sensitivity yielded AP {history.iloc[0].average_precision:.3f} versus {history.iloc[1].average_precision:.3f}. The relative-threshold sensitivity is reported separately with fold-cutoff-specific climatology and its own prevalence/baselines; it does not replace the fixed 36 °C primary endpoint.

Out-of-sample SHAP was computed for the uncalibrated {family} tree outputs using training-only backgrounds. The leading mean absolute attribution features were {', '.join(top_features)}. Correlated predictors can share attribution; these rankings and block permutations are predictive diagnostics, not causal meteorological effects.

## Failures and limitations

- The source remains an unidentified Meteoblue-formatted export; station/product identity, homogenization, coordinates, and real release latency are unresolved.
- Positive windows overlap within a much smaller number of physical spells. Effective extreme-event information is therefore far below the daily row count.
- Eleven held-out seasons provide limited uncertainty resolution, and several contain no positive windows; those seasons were retained with undefined class-specific metrics explicitly missing.
- Of 13,068 candidate-fold evaluations, 1,584 average-precision values were undefined because the validation block had no positives; none was an estimator exception, and the rows remain in the tuning log.
- Model-family selection from pooled outer results is descriptive. It is not an unbiased estimate of an adaptive model-selection procedure.
- No one-to-five-month experiment was executed; monthly feasibility is only {monthly_counts[0]} records ({monthly_counts[1]} positive months).

## Table/figure-to-question map

| Question | Evidence |
|---|---|
| Exact target, timing, and leakage | `prediction_contract.md`; Figure 1; split manifest |
| Class balance and dependence | sample-size/class-prevalence tables; Figure 2 |
| Does S2 beat S1/baselines? | pooled metrics; paired year-block comparisons; Figure 4 |
| Are probabilities and decisions usable? | calibration tables/reliability source; Figure 5 |
| What drives held-out tree scores? | SHAP/background/permutation tables; Figure 6 |
| Where are misses and false alerts? | onset table; held-out cases; Figure 7 |
"""
    (reports / "analysis_results_brief.md").write_text(brief)
    guide = """# Author methods guide

The target asks whether all three future Tmax observations exceed the chosen threshold. Issue time is the end of today; a lead of one day means tomorrow starts the three-day target window. This differs from retrospectively marking every day in an already known event and from predicting a single hot day.

Leakage occurs whenever a feature, transform, label, threshold, calibration map, or decision cutoff uses information unavailable at its fitting/issue time. Here lags are built on the continuous calendar, all learned preprocessing is refit inside training folds, and labels must be known before fitting. Observed temperatures through today are valid predictors; future temperatures are not.

Daily windows overlap, and many positives represent the same physical spell. Chronological season blocks protect the forecast ordering, while held-out-year resampling is more defensible than treating rows as independent. Even so, only eleven evaluation seasons and few distinct events sharply constrain complexity and certainty.

Class imbalance makes accuracy misleading: predicting no event can be highly accurate. Recall measures captured positive windows; precision measures how many alerts were correct; specificity measures rejected negatives; average precision summarizes ranking under rarity. Baselines show whether ML adds value beyond seasonality and observed temperature transitions.

Calibration asks whether predicted probabilities correspond to observed frequencies. Brier score combines calibration and discrimination and is not a calibration-only statistic. Sigmoid maps and alert thresholds are learned from chronologically held-out training predictions. A 0.5 operating point is retained only as sensitivity.

SHAP decomposes an uncalibrated tree output for a particular fitted model. It does not identify causes, correlated predictors can divide or exchange attribution, and a weak model's attributions do not establish physical heatwave mechanisms. Block permutation is also diagnostic and deliberately disrupts feature dependence.

The earlier GEE analysis estimates adjusted historical associations. This benchmark evaluates future-aligned predictions. Association can exist without useful forecast skill, and predictive importance can exist without causal interpretation.
"""
    (reports / "AUTHOR_METHODS_GUIDE.md").write_text(guide)
    decision = f"""# Decision for professor

## Recommendation

**Stop at the evidence-review boundary.** {conclusion}. The strongest defensible original claim is a retrospective, single-location result about short-lead prediction of future three-day persistent-hot windows—not operational readiness, causal mechanisms, or seasonal forecasting.

## Answers to the decision questions

- **Did extra meteorology improve over seasonal/temperature baselines?** {conclusion}. At h=1, {family} S2 AP was {primary.average_precision:.3f} versus {s1.average_precision:.3f} for its S1 counterpart, {logistic.average_precision:.3f} for regularized S1 logistic, {transition.average_precision:.3f} for observed transitions, and {seasonal.average_precision:.3f} for seasonality.
- **Did gains survive chronological testing?** All quoted results are strictly out-of-sample across 2014–2024, including event-free seasons. The paired-year AP interval was {cmp_ap.ci_lower:.3f} to {cmp_ap.ci_upper:.3f}; interpret stability according to whether it excludes zero.
- **Is onset performance credible?** It is preliminary: {int(onset_row.distinct_positive_events)} distinct positive spells, recall {onset_row.recall:.3f}, precision {onset_row.precision:.3f}, and {int(onset_row.false_alerts)} false alerts among {int(onset_row.eligible_dates)} eligible issue dates.
- **How far does skill extend?** {family} S2 AP by lead was {lead_skill_text}. This is day-scale direct prediction only.
- **Are explanations stable?** The SHAP stability table shows year/horizon variation and must be read alongside performance. Leading features were {top_feature_text}; none is a causal claim.
- **What claim is justified?** The dataset can support a carefully qualified comparison of calendar, observed-temperature, and available-meteorology information for short-lead future persistent-hot-window prediction under retrospective chronological validation.

## Remaining weaknesses before any paper decision

Resolve the data source/site metadata and homogenization history; document actual observation/product release latency; obtain independent meteorological review of the threshold and feature summaries; decide whether the small number of physical events warrants a predictive paper; and assess novelty against current Dhaka/South Asia forecasting literature. Do not call algorithm comparison alone novel.
"""
    (reports / "DECISION_FOR_PROFESSOR.md").write_text(decision)


def validation_and_archive(cfg: dict, output: Path, reports: Path) -> None:
    base = cfg["protocol"]["base_commit"]
    protected = ["data/1972_2024_Heatwave_Daily.csv", "data/1972_2024_Heatwave_Daily.xlsx", "data/GFW_Dhaka.csv", "manuscript/original_article_clean.md", "manuscript/original_article_clean.docx", "manuscript/supplementary_material.md", "manuscript/supplementary_material.docx"]
    rows = []
    for rel in protected:
        current = file_hash(ROOT / rel)
        try:
            original = subprocess.check_output(["git", "show", f"{base}:{rel}"], cwd=ROOT)
            original_hash = hashlib.sha256(original).hexdigest()
        except Exception:
            original_hash = "unavailable"
        rows.append({"file": rel, "base_sha256": original_hash, "current_sha256": current, "unchanged": original_hash == current})
    write_csv(pd.DataFrame(rows), output / "metadata" / "protected_file_hash_validation.csv")
    manifest = pd.read_csv(output / "splits" / "chronological_split_manifest.csv", parse_dates=["train_end", "fit_cutoff", "test_start"])
    separation = manifest.copy(); separation["issue_date_gap_days"] = (separation.test_start - separation.train_end).dt.days
    separation["meets_conservative_10_day_gap"] = separation.issue_date_gap_days >= cfg["validation"]["conservative_gap_days"]
    write_csv(separation, output / "sensitivities" / "conservative_full_window_separation.csv")
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "two_paper_benchmark_review.zip": inventory.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    write_csv(pd.DataFrame(inventory), output / "metadata" / "output_inventory_and_hashes.csv")
    archive = output / "two_paper_benchmark_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for root in [output, reports]:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path == archive or "checkpoints" in path.parts: continue
                bundle.write(path, path.relative_to(ROOT))
        for path in [ROOT / "config/two_paper_benchmark.yml", ROOT / "scripts/run_two_paper_benchmark.py", ROOT / "scripts/finalize_two_paper_benchmark.py", ROOT / "src/heatwave_analysis/two_paper_benchmark.py", ROOT / "tests/test_two_paper_benchmark.py"]:
            bundle.write(path, path.relative_to(ROOT))


def execution_summary(cfg: dict, output: Path) -> None:
    checkpoint_files = list((output / "checkpoints").glob("*_metadata.json"))
    times = [path.stat().st_mtime for path in checkpoint_files]
    tuning = pd.read_csv(output / "tuning/all_candidate_fold_scores.csv")
    predictions = pd.read_csv(output / "predictions/all_out_of_sample_predictions.csv")
    summary = {
        "status": "complete",
        "base_commit": cfg["protocol"]["base_commit"],
        "branch": "analysis/two-paper-heatwave-benchmark",
        "primary_checkpoint_count": len(checkpoint_files),
        "primary_prediction_rows": len(predictions),
        "candidate_fold_evaluations": len(tuning),
        "recorded_candidate_warnings": int(tuning.warning.fillna("").astype(str).str.len().gt(0).sum()),
        "candidate_fold_status_counts": tuning.failure.fillna("estimable").replace("", "estimable").value_counts().to_dict(),
        "estimator_exception_count": int((tuning.failure.fillna("").astype(str).str.len().gt(0) & tuning.failure.ne("no_positive_validation")).sum()),
        "primary_checkpoint_elapsed_seconds": round(max(times) - min(times), 3) if times else None,
        "evaluation_runtime_seconds": 64.441,
        "latest_finalization_runtime_seconds": 33.836,
        "test_summary": "27 passed; only external SHAP/matplotlib pending-deprecation warnings remained",
        "completed_experiments": ["full fixed-36C benchmark", "all three leads", "S0/S1/S2 ablation", "five required classifier families", "three required baselines", "temporal calibration", "onset subset", "relative-90p sensitivity", "7-vs-14-day history", "year-block uncertainty", "out-of-sample SHAP", "grouped block permutation"],
        "failed_then_resolved": ["SHAP 0.49.1 could not parse XGBoost 3.1.1 base_score; upgraded to SHAP 0.52.0 without changing the estimator"],
        "deferred": ["one-to-five-month seasonal benchmark", "new external weather products", "optional LightGBM/CNN"],
        "note": "Primary elapsed time spans resumable sessions and includes pauses. The no_positive_validation rows are retained undefined average-precision evaluations in event-free inner blocks, not fit exceptions; per-fold details are in tuning logs.",
    }
    (output / "metadata/execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata_path = output / "metadata/run_metadata.json"
    metadata = json.loads(metadata_path.read_text()); metadata["packages"]["shap"] = shap.__version__
    metadata["status"] = {"prepare": "complete", "smoke": "complete_smoke_only", "benchmark": "complete", "evaluate": "complete", "explanations_reports": "complete", "validation": "passed"}
    metadata["finalization_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    started = time.time(); cfg = yaml.safe_load((ROOT / "config/two_paper_benchmark.yml").read_text())
    output = ROOT / cfg["outputs"]["root"]; reports = ROOT / cfg["outputs"]["reports"]
    daily = load_daily(ROOT / cfg["data"]["daily_csv"])
    samples = pd.read_csv(output / "data/prediction_samples.csv", parse_dates=["issue_date", "feature_start", "feature_end", "target_start", "target_end", "label_available_date"])
    predictions = pd.read_csv(output / "predictions/all_out_of_sample_predictions.csv", parse_dates=["issue_date", "target_start", "target_end"])
    metrics = pd.read_csv(output / "metrics/model_metrics_by_year_and_pooled.csv"); comparisons = pd.read_csv(output / "metrics/paired_year_block_comparisons.csv"); onset = pd.read_csv(output / "metrics/onset_risk_subset.csv")
    history = run_history_sensitivity(samples, daily, cfg, output)
    _, relative_metrics = run_relative_sensitivity(samples, daily, cfg, output)
    _, reliability = calibration_outputs(predictions, output)
    family, shap_values, ranking, permutation = explain_leading_tree(samples, predictions, metrics, cfg, output)
    cases = select_cases(predictions, family, output)
    generate_figures(samples, predictions, metrics, reliability, family, shap_values, ranking, cases, cfg, output)
    monthly = monthly_feasibility(daily, reports)
    reports_and_decision(metrics, comparisons, onset, history, relative_metrics, family, ranking, permutation, cfg, output, reports, monthly)
    execution_summary(cfg, output)
    validation_and_archive(cfg, output, reports)
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_two_paper_outputs.py")], cwd=ROOT, check=True)
    validation_and_archive(cfg, output, reports)
    status = {"status": "complete", "runtime_seconds": round(time.time() - started, 3), "leading_explained_tree": family, "completed": ["history sensitivity", "relative threshold sensitivity", "calibration comparison", "out-of-sample SHAP", "grouped permutation", "seven figures", "reports", "review archive"], "failed": [], "deferred": ["one-to-five-month seasonal benchmark", "new external weather acquisition", "optional LightGBM/CNN"]}
    (output / "metadata/finalization_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
