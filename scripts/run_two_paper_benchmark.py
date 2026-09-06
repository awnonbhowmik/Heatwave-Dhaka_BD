#!/usr/bin/env python3
"""Single-command execution of the two-paper future-hot-window benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/two-paper-mpl")

import imblearn
import matplotlib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import xgboost
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatwave_analysis.climatology import calendar_day_threshold
from heatwave_analysis.data_io import load_daily, source_hashes
from heatwave_analysis.quality_control import quality_findings
from heatwave_analysis.two_paper_benchmark import (
    S0_FEATURES,
    assert_prediction_provenance,
    baseline_probabilities,
    classification_metrics,
    construct_prediction_samples,
    feature_dictionary,
    feature_sets,
    inner_year_splits,
    make_pipeline,
    metrics_from_predictions,
    prediction_rows,
    select_threshold,
    tune_outer_model,
    year_block_bootstrap,
)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_meta(meta: dict) -> dict:
    return {key: value for key, value in meta.items() if key != "calibrator"}


def prepare(cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = output / "data"; descriptive_dir = output / "descriptive"
    data_dir.mkdir(parents=True, exist_ok=True); descriptive_dir.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / cfg["data"]["daily_csv"]
    daily = load_daily(raw_path)
    samples = construct_prediction_samples(
        daily,
        leads=cfg["prediction"]["leads_days"],
        lookback_days=cfg["prediction"]["lookback_days"],
        threshold=cfg["prediction"]["fixed_threshold_c"],
        months=cfg["prediction"]["hot_season_months"],
    )
    assert_prediction_provenance(samples)
    write_csv(samples, data_dir / "prediction_samples.csv")
    write_csv(feature_dictionary(), data_dir / "predictor_dictionary.csv")
    hashes = source_hashes(ROOT / "data")
    (data_dir / "source_data_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    q = quality_findings(daily)
    expected = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
    audit = pd.DataFrame([
        {"quantity": "raw_rows", "value": len(daily)},
        {"quantity": "unique_dates", "value": daily.date.nunique()},
        {"quantity": "duplicate_dates", "value": int(daily.date.duplicated().sum())},
        {"quantity": "missing_calendar_dates", "value": len(expected.difference(daily.date))},
        {"quantity": "earliest_date", "value": daily.date.min().date().isoformat()},
        {"quantity": "latest_date", "value": daily.date.max().date().isoformat()},
        {"quantity": "unresolved_air_temperature_source", "value": "Meteoblue-formatted export; station/product identity unresolved"},
    ])
    write_csv(audit, data_dir / "raw_date_and_provenance_audit.csv")
    (data_dir / "quality_findings.json").write_text(json.dumps(q, indent=2, default=str) + "\n")
    class_balance = samples.groupby(["lead", "issue_year"]).agg(eligible_windows=("outcome", "size"), positive_windows=("outcome", "sum"), distinct_events=("associated_event_id", "nunique")).reset_index()
    class_balance["prevalence"] = class_balance.positive_windows / class_balance.eligible_windows
    write_csv(class_balance, descriptive_dir / "class_prevalence_by_year.csv")
    counts = samples.groupby("lead").agg(eligible_windows=("outcome", "size"), positive_windows=("outcome", "sum"), distinct_events=("associated_event_id", "nunique"), years=("issue_year", "nunique")).reset_index()
    positive_years = samples.groupby("lead").apply(lambda f: f.groupby("issue_year").outcome.sum().gt(0).sum(), include_groups=False).rename("event_positive_years").reset_index()
    write_csv(counts.merge(positive_years, on="lead"), data_dir / "sample_size_levels.csv")
    features = feature_sets()["S2"]
    desc = samples[features].describe(percentiles=[.05, .25, .5, .75, .95]).T.reset_index(names="feature")
    write_csv(desc, descriptive_dir / "predictor_distributions.csv")
    primary = samples[samples.lead.eq(cfg["prediction"]["onset_primary_lead"])]
    primary[features].corr(method="pearson").to_csv(descriptive_dir / "pearson_correlations.csv")
    primary[features].corr(method="spearman").to_csv(descriptive_dir / "spearman_correlations.csv")
    scatter = primary[["issue_date", "outcome", "tmax_latest", "tmax_mean7", "rh_mean_latest", "soil_moisture_mean_latest"]].copy()
    write_csv(scatter, descriptive_dir / "selected_scatterplot_source.csv")
    return daily, samples


def baseline_oof_threshold(train: pd.DataFrame, kind: str, folds: int) -> float:
    rows = []
    for split in inner_year_splits(train, folds=folds):
        tr = train.loc[split["train_index"]]; va = train.loc[split["validation_index"]]
        p = baseline_probabilities(tr, va, kind)
        rows.append(pd.DataFrame({"outcome": va.outcome.to_numpy(int), "probability": p}))
    oof = pd.concat(rows, ignore_index=True)
    return select_threshold(oof.outcome, oof.probability)


def smoke_test(samples: pd.DataFrame, cfg: dict, output: Path) -> None:
    year = max(cfg["prediction"]["outer_test_years"]); lead = cfg["prediction"]["leads_days"][0]
    frame = samples[samples.lead.eq(lead)].copy()
    test = frame[frame.issue_year.eq(year)]
    cutoff = test.issue_date.min()
    train = frame[(frame.issue_year < year) & (frame.label_available_date < cutoff)]
    tuning, pred, meta, _ = tune_outer_model(train, test, feature_sets()["S1"], "logistic", cfg["seed"], candidates=2, inner_folds=2, n_estimators=20)
    directory = output / "smoke_test"; directory.mkdir(parents=True, exist_ok=True)
    write_csv(tuning, directory / "tuning_log.csv")
    write_csv(prediction_rows(pred, "logistic", "S1", year, meta["calibration"]), directory / "predictions.csv")
    (directory / "status.json").write_text(json.dumps({"status": "smoke_only_not_full_analysis", "outer_year": year, "lead": lead, "candidates": 2, "metadata": serialize_meta(meta)}, indent=2, default=str) + "\n")


def run_primary(samples: pd.DataFrame, cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint = output / "checkpoints"; checkpoint.mkdir(parents=True, exist_ok=True)
    split_rows = []; selection_rows = []
    fsets = feature_sets()
    for lead in cfg["prediction"]["leads_days"]:
        frame = samples[samples.lead.eq(lead)].copy()
        for outer_year in cfg["prediction"]["outer_test_years"]:
            test = frame[frame.issue_year.eq(outer_year)].copy()
            if test.empty: continue
            cutoff = test.issue_date.min()
            train = frame[(frame.issue_year < outer_year) & (frame.label_available_date < cutoff)].copy()
            if train.empty: continue
            split_rows.append({"level": "outer", "lead": lead, "outer_year": outer_year, "inner_fold": np.nan, "train_start": train.issue_date.min(), "train_end": train.issue_date.max(), "fit_cutoff": cutoff, "test_start": test.issue_date.min(), "test_end": test.issue_date.max(), "train_n": len(train), "train_positive_n": int(train.outcome.sum()), "test_n": len(test), "test_positive_n": int(test.outcome.sum()), "labels_available_before_fit": bool((train.label_available_date < cutoff).all()), "event_free_test_season": bool(test.outcome.sum() == 0)})
            for split in inner_year_splits(train, folds=cfg["validation"]["inner_folds"]):
                tr = train.loc[split["train_index"]]; va = train.loc[split["validation_index"]]
                split_rows.append({"level": "inner", "lead": lead, "outer_year": outer_year, "inner_fold": split["inner_fold"], "train_start": tr.issue_date.min(), "train_end": tr.issue_date.max(), "fit_cutoff": split["validation_start"], "test_start": va.issue_date.min(), "test_end": va.issue_date.max(), "train_n": len(tr), "train_positive_n": int(tr.outcome.sum()), "test_n": len(va), "test_positive_n": int(va.outcome.sum()), "labels_available_before_fit": bool((tr.label_available_date < split["validation_start"]).all()), "event_free_test_season": bool(va.outcome.sum() == 0)})
            for kind in ["always_negative", "seasonal_probability", "temperature_transition"]:
                stem = f"h{lead}_{outer_year}_{kind}_S0"
                path = checkpoint / f"{stem}_predictions.csv"
                if not path.exists():
                    threshold = 0.5 if kind == "always_negative" else baseline_oof_threshold(train, kind, cfg["validation"]["inner_folds"])
                    p = baseline_probabilities(train, test, kind)
                    pred = test.copy(); pred["raw_score"] = p; pred["probability"] = p; pred["threshold"] = threshold
                    write_csv(prediction_rows(pred, kind, "S0", outer_year, "not_applicable_training_only_probability"), path)
            combinations = [("s0_logistic", "logistic", "S0")] + [(family, family, feature_set) for family in cfg["models"]["families"] for feature_set in ["S1", "S2"]]
            for model_name, family, feature_set in combinations:
                stem = f"h{lead}_{outer_year}_{model_name}_{feature_set}"
                pred_path = checkpoint / f"{stem}_predictions.csv"; tune_path = checkpoint / f"{stem}_tuning.csv"; meta_path = checkpoint / f"{stem}_metadata.json"
                if pred_path.exists() and tune_path.exists() and meta_path.exists():
                    continue
                tuning, pred, meta, _ = tune_outer_model(
                    train, test, fsets[feature_set], family,
                    seed=cfg["seed"] + lead * 100 + outer_year,
                    candidates=cfg["validation"]["random_search_candidates"],
                    inner_folds=cfg["validation"]["inner_folds"],
                    n_estimators=cfg["models"]["n_estimators"],
                )
                tuning.insert(0, "outer_year", outer_year); tuning.insert(1, "lead", lead); tuning.insert(2, "feature_set", feature_set); tuning.insert(3, "model", model_name)
                write_csv(tuning, tune_path)
                write_csv(prediction_rows(pred, model_name, feature_set, outer_year, meta["calibration"]), pred_path)
                meta_path.write_text(json.dumps(serialize_meta(meta), indent=2, default=str) + "\n")
    prediction_files = sorted(checkpoint.glob("*_predictions.csv")); tuning_files = sorted(checkpoint.glob("*_tuning.csv")); metadata_files = sorted(checkpoint.glob("*_metadata.json"))
    predictions = pd.concat([pd.read_csv(path, parse_dates=["issue_date", "feature_start", "feature_end", "target_start", "target_end", "label_available_date"]) for path in prediction_files], ignore_index=True)
    tuning = pd.concat([pd.read_csv(path) for path in tuning_files], ignore_index=True)
    for path in metadata_files:
        meta = json.loads(path.read_text())
        parts = path.stem.removesuffix("_metadata").split("_")
        selection_rows.append({"checkpoint": path.name, **meta})
    write_csv(predictions.sort_values(["lead", "model", "feature_set", "outer_fold", "issue_date"]), output / "predictions" / "all_out_of_sample_predictions.csv")
    write_csv(tuning, output / "tuning" / "all_candidate_fold_scores.csv")
    write_csv(pd.DataFrame(selection_rows), output / "tuning" / "selected_parameters_and_features.csv")
    manifests = pd.DataFrame(split_rows).drop_duplicates()
    write_csv(manifests, output / "splits" / "chronological_split_manifest.csv")
    return predictions, tuning, manifests


def evaluate(predictions: pd.DataFrame, cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = metrics_from_predictions(predictions)
    write_csv(metrics, output / "metrics" / "model_metrics_by_year_and_pooled.csv")
    comparisons = year_block_bootstrap(predictions, cfg["uncertainty"]["year_block_bootstrap_draws"], cfg["seed"])
    write_csv(comparisons, output / "metrics" / "paired_year_block_comparisons.csv")
    onset = predictions[(predictions.lead == cfg["prediction"]["onset_primary_lead"])].merge(
        pd.read_csv(output / "data" / "prediction_samples.csv", usecols=["issue_date", "lead", "issue_tmax"], parse_dates=["issue_date"]),
        on=["issue_date", "lead"], how="left",
    )
    onset = onset[onset.issue_tmax < cfg["prediction"]["fixed_threshold_c"]]
    rows = []
    for key, frame in onset.groupby(["model", "feature_set"]):
        m = classification_metrics(frame.outcome, frame.probability, 0.5)
        pred_class = frame.predicted_class.to_numpy(int); y = frame.outcome.to_numpy(int)
        tp = int(((pred_class == 1) & (y == 1)).sum()); fp = int(((pred_class == 1) & (y == 0)).sum()); fn = int(((pred_class == 0) & (y == 1)).sum())
        m.update({"tp": tp, "fp": fp, "fn": fn, "recall": tp / (tp + fn) if tp + fn else np.nan, "precision": tp / (tp + fp) if tp + fp else np.nan, "false_alerts": fp, "eligible_dates": len(frame), "distinct_positive_events": frame.loc[frame.outcome.eq(1), "associated_event_id"].nunique(), "threshold": np.nan})
        rows.append({"model": key[0], "feature_set": key[1], **m})
    write_csv(pd.DataFrame(rows), output / "metrics" / "onset_risk_subset.csv")
    return metrics, comparisons


def environment_metadata(cfg: dict, output: Path, started: float, stage_status: dict) -> None:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        branch = commit = "unavailable"
    metadata = {
        "status": stage_status,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "branch": branch,
        "execution_commit": commit,
        "protocol_base_commit": cfg["protocol"]["base_commit"],
        "seed": cfg["seed"],
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__, "matplotlib": matplotlib.__version__},
        "thread_caps": {name: os.environ.get(name) for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "config": cfg,
    }
    path = output / "metadata" / "run_metadata.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/two_paper_benchmark.yml")
    parser.add_argument("--stage", choices=["all", "prepare", "smoke", "benchmark", "evaluate"], default="all")
    args = parser.parse_args(); started = time.time()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    output = ROOT / cfg["outputs"]["root"]; output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "metadata" / "frozen_config_snapshot.yml"; snapshot.parent.mkdir(parents=True, exist_ok=True)
    config_text = (ROOT / args.config).read_text()
    if not snapshot.exists():
        snapshot.write_text(config_text)
    elif snapshot.read_text() != config_text:
        (output / "metadata" / "runtime_amended_config_snapshot.yml").write_text(config_text)
    log_path = output / "metadata" / "execution_status.json"
    status = {"prepare": "not_run", "smoke": "not_run", "benchmark": "not_run", "evaluate": "not_run", "explanations_reports": "deferred"}
    daily = samples = predictions = None
    if args.stage in ["all", "prepare", "smoke"] or not (output / "data" / "prediction_samples.csv").exists():
        daily, samples = prepare(cfg, output); status["prepare"] = "complete"
    else:
        samples = pd.read_csv(output / "data" / "prediction_samples.csv", parse_dates=["issue_date", "feature_start", "feature_end", "target_start", "target_end", "label_available_date"])
        status["prepare"] = "reused_complete_output"
    if args.stage in ["all", "smoke"]:
        smoke_test(samples, cfg, output); status["smoke"] = "complete_smoke_only"
    if args.stage in ["all", "benchmark"]:
        predictions, _, _ = run_primary(samples, cfg, output); status["benchmark"] = "complete"
    if args.stage in ["all", "evaluate"]:
        if predictions is None:
            predictions = pd.read_csv(output / "predictions" / "all_out_of_sample_predictions.csv", parse_dates=["issue_date", "target_start", "target_end"])
        evaluate(predictions, cfg, output); status["evaluate"] = "complete"
    if args.stage == "all":
        subprocess.run([sys.executable, str(ROOT / "scripts" / "finalize_two_paper_benchmark.py")], cwd=ROOT, check=True)
        status["explanations_reports"] = "complete"
    log_path.write_text(json.dumps(status, indent=2) + "\n")
    environment_metadata(cfg, output, started, status)
    print(json.dumps({"status": status, "runtime_seconds": round(time.time() - started, 2), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
