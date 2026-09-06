#!/usr/bin/env python3
"""Reconcile benchmark predictions, metrics, figures, reports, and protected files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config/two_paper_benchmark.yml").read_text())
    output = ROOT / cfg["outputs"]["root"]; reports = ROOT / cfg["outputs"]["reports"]
    checks = []
    pred = pd.read_csv(output / "predictions/all_out_of_sample_predictions.csv", parse_dates=["issue_date", "feature_start", "feature_end", "target_start", "target_end", "label_available_date"])
    checks.extend([
        {"check": "feature_window_ends_at_issue", "passed": bool((pred.feature_end == pred.issue_date).all())},
        {"check": "target_strictly_future", "passed": bool((pred.target_start > pred.issue_date).all())},
        {"check": "label_available_at_target_end", "passed": bool((pred.label_available_date == pred.target_end).all())},
        {"check": "all_outer_years_retained", "passed": set(pred.outer_fold.unique()) == set(cfg["prediction"]["outer_test_years"])},
    ])
    manifest = pd.read_csv(output / "splits/chronological_split_manifest.csv")
    checks.append({"check": "all_training_labels_available", "passed": bool(manifest.labels_available_before_fit.astype(str).str.lower().eq("true").all())})
    metrics = pd.read_csv(output / "metrics/model_metrics_by_year_and_pooled.csv")
    maximum_error = 0.0; confusion_error = 0
    for key, frame in pred.groupby(["model", "feature_set", "lead"]):
        row = metrics[(metrics.model == key[0]) & (metrics.feature_set == key[1]) & (metrics.lead == key[2]) & (metrics.scope == "pooled_strictly_out_of_sample")].iloc[0]
        ap = average_precision_score(frame.outcome, frame.probability) if frame.outcome.sum() else np.nan
        brier = brier_score_loss(frame.outcome, frame.probability)
        maximum_error = max(maximum_error, abs(ap - row.average_precision) if np.isfinite(ap) else 0, abs(brier - row.brier_score))
        tn, fp, fn, tp = confusion_matrix(frame.outcome, frame.predicted_class, labels=[0, 1]).ravel()
        confusion_error += abs(tp - row.tp) + abs(fp - row.fp) + abs(tn - row.tn) + abs(fn - row.fn)
    checks.append({"check": "pooled_metric_reconciliation", "passed": maximum_error < 1e-8, "maximum_absolute_error": maximum_error})
    checks.append({"check": "confusion_reconciliation", "passed": confusion_error == 0, "count_error": int(confusion_error)})
    shap_meta = pd.read_csv(output / "explanations/shap_background_and_additivity.csv")
    checks.append({"check": "shap_additivity", "passed": bool((shap_meta.maximum_additivity_absolute_error < 1e-4).all()), "maximum_absolute_error": float(shap_meta.maximum_additivity_absolute_error.max())})
    protected = pd.read_csv(output / "metadata/protected_file_hash_validation.csv")
    checks.append({"check": "raw_data_and_manuscripts_unchanged", "passed": bool(protected.unchanged.astype(str).str.lower().eq("true").all())})
    figure_ok = all((output / f"figures/figure{i:02d}_{stem}.png").exists() and (output / f"figures/figure{i:02d}_{stem}.pdf").exists() for i, stem in [(1, "workflow_timeline"), (2, "descriptive_class_balance"), (3, "predictor_relationships"), (4, "model_skill_by_lead"), (5, "precision_confusion_calibration"), (6, "shap_importance_stability"), (7, "held_out_episode_timeline")])
    checks.append({"check": "seven_png_and_vector_figures", "passed": figure_ok})
    required_reports = ["source_method_transfer.md", "implementation_gap_audit.md", "prediction_contract.md", "analysis_results_brief.md", "monthly_feasibility.md", "AUTHOR_METHODS_GUIDE.md", "DECISION_FOR_PROFESSOR.md"]
    checks.append({"check": "required_reports_present", "passed": all((reports / name).exists() for name in required_reports)})
    leading = metrics[(metrics.scope == "pooled_strictly_out_of_sample") & (metrics.feature_set.isin(["S1", "S2"])) & (metrics.lead == 1) & metrics.model.isin(cfg["models"]["families"])].sort_values("average_precision", ascending=False).iloc[0]
    brief = (reports / "analysis_results_brief.md").read_text()
    checks.append({"check": "narrative_primary_ap_reconciles", "passed": f"AP **{leading.average_precision:.3f}**" in brief})
    result = {"status": "passed" if all(row["passed"] for row in checks) else "failed", "checks": checks}
    json_default = lambda value: value.item() if isinstance(value, np.generic) else str(value)
    path = output / "metadata/output_validation.json"; path.write_text(json.dumps(result, indent=2, default=json_default) + "\n")
    print(json.dumps(result, indent=2, default=json_default))
    if result["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__":
    main()
