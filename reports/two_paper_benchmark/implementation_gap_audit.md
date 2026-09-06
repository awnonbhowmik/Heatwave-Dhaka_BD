# Implementation-gap audit

Audit base: `29aa3ef2f3dce8c55c4a39a18b8717f77dcbf662`. Baseline tests executed in the isolated clean worktree: **17 passed in 5.91 seconds**. Existing result files were inspected but not overwritten.

## Already implemented and verified

- Raw CSV schema mapping, immutable-source hashing, date parsing, quality checks, and physical units.
- Continuous-date persistent-event grouping that breaks runs at missing dates.
- Descriptive, correlation, anomaly-context, definition-sensitivity, trend, count, and adjusted antecedent-association analyses.
- Some expanding/rolling validation for the earlier association and monthly-temperature analyses.
- Existing tests for event construction, percentile thresholds, partial years, time splits, leakage rules, forecast intervals, and output consistency.

## Implemented but needs correction for this benchmark

- Existing `assert_no_target_leakage` rejects all temperature-derived inputs; past temperatures known at issue time are legitimate here, so benchmark leakage checks must use timestamps/provenance rather than name prefixes.
- Existing binary validation uses historical event-status labels rather than explicitly future-aligned three-day targets.
- Its threshold is learned from in-sample fitted probabilities rather than temporally cross-fitted calibrated predictions.
- It is not a comparable nested classifier experiment and does not retain per-prediction provenance.

## Missing

- Explicit future targets at 1-, 3-, and 7-day leads and onset-risk evaluation.
- S0/S1/S2 information-set ablations and a predictor dictionary.
- Required logistic, weighted RF, balanced RF, XGBoost, and weighted SVC comparison.
- Nested year-blocked tuning, target-label availability purging, held-out calibration, temporal threshold selection, and complete split manifests.
- Always-negative, exact-target seasonal-probability, and observed-temperature transition baselines.
- Per-year and pooled rare-event metrics, event-free-season handling, year-block paired uncertainty, and relative-threshold/history sensitivities.
- Out-of-sample SHAP, grouped permutation diagnostics, rank stability, and documented held-out cases.
- Single-command entry point and the dedicated evidence/decision package.

## Not applicable

- Spatial-pixel modelling, graph networks, and spatial validation: the repository has one location series.
- One-to-five-month execution: the available record provides at most 212 March–June target-month rows before exclusions and lacks the regional predictor field used by Paper A.
- New external weather acquisition, artificial spatial pixels, sequence augmentation, LSTM/Transformer/GNN/autoencoder model-count expansion.
