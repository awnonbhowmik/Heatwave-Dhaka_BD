# Source-method transfer

The two PDFs named in the task were not present in the repository or available attachment directory at audit time. The mapping therefore uses the exact open publisher articles: [Kan et al. (2025)](https://link.springer.com/article/10.1007/s00477-025-03020-1) and [the Scientific Reports India comparison (2025)](https://www.nature.com/articles/s41598-025-04634-9), supplemented only by the task's explicit descriptions.

## Paper A: Kan et al.

The paper predicts monthly, pixel-level Swedish heatwave occurrence at one-to-five-month leads from Google Earth Engine atmospheric and land-surface predictors. It compares five classifiers, includes balanced random forest and a historical occurrence-probability baseline, uses training-only imbalance handling, evaluates by test years, and applies SHAP to model outputs. It defines heatwaves using Sweden's 27 °C warning threshold and excludes an event-free test year from some class-specific summaries.

Transferable here: explicit lead-specific classification, historical summaries, chronological model assessment, an occurrence baseline, imbalance-aware learners, class-specific metrics, and held-out predictive explanation.

Not transferred: Sweden's threshold, spatial pixels, unavailable runoff/geopotential-height/specific-humidity features, cluster-centroid undersampling, exclusion of event-free seasons, causal language for feature attributions, or any published performance/model ranking. This Dhaka benchmark is daily and single-location; it does not demonstrate seasonal predictability.

## Paper B: India classifier comparison

The paper uses a single-location weather sequence with seven-day inputs for next-day classification, compares conventional and deep classifier families, uses class weighting for rare events, and emphasizes confusion matrices plus observed-versus-predicted evaluation. Its published implementation also reports random/k-fold validation language and a broad nine-model comparison.

Transferable here: the seven-day historical window, next-day framing, manageable classifier comparison, class-weighted losses, rare-event miss analysis, and confusion/observed-predicted displays.

Not transferred: random stratification of overlapping windows, synthetic or sequence augmentation, a model-count target, unsupported deep/graph architectures, next-day hot-day labels presented as persistent heatwaves, or published accuracies/rankings.

## Dhaka adaptation

The adaptation predicts a fully specified future three-day persistent-hot window at direct 1-, 3-, and 7-day leads. It uses three nested information sets, strict expanding-year validation with label-availability purging, temporal cross-fitted calibration and thresholding, strong leakage-safe baselines, held-out-year uncertainty, and non-causal out-of-sample explanations. It preserves the existing association and warming/count analyses as context rather than treating them as this prediction experiment.
