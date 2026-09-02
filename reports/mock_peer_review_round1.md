# Mock peer review: round 1

## Reviewer 1: meteorology and climate extremes

### Overall assessment

The manuscript addresses a useful question and is unusually transparent about definition dependence. Its strongest contribution is the separation of continuous-temperature warming from persistent-event counts. I would consider it after major revision, principally because the provenance and homogeneity of the meteorological record cannot yet be independently verified.

### Fatal flaw assessment

No demonstrated fatal analytical flaw, but unavailable station provenance and homogeneity history are a potential submission blocker. If the authors cannot document the series sufficiently, long-term trend attribution must be narrowed further or independently checked against a verified station/reanalysis source.

### Major comments

1. **Station provenance and homogeneity are unresolved.** The record is labeled for Dhaka, but the exact station identifier, coordinates, instruments, observation-time conventions, relocation history, and homogenization status are unavailable. Long-term trends can reflect nonclimatic changes. The paper must not imply that the supplied series is a homogenized citywide climate record. Add this limitation prominently, retain the decision not to plot an unsupported station point, and require author verification before submission.

2. **Clarify the seasonal domain.** March–June combines the conventional pre-monsoon months with June, often treated as monsoon onset. Explain that the window was selected as an analysis window covering the local hot season, not as a claim that all four months are meteorologically identical. Report complete-season coverage and avoid calling March–June simply “pre-monsoon” without qualification.

3. **Definition sensitivity needs multiplicity context.** Six count trends are shown. The nominal compound day/night result is interesting, but it is secondary and one of several tests. Report an across-definition adjustment and temper the conclusion if it does not remain significant.

4. **Percentile construction should be fully reproducible.** The circular calendar-day window, quantile algorithm, leap-day handling, reference periods, strictness of threshold comparison, and missing-day event breaks should remain explicit. These choices materially affect event counts.

5. **Antecedent conditions are not physical causes.** Lower humidity and precipitation plausibly mark dry conditions, but the positive wind estimate and unstable pressure estimate require caution. Units are also unverified. Limit interpretation to adjusted temporal associations and discuss the pressure sign reversal.

### Minor comments

- Keep years as whole numbers throughout axes and tables.
- Figure 1 appropriately avoids an unsupported point location; the caption should say why.
- Figure 3 is clearer without a redundant overall title. Keep the matrices close and the colorbar equal to the matrix height.
- The article should use mathematical notation such as \(T_{\max}\), \(T_{\min}\), \(P_{90}\), IRR, OR, and \(q\) consistently.

### Unsupported claims, diagnostics, and presentation

- Unsupported claims to remove: citywide representativeness, urban amplification, causal meteorological “drivers,” and a confirmatory compound trend.
- Missing diagnostic: an external homogeneity check, which cannot be completed from the supplied files and must remain a limitation.
- Figure/table issue: identify secondary definition tests and their multiplicity adjustment in Table 3 and Figure 5's interpretation.
- Terminology issue: use “March–June hot-season window” where June's monsoon-transition status matters.
- Novelty concern: claim an integrated methodological contribution, not a first Dhaka or Bangladesh heatwave study.

## Reviewer 2: statistics and epidemiologic methods

### Overall assessment

The workflow is substantially stronger than a descriptive heatwave paper: it prespecifies a primary outcome, recognizes overdispersion, uses cluster-aware binary models, separates screening correlations from adjusted associations, and performs chronological validation. Major revision is needed to align every validation and multiplicity claim with the inferential estimator.

### Fatal flaw assessment

No fatal flaw after estimator-aligned validation, provided the paper does not interpret wide null-compatible intervals as proof of no change. Using a non-GEE substitute in validation would have been a major internal inconsistency.

### Major comments

1. **Validation must refit the reported estimator.** A generalized linear model would not be an adequate stand-in for the article’s year-clustered logistic GEE. Each training origin should refit the same GEE specification, with training-only scaling and no future-season leakage.

2. **Class imbalance requires precision–recall and calibration reporting.** ROC-AUC alone would be optimistic with relatively few positive days. Report pooled PR-AUC, Brier score, calibration intercept and slope, sensitivity, specificity, and the number of positives. Do not turn undefined within-season AUCs into zero.

3. **The full model appears undercalibrated.** A large positive calibration intercept indicates underprediction even if discrimination improves. This should preclude an operational-warning claim unless recalibration is performed and independently evaluated.

4. **Multiplicity families should be defensible.** The four antecedent weather coefficients form one family; base seasonality and trend terms should not dilute their BH adjustment. Conversely, the six definition-specific count trends need their own explicit adjustment because the compound result is emphasized.

5. **NB2 diagnostics should match the selected distribution.** Poisson residuals or influence diagnostics cannot validate an NB2 selection. Use NB2 Pearson/deviance summaries, randomized quantile residuals, and a transparent case-deletion sensitivity. Retain influential seasons unless there is a prespecified data-quality reason to remove them.

6. **Power and estimand language need restraint.** Forty-nine primary events and 53 seasons provide limited information for a monotonic rate trend. A wide IRR interval is evidence of uncertainty, not evidence of no change. State the modeled estimand precisely as March–June heatwave days per year, not event incidence unless events themselves are modeled.

### Minor comments

- State the HAC lag rule and time scaling for every slope.
- Report the formal \(T_{\min}-T_{\max}\) slope contrast rather than comparing separate significance tests.
- Make clear that marginal-probability curves hold remaining covariates and calendar day fixed.
- Preserve exact output values in machine-readable tables and use rounded values only in prose.

### Unsupported claims, diagnostics, and presentation

- Unsupported claims to remove: “no trend,” independent causal effects, and operational calibration.
- Missing diagnostics before revision: NB2-consistent influence/residual checks and pooled calibration; both are required.
- Figure/table issue: Figure 7 must show uncertainty for marginal probabilities, and Table 8 must expose sample size and positive count.
- Terminology issue: distinguish heatwave-day rate from event incidence.
- Novelty concern: robust workflow integration is useful, but statistical methods themselves are not novel.

## Reviewer 3: forecasting and reproducibility

### Overall assessment

The decision to demote forecasting is scientifically appropriate. The validation shows that complexity did not reliably outperform simple baselines. The article should retain forecasting only as a methodological supplement.

### Fatal flaw assessment

No fatal flaw if forecasting remains supplementary. Treating monthly-mean forecasts as validated daily persistent-event forecasts would be fatal to the forecasting claim.

### Major comments

1. **Do not report exact future heatwave counts from monthly temperature forecasts.** Persistence is a daily property; monthly means cannot identify runs of daily threshold exceedance. Remove 2025–2029 event-count projections from manuscript evidence.

2. **Compare all models at identical rolling origins.** Forecast accuracy, empirical coverage, and interval width must use the same held-out targets. Report the number and dates of origins.

3. **Interpret intervals jointly by coverage and sharpness.** Wider intervals can achieve better coverage without improving point accuracy. The SARIMAX result should not be called superior merely because its 95% coverage is high.

4. **Reproducibility requires more than code presence.** Record source hashes, configuration, starting and ending commits, software versions, generated paths, and an exact build command. Validate that manuscript claims match generated tables.

5. **Editable deliverables are required.** Provide one-column Word documents with black text, editable tables, figures followed by captions, a clean version, and a version highlighting only substantively new manuscript prose.

### Recommendation

Major revision. The revised work would be publishable in scope if the station metadata caveat is made non-negotiable and claims remain constrained by the validation evidence.

### Unsupported claims, diagnostics, and presentation

- Unsupported claims to remove: SARIMAX superiority and exact 2025–2029 heatwave-day counts.
- Missing diagnostic before revision: like-for-like origin coverage and interval sharpness alongside coverage.
- Figure/table issue: keep forecast comparisons in the supplement and use identical scales and origins.
- Terminology issue: distinguish a forecast interval from a confidence interval and monthly temperature from daily events.
- Novelty concern: forecasting does not add originality here because it lacks superior validated skill.
