# Response to mock peer review

We thank the three reviewers. The revision prioritizes estimator consistency, explicit multiplicity control, provenance limitations, and a manuscript scope supported by the data. Page and line numbers should be added after journal formatting.

## Response to Reviewer 1

### Comment 1: station provenance and homogeneity

**Response.** Agreed. The revised Methods and Limitations identify the unavailable station identifier, coordinates, units metadata, observation conventions, relocation history, and homogenization record. Figure 1 does not infer or display a station point. The manuscript treats provider verification as a pre-submission requirement and does not describe the series as homogenized or representative of all Dhaka.

### Comment 2: seasonal domain

**Response.** The revision calls March–June the prespecified hot-season analysis window and distinguishes it from narrower uses of “pre-monsoon.” Completeness is stated explicitly: each March–June season contains 122 dates, including 2024, while calendar year 2024 is incomplete.

### Comment 3: definition multiplicity

**Response.** We added BH adjustment across the six primary-reference definition-specific trend tests. The compound result changed from a positive-trend conclusion to an exploratory nominal signal: IRR 1.320 (95% CI 1.043–1.672), nominal \(p=0.0209\), BH \(q=0.126\). This correction appears in the abstract, Results, Discussion, Conclusions, supplement, cover letter, Table 3, and claim matrix.

### Comment 4: percentile reproducibility

**Response.** The Methods and supplement specify the \(d\pm7\) circular window, NumPy linear quantile algorithm, 1981–2010 primary and 1991–2020 sensitivity periods, climatological-day mapping, 29 February interpolation, strict percentile exceedance, and event breaks at missing dates.

### Comment 5: causal language and units

**Response.** All weather results are described as antecedent associations. Pressure is not a principal finding because its sign reverses in the onset analysis. Wind and pressure units are explicitly flagged for provider confirmation.

## Response to Reviewer 2

### Comment 1: estimator-aligned validation

**Response.** Corrected. Every rolling origin now refits logistic GEE with year clusters, the same covariates, and training-only means and standard deviations. Predictions cover 25 held-out seasons from 2000 through 2024.

### Comments 2 and 3: imbalance and calibration

**Response.** Table 8 and Figure 7 now report pooled Brier score, ROC-AUC, PR-AUC, calibration intercept and slope, sensitivity, specificity, sample size, and positive count. The full model improved Brier score from 0.0382 to 0.0333 and PR-AUC from 0.198 to 0.473, but its calibration intercept was 1.512 and slope 1.136. The manuscript interprets this as systematic underprediction and explicitly precludes operational-warning use. Seasons with no positives retain valid Brier and specificity values while discrimination and calibration metrics remain missing.

### Comment 4: multiplicity families

**Response.** BH adjustment for antecedent associations now applies only to the four prespecified weather predictors; it excludes nuisance seasonality and decade terms. A separate six-test BH family was added for definition-specific count trends.

### Comment 5: selected-distribution diagnostics

**Response.** The diagnostics now use NB2 Pearson and deviance summaries, randomized quantile residuals, and NB2 case-deletion parameter distance. The flagged seasons 1979, 2023, and 2024 remain in the primary model; their separate exclusions do not change the uncertainty conclusion.

### Comment 6: power and estimand

**Response.** The manuscript now consistently describes annual March–June heatwave-day counts and states that the broad IRR interval reflects limited precision. It does not equate \(p>0.05\) with proof of no change.

## Response to Reviewer 3

### Comment 1: future event counts

**Response.** Exact 2025–2029 heatwave-count projections were removed from the article and supplement as evidence. The legacy output remains reproducibility-only and is explicitly identified as not interpretable from monthly means.

### Comments 2 and 3: common origins and intervals

**Response.** Models are compared at the same ten rolling origins, 2014–2023, for March–June monthly mean \(T_{\max}\). The supplement reports RMSE, MAE, MASE, empirical 80% and 95% coverage, and interval widths. SARIMAX's wider intervals are not treated as superior skill.

### Comment 4: reproducibility record

**Response.** The rebuilt pipeline records source hashes, configuration, runtime metadata, software versions, repository state, tables, figures, tests, and validator results. `make article` is the complete reproduction command.

### Comment 5: editable deliverables

**Response.** The build now creates clean and yellow-highlighted one-column Word manuscripts plus an editable supplementary Word file. Text is black, tables are native Word tables, and captions follow their figures.
