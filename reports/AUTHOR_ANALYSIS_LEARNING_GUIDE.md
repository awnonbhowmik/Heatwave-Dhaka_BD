# Author analysis learning guide

Use this guide to prepare for meetings, peer review, and oral defense. Answers should be explained in your own words and checked against the linked outputs.

## 1. What makes this an original article rather than a review?

**Simple explanation.** The paper asks questions of the Dhaka data and answers them with new statistical analyses. Previous studies explain why the questions matter; they are not the paper's main evidence.

**Technical explanation.** The contribution is an integrated inferential design: six definitions, HAC temperature trends, NB2 count regression, clustered GEE, sensitivity analysis, and chronological validation on a 1972–2024 daily series.

**Repository output.** `reports/manuscript_argument_map.md`; Tables 3–8.

**Likely question.** “What is original if Dhaka heatwaves have already been studied?”

**Model answer.** “The originality is not the first observation of Dhaka heat. It is the direct test of how conclusions change across definitions and models, including overdispersed count inference and leakage-safe, blocked-validated antecedent associations.”

## 2. What is the dependent variable?

**Simple explanation.** The main dependent variable is how many March–June days in each year belonged to a persistent heatwave.

**Technical explanation.** \(Y_y\) is the annual count of hot-season days in events satisfying \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive calendar days.

**Repository output.** `results/tables/main_table06_primary_count_model.csv`.

**Likely question.** “Why count days instead of events?”

**Model answer.** “Days capture both occurrence and persistence and were prespecified as the primary rate outcome. Event count and duration remain secondary descriptive outcomes.”

## 3. What is the primary heatwave definition?

**Simple explanation.** At least three consecutive days, each reaching 36 °C or more in daily maximum temperature.

**Technical explanation.** A Boolean exceedance sequence is grouped only across consecutive calendar dates; runs shorter than three days are excluded. A missing date would break a run.

**Repository output.** `src/heatwave_analysis/heatwave_events.py`; Table 3.

**Likely question.** “Why three days?”

**Model answer.** “Persistence distinguishes sustained events from isolated hot days and aligns with Bangladesh literature; one- and two-day definitions are reported as sensitivity analyses.”

## 4. Why do descriptive statistics come first?

**Simple explanation.** We must understand data quality and the outcome's shape before choosing a model.

**Technical explanation.** The primary counts were zero in 58.5% of seasons and had variance 58.89 versus mean 5.09. Those facts justify count regression and show why Gaussian assumptions are inappropriate.

**Repository output.** Table 1; Figure 2; `table10_count_distribution_diagnostics.csv`.

**Likely question.** “Could you have gone directly to regression?”

**Model answer.** “No. Without inspecting completeness, skewness, zeros, and dispersion, the model family would be unmotivated and potentially wrong.”

## 5. Why is the raw heatmap not a final analysis?

**Simple explanation.** A heatmap compares variables two at a time and ignores other variables and repeated observations.

**Technical explanation.** Pairwise Spearman \(\rho\) does not condition on seasonal timing, long-term trend, collinearity, or within-year dependence and cannot identify causal effects.

**Repository output.** Figure 3A; Table 2.

**Likely question.** “Why not call the strongest correlations drivers?”

**Model answer.** “Because correlation is symmetric and unadjusted. A driver claim would require temporal, mechanistic, and causal evidence beyond this design.”

## 6. Why were de-seasonalized correlations calculated?

**Simple explanation.** Weather variables rise and fall together across March–June simply because the season changes. We removed the normal calendar-day pattern to see whether associations remained.

**Technical explanation.** Each observation was centered on a smoothed 1981–2010 calendar-day mean from a circular ±7-day window; Spearman correlations were then calculated on the anomalies.

**Repository output.** Figure 3B; `table06_spearman_correlations_anomalies.csv`.

**Likely question.** “Did de-seasonalization remove the climate trend?”

**Model answer.** “No. It removed the expected within-year cycle, not the long-term trend.”

## 7. Why was Spearman correlation used?

**Simple explanation.** Some variables are skewed or have relationships that are monotonic but not straight lines.

**Technical explanation.** Spearman correlation ranks observations, reducing sensitivity to extreme values and linearity assumptions. Pearson results remain supplementary.

**Repository output.** Tables 2, 5, and 6 of the rebuilt analysis; Figure 3.

**Likely question.** “Is Spearman immune to outliers?”

**Model answer.** “No, but ranks make it less dominated by magnitude extremes. It still requires careful interpretation and does not solve confounding.”

## 8. Why does correlation not establish an independent effect?

**Simple explanation.** Two variables can correlate because both respond to season, circulation, or another unmeasured factor.

**Technical explanation.** A bivariate coefficient is not a partial regression coefficient and has no counterfactual interpretation. Even the adjusted GEE estimates associations, not effects.

**Repository output.** Table 2 versus Table 7.

**Likely question.** “What changed after adjustment?”

**Model answer.** “The GEE estimates each lagged predictor conditional on seasonal Fourier terms, decade, and the other retained predictors while accounting for clustering.”

## 9. Why is OLS not the principal model for heatwave counts?

**Simple explanation.** Counts cannot be negative and had many zeros and much more variance than the mean.

**Technical explanation.** OLS assumes an approximately continuous outcome with constant conditional variance. The data are discrete and overdispersed, requiring a log-link count model.

**Repository output.** Figure 2C; Tables 5–6.

**Likely question.** “Could robust OLS standard errors fix it?”

**Model answer.** “Robust standard errors do not fix impossible negative fitted values or the wrong mean–variance relationship.”

## 10. Why was Poisson tested first?

**Simple explanation.** Poisson is the standard starting model for counts and provides a clear test of whether variance is too large.

**Technical explanation.** Under Poisson, conditional variance equals the mean. Its Pearson dispersion of 11.64 showed severe overdispersion.

**Repository output.** Table 5.

**Likely question.** “Why show a model you reject?”

**Model answer.** “It documents the model-selection evidence and prevents an arbitrary choice of a more flexible distribution.”

## 11. What does overdispersion mean?

**Simple explanation.** Year-to-year heatwave counts vary far more than a Poisson process expects.

**Technical explanation.** The observed variance-to-mean ratio was 11.56 and Poisson Pearson dispersion was 11.64. Unmodeled heterogeneity and event clustering can create this pattern.

**Repository output.** Tables 1, 5, and 6.

**Likely question.** “Does overdispersion prove the NB2 model is true?”

**Model answer.** “No. It shows Poisson is inadequate; NB2 is a practical variance model that fit substantially better and passed diagnostics.”

## 12. Why was negative binomial selected?

**Simple explanation.** It allowed extra year-to-year variation and fit much better.

**Technical explanation.** NB2 models \(\operatorname{Var}(Y)=\mu+\alpha\mu^2\), converged, and reduced AIC from 659.8 to 245.2 under the prespecified rule.

**Repository output.** Table 5; Figure 6.

**Likely question.** “Was NB selected because it gave the desired p-value?”

**Model answer.** “No. Selection used dispersion, AIC improvement, convergence, and diagnostics, independent of the trend's significance.”

## 13. What does an incidence-rate ratio mean?

**Simple explanation.** It is the multiplicative change in the expected annual count for each decade.

**Technical explanation.** \(\mathrm{IRR}=\exp(\beta_1)\). The primary IRR 1.029 corresponds to an estimated 2.9% higher expected count per decade.

**Repository output.** Table 6.

**Likely question.** “Is 1.029 a 102.9% increase?”

**Model answer.** “No. It is a 2.9% estimated increase per decade because \((1.029-1)\times100=2.9\).”

## 14. How do you interpret an IRR confidence interval containing 1?

**Simple explanation.** The data do not distinguish a decrease, no change, or an increase precisely enough.

**Technical explanation.** The 95% CI 0.727–1.455 includes the null value 1 and is wide. The two-sided \(p=0.872\) provides no evidence against \(H_0\).

**Repository output.** Table 6; Figure 6A.

**Likely question.** “Can you say heatwaves did not increase?”

**Model answer.** “We can say no monotonic increase was detected. We cannot prove exact absence of change because the interval is broad.”

## 15. Why is a nonsignificant result scientifically meaningful?

**Simple explanation.** It corrects the assumption that warming automatically means more persistent events under every definition.

**Technical explanation.** The null-compatible primary count result contrasts with significant continuous-temperature slopes and a positive compound-definition trend, exposing definition dependence and uncertainty.

**Repository output.** Tables 3, 4, and 6.

**Likely question.** “Does a null result weaken the paper?”

**Model answer.** “It strengthens scientific honesty and reveals a methodological distinction that a significance-only narrative would miss.”

## 16. What do HAC standard errors do?

**Simple explanation.** They make trend uncertainty more reliable when residual variance changes or neighboring years are correlated.

**Technical explanation.** Newey–West-type HAC covariance with three lags adjusts the coefficient covariance while leaving the OLS slope itself unchanged.

**Repository output.** Table 4; `trend_models.py`.

**Likely question.** “Why not fit ARIMA to the temperature trend?”

**Model answer.** “The estimand is a transparent long-term slope. HAC covariance addresses modest residual dependence, while nonparametric slopes provide sensitivity.”

## 17. Why does separate Tmax and Tmin significance not prove their slopes differ?

**Simple explanation.** Two estimates can each differ from zero without differing from each other.

**Technical explanation.** A direct interaction tests \(\beta_{T_{\min}}-\beta_{T_{\max}}\). The difference was 0.016 °C/decade with CI -0.034–0.067 and \(p=0.527\).

**Repository output.** Table 4; Figure 4D.

**Likely question.** “Can you call Tmin warming faster?”

**Model answer.** “Only numerically. The formal difference was not statistically detectable, so ‘significantly faster’ is prohibited.”

## 18. What does GEE do?

**Simple explanation.** It estimates average associations while allowing repeated days in the same year to be correlated.

**Technical explanation.** Logistic GEE used year clusters, AR(1) working dependence, and robust sandwich covariance. Coefficients are population-averaged log odds ratios.

**Repository output.** Table 7; `association_models.py`.

**Likely question.** “Why not ordinary logistic regression?”

**Model answer.** “Ordinary logistic standard errors would treat thousands of correlated days as independent and could be too small.”

## 19. Why were predictors lagged?

**Simple explanation.** A condition can only be antecedent if it occurs before the heatwave day.

**Technical explanation.** `shift(1)` is applied before each rolling mean or sum, ensuring the window ends on the previous day.

**Repository output.** `association_models.py`; `test_no_target_leakage.py`.

**Likely question.** “Does a three-day mean include today's humidity?”

**Model answer.** “No. It contains the three completed days before the outcome date.”

## 20. What is target leakage?

**Simple explanation.** Leakage occurs when a predictor contains the answer or information unavailable at prediction time.

**Technical explanation.** Same-day \(T_{\max}\), temperature-derived VPD/heat index, event duration, and future rolling values would encode the target or future event structure.

**Repository output.** `variable_dictionary.py`; leakage tests.

**Likely question.** “Why exclude VPD if it is meteorologically relevant?”

**Model answer.** “Its formulation uses temperature and could make a \(T_{\max}\)-defined outcome tautological. Relevance does not override leakage risk.”

## 21. What does an adjusted odds ratio mean?

**Simple explanation.** It compares the odds of a persistent day for a one-standard-deviation predictor difference while holding modeled covariates constant.

**Technical explanation.** \(\exp(\theta_j)\) is conditional on Fourier seasonality, decade, and the other lagged predictors in the population-averaged GEE.

**Repository output.** Table 7; Figure 7A.

**Likely question.** “Does OR 0.307 mean probability falls by 69.3 percentage points?”

**Model answer.** “No. It is a 69.3% reduction in odds, not an absolute probability change.”

## 22. Why is adjusted association not causation?

**Simple explanation.** Adjustment cannot remove all unmeasured weather processes or prove what would happen under intervention.

**Technical explanation.** The model is observational, predictors are correlated, synoptic conditions are incompletely represented, and no causal identification assumptions or intervention were available.

**Repository output.** Table 7; claim-to-evidence matrix.

**Likely question.** “Can rainfall suppression be called a cause?”

**Model answer.** “The result is physically plausible but remains an antecedent association. Causation would require a stronger design and mechanistic evidence.”

## 23. Why is chronological validation required?

**Simple explanation.** A real model learns from the past and predicts the future.

**Technical explanation.** Each origin trained only on earlier years; standardization and thresholds were also training-only. This evaluates temporal transport rather than interpolation across randomly mixed dates.

**Repository output.** Table 8; Figure 7C–D.

**Likely question.** “Why start pooled validation in 2000?”

**Model answer.** “It leaves a substantial historical training period while providing 25 independent hot-season origins through 2024.”

## 24. Why is random train–test splitting inappropriate here?

**Simple explanation.** Random splitting would let nearby days and the same event appear in both training and test sets.

**Technical explanation.** Serial dependence and nonstationarity would leak event structure and future climatology, producing optimistically biased performance.

**Repository output.** `test_time_splits.py`; Table 8.

**Likely question.** “Would cross-validation with shuffled folds give more samples?”

**Model answer.** “More folds would not compensate for temporal contamination. The question is future-season prediction, so folds must respect time.”

## 25. Why can a simple baseline outperform SARIMAX or LSTM?

**Simple explanation.** Complexity adds estimation error when the dominant signal is stable seasonality plus a modest trend.

**Technical explanation.** Climatology plus trend achieved RMSE 1.114 °C versus 1.607 °C for SARIMAX on identical origins and targets. In-sample fit criteria do not guarantee out-of-sample accuracy.

**Repository output.** Supplementary Table S2 and Figure S2.

**Likely question.** “Is SARIMAX always inferior?”

**Model answer.** “No. It was inferior in this design and dataset. Different targets, tuning, or more data could change the ranking.”

## 26. Why are exact 2025–2029 heatwave counts not defensible?

**Simple explanation.** Monthly average temperature forecasts cannot tell us which individual days cross a threshold consecutively.

**Technical explanation.** Event counts require a validated daily joint simulation of temperature and persistence. Direct count extrapolations had wide parameter and observation uncertainty and no five-year validation.

**Repository output.** Forecast supplement; claim matrix.

**Likely question.** “Why keep future simulations in the repository?”

**Model answer.** “They document uncertainty and the analytical history, but they are labeled scenarios and excluded from article claims.”

## 27. What does each principal table demonstrate?

**Simple explanation.** Table 1 describes data; 2 screens correlations; 3 compares definitions; 4 estimates temperature trends; 5 selects the count family; 6 gives the primary trend; 7 gives adjusted antecedent associations; 8 validates and stress-tests them.

**Technical explanation.** The table order mirrors the dependency chain from quality control through exploration, estimand definition, inference, diagnostics, and validation.

**Repository output.** `results/tables/main/`.

**Likely question.** “Which table contains the primary answer?”

**Model answer.** “Table 6 contains the primary NB2 IRR; Tables 5 and 8 provide model-selection and robustness context.”

## 28. What does each principal figure demonstrate?

**Simple explanation.** Figure 1 establishes place and coverage; 2 motivates the model; 3 explores covariance; 4 shows warming; 5 shows definition sensitivity; 6 shows count inference and diagnostics; 7 shows adjusted associations and validation.

**Technical explanation.** No main figure is decorative: each answers a research question or justifies a statistical choice.

**Repository output.** `results/figures/main/`.

**Likely question.** “Why is forecasting not Figure 8?”

**Model answer.** “It did not add reliable main-text event evidence, so it is Supplementary Figure S2.”

## 29. What is the strongest conclusion?

**Simple explanation.** Dhaka warmed, but whether heatwave occurrence increased depends on the definition; the primary persistent-count trend is uncertain.

**Technical explanation.** Continuous-temperature CIs excluded zero, the primary NB2 IRR CI included 1 broadly, and only the compound definition showed a positive count-trend CI excluding 1.

**Repository output.** Tables 3, 4, and 6; Figures 4–6.

**Likely question.** “Give the paper's message in one sentence.”

**Model answer.** “Clear background warming and uncertain persistent-event count change coexist in Dhaka, and heatwave conclusions must be tied to explicit definitions.”

## 30. What can the paper not claim?

**Simple explanation.** It cannot claim measured vulnerability, health effects, urban amplification, tree-cover causation, meteorological causal drivers, or precise future counts.

**Technical explanation.** The dataset lacks health outcomes, exposure measures, a rural comparator, a causal design, validated daily event forecasts, and fully documented station homogeneity.

**Repository output.** `reports/claim_to_evidence_matrix.csv`; limitations section.

**Likely question.** “What additional data would change those limits?”

**Model answer.** “Verified station metadata and homogenized multi-station urban–rural observations, health/exposure data, causal or mechanistic designs, and externally validated daily forecasts would support broader claims.”
