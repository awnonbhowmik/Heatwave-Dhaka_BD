# Long-Term Warming and Definition-Dependent Heatwaves in Dhaka, Bangladesh: Count Trends and Antecedent Meteorological Associations, 1972–2024

**Provisional title—professor and coauthor approval required**

**Article type:** Original Article  
**Authors:** [Insert approved author list]  
**Affiliations:** [Insert approved affiliations]  
**Corresponding author:** [Insert name and email]

## Abstract

Long-term warming does not guarantee that heatwave counts increase uniformly, because event estimates depend on thresholds, persistence requirements, and day/night criteria. We analyzed 19,316 consecutive daily meteorological records for Dhaka from 1972–2024. Complete-calendar-year temperature trends through 2023 and complete March–June trends through 2024 were estimated using ordinary least squares with heteroskedasticity-and-autocorrelation-consistent standard errors, Theil–Sen slopes, and Mann–Kendall tests. Six heatwave definitions were compared. The primary outcome was the annual March–June count of days in events with \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive days. Poisson and NB2 count models were compared. Leakage-safe lagged predictors were evaluated with logistic generalized estimating equations and chronological validation. Annual mean \(T_{\max}\) increased by 0.192 °C per decade (95% confidence interval [CI] 0.130–0.253), and March–June mean \(T_{\max}\) increased by 0.165 °C per decade (95% CI 0.044–0.287). The primary count distribution was strongly overdispersed (variance/mean 11.56), favoring NB2 over Poisson (Akaike information criterion 245.2 versus 659.8). The primary trend was uncertain (incidence-rate ratio [IRR] 1.029 per decade, 95% CI 0.727–1.455; \(p=0.872\)). A secondary compound day/night definition had IRR 1.320 (95% CI 1.043–1.672; nominal \(p=0.0209\)), but this did not remain significant after adjustment across six definitions (Benjamini–Hochberg \(q=0.126\)). Lower antecedent humidity (adjusted odds ratio [OR] 0.307), lower precipitation (OR 0.140), and higher wind speed (OR 1.337) were associated with persistent days; directions remained stable for event onset. Pooled out-of-sample precision–recall area under the curve improved from 0.198 for seasonality and trend alone to 0.473 for the antecedent model, although calibration showed underprediction. Dhaka warmed detectably, but heatwave-count conclusions were definition-dependent and the primary persistent-event rate showed no clear monotonic change. Antecedent associations were predictive rather than causal, and forecast validation did not justify exact five-year heatwave counts.

**Keywords:** extreme temperature; heatwave definition; negative binomial regression; generalized estimating equations; blocked validation; Bangladesh

## 1 Introduction

Heatwaves are sustained periods of unusually or operationally high temperature, yet there is no single definition that is optimal for every scientific or decision context. Absolute thresholds can align with local warning practices, percentile thresholds adjust to the seasonal temperature distribution, persistence requirements distinguish isolated hot days from sustained events, and combined maximum–minimum criteria capture limited nighttime relief. These choices alter the number, duration, and timing of identified events and can therefore alter estimated trends. A credible long-term analysis must distinguish background warming from changes in a prespecified heatwave outcome rather than assuming that the two are equivalent.

Bangladesh experiences its most consequential dry heat during the pre-monsoon transition, when suppressed rainfall, land-surface dryness, and regional circulation can favor high temperatures. Nissan et al. (2017) developed a Bangladesh heatwave definition combining high daytime and nighttime temperatures and described precipitation and soil-moisture antecedents. Multi-station studies have subsequently documented threshold-day climatology and spatially heterogeneous trends (Rashid et al. 2024; Mallik et al. 2024), while newer studies have examined percentile definitions, reanalysis agreement, circulation, and short-lead heat-stress prediction (Molla et al. 2025; Chaki et al. 2025; Farukh et al. 2026). These studies establish the national meteorological relevance of definition choice and persistence.

Dhaka-specific studies provide important but different evidence. Khatun et al. (2017) estimated temperature and rainfall trends across Dhaka Division. Islam et al. (2024) combined land-surface temperature, meteorological descriptions, and survey evidence for Dhaka Metropolitan City. Tabassum et al. (2024) used an urban–rural station comparison and reanalysis to evaluate the urban heat island and its interaction with percentile heatwaves. Those designs should not be conflated with the present single-series dataset, which cannot measure an urban–rural contrast, human vulnerability, or health impacts. The remaining analytical gap is narrower: few Dhaka-focused studies have combined alternative event definitions, direct count regression, adjusted antecedent meteorological associations, and chronological model validation within one long daily series.

Descriptive counts and pairwise correlations alone cannot fill this gap. Annual heatwave counts are discrete, zero-heavy, and potentially overdispersed; ordinary least squares is therefore not the natural primary model. Raw meteorological correlations also mix within-season covariance with the shared seasonal cycle and do not estimate independent associations. Repeated daily outcomes require attention to within-year dependence, and predictive performance must be evaluated on future seasons rather than by random train–test splitting. These considerations motivate direct count modeling, de-seasonalized exploration, leakage-safe lag construction, generalized estimating equations, model diagnostics, and blocked validation.

We addressed three questions. First, how did annual and March–June maximum and minimum temperatures change in Dhaka during 1972–2024? Second, how did heatwave frequency, duration, intensity, and estimated trend vary across operational, persistence-based, percentile, and compound day/night definitions? Third, which antecedent meteorological conditions were associated with persistent heatwave days after accounting for seasonality, long-term change, collinearity, and within-year temporal dependence? We prespecified March–June as the hot-season analysis window; because June can include monsoon onset, this label does not imply that all four months share one meteorological regime. The primary outcome was days belonging to events with \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive days. The primary null hypothesis was \(H_0:\mathrm{IRR}=1\) per decade; the two-sided alternative was \(H_1:\mathrm{IRR}\ne1\).

## 2 Methods

### 2.1 Study design and setting

We conducted a retrospective, single-location meteorological time-series study for Dhaka, Bangladesh. The supplied dataset is described as daily Dhaka meteorology, but the repository does not contain a verified station identifier or exact coordinate. We therefore show administrative geography without inferring a station point and do not estimate urban-heat-island intensity. The analysis is observational and evaluates statistical trends and associations, not intervention effects or causal mechanisms.

### 2.2 Data source and temporal coverage

The immutable daily CSV contains 19,316 dates from 1 January 1972 through 18 November 2024 and 32 source fields. Variables used here include daily maximum, minimum, and mean air temperature; precipitation; relative humidity; wind speed; cloud cover; sunshine duration; shortwave and longwave radiation; pressure; evapotranspiration; soil temperature; and soil moisture. Source-file SHA-256 hashes are recorded in `results/metadata/source_data_hashes.json`. The repository documentation attributes the meteorological data to the Bangladesh Meteorological Department; station provenance and homogenization metadata require confirmation by the authors before submission.

### 2.3 Data quality control

Dates were parsed, sorted, and compared with a complete daily calendar. We evaluated duplicate dates, missing dates, field-level missingness, leap days, physical-range flags, and abrupt day-to-day changes. The record had no duplicate or missing dates and nine missing field values. Flagged values were retained because no independent quality-control metadata justified deletion. Calendar year 2024 was incomplete and was excluded from complete-year annual means. Every March–June season, including 2024, contained the expected 122 dates and was eligible for hot-season analyses. Leap days remained in descriptive analyses. For calendar-day climatologies, dates after February in leap years were mapped to a 365-day climatological calendar; 29 February used the mean of 28 February and 1 March thresholds.

### 2.4 Heatwave definitions and event construction

The primary definition was daily \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive calendar days. Sensitivity definitions were: (A) \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least one day; (B) the same threshold for at least two consecutive days; (C) the primary three-day definition; (D) calendar-day 90th-percentile \(T_{\max}\) for at least three days; (E) calendar-day 95th-percentile \(T_{\max}\) for at least three days; and (F) concurrent calendar-day 90th-percentile \(T_{\max}\) and \(T_{\min}\) for at least two days.

The primary percentile reference was 1981–2010, with 1991–2020 as a sensitivity period. For climatological day \(d\), the threshold \(q_p(d)\) was the linear sample quantile from reference-period observations within a circular centered window \(d\pm7\) days. A date was classified as a percentile exceedance when its observed temperature was strictly above the relevant threshold. Consecutive dates were grouped into events; a missing date would break an event. For event \(e\), cumulative intensity was

\[
I_e=\sum_{t\in e}\{T_{\max,t}-q_t\},
\]

where \(q_t=36\,{}^\circ\mathrm{C}\) for absolute definitions and the calendar-day threshold for relative definitions. We calculated qualifying days, events, duration, maximum excess, cumulative intensity, onset timing, month, annual distribution, and decadal count trends for March–June.

### 2.5 Descriptive statistics

Before inferential modeling, we summarized valid and missing observations, mean, standard deviation, median, quartiles, interquartile range, minimum, maximum, skewness, and excess kurtosis for all daily observations, March–June observations, persistent heatwave days, and non-heatwave hot-season days. For annual primary counts, we reported sample size, mean, variance, variance-to-mean ratio, zero proportion, range, and skewness. These diagnostics determined whether a Gaussian count model was defensible.

### 2.6 Correlation and collinearity analysis

We emphasized Spearman rank correlation because several meteorological variables, notably precipitation, were skewed and relationships need not be linear. Raw correlations used March–June observations. De-seasonalized correlations used anomalies

\[
X'_{t}=X_t-\bar X_{1981:2010}(d_t),
\]

where \(\bar X_{1981:2010}(d_t)\) was the smoothed calendar-day mean within a circular \(\pm7\)-day window. This removed the expected within-season cycle but did not detrend the long-term record. Pairwise sample sizes, absolute correlations, variance-inflation factors (VIFs), and domain knowledge informed predictor screening. Same-day temperature, heat index, apparent temperature, vapor-pressure deficit, and event-derived variables were excluded from the association model because they were target-derived, tautological, or used future event information. Correlations were exploratory and were not interpreted as independent effects or causation.

### 2.7 Temperature-trend analysis

For each annual or March–June outcome, we fitted

\[
T_y=\alpha+\beta\left(\frac{y-\bar y}{10}\right)+\varepsilon_y.
\]

The coefficient \(\beta\) is the temperature change in degrees Celsius per decade. Ordinary least squares used heteroskedasticity-and-autocorrelation-consistent (HAC) covariance with three lags. We report \(\beta\), its standard error, 95% CI, two-sided \(p\)-value, and \(R^2\) as descriptive fit. Robustness analyses included Theil–Sen slopes, Mann–Kendall tests, residual lag-1 correlation, and March–June refits excluding 2024. A stacked model with temperature type, decade, and their interaction formally tested whether annual \(T_{\min}\) and \(T_{\max}\) slopes differed; separate statistical significance was not treated as evidence of a difference.

### 2.8 Count-regression analysis

Let \(Y_y\) be the March–June primary persistent heatwave-day count in year \(y\). The Poisson model was

\[
Y_y\sim\operatorname{Poisson}(\mu_y),\qquad
\log(\mu_y)=\beta_0+\beta_1\left(\frac{y-\bar y}{10}\right).
\]

The NB2 alternative used the same mean model and

\[
\operatorname{Var}(Y_y)=\mu_y+\alpha\mu_y^2.
\]

We compared empirical Poisson dispersion, residual deviance, Pearson statistic, Akaike information criterion (AIC), Bayesian information criterion, log-likelihood, and convergence. NB2 was selected when Poisson dispersion exceeded 1.2, NB2 improved AIC by more than 2, and convergence succeeded. The primary effect was

\[
\mathrm{IRR}_{10}=\exp(\beta_1),
\]

with percentage change \(100\{\exp(\beta_1)-1\}\). Diagnostics used selected-distribution randomized quantile residuals and case-deletion parameter distance. Years exceeding \(4/n\) were labeled for leave-one-year-out sensitivity but were not deleted from the primary model. A monthly Poisson model with month effects and the log of observed days as an offset was a secondary specification.

### 2.9 Adjusted antecedent association analysis

The daily binary outcome indicated membership in a primary persistent event. Candidate features used only information before the outcome date: prior-three-day mean relative humidity, prior-seven-day cumulative precipitation, prior-three-day mean wind speed, and prior-three-day mean pressure. Every rolling feature was calculated after a one-day shift. This ordering was enforced by unit tests.

The base logistic generalized estimating equation (GEE) contained centered decade and three Fourier harmonics for day of year. The full model added the four standardized antecedent predictors:

\[
\operatorname{logit}\{P(Y_t=1)\}=\alpha+\gamma D_t+
\sum_{k=1}^{3}\left[a_k\sin\left(\frac{2\pi k d_t}{365.25}\right)+b_k\cos\left(\frac{2\pi k d_t}{365.25}\right)\right]+
\sum_{j=1}^{4}\theta_j Z_{j,t-}.
\]

Here \(D_t\) is centered decade and \(Z_{j,t-}\) is an antecedent predictor standardized over the analyzed hot-season data. Year was the clustering unit; an AR(1) working correlation represented within-year temporal dependence; robust sandwich standard errors were used. We report adjusted ORs per one standard deviation, 95% CIs, two-sided \(p\)-values, and Benjamini–Hochberg \(q\)-values for secondary predictor tests.

Sensitivity analyses changed the outcome to the first day of each event, substituted one-day or seven-day lag structures, and excluded count-influential seasons separately. Predictor VIFs and rank correlations were reviewed. Pressure was not considered robust if its sign changed across outcome definitions.

### 2.10 Chronological validation

For each held-out March–June season from 2000 through 2024, models were trained only on prior years. Standardization parameters and the Youden classification threshold were estimated from training data only. We calculated Brier score, receiver-operating-characteristic area under the curve (ROC-AUC), precision–recall AUC (PR-AUC), sensitivity, specificity, calibration intercept, and calibration slope. ROC-AUC, PR-AUC, and calibration were reported as not estimable for seasons without positive outcomes and were never replaced by zero. All strictly out-of-sample daily probabilities were pooled for an overall assessment.

### 2.11 Forecast validation

Forecasting was prespecified as secondary and is reported in the supplement. Ten rolling origins (2014–2023) compared seasonal naive, monthly climatology, climatology plus linear trend, exponential smoothing, and SARIMAX on identical March–June monthly mean \(T_{\max}\) targets. Metrics were mean absolute error, root-mean-square error (RMSE), bias, mean absolute scaled error,

\[
\mathrm{RMSE}=\sqrt{n^{-1}\sum_{i=1}^{n}(y_i-\hat y_i)^2},
\]

and empirical 80%/95% interval coverage and width. No monthly forecast was converted into exact daily heatwave events.

### 2.12 Software and reproducibility

The analysis used Python 3.14.4 for the audited run, NumPy 2.5.2, pandas 2.3.3, SciPy 1.16.1, statsmodels 0.14.5, scikit-learn 1.7.2, matplotlib 3.11.1, and seaborn 0.13.2. The prespecified seed was 20260901. `make article` regenerates the analytical results, article outputs, tests, validation, notebooks, and Word manuscripts. Source hashes, package versions, runtime, platform, and starting commit are machine-readable.

## 3 Results

### 3.1 Data quality and descriptive characteristics

The record contained 19,316 consecutive dates with no duplicates or missing calendar dates. Nine field values were missing; \(T_{\max}\) was complete. All 52 calendar years from 1972 through 2023 were complete. The 2024 calendar year ended on 18 November, but its March–June season contained all 122 dates. Field-level range and discontinuity flags were retained for transparency and are provided in machine-readable metadata.

Across the full record, mean \(T_{\max}\) was 30.14 °C and the maximum was 40.2 °C on 9 May 2023. Descriptive distributions differed between persistent and non-persistent hot-season days (Table 1; Figure 2). The annual primary count had \(n=53\), mean 5.09 days, variance 58.89, variance-to-mean ratio 11.56, skewness 1.51, range 0–28, and 31 zero seasons (58.5%). The zero-heavy overdispersed distribution made ordinary least squares and equidispersed Poisson inference inadequate as the sole primary approach.

**Table 1.** Data completeness and descriptive statistics for all-year observations, March–June observations, persistent heatwave days, and non-heatwave hot-season days. Source: `results/tables/main/main_table01_data_and_descriptive_statistics.csv`.

### 3.2 Correlations and collinearity assessment

Raw and anomaly correlations differed materially (Figure 3; Table 2). For \(T_{\max}\), the raw/anomaly Spearman correlations were 0.418/0.654 with \(T_{\min}\), -0.555/-0.700 with mean relative humidity, -0.470/-0.526 with precipitation, 0.646/0.678 with shortwave radiation, and -0.489/-0.722 with soil moisture. Removing the 1981–2010 calendar-day climatology therefore strengthened several relationships rather than eliminating them, showing that shared seasonality was not their only source.

The four antecedent predictors had VIFs from 1.44 to 2.26. Although humidity and precipitation had a high Spearman correlation, their linear VIFs remained below 5 and each represented a distinct prespecified antecedent window. Temperature-derived and same-day variables were excluded. These correlations informed model construction but did not provide adjusted or causal estimates.

**Table 2.** Exploratory correlations, predictor screening, and collinearity decisions. Source: `results/tables/main/main_table02_correlations_and_collinearity.csv`.

### 3.3 Temperature trends

Annual mean \(T_{\max}\) increased by 0.192 °C per decade (95% CI 0.130–0.253; \(p<0.001\); \(R^2=0.405\)), and annual mean \(T_{\min}\) increased by 0.208 °C per decade (95% CI 0.152–0.264; \(p<0.001\); \(R^2=0.543\)). The annual mean day–night range slope was -0.016 °C per decade (95% CI -0.069–0.036; \(p=0.538\)). The formal \(T_{\min}-T_{\max}\) slope difference was 0.016 °C per decade (95% CI -0.034–0.067; \(p=0.527\)); the data therefore did not show that minimum temperature warmed faster than maximum temperature.

March–June mean \(T_{\max}\) increased by 0.165 °C per decade through 2024 (95% CI 0.044–0.287; \(p=0.0076\)), and March–June mean \(T_{\min}\) increased by 0.153 °C per decade (95% CI 0.070–0.236; \(p<0.001\)). Excluding the 2024 hot season yielded corresponding slopes of 0.151 °C (95% CI 0.034–0.269) and 0.137 °C (95% CI 0.060–0.214) per decade. Theil–Sen and Mann–Kendall analyses supported the direction of the principal temperature trends (Table 4; Figure 4).

**Table 4.** Temperature trends, nonparametric robustness estimates, slope contrast, and 2024 endpoint sensitivity. Source: `results/tables/main/main_table04_temperature_trends.csv`.

### 3.4 Heatwave definition sensitivity and event climatology

During March–June, the one-day absolute definition identified 369 days in 128 events, while requiring three consecutive days identified 270 days in 49 events (Table 3; Figure 5). Primary events had median duration 5 days, mean duration 5.51 days, maximum duration 15 days, and peak occurrence in April. The 1981–2010 90th-percentile definition identified 490 March–June days in 93 events; the 95th-percentile definition identified 248 days in 52 events; and the compound 90th-percentile day/night definition identified 228 days in 71 events.

Count-trend conclusions depended on definition. IRRs per decade were 1.018 (95% CI 0.799–1.298) for one-day absolute exceedances, 1.020 (0.775–1.344) for two-day persistence, 1.029 (0.727–1.455) for the primary three-day definition, 1.069 (0.842–1.358) for 90th-percentile three-day events, and 1.052 (0.779–1.420) for 95th-percentile three-day events. Only the compound day/night definition had a CI excluding 1: IRR 1.320 (95% CI 1.043–1.672; nominal \(p=0.0209\)). Because this was one of six definition-specific trend tests, its Benjamini–Hochberg-adjusted \(q\)-value was 0.126; it is therefore an exploratory signal rather than confirmatory evidence. With a 1991–2020 reference, its IRR was 1.331 (95% CI 1.021–1.735), while the relative \(T_{\max}\)-only trends remained uncertain. Thus, estimated heatwave occurrence was definition-dependent, but no definition-specific trend survived multiplicity adjustment.

**Table 3.** March–June heatwave statistics and count trends under alternative definitions and reference periods. Source: `results/tables/main/main_table03_definition_sensitivity.csv`.

### 3.5 Poisson and negative-binomial model selection

The Poisson model had Pearson dispersion 11.64, residual deviance 563.9, and AIC 659.8. The NB2 model converged and had AIC 245.2, log-likelihood -119.6, deviance-style statistic 43.4, and Pearson statistic 20.3 (Table 5). The prespecified selection rule therefore favored NB2. Randomized quantile residuals were broadly consistent with the fitted distribution, while case-deletion screening identified 1979, 2023, and 2024 as influential seasons (Figure 6). These years were retained.

**Table 5.** Poisson and NB2 comparison for annual March–June persistent heatwave-day counts. Source: `results/tables/main/main_table05_poisson_nb_comparison.csv`.

### 3.6 Primary persistent heatwave-day trend

The selected NB2 model estimated a log-rate coefficient of 0.0284 per decade (standard error 0.1768), equivalent to IRR 1.029 per decade (95% CI 0.727–1.455; \(p=0.872\)) and an estimated 2.9% change per decade. Because the CI included 1 broadly, the data did not provide evidence of a monotonic change in the primary persistent heatwave-day rate (Table 6). Excluding influential seasons one at a time produced IRRs near 1 and did not change that conclusion.

**Table 6.** Distributional evidence and selected NB2 estimate for the primary count outcome. Source: `results/tables/main/main_table06_primary_count_model.csv`.

### 3.7 Adjusted antecedent meteorological associations

After adjustment for seasonal Fourier terms and decade, the odds of a persistent heatwave day were lower per one-standard-deviation increase in prior-three-day mean relative humidity (OR 0.307, 95% CI 0.236–0.399; \(q<0.001\)) and prior-seven-day cumulative precipitation (OR 0.140, 95% CI 0.064–0.307; \(q<0.001\)). Higher prior-three-day wind speed was positively associated (OR 1.337, 95% CI 1.107–1.614; \(q=0.003\)). Prior-three-day pressure was imprecise (OR 0.656, 95% CI 0.404–1.065; \(q=0.088\)). One standard deviation corresponded to 9.19 relative-humidity percentage points, 55.83 mm of seven-day precipitation, 3.23 wind-speed units as recorded in the dataset, and 4.21 pressure units.

The fitted AR(1) working-correlation parameter was 0.576. When the outcome was restricted to 49 event-onset days, humidity (OR 0.464, 95% CI 0.338–0.638), precipitation (OR 0.298, 95% CI 0.097–0.920), and wind speed (OR 1.526, 95% CI 1.121–2.076) retained their directions. Pressure changed direction (OR 1.688, 95% CI 1.107–2.574), so it was not interpreted as robust. Excluding 1979, 2023, or 2024 separately preserved the three principal directions (Table 8; Figure 7).

**Table 7.** Adjusted logistic GEE associations per one-standard-deviation antecedent predictor. Source: `results/tables/main/main_table07_adjusted_associations.csv`.

### 3.8 Blocked validation

Across 3,050 strictly out-of-sample days from 25 held-out hot seasons, 125 were persistent heatwave days. The seasonal-trend base GEE had Brier score 0.0382, ROC-AUC 0.772, PR-AUC 0.198, calibration intercept 0.396, and calibration slope 0.895. The antecedent GEE improved Brier score to 0.0333, ROC-AUC to 0.924, and PR-AUC to 0.473; its calibration intercept was 1.512 and slope 1.136, indicating systematic underprediction despite improved ranking and overall probability error. At origin-specific training thresholds, pooled sensitivity was 0.664 and specificity 0.919. Seasons without positive outcomes contributed to Brier score but had undefined discrimination and calibration metrics, which were not averaged as zeros. The calibration limitation precludes an operational-warning claim.

**Table 8.** Held-out-season and pooled validation plus association sensitivity. Source: `results/tables/main/main_table08_blocked_validation.csv`.

### 3.9 Forecast validation

Forecasting did not meet the criterion for inclusion as a central analysis. Climatology plus linear trend had the lowest mean rolling-origin RMSE (1.114 °C), closely followed by exponential smoothing (1.120 °C) and monthly climatology (1.175 °C). SARIMAX RMSE was 1.607 °C, and its 95% intervals were nearly twice as wide as those from climatology plus trend. The climatology-plus-trend 80% and 95% coverages were 0.675 and 0.875, showing imperfect calibration. These results are reported in the supplement; no exact 2025–2029 heatwave count is presented.

## 4 Discussion

### 4.1 Principal findings

Dhaka's annual and hot-season temperatures warmed over the observation period, but the primary persistent heatwave-day count did not exhibit a detectable monotonic trend. That contrast is not paradoxical. A shift in the center of the temperature distribution can coexist with large interannual variability, zero-heavy counts, threshold sensitivity, and limited power for rare persistent events. The NB2 CI is compatible with both decreases and increases of practical interest, so the correct interpretation is uncertainty—not proof of no change and not evidence of a clear increase.

The adjusted antecedent analysis contributed separate evidence. Lower humidity and precipitation and higher wind speed preceded persistent heatwave days after seasonal and long-term adjustment, and these directions remained in the event-onset analysis and after influential-season exclusions. The full model also improved strictly out-of-sample discrimination and probability error. These results support meteorological monitoring value, but they do not isolate causal effects. Synoptic circulation, radiation, land-surface state, and measurement dependencies may jointly produce the observed associations.

### 4.2 Why heatwave definition matters

The compound day/night definition was the only primary-reference definition with a nominal positive count trend, but it did not survive adjustment across the six definition tests. Absolute one-day and persistence-based definitions and relative \(T_{\max}\)-only definitions were uncertain. This pattern shows why “heatwaves increased” is incomplete without stating the threshold, duration, reference period, season, day/night variables, and multiplicity context. It also connects with prior Bangladesh work emphasizing compound high minimum and maximum temperatures (Nissan et al. 2017) and percentile heatwaves (Tabassum et al. 2024; Molla et al. 2025). The present contribution is not to choose one universal definition, but to show which conclusions are stable and which are conditional.

### 4.3 Antecedent meteorological conditions

Dryer antecedent conditions are physically plausible during suppressed pre-monsoon rainfall and enhanced surface heating. Nissan et al. (2017) likewise described below-normal precipitation and soil moisture before Bangladesh heatwaves, and Mallik et al. (2024) discussed pressure and wind patterns associated with threshold days. In the current analysis, however, humidity and precipitation were correlated and the data came from one series. Odds ratios represent conditional associations per sample standard deviation, not intervention effects. Wind's positive association may reflect regional advection rather than locally generated wind effects. Pressure's sign reversal between persistent-day and onset models demonstrates the danger of elevating an unstable coefficient into a mechanistic claim.

### 4.4 Methodological implications

Raw heatmaps were useful for identifying covariance and possible collinearity but could not estimate independent relationships. De-seasonalization showed that several strong correlations were not explained solely by the March–June cycle. Direct count diagnostics were equally important: Poisson dispersion above 11 and a more than 400-point AIC difference made an equidispersed count model untenable. GEE accounted for repeated daily observations within years, while blocked validation showed whether association estimates translated into future-season discrimination. These safeguards change the role of apparently strong associations from explanatory claims to validated but observational signals.

### 4.5 Forecasting limitations

Complexity did not guarantee better forecasting. SARIMAX underperformed climatology plus trend and required much wider intervals. Monthly mean-temperature skill also cannot validate the consecutive daily threshold runs needed for event forecasts. Accordingly, the analysis does not report deterministic annual heatwave counts for 2025–2029. Future prediction would require a validated daily multivariate simulation or event-occurrence framework, external stations, forecast-origin climatologies, and calibration across many more independent seasons.

### 4.6 Practical relevance

Primary persistent events were concentrated in April and May, and antecedent weather improved prediction beyond seasonal timing and long-term trend. These findings may inform monitoring priorities: sustained absolute heat, limited rainfall and humidity, and changing wind conditions can be tracked jointly. The study does not demonstrate that any monitoring system reduces health impacts, and preparedness decisions should integrate health, exposure, vulnerability, and operational forecast data not present here.

### 4.7 Strengths and limitations

Strengths include a long continuous daily record, explicit partial-year handling, reproducible event construction, six definitions, two climatological reference periods, direct count regression, model-consistent diagnostics, formal slope comparison, leakage-safe lag construction, repeated-measures inference, event-onset sensitivity, and chronological validation. All principal tables and figures are generated from code and linked to claims.

Several limitations constrain interpretation. The study represents one location and lacks a rural comparator, so it cannot quantify urban amplification. Station coordinates, relocations, instrumentation changes, observation practices, and homogenization metadata were unavailable in the repository; an unmodeled discontinuity could affect trends. The dataset has no health, mortality, morbidity, exposure, or vulnerability outcome. The primary definition yielded only 49 events, limiting power and the complexity of event-level models. Meteorological variables may share upstream algorithms or measurement errors. GEE associations are exploratory and observational. The 2024 calendar year was partial, although its hot season was complete. Finally, monthly forecast validation cannot establish long-horizon event-count skill.

## 5 Conclusions

Dhaka experienced statistically detectable annual and March–June warming from 1972–2024. Heatwave days were concentrated within the March–June hot-season window, but estimated count trends depended on definition. The selected NB2 model did not show a clear monotonic change in the prespecified persistent heatwave-day rate; the secondary compound day/night definition showed a nominal positive trend that did not survive adjustment across definitions. Lower antecedent humidity and precipitation and higher antecedent wind speed were adjusted, temporally validated associations, not causal effects. Simple temperature baselines equaled or outperformed more complex forecasting models, and exact five-year heatwave counts were unsupported. A scientifically defensible interpretation therefore separates clear background warming from uncertain persistent-event counts and makes every heatwave conclusion conditional on an explicit definition.

## Declarations

**Data availability.** The analytical repository contains the supplied source data, source hashes, code, configuration, tests, and generated outputs. The authors must confirm redistribution rights and provide the final repository DOI or archive link.

**Code availability.** The reproducible command is `make article`. The authors must provide the final branch/tag and permanent archive.

**Funding.** [Insert verified funding statement.]

**Competing interests.** [Insert author-approved declaration.]

**Author contributions.** [Insert CRediT roles after author approval.]

**Ethics approval.** The meteorological analysis itself uses no individual-level data. The authors must separately verify whether any legacy survey material is excluded and whether a statement is required by the journal.

## Figure captions

**Figure 1. Study area and data coverage.** Bangladesh and Dhaka District administrative boundaries and annual daily-record completeness. No station point is shown because exact coordinate provenance was unavailable. Calendar year 2024 is partial; March–June 2024 is complete. Administrative boundaries do not imply official endorsement.

**Figure 2. Descriptive climatology and primary count distribution.** (A) Daily maximum-temperature distribution. (B) Monthly mean maximum and minimum temperatures. (C) Annual March–June count distribution for days in events with \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive days, with overdispersion statistics. (D) March–June maximum-temperature distributions and the 36 °C threshold.

**Figure 3. March–June correlation analysis.** (A) Raw Spearman correlations. (B) Spearman correlations after subtracting smoothed 1981–2010 calendar-day means. Correlations describe pairwise covariance and possible collinearity. They do not estimate independent effects or establish causation.

**Figure 4. Long-term temperature trends.** (A) Complete-year annual mean \(T_{\max}\) through 2023. (B) Complete-year annual mean \(T_{\min}\). (C) Complete March–June mean \(T_{\max}\) through 2024. Lines are OLS fits and bands are 95% CIs based on HAC covariance. (D) Per-decade slope estimates and 95% CIs, including the formal annual \(T_{\min}-T_{\max}\) contrast.

**Figure 5. Heatwave definition sensitivity.** (A) March–June annual counts under selected definitions. (B) Event-duration distribution for the primary three-day absolute definition only. (C) Monthly distribution of primary persistent heatwave days. (D) Count-model IRRs per decade and 95% CIs for six definitions. The log-scale reference line is IRR 1.

**Figure 6. Primary NB2 count model and diagnostics.** (A) Observed March–June persistent heatwave-day counts, fitted mean, and parameter-based 95% mean CI. (B) Selected-distribution randomized quantile residuals versus fitted means. (C) randomized-residual Q–Q plot. (D) case-deletion parameter distance by year; labels identify years exceeding the \(4/n\) screening threshold. Influential years remain in the primary model.

**Figure 7. Adjusted antecedent associations and validation.** (A) Logistic GEE adjusted ORs and 95% CIs per one-standard-deviation antecedent predictor. (B) model-implied probabilities across humidity and precipitation values at a representative pre-monsoon calendar day, holding other standardized variables at their means. (C) Brier score by held-out hot season. (D) pooled strictly out-of-sample discrimination and probability error for the seasonal-trend base and full antecedent models. Brier score is lower-is-better, unlike AUC metrics.

## References

Chaki S, Samad MA, Mallik MAK, Hassan SMQ (2025) Forecasting human heat stress: Insights from observations and WRF simulations during Bangladesh heatwaves. *PLOS Climate* 4(8):e0000690. https://doi.org/10.1371/journal.pclm.0000690

Farukh MA, Brahma PP, Hossain MS, Hoque MJ, Sejuti SI, Shammy US, Arefin KS (2026) Climatological assessment of pre-monsoon heatwave days in Bangladesh and their relationship to Indo-Pacific circulation anomalies. *Natural Hazards* 122(1):1–32. https://doi.org/10.1007/s11069-025-07746-7

Islam MY, Mohiuddin M, Tanvir Hossain K, Salauddin M, Farin S (2024) Trend of heat waves in Dhaka Metropolitan City and its impact on livelihood and health of exposed people. *Arabian Journal of Geosciences* 17:232. https://doi.org/10.1007/s12517-024-12027-x

Khatun K, Samad MA, Rashid MB (2017) Time Series Analysis of Temperature and Rainfall Data of Dhaka Division. *Dhaka University Journal of Science* 65(2):119–123. https://doi.org/10.3329/dujs.v65i2.54519

Mallik MAK, Sultana A, Islam MK, Akter MY, Alam E, Islam ARMT (2024) Are hotspots and frequencies of heat waves changing over time? Exploring causes of heat waves in a tropical country. *PLOS ONE* 19:e0300070. https://doi.org/10.1371/journal.pone.0300070

Molla MAM, Hassan QK, Dewan A (2025) Unveiling heatwave events in Bangladesh: Insights from observational records and ERA5 reanalysis data. *Climate Services* 40:100609. https://doi.org/10.1016/j.cliser.2025.100609

Nissan H, Burkart K, Coughlan de Perez E, Van Aalst M, Mason S (2017) Defining and Predicting Heat Waves in Bangladesh. *Journal of Applied Meteorology and Climatology* 56(10):2653–2670. https://doi.org/10.1175/JAMC-D-17-0035.1

Rashid GM, Hossain MMT, Akhter MAE, Mallik MAK (2024) A Study on the Heat Wave Conditions over Bangladesh During 1990–2019. *Journal of Engineering Science* 14(2):59–67. https://doi.org/10.3329/jes.v14i2.71227

Tabassum A, Park K, Seo JM, Han JY, Hong SH, Baik JJ (2024) Characteristics of the Urban Heat Island in Dhaka, Bangladesh, and Its Interaction with Heat Waves. *Asia-Pacific Journal of Atmospheric Sciences* 60:479–493. https://doi.org/10.1007/s13143-024-00362-8
