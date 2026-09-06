# Long-Term Warming, Definition-Dependent Heatwaves, and Short-Lead Persistent-Hot-Window Prediction in Dhaka, Bangladesh, 1972–2024

**Article type:** Original Article

**Authors:** Awnon Bhowmik¹; Goutam Saha²

**Affiliations:** ¹College of Engineering & Computer Science, Colorado Technical University, Colorado Springs, CO 80907, USA; ²Department of Mathematics, International University of Business Agriculture and Technology, Dhaka 1230, Bangladesh

**Corresponding author:** Goutam Saha (gsahamath@du.ac.bd)

**Author emails:** Awnon Bhowmik (awnonbhowmik@outlook.com); Goutam Saha (gsahamath@du.ac.bd)

## Abstract

Long-term warming does not guarantee that heatwave counts increase uniformly, because event estimates depend on thresholds, persistence requirements, and day/night criteria. We analyzed 19,316 consecutive daily meteorological records for Dhaka from 1972–2024. Temperature trends were estimated using ordinary least squares with heteroskedasticity-and-autocorrelation-consistent standard errors, Theil–Sen slopes, and Mann–Kendall tests; six heatwave definitions and Poisson versus NB2 count models were compared. Leakage-safe antecedent associations were estimated with logistic generalized estimating equations. A separate benchmark predicted whether all three days beginning 1, 3, or 7 days after an issue date would have \(T_{\max}\geq36\,{}^\circ\mathrm{C}\). Calendar-only, temperature-history, and full-meteorology information sets were compared across five imbalance-aware classifiers using nested chronological tuning, temporal calibration, and held-out seasons from 2014–2024. Annual mean \(T_{\max}\) increased by 0.192 °C per decade (95% confidence interval [CI] 0.130–0.253), whereas the primary persistent-day count trend was uncertain (incidence-rate ratio 1.029 per decade, 95% CI 0.727–1.455). At one-day lead, temperature-history logistic regression had the strongest overall average precision (AP 0.522; Brier score 0.0276). Adding meteorology improved XGBoost within-family AP by 0.138 (paired held-out-year 95% interval 0.012–0.188) and Brier score by 0.0040, but full-meteorology XGBoost remained below temperature-history logistic regression (AP 0.427). XGBoost full-meteorology AP was 0.165 and 0.216 at 3- and 7-day leads. On 1,215 eligible onset-risk dates representing 15 positive spells, it detected 11 spells but produced 182 false alerts (precision 0.057). Dhaka warmed detectably, but persistent-event trends were definition-dependent, additional meteorology produced model-family-specific rather than universal gains, and the limited event count and unresolved data latency preclude operational-warning claims.

**Keywords:** extreme temperature; heatwave definition; negative binomial regression; machine learning; chronological validation; Bangladesh

## 1 Introduction

Heatwaves are sustained periods of unusually or operationally high temperature, yet there is no single definition that is optimal for every scientific or decision context (Boni et al., 2023; Perkins & Alexander, 2013; Robinson, 2001). Absolute thresholds can align with local warning practices, percentile thresholds adjust to the seasonal temperature distribution, persistence requirements distinguish isolated hot days from sustained events, and combined maximum–minimum criteria capture limited nighttime relief (Russo et al., 2014; Zscheischler et al., 2018). These choices alter the number, duration, and timing of identified events and can therefore alter estimated trends. Regional evidence shows widespread increases in heatwave frequency, duration, and cumulative heat, while also demonstrating that metric choice and internal variability complicate comparisons (Perkins-Kirkpatrick & Lewis, 2020). Anthropogenic warming has increased the likelihood or burden of many extreme-heat outcomes, but attribution of a global or regional signal does not itself establish a monotonic trend in every local threshold count (Diffenbaugh et al., 2017; Intergovernmental Panel on Climate Change [IPCC], 2021; Vicedo-Cabrera et al., 2021). A credible long-term analysis must therefore distinguish background warming from changes in a prespecified heatwave outcome rather than assuming that the two are equivalent.

Bangladesh experiences its most consequential dry heat during the pre-monsoon transition, when suppressed rainfall, land-surface dryness, and regional circulation can favor high temperatures (Karmakar & Das, 2020). Projected humid heat is also a concern across South Asia (Im et al., 2017; Saeed et al., 2021; Sharma et al., 2022), while global observations show that short-duration combinations of heat and humidity have approached physiological limits (Mora et al., 2017; Raymond et al., 2020). Experimental evidence further shows that physiological limits depend on exposure conditions and cannot be reduced to one universal wet-bulb threshold (Vecellio et al., 2022). Those contextual findings do not permit physiological tolerance to be inferred from the daily meteorological associations analyzed here. Nissan et al. (2017) developed a Bangladesh heatwave definition combining high daytime and nighttime temperatures and described precipitation and soil-moisture antecedents. Multi-station studies have subsequently documented threshold-day climatology and spatially heterogeneous trends (Mallik et al., 2024; Rashid et al., 2024), while newer studies have examined percentile definitions, reanalysis agreement, circulation, and short-lead heat-stress prediction (Chaki et al., 2025; Farukh et al., 2026; Molla et al., 2025). Related work in India illustrates the wider regional importance of definition, circulation, and attribution choices (Ravindra et al., 2024; Singh et al., 2024). These studies establish the national and regional meteorological relevance of definition choice and persistence.

Dhaka-specific studies provide important but different evidence. Khatun et al. (2017) estimated temperature and rainfall trends across Dhaka Division. Islam et al. (2024) combined land-surface temperature, meteorological descriptions, and survey evidence for Dhaka Metropolitan City. Bangladesh-wide and Dhaka-focused studies have examined surface urban heat islands, metropolitan vulnerability, and adaptation among informal workers (Adnan et al., 2024; Dewan et al., 2021; Shahrujjaman et al., 2025). Tabassum et al. (2024) used an urban–rural station comparison and reanalysis to evaluate the urban heat island and its interaction with percentile heatwaves. Separate time-series studies linked temperature variability to cardiorespiratory emergency visits and defined heatwave days to diarrhoeal hospitalization in Dhaka (Haque et al., 2024; Rahman et al., 2022). Those designs should not be conflated with the present single-series dataset, which cannot measure an urban–rural contrast, human vulnerability, or health impacts. The remaining analytical gap is narrower: few Dhaka-focused studies have combined alternative event definitions, direct count regression, adjusted antecedent meteorological associations, and chronological model validation within one long daily series.

The distinction between ambient temperature, urban amplification, and human exposure is important. Across cities, background climate, population, urban form, and urbanization can modify heat-island magnitude, nighttime heat, and population exposure (Liu et al., 2022; Manoli et al., 2019; Tuholske et al., 2021; Zhao et al., 2014). Urbanization has also been associated with more persistent compound or nighttime heat in other settings (Ma & Yuan, 2021; Sarangi et al., 2021; Shi et al., 2021). These studies motivate careful local work, but they do not justify attributing trends in one Dhaka meteorological series to urbanization.

Extreme heat is consequential for health, although the present study has no health outcome. Mortality effects have been documented in Shanghai and Karachi and during recent European summers (Ballester et al., 2023; Gallo et al., 2024; Ghumman & Horney, 2016; Huang et al., 2010), and a systematic review shows that cardiovascular findings depend in part on heatwave definition (Nawaro et al., 2023). This evidence motivates explicit, reproducible exposure definitions while remaining external context rather than an outcome of the present analysis.

Predictive heatwave studies span very different targets and lead times. Kan et al. (2025) classified monthly heatwave occurrence at Swedish spatial pixels at one-to-five-month leads using remote-sensing histories, imbalance-aware models, forward-time validation, and SHAP. Choudary V et al. (2025) compared multiple classifier families for next-day heatwave-event prediction from a single Chennai weather series. Those designs motivate chronological classifier comparison and out-of-sample explanation, but their thresholds, spatial supports, and forecast horizons cannot be transferred directly to Dhaka. Here, the predictive estimand is deliberately narrower: whether a future three-day persistent-hot window begins 1, 3, or 7 days after the latest observed day.

Descriptive counts and pairwise correlations alone cannot fill this gap. Annual heatwave counts are discrete, zero-heavy, and potentially overdispersed; ordinary least squares is therefore not the natural primary model. Raw meteorological correlations also mix within-season covariance with the shared seasonal cycle and do not estimate independent associations. Repeated daily outcomes require attention to within-year dependence, and predictive performance must be evaluated on future seasons rather than by random train–test splitting. These considerations motivate direct count modeling, de-seasonalized exploration, leakage-safe lag construction, generalized estimating equations, model diagnostics, and blocked validation.

We addressed four questions. First, how did annual and March–June maximum and minimum temperatures change in Dhaka during 1972–2024? Second, how did heatwave frequency, duration, intensity, and estimated trend vary across operational, persistence-based, percentile, and compound day/night definitions? Third, which antecedent meteorological conditions were associated with persistent heatwave days after accounting for seasonality, long-term change, collinearity, and within-year temporal dependence? Fourth, did recent meteorological history improve prediction of a forthcoming three-day persistent-hot window beyond seasonality and recent temperature alone, and how did performance change with lead time? We prespecified March–June as the hot-season analysis window; because June can include monsoon onset, this label does not imply that all four months share one meteorological regime. The primary descriptive and count outcome was days belonging to events with \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive days. The primary count-model null hypothesis was \(H_0:\mathrm{IRR}=1\) per decade; the predictive primary contrast was the difference in held-out AP between the full-meteorology and temperature-history information sets at one-day lead.

## 2 Methods

### 2.1 Study design and setting

We conducted a retrospective, single-location meteorological time-series study for Dhaka, Bangladesh. The supplied dataset is described as daily Dhaka meteorology, but the repository does not contain a verified station identifier or exact coordinate. We therefore show administrative geography without inferring a station point and do not estimate urban-heat-island intensity. The analysis is observational and evaluates statistical trends and associations, not intervention effects or causal mechanisms.

### 2.2 Data source and temporal coverage

The immutable daily CSV contains 19,316 dates from 1 January 1972 through 18 November 2024 and 32 source fields. Variables used here include daily maximum, minimum, and mean air temperature; precipitation; relative humidity; wind speed; cloud cover; sunshine duration; shortwave and longwave radiation; pressure; evapotranspiration; soil temperature; and soil moisture. Source-file SHA-256 hashes are recorded in `results/metadata/source_data_hashes.json`. Column labels and the supplied materials are consistent with a Meteoblue-formatted historical export (Meteoblue, 2024), but the repository does not establish whether the values are station observations, model output, reanalysis, or a blended product. Provider, site identity, coordinates, release latency, and homogenization history require confirmation before submission.

### 2.3 Data quality control

Dates were parsed, sorted, and compared with a complete daily calendar. We evaluated duplicate dates, missing dates, field-level missingness, leap days, physical-range flags, and abrupt day-to-day changes. The record had no duplicate or missing dates and nine missing field values. Flagged values were retained because no independent quality-control metadata justified deletion. Calendar year 2024 was incomplete and was excluded from complete-year annual means. Every March–June season, including 2024, contained the expected 122 dates and was eligible for hot-season analyses. Leap days remained in descriptive analyses. For calendar-day climatologies, dates after February in leap years were mapped to a 365-day climatological calendar; 29 February used the mean of 28 February and 1 March thresholds.

### 2.4 Heatwave definitions and event construction

The primary definition was daily \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive calendar days. Sensitivity definitions were: (A) \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least one day; (B) the same threshold for at least two consecutive days; (C) the primary three-day definition; (D) calendar-day 90th-percentile \(T_{\max}\) for at least three days; (E) calendar-day 95th-percentile \(T_{\max}\) for at least three days; and (F) concurrent calendar-day 90th-percentile \(T_{\max}\) and \(T_{\min}\) for at least two days. The last sensitivity recognizes that concurrent daytime and nighttime extremes can represent a distinct compound exposure (Zhang et al., 2020; Zscheischler et al., 2018).

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

where \(\bar X_{1981:2010}(d_t)\) was the smoothed calendar-day mean within a circular \(\pm7\)-day window. This removed the expected within-season cycle but did not detrend the long-term record. Pairwise sample sizes, absolute correlations, variance-inflation factors (VIFs), and domain knowledge informed predictor screening. Same-day temperature, heat index, apparent temperature, vapor-pressure deficit, and event-derived variables were excluded from the association model because they were target-derived, tautological, or used future event information. Heat index also requires internally consistent temperature--humidity pairing and is a distinct exposure metric rather than an antecedent predictor of a temperature-defined outcome (Anderson et al., 2013; Lu & Romps, 2022). Correlations were exploratory and were not interpreted as independent effects or causation.

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

For each held-out March–June season from 2000 through 2024, models were trained only on prior years. This temporally ordered design follows established guidance that forecast evaluation must preserve ordering and assess genuine future observations rather than rely on random resampling (Bergmeir et al., 2018; Tashman, 2000). Standardization parameters and the Youden classification threshold were estimated from training data only. We calculated Brier score, receiver-operating-characteristic area under the curve (ROC-AUC), precision–recall AUC (PR-AUC), sensitivity, specificity, calibration intercept, and calibration slope. ROC-AUC, PR-AUC, and calibration were reported as not estimable for seasons without positive outcomes and were never replaced by zero. All strictly out-of-sample daily probabilities were pooled for an overall assessment.

### 2.11 Future persistent-hot-window prediction

Issue time was the end of day \(t\), after that day's measurements were assumed available. This is a retrospective availability assumption because source-product release latency was not documented. Predictors summarized the seven observed days \(t-6\) through \(t\). The binary target was 1 when \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) on all three dates \(t+h,t+h+1,t+h+2\), and 0 otherwise when the complete target window was observed. The primary lead was \(h=1\) day, with direct separate models for \(h=3\) and \(h=7\); every target window had to remain within March–June. This endpoint can describe continuation of existing heat and is therefore called prediction of a future three-day persistent-hot window, not event onset or a one-to-five-month forecast.

We compared three nested information sets. S0 contained target-date seasonal Fourier terms and a prespecified long-term time term. S1 added available temperature history, including latest, three-day, and seven-day maximum and minimum temperature summaries, seven-day variability and trend, and the observed hot-day run length at issue time. S2 added parsimonious summaries of relative humidity, precipitation, wind, pressure, cloud cover, shortwave radiation, and soil moisture. Predictors derived from the future target window were prohibited. Latest values, three- and seven-day means, seven-day variability or trend, and seven-day precipitation totals were used selectively rather than expanding every possible lag and interaction.

Models were regularized logistic regression, class-weighted support-vector classification, class-weighted random forest, balanced random forest, and XGBoost; calendar logistic, seasonal probability, observed-temperature transition, and always-negative models served as simpler baselines (Breiman, 2001; Chen & Guestrin, 2016; Choudary V et al., 2025; Kan et al., 2025). Outer evaluation held out each complete March–June season from 2014–2024 and retained seasons without positive windows. Within every outer training set, forward-time inner folds selected hyperparameters by average precision (AP); imputation, scaling, high-correlation filtering, calibration, and alert-threshold selection were fitted using training data only. AP was the primary discrimination metric because only 56 of 1,320 held-out issue dates were positive; ROC-AUC, Brier score, balanced accuracy, recall, precision, and specificity were secondary (Saito & Rehmsmeier, 2015). Calibration intercepts, slopes, and reliability curves were examined because good discrimination does not guarantee reliable probabilities (Van Calster et al., 2019).

The primary comparison was S2 minus S1 within model family at \(h=1\). We used 2,000 paired bootstrap resamples of complete held-out years to obtain percentile intervals while preserving within-year dependence. These intervals measure sensitivity to which complete seasons were represented; they do not make overlapping three-day windows independent. Secondary analyses evaluated the same comparison at longer leads, a training-cutoff-specific calendar-day 90th-percentile target, and 14 rather than seven history days. Onset performance at \(h=1\) was evaluated only when \(T_{\max}(t)<36\,{}^\circ\mathrm{C}\), with eligible dates, distinct spells, recall, precision, and false alerts reported separately.

For the leading S2 tree family, SHAP values were computed only for outer-test observations using training-only background samples, and grouped seven-row block permutations provided a complementary diagnostic (Lundberg & Lee, 2017). SHAP values describe contributions to this fitted model's uncalibrated output; correlated predictors can share attribution, and neither SHAP nor permutation importance identifies causal meteorological effects. The tree explanation was secondary to the overall benchmark because the highest-AP model could belong to a different family.

### 2.12 Secondary monthly forecast validation

As a target-mismatch sensitivity, ten rolling origins (2014–2023) compared seasonal naive, monthly climatology, climatology plus linear trend, exponential smoothing, and a conventional SARIMAX specification (Box et al., 2015) on March–June monthly mean \(T_{\max}\). Metrics included mean absolute error, root-mean-square error (RMSE), bias, mean absolute scaled error,

\[
\mathrm{RMSE}=\sqrt{n^{-1}\sum_{i=1}^{n}(y_i-\hat y_i)^2},
\]

and empirical 80%/95% interval coverage and width. These monthly results were not used to claim persistent-window or event-count skill.

### 2.13 Software and reproducibility

The benchmark run used Python 3.14.4, NumPy 2.5.2, pandas 2.3.3, SciPy 1.16.1, statsmodels 0.14.5, scikit-learn 1.7.2, matplotlib 3.11.1, seaborn 0.13.2, imbalanced-learn 0.14.0, XGBoost 3.1.1, and SHAP 0.52.0. The prespecified seed was 20260901. `make article` regenerates the original analytical outputs, tests, notebooks, and manuscripts; `make two-paper-benchmark` and `make validate-two-paper` regenerate and validate the future-window analysis. Source hashes, configuration snapshots, split manifests, package versions, runtime, platform, and starting commit are machine-readable.

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

Count-trend conclusions depended on definition. IRRs per decade were 1.018 (95% CI 0.799–1.298) for one-day absolute exceedances, 1.020 (0.775–1.344) for two-day persistence, 1.029 (0.727–1.455) for the primary three-day definition, 1.069 (0.842–1.358) for 90th-percentile three-day events, and 1.052 (0.779–1.420) for 95th-percentile three-day events. Only the compound day/night definition had a CI excluding 1: IRR 1.320 (95% CI 1.043–1.672; nominal \(p=0.0209\)). Because this was one of six definition-specific trend tests, the Benjamini–Hochberg-adjusted result was \(q=0.126\); it is therefore an exploratory signal rather than confirmatory evidence. With a 1991–2020 reference, its IRR was 1.331 (95% CI 1.021–1.735), while the relative \(T_{\max}\)-only trends remained uncertain. Thus, estimated heatwave occurrence was definition-dependent, but no definition-specific trend survived multiplicity adjustment.

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

### 3.9 Future persistent-hot-window prediction

The future-window benchmark contained 1,320 strictly held-out issue dates from 2014–2024 at each lead, including event-free seasons; 56 dates (4.24%) were positive. At the primary one-day lead, temperature-history logistic regression (S1) was the strongest overall model, with AP 0.522, ROC-AUC 0.927, Brier score 0.0276, balanced accuracy 0.849, recall 0.839, and precision 0.209. It exceeded the observed-temperature transition baseline (AP 0.357), seasonal probability baseline (0.178), calendar logistic model (0.087), and always-negative prevalence benchmark (0.042). Thus, most usable information came from recent observed temperature rather than model complexity or calendar timing alone (Table 9; Figure 8A).

The effect of adding non-temperature meteorology was model-family-specific. At \(h=1\), S2 improved XGBoost AP from 0.288 to 0.427, a paired held-out-year difference of 0.138 (95% interval 0.012–0.188), and reduced Brier score by 0.0040 (95% interval 0.0002–0.0086 improvement). Weighted random forest showed an AP gain of 0.107 (95% interval 0.035–0.172) and a Brier improvement of 0.0034 (95% interval 0.0005–0.0066). Conversely, adding S2 predictors to logistic regression reduced AP by 0.077 (95% interval 0.016–0.168 lower) and worsened Brier score by 0.0029 (95% interval 0.0004–0.0062). Neither XGBoost S2 (AP 0.427) nor weighted-random-forest S2 (0.377) surpassed logistic S1 (0.522), so the evidence supports within-family gains for two tree ensembles rather than a universal benefit from extra meteorology (Figure 8B).

Predictive skill was concentrated at short lead. Logistic S1 AP declined from 0.522 at \(h=1\) to 0.256 at \(h=3\) and 0.131 at \(h=7\). XGBoost S2 AP was 0.427, 0.165, and 0.216 at the same leads. The nonmonotonic XGBoost values do not demonstrate recovery of longer-lead skill because the year-block intervals were wide and the same 56 overlapping positive windows underlay each horizon (Figure 8C). Temporal calibration reduced Brier score for logistic S1 from 0.0804 to 0.0276 and for XGBoost S2 from 0.0514 to 0.0310; calibrated slopes were 0.835 and 1.034, respectively (Figure 9A–B).

On the onset-risk subset, 1,215 issue dates with \(T_{\max}(t)<36\,{}^\circ\mathrm{C}\) contained only 15 positive spell starts. Logistic S1 and XGBoost S2 each detected 11 of 15 (recall 0.733), but they generated 117 and 182 false alerts, respectively, yielding precision 0.086 and 0.057 (Figure 9C). These results do not support operational onset alerts. Extending logistic S1 history from seven to 14 days reduced AP from 0.522 to 0.494. Under the training-cutoff-specific 90th-percentile sensitivity target, logistic S1 AP was 0.431 with 94 positives, indicating that predictive conclusions remained prevalence- and definition-dependent.

XGBoost S2 was examined to understand why extra meteorology helped that family, not because it was the overall winner. Across held-out predictions, the leading mean absolute SHAP features were issue-day \(T_{\max}\), seven-day mean relative humidity, seven-day mean \(T_{\max}\), long-term time, and latest soil moisture (Figure 10). Their ranks varied across held-out years and leads, and correlated temperature and moisture summaries could share attribution. The explanations therefore identify predictive contributions within fitted XGBoost models, not physical causes.

**Table 9.** Strictly held-out future persistent-hot-window performance and within-family S2-minus-S1 contrasts. Source: `results/tables/main/main_table09_short_lead_prediction.csv`.

### 3.10 Secondary monthly forecast validation

For March–June monthly mean \(T_{\max}\), climatology plus linear trend had the lowest mean rolling-origin RMSE (1.114 °C), closely followed by exponential smoothing (1.120 °C); monthly climatology, SARIMAX, and seasonal naive had RMSEs of 1.175, 1.607, and 1.661 °C. The comparison concerns a continuous monthly temperature target and cannot be translated into exact persistent-window or annual heatwave-event predictions.

## 4 Discussion

### 4.1 Principal findings

Dhaka's annual and hot-season temperatures warmed over the observation period, but the primary persistent heatwave-day count did not exhibit a detectable monotonic trend. That contrast is not paradoxical. A shift in the center of the temperature distribution can coexist with large interannual variability, zero-heavy counts, threshold sensitivity, and limited power for rare persistent events. The NB2 CI is compatible with both decreases and increases of practical interest, so the correct interpretation is uncertainty—not proof of no change and not evidence of a clear increase.

The adjusted antecedent analysis contributed separate evidence. Lower humidity and precipitation and higher wind speed preceded persistent heatwave days after seasonal and long-term adjustment, and these directions remained in the event-onset analysis and after influential-season exclusions. The future-window experiment then addressed a genuinely prospective target: conditions on three not-yet-observed dates. Recent temperature history supplied the strongest overall short-lead signal. Additional meteorology improved two tree ensembles within family, but did not improve logistic regression and did not displace the simpler temperature-history logistic model. These results support a qualified claim about retrospective short-lead prediction, not causal explanation or operational forecast skill.

### 4.2 Why heatwave definition matters

The compound day/night definition was the only primary-reference definition with a nominal positive count trend, but it did not survive adjustment across the six definition tests. Absolute one-day and persistence-based definitions and relative \(T_{\max}\)-only definitions were uncertain. This pattern shows why “heatwaves increased” is incomplete without stating the threshold, duration, reference period, season, day/night variables, and multiplicity context. It also connects with prior Bangladesh work emphasizing compound high minimum and maximum temperatures (Nissan et al., 2017) and percentile heatwaves (Molla et al., 2025; Tabassum et al., 2024). The present contribution is not to choose one universal definition, but to show which conclusions are stable and which are conditional.

### 4.3 Antecedent meteorological conditions

Dryer antecedent conditions are physically plausible during suppressed pre-monsoon rainfall and enhanced surface heating. Nissan et al. (2017) likewise described below-normal precipitation and soil moisture before Bangladesh heatwaves, and Mallik et al. (2024) discussed pressure and wind patterns associated with threshold days. In the current analysis, however, humidity and precipitation were correlated and the data came from one series. Odds ratios represent conditional associations per sample standard deviation, not intervention effects. Wind's positive association may reflect regional advection rather than locally generated wind effects. Pressure's sign reversal between persistent-day and onset models demonstrates the danger of elevating an unstable coefficient into a mechanistic claim.

### 4.4 Methodological implications

Raw heatmaps were useful for identifying covariance and possible collinearity but could not estimate independent relationships. De-seasonalization showed that several strong correlations were not explained solely by the March–June cycle. Direct count diagnostics were equally important: Poisson dispersion above 11 and a more than 400-point AIC difference made an equidispersed count model untenable. GEE accounted for repeated daily observations within years, while blocked validation showed whether association estimates translated into future-season discrimination. These safeguards change the role of apparently strong associations from explanatory claims to validated but observational signals.

### 4.5 What the predictive comparison establishes

The primary predictive result was not that the most complex model won. Logistic S1 produced the highest overall one-day AP and lowest Brier score, showing that recent observed temperature carried most of the recoverable short-lead information. The positive S2-minus-S1 contrasts for XGBoost and weighted random forest nevertheless show that humidity, soil moisture, radiation, and related summaries can help particular nonlinear learners organize the remaining signal. The negative logistic contrast prevents a broader claim that more predictors are inherently better. This distinction is more informative than a model leaderboard because it separates an information-set question from an algorithm-selection question.

The contrast with prior work also requires precise language. Kan et al. (2025) studied seasonal monthly classification over Swedish spatial pixels, whereas Choudary V et al. (2025) studied next-day classification at a Chennai station. Our one-, three-, and seven-day direct targets are neither seasonal forecasts nor copies of a next-day label. The sharp decline after one day suggests that observed-history-only models have limited reach without numerical weather prediction or other genuinely future-available inputs. The relatively high one-day ROC-AUCs coexist with modest precision because positive windows were rare, illustrating why prevalence-sensitive AP, probability calibration, and false-alert counts were central (Saito & Rehmsmeier, 2015; Van Calster et al., 2019).

The onset analysis is the clearest operational caution. Detecting 11 of 15 spell starts may appear encouraging, but 117–182 false alerts for those 11 detections would create a poor warning burden. Moreover, the data source is an unidentified Meteoblue-formatted export, and actual observation or product-release latency is unknown. Independent sites, verified real-time availability, event-level external validation, and a threshold tied to an explicit decision and loss function would be required before calling the models an early-warning system. This follows Bangladesh climate-services work emphasizing evaluation against the statistics and decisions a forecast is meant to support (Nissan et al., 2020). The monthly temperature exercise likewise remains a target-mismatch sensitivity; its results cannot be converted into exact future event counts. Recent hybrid machine-learning work demonstrates active methodological interest but does not by itself establish transportable skill (Qureshi et al., 2025).

### 4.6 Practical relevance

Primary persistent events were concentrated in April and May, and recent temperature history improved prediction beyond seasonal timing and long-term trend. These findings may inform research monitoring priorities, but the low onset precision means that the fitted alert thresholds should not guide public warnings. Preparedness decisions should integrate numerical forecasts, verified observation latency, health, exposure, vulnerability, and operational cost information not present here.

The distinction matters because separate Dhaka health research has associated temperature variability with emergency visits and heatwave definitions with diarrhoeal hospitalization (Haque et al., 2024; Rahman et al., 2022), whereas the present study contains no patient-level or population-health outcome. Heat-action-plan evidence and multi-country analyses also show that preparedness and vulnerability depend on social and urban context (Hess et al., 2018; Sera et al., 2019). The meteorological evidence can help define exposure windows for future interdisciplinary work, but it cannot estimate attributable morbidity or intervention effectiveness.

### 4.7 Strengths and limitations

Strengths include a long continuous daily record, explicit partial-year handling, reproducible event construction, six definitions, two climatological reference periods, direct count regression, model-consistent diagnostics, formal slope comparison, leakage-safe lag construction, repeated-measures inference, and event-onset sensitivity. The prediction benchmark used explicit issue and target times, direct horizons, nested chronological tuning, training-only transformations and calibration, simple baselines, information-set ablations, event-free seasons, paired-year uncertainty, sensitivity targets, and out-of-sample explanations. All principal tables and figures are generated from code and linked to claims.

Several limitations constrain interpretation. The study represents one location and lacks a rural comparator, so it cannot quantify urban amplification. Station coordinates, relocations, instrumentation changes, observation practices, release latency, and homogenization metadata were unavailable in the repository; an unmodeled discontinuity could affect trends, and unavailable real-time fields could invalidate retrospective issue times. The dataset has no health, mortality, morbidity, exposure, or vulnerability outcome. The primary definition yielded only 49 historical events and the predictive onset subset only 15 held-out spells, limiting model complexity and uncertainty resolution. Overlapping positive windows from the same spell were not independent; year-block resampling preserved seasons but eleven held-out years still provide coarse intervals. Meteorological variables may share upstream algorithms or measurement errors. GEE associations are exploratory and observational, while SHAP and permutation results are predictive diagnostics rather than causal effects. The 2024 calendar year was partial, although its hot season was complete. No external location or operational feed was available for transportability testing.

## 5 Conclusions

Dhaka experienced statistically detectable annual and March–June warming from 1972–2024, but the selected NB2 model did not show a clear monotonic change in the prespecified persistent heatwave-day rate and definition-specific conclusions varied. Lower antecedent humidity and precipitation and higher antecedent wind speed were adjusted associations, not causal effects. For future three-day persistent-hot windows, recent temperature history supported useful retrospective discrimination at one-day lead; added meteorology improved selected tree families but not the strongest overall model, and skill weakened at longer leads. Low onset precision, few distinct held-out spells, unresolved source latency, and the absence of external validation preclude operational-warning claims. The defensible contribution is therefore an integrated account of warming, definition sensitivity, antecedent association, and strictly chronological short-lead prediction—with each conclusion tied to its specific outcome and evidence.

## Declarations

**Author contributions.** Awnon Bhowmik: methodology, software, data curation, investigation, visualization, and original draft preparation. Goutam Saha: conceptualization, supervision, data review, and manuscript review and editing.

**Funding.** The authors received no specific funding for this work.

**Competing interests.** The authors declare no competing financial or non-financial interests.

**Data and code availability.** The daily climate file, analysis code, reproducibility tests, and figure-generation materials are available in the public repository at https://github.com/awnonbhowmik/Heatwave-Dhaka_BD. The complete analysis can be regenerated with `make article`.

**Ethics approval.** Not applicable. This meteorological analysis used no human participants, personal data, or individual-level health records.

**Use of AI-assisted tools.** AI-assisted tools were used for language editing and organizational support. The authors reviewed and verified the scientific content, numerical results, interpretations, and references and remain responsible for the final manuscript.

## Figure captions

**Figure 1. Study area and data coverage.** Bangladesh and Dhaka District administrative boundaries and annual daily-record completeness. No station point is shown because exact coordinate provenance was unavailable. Calendar year 2024 is partial; March–June 2024 is complete. Administrative boundaries do not imply official endorsement.

**Figure 2. Descriptive climatology and primary count distribution.** (A) Daily maximum-temperature distribution. (B) Monthly mean maximum and minimum temperatures. (C) Annual March–June count distribution for days in events with \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive days, with overdispersion statistics. (D) March–June maximum-temperature distributions and the 36 °C threshold.

**Figure 3. March–June correlation analysis.** (A) Raw Spearman correlations. (B) Spearman correlations after subtracting smoothed 1981–2010 calendar-day means. Correlations describe pairwise covariance and possible collinearity. They do not estimate independent effects or establish causation.

**Figure 4. Long-term temperature trends.** (A) Complete-year annual mean \(T_{\max}\) through 2023. (B) Complete-year annual mean \(T_{\min}\). (C) Complete March–June mean \(T_{\max}\) through 2024. Lines are OLS fits and bands are 95% CIs based on HAC covariance. (D) Per-decade slope estimates and 95% CIs, including the formal annual \(T_{\min}-T_{\max}\) contrast.

**Figure 5. Heatwave definition sensitivity.** (A) March–June annual counts under selected definitions. (B) Event-duration distribution for the primary three-day absolute definition only. (C) Monthly distribution of primary persistent heatwave days. (D) Count-model IRRs per decade and 95% CIs for six definitions. The log-scale reference line is IRR 1.

**Figure 6. Primary NB2 count model and diagnostics.** (A) Observed March–June persistent heatwave-day counts, fitted mean, and parameter-based 95% mean CI. (B) Selected-distribution randomized quantile residuals versus fitted means. (C) randomized-residual Q–Q plot. (D) case-deletion parameter distance by year; labels identify years exceeding the \(4/n\) screening threshold. Influential years remain in the primary model.

**Figure 7. Adjusted antecedent associations and validation.** (A) Logistic GEE adjusted ORs and 95% CIs per one-standard-deviation antecedent predictor. (B) model-implied probabilities across humidity and precipitation values at a representative pre-monsoon calendar day, holding other standardized variables at their means. (C) Brier score by held-out hot season. (D) pooled strictly out-of-sample discrimination and probability error for the seasonal-trend base and full antecedent models. Brier score is lower-is-better, unlike AUC metrics.

**Figure 8. Primary future-window comparisons.** (A) One-day average precision for the prespecified baselines and focal S1/S2 models. (B) Within-family S2-minus-S1 average-precision differences and paired held-out-year 95% bootstrap intervals at one-day lead. (C) Average precision by lead for the strongest overall temperature-history model, the explained full-meteorology tree, and simple baselines. All values are strictly out of sample.

**Figure 9. Discrimination, calibration, and onset alert burden.** (A) Pooled held-out precision–recall curves at one-day lead. (B) temporal-calibration reliability for logistic S1 and XGBoost S2. (C) detected and missed onset spells with false alerts at training-selected thresholds. The onset panel summarizes only 15 positive spells and is not an operational evaluation.

**Figure 10. Out-of-sample XGBoost explanation.** (A) Global mean absolute SHAP values. (B) feature-rank stability across leads and held-out years. (C) held-out SHAP distribution for leading predictors. (D) dependence diagnostic for the prespecified leading feature. SHAP values explain fitted uncalibrated model outputs and are not causal meteorological effects.

## References

Adnan, M. S. G., Kabir, I., Hossain, M. A., Chakma, S., Tasneem, S. N., Saha, C. R., Hassan, Q. K., & Dewan, A. (2024). Heatwave vulnerability of large metropolitans in Bangladesh: An evaluation. *Geomatica, 76*(2), 100020. https://doi.org/10.1016/j.geomat.2024.100020

Anderson, G. B., Bell, M. L., & Peng, R. D. (2013). Methods to calculate the heat index as an exposure metric in environmental health research. *Environmental Health Perspectives, 121*(10), 1111–1119. https://doi.org/10.1289/ehp.1206273

Ballester, J., Quijal-Zamorano, M., Méndez Turrubiates, R. F., Pegenaute, F., Herrmann, F. R., Robine, J. M., Basagaña, X., Tonne, C., Antó, J. M., & Achebak, H. (2023). Heat-related mortality in Europe during the summer of 2022. *Nature Medicine, 29*, 1857–1866. https://doi.org/10.1038/s41591-023-02419-z

Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis, 120*, 70–83. https://doi.org/10.1016/j.csda.2017.11.003

Boni, Z., Bieńkowska, Z., Chwałczyk, F., Jancewicz, B., Marginean, I., & Yáñez Serrano, P. (2023). What is a heat(wave)? An interdisciplinary perspective. *Climatic Change, 176*, 129. https://doi.org/10.1007/s10584-023-03592-3

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: Forecasting and control* (5th ed.). Wiley.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chaki, S., Samad, M. A., Mallik, M. A. K., & Hassan, S. M. Q. (2025). Forecasting human heat stress: Insights from observations and WRF simulations during Bangladesh heatwaves. *PLOS Climate, 4*(8), e0000690. https://doi.org/10.1371/journal.pclm.0000690

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Choudary V, R., Johnvictor, A. C., & Sankar N, P. (2025). Comparative analysis of machine learning approaches for heatwave event prediction in India. *Scientific Reports, 15*, 22431. https://doi.org/10.1038/s41598-025-04634-9

Dewan, A., Kiselev, G., Botje, D., Mahmud, G. I., Bhuian, M. H., & Hassan, Q. K. (2021). Surface urban heat island intensity in five major cities of Bangladesh: Patterns, drivers and trends. *Sustainable Cities and Society, 71*, 102926. https://doi.org/10.1016/j.scs.2021.102926

Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association, 74*(366), 427–431. https://doi.org/10.1080/01621459.1979.10482531

Diffenbaugh, N. S., Singh, D., Mankin, J. S., Horton, D. E., Swain, D. L., Touma, D., Charland, A., Liu, Y., Haugen, M., Tsiang, M., & Rajaratnam, B. (2017). Quantifying the influence of global warming on unprecedented extreme climate events. *Proceedings of the National Academy of Sciences of the United States of America, 114*(19), 4881–4886. https://doi.org/10.1073/pnas.1618082114

Farukh, M. A., Brahma, P. P., Hossain, M. S., Hoque, M. J., Sejuti, S. I., Shammy, U. S., & Arefin, K. S. (2026). Climatological assessment of pre-monsoon heatwave days in Bangladesh and their relationship to Indo-Pacific circulation anomalies. *Natural Hazards, 122*(1), 1–32. https://doi.org/10.1007/s11069-025-07746-7

Gallo, E., Quijal-Zamorano, M., Méndez Turrubiates, R. F., Tonne, C., Basagaña, X., Achebak, H., & Ballester, J. (2024). Heat-related mortality in Europe during 2023 and the role of adaptation in protecting health. *Nature Medicine, 30*, 3101–3105. https://doi.org/10.1038/s41591-024-03186-1

Ghumman, U., & Horney, J. (2016). Characterizing the impact of extreme heat on mortality, Karachi, Pakistan, June 2015. *Prehospital and Disaster Medicine, 31*(3), 263–266. https://doi.org/10.1017/S1049023X16000273

Global Forest Watch. (2024). *Tree cover loss data for Dhaka Division, Bangladesh, 2001–2023*. World Resources Institute. https://www.globalforestwatch.org/

Haque, F., Lampe, F. C., Hajat, S., Stavrianaki, K., Hasan, S. M. T., Faruque, A. S. G., Ahmed, T., Jubayer, S., & Kelman, I. (2024). Is heat wave a predictor of diarrhoea in Dhaka, Bangladesh? A time-series analysis in a South Asian tropical monsoon climate. *PLOS Global Public Health, 4*(9), e0003629. https://doi.org/10.1371/journal.pgph.0003629

Hess, J. J., Sathish, L. M., Knowlton, K., Saha, S., Dutta, P., Ganguly, P., Tiwari, A., Jaiswal, A., Sheffield, P., Sarkar, J., Bhan, S. C., Begda, A., Shah, T., Solanki, B., & Mavalankar, D. (2018). Building resilience to climate change: Pilot evaluation of the impact of India's first heat action plan on all-cause mortality. *Journal of Environmental and Public Health, 2018*, 7973519. https://doi.org/10.1155/2018/7973519

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Huang, W., Kan, H., & Kovats, S. (2010). The impact of the 2003 heat wave on mortality in Shanghai, China. *Science of the Total Environment, 408*(11), 2418–2420. https://doi.org/10.1016/j.scitotenv.2010.02.009

Im, E.-S., Pal, J. S., & Eltahir, E. A. B. (2017). Deadly heat waves projected in the densely populated agricultural regions of South Asia. *Science Advances, 3*(8), e1603322. https://doi.org/10.1126/sciadv.1603322

Intergovernmental Panel on Climate Change. (2021). *Climate change 2021: The physical science basis: Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change*. Cambridge University Press. https://doi.org/10.1017/9781009157896

Islam, M. Y., Mohiuddin, M., Tanvir Hossain, K., Salauddin, M., & Farin, S. (2024). Trend of heat waves in Dhaka Metropolitan City and its impact on livelihood and health of exposed people. *Arabian Journal of Geosciences, 17*, 232. https://doi.org/10.1007/s12517-024-12027-x

Kan, J.-C., Vieira Passos, M., Destouni, G., Barquet, K., Ferreira, C. S. S., & Kalantari, Z. (2025). Seasonal heatwave forecasting with explainable machine learning and remote sensing data. *Stochastic Environmental Research and Risk Assessment, 39*, 3333–3352. https://doi.org/10.1007/s00477-025-03020-1

Karmakar, S., & Das, M. K. (2020). On the heat waves in Bangladesh, their trends and associated large-scale tropospheric conditions. *Journal of Engineering Science, 11*(1), 19–36. https://doi.org/10.3329/jes.v11i1.49544

Khatun, K., Samad, M. A., & Rashid, M. B. (2017). Time series analysis of temperature and rainfall data of Dhaka Division. *Dhaka University Journal of Science, 65*(2), 119–123. https://doi.org/10.3329/dujs.v65i2.54519

Liu, Z., Zhan, W., Bechtel, B., Voogt, J., Lai, J., Chakraborty, T., Wang, Z.-H., Li, M., Huang, F., & Lee, X. (2022). Surface warming in global cities is substantially more rapid than in rural background areas. *Communications Earth & Environment, 3*, 219. https://doi.org/10.1038/s43247-022-00539-x

Lu, Y.-C., & Romps, D. M. (2022). Extending the heat index. *Journal of Applied Meteorology and Climatology, 61*(10), 1367–1383. https://doi.org/10.1175/JAMC-D-22-0021.1

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774).

Ma, F., & Yuan, X. (2021). More persistent summer compound hot extremes caused by global urbanization. *Geophysical Research Letters, 48*(15), e2021GL093721. https://doi.org/10.1029/2021GL093721

Mallik, M. A. K., Sultana, A., Islam, M. K., Akter, M. Y., Alam, E., & Islam, A. R. M. T. (2024). Are hotspots and frequencies of heat waves changing over time? Exploring causes of heat waves in a tropical country. *PLOS ONE, 19*, e0300070. https://doi.org/10.1371/journal.pone.0300070

Manoli, G., Fatichi, S., Schläpfer, M., Yu, K., Crowther, T. W., Meili, N., Burlando, P., Katul, G. G., & Bou-Zeid, E. (2019). Magnitude of urban heat islands largely explained by climate and population. *Nature, 573*, 55–60. https://doi.org/10.1038/s41586-019-1512-9

Meteoblue. (2024). *Historical weather data for Dhaka, Bangladesh*. https://www.meteoblue.com/

Molla, M. A. M., Hassan, Q. K., & Dewan, A. (2025). Unveiling heatwave events in Bangladesh: Insights from observational records and ERA5 reanalysis data. *Climate Services, 40*, 100609. https://doi.org/10.1016/j.cliser.2025.100609

Mora, C., Dousset, B., Caldwell, I. R., Powell, F. E., Geronimo, R. C., Bielecki, C. R., Counsell, C. W. W., Dietrich, B. S., Johnston, E. T., Louis, L. V., Lucas, M. P., McKenzie, M. M., Shea, A. G., Tseng, H., Giambelluca, T. W., Leon, L. R., Hawkins, E., & Trauernicht, C. (2017). Global risk of deadly heat. *Nature Climate Change, 7*(7), 501–506. https://doi.org/10.1038/nclimate3322

Nawaro, J., Gianquintieri, L., Pagliosa, A., Sechi, G. M., & Caiani, E. G. (2023). Heatwave definition and impact on cardiovascular health: A systematic review. *Public Health Reviews, 44*, 1606266. https://doi.org/10.3389/phrs.2023.1606266

Nissan, H., Burkart, K., Coughlan de Perez, E., van Aalst, M., & Mason, S. (2017). Defining and predicting heat waves in Bangladesh. *Journal of Applied Meteorology and Climatology, 56*(10), 2653–2670. https://doi.org/10.1175/JAMC-D-17-0035.1

Nissan, H., Muñoz, Á. G., & Mason, S. J. (2020). Targeted model evaluations for climate services: A case study on heat waves in Bangladesh. *Climate Risk Management, 28*, 100213. https://doi.org/10.1016/j.crm.2020.100213

Perkins, S. E., & Alexander, L. V. (2013). On the measurement of heat waves. *Journal of Climate, 26*(13), 4500–4517. https://doi.org/10.1175/JCLI-D-12-00383.1

Perkins-Kirkpatrick, S. E., & Lewis, S. C. (2020). Increasing trends in regional heatwaves. *Nature Communications, 11*, 3357. https://doi.org/10.1038/s41467-020-16970-7

Qureshi, M. M. U., Ahmed, A. B., Dulmini, A., Khan, M. M. H., & Rois, R. (2025). Developing a seasonal-adjusted machine-learning-based hybrid time-series model to forecast heatwave warning. *Scientific Reports, 15*, 8699. https://doi.org/10.1038/s41598-025-93227-7

Rahman, M. M., Garcia, E., Lim, C. C., Ghazipura, M., Alam, N., Palinkas, L. A., McConnell, R., & Thurston, G. (2022). Temperature variability associations with cardiovascular and respiratory emergency department visits in Dhaka, Bangladesh. *Environment International, 164*, 107267. https://doi.org/10.1016/j.envint.2022.107267

Ravindra, K., Bhardwaj, S., Ram, C., Goyal, A., Singh, V., Venkataraman, C., Bhan, S., Sokhi, R., & Mor, S. (2024). Temperature projections and heatwave attribution scenarios over India: A systematic review. *Heliyon, 10*(4), e26431. https://doi.org/10.1016/j.heliyon.2024.e26431

Raymond, C., Matthews, T., & Horton, R. M. (2020). The emergence of heat and humidity too severe for human tolerance. *Science Advances, 6*(19), eaaw1838. https://doi.org/10.1126/sciadv.aaw1838

Rashid, G. M., Hossain, M. M. T., Akhter, M. A. E., & Mallik, M. A. K. (2024). A study on the heat wave conditions over Bangladesh during 1990–2019. *Journal of Engineering Science, 14*(2), 59–67. https://doi.org/10.3329/jes.v14i2.71227

Robinson, P. J. (2001). On the definition of a heat wave. *Journal of Applied Meteorology, 40*(4), 762–775. https://doi.org/10.1175/1520-0450(2001)040%3C0762:OTDOAH%3E2.0.CO;2

Rothfusz, L. P. (1990). *The heat index “equation” (or, more than you ever wanted to know about heat index)*. National Weather Service, Southern Region Headquarters.

Russo, S., Dosio, A., Graversen, R. G., Sillmann, J., Carrao, H., Dunbar, M. B., Singleton, A., Montagna, P., Barbola, P., & Vogt, J. V. (2014). Magnitude of extreme heat waves in present climate and their projection in a warming world. *Journal of Geophysical Research: Atmospheres, 119*(22), 12,500–12,512. https://doi.org/10.1002/2014JD022098

Saito, T., & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Saeed, F., Schleussner, C.-F., & Ashfaq, M. (2021). Deadly heat stress to become commonplace across South Asia already at 1.5 °C of global warming. *Geophysical Research Letters, 48*(7), e2020GL091191. https://doi.org/10.1029/2020GL091191

Sarangi, C., Qian, Y., Li, J., Leung, L. R., Chakraborty, T., & Liu, Y. (2021). Urbanization amplifies nighttime heat stress on warmer days over the United States. *Geophysical Research Letters, 48*, e2021GL095678. https://doi.org/10.1029/2021GL095678

Sera, F., Armstrong, B., Tobias, A., Vicedo-Cabrera, A. M., Åström, C., Bell, M. L., Chen, B.-Y., de Sousa Zanotti Stagliorio Coelho, M., Matus Correa, P., Cruz, J. C., Dang, T. N., Hurtado-Díaz, M., Do Van, D., Forsberg, B., Guo, Y.-L. L., Guo, Y., Hashizume, M., Honda, Y., Iñiguez, C., … Gasparrini, A. (2019). How urban characteristics affect vulnerability to heat and cold: A multi-country analysis. *International Journal of Epidemiology, 48*(4), 1101–1112. https://doi.org/10.1093/ije/dyz008

Shahrujjaman, S. M., Sikder, B. B., Zahid, D., & Pal, B. (2025). Heat wave adaptation strategies among informal workers in an urban setting: A study in Dhaka city, Bangladesh. *Natural Hazards Research, 5*(3), 509–522. https://doi.org/10.1016/j.nhres.2025.01.006

Sharma, A., Andhikaputra, G., & Wang, Y.-C. (2022). Heatwaves in South Asia: Characterization, consequences on human health, and adaptation strategies. *Atmosphere, 13*(5), 734. https://doi.org/10.3390/atmos13050734

Shi, Z., Xu, X., & Jia, G. (2021). Urbanization magnified nighttime heat waves in China. *Geophysical Research Letters, 48*(15), e2021GL093603. https://doi.org/10.1029/2021GL093603

Singh, S., Yadav, A., & Goyal, M. K. (2024). Univariate and bivariate spatiotemporal characteristics of heat waves and relative influence of large-scale climate oscillations over India. *Journal of Hydrology, 628*, 130596. https://doi.org/10.1016/j.jhydrol.2023.130596

Tabassum, A., Park, K., Seo, J. M., Han, J.-Y., Hong, S. H., & Baik, J.-J. (2024). Characteristics of the urban heat island in Dhaka, Bangladesh, and its interaction with heat waves. *Asia-Pacific Journal of Atmospheric Sciences, 60*, 479–493. https://doi.org/10.1007/s13143-024-00362-8

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting, 16*(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

Tuholske, C., Caylor, K., Funk, C., Verdin, A., Sweeney, S., Grace, K., Peterson, P., & Evans, T. (2021). Global urban population exposure to extreme heat. *Proceedings of the National Academy of Sciences of the United States of America, 118*(41), e2024792118. https://doi.org/10.1073/pnas.2024792118

Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. *BMC Medicine, 17*, 230. https://doi.org/10.1186/s12916-019-1466-7

Vecellio, D. J., Wolf, S. T., Cottle, R. M., & Kenney, W. L. (2022). Evaluating the 35 °C wet-bulb temperature adaptability threshold for young, healthy subjects (PSU HEAT Project). *Journal of Applied Physiology, 132*(2), 340–345. https://doi.org/10.1152/japplphysiol.00738.2021

Vicedo-Cabrera, A. M., Scovronick, N., Sera, F., Royé, D., Schneider, R., Tobias, A., Åström, C., Guo, Y., Honda, Y., Hondula, D. M., Abrutzky, R., Tong, S., de Sousa Zanotti Stagliorio Coelho, M., Nascimento Saldiva, P. H., Lavigne, E., Matus Correa, P., Valdés Ortega, N., Kan, H., Osorio, S., … Gasparrini, A. (2021). The burden of heat-related mortality attributable to recent human-induced climate change. *Nature Climate Change, 11*(6), 492–500. https://doi.org/10.1038/s41558-021-01058-x

Zhang, Y., Mao, G., Chen, C., Lu, Z., Luo, Z., & Zhou, W. (2020). Population exposure to concurrent daytime and nighttime heatwaves in Huai River Basin, China. *Sustainable Cities and Society, 61*, 102309. https://doi.org/10.1016/j.scs.2020.102309

Zhao, L., Lee, X., Smith, R. B., & Oleson, K. (2014). Strong contributions of local background climate to urban heat islands. *Nature, 511*, 216–219. https://doi.org/10.1038/nature13462

Zscheischler, J., Westra, S., van den Hurk, B. J. J. M., Seneviratne, S. I., Ward, P. J., Pitman, A., AghaKouchak, A., Bresch, D. N., Leonard, M., Wahl, T., & Zhang, X. (2018). Future climate risk from compound events. *Nature Climate Change, 8*, 469–477. https://doi.org/10.1038/s41558-018-0156-3
