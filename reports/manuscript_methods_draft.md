# Manuscript Methods draft

## Design, data, and reproducibility

We conducted a retrospective single-location time-series analysis of daily meteorological data for Dhaka from 1 January 1972 to 18 November 2024. The source CSV was read without editing and its SHA-256 digest recorded. Dates were parsed under ISO-8601, sorted, compared with a complete daily index, and checked for duplication, missingness, physical ranges, discontinuities, and partial years. Complete-calendar-year analyses ended in 2023; March–June 2024 was retained because all 122 dates were present. The prespecified seed was 20260901.

## Outcomes and event construction

The primary heatwave was daily Tmax >= 36 °C for at least three consecutive calendar days. Sensitivity definitions used one and two days, calendar-day 90th and 95th percentiles for three days, and compound 90th-percentile Tmax/Tmin for two days. Calendar-day quantiles used NumPy's linear algorithm in a circular +/-7-day window over 1981–2010; 1991–2020 was a sensitivity reference. Leap-year dates after February were mapped to a 365-day climatological calendar; 29 February used the mean of 28 February and 1 March thresholds. A missing date breaks an event. Event intensity was the sum of daily Tmax excess above its definition-specific threshold.

## Descriptions, correlations, and trends

We summarized n, missing n, mean, SD, median, quartiles, IQR, range, skewness, and excess kurtosis before modeling. March–June Spearman matrices were calculated on raw variables and on anomalies after subtracting smoothed 1981–2010 calendar-day climatologies. VIFs, absolute-correlation clustering, condition indices, and domain knowledge determined the adjusted predictor set.

For continuous annual and March–June temperatures, $T_y=\alpha+\beta((y-\bar y)/10)+\epsilon_y$ was fitted by OLS with three-lag HAC standard errors. We report per-decade slopes and 95% CIs, Theil–Sen slopes, Kendall trend tests, and a prewhitened sensitivity when lag-1 residual correlation exceeded $1.96/\sqrt n$. A stacked interaction model formally tested the Tmin–Tmax slope difference.

## Count and association models

For primary March–June counts, $Y_y\sim Poisson(\mu_y)$ and NB2 alternatives with $Var(Y)=\mu+\alpha\mu^2$ were compared. Model selection used empirical dispersion, AIC, convergence, and residual diagnostics. We report $e^{\beta_1}$ as the incidence-rate ratio per decade. Cook's distance from the Poisson influence representation flagged years for leave-one-year-out sensitivity only.

Antecedent variables were created with `shift(1)` before rolling aggregation, ensuring that outcome-day and future information were unavailable. The full logistic GEE added four prespecified, standardized antecedent variables to three Fourier harmonics and long-term trend. Years defined clusters, an AR(1) working correlation represented serial dependence, and robust sandwich covariance was used. No current-day temperature, VPD, heat index, apparent temperature, or event-derived variable was used. Chronological validation trained only on years preceding each held-out complete hot season; standardization and the Youden operating threshold were fitted on training data only.

## Forecasting and uncertainty

Monthly mean Tmax forecasts used repeated origins 2014–2023. Seasonal naive, training-only monthly climatology, climatology with linear trend, SARIMA(1,1,1)x(1,0,0)[12], and additive ETS were scored on identical March–June targets using MAE, RMSE, bias, MASE, and 80%/95% coverage and width. Model AIC was not compared with out-of-sample scores. Because monthly forecasts cannot reconstruct daily threshold runs, 2025–2029 heatwave counts were generated only as exploratory direct-count trend scenarios. We drew 5,000 coefficient vectors from the fitted covariance and then sampled the selected count distribution, reporting medians and 80%/95% quantiles.

## Software

The command `python scripts/run_all.py --config config/analysis.yml` generated all results. Pinned package versions, platform, configuration, hashes, starting commit, timestamp, and runtime are stored in `results/metadata/run_metadata.json`.
