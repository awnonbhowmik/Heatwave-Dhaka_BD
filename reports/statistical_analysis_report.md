# Statistical analysis report

## Research questions and prespecification

RQ1 concerns trends in heatwave frequency, persistence, duration, and intensity; RQ2 concerns adjusted associations between antecedent weather and persistent heatwave status; RQ3 concerns prediction on unseen hot seasons. The primary definition is Tmax >= 36 °C for at least three consecutive days; the primary count outcome is March–June persistent heatwave days.

## Descriptive and exploratory results

The full-record mean Tmax was **30.14 °C**, and the maximum was **40.2 °C** on 2023-05-09. The one-day operational definition identified **377 days in 136 runs**; the primary three-day definition identified **270 days in 49 events**, with a longest event of **15 days**. Descriptive distributions precede inference in Tables 1–8. Raw and calendar-day-anomaly Spearman matrices are separate. Correlation is interpreted as exploratory association, never causation. Temperature-derived VPD and all same-day temperatures were excluded from the adjusted binary model.

## Trend and count inference

March–June mean Tmax changed by **0.165 °C/decade** (95% CI 0.044, 0.287) using OLS with HAC standard errors; Sen and Mann–Kendall sensitivity estimates appear in Table 9.

Counts were modeled as $\log E(Y_y)=\beta_0+\beta_1((y-\bar y)/10)$. Poisson and NB2 were compared using dispersion, likelihood fit, AIC, and diagnostics. **negative_binomial** was selected. Its IRR was **1.029 per decade** (95% CI 0.727, 1.455; p=0.8725). Influential years remain in the primary analysis and are removed only one at a time in sensitivity checks.

## Adjusted association model

The model is $\operatorname{logit}(p_t)=\alpha+f_{Fourier}(DOY_t)+\gamma year_t+\sum_j\theta_j X_{j,t-}$. Predictors are prior-three-day mean humidity, prior-seven-day precipitation, prior-three-day wind speed, and prior-three-day pressure. A logistic GEE grouped by year used an AR(1) working correlation and robust sandwich covariance. Per-SD estimates are in Table 15. Blocked validation results, emphasizing Brier and precision–recall AUC, are in Table 16.

## Forecast validation and uncertainty

Ten rolling origins compared seasonal naive, monthly climatology, climatology plus linear trend, SARIMAX, and ETS on identical March–June monthly means. The lowest pooled RMSE was **1.114 °C (climatology_linear_trend)** versus **1.661 °C** for seasonal naive. Interval calibration is reported rather than inferred from in-sample AIC.

Future count output is a direct negative_binomial trend scenario with parameter and count uncertainty from 5,000 simulations. It is not a validated deterministic five-year forecast. Long-horizon precision is unsupported by a single-site historical series and rolling validation.

## Conclusions

Conclusions are separated into descriptive, inferential, and predictive claims. No urban–rural contrast, health outcome, or causal intervention is observed. Tree-cover and heat-index claims remain supplementary and exploratory.
