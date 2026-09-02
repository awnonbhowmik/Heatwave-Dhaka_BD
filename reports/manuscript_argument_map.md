# Manuscript argument map

## 1. Scientific problem

Long-term warming does not imply that every operational or persistence-based heatwave metric changes at the same rate. Heatwave conclusions can depend strongly on the absolute or percentile threshold, persistence requirement, day/night criterion, reference climatology, and statistical model.

## 2. Specific knowledge gap

Dhaka-focused evidence has commonly emphasized descriptive temperature trends, threshold counts, correlations, or projections. The present study tests whether a long daily record supports a coherent conclusion when alternative definitions, formal temperature trends, overdispersed count regression, leakage-safe antecedent associations, diagnostics, and chronological validation are considered together.

## 3. Primary research question

How did March–June persistent heatwave-day incidence change in Dhaka from 1972 through 2024 when persistent heat is defined as daily maximum temperature at or above 36 °C for at least three consecutive calendar days?

The two secondary research questions are: (i) how did annual and March–June maximum and minimum temperatures change, and how sensitive are event statistics to alternative heatwave definitions; and (ii) which antecedent meteorological conditions are associated with persistent heatwave days after adjustment for seasonality, long-term change, and within-year dependence?

## 4. Primary outcome

The primary outcome is the annual March–June count of days belonging to an event with \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) for at least three consecutive calendar days.

## 5. Primary model

The primary model is NB2 count regression,

\[
\log\{E(Y_y)\}=\beta_0+\beta_1\left(\frac{y-\bar y}{10}\right),
\qquad
\operatorname{Var}(Y_y)=\mu_y+\alpha\mu_y^2,
\]

with \(\exp(\beta_1)\) interpreted as the incidence-rate ratio per decade. The two-sided null hypothesis is \(H_0:\exp(\beta_1)=1\); the alternative is \(H_1:\exp(\beta_1)\ne1\).

## 6. Secondary analyses

- OLS temperature trends with HAC standard errors, Theil–Sen slopes, Mann–Kendall tests, residual autocorrelation, and endpoint sensitivity.
- A formal annual \(T_{\min}\)-versus-\(T_{\max}\) slope contrast.
- Six primary heatwave definitions and 1991–2020 percentile-reference sensitivity.
- Raw and de-seasonalized March–June Spearman correlations for exploration and predictor screening.
- Logistic GEE for adjusted antecedent associations, clustered by year with AR(1) working dependence and robust sandwich covariance.
- Event-onset, alternate-lag, and influential-season sensitivity analyses.
- Chronological held-out-season validation and pooled strictly out-of-sample performance.
- Supplementary rolling-origin monthly-temperature forecasting against simple baselines.

## 7. Most robust finding

Dhaka warmed over the record. Annual mean \(T_{\max}\) increased by 0.192 °C per decade (95% CI 0.130–0.253), and March–June mean \(T_{\max}\) increased by 0.165 °C per decade (95% CI 0.044–0.287). Parametric and nonparametric analyses agreed in direction.

## 8. Most important null finding

The selected NB2 model did not provide evidence of a monotonic change in the primary persistent heatwave-day rate: IRR 1.029 per decade (95% CI 0.727–1.455; p = 0.872). This null finding is stable to case deletion and contrasts with clear background temperature warming.

## 9. Main contribution

The study shows empirically that long-term warming and heatwave-count trends are not interchangeable: conclusions for Dhaka depend on how heatwaves are defined. It couples that comparison with direct overdispersed count modeling and validated, leakage-safe antecedent associations rather than treating raw correlations as final evidence.

## 10. Main limitation

The analysis uses one meteorological series without a rural comparator, health outcomes, or complete station-homogeneity metadata. It contains only 49 primary persistent events, so antecedent associations remain observational and exploratory even when statistically adjusted and temporally validated.

## 11. One-sentence conclusion

Dhaka experienced statistically detectable warming from 1972–2024, but the long-term behavior of heatwave occurrence was definition-dependent, the primary persistent heatwave-day rate showed no clear monotonic change, and lower antecedent humidity and precipitation and higher antecedent wind speed distinguished persistent heatwave days without establishing causal effects.
