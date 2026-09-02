# Article viability assessment

## Decision framework

The available daily series can support a focused original meteorological article. It cannot support causal claims about urbanization, tree cover, health, or vulnerability, and it cannot support precise five-year heatwave-count forecasts. Viability therefore depends on a narrow analytical question, direct modeling of the principal heatwave outcome, honest interpretation of the null count trend, and explicit separation of association from causation.

## Scope A — core article

**Content.** Long-term temperature trends, sensitivity to six heatwave definitions and two percentile reference periods, Poisson-versus-NB2 comparison, primary persistent heatwave-day count regression, diagnostics, and influential-year sensitivity.

- Scientific coherence: high. It directly connects background warming with definition-dependent event behavior.
- Originality: moderate to high for a Dhaka-specific long record analyzed with formal definition sensitivity and direct count regression.
- Analytical depth: adequate. It goes beyond descriptive counts and ordinary least squares.
- Sample-size adequacy: adequate for a parsimonious one-predictor annual NB2 model (53 hot seasons), but not for complex count trajectories.
- Model reliability: acceptable after NB2-consistent residual and case-deletion diagnostics.
- Risk of overclaiming: low if the null persistent-count result is stated directly.
- Suitability for *Meteorology and Atmospheric Physics*: plausible, subject to data-source documentation and homogenization limitations.
- Main figures: approximately six without the association analysis.
- Main tables: approximately six.
- Principal limitation: one daily location series with only 49 primary persistent events and no independent homogenization metadata.
- Likely reviewer criticism: a statistically uncertain persistent-event trend may appear modest unless definition sensitivity and the distinction from background warming are explained clearly.

**Assessment.** Scope A is independently defensible and is the fallback if the association model fails sensitivity or validation.

## Scope B — expanded original article

**Content.** Scope A plus logistic GEE estimates for antecedent humidity, precipitation, wind speed, and pressure; event-onset sensitivity; alternate lag windows; influential-season exclusion; chronological validation; and pooled strictly out-of-sample performance.

- Scientific coherence: high. Antecedent conditions address why persistent days are distinguishable without being called causal drivers.
- Originality: higher than Scope A because it combines direct count modeling, definition sensitivity, leakage-safe antecedent construction, repeated-measures inference, and blocked validation.
- Analytical depth: strong for the available single-location dataset.
- Sample-size adequacy: 6,466 March–June daily observations after lag construction, 270 persistent days, and 49 onset days. This is adequate for the parsimonious four-predictor model but requires clustered inference and cautious event-level interpretation.
- Model reliability: acceptable with qualifications. The primary GEE converged; antecedent-predictor VIFs were 1.44–2.26; the AR(1) dependence estimate was 0.576; strictly out-of-sample pooled predictions comprised 3,050 days and 125 positives.
- Risk of overclaiming: moderate. Strong odds ratios can be mistaken for causal effects or independent physical mechanisms.
- Suitability for *Meteorology and Atmospheric Physics*: plausible if physical interpretation remains cautious and model validation is prominent.
- Main figures: seven.
- Main tables: eight.
- Principal limitation: outcome days within the same event are dependent and the model is observational. Event-onset sensitivity reduces, but does not eliminate, this concern.
- Likely reviewer criticism: humidity and precipitation are correlated in rank, predictor definitions were not prospectively registered, and pressure reverses direction in the onset analysis.

The pooled full model improved Brier score from 0.0383 to 0.0301, ROC-AUC from 0.772 to 0.937, and PR-AUC from 0.192 to 0.453 relative to the seasonal-trend base model. Its pooled calibration intercept was 0.651 and slope 0.914. Per-season AUC metrics were left undefined when no positive days occurred. The onset sensitivity preserved the directions for humidity (OR 0.464, 95% CI 0.338–0.638), precipitation (OR 0.298, 95% CI 0.097–0.920), and wind speed (OR 1.526, 95% CI 1.121–2.076). Pressure changed from an imprecise inverse association in the persistent-day model (OR 0.656, 95% CI 0.404–1.065) to a positive onset association (OR 1.688, 95% CI 1.107–2.574), so pressure is not a robust principal finding.

**Assessment.** Scope B passes the prespecified leakage, collinearity, temporal-dependence, calibration, blocked-validation, event-onset, alternate-lag, and influential-season checks sufficiently to remain in the main article. Its claims must remain “adjusted antecedent associations,” not effects or drivers.

## Scope C — forecasting-enhanced article

**Content.** Scope B plus rolling-origin monthly-temperature forecasting and future count scenarios.

- Scientific coherence: low to moderate. Monthly temperature prediction is adjacent to, but does not validate, daily persistent-event prediction.
- Originality: not improved by model complexity alone.
- Analytical depth: technically adequate for monthly temperature, inadequate for exact future heatwave events.
- Sample-size adequacy: ten forecast origins with four March–June targets each support model comparison but not five-year daily-event claims.
- Model reliability: simple climatology plus trend had the lowest mean RMSE (1.114 °C); SARIMAX was substantially worse (1.607 °C). Interval calibration was imperfect.
- Risk of overclaiming: high because monthly means cannot reconstruct consecutive daily threshold exceedances.
- Suitability for *Meteorology and Atmospheric Physics*: useful as a supplementary negative result, not as the article center.
- Main figures: eight if retained.
- Main tables: nine if retained.
- Principal limitation: no validated daily simulation links monthly forecasts to persistent event counts.
- Likely reviewer criticism: forecasting broadens an already multifaceted article and the complex time-series model does not beat the simple baseline.

**Assessment.** Scope C is rejected. Forecast validation is retained only in the supplement to demonstrate that reliable five-year heatwave-count prediction was not established. Future-count scenarios are excluded from the manuscript's principal evidence.

## Final decision

**GO: a Scope B original article is defensible.** The narrowest coherent paper includes long-term temperature trends, heatwave-definition sensitivity, NB2 modeling of March–June persistent heatwave-day counts, and carefully qualified adjusted antecedent meteorological associations with chronological validation. The primary persistent-count trend is a scientifically informative null result: IRR 1.029 per decade (95% CI 0.727–1.455; p = 0.872). Background temperatures warmed, but the monotonic trend in primary persistent counts was not statistically detectable. Forecasting belongs in the supplement, and no exact 2025–2029 heatwave count belongs in the article.

Outstanding submission conditions are independent verification of data provenance and homogeneity, confirmation of the meteorological observation/site description, and manual bibliographic review by the authors.
