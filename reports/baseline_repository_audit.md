# Baseline repository audit

- Starting main commit: `926400ce49ebf2e8e87561beeedb7be93a19dcaf`.
- Baseline branch was clean and synchronized with `origin/main`.
- Legacy contents: one 49-cell notebook (24 code cells), 11 main PNG figures plus two supplementary PNGs, three tabular source files, and shapefiles.
- Unchanged notebook execution: **failed** after 4.523 seconds at the study-area cell with `ModuleNotFoundError: geopandas`. The README environment omits that import.
- The notebook uses a one-day Tmax >= 36 °C definition, descriptive OLS trends, Pearson correlations, annual random/holdout ML procedures, and SARIMA AIC to support model preference. It does not provide direct count regression, persistent-event primary inference, leakage-safe adjusted associations, rolling-origin comparison, or calibrated interval validation.
- Forecast conversion regresses annual threshold-day counts on annual mean Tmax, then converts model mean-temperature extrapolations into deterministic counts. This conversion was not validated on unseen seasons.
- Potential leakage/tautology: heat index and VPD contain temperature information and cannot serve as predictors of a Tmax-defined outcome; current-day variables are not temporally antecedent.
- Unsupported design claims include an urban heat-island strengthening attribution, human vulnerability effects, causal tree-loss effects, and precise 2029 counts.
- README values were treated as claims, not truth; all rebuilt numbers come from the raw CSV. Its basic full-record values do reproduce closely (mean Tmax **30.14 °C**, maximum **40.2 °C** on 2023-05-09, **377** one-day threshold days and **136** contiguous runs). Its annual trend estimates do not follow the required complete-year rule: rebuilt complete-year slopes are 0.192 °C/decade for Tmax and 0.208 °C/decade for Tmin, and their difference is not significant. The notebook cannot reproduce fully from top to bottom because of the missing dependency.
