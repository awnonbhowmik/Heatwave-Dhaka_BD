# Decision for professor

## Recommendation

**Stop at the evidence-review boundary.** Extra meteorology improved h=1 AP and Brier score with paired-year support for weighted_rf, xgboost, but the strongest overall model was logistic S1; gains were therefore family-specific rather than universal. The strongest defensible original claim is a retrospective, single-location result about short-lead prediction of future three-day persistent-hot windows—not operational readiness, causal mechanisms, or seasonal forecasting.

## Answers to the decision questions

- **Did extra meteorology improve over seasonal/temperature baselines?** Extra meteorology improved h=1 AP and Brier score with paired-year support for weighted_rf, xgboost, but the strongest overall model was logistic S1; gains were therefore family-specific rather than universal. At h=1, xgboost S2 AP was 0.427 versus 0.288 for its S1 counterpart, 0.522 for regularized S1 logistic, 0.357 for observed transitions, and 0.178 for seasonality.
- **Did gains survive chronological testing?** All quoted results are strictly out-of-sample across 2014–2024, including event-free seasons. The paired-year AP interval was 0.012 to 0.188; interpret stability according to whether it excludes zero.
- **Is onset performance credible?** It is preliminary: 15 distinct positive spells, recall 0.733, precision 0.057, and 182 false alerts among 1215 eligible issue dates.
- **How far does skill extend?** xgboost S2 AP by lead was h=1: 0.427, h=3: 0.165, h=7: 0.216. This is day-scale direct prediction only.
- **Are explanations stable?** The SHAP stability table shows year/horizon variation and must be read alongside performance. Leading features were tmax_latest, rh_mean_mean7, tmax_mean7, time_decades, soil_moisture_mean_latest; none is a causal claim.
- **What claim is justified?** The dataset can support a carefully qualified comparison of calendar, observed-temperature, and available-meteorology information for short-lead future persistent-hot-window prediction under retrospective chronological validation.

## Remaining weaknesses before any paper decision

Resolve the data source/site metadata and homogenization history; document actual observation/product release latency; obtain independent meteorological review of the threshold and feature summaries; decide whether the small number of physical events warrants a predictive paper; and assess novelty against current Dhaka/South Asia forecasting literature. Do not call algorithm comparison alone novel.
