# Manuscript Results draft

The raw series contained 19,316 consecutive daily dates; 2024 was incomplete overall but complete for March–June. Mean $T_{\max}$ was 30.14 $^\circ$C and the observed maximum was 40.2 $^\circ$C. The operational definition identified 377 days, whereas requiring at least three consecutive days identified 270 days in 49 events. Descriptive statistics and distributions are reported first in Tables 1–4 and Figures 1–3. Raw hot-season correlations changed in magnitude after removal of the calendar-day climatology, indicating that shared seasonality explained part of several unadjusted relationships.

Across complete annual series and complete hot seasons, trend estimates were outcome-dependent. March–June mean $T_{\max}$ increased by 0.165 $^\circ$C per decade (95% CI 0.044 to 0.287). Parametric and nonparametric sensitivity estimates are shown together in Table 9.

The count distribution was assessed before regression. Comparison of Poisson and NB2 supported the **negative-binomial** model under the prespecified dispersion/AIC rule. Persistent March–June heatwave-day incidence had $\mathrm{IRR}=1.029$ per decade (95% CI 0.727 to 1.455). Residual, influence, and leave-one-year-out results are presented in Figure 6 and Table 13; influential years were not deleted from the primary estimate.

After adjustment for seasonal Fourier terms and long-term trend, antecedent associations per $1\,\mathrm{SD}$ were heterogeneous (Table 15, Figure 7). These are predictive associations, not meteorological causal effects. Blocked validation provided mean full-model $\mathrm{PR\text{-}AUC}=0.638$ and Brier score 0.033; comparison with the seasonal-trend base model is in Table 16.

In rolling-origin temperature validation, `climatology_linear_trend` had the lowest mean $\mathrm{RMSE}$ (1.114 $^\circ$C), compared with 1.661 $^\circ$C for seasonal naive and 1.175 $^\circ$C for climatology. Coverage departures in Table 19 caution against precise long-range predictions. Accordingly, 2025–2029 results are labeled conditional trend scenarios with wide count intervals, not deterministic forecasts. Five-year forecast skill is not established.
