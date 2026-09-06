# Analytical results brief

## Computed benchmark results

Across 2014–2024 at h=1 there were **1320** eligible held-out issue dates and **56** positive future three-day windows. The strongest required model was **logistic S1** (AP **0.522**, Brier **0.0276**). The leading suitable S2 tree selected for explanation was **xgboost**. Its S2 result was AP **0.427**, balanced accuracy **0.868**, recall **0.929**, precision **0.176**, and Brier score **0.0310**. The same family's S1 result was AP **0.288** and Brier **0.0350**. The S1 logistic temperature benchmark had AP **0.522** and Brier **0.0276**; the observed-transition baseline had AP **0.357**, and the seasonal baseline had AP **0.178**.

|   lead |        n |   positive_n |   average_precision |   balanced_accuracy |   recall |   precision |   brier_score |
|-------:|---------:|-------------:|--------------------:|--------------------:|---------:|------------:|--------------:|
|  1.000 | 1320.000 |       56.000 |               0.427 |               0.868 |    0.929 |       0.176 |         0.031 |
|  3.000 | 1320.000 |       56.000 |               0.165 |               0.709 |    0.768 |       0.089 |         0.038 |
|  7.000 | 1320.000 |       56.000 |               0.216 |               0.647 |    0.679 |       0.073 |         0.039 |

The paired held-out-year bootstrap estimated S2−S1 AP = **0.138** (95% interval **0.012 to 0.188**) and Brier difference = **-0.0040** (95% interval **-0.0086 to -0.0002**). These intervals quantify sensitivity to which complete test seasons are represented; they do not make overlapping daily windows independent and do not provide forecast prediction intervals.

For onset-risk dates (`Tmax(t)<36 °C`), xgboost S2 evaluated **1215** dates spanning **15** positive spells: recall **0.733**, precision **0.057**, with **182** false alerts. This is day-level spell-start detection, not whole-event detection.

The 7-versus-14-day logistic sensitivity yielded AP 0.522 versus 0.494. The relative-threshold sensitivity is reported separately with fold-cutoff-specific climatology and its own prevalence/baselines; it does not replace the fixed 36 °C primary endpoint.

Out-of-sample SHAP was computed for the uncalibrated xgboost tree outputs using training-only backgrounds. The leading mean absolute attribution features were tmax_latest, rh_mean_mean7, tmax_mean7, time_decades, soil_moisture_mean_latest. Correlated predictors can share attribution; these rankings and block permutations are predictive diagnostics, not causal meteorological effects.

## Failures and limitations

- The source remains an unidentified Meteoblue-formatted export; station/product identity, homogenization, coordinates, and real release latency are unresolved.
- Positive windows overlap within a much smaller number of physical spells. Effective extreme-event information is therefore far below the daily row count.
- Eleven held-out seasons provide limited uncertainty resolution, and several contain no positive windows; those seasons were retained with undefined class-specific metrics explicitly missing.
- Of 13,068 candidate-fold evaluations, 1,584 average-precision values were undefined because the validation block had no positives; none was an estimator exception, and the rows remain in the tuning log.
- Model-family selection from pooled outer results is descriptive. It is not an unbiased estimate of an adaptive model-selection procedure.
- No one-to-five-month experiment was executed; monthly feasibility is only 212 records (39 positive months).

## Table/figure-to-question map

| Question | Evidence |
|---|---|
| Exact target, timing, and leakage | `prediction_contract.md`; Figure 1; split manifest |
| Class balance and dependence | sample-size/class-prevalence tables; Figure 2 |
| Does S2 beat S1/baselines? | pooled metrics; paired year-block comparisons; Figure 4 |
| Are probabilities and decisions usable? | calibration tables/reliability source; Figure 5 |
| What drives held-out tree scores? | SHAP/background/permutation tables; Figure 6 |
| Where are misses and false alerts? | onset table; held-out cases; Figure 7 |
