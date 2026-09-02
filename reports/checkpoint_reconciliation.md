# Checkpoint reconciliation

All checkpoints were regenerated from the immutable raw files; none was entered as a target or edited manually.

| Checkpoint | Recomputed value | Reconciliation |
|---|---:|---|
| Poisson dispersion | 11.6375 | Reproduces the expected value near 11.64. |
| Poisson AIC | 659.8045 | Reproduces the expected value near 659.8. |
| NB2 AIC | 245.1811 | Reproduces the expected value near 245.2. |
| Primary persistent-day IRR per decade | 1.0288 | Reproduces the expected value near 1.029. |
| Primary IRR 95% CI | 0.7275–1.4549 | Reproduces the expected interval near 0.727–1.455. |
| Primary trend p-value | 0.8725 | Reproduces the expected value near 0.872. |
| March–June mean \(T_{\max}\) trend | 0.1653 °C/decade | Reproduces the expected value near 0.165. |
| Antecedent 3-day mean RH adjusted OR | 0.3067 | Reproduces the expected value near 0.307. |
| Antecedent 7-day precipitation adjusted OR | 0.1400 | Reproduces the expected value near 0.140. |
| Antecedent 3-day wind-speed adjusted OR | 1.3370 | Reproduces the expected value near 1.337. |
| Climatology-plus-trend rolling RMSE | 1.1135 °C | Reproduces the expected value near 1.11 °C. |
| SARIMAX rolling RMSE | 1.6072 °C | Remains substantially worse than climatology plus trend. |
| 2025–2029 count scenarios | Medians near zero with wide upper limits | Reproduces the qualitative checkpoint; excluded from main article claims because it is not a validated event forecast. |

## Corrections that changed dependent outputs without changing checkpoints

The checkpoint estimates themselves did not change. The following diagnostic and presentation corrections were made:

- Randomized quantile residuals now use the selected NB2 distribution rather than a Poisson distribution.
- Influence screening now uses case-deletion NB2 parameter distance. It identifies 1979, 2023, and 2024, rather than importing Poisson influence measures into the selected model.
- The model-comparison table now reports NB2 Pearson and deviance-style diagnostics and convergence.
- The primary duration histogram includes only \(T_{\max}\geq36\,{}^\circ\mathrm{C}\) events lasting at least three days.
- Definition summaries use a consistent March–June analysis season.
- Blocked binary validation now covers every eligible held-out season from 2000 through 2024 and reports pooled strictly out-of-sample metrics.
- Run metadata now records the merge-base starting SHA rather than a hard-coded historical SHA.
- The map no longer displays an inferred station/reference point without coordinate provenance.

These changes update figures, diagnostics, validation tables, reports, and manuscript wording, while preserving the reproducible primary effect estimates.
