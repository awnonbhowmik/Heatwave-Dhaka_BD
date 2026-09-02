# Supplementary material

## Long-Term Warming and Definition-Dependent Heatwaves in Dhaka, Bangladesh: Count Trends and Antecedent Meteorological Associations, 1972–2024

**Provisional title—professor and coauthor approval required**

## S1 Reproducibility and data-quality details

The analysis reads `data/1972_2024_Heatwave_Daily.csv` without modification and records its SHA-256 digest. The raw record contains 19,316 rows from 1972-01-01 through 2024-11-18, no duplicate dates, no missing calendar dates, and 14 leap-day observations. Nine cells are missing across non-\(T_{\max}\) fields. Calendar year 2024 is incomplete, while all March–June seasons contain 122 dates.

Physical-range checks flagged 168 daily maximum-relative-humidity values outside the nominal range used by the screening code. Abrupt-change screening flagged observations in several variables, including soil moisture. These flags were retained rather than deleted because station metadata and an independent quality-control source were unavailable. The corresponding dates and rules are preserved in `results/metadata/quality_findings.json`; authors should verify them against the data provider before submission.

Three infinite VIF warnings occur when the exploratory screen includes exact mathematical temperature relationships. They do not arise in the four-predictor antecedent model, whose VIFs were 1.44–2.26.

## S2 Calendar-day climatology

Dates were mapped to a 365-day climatological calendar. For reference years 1981–2010, the threshold for each climatological day was estimated from observations in the circular window \(d\pm7\), including 15 climatological dates. The primary algorithm used NumPy's linear sample quantile. Leap-year dates after February were shifted one climatological index, and 29 February used the mean of the thresholds for 28 February and 1 March. The 1991–2020 reference was recomputed independently.

De-seasonalized anomalies used the same 1981–2010 calendar but subtracted a smoothed calendar-day mean rather than a percentile threshold. This removes expected within-year timing but retains long-term change. It is therefore a seasonal adjustment, not a detrending operation.

## S3 Definition-reference sensitivity

With the 1981–2010 reference, the March–June count-trend IRRs were 1.069 (95% CI 0.842–1.358) for the 90th-percentile \(T_{\max}\) definition, 1.052 (0.779–1.420) for the 95th-percentile definition, and 1.320 (1.043–1.672; nominal \(p=0.0209\)) for compound 90th-percentile \(T_{\max}\)/\(T_{\min}\). The compound result had Benjamini–Hochberg \(q=0.126\) across the six primary-reference definition tests. With 1991–2020, corresponding IRRs were 1.031 (0.824–1.290), 0.944 (0.711–1.252), and 1.331 (1.021–1.735). The compound point estimate and nominal interval were stable across reference periods, while \(T_{\max}\)-only relative trends remained uncertain; this secondary signal is exploratory rather than confirmatory.

## S4 Count influence sensitivity

Selected-distribution case deletion identified 1979, 2023, and 2024 above the \(4/n\) screening threshold. Each was omitted separately in a sensitivity refit; none was removed from the primary model. Exclusion estimates remained compatible with no monotonic count trend. The complete estimates are provided in Supplementary Table S1.

**Supplementary Table S1.** Leave-one-influential-year-out NB2 count sensitivity. Source: `results/tables/supplement/supplement_tableS01_count_influence_sensitivity.csv`.

## S5 Association-model sensitivity

The full GEE used 270 persistent heatwave days across 49 events. The event-onset model reduced the positive outcome to one day per event and estimated an AR(1) dependence parameter close to zero, as expected after removing within-event sequences. Humidity, precipitation, and wind retained their primary directions. Pressure reversed, so it was excluded from the principal interpretation. One-day and seven-day feature sets provided lag-window checks. Separate exclusion of count-influential seasons preserved the three principal directions.

The primary one-standard-deviation scales were 9.19 percentage points for prior-three-day relative humidity, 55.83 mm for prior-seven-day precipitation, 3.23 recorded units for prior-three-day wind speed, and 4.21 recorded units for prior-three-day pressure. Unit labels for wind and pressure must be confirmed against provider documentation.

## S6 Binary validation by season

Each validation origin used only prior years and refitted the same GEE estimator used for the full-sample association analysis. Origin-specific discrimination metrics were undefined for a held-out season with no persistent heatwave day and were stored as missing. Such seasons remained valid for Brier score and specificity. Pooled metrics combined all strictly out-of-sample probabilities rather than averaging undefined season-level values. The full model's pooled metrics were Brier 0.0333, ROC-AUC 0.9244, PR-AUC 0.4732, calibration intercept 1.5119, and calibration slope 1.1361. The nonzero calibration intercept indicates systematic underprediction and prevents operational use without recalibration.

## S7 Forecast validation

Monthly mean \(T_{\max}\) was forecast for March–June at ten origins, 2014–2023. All models used identical training cutoffs and targets. Mean RMSEs were 1.114 °C for climatology plus linear trend, 1.120 °C for additive exponential smoothing, 1.175 °C for monthly climatology, 1.607 °C for SARIMAX, and 1.661 °C for seasonal naive. The corresponding mean absolute scaled errors were 0.994, 0.994, 1.045, 1.398, and 1.468.

Climatology-plus-trend 80% and 95% coverages were 0.675 and 0.875 with mean widths 2.367 °C and 3.620 °C. SARIMAX coverages were 0.775 and 1.000, but widths were 4.609 °C and 7.049 °C. This illustrates the accuracy–sharpness tradeoff and does not establish superior SARIMAX skill.

**Supplementary Table S2.** Forecast metrics by rolling origin and model. Source: `results/tables/supplement/supplement_tableS02_forecast_validation.csv`.

**Supplementary Figure S1. Selected meteorological pairplots.** Pairwise distributions for humidity, precipitation, wind speed, and pressure in a reproducible hot-season sample, colored by primary persistent-day status. The figure is exploratory and does not represent adjusted effects.

**Supplementary Figure S2. Rolling-origin forecast validation.** (A) RMSE distributions across origins. (B) observed versus predicted held-out March–June monthly mean \(T_{\max}\). (C) empirical 80% and 95% interval coverage. (D) mean interval widths.

## S8 Analyses not retained as article evidence

The legacy tree-cover correlation was not spatially matched to the meteorological series and changed sign after detrending; it is not used to infer causation. Legacy heat-index calculations paired daily \(T_{\max}\) with daily mean humidity rather than simultaneous observations; they are not used as physiological exposure. Monthly mean-temperature forecasts cannot be transformed into persistent daily events. Direct 2025–2029 count simulations therefore remain repository diagnostics and are not reported as forecasts in the article.

## S9 Reproducibility files

- Configuration: `config/analysis.yml`
- Full runner: `scripts/run_all.py`
- Article output generator: `src/heatwave_analysis/article_outputs.py`
- Test suite: `tests/`
- Output validator: `scripts/validate_outputs.py`
- Reproducibility audit: `reports/original_article_reproducibility_audit.md`
- Claim matrix: `reports/claim_to_evidence_matrix.csv`
- Exact reproduction command: `make article`
