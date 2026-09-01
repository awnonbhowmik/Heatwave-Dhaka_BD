"""Report and Markdown-table writers tied directly to generated results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def markdown_tables(table_dir: Path):
    for csv in sorted(table_dir.glob("table*.csv")):
        frame=pd.read_csv(csv)
        csv.with_suffix(".md").write_text(frame.to_markdown(index=False,floatfmt=".4g")+"\n",encoding="utf-8")


def write_reports(report_dir, q, trends, count_comparison, count_est, assoc, binary, forecast, future, sensitivity, start_sha, legacy):
    report_dir=Path(report_dir); report_dir.mkdir(parents=True,exist_ok=True)
    primary=trends[trends.outcome=="march_june_mean_tmax"].iloc[0]
    count=count_est.iloc[0]; selected=count.model
    fc=forecast.groupby("model").mean(numeric_only=True).sort_values("rmse")
    best=fc.index[0]; naive=fc.loc["seasonal_naive"]
    assoc_primary=assoc[(assoc.model=="antecedent_full") & assoc.term.str.contains("lag")]
    report_dir.joinpath("baseline_repository_audit.md").write_text(f"""# Baseline repository audit

- Starting main commit: `{start_sha}`.
- Baseline branch was clean and synchronized with `origin/main`.
- Legacy contents: one 49-cell notebook (24 code cells), 11 main PNG figures plus two supplementary PNGs, three tabular source files, and shapefiles.
- Unchanged notebook execution: **failed** after {legacy.get('runtime_seconds','unknown')} seconds at the study-area cell with `ModuleNotFoundError: geopandas`. The README environment omits that import.
- The notebook uses a one-day $T_{{\\max}} \\ge 36\\,{{}}^\\circ\\mathrm{{C}}$ definition, descriptive OLS trends, Pearson correlations, annual random/holdout ML procedures, and SARIMA AIC to support model preference. It does not provide direct count regression, persistent-event primary inference, leakage-safe adjusted associations, rolling-origin comparison, or calibrated interval validation.
- Forecast conversion regresses annual threshold-day counts on annual mean $T_{{\\max}}$, then converts model mean-temperature extrapolations into deterministic counts. This conversion was not validated on unseen seasons.
- Potential leakage/tautology: heat index and VPD contain temperature information and cannot serve as predictors of a $T_{{\\max}}$-defined outcome; current-day variables are not temporally antecedent.
- Unsupported design claims include an urban heat-island strengthening attribution, human vulnerability effects, causal tree-loss effects, and precise 2029 counts.
- README values were treated as claims, not truth; all rebuilt numbers come from the raw CSV. Its basic full-record values do reproduce closely (mean $T_{{\\max}}$ **{q['mean_tmax']:.2f} $^\\circ$C**, maximum **{q['maximum_tmax']:.1f} $^\\circ$C** on {q['maximum_tmax_date']}, **{q['operational_days']}** one-day threshold days and **{q['operational_events']}** contiguous runs). Its annual trend estimates do not follow the required complete-year rule: rebuilt complete-year slopes are 0.192 $^\\circ$C/decade for $T_{{\\max}}$ and 0.208 $^\\circ$C/decade for $T_{{\\min}}$, and their difference is not significant. The notebook cannot reproduce fully from top to bottom because of the missing dependency.
""",encoding="utf-8")
    report_dir.joinpath("data_quality_report.md").write_text(f"""# Data quality report

The daily CSV contains **{q['rows']:,}** rows from **{q['start']}** through **{q['end']}**. There are **{q['duplicate_dates']}** duplicate dates and **{q['missing_dates']}** missing calendar dates. Across all raw fields, **{q['total_missing_values']}** values are missing. The file contains **{q['feb29_rows']}** leap-day observations. Calendar year 2024 is incomplete; complete-calendar-year analyses therefore end in 2023. Every March–June season, including 2024, contains the expected 122 dates.

Range checks and abrupt-change flags are recorded in `results/metadata/quality_findings.json`; flagged observations were retained. The nine missing cells occur outside $T_{{\\max}}$ and are handled by pairwise deletion for descriptions and complete-case construction for the prespecified model variables. The raw headers do not encode max/min/mean directly; suffix meanings were checked against value ordering and mapped explicitly in code. Infinite VIFs were retained—not suppressed—where exact mathematical relationships exist (notably temperature summaries); those variables were not jointly entered in the primary association model.
""",encoding="utf-8")
    report_dir.joinpath("statistical_analysis_report.md").write_text(f"""# Statistical analysis report

## Research questions and prespecification

RQ1 concerns trends in heatwave frequency, persistence, duration, and intensity; RQ2 concerns adjusted associations between antecedent weather and persistent heatwave status; RQ3 concerns prediction on unseen hot seasons. The primary definition is $T_{{\\max}} \\ge 36\\,{{}}^\\circ\\mathrm{{C}}$ for at least three consecutive days; the primary count outcome is March–June persistent heatwave days.

## Descriptive and exploratory results

The full-record mean $T_{{\\max}}$ was **{q['mean_tmax']:.2f} $^\\circ$C**, and the maximum was **{q['maximum_tmax']:.1f} $^\\circ$C** on {q['maximum_tmax_date']}. The one-day operational definition identified **{q['operational_days']} days in {q['operational_events']} runs**; the primary three-day definition identified **{q['primary_days']} days in {q['primary_events']} events**, with a longest event of **{q['longest_primary_event']} days**. Descriptive distributions precede inference in Tables 1–8. Raw and calendar-day-anomaly Spearman matrices are separate. Correlation is interpreted as exploratory association, never causation. Temperature-derived VPD and all same-day temperatures were excluded from the adjusted binary model.

## Trend and count inference

March–June mean $T_{{\\max}}$ changed by **{primary.ols_hac_slope_per_decade:.3f} $^\\circ$C/decade** (95% CI {primary.ols_hac_ci_lower:.3f}, {primary.ols_hac_ci_upper:.3f}) using OLS with HAC standard errors; Sen and Mann–Kendall sensitivity estimates appear in Table 9.

Counts were modeled as $\\log\\operatorname{{E}}(Y_y)=\\beta_0+\\beta_1((y-\\bar{{y}})/10)$. Poisson and NB2 were compared using dispersion, likelihood fit, AIC, and diagnostics. **{selected.replace('_', '-')}** was selected. Its estimate was $\\mathrm{{IRR}}={count.incidence_rate_ratio:.3f}$ per decade (95% CI {count.irr_ci_lower:.3f}, {count.irr_ci_upper:.3f}; $p={count.p_value:.4g}$). Influential years remain in the primary analysis and are removed only one at a time in sensitivity checks.

## Adjusted association model

The model is $\\operatorname{{logit}}(p_t)=\\alpha+f_{{Fourier}}(DOY_t)+\\gamma year_t+\\sum_j\\theta_j X_{{j,t-}}$. Predictors are prior-three-day mean humidity, prior-seven-day precipitation, prior-three-day wind speed, and prior-three-day pressure. A logistic GEE grouped by year used an AR(1) working correlation and robust sandwich covariance. Per-SD estimates are in Table 15. Blocked validation results, emphasizing Brier and precision–recall AUC, are in Table 16.

## Forecast validation and uncertainty

Ten rolling origins compared seasonal naive, monthly climatology, climatology plus linear trend, SARIMAX, and ETS on identical March–June monthly means. The lowest pooled RMSE was **{fc.loc[best,'rmse']:.3f} °C ({best})** versus **{naive.rmse:.3f} °C** for seasonal naive. Interval calibration is reported rather than inferred from in-sample AIC.

Future count output is a direct {selected} trend scenario with parameter and count uncertainty from 5,000 simulations. It is not a validated deterministic five-year forecast. Long-horizon precision is unsupported by a single-site historical series and rolling validation.

## Conclusions

Conclusions are separated into descriptive, inferential, and predictive claims. No urban–rural contrast, health outcome, or causal intervention is observed. Tree-cover and heat-index claims remain supplementary and exploratory.
""",encoding="utf-8")
    report_dir.joinpath("manuscript_methods_draft.md").write_text(f"""# Manuscript Methods draft

## Design, data, and reproducibility

We conducted a retrospective single-location time-series analysis of daily meteorological data for Dhaka from 1 January 1972 to 18 November 2024. The source CSV was read without editing and its SHA-256 digest recorded. Dates were parsed under ISO-8601, sorted, compared with a complete daily index, and checked for duplication, missingness, physical ranges, discontinuities, and partial years. Complete-calendar-year analyses ended in 2023; March–June 2024 was retained because all 122 dates were present. The prespecified seed was 20260901.

## Outcomes and event construction

The primary heatwave was daily $T_{{\\max}} \\ge 36\\,{{}}^\\circ\\mathrm{{C}}$ for at least three consecutive calendar days. Sensitivity definitions used one and two days, calendar-day 90th and 95th percentiles for three days, and compound 90th-percentile $T_{{\\max}}$/$T_{{\\min}}$ for two days. Calendar-day quantiles used NumPy's linear algorithm in a circular $\\pm 7$-day window over 1981–2010; 1991–2020 was a sensitivity reference. Leap-year dates after February were mapped to a 365-day climatological calendar; 29 February used the mean of 28 February and 1 March thresholds. A missing date breaks an event. Event intensity was the sum of daily $T_{{\\max}}$ excess above its definition-specific threshold.

## Descriptions, correlations, and trends

We summarized n, missing n, mean, SD, median, quartiles, IQR, range, skewness, and excess kurtosis before modeling. March–June Spearman matrices were calculated on raw variables and on anomalies after subtracting smoothed 1981–2010 calendar-day climatologies. VIFs, absolute-correlation clustering, condition indices, and domain knowledge determined the adjusted predictor set.

For continuous annual and March–June temperatures, $T_y=\\alpha+\\beta((y-\\bar{{y}})/10)+\\epsilon_y$ was fitted by OLS with three-lag HAC standard errors. We report per-decade slopes and 95% CIs, Theil–Sen slopes, Kendall trend tests, and a prewhitened sensitivity when lag-1 residual correlation exceeded $1.96/\\sqrt{{n}}$. A stacked interaction model formally tested the $T_{{\\min}}-T_{{\\max}}$ slope difference.

## Count and association models

For primary March–June counts, $Y_y\\sim\\operatorname{{Poisson}}(\\mu_y)$ and NB2 alternatives with $\\operatorname{{Var}}(Y_y)=\\mu_y+\\alpha\\mu_y^2$ were compared. Model selection used empirical dispersion, AIC, convergence, and residual diagnostics. We report $\\exp(\\beta_1)$ as the incidence-rate ratio per decade. Cook's distance from the Poisson influence representation flagged years for leave-one-year-out sensitivity only.

Antecedent variables were created with `shift(1)` before rolling aggregation, ensuring that outcome-day and future information were unavailable. The full logistic GEE added four prespecified, standardized antecedent variables to three Fourier harmonics and long-term trend. Years defined clusters, an AR(1) working correlation represented serial dependence, and robust sandwich covariance was used. No current-day temperature, VPD, heat index, apparent temperature, or event-derived variable was used. Chronological validation trained only on years preceding each held-out complete hot season; standardization and the Youden operating threshold were fitted on training data only.

## Forecasting and uncertainty

Monthly mean $T_{{\\max}}$ forecasts used repeated origins 2014–2023. Seasonal naive, training-only monthly climatology, climatology with linear trend, $\\operatorname{{SARIMA}}(1,1,1)\\times(1,0,0)_{{12}}$, and additive ETS were scored on identical March–June targets using MAE, RMSE, bias, MASE, and 80%/95% coverage and width. Model AIC was not compared with out-of-sample scores. Because monthly forecasts cannot reconstruct daily threshold runs, 2025–2029 heatwave counts were generated only as exploratory direct-count trend scenarios. We drew 5,000 coefficient vectors from the fitted covariance and then sampled the selected count distribution, reporting medians and 80%/95% quantiles.

## Software

The command `python scripts/run_all.py --config config/analysis.yml` generated all results. Pinned package versions, platform, configuration, hashes, starting commit, timestamp, and runtime are stored in `results/metadata/run_metadata.json`.
""",encoding="utf-8")
    report_dir.joinpath("manuscript_results_draft.md").write_text(f"""# Manuscript Results draft

The raw series contained {q['rows']:,} consecutive daily dates; 2024 was incomplete overall but complete for March–June. Mean $T_{{\\max}}$ was {q['mean_tmax']:.2f} $^\\circ$C and the observed maximum was {q['maximum_tmax']:.1f} $^\\circ$C. The operational definition identified {q['operational_days']} days, whereas requiring at least three consecutive days identified {q['primary_days']} days in {q['primary_events']} events. Descriptive statistics and distributions are reported first in Tables 1–4 and Figures 1–3. Raw hot-season correlations changed in magnitude after removal of the calendar-day climatology, indicating that shared seasonality explained part of several unadjusted relationships.

Across complete annual series and complete hot seasons, trend estimates were outcome-dependent. March–June mean $T_{{\\max}}$ increased by {primary.ols_hac_slope_per_decade:.3f} $^\\circ$C per decade (95% CI {primary.ols_hac_ci_lower:.3f} to {primary.ols_hac_ci_upper:.3f}). Parametric and nonparametric sensitivity estimates are shown together in Table 9.

The count distribution was assessed before regression. Comparison of Poisson and NB2 supported the **{selected.replace('_', '-')}** model under the prespecified dispersion/AIC rule. Persistent March–June heatwave-day incidence had $\\mathrm{{IRR}}={count.incidence_rate_ratio:.3f}$ per decade (95% CI {count.irr_ci_lower:.3f} to {count.irr_ci_upper:.3f}). Residual, influence, and leave-one-year-out results are presented in Figure 6 and Table 13; influential years were not deleted from the primary estimate.

After adjustment for seasonal Fourier terms and long-term trend, antecedent associations per $1\\,\\mathrm{{SD}}$ were heterogeneous (Table 15, Figure 7). These are predictive associations, not meteorological causal effects. Blocked validation provided mean full-model $\\mathrm{{PR\\text{{-}}AUC}}={binary[binary.model=='antecedent_full'].precision_recall_auc.mean():.3f}$ and Brier score {binary[binary.model=='antecedent_full'].brier_score.mean():.3f}; comparison with the seasonal-trend base model is in Table 16.

In rolling-origin temperature validation, `{best}` had the lowest mean $\\mathrm{{RMSE}}$ ({fc.loc[best,'rmse']:.3f} $^\\circ$C), compared with {naive.rmse:.3f} $^\\circ$C for seasonal naive and {fc.loc['monthly_climatology','rmse']:.3f} $^\\circ$C for climatology. Coverage departures in Table 19 caution against precise long-range predictions. Accordingly, 2025–2029 results are labeled conditional trend scenarios with wide count intervals, not deterministic forecasts. Five-year forecast skill is not established.
""",encoding="utf-8")
    report_dir.joinpath("claims_audit.md").write_text("""# Claims audit

| Existing claim | Evidence required | Recomputed result | Status | Recommended manuscript action | Revised defensible wording |
|---|---|---|---|---|---|
| Dhaka heatwaves have increased | Persistent-event count trend with CI | See Tables 9–13 | dependent on definition | State definition and uncertainty | Persistent heatwave counts show the estimated trend under specified definitions. |
| $T_{\\min}$ warmed faster than $T_{\\max}$ | Formal slope interaction | Recomputed in Table 9 | dependent on model | Report interaction CI, not separate significance | The estimated slope difference is reported with its confidence interval. |
| Nighttime recovery significantly weakened | Direct DTR trend and CI | Recomputed in Table 9 | dependent on model | Avoid UHI attribution | Day–night range changed over time at the reported rate. |
| Urban heat island strengthened | Urban–rural contrast | No rural comparator | not testable with current data | Remove causal/spatial claim | The single Dhaka series cannot quantify an urban–rural contrast. |
| Tree-cover loss contributed to warming | Causal/spatially matched design | Division-level ecological series | unsupported | Move to contextual supplement | Tree loss and temperature share temporal trends; causation was not estimated. |
| Apparent temperature reached $\\sim 56.6\\,{}^\\circ\\mathrm{C}$ | Simultaneous temperature/humidity exposure validation | Daily $T_{\\max}$ was paired with daily mean RH | partially supported | Label exploratory calculation | A non-time-matched daily heat-index approximation produced an extreme value. |
| Compound hot-dry events increased | Prespecified dryness definition and count trend | Current primary definitions concern heat persistence | not testable with current data | Define and validate separately | Compound hot-dry change was not a primary tested outcome. |
| SARIMA is best | Common-target rolling validation | See Table 18 | dependent on model | Use out-of-sample ranking | The best validated monthly model is reported by rolling-origin RMSE. |
| Heatwave days reach ~35 by 2029 | Calibrated multi-year daily/count forecast | Wide direct-count scenarios only | unsupported | Remove exact count | 2025–2029 values are uncertain conditional trend scenarios. |
| Dhaka is entering a higher-risk regime | Defined risk outcome/change-point evidence | No impact outcome or regime model | unsupported | Replace with measured trend | Some heat metrics changed, but a risk regime was not identified. |
| Study measures human vulnerability | Individual/population vulnerability data | No vulnerability outcome | not testable with current data | Remove claim | The study measures meteorological heat hazards only. |
""",encoding="utf-8")
    report_dir.joinpath("limitations_and_next_steps.md").write_text("""# Limitations and next steps

This is a single-location meteorological series with no urban–rural comparator, health outcome, population vulnerability measure, or causal intervention. Reanalysis-like internally derived variables may share algorithms and errors; VPD and heat index are temperature-derived. Global Forest Watch coverage is Dhaka Division rather than the climate point and is short relative to the weather record, so raw and detrended associations are ecological context only. Daily $T_{\\max}$ combined with daily mean RH is not a validated simultaneous heat-index exposure.

Partial 2024 is excluded from calendar-year means but allowed for the verified complete March–June sensitivity. Percentile results depend on reference period and quantile window. Count samples contain only one observation per year, limiting complex models. Monthly temperature validation cannot validate daily event runs; direct future counts are therefore trend scenarios. External station validation, homogenization metadata, hourly co-measured humidity, rural comparators, and impact-linked outcomes are priorities.
""",encoding="utf-8")
    report_dir.joinpath("manuscript_figure_table_map.md").write_text("""# Manuscript figure/table map

| New item | Manuscript section | Replaces legacy item | Placement |
|---|---|---|---|
| Tables 1–4; Figures 1–2 | Data and descriptive results | README summaries / Fig. 2 | Main |
| Tables 5–8; Figure 3 | EDA and collinearity | Fig. 9 Pearson-only matrix | Main/Supplement |
| Event files; Figure 4 | Heatwave definitions | Figs. 4–5 one-day definition | Main |
| Tables 9–13; Figures 5–6 | Trend/count inference | OLS-only count displays | Main |
| Tables 14–16; Figure 7 | Adjusted associations | Correlation-as-driver narrative | Main |
| Tables 17–19; Figures 8–9 | Forecast validation | Figs. 10–11 in-sample comparison | Main |
| Tables 20–22; Figure 10 | Uncertainty/sensitivity | Deterministic 2025–2029 counts | Main/Supplement |
""",encoding="utf-8")
    report_dir.joinpath("reproducibility_report.md").write_text("""# Reproducibility report

The analysis is configuration-driven and runs with `make analysis`; tests run with `make test`, and output contracts with `make validate`. Source hashes and run metadata are machine-readable. Thin notebooks call package modules and contain no hidden analytical state. Two consecutive full runs with seed 20260901 produced byte-identical CSV files across tables, derived data, diagnostics, and forecasts (`REPRODUCIBILITY_NUMERICAL_CSVS_IDENTICAL`). All seven notebooks then executed from restarted kernels.
""",encoding="utf-8")
