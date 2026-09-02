"""Original-article tables and figures derived from the rebuilt analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .association_models import _design
from .count_models import fit_count_models
from .exploratory import descriptive_statistics
from .heatwave_events import DEFINITIONS, construct_definition
from .plotting import PALETTE, _save, _study_area_figure
from .variable_dictionary import DESCRIPTIVE_VARIABLES, PRIMARY_ASSOCIATION_PREDICTORS


DEFINITION_LABELS = {
    "operational_36c_1d": r"$T_{\max}\geq36\,{}^\circ$C, $\geq1$ day",
    "persistent_36c_2d": r"$T_{\max}\geq36\,{}^\circ$C, $\geq2$ days",
    "persistent_36c_3d": r"$T_{\max}\geq36\,{}^\circ$C, $\geq3$ days",
    "relative_90p_3d": r"Calendar-day $T_{\max}>P_{90}$, $\geq3$ days",
    "relative_95p_3d": r"Calendar-day $T_{\max}>P_{95}$, $\geq3$ days",
    "compound_90p_2d": r"Concurrent $T_{\max},T_{\min}>P_{90}$, $\geq2$ days",
}


def _write(frame: pd.DataFrame, root: Path, name: str) -> None:
    """Write exact requested paths and organized main/supplement copies."""
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / f"{name}.csv", index=False, float_format="%.10g")
    collection = root / ("supplement" if name.startswith("supplement") else "main")
    collection.mkdir(parents=True, exist_ok=True)
    frame.to_csv(collection / f"{name}.csv", index=False, float_format="%.10g")


def _long_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    hot = df[df.month.isin([3, 4, 5, 6])]
    strata = {
        "all_year_daily": df,
        "march_june_daily": hot,
        "persistent_heatwave_days": hot[hot.persistent_36c_3d.astype(bool)],
        "non_heatwave_hot_season_days": hot[~hot.persistent_36c_3d.astype(bool)],
    }
    for label, part in strata.items():
        frames.append(descriptive_statistics(part, DESCRIPTIVE_VARIABLES, label))
    return pd.concat(frames, ignore_index=True)


def definition_summary(df, daily_events, events) -> pd.DataFrame:
    rows = []
    hot_mask = df.month.isin([3, 4, 5, 6])
    variants = [(name, "1981–2010" if DEFINITIONS[name]["kind"] != "fixed" else "absolute") for name in DEFINITIONS]
    for name, reference in variants:
        status = daily_events[name].astype(bool)
        event_frame = events[events.definition.eq(name) & events.start_month.isin([3, 4, 5, 6])].copy()
        counts = pd.DataFrame({"year": df.year[hot_mask], "heatwave_days": status[hot_mask].astype(int)}).groupby("year", as_index=False).heatwave_days.sum()
        _, _, _, model, model_name, _ = fit_count_models(counts)
        ci = model.conf_int().loc["decade"]
        monthly = pd.DataFrame({"month": df.month[status & hot_mask], "status": 1}).groupby("month").status.sum()
        rows.append({
            "definition": name,
            "definition_label": DEFINITION_LABELS[name],
            "reference_period": reference,
            "analysis_season": "March–June",
            "qualifying_days": int(status[hot_mask].sum()),
            "events": len(event_frame),
            "median_duration_days": event_frame.duration.median(),
            "mean_duration_days": event_frame.duration.mean(),
            "maximum_duration_days": event_frame.duration.max(),
            "peak_intensity_celsius": event_frame.maximum_excess.max(),
            "cumulative_intensity_degree_days": event_frame.cumulative_excess.sum(),
            "median_onset_day_of_year": event_frame.onset_day_of_year.median(),
            "peak_month": int(monthly.idxmax()) if len(monthly) else np.nan,
            "selected_count_model": model_name,
            "irr_per_decade": np.exp(model.params["decade"]),
            "irr_ci_lower": np.exp(ci[0]),
            "irr_ci_upper": np.exp(ci[1]),
            "p_value": model.pvalues["decade"],
        })
    for name in ("relative_90p_3d", "relative_95p_3d", "compound_90p_2d"):
        status, event_frame, _ = construct_definition(df, name, (1991, 2020))
        event_frame = event_frame[event_frame.start_month.isin([3, 4, 5, 6])].copy()
        counts = pd.DataFrame({"year": df.year[hot_mask], "heatwave_days": status[hot_mask].astype(int)}).groupby("year", as_index=False).heatwave_days.sum()
        _, _, _, model, model_name, _ = fit_count_models(counts)
        ci = model.conf_int().loc["decade"]
        monthly = pd.DataFrame({"month": df.month[status & hot_mask], "status": 1}).groupby("month").status.sum()
        rows.append({
            "definition": f"{name}_reference_1991_2020",
            "definition_label": DEFINITION_LABELS[name],
            "reference_period": "1991–2020",
            "analysis_season": "March–June",
            "qualifying_days": int(status[hot_mask].sum()), "events": len(event_frame),
            "median_duration_days": event_frame.duration.median(), "mean_duration_days": event_frame.duration.mean(),
            "maximum_duration_days": event_frame.duration.max(), "peak_intensity_celsius": event_frame.maximum_excess.max(),
            "cumulative_intensity_degree_days": event_frame.cumulative_excess.sum(),
            "median_onset_day_of_year": event_frame.onset_day_of_year.median(),
            "peak_month": int(monthly.idxmax()) if len(monthly) else np.nan,
            "selected_count_model": model_name, "irr_per_decade": np.exp(model.params["decade"]),
            "irr_ci_lower": np.exp(ci[0]), "irr_ci_upper": np.exp(ci[1]), "p_value": model.pvalues["decade"],
        })
    return pd.DataFrame(rows)


def correlation_screening_table(df, raw_corr, anomaly_corr) -> pd.DataFrame:
    selected = [
        "tmax", "tmin", "rh_mean", "precipitation", "wind_speed_mean",
        "cloud_cover", "sunshine_duration", "shortwave_radiation",
        "pressure_mean", "soil_moisture_mean",
    ]
    rows = []
    for matrix_name, matrix in [("raw_spearman", raw_corr), ("deseasonalized_spearman", anomaly_corr)]:
        for i, left in enumerate(selected):
            for right in selected[i + 1:]:
                rows.append({"section": matrix_name, "variable_1": left, "variable_2": right,
                             "estimate": matrix.loc[left, right]})
    hot = df[df.month.isin([3, 4, 5, 6])]
    x = hot[PRIMARY_ASSOCIATION_PREDICTORS].dropna()
    z = (x - x.mean()) / x.std(ddof=0)
    for i, predictor in enumerate(PRIMARY_ASSOCIATION_PREDICTORS):
        rows.append({
            "section": "antecedent_predictor_screening", "variable_1": predictor,
            "variable_2": "retained", "estimate": variance_inflation_factor(z.to_numpy(), i),
            "decision": "retained; antecedent, non-target-derived, interpretable, VIF below 5",
        })
    for excluded, reason in [
        ("same-day Tmax/Tmin/Tmean", "defines or directly encodes the outcome"),
        ("heat index/apparent temperature/VPD", "temperature-derived; leakage or tautology risk"),
        ("event duration/intensity", "defined using outcome or future event information"),
    ]:
        rows.append({"section": "excluded_predictor_group", "variable_1": excluded,
                     "variable_2": "excluded", "estimate": np.nan, "decision": reason})
    return pd.DataFrame(rows)


def _temperature_line(ax, series, outcome, title, ylabel):
    d = series[["year", outcome]].dropna()
    decade = (d.year - d.year.mean()) / 10
    x = sm.add_constant(decade)
    model = sm.OLS(d[outcome], x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    grid_year = np.linspace(d.year.min(), d.year.max(), 250)
    grid_x = sm.add_constant((grid_year - d.year.mean()) / 10)
    fitted = np.asarray(grid_x) @ np.asarray(model.params)
    covariance = np.asarray(model.cov_params())
    se = np.sqrt(np.einsum("ij,jk,ik->i", grid_x, covariance, grid_x))
    ax.scatter(d.year, d[outcome], s=20, color=PALETTE[0], alpha=.75)
    ax.plot(grid_year, fitted, color=PALETTE[1], lw=2)
    ax.fill_between(grid_year, fitted - 1.96 * se, fitted + 1.96 * se, color=PALETTE[1], alpha=.18)
    ax.set(title=title, xlabel="Year", ylabel=ylabel)


def _figure02(df, count_distribution, main_figures):
    hot = df[df.month.isin([3, 4, 5, 6])]
    counts = hot.groupby("year").persistent_36c_3d.sum()
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    sns.histplot(df.tmax, bins=35, kde=True, ax=ax[0, 0], color=PALETTE[0])
    ax[0, 0].set(xlabel=r"$T_{\max}$ ($^\circ$C)", title=r"Daily $T_{\max}$ distribution")
    clim = df.groupby("month")[["tmax", "tmin"]].mean()
    ax[0, 1].plot(clim.index, clim.tmax, marker="o", label=r"$T_{\max}$", color=PALETTE[1])
    ax[0, 1].plot(clim.index, clim.tmin, marker="o", label=r"$T_{\min}$", color=PALETTE[0])
    ax[0, 1].set(xlabel="Month", ylabel=r"Temperature ($^\circ$C)", title="Monthly climatology")
    ax[0, 1].legend(frameon=False)
    bins = np.arange(-.5, counts.max() + 1.5)
    ax[1, 0].hist(counts, bins=bins, color=PALETTE[2], edgecolor="white")
    ax[1, 0].set(xlabel="March–June persistent heatwave days per year", ylabel="Years", title="Primary count-outcome distribution")
    row = count_distribution.iloc[0]
    ax[1, 0].text(.98, .95, f"Mean = {row['mean']:.2f}\nVariance = {row['variance']:.2f}\nVariance/mean = {row['variance_to_mean']:.2f}\nZero seasons = {int(row['zero_count'])}/{int(row['n'])}",
                  transform=ax[1, 0].transAxes, ha="right", va="top", fontsize=9,
                  bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#BBBBBB"})
    sns.boxplot(data=hot, x="month", y="tmax", ax=ax[1, 1], color="#9ECAE1")
    ax[1, 1].axhline(36, color=PALETTE[1], ls="--", lw=1.2, label=r"$36\,{}^\circ$C threshold")
    ax[1, 1].set(xlabel="Month", ylabel=r"$T_{\max}$ ($^\circ$C)", title="Hot-season temperature distributions")
    ax[1, 1].legend(frameon=False, fontsize=8)
    for label, panel in zip("ABCD", ax.ravel()):
        panel.text(.01, .99, label, transform=panel.transAxes, va="top", fontweight="bold")
    _save(fig, main_figures, "figure02_descriptive_climatology_and_count_distribution")


def _figure03(raw_corr, anomaly_corr, main_figures):
    selected = ["tmax", "tmin", "rh_mean", "precipitation", "wind_speed_mean", "cloud_cover", "sunshine_duration", "shortwave_radiation", "pressure_mean", "soil_moisture_mean"]
    labels = [r"$T_{\max}$", r"$T_{\min}$", r"Mean $RH$", "Precipitation", "Wind speed", "Cloud cover", "Sunshine duration", "Shortwave radiation", "MSLP", "Soil moisture"]
    fig, ax = plt.subplots(1, 2, figsize=(16.5, 7.5), layout="constrained", gridspec_kw={"wspace": .08})
    for i, (matrix, title) in enumerate([(raw_corr.loc[selected, selected], "Raw March–June Spearman correlations"), (anomaly_corr.loc[selected, selected], "De-seasonalized March–June Spearman correlations")]):
        sns.heatmap(matrix, ax=ax[i], cmap="vlag", vmin=-1, vmax=1, center=0, square=True,
                    annot=True, fmt=".2f", annot_kws={"fontsize": 7.5}, linewidths=.45,
                    linecolor="white", xticklabels=labels, yticklabels=labels if i == 0 else False,
                    cbar=i == 1, cbar_kws={"label": r"Spearman $\rho$", "shrink": 1.0, "ticks": [-1, -.5, 0, .5, 1]})
        ax[i].set_title(title, pad=12); ax[i].tick_params(axis="x", rotation=45, labelsize=8); ax[i].tick_params(axis="y", rotation=0, labelsize=8)
        ax[i].text(-.04, 1.03, "AB"[i], transform=ax[i].transAxes, fontweight="bold", fontsize=12)
        for annotation in ax[i].texts:
            try: value = float(annotation.get_text())
            except ValueError: continue
            annotation.set_color("white" if abs(value) >= .55 else "black")
    _save(fig, main_figures, "figure03_correlation_analysis", tight=False)


def _figure04(series, trends, main_figures):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    _temperature_line(ax[0, 0], series, "annual_mean_tmax", r"Annual mean $T_{\max}$", r"$T_{\max}$ ($^\circ$C)")
    _temperature_line(ax[0, 1], series, "annual_mean_tmin", r"Annual mean $T_{\min}$", r"$T_{\min}$ ($^\circ$C)")
    _temperature_line(ax[1, 0], series, "march_june_mean_tmax", r"March–June mean $T_{\max}$", r"$T_{\max}$ ($^\circ$C)")
    plot = trends[trends.outcome.isin(["annual_mean_tmax", "annual_mean_tmin", "annual_mean_day_night_range", "march_june_mean_tmax", "march_june_mean_tmin", "formal_tmin_minus_tmax_slope_difference"])].copy()
    labels = {"annual_mean_tmax": r"Annual $T_{\max}$", "annual_mean_tmin": r"Annual $T_{\min}$", "annual_mean_day_night_range": "Annual day–night range", "march_june_mean_tmax": r"March–June $T_{\max}$", "march_june_mean_tmin": r"March–June $T_{\min}$", "formal_tmin_minus_tmax_slope_difference": r"Annual $T_{\min}-T_{\max}$ slope contrast"}
    y = np.arange(len(plot))
    ax[1, 1].errorbar(plot.ols_hac_slope_per_decade, y,
                      xerr=[plot.ols_hac_slope_per_decade - plot.ols_hac_ci_lower, plot.ols_hac_ci_upper - plot.ols_hac_slope_per_decade],
                      fmt="o", color=PALETTE[0], ecolor="#555555", capsize=3)
    ax[1, 1].axvline(0, color="black", lw=.8, ls="--")
    ax[1, 1].set(yticks=y, yticklabels=[labels[x] for x in plot.outcome], xlabel=r"Slope ($^\circ$C per decade)", title="Trend estimates and 95% confidence intervals")
    ax[1, 1].invert_yaxis()
    for label, panel in zip("ABCD", ax.ravel()): panel.text(.01, .99, label, transform=panel.transAxes, va="top", fontweight="bold")
    _save(fig, main_figures, "figure04_temperature_trends")


def _figure05(daily_events, events, definition_table, main_figures):
    hot = daily_events[daily_events.month.isin([3, 4, 5, 6])]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for name, color in zip(["operational_36c_1d", "persistent_36c_3d", "relative_90p_3d", "compound_90p_2d"], PALETTE):
        yearly = hot.groupby("year")[name].sum()
        ax[0, 0].plot(yearly.index, yearly, lw=1.1, label=DEFINITION_LABELS[name], color=color)
    ax[0, 0].set(xlabel="Year", ylabel="Heatwave days", title="Definition-dependent annual counts"); ax[0, 0].legend(fontsize=7, frameon=False)
    primary = events[events.definition.eq("persistent_36c_3d")]
    ax[0, 1].hist(primary.duration, bins=np.arange(.5, primary.duration.max() + 1.5), color=PALETTE[2], edgecolor="white")
    ax[0, 1].set(xlabel="Event duration (days)", ylabel="Events", title=r"Primary-definition event duration only")
    monthly = hot.groupby("month").persistent_36c_3d.sum()
    ax[1, 0].bar(monthly.index, monthly.values, color=PALETTE[1]); ax[1, 0].set(xlabel="Month", ylabel="Persistent heatwave days", title="March–June seasonality")
    forest = definition_table[definition_table.reference_period.isin(["absolute", "1981–2010"])].copy()
    y = np.arange(len(forest))
    ax[1, 1].errorbar(forest.irr_per_decade, y, xerr=[forest.irr_per_decade - forest.irr_ci_lower, forest.irr_ci_upper - forest.irr_per_decade], fmt="o", color=PALETTE[0], capsize=3)
    ax[1, 1].axvline(1, color="black", ls="--", lw=.8); ax[1, 1].set_xscale("log")
    ax[1, 1].set(yticks=y, yticklabels=[DEFINITION_LABELS[x] for x in forest.definition], xlabel="IRR per decade (log scale)", title="Count-trend sensitivity with 95% CIs"); ax[1, 1].invert_yaxis()
    for label, panel in zip("ABCD", ax.ravel()): panel.text(.01, .99, label, transform=panel.transAxes, va="top", fontweight="bold")
    _save(fig, main_figures, "figure05_definition_sensitivity")


def _figure06(count_data, selected, selected_name, diagnostics, main_figures):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    order = np.argsort(count_data.year.to_numpy()); years = count_data.year.to_numpy()[order]
    fitted = np.asarray(selected.predict())[order]
    x = sm.add_constant(count_data[["decade"]]).to_numpy()[order]
    covariance = np.asarray(selected.cov_params())[:2, :2]
    eta_se = np.sqrt(np.einsum("ij,jk,ik->i", x, covariance, x))
    ax[0, 0].scatter(count_data.year, count_data.heatwave_days, color=PALETTE[0], label="Observed")
    ax[0, 0].plot(years, fitted, color=PALETTE[1], label=f"Fitted {selected_name.replace('_', ' ')} mean")
    ax[0, 0].fill_between(years, np.exp(np.log(fitted) - 1.96 * eta_se), np.exp(np.log(fitted) + 1.96 * eta_se), color=PALETTE[1], alpha=.18, label="95% mean CI")
    ax[0, 0].set(xlabel="Year", ylabel="Persistent heatwave days", title="Observed and fitted primary counts"); ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 1].scatter(diagnostics.fitted, diagnostics.randomized_quantile_residual, color=PALETTE[0], alpha=.75); ax[0, 1].axhline(0, color="black", lw=.8)
    ax[0, 1].set(xlabel="Fitted mean", ylabel="Randomized quantile residual", title="Residuals versus fitted values")
    residuals = np.sort(diagnostics.randomized_quantile_residual.to_numpy()); theoretical = stats.norm.ppf((np.arange(len(residuals)) + .5) / len(residuals))
    ax[1, 0].scatter(theoretical, residuals, color=PALETTE[0]); limits = [min(theoretical.min(), residuals.min()), max(theoretical.max(), residuals.max())]; ax[1, 0].plot(limits, limits, color="black", ls="--", lw=.8)
    ax[1, 0].set(xlabel="Theoretical normal quantile", ylabel="Observed residual quantile", title="Randomized-residual Q–Q plot")
    cutoff = 4 / len(diagnostics); ax[1, 1].plot(diagnostics.year, diagnostics.cooks_distance, marker="o", lw=1, color=PALETTE[0]); ax[1, 1].axhline(cutoff, color=PALETTE[1], ls="--", lw=1, label=r"$4/n$ screening threshold")
    for _, row in diagnostics[diagnostics.influential].iterrows(): ax[1, 1].annotate(str(int(row.year)), (row.year, row.cooks_distance), xytext=(3, 4), textcoords="offset points", fontsize=8)
    ax[1, 1].set(xlabel="Year", ylabel="Case-deletion parameter distance", title="Influence screening"); ax[1, 1].legend(frameon=False, fontsize=8)
    for label, panel in zip("ABCD", ax.ravel()): panel.text(.01, .99, label, transform=panel.transAxes, va="top", fontweight="bold")
    _save(fig, main_figures, "figure06_count_model_and_diagnostics")


def _figure07(hot_model, full_model, full_standardization, association, binary, main_figures):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    forest = association[(association.model == "antecedent_full") & association.term.isin(PRIMARY_ASSOCIATION_PREDICTORS)].copy()
    pretty = {"rh_mean_lag3_mean": "Prior 3-day mean RH", "precipitation_lag7_sum": "Prior 7-day precipitation", "wind_speed_mean_lag3_mean": "Prior 3-day wind speed", "pressure_mean_lag3_mean": "Prior 3-day pressure"}
    y = np.arange(len(forest)); ax[0, 0].errorbar(forest.adjusted_odds_ratio, y, xerr=[forest.adjusted_odds_ratio - forest.or_ci_lower, forest.or_ci_upper - forest.adjusted_odds_ratio], fmt="o", color=PALETTE[0], capsize=3)
    ax[0, 0].axvline(1, color="black", ls="--", lw=.8); ax[0, 0].set_xscale("log"); ax[0, 0].set(yticks=y, yticklabels=[pretty[x] for x in forest.term], xlabel="Adjusted odds ratio per 1 SD (log scale)", title="Adjusted antecedent associations"); ax[0, 0].invert_yaxis()
    terms = full_model._analysis_terms; params = pd.Series(np.asarray(full_model.params), index=terms)
    for predictor, color in zip(["rh_mean_lag3_mean", "precipitation_lag7_sum"], PALETTE[:2]):
        z = np.linspace(-2, 2, 100); design = pd.DataFrame(0.0, index=np.arange(len(z)), columns=terms); design["const"] = 1; design[predictor] = z
        representative_day = 110
        for harmonic in (1, 2, 3):
            design[f"sin_doy_{harmonic}"] = np.sin(2 * np.pi * harmonic * representative_day / 365.25)
            design[f"cos_doy_{harmonic}"] = np.cos(2 * np.pi * harmonic * representative_day / 365.25)
        probability = expit(design.to_numpy() @ params.to_numpy())
        ax[0, 1].plot(z, probability, color=color, label=pretty[predictor])
    ax[0, 1].set(xlabel="Antecedent predictor (SD from hot-season mean)", ylabel="Adjusted probability", title="Model-implied marginal probabilities"); ax[0, 1].legend(frameon=False, fontsize=8)
    held = binary[binary.validation_scope.eq("held_out_season")]
    for model, color in zip(["seasonal_trend_base", "antecedent_full"], PALETTE[:2]):
        part = held[held.model.eq(model)]; ax[1, 0].plot(part.validation_year, part.brier_score, marker="o", lw=1, color=color, label=model.replace("_", " "))
    ax[1, 0].set(xlabel="Held-out hot season", ylabel="Brier score (lower is better)", title="Chronological blocked validation"); ax[1, 0].legend(frameon=False, fontsize=8)
    pooled = binary[binary.validation_scope.eq("pooled_strictly_out_of_sample")].set_index("model")
    metrics = ["roc_auc", "precision_recall_auc", "brier_score"]; xpos = np.arange(len(metrics)); width = .34
    for j, model in enumerate(["seasonal_trend_base", "antecedent_full"]):
        ax[1, 1].bar(xpos + (j - .5) * width, pooled.loc[model, metrics], width, label=model.replace("_", " "), color=PALETTE[j])
    ax[1, 1].set(xticks=xpos, xticklabels=["ROC-AUC", "PR-AUC", "Brier"], ylim=(0, 1), title="Pooled strictly out-of-sample performance"); ax[1, 1].legend(frameon=False, fontsize=8)
    for label, panel in zip("ABCD", ax.ravel()): panel.text(.01, .99, label, transform=panel.transAxes, va="top", fontweight="bold")
    _save(fig, main_figures, "figure07_adjusted_associations")


def _supplement_figures(df, performance, predictions, supplement_figures):
    hot = df[df.month.isin([3, 4, 5, 6])].copy()
    hot = hot[["rh_mean", "precipitation", "wind_speed_mean", "pressure_mean", "persistent_36c_3d"]].dropna()
    sample = hot.sample(min(1800, len(hot)), random_state=20260901)
    sample["Persistent day"] = sample.persistent_36c_3d.map({False: "No", True: "Yes"})
    grid = sns.pairplot(sample, vars=["rh_mean", "precipitation", "wind_speed_mean", "pressure_mean"], hue="Persistent day", corner=True, plot_kws={"s": 10, "alpha": .45}, diag_kws={"common_norm": False})
    grid.fig.savefig(supplement_figures / "figureS01_selected_pairplots.png", dpi=300, bbox_inches="tight"); grid.fig.savefig(supplement_figures / "figureS01_selected_pairplots.pdf", bbox_inches="tight"); plt.close(grid.fig)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    sns.boxplot(data=performance, x="model", y="rmse", ax=ax[0, 0], color="#9ECAE1"); ax[0, 0].tick_params(axis="x", rotation=30); ax[0, 0].set(title="Rolling-origin RMSE", xlabel="", ylabel=r"RMSE ($^\circ$C)")
    sns.scatterplot(data=predictions, x="observed", y="predicted", hue="model", ax=ax[0, 1], s=28); limits=[min(predictions.observed.min(),predictions.predicted.min()),max(predictions.observed.max(),predictions.predicted.max())]; ax[0, 1].plot(limits,limits,color="black",ls="--",lw=.8); ax[0, 1].set(title="Strictly out-of-sample predictions", xlabel=r"Observed $T_{\max}$", ylabel=r"Predicted $T_{\max}$"); ax[0, 1].legend(fontsize=6)
    calibration=performance.groupby("model")[["coverage_80","coverage_95"]].mean(); calibration.plot.bar(ax=ax[1,0],color=PALETTE[:2]); ax[1,0].axhline(.8,color=PALETTE[0],ls="--",lw=.8); ax[1,0].axhline(.95,color=PALETTE[1],ls="--",lw=.8); ax[1,0].set(title="Empirical interval coverage",xlabel="",ylabel="Coverage",ylim=(0,1.05)); ax[1,0].tick_params(axis="x",rotation=30); ax[1,0].legend(fontsize=7)
    widths=performance.groupby("model")[["width_80","width_95"]].mean(); widths.plot.bar(ax=ax[1,1],color=PALETTE[2:4]); ax[1,1].set(title="Mean prediction-interval width",xlabel="",ylabel=r"Width ($^\circ$C)"); ax[1,1].tick_params(axis="x",rotation=30); ax[1,1].legend(fontsize=7)
    for label,panel in zip("ABCD",ax.ravel()): panel.text(.01,.99,label,transform=panel.transAxes,va="top",fontweight="bold")
    _save(fig,supplement_figures,"figureS02_forecast_validation")


def generate_article_outputs(
    *, df, completeness, raw_corr, anomaly_corr, daily_events, events,
    temperature_series, temperature_trends, temperature_sensitivity,
    count_distribution, count_comparison, count_estimates, count_data,
    count_model, count_model_name, count_diagnostics, influence_sensitivity,
    hot_model, association_model, association_standardization,
    association_estimates_frame, association_sensitivity_frame,
    binary_validation, forecast_performance, forecast_predictions, output_root: Path,
) -> dict[str, pd.DataFrame]:
    tables_root = output_root / "tables"; main_figures = output_root / "figures" / "main"; supplement_figures = output_root / "figures" / "supplement"
    main_figures.mkdir(parents=True, exist_ok=True); supplement_figures.mkdir(parents=True, exist_ok=True)
    descriptive = _long_descriptives(df)
    count_long = count_distribution.copy(); count_long.insert(0, "stratum", "annual_persistent_count_distribution")
    table01 = pd.concat([descriptive, count_long], ignore_index=True, sort=False)
    table02 = correlation_screening_table(df, raw_corr, anomaly_corr)
    table03 = definition_summary(df, daily_events, events)
    table04 = pd.concat([temperature_trends.assign(analysis="primary"), temperature_sensitivity.assign(analysis="hot_season_endpoint_sensitivity")], ignore_index=True, sort=False)
    table07 = association_estimates_frame.copy()
    hot = df[df.month.isin([3,4,5,6])]
    sd_map = hot[PRIMARY_ASSOCIATION_PREDICTORS].std(ddof=0).to_dict()
    table07["one_sd_original_units"] = table07.term.map(sd_map)
    table08 = pd.concat([binary_validation.assign(section="blocked_validation"), association_sensitivity_frame.assign(section="association_sensitivity")], ignore_index=True, sort=False)
    outputs = {
        "main_table01_data_and_descriptive_statistics": table01,
        "main_table02_correlations_and_collinearity": table02,
        "main_table03_definition_sensitivity": table03,
        "main_table04_temperature_trends": table04,
        "main_table05_poisson_nb_comparison": count_comparison,
        "main_table06_primary_count_model": pd.concat([count_distribution.assign(section="distribution"), count_estimates.assign(section="model")], ignore_index=True, sort=False),
        "main_table07_adjusted_associations": table07,
        "main_table08_blocked_validation": table08,
        "supplement_tableS01_count_influence_sensitivity": influence_sensitivity,
        "supplement_tableS02_forecast_validation": forecast_performance,
    }
    for name, frame in outputs.items(): _write(frame, tables_root, name)
    _study_area_figure(df, completeness, main_figures)
    _figure02(df, count_distribution, main_figures)
    _figure03(raw_corr, anomaly_corr, main_figures)
    _figure04(temperature_series, temperature_trends, main_figures)
    _figure05(daily_events, events, table03, main_figures)
    _figure06(count_data, count_model, count_model_name, count_diagnostics, main_figures)
    _figure07(hot_model, association_model, association_standardization, association_estimates_frame, binary_validation, main_figures)
    _supplement_figures(df, forecast_performance, forecast_predictions, supplement_figures)
    return outputs
