"""Publication-quality, colorblind-safe figures generated only from rebuilt outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PALETTE=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9"]


def _save(fig, out: Path, stem: str):
    fig.tight_layout(); fig.savefig(out/f"{stem}.png",dpi=300,bbox_inches="tight"); fig.savefig(out/f"{stem}.pdf",bbox_inches="tight"); plt.close(fig)


def generate_figures(df, completeness, raw_corr, anomaly_corr, daily_events, events, annual_metrics,
                     count_diag, count_estimates, association_estimates, binary_validation,
                     forecast_predictions, forecast_performance, future_counts, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid",context="paper",font_scale=1.05)
    # 1: study/data/workflow (avoids adding an undocumented cartographic dependency)
    fig,ax=plt.subplots(1,2,figsize=(11,4)); ax[0].plot(df.date,df.tmax,color=PALETTE[0],lw=.25); ax[0].set(title=r"Dhaka daily $T_{\max}$",ylabel=r"$T_{\max}$ ($^\circ$C)",xlabel="Date")
    steps=["Audit","Describe","Define events","Infer","Diagnose","Validate","Quantify uncertainty"]
    ax[1].axis("off");
    for i,s in enumerate(steps): ax[1].text(.5,1-i/7.2,s,ha="center",va="center",bbox=dict(boxstyle="round",fc="#E6F2F8",ec=PALETTE[0]))
    ax[1].set_title("Methods-first workflow"); _save(fig,out,"figure01_study_data_workflow")
    # 2
    fig,ax=plt.subplots(1,3,figsize=(12,3.7)); ax[0].bar(completeness.year,completeness.observed_days,color=np.where(completeness.calendar_complete,PALETTE[2],PALETTE[1])); ax[0].set(title="Calendar completeness",ylabel="Observed days")
    sns.histplot(df.tmax,bins=35,kde=True,ax=ax[1],color=PALETTE[0]); ax[1].set(title=r"$T_{\max}$ distribution",xlabel=r"$T_{\max}$ ($^\circ$C)")
    clim=df.groupby("month")[["tmax","tmin"]].mean(); ax[2].plot(clim.index,clim.tmax,label=r"$T_{\max}$",color=PALETTE[1]); ax[2].plot(clim.index,clim.tmin,label=r"$T_{\min}$",color=PALETTE[0]); ax[2].legend(); ax[2].set(title="Monthly climatology",xlabel="Month",ylabel=r"Temperature ($^\circ$C)"); _save(fig,out,"figure02_completeness_distributions_climatology")
    # 3
    fig,ax=plt.subplots(1,3,figsize=(13,4)); hot=df[df.month.isin([3,4,5,6])]; sns.scatterplot(data=hot.sample(min(2500,len(hot)),random_state=20260901),x="rh_mean",y="tmax",alpha=.25,s=10,ax=ax[0],color=PALETTE[0]); ax[0].set_title(r"Hot-season $T_{\max}$ and humidity"); ax[0].set(xlabel=r"Mean relative humidity, $RH$ (%)",ylabel=r"$T_{\max}$ ($^\circ$C)")
    sns.heatmap(raw_corr,ax=ax[1],cmap="vlag",vmin=-1,vmax=1,cbar=False,xticklabels=False,yticklabels=False); ax[1].set_title("Raw Spearman correlations")
    sns.heatmap(anomaly_corr,ax=ax[2],cmap="vlag",vmin=-1,vmax=1,cbar=True,xticklabels=False,yticklabels=False); ax[2].set_title("De-seasonalized correlations"); _save(fig,out,"figure03_relationships_correlations")
    # 4
    fig,ax=plt.subplots(1,3,figsize=(12,3.7)); am=annual_metrics[annual_metrics.definition.isin(["operational_36c_1d","persistent_36c_3d","relative_90p_3d"])]
    for name,g in am.groupby("definition"): ax[0].plot(g.year,g.heatwave_days,label=name,lw=1); ax[0].legend(fontsize=6); ax[0].set(title="Heatwave-day frequency",ylabel="Days")
    sns.histplot(events.duration,bins=range(1,int(events.duration.max())+2),ax=ax[1],color=PALETTE[2]); ax[1].set_title("Event-duration distribution")
    month=daily_events.groupby("month")[[c for c in daily_events if c in ["operational_36c_1d","persistent_36c_3d"]]].sum(); month.plot.bar(ax=ax[2],color=PALETTE[:2],legend=False); ax[2].set_title("Monthly seasonality"); _save(fig,out,"figure04_heatwave_characteristics")
    # 5 count fit
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(count_diag.year,count_diag.observed,"o",color=PALETTE[0],label="Observed"); ax.plot(count_diag.year,count_diag.fitted,color=PALETTE[1],label="Fitted"); ax.legend(); ax.set(title="Persistent heatwave-day count trend",xlabel="Year",ylabel="March–June days"); _save(fig,out,"figure05_selected_count_trend")
    # 6 diagnostics
    fig,ax=plt.subplots(2,2,figsize=(9,7)); ax=ax.ravel(); ax[0].scatter(count_diag.fitted,count_diag.pearson_residual,c=PALETTE[0]); ax[0].axhline(0,color="black",lw=.8); ax[0].set(title="Residuals vs fitted",xlabel="Fitted",ylabel="Pearson residual")
    stats.probplot(count_diag.randomized_quantile_residual,dist="norm",plot=ax[1]); ax[1].set_title("Randomized-residual Q–Q")
    ax[2].stem(count_diag.year,count_diag.cooks_distance,linefmt=PALETTE[1],markerfmt="o",basefmt=" "); ax[2].set_title("Cook's distance")
    ax[3].scatter(count_diag.observed,count_diag.fitted,c=PALETTE[2]); lim=max(count_diag.observed.max(),count_diag.fitted.max()); ax[3].plot([0,lim],[0,lim],"k--"); ax[3].set(title="Observed vs fitted",xlabel="Observed",ylabel="Fitted"); _save(fig,out,"figure06_count_diagnostics")
    # 7 OR plot
    a=association_estimates[(association_estimates.model=="antecedent_full") & association_estimates.term.str.contains("lag")].copy(); fig,ax=plt.subplots(figsize=(8,4)); y=np.arange(len(a)); ax.errorbar(a.adjusted_odds_ratio,y,xerr=[a.adjusted_odds_ratio-a.or_ci_lower,a.or_ci_upper-a.adjusted_odds_ratio],fmt="o",color=PALETTE[0]); ax.axvline(1,color="black",ls="--"); labels={"rh_mean_lag3_mean":r"$\overline{RH}_{t-1:t-3}$","precipitation_lag7_sum":r"$\sum_{k=1}^{7} P_{t-k}$","wind_speed_mean_lag3_mean":r"$\overline{WS}_{t-1:t-3}$","pressure_mean_lag3_mean":r"$\overline{MSLP}_{t-1:t-3}$"}; ax.set_yticks(y,[labels.get(term,term) for term in a.term]); ax.set(title=r"Adjusted antecedent associations (per $1\,\mathrm{SD}$)",xlabel=r"Odds ratio, $\exp(\beta)$ (95% CI)"); _save(fig,out,"figure07_adjusted_associations")
    # 8 validation
    fig,ax=plt.subplots(1,2,figsize=(11,4)); bp=binary_validation.groupby("model")[["brier_score","precision_recall_auc"]].mean(); bp.plot.bar(ax=ax[0],color=PALETTE[:2]); ax[0].set_title("Binary blocked validation")
    fp=forecast_performance.groupby("model")["rmse"].mean().sort_values(); fp.plot.bar(ax=ax[1],color=PALETTE[0]); ax[1].set(title=r"Rolling-origin monthly $T_{\max}$",ylabel=r"$\mathrm{RMSE}$ ($^\circ$C)"); _save(fig,out,"figure08_validation_against_baselines")
    # 9 forecast validation predictions
    fig,ax=plt.subplots(figsize=(10,4)); selected=forecast_performance.groupby("model").rmse.mean().idxmin(); p=forecast_predictions[forecast_predictions.model==selected]; ax.plot(p.date,p.observed,"o",label="Observed",color=PALETTE[0]); ax.plot(p.date,p.predicted,"-",label=selected,color=PALETTE[1]); ax.fill_between(p.date,p.lower_95,p.upper_95,color=PALETTE[1],alpha=.15,label="95% interval"); ax.legend(); ax.set(title="Out-of-sample hot-season forecasts",ylabel=r"Monthly mean $T_{\max}$ ($^\circ$C)"); _save(fig,out,"figure09_rolling_forecast_intervals")
    # 10 scenario uncertainty
    fig,ax=plt.subplots(figsize=(8,4)); years=future_counts.year.astype(int).to_numpy(); ax.plot(years,future_counts.median_heatwave_days,"o-",color=PALETTE[1]); ax.fill_between(years,future_counts.p10,future_counts.p90,alpha=.3,color=PALETTE[1],label=r"$10^{\mathrm{th}}$--$90^{\mathrm{th}}$ percentiles"); ax.fill_between(years,future_counts.p2_5,future_counts.p97_5,alpha=.15,color=PALETTE[1],label="95% interval"); ax.legend(); ax.set(title="Trend-based count scenarios (not validated long-range forecasts)",xlabel="Year",ylabel="Persistent heatwave days"); ax.set_xticks(years); ax.set_xlim(years.min()-.25,years.max()+.25); _save(fig,out,"figure10_future_count_scenarios")
