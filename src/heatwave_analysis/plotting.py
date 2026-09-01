"""Publication-quality, colorblind-safe figures generated only from rebuilt outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapefile
from scipy import stats

PALETTE=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9"]


def _save(fig, out: Path, stem: str, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(out/f"{stem}.png",dpi=300,bbox_inches="tight"); fig.savefig(out/f"{stem}.pdf",bbox_inches="tight"); plt.close(fig)


def _polygon_parts(shape):
    """Yield individual polygon rings from a pyshp shape."""
    points=np.asarray(shape.points)
    stops=list(shape.parts)+[len(points)]
    for start,end in zip(stops[:-1],stops[1:]):
        if end-start >= 3:
            yield points[start:end]


def _draw_shape(ax,shape,facecolor="none",edgecolor="black",linewidth=.5,alpha=1.0,zorder=1):
    for ring in _polygon_parts(shape):
        ax.fill(ring[:,0],ring[:,1],facecolor=facecolor,edgecolor=edgecolor,
                linewidth=linewidth,alpha=alpha,zorder=zorder)


def _study_area_figure(df,completeness,out):
    """Bangladesh/Dhaka geography and record coverage without assuming station coordinates."""
    shape_dir=Path(__file__).resolve().parents[2]/"data"/"shapefiles"
    adm2=shapefile.Reader(str(shape_dir/"bgd_admbnda_adm2_bbs_20201113"))
    adm3=shapefile.Reader(str(shape_dir/"bgd_admbnda_adm3_bbs_20201113"))
    districts=list(zip(adm2.shapes(),adm2.records()))
    dhaka_shape=next(shape for shape,record in districts if record.as_dict()["ADM2_EN"]=="Dhaka")
    dhaka_upazilas=[shape for shape,record in zip(adm3.shapes(),adm3.records()) if record.as_dict()["ADM2_EN"]=="Dhaka"]

    fig,ax=plt.subplots(1,3,figsize=(14,4.4),gridspec_kw={"width_ratios":[.9,1,1.45]})
    for shape,_ in districts:
        _draw_shape(ax[0],shape,facecolor="#E8E8E8",edgecolor="white",linewidth=.25)
    _draw_shape(ax[0],dhaka_shape,facecolor=PALETTE[1],edgecolor="#7A2E00",linewidth=1.0,zorder=3)
    ax[0].set(xlim=(87.8,92.9),ylim=(20.4,26.8),title="Bangladesh and Dhaka District",xlabel="Longitude (°E)",ylabel="Latitude (°N)")
    ax[0].set_aspect("equal"); ax[0].grid(False); ax[0].annotate("Dhaka",xy=(90.35,23.75),xytext=(89.1,24.8),arrowprops={"arrowstyle":"->","color":"#7A2E00"},color="#7A2E00",fontweight="bold")

    for shape in dhaka_upazilas:
        _draw_shape(ax[1],shape,facecolor="#F7D7C4",edgecolor="white",linewidth=.7)
    _draw_shape(ax[1],dhaka_shape,facecolor="none",edgecolor="#7A2E00",linewidth=1.5,zorder=3)
    xmin,ymin,xmax,ymax=dhaka_shape.bbox; padx=(xmax-xmin)*.08; pady=(ymax-ymin)*.08
    ax[1].set(xlim=(xmin-padx,xmax+padx),ylim=(ymin-pady,ymax+pady),title="Dhaka District boundary",xlabel="Longitude (°E)",ylabel="Latitude (°N)")
    ax[1].set_aspect("equal"); ax[1].grid(False)
    ax[1].scatter(90.4125,23.8103,s=42,color=PALETTE[0],edgecolor="white",linewidth=.7,zorder=5)
    ax[1].annotate("Dhaka city reference",xy=(90.4125,23.8103),xytext=(90.05,24.05),arrowprops={"arrowstyle":"->","color":PALETTE[0]},color=PALETTE[0],fontsize=8)
    ax[1].text(.02,.02,"Reference point only; exact station\ncoordinates are not documented.",transform=ax[1].transAxes,fontsize=7,va="bottom",bbox={"boxstyle":"round","facecolor":"white","alpha":.85,"edgecolor":"#BBBBBB"})

    colors=np.where(completeness.calendar_complete,PALETTE[2],PALETTE[1])
    ax[2].bar(completeness.year,completeness.observed_days,color=colors,width=.82)
    ax[2].plot(completeness.year,completeness.expected_days,color="#444444",linewidth=.7,label="Expected calendar days")
    ax[2].set(title="Daily meteorological record coverage",xlabel="Year",ylabel="Observed days",ylim=(0,390),xlim=(1970,2026))
    ax[2].legend(loc="lower left",fontsize=7)
    ax[2].annotate("2024 partial year\nthrough 18 November",xy=(2024,323),xytext=(2006,245),arrowprops={"arrowstyle":"->","color":PALETTE[1]},color=PALETTE[1],fontsize=8)
    ax[2].text(.02,.93,f"{len(df):,} consecutive daily observations\n1972-01-01 to 2024-11-18; no missing dates\nMarch–June 2024 complete (122 days)",transform=ax[2].transAxes,va="top",fontsize=8,bbox={"boxstyle":"round","facecolor":"white","alpha":.9,"edgecolor":"#BBBBBB"})
    for label,a in zip(["A","B","C"],ax):
        a.text(.01,.99,label,transform=a.transAxes,ha="left",va="top",fontweight="bold",fontsize=11)
    _save(fig,out,"figure01_study_area_data_coverage")


def generate_figures(df, completeness, raw_corr, anomaly_corr, daily_events, events, annual_metrics,
                     count_diag, count_estimates, association_estimates, binary_validation,
                     forecast_predictions, forecast_performance, future_counts, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid",context="paper",font_scale=1.05)
    # 1: study area and verified temporal coverage
    _study_area_figure(df,completeness,out)
    # 2: field-level quality control, distribution, and climatology
    fig,ax=plt.subplots(1,3,figsize=(12,3.7))
    affected=[name for name in df.columns if df[name].isna().any()]
    missing_by_year=df.groupby("year")[affected].agg(lambda x:x.isna().sum()).T
    aliases={"precipitation":"Precipitation","wind_gust_max":"Wind gust, max","wind_gust_min":"Wind gust, min","wind_gust_mean":"Wind gust, mean","shortwave_radiation":"Shortwave radiation","longwave_radiation":"Longwave radiation","uv_radiation":"UV radiation","direct_shortwave_radiation":"Direct shortwave","evapotranspiration":"Evapotranspiration"}
    tick_years=[str(year) if year in (1972,1979,2024) or (year%10==0 and year!=1980) else "" for year in missing_by_year.columns]
    sns.heatmap(missing_by_year,ax=ax[0],cmap=["#F2F2F2",PALETTE[1]],vmin=0,vmax=1,cbar=False,xticklabels=tick_years,yticklabels=[aliases.get(name,name) for name in affected])
    ax[0].set(title="Field-level missingness\n9 cells, all on 1979-01-01",xlabel="Year",ylabel="Variable")
    ax[0].tick_params(axis="x",rotation=45,labelsize=7); ax[0].tick_params(axis="y",rotation=0,labelsize=7)
    sns.histplot(df.tmax,bins=35,kde=True,ax=ax[1],color=PALETTE[0]); ax[1].set(title=r"$T_{\max}$ distribution",xlabel=r"$T_{\max}$ ($^\circ$C)")
    clim=df.groupby("month")[["tmax","tmin"]].mean(); ax[2].plot(clim.index,clim.tmax,label=r"$T_{\max}$",color=PALETTE[1]); ax[2].plot(clim.index,clim.tmin,label=r"$T_{\min}$",color=PALETTE[0]); ax[2].legend(); ax[2].set(title="Monthly climatology",xlabel="Month",ylabel=r"Temperature ($^\circ$C)"); _save(fig,out,"figure02_completeness_distributions_climatology")
    # 3: readable selected-variable correlation matrices
    selected=["tmax","tmin","rh_mean","precipitation","wind_speed_mean","cloud_cover","sunshine_duration","shortwave_radiation","pressure_mean","soil_moisture_mean"]
    labels=[r"$T_{\max}$",r"$T_{\min}$",r"Mean $RH$","Precipitation","Wind speed","Cloud cover","Sunshine duration","Shortwave radiation","MSLP","Soil moisture"]
    matrices=[raw_corr.loc[selected,selected],anomaly_corr.loc[selected,selected]]
    titles=[r"Raw March–June Spearman correlations",r"De-seasonalized March–June Spearman correlations"]
    fig,ax=plt.subplots(1,2,figsize=(18,7.5),layout="constrained",gridspec_kw={"wspace":.28})
    for i,(matrix,title) in enumerate(zip(matrices,titles)):
        hm=sns.heatmap(matrix,ax=ax[i],cmap="vlag",vmin=-1,vmax=1,center=0,square=True,
                       annot=True,fmt=".2f",annot_kws={"fontsize":7.5},linewidths=.45,linecolor="white",
                       xticklabels=labels,yticklabels=labels,cbar=i==1,
                       cbar_kws={"label":r"Spearman $\rho$","shrink":.78,"ticks":[-1,-.5,0,.5,1]})
        ax[i].set_title(title,pad=12); ax[i].tick_params(axis="x",rotation=45,labelsize=8); ax[i].tick_params(axis="y",rotation=0,labelsize=8)
        ax[i].set_xlabel(""); ax[i].set_ylabel(""); ax[i].text(-.04,1.03,chr(65+i),transform=ax[i].transAxes,fontweight="bold",fontsize=12)
        for annotation in ax[i].texts:
            try: value=float(annotation.get_text())
            except ValueError: continue
            annotation.set_color("white" if abs(value)>=.55 else "black")
    fig.suptitle("Hot-season meteorological correlation structure, 1972–2024",fontsize=15,fontweight="bold")
    _save(fig,out,"figure03_spearman_correlation_matrices",tight=False)
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
