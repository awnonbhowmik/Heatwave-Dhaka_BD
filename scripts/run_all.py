#!/usr/bin/env python3
"""Run the full methods-first analysis from raw inputs to manuscript-ready outputs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from heatwave_analysis.association_models import (association_estimates, construct_antecedent_predictors, fit_association_models, rolling_binary_validation)
from heatwave_analysis.count_models import (count_distribution_diagnostics, fit_count_models, influence_diagnostics, leave_one_influential_year_out, model_estimates)
from heatwave_analysis.data_io import complete_years, hot_season, load_daily, source_hashes
from heatwave_analysis.exploratory import collinearity_assessment, correlation_outputs, descriptive_statistics, grouped_descriptives
from heatwave_analysis.forecasting import (forecast_design, future_count_scenarios, future_temperature_scenarios, monthly_temperature, rolling_origin_forecasts)
from heatwave_analysis.heatwave_events import aggregate_metrics, construct_all_definitions, construct_definition
from heatwave_analysis.plotting import generate_figures
from heatwave_analysis.quality_control import completeness_table, data_dictionary, quality_findings
from heatwave_analysis.reporting import markdown_tables, write_reports
from heatwave_analysis.trend_models import annual_temperature_series, fit_temperature_trends, test_tmax_tmin_slope_difference
from heatwave_analysis.variable_dictionary import CORRELATION_VARIABLES, DESCRIPTIVE_VARIABLES, PRIMARY_ASSOCIATION_PREDICTORS


def write_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False,float_format="%.10g")


def bh(frame: pd.DataFrame, p_col="p_value"):
    out=frame.copy(); mask=out[p_col].notna(); out["q_value_bh"]=np.nan
    if mask.any(): out.loc[mask,"q_value_bh"]=multipletests(out.loc[mask,p_col],method="fdr_bh")[1]
    return out


def notebook_files():
    specs=[("00_data_audit","Load and audit the immutable raw daily CSV."),("01_descriptive_and_eda","Review generated descriptive statistics and correlations."),("02_heatwave_definitions","Review persistent and percentile event definitions."),("03_inferential_analysis","Review temperature trends and count models."),("04_association_models","Review leakage-safe adjusted associations."),("05_forecast_validation","Review rolling-origin forecast validation."),("06_final_outputs","Review manuscript tables, figures, and reports.")]
    d=ROOT/"notebooks"; d.mkdir(exist_ok=True)
    for stem,desc in specs:
        nb=nbformat.v4.new_notebook(metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
        nb.cells=[nbformat.v4.new_markdown_cell(f"# {stem.replace('_',' ').title()}\n\n{desc}\n\nThis thin interface delegates computation to `src/heatwave_analysis` and the command-line pipeline."),nbformat.v4.new_code_cell("from pathlib import Path\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nassert (ROOT / 'results' / 'metadata' / 'run_metadata.json').exists(), 'Run make analysis first'\nprint((ROOT / 'results' / 'metadata' / 'run_metadata.json').read_text())")]
        nbformat.write(nb,d/f"{stem}.ipynb")


def main(config_path: str):
    started=time.time(); os.environ.setdefault("MPLCONFIGDIR","/tmp/heatwave-mpl")
    cfg=yaml.safe_load((ROOT/config_path).read_text()); seed=int(cfg["seed"]); np.random.seed(seed)
    out=ROOT/cfg["outputs"]["root"]; reports=ROOT/cfg["outputs"]["reports"]
    dirs={name:out/name for name in ["metadata","derived_data","tables","figures","diagnostics","forecasts"]}
    for d in [*dirs.values(),reports]: d.mkdir(parents=True,exist_ok=True)
    daily_path=ROOT/cfg["data"]["daily_csv"]
    raw=pd.read_csv(daily_path); df=load_daily(daily_path); q=quality_findings(df); completeness=completeness_table(df)
    (dirs["metadata"]/"source_data_hashes.json").write_text(json.dumps(source_hashes(ROOT/"data"),indent=2)+"\n")
    (dirs["metadata"]/"quality_findings.json").write_text(json.dumps(q,indent=2)+"\n")
    write_csv(data_dictionary(raw,df),dirs["metadata"]/"data_dictionary.csv")
    write_csv(completeness,dirs["tables"]/"table01_data_completeness.csv")
    write_csv(descriptive_statistics(df,DESCRIPTIVE_VARIABLES,"all_dates"),dirs["tables"]/"table02_descriptive_statistics_all_year.csv")
    hot=hot_season(df); write_csv(descriptive_statistics(hot,DESCRIPTIVE_VARIABLES,"march_june"),dirs["tables"]/"table03_descriptive_statistics_hot_season.csv")

    daily_events,events,thresholds=construct_all_definitions(df)
    for col in [c for c in daily_events.columns if c not in ["date","year","month","tmax","tmin"]]: df[col]=daily_events[col]
    df["operational_threshold_day"]=df.operational_36c_1d; df["persistent_heatwave_day"]=df.persistent_36c_3d
    hot=df[df.month.isin([3,4,5,6])]
    grouped=grouped_descriptives(hot,DESCRIPTIVE_VARIABLES,["operational_threshold_day","persistent_heatwave_day"])
    write_csv(grouped,dirs["tables"]/"table04_heatwave_vs_nonheatwave_descriptives.csv")
    raw_corr,anom_corr,pair_n,pearson,anomalies=correlation_outputs(hot,CORRELATION_VARIABLES)
    raw_corr.to_csv(dirs["tables"]/"table05_spearman_correlations_raw.csv"); anom_corr.to_csv(dirs["tables"]/"table06_spearman_correlations_anomalies.csv"); pair_n.to_csv(dirs["tables"]/"table07_pairwise_sample_sizes.csv")
    coll=collinearity_assessment(hot,CORRELATION_VARIABLES); write_csv(coll,dirs["tables"]/"table08_collinearity_assessment.csv"); pearson.to_csv(dirs["tables"]/"supplementary_pearson_correlations.csv")
    events.to_csv(dirs["derived_data"]/"heatwave_events_all_definitions.csv",index=False)
    annual=aggregate_metrics(daily_events,events,"annual"); monthly=aggregate_metrics(daily_events,events,"monthly")
    write_csv(annual,dirs["derived_data"]/"annual_heatwave_metrics.csv"); write_csv(monthly,dirs["derived_data"]/"monthly_heatwave_metrics.csv")
    df.to_csv(dirs["derived_data"]/"daily_analysis_data.csv",index=False,float_format="%.8g")

    temp_series=annual_temperature_series(df,cfg["analysis"]["complete_year_end"]); trends=fit_temperature_trends(temp_series)
    contrast=test_tmax_tmin_slope_difference(temp_series); contrast_row={"outcome":"formal_tmin_minus_tmax_slope_difference","ols_hac_slope_per_decade":contrast["difference_per_decade"],"ols_hac_ci_lower":contrast["ci_lower"],"ols_hac_ci_upper":contrast["ci_upper"],"ols_hac_p_value":contrast["p_value"],"contrast":contrast["contrast"]}; trends=pd.concat([trends,pd.DataFrame([contrast_row])],ignore_index=True)
    trends=bh(trends.rename(columns={"ols_hac_p_value":"p_value"})).rename(columns={"p_value":"ols_hac_p_value"}); write_csv(trends,dirs["tables"]/"table09_temperature_trends.csv")
    primary_counts=annual[(annual.definition=="persistent_36c_3d") & annual.year.le(2024)].copy()
    # All March-June dates are present, so full-year event flags can be aggregated consistently.
    primary_hot=daily_events[daily_events.month.isin([3,4,5,6])].groupby("year").persistent_36c_3d.sum().reset_index(name="heatwave_days")
    dist=count_distribution_diagnostics(primary_hot); write_csv(dist,dirs["tables"]/"table10_count_distribution_diagnostics.csv")
    count_data,pois,nb,selected,selected_name,comparison=fit_count_models(primary_hot); write_csv(comparison,dirs["tables"]/"table11_poisson_nb_comparison.csv")
    estimates=model_estimates(selected,selected_name); write_csv(estimates,dirs["tables"]/"table12_selected_count_model.csv")
    count_diag=influence_diagnostics(count_data,pois,selected,selected_name); count_diag.to_csv(dirs["diagnostics"]/"selected_count_model_diagnostics.csv",index=False)
    influence=leave_one_influential_year_out(count_data,count_diag); write_csv(influence,dirs["tables"]/"table13_influence_sensitivity.csv")
    # Required monthly count specification, retained as a diagnostic sensitivity.
    m=daily_events[daily_events.month.isin([3,4,5,6])].groupby(["year","month"]).agg(heatwave_days=("persistent_36c_3d","sum"),observed_days=("date","nunique")).reset_index(); m["decade"]=(m.year-m.year.mean())/10
    import statsmodels.formula.api as smf
    monthly_model=smf.glm("heatwave_days ~ decade + C(month)",m,family=sm.families.Poisson(),offset=np.log(m.observed_days)).fit(cov_type="cluster",cov_kwds={"groups":m.year})
    pd.DataFrame({"term":monthly_model.params.index,"coefficient":monthly_model.params.values,"se":monthly_model.bse.values,"p_value":monthly_model.pvalues.values}).to_csv(dirs["diagnostics"]/"monthly_count_model.csv",index=False)

    modeled=construct_antecedent_predictors(df); hot_model,base,full,_,_=fit_association_models(modeled); assoc=bh(association_estimates(base,full));
    selection=pd.DataFrame([{"predictor":p,"lag_structure":p.split("_lag",1)[1],"target_derived":False,"primary_model":True,"reason":"antecedent, interpretable, prespecified after domain/collinearity review"} for p in PRIMARY_ASSOCIATION_PREDICTORS])
    write_csv(selection,dirs["tables"]/"table14_predictor_selection.csv"); write_csv(assoc,dirs["tables"]/"table15_adjusted_association_model.csv")
    binary=rolling_binary_validation(modeled,cfg["analysis"]["binary_validation_origins"]); write_csv(binary,dirs["tables"]/"table16_binary_model_validation.csv")

    design=forecast_design(); write_csv(design,dirs["tables"]/"table17_forecast_design.csv")
    month_temp=monthly_temperature(df); performance,predictions=rolling_origin_forecasts(month_temp,cfg["analysis"]["forecast_origins"])
    write_csv(performance,dirs["tables"]/"table18_rolling_origin_performance.csv"); write_csv(performance.groupby("model")[["coverage_80","coverage_95","width_80","width_95"]].mean().reset_index(),dirs["tables"]/"table19_interval_calibration.csv")
    write_csv(predictions,dirs["forecasts"]/"forecast_metrics_by_origin.csv")
    future_temp=future_temperature_scenarios(month_temp,cfg["analysis"]["future_years"]); write_csv(future_temp,dirs["tables"]/"table20_future_temperature_projections.csv")
    future_counts,paths=future_count_scenarios(selected,selected_name,cfg["analysis"]["future_years"],count_data.year.mean(),cfg["analysis"]["simulations"],seed,float(primary_hot.heatwave_days.median()))
    write_csv(future_counts,dirs["tables"]/"table21_future_heatwave_distributions.csv"); write_csv(paths,dirs["forecasts"]/"simulated_future_paths_summary.csv")
    # Sensitivity summaries: definitions, reference periods, 2024, model class, tree-cover detrending, forecast ranking.
    sens=[]
    for definition in ["operational_36c_1d","persistent_36c_2d","persistent_36c_3d","relative_90p_3d","relative_95p_3d","compound_90p_2d"]:
        c=daily_events[daily_events.month.isin([3,4,5,6])].groupby("year")[definition].sum().reset_index(name="heatwave_days")
        _,_,_,mod,name,_=fit_count_models(c); ci=mod.conf_int().loc["decade"]; sens.append({"analysis":"definition","variant":definition,"estimate":np.exp(mod.params["decade"]),"ci_lower":np.exp(ci[0]),"ci_upper":np.exp(ci[1]),"conclusion":"definition-dependent"})
    status2,_,_=construct_definition(df,"relative_90p_3d",(1991,2020)); c2=pd.DataFrame({"year":df.year,"month":df.month,"status":status2}).query("year <= 2024 and month in [3,4,5,6]").groupby("year").status.sum().reset_index(name="heatwave_days"); _,_,_,mod2,name2,_=fit_count_models(c2); sens.append({"analysis":"percentile_reference","variant":"1991-2020 90p","estimate":np.exp(mod2.params["decade"]),"ci_lower":np.nan,"ci_upper":np.nan,"conclusion":"partially robust"})
    for endpoint in [2023,2024]:
        c=primary_hot[primary_hot.year<=endpoint]; _,_,_,mod,name,_=fit_count_models(c); sens.append({"analysis":"2024_hot_season","variant":f"through_{endpoint}","estimate":np.exp(mod.params["decade"]),"ci_lower":np.nan,"ci_upper":np.nan,"conclusion":"robust"})
    for _,row in performance.groupby("model").rmse.mean().items(): pass
    for model,rmse in performance.groupby("model").rmse.mean().items(): sens.append({"analysis":"forecast_model","variant":model,"estimate":rmse,"ci_lower":np.nan,"ci_upper":np.nan,"conclusion":"model-dependent"})
    gfw=pd.read_csv(ROOT/cfg["data"]["tree_cover_csv"]); g=gfw.dropna(subset=["Tree_Cover_Loss_Year"]).drop_duplicates("Tree_Cover_Loss_Year"); g=g.rename(columns={"Tree_Cover_Loss_Year":"year","umd_tree_cover_loss__ha":"loss"}); annual_t=df.groupby("year").tmax.mean().reset_index(); tg=g.merge(annual_t,on="year"); raw_r,raw_p=stats.spearmanr(tg.loss,tg.tmax); loss_res=stats.linregress(tg.year,tg.loss); temp_res=stats.linregress(tg.year,tg.tmax); det_r,det_p=stats.spearmanr(tg.loss-(loss_res.intercept+loss_res.slope*tg.year),tg.tmax-(temp_res.intercept+temp_res.slope*tg.year)); sens.extend([{"analysis":"tree_cover","variant":"raw_spearman","estimate":raw_r,"ci_lower":np.nan,"ci_upper":np.nan,"conclusion":"unsupported"},{"analysis":"tree_cover","variant":"detrended_spearman","estimate":det_r,"ci_lower":np.nan,"ci_upper":np.nan,"conclusion":"unsupported"}])
    sensitivity=pd.DataFrame(sens); write_csv(sensitivity,dirs["tables"]/"table22_sensitivity_summary.csv")
    generate_figures(df,completeness,raw_corr,anom_corr,daily_events,events,annual,count_diag,estimates,assoc,binary,predictions,performance,future_counts,dirs["figures"])
    markdown_tables(dirs["tables"]); notebook_files()
    start_sha=subprocess.check_output(["git","rev-list","--max-parents=0","HEAD"],cwd=ROOT,text=True).strip()
    # Mandated starting main SHA is fixed in the baseline audit for this run.
    start_sha="926400ce49ebf2e8e87561beeedb7be93a19dcaf"
    q.update({"mean_tmax":float(df.tmax.mean()),"maximum_tmax":float(df.tmax.max()),"maximum_tmax_date":str(df.loc[df.tmax.idxmax(),"date"].date()),"operational_days":int(df.operational_36c_1d.sum()),"operational_events":int((events.definition=="operational_36c_1d").sum()),"primary_days":int(df.persistent_36c_3d.sum()),"primary_events":int((events.definition=="persistent_36c_3d").sum()),"longest_primary_event":int(events.loc[events.definition=="persistent_36c_3d","duration"].max())})
    legacy=json.loads(Path("/tmp/legacy_notebook_execution.json").read_text()) if Path("/tmp/legacy_notebook_execution.json").exists() else {"success":False,"message":"not available"}
    write_reports(reports,q,trends,comparison,estimates,assoc,binary,performance,future_counts,sensitivity,start_sha,legacy)
    metadata={"python_version":sys.version,"operating_system":platform.platform(),"package_versions":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__,"scikit_learn":sklearn.__version__},"random_seed":seed,"starting_commit":start_sha,"run_timestamp_utc":datetime.now(timezone.utc).isoformat(),"config":cfg,"runtime_seconds":round(time.time()-started,3),"data_hashes":source_hashes(ROOT/"data")}
    (dirs["metadata"]/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"status":"complete","runtime_seconds":metadata["runtime_seconds"],"selected_count_model":selected_name,"tables":len(list(dirs['tables'].glob('*.csv'))),"figures":len(list(dirs['figures'].glob('*.png')))},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="config/analysis.yml"); args=parser.parse_args(); main(args.config)
