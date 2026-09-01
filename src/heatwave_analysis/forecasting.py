"""Chronological monthly-temperature validation and trend-based count scenarios."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def monthly_temperature(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("date").tmax.resample("MS").mean().rename("tmax").reset_index()


def forecast_design() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "seasonal_naive", "target": "monthly mean Tmax", "information": "same month previous year", "interval": "empirical training residual"},
        {"model": "monthly_climatology", "target": "monthly mean Tmax", "information": "training month-of-year mean", "interval": "empirical training residual"},
        {"model": "climatology_linear_trend", "target": "monthly mean Tmax", "information": "training-only month effects plus trend", "interval": "normal OLS prediction"},
        {"model": "sarimax", "target": "monthly mean Tmax", "information": "SARIMA(1,1,1)x(1,0,0,12)", "interval": "state-space forecast"},
        {"model": "ets", "target": "monthly mean Tmax", "information": "additive trend and 12-month seasonality", "interval": "empirical training residual"},
    ])


def _metrics(observed, predicted, train, lower80, upper80, lower95, upper95):
    observed=np.asarray(observed); predicted=np.asarray(predicted)
    naive_scale = np.mean(np.abs(np.asarray(train)[12:] - np.asarray(train)[:-12]))
    return {
        "mae": np.mean(np.abs(observed-predicted)), "rmse": np.sqrt(np.mean((observed-predicted)**2)),
        "mean_bias": np.mean(predicted-observed), "mase": np.mean(np.abs(observed-predicted))/naive_scale,
        "coverage_80": np.mean((observed>=lower80)&(observed<=upper80)),
        "coverage_95": np.mean((observed>=lower95)&(observed<=upper95)),
        "width_80": np.mean(upper80-lower80), "width_95": np.mean(upper95-lower95),
    }


def rolling_origin_forecasts(monthly: pd.DataFrame, origins: list[int]):
    rows=[]; predictions=[]
    for year in origins:
        train=monthly[monthly.date < pd.Timestamp(year,1,1)].copy()
        test=monthly[(monthly.date.dt.year==year)&monthly.date.dt.month.isin([3,4,5,6])].copy()
        if len(test)!=4: continue
        y=train.set_index("date").tmax.asfreq("MS")
        residual_scale=max(np.std(y.values[12:]-y.values[:-12],ddof=1),.05)
        models={}
        pred_naive=np.array([y.get(pd.Timestamp(year-1,m,1),np.nan) for m in [3,4,5,6]])
        models["seasonal_naive"]=(pred_naive,residual_scale)
        clim=train.groupby(train.date.dt.month).tmax.mean()
        clim_resid=train.tmax-train.date.dt.month.map(clim)
        models["monthly_climatology"]=(np.array([clim[m] for m in [3,4,5,6]]),clim_resid.std(ddof=1))
        tr=train.copy(); tr["time"]=np.arange(len(tr)); X=sm.add_constant(pd.concat([tr.time,pd.get_dummies(tr.date.dt.month,prefix="m",drop_first=True,dtype=float)],axis=1))
        ols=sm.OLS(tr.tmax,X).fit(); future=pd.DataFrame({"time":[len(tr)+(pd.Timestamp(year,m,1)-tr.date.max()).days/30.4375-1 for m in [3,4,5,6]],"month":[3,4,5,6]})
        dummies=pd.get_dummies(future.month,prefix="m",dtype=float).reindex(columns=[c for c in X.columns if c.startswith("m_")],fill_value=0)
        Xf=sm.add_constant(pd.concat([future.time,dummies],axis=1),has_constant="add").reindex(columns=X.columns,fill_value=0)
        models["climatology_linear_trend"]=(ols.predict(Xf).to_numpy(),np.sqrt(ols.mse_resid))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sar=SARIMAX(y,order=(1,1,1),seasonal_order=(1,0,0,12),trend="t",enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
            sf=sar.get_forecast(steps=year*12+6-(y.index[-1].year*12+y.index[-1].month))
            sar_frame=sf.summary_frame(); sar_pred=np.array([sar_frame.loc[pd.Timestamp(year,m,1),"mean"] for m in [3,4,5,6]])
            models["sarimax"]=(sar_pred,float(np.std(sar.resid.dropna(),ddof=1)))
            ets=ExponentialSmoothing(y,trend="add",seasonal="add",seasonal_periods=12,initialization_method="estimated").fit(optimized=True)
            ef=ets.forecast(year*12+6-(y.index[-1].year*12+y.index[-1].month)); ets_pred=np.array([ef.loc[pd.Timestamp(year,m,1)] for m in [3,4,5,6]])
            models["ets"]=(ets_pred,float(np.std(ets.resid,ddof=1)))
        observed=test.tmax.to_numpy()
        for name,(pred,se) in models.items():
            lower80=pred-stats.norm.ppf(.9)*se; upper80=pred+stats.norm.ppf(.9)*se
            lower95=pred-stats.norm.ppf(.975)*se; upper95=pred+stats.norm.ppf(.975)*se
            rows.append({"origin_year":year,"model":name,**_metrics(observed,pred,y.values,lower80,upper80,lower95,upper95)})
            for date,obs,p,l80,u80,l95,u95 in zip(test.date,observed,pred,lower80,upper80,lower95,upper95):
                predictions.append({"origin_year":year,"date":date,"model":name,"observed":obs,"predicted":p,"lower_80":l80,"upper_80":u80,"lower_95":l95,"upper_95":u95})
    return pd.DataFrame(rows),pd.DataFrame(predictions)


def future_count_scenarios(count_model, selected_name: str, years: list[int], center_year: float, simulations: int, seed: int, historical_median: float):
    rng=np.random.default_rng(seed); decade=(np.asarray(years)-center_year)/10
    X=sm.add_constant(pd.DataFrame({"decade":decade}),has_constant="add")
    beta=rng.multivariate_normal(np.asarray(count_model.params)[:2],np.asarray(count_model.cov_params())[:2,:2],size=simulations)
    mu=np.exp(beta[:,0,None]+beta[:,1,None]*decade[None,:])
    if selected_name=="negative_binomial":
        alpha=max(float(count_model.params.get("alpha",.1)),1e-8); shape=1/alpha; p=shape/(shape+mu)
        draws=rng.negative_binomial(shape,p)
    else: draws=rng.poisson(mu)
    rows=[]
    for j,year in enumerate(years):
        x=draws[:,j]; rows.append({"year":year,"model":f"direct_{selected_name}_trend_scenario","median_heatwave_days":np.median(x),
            "p10":np.quantile(x,.1),"p90":np.quantile(x,.9),"p2_5":np.quantile(x,.025),"p97_5":np.quantile(x,.975),
            "probability_exceeds_historical_median":np.mean(x>historical_median),"probability_exceeds_20_days":np.mean(x>20)})
    summary=pd.DataFrame(rows)
    path_summary=pd.DataFrame({"simulation":np.arange(1,simulations+1),**{str(y):draws[:,j] for j,y in enumerate(years)}})
    return summary,path_summary


def future_temperature_scenarios(monthly: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Conditional month-effect plus linear-trend projections with prediction intervals."""
    tr=monthly.copy(); tr["time"]=np.arange(len(tr)); tr["month"]=tr.date.dt.month
    X=sm.add_constant(pd.concat([tr.time,pd.get_dummies(tr.month,prefix="m",drop_first=True,dtype=float)],axis=1))
    model=sm.OLS(tr.tmax,X).fit(cov_type="HAC",cov_kwds={"maxlags":12})
    rows=[]
    for year in years:
        future=pd.DataFrame({"date":[pd.Timestamp(year,m,1) for m in [3,4,5,6]]})
        future["time"]=[len(tr)+(d.year-tr.date.max().year)*12+d.month-tr.date.max().month-1 for d in future.date]
        dummies=pd.get_dummies(future.date.dt.month,prefix="m",dtype=float).reindex(columns=[c for c in X if c.startswith("m_")],fill_value=0)
        Xf=sm.add_constant(pd.concat([future.time,dummies],axis=1),has_constant="add").reindex(columns=X.columns,fill_value=0)
        pred=model.get_prediction(Xf).summary_frame(alpha=.05)
        for date,values in zip(future.date,pred.to_dict("records")):
            rows.append({"year":year,"month":date.month,"model":"climatology_linear_trend_scenario",
                         "mean_tmax":values["mean"],"ci_lower":values["mean_ci_lower"],"ci_upper":values["mean_ci_upper"],
                         "prediction_lower":values["obs_ci_lower"],"prediction_upper":values["obs_ci_upper"]})
    return pd.DataFrame(rows)
