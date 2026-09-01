"""Shared diagnostic helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan


def time_series_residual_diagnostics(residuals):
    x=np.asarray(residuals); x=x[np.isfinite(x)]
    lb=acorr_ljungbox(x,lags=[min(12,len(x)//5)],return_df=True).iloc[0]
    return {"n":len(x),"mean":np.mean(x),"variance":np.var(x,ddof=1),"lag1_autocorrelation":np.corrcoef(x[:-1],x[1:])[0,1],
            "ljung_box_statistic":lb.lb_stat,"ljung_box_p_value":lb.lb_pvalue,"shapiro_p_value":stats.shapiro(x).pvalue if len(x)<=5000 else np.nan}
