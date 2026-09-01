import numpy as np
import pandas as pd
from heatwave_analysis.climatology import calendar_day_threshold, climatology_day

def test_leap_day_handling_and_thresholds():
    dates=pd.date_range("1981-01-01","2010-12-31")
    d=pd.DataFrame({"date":dates}); d["year"]=d.date.dt.year; d["x"]=climatology_day(d.date).astype(float)
    t=calendar_day_threshold(d,"x",.9,(1981,2010))
    leap=d.date.eq(pd.Timestamp("1984-02-29"))
    assert leap.sum()==1 and np.isfinite(t[leap]).all()

def test_training_cutoff_never_uses_future_reference_data():
    dates=pd.date_range("1981-01-01","2010-12-31"); d=pd.DataFrame({"date":dates}); d["year"]=d.date.dt.year; d["x"]=1.0; d.loc[d.year>2000,"x"]=999
    t=calendar_day_threshold(d,"x",.9,(1981,2010),training_end_year=2000)
    assert t.max()==1
