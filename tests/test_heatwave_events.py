import pandas as pd
from heatwave_analysis.heatwave_events import group_persistent_events

def run(values,minimum=3,dates=None):
    dates=pd.Series(pd.to_datetime(dates or pd.date_range("2000-01-01",periods=len(values))))
    x=pd.Series(values,dtype=float)
    return group_persistent_events(dates,x>=36,minimum,36.0,x)

def test_isolated_and_consecutive_sequences():
    status,events=run([35,36,35,36,37,38,35])
    assert status.sum()==3 and len(events)==1 and events.iloc[0].cumulative_excess==3

def test_cool_day_and_missing_date_break_runs():
    status,events=run([36,37,35,36,37,38]); assert len(events)==1
    status,events=run([36,37,38],dates=["2000-01-01","2000-01-02","2000-01-04"]); assert len(events)==0

def test_year_boundary_is_one_event():
    status,events=run([36,37,38],dates=["1999-12-31","2000-01-01","2000-01-02"])
    assert len(events)==1 and events.iloc[0].duration==3
