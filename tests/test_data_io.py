import pandas as pd
from heatwave_analysis.data_io import complete_years, load_daily

def test_raw_dates_are_continuous():
    d=load_daily("data/1972_2024_Heatwave_Daily.csv")
    assert d.date.min()==pd.Timestamp("1972-01-01")
    assert d.date.max()==pd.Timestamp("2024-11-18")
    assert not d.date.duplicated().any()
    assert len(pd.date_range(d.date.min(),d.date.max()).difference(d.date))==0

def test_incomplete_year_excluded():
    d=load_daily("data/1972_2024_Heatwave_Daily.csv")
    assert 2024 not in complete_years(d).year.unique()
