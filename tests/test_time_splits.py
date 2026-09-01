import pandas as pd
from heatwave_analysis.data_io import load_daily

def test_every_validation_split_is_chronological():
    d=load_daily("data/1972_2024_Heatwave_Daily.csv")
    for year in [2014,2016,2018,2020,2022,2024]:
        assert d[d.year<year].date.max() < d[d.year==year].date.min()
