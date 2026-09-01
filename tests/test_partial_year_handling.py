from heatwave_analysis.data_io import load_daily
from heatwave_analysis.quality_control import completeness_table

def test_2024_hot_season_complete_but_calendar_incomplete():
    c=completeness_table(load_daily("data/1972_2024_Heatwave_Daily.csv")).set_index("year").loc[2024]
    assert not bool(c.calendar_complete) and bool(c.hot_season_complete) and c.hot_season_days==122
