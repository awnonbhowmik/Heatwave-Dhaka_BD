import pandas as pd

def test_generated_interval_order_if_outputs_exist():
    path="results/tables/table20_future_temperature_projections.csv"
    try: d=pd.read_csv(path)
    except FileNotFoundError: return
    assert (d.prediction_lower<=d.mean_tmax).all() and (d.mean_tmax<=d.prediction_upper).all()
