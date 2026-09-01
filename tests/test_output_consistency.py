from pathlib import Path
import pandas as pd

def test_expected_outputs_if_pipeline_has_run():
    root=Path("results/tables")
    if not root.exists(): return
    assert len(list(root.glob("table*.csv")))==22
    estimate=pd.read_csv(root/"table12_selected_count_model.csv").iloc[0]
    assert f"{estimate.incidence_rate_ratio:.3f}" in Path("reports/statistical_analysis_report.md").read_text()
