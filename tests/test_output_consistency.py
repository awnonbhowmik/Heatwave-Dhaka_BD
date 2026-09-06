from pathlib import Path
from zipfile import ZipFile
import re
import pandas as pd

def test_expected_outputs_if_pipeline_has_run():
    root=Path("results/tables")
    if not root.exists(): return
    assert len(list(root.glob("table*.csv")))==22
    estimate=pd.read_csv(root/"table12_selected_count_model.csv").iloc[0]
    assert f"{estimate.incidence_rate_ratio:.3f}" in Path("reports/statistical_analysis_report.md").read_text()


def test_article_claims_and_definition_multiplicity_if_pipeline_has_run():
    path=Path("results/tables/main_table03_definition_sensitivity.csv")
    if not path.exists(): return
    table=pd.read_csv(path)
    compound=table.loc[table.definition.eq("compound_90p_2d")].iloc[0]
    assert compound.p_value < 0.05
    assert compound.q_value_bh_across_primary_definitions > 0.05
    manuscript=Path("manuscript/original_article_clean.md").read_text()
    assert "exploratory signal rather than confirmatory evidence" in manuscript


def test_word_deliverables_are_editable_if_built():
    clean=Path("manuscript/original_article_clean.docx")
    if not clean.exists(): return
    with ZipFile(clean) as archive:
        xml=archive.read("word/document.xml").decode("utf-8")
    assert "w:tbl" in xml
    assert "Figure 7." in xml
    assert "Figure 10." in xml


def test_integrated_manuscript_preserves_and_augments_references():
    manuscript=Path("manuscript/original_article_clean.md").read_text()
    reference_text=manuscript.split("## References\n",1)[1]
    references=[entry for entry in reference_text.split("\n\n") if entry.strip()]
    assert len(references)==66
    baseline=Path("config/manuscript_reference_baseline.txt").read_text().splitlines()
    assert len(baseline)==54
    for item in baseline:
        author,year=item.split("|")
        assert re.search(re.escape(author)+r".*\("+year+r"\)",reference_text)
    assert "Kan, J.-C." in manuscript
    assert "Choudary V, R." in manuscript
    assert "Saito, T., & Rehmsmeier, M." in manuscript
    assert "Van Calster, B." in manuscript
    assert "Lundberg, S. M., & Lee, S.-I." in manuscript


def test_integrated_prediction_table_matches_benchmark_if_built():
    table_path=Path("results/tables/main/main_table09_short_lead_prediction.csv")
    if not table_path.exists(): return
    table=pd.read_csv(table_path)
    logistic=table[(table.model=="Logistic") & (table.information_set=="S1") & (table.lead_days==1)].iloc[0]
    xgboost=table[(table.model=="XGBoost") & (table.information_set=="S2") & (table.lead_days==1)].iloc[0]
    assert abs(logistic.average_precision-0.5223125539)<1e-3
    assert abs(xgboost.average_precision-0.4268476804)<1e-3
