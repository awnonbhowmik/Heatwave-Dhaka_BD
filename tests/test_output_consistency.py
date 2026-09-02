from pathlib import Path
from zipfile import ZipFile
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
