#!/usr/bin/env python3
"""Validate output contracts, intervals, report/table consistency, and source immutability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_TABLES=[f"table{i:02d}" for i in range(1,23)]


def main(config_path):
    cfg=yaml.safe_load((ROOT/config_path).read_text()); out=ROOT/cfg["outputs"]["root"]
    csvs=list((out/"tables").glob("table*.csv")); stems=[p.stem[:7] for p in csvs]
    missing=[name for name in EXPECTED_TABLES if name not in stems]
    assert not missing,f"Missing tables: {missing}"
    assert len(list((out/"figures").glob("figure*.png")))==10
    assert len(list((out/"figures").glob("figure*.pdf")))==10
    required_main_tables=[
        "main_table01_data_and_descriptive_statistics.csv",
        "main_table02_correlations_and_collinearity.csv",
        "main_table03_definition_sensitivity.csv",
        "main_table04_temperature_trends.csv",
        "main_table05_poisson_nb_comparison.csv",
        "main_table06_primary_count_model.csv",
        "main_table07_adjusted_associations.csv",
        "main_table08_blocked_validation.csv",
    ]
    for name in required_main_tables:
        assert (out/"tables"/name).exists(),f"Missing original-article table: {name}"
        assert (out/"tables"/"main"/name).exists(),f"Missing organized main table: {name}"
    assert len(list((out/"figures"/"main").glob("figure*.png")))==7
    assert len(list((out/"figures"/"main").glob("figure*.pdf")))==7
    assert len(list((out/"figures"/"supplement").glob("figureS*.png")))>=2
    future=pd.read_csv(out/"tables"/"table20_future_temperature_projections.csv")
    assert (future.ci_lower<=future.mean_tmax).all() and (future.mean_tmax<=future.ci_upper).all()
    assert (future.prediction_lower<=future.mean_tmax).all() and (future.mean_tmax<=future.prediction_upper).all()
    counts=pd.read_csv(out/"tables"/"table21_future_heatwave_distributions.csv")
    assert (counts.p2_5<=counts.p10).all() and (counts.p10<=counts.median_heatwave_days).all() and (counts.median_heatwave_days<=counts.p90).all() and (counts.p90<=counts.p97_5).all()
    meta=json.loads((out/"metadata"/"run_metadata.json").read_text())
    for path,digest in meta["data_hashes"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,f"Source changed: {path}"
    estimate=pd.read_csv(out/"tables"/"table12_selected_count_model.csv").iloc[0]
    report=(ROOT/"reports"/"statistical_analysis_report.md").read_text()
    assert f"{estimate.incidence_rate_ratio:.3f}" in report
    definition_table=pd.read_csv(out/"tables"/"main_table03_definition_sensitivity.csv")
    compound=definition_table.loc[definition_table.definition.eq("compound_90p_2d")].iloc[0]
    assert 0.12 < compound.q_value_bh_across_primary_definitions < 0.13
    manuscript=(ROOT/"manuscript"/"original_article_clean.md").read_text()
    assert "\\(q=0.126\\)" in manuscript
    assert "systematic underprediction" in manuscript
    required_docs=[
        "original_article_clean.docx",
        "original_article_updates_highlighted_yellow.docx",
        "supplementary_material.docx",
    ]
    for name in required_docs:
        path=ROOT/"manuscript"/name
        assert path.exists() and path.stat().st_size>10_000,f"Missing or empty Word deliverable: {name}"
        with ZipFile(path) as archive:
            xml=archive.read("word/document.xml").decode("utf-8")
            assert "Long-Term Warming" in xml
            assert "w:tbl" in xml
    highlighted=ROOT/"manuscript"/"original_article_updates_highlighted_yellow.docx"
    with ZipFile(highlighted) as archive:
        xml=archive.read("word/document.xml").decode("utf-8")
        assert 'w:highlight w:val="yellow"' in xml
    print(f"Validated {len(csvs)} numbered CSV tables, 8 main article tables, 7 main figures, supplementary figures, intervals, source hashes, manuscript claims, and editable Word deliverables.")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/analysis.yml"); main(p.parse_args().config)
