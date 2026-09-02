import numpy as np
import pandas as pd

from heatwave_analysis.count_models import fit_count_models, influence_diagnostics


def test_selected_count_diagnostics_use_selected_distribution():
    counts = pd.DataFrame({
        "year": np.arange(2000, 2012),
        "heatwave_days": [0, 0, 1, 0, 12, 0, 2, 0, 18, 0, 1, 0],
    })
    data, poisson, _, selected, name, _ = fit_count_models(counts)
    diagnostics = influence_diagnostics(data, poisson, selected, name, seed=17)
    assert diagnostics.influence_method.str.contains(name).all()
    assert np.isfinite(diagnostics.randomized_quantile_residual).all()
    assert (diagnostics.cooks_distance >= 0).all()


def test_article_outputs_exist_if_pipeline_has_run():
    from pathlib import Path

    root = Path("results")
    if not root.exists():
        return
    assert (root / "tables" / "main_table04_temperature_trends.csv").exists()
    assert (root / "figures" / "main" / "figure04_temperature_trends.png").exists()
    assert (root / "figures" / "supplement" / "figureS01_selected_pairplots.png").exists()
