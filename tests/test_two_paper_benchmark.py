import numpy as np
import pandas as pd
import pytest
import shap
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier

from heatwave_analysis.two_paper_benchmark import (
    CorrelationFilter,
    assert_prediction_provenance,
    classification_metrics,
    construct_prediction_samples,
    inner_year_splits,
)


def synthetic_daily(start="2000-02-20", end="2024-07-10"):
    dates = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame({"date": dates})
    frame["tmax"] = 32.0
    frame["tmin"] = 23.0
    for variable in ["rh_mean", "precipitation", "wind_speed_mean", "pressure_mean", "cloud_cover", "shortwave_radiation", "soil_moisture_mean"]:
        frame[variable] = np.linspace(0.1, 1.0, len(frame))
    return frame


def test_exact_lookback_and_future_label_alignment():
    daily = synthetic_daily("2000-02-20", "2000-07-10")
    daily.loc[daily.date.between("2000-03-10", "2000-03-12"), "tmax"] = [36.0, 37.0, 38.0]
    samples = construct_prediction_samples(daily, leads=[1], lookback_days=7)
    row = samples.loc[samples.issue_date.eq(pd.Timestamp("2000-03-09"))].iloc[0]
    assert row.feature_start == pd.Timestamp("2000-03-03")
    assert row.feature_end == row.issue_date
    assert row.target_start == pd.Timestamp("2000-03-10")
    assert row.target_end == pd.Timestamp("2000-03-12")
    assert row.label_available_date == row.target_end
    assert row.outcome == 1
    assert pd.notna(row.associated_event_id)


def test_missing_target_date_is_unknown_not_negative():
    daily = synthetic_daily("2000-02-20", "2000-07-10")
    daily = daily[daily.date.ne(pd.Timestamp("2000-03-11"))]
    samples = construct_prediction_samples(daily, leads=[1], lookback_days=7)
    assert not samples.issue_date.eq(pd.Timestamp("2000-03-09")).any()


def test_target_window_must_remain_inside_march_june():
    samples = construct_prediction_samples(synthetic_daily("2000-02-20", "2000-07-10"), leads=[1, 3, 7])
    assert samples.target_start.dt.month.isin([3, 4, 5, 6]).all()
    assert samples.target_end.dt.month.isin([3, 4, 5, 6]).all()
    assert samples.target_end.max() == pd.Timestamp("2000-06-30")


def test_temperature_history_is_allowed_but_never_future_dated():
    samples = construct_prediction_samples(synthetic_daily(), leads=[1])
    assert "tmax_latest" in samples
    assert_prediction_provenance(samples)
    assert (samples.feature_end < samples.target_start).all()


def test_persistence_is_observed_run_not_retrospective_event_status():
    daily = synthetic_daily("2000-02-20", "2000-07-10")
    daily.loc[daily.date.between("2000-03-08", "2000-03-12"), "tmax"] = 37
    samples = construct_prediction_samples(daily, leads=[1])
    row = samples.loc[samples.issue_date.eq(pd.Timestamp("2000-03-09"))].iloc[0]
    assert row.observed_hot_run_length == 2
    assert row.outcome == 1


def test_inner_training_labels_available_and_event_free_year_retained():
    samples = construct_prediction_samples(synthetic_daily(), leads=[1])
    splits = inner_year_splits(samples[samples.issue_year < 2024], folds=3)
    assert len(splits) == 3
    for split in splits:
        train = samples.loc[split["train_index"]]
        assert (train.label_available_date < split["validation_start"]).all()
        validation = samples.loc[split["validation_index"]]
        assert validation.outcome.sum() == 0
        assert len(validation) > 0


def test_metric_formulas_and_undefined_values():
    metric = classification_metrics([0, 0, 1, 1], [.1, .8, .7, .2], .5)
    assert (metric["tn"], metric["fp"], metric["fn"], metric["tp"]) == (1, 1, 1, 1)
    assert metric["recall"] == pytest.approx(.5)
    assert metric["specificity"] == pytest.approx(.5)
    undefined = classification_metrics([0, 0], [.1, .2], .5)
    assert np.isnan(undefined["average_precision"])
    assert "no positives" in undefined["undefined_reason"]


def test_redundancy_filter_learns_from_training_only():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 1, 0, 0]})
    filt = CorrelationFilter(.95).fit(train)
    assert filt.keep_ == ["a", "c"]
    assert filt.drop_ == ["b"]
    assert filt.transform(train).shape == (4, 2)


def test_conservative_inner_gap_preserves_label_availability():
    samples = construct_prediction_samples(synthetic_daily(), leads=[7])
    splits = inner_year_splits(samples[samples.issue_year < 2024], folds=3, gap_days=10)
    for split in splits:
        train = samples.loc[split["train_index"]]
        assert (train.label_available_date < split["validation_start"] - timedelta(days=10)).all()


def test_tree_shap_class_one_additivity_and_scale():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 3)); y = (x[:, 0] + .2 * x[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(x, y)
    explanation = shap.TreeExplainer(model, data=x[:30], feature_perturbation="interventional", model_output="raw")(x[30:40])
    values = np.asarray(explanation.values)[:, :, 1]
    base = np.asarray(explanation.base_values)[:, 1]
    reconstructed = base + values.sum(axis=1)
    assert np.max(np.abs(reconstructed - model.predict_proba(x[30:40])[:, 1])) < 1e-6
