#!/usr/bin/env python3
"""Build compact manuscript assets from the two-paper benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "two_paper_benchmark"
FIGURES = BENCHMARK / "figures"
FIGURE_SOURCES = FIGURES / "source_tables"
MAIN_TABLE = ROOT / "results" / "tables" / "main" / "main_table09_short_lead_prediction.csv"
ROOT_TABLE = ROOT / "results" / "tables" / "main_table09_short_lead_prediction.csv"


MODEL_LABELS = {
    "always_negative": "Always negative",
    "s0_logistic": "Calendar logistic",
    "seasonal_probability": "Seasonal probability",
    "temperature_transition": "Temperature transition",
    "logistic": "Logistic",
    "weighted_rf": "Weighted RF",
    "balanced_rf": "Balanced RF",
    "weighted_svc": "Weighted SVC",
    "xgboost": "XGBoost",
}


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pooled_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(BENCHMARK / "metrics" / "model_metrics_by_year_and_pooled.csv")
    return metrics[metrics["scope"].eq("pooled_strictly_out_of_sample")].copy()


def build_main_table(pooled: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    requested = [
        ("always_negative", "S0", 1),
        ("seasonal_probability", "S0", 1),
        ("temperature_transition", "S0", 1),
        ("logistic", "S1", 1),
        ("logistic", "S2", 1),
        ("weighted_rf", "S1", 1),
        ("weighted_rf", "S2", 1),
        ("xgboost", "S1", 1),
        ("xgboost", "S2", 1),
        ("logistic", "S1", 3),
        ("logistic", "S1", 7),
        ("xgboost", "S2", 3),
        ("xgboost", "S2", 7),
    ]
    rows: list[dict[str, object]] = []
    for model, feature_set, lead in requested:
        match = pooled[
            pooled["model"].eq(model)
            & pooled["feature_set"].eq(feature_set)
            & pooled["lead"].eq(lead)
        ]
        if len(match) != 1:
            raise RuntimeError(f"Expected one pooled row for {(model, feature_set, lead)}, found {len(match)}")
        row = match.iloc[0]
        result: dict[str, object] = {
            "model": MODEL_LABELS[model],
            "information_set": feature_set,
            "lead_days": int(lead),
            "average_precision": row["average_precision"],
            "brier_score": row["brier_score"],
            "recall": row["recall"],
            "precision": row["precision"],
            "S2_minus_S1_AP_95pct_interval": "",
            "S2_minus_S1_Brier_95pct_interval": "",
        }
        if feature_set == "S2":
            family = comparisons[
                comparisons["model"].eq(model)
                & comparisons["lead"].eq(lead)
            ]
            for metric, destination in [
                ("average_precision", "S2_minus_S1_AP_95pct_interval"),
                ("brier_score", "S2_minus_S1_Brier_95pct_interval"),
            ]:
                contrast = family[family["metric"].eq(metric)]
                if len(contrast) == 1:
                    value = contrast.iloc[0]
                    digits = 3 if metric == "average_precision" else 4
                    result[destination] = (
                        f"{value['observed_difference']:.{digits}f} "
                        f"({value['ci_lower']:.{digits}f}, {value['ci_upper']:.{digits}f})"
                    )
        rows.append(result)
    table = pd.DataFrame(rows)
    MAIN_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(MAIN_TABLE, index=False, float_format="%.4f")
    table.to_csv(ROOT_TABLE, index=False, float_format="%.4f")
    return table


def figure_primary_comparisons(pooled: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    selected = [
        ("always_negative", "S0"),
        ("seasonal_probability", "S0"),
        ("temperature_transition", "S0"),
        ("logistic", "S1"),
        ("logistic", "S2"),
        ("weighted_rf", "S1"),
        ("weighted_rf", "S2"),
        ("xgboost", "S1"),
        ("xgboost", "S2"),
    ]
    h1 = pooled[pooled["lead"].eq(1)].set_index(["model", "feature_set"])
    labels = [f"{MODEL_LABELS[model]} {feature_set}" for model, feature_set in selected]
    values = [h1.loc[(model, feature_set), "average_precision"] for model, feature_set in selected]
    colors = ["0.72" if feature_set == "S0" else "#4472C4" if feature_set == "S1" else "#ED7D31" for _, feature_set in selected]
    positions = np.arange(len(labels))
    axes[0].barh(positions, values, color=colors)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Average precision", title="A  One-day held-out performance", xlim=(0, 0.58))
    for y, value in zip(positions, values):
        axes[0].text(value + 0.008, y, f"{value:.3f}", va="center", fontsize=8)

    forest = comparisons[
        comparisons["lead"].eq(1) & comparisons["metric"].eq("average_precision")
    ].copy()
    family_order = ["logistic", "weighted_svc", "balanced_rf", "weighted_rf", "xgboost"]
    forest["order"] = forest["model"].map({name: i for i, name in enumerate(family_order)})
    forest = forest.sort_values("order")
    y = np.arange(len(forest))
    x = forest["observed_difference"].to_numpy()
    xerr = np.vstack([x - forest["ci_lower"].to_numpy(), forest["ci_upper"].to_numpy() - x])
    axes[1].errorbar(x, y, xerr=xerr, fmt="o", color="black", ecolor="0.35", capsize=3)
    axes[1].axvline(0, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_yticks(y, [MODEL_LABELS[name] for name in forest["model"]])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="S2 − S1 average precision", title="B  Added-meteorology contrast")

    styles = [
        ("logistic", "S1", "Logistic S1", "#4472C4", "o"),
        ("xgboost", "S2", "XGBoost S2", "#ED7D31", "s"),
        ("temperature_transition", "S0", "Temperature transition", "#70AD47", "^"),
        ("seasonal_probability", "S0", "Seasonal probability", "0.45", "D"),
    ]
    for model, feature_set, label, color, marker in styles:
        frame = pooled[pooled["model"].eq(model) & pooled["feature_set"].eq(feature_set)].sort_values("lead")
        axes[2].plot(frame["lead"], frame["average_precision"], marker=marker, color=color, label=label)
    FIGURE_SOURCES.mkdir(parents=True, exist_ok=True)
    pooled[
        pooled[["model", "feature_set"]].apply(tuple, axis=1).isin(
            {(model, feature_set) for model, feature_set, *_ in styles}
            | set(selected)
        )
    ].to_csv(FIGURE_SOURCES / "figure08_manuscript_primary_prediction.csv", index=False)
    forest.to_csv(FIGURE_SOURCES / "figure08_h1_AP_contrasts.csv", index=False)
    axes[2].set_xticks([1, 3, 7])
    axes[2].set(xlabel="Lead (days)", ylabel="Average precision", title="C  Skill by direct lead")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    save_figure(fig, FIGURES / "figure08_manuscript_primary_prediction")


def figure_diagnostics() -> None:
    predictions = pd.read_csv(BENCHMARK / "predictions" / "all_out_of_sample_predictions.csv")
    reliability = pd.read_csv(BENCHMARK / "calibration" / "reliability_source.csv")
    onset = pd.read_csv(BENCHMARK / "metrics" / "onset_risk_subset.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    styles = [
        ("logistic", "S1", "Logistic S1", "#4472C4"),
        ("xgboost", "S1", "XGBoost S1", "0.5"),
        ("xgboost", "S2", "XGBoost S2", "#ED7D31"),
        ("temperature_transition", "S0", "Temperature transition", "#70AD47"),
    ]
    for model, feature_set, label, color in styles:
        frame = predictions[
            predictions["model"].eq(model)
            & predictions["feature_set"].eq(feature_set)
            & predictions["lead"].eq(1)
        ]
        precision, recall, _ = precision_recall_curve(frame["outcome"], frame["probability"])
        axes[0].plot(recall, precision, label=label, color=color)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="A  Held-out precision–recall")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    for model, feature_set, label, color in [styles[0], styles[2]]:
        frame = reliability[
            reliability["model"].eq(model)
            & reliability["feature_set"].eq(feature_set)
            & reliability["lead"].eq(1)
        ].sort_values("mean_probability")
        axes[1].plot(frame["mean_probability"], frame["observed_fraction"], marker="o", label=label, color=color)
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed fraction", title="B  Temporal reliability", xlim=(0, 0.75), ylim=(0, 0.85))
    axes[1].legend(frameon=False, fontsize=8)

    subset = onset[
        ((onset["model"].eq("logistic")) & onset["feature_set"].eq("S1"))
        | ((onset["model"].eq("xgboost")) & onset["feature_set"].eq("S2"))
    ].copy()
    selected_predictions = predictions[
        predictions["lead"].eq(1)
        & predictions[["model", "feature_set"]].apply(tuple, axis=1).isin(
            {(model, feature_set) for model, feature_set, *_ in styles}
        )
    ]
    selected_reliability = reliability[
        reliability["lead"].eq(1)
        & reliability[["model", "feature_set"]].apply(tuple, axis=1).isin(
            {("logistic", "S1"), ("xgboost", "S2")}
        )
    ]
    FIGURE_SOURCES.mkdir(parents=True, exist_ok=True)
    selected_predictions.to_csv(FIGURE_SOURCES / "figure09_precision_recall_predictions.csv", index=False)
    selected_reliability.to_csv(FIGURE_SOURCES / "figure09_reliability.csv", index=False)
    subset.to_csv(FIGURE_SOURCES / "figure09_onset_alert_burden.csv", index=False)
    categories = ["Detected spells", "Missed spells", "False alerts"]
    x = np.arange(len(categories))
    width = 0.34
    for offset, row in zip([-width / 2, width / 2], subset.itertuples()):
        values = [row.tp, row.fn, row.false_alerts]
        label = f"{MODEL_LABELS[row.model]} {row.feature_set}"
        axes[2].bar(x + offset, values, width, label=label)
        for xpos, value in zip(x + offset, values):
            axes[2].text(xpos, value * 1.08, str(int(value)), ha="center", va="bottom", fontsize=8)
    axes[2].set_yscale("log")
    axes[2].set_xticks(x, categories, rotation=18, ha="right")
    axes[2].set(ylabel="Count (log scale)", title="C  Onset alert burden")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    save_figure(fig, FIGURES / "figure09_manuscript_prediction_diagnostics")


def main() -> None:
    pooled = pooled_metrics()
    comparisons = pd.read_csv(BENCHMARK / "metrics" / "paired_year_block_comparisons.csv")
    build_main_table(pooled, comparisons)
    figure_primary_comparisons(pooled, comparisons)
    figure_diagnostics()
    print(f"Wrote {MAIN_TABLE.relative_to(ROOT)} and manuscript Figures 8–9.")


if __name__ == "__main__":
    main()
