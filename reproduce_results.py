#!/usr/bin/env python3
"""Recompute released gat_edge_v4 metrics and derive a split comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


SPLITS = (
    "random_split",
    "scaffold_split",
    "reaction_core_split",
    "drfp_kmeans100_greedy_split",
)
SEEDS = (42, 43, 44, 45, 46)
MODEL = "gat_edge_v4"
METRICS = ("mcc", "f0_5", "precision", "recall", "average_precision")


def recompute_seed(seed_dir: Path, split: str, seed: int) -> dict:
    predictions_path = seed_dir / "test_predictions.csv"
    metrics_path = seed_dir / "metrics.json"
    threshold_path = seed_dir / "threshold.json"
    for path in (predictions_path, metrics_path, threshold_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(predictions_path, dtype={"ID": "string"})
    expected_columns = {"ID", "y_true", "y_probability", "threshold", "y_pred"}
    if not expected_columns.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns in {predictions_path}")
    if predictions.empty or predictions["ID"].isna().any() or predictions["ID"].duplicated().any():
        raise ValueError(f"Missing, empty, or duplicate IDs in {predictions_path}")

    y_true = predictions["y_true"].to_numpy(dtype=np.int8)
    probabilities = predictions["y_probability"].to_numpy(dtype=np.float64)
    threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = float(threshold_data["threshold"])
    if not np.allclose(predictions["threshold"].to_numpy(dtype=float), threshold, rtol=0, atol=1e-12):
        raise ValueError(f"Prediction thresholds differ from {threshold_path}")
    y_pred = (probabilities >= threshold).astype(np.int8)
    if not np.array_equal(y_pred, predictions["y_pred"].to_numpy(dtype=np.int8)):
        raise ValueError(f"Predicted labels do not match the probabilities in {predictions_path}")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    computed = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }
    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
    for name, value in computed.items():
        if not np.isclose(value, float(reported[name]), rtol=0, atol=1e-6):
            raise ValueError(f"{split}/seed_{seed}: {name} differs from metrics.json")
    if matrix.tolist() != reported["confusion_matrix"]:
        raise ValueError(f"{split}/seed_{seed}: confusion matrix differs from metrics.json")
    if not np.isclose(threshold, float(reported["threshold"]), rtol=0, atol=1e-12):
        raise ValueError(f"{split}/seed_{seed}: threshold differs from metrics.json")

    return {
        "split": split,
        "model": MODEL,
        "seed": seed,
        "n_test": len(predictions),
        **computed,
        "ap": computed["average_precision"],
        "threshold": threshold,
        "confusion_matrix": json.dumps(matrix.tolist(), separators=(",", ":")),
    }


def summarize(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summaries = []
    for split in SPLITS:
        group = frame.loc[frame["split"] == split]
        if len(group) != len(SEEDS):
            raise ValueError(f"Expected {len(SEEDS)} seeds for {split}; found {len(group)}")
        row = {"split": split, "model": MODEL, "n_seeds": len(group), "seeds": ",".join(map(str, SEEDS))}
        for name in (*METRICS, "ap"):
            values = group[name].to_numpy(dtype=float)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=0))
        matrices = np.asarray([json.loads(value) for value in group["confusion_matrix"]], dtype=np.int64)
        row["confusion_matrix_sum"] = json.dumps(matrices.sum(axis=0).tolist(), separators=(",", ":"))
        row["confusion_matrix_mean"] = json.dumps(matrices.mean(axis=0).tolist(), separators=(",", ":"))
        summaries.append(row)
    return pd.DataFrame(summaries)


def draw_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = ("Random", "Scaffold", "Reaction core", "DRFP greedy")
    names = ("MCC", "F0.5", "Precision", "Recall", "AP")
    positions = np.arange(len(SPLITS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for index, (metric, label) in enumerate(zip(METRICS, names)):
        ax.bar(
            positions + (index - 2) * width,
            summary[f"{metric}_mean"],
            width,
            yerr=summary[f"{metric}_std"],
            capsize=2,
            label=label,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test metric (mean ± population SD across five seeds)")
    ax.set_title("Derived comparison from released gat_edge_v4 test predictions")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "gat_edge_v4_split_metrics.png", dpi=200)
    fig.savefig(output_dir / "gat_edge_v4_split_metrics.svg")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=root / "results")
    parser.add_argument("--output-dir", type=Path, default=root / "derived")
    args = parser.parse_args()
    rows = [
        recompute_seed(args.results_root / split / MODEL / f"seed_{seed}", split, seed)
        for split in SPLITS
        for seed in SEEDS
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "recomputed_seed_metrics.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(args.output_dir / "recomputed_split_summary.csv", index=False)
    draw_figure(summary, args.output_dir)
    print(f"Verified {len(rows)} seed predictions; wrote tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
