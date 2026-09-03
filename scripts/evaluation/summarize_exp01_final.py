#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from paper1_hef.evaluate import classification_metrics, paired_bootstrap_f1, select_threshold


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs" / "experiment.yaml").read_text())
    model_id = config["experiments"]["exp01_pair_classification"]["primary_backbone"]
    revision = next(x["revision"] for x in config["frozen_backbones"] if x["id"] == model_id)
    seeds = [int(seed) for seed in config["protocol"]["seeds"]]
    headline = config["dataset_groups"]["exp01_headline"]
    secondary = config["dataset_groups"]["exp01_secondary"]
    datasets = headline + secondary
    model_dir = model_id.replace("/", "__")

    table_rows: list[dict] = []
    comparisons: list[dict] = []
    completeness: dict[str, dict] = {}
    for dataset in datasets:
        fusion_dir = root / "artifacts" / "exp01_final" / dataset / model_dir / revision
        fusion_metrics = read_json(fusion_dir / "metrics.json")
        completeness[dataset] = {
            "fusion_metrics": True,
            "three_fusion_seeds": all(
                (fusion_dir / f"hef_gbdt_scores_seed_{seed}.npz").exists()
                and (fusion_dir / f"hef_linear_scores_seed_{seed}.npz").exists()
                for seed in seeds
            ),
            "cross_encoder_three_seeds": (
                all(
                    (
                        root
                        / "artifacts"
                        / "exp01_cross_encoder"
                        / dataset
                        / f"seed_{seed}"
                        / "metrics.json"
                    ).exists()
                    for seed in seeds
                )
                if dataset in headline
                else None
            ),
        }
        for method in ("rules", "frozen_embedding", "equal_fusion", "convex_fusion"):
            result = fusion_metrics["methods"][method]["test"]
            table_rows.append(
                {
                    "dataset": dataset,
                    "role": "headline" if dataset in headline else "secondary",
                    "method": method,
                    "f1_mean": result["f1"],
                    "f1_std": 0.0,
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "average_precision": result["average_precision"],
                    "roc_auc": result["roc_auc"],
                    "brier": result["brier"],
                    "ece": result["ece"],
                }
            )

        embedding_path = (
            root / "artifacts" / "embeddings" / dataset / model_dir / revision / "test.npz"
        )
        embedding_valid_path = (
            root / "artifacts" / "embeddings" / dataset / model_dir / revision / "valid.npz"
        )
        emb_test = np.load(embedding_path)
        emb_valid = np.load(embedding_valid_path)
        y_test = emb_test["label"].astype(int)
        y_valid = emb_valid["label"].astype(int)
        raw_train = np.load(
            root / "artifacts" / "embeddings" / dataset / model_dir / revision / "train.npz"
        )["embedding_score"].astype(float)
        low, high = float(raw_train.min()), float(raw_train.max())
        emb_test_scores = np.clip((emb_test["embedding_score"] - low) / (high - low), 0, 1)
        emb_valid_scores = np.clip((emb_valid["embedding_score"] - low) / (high - low), 0, 1)
        emb_threshold = select_threshold(y_valid, emb_valid_scores)

        for method in ("hef_linear", "hef_gbdt"):
            valid_scores = []
            test_scores = []
            per_seed_f1 = []
            for seed in seeds:
                values = np.load(fusion_dir / f"{method}_scores_seed_{seed}.npz")
                valid_scores.append(values["valid_score"].astype(float))
                test_scores.append(values["test_score"].astype(float))
                per_seed_f1.append(
                    next(
                        item["test"]["f1"]
                        for item in fusion_metrics["methods"][method]["repetitions"]
                        if int(item["seed"]) == seed
                    )
                )
            mean_valid = np.mean(valid_scores, axis=0)
            mean_test = np.mean(test_scores, axis=0)
            threshold = select_threshold(y_valid, mean_valid)
            result = classification_metrics(y_test, mean_test, threshold)
            comparison = paired_bootstrap_f1(
                y_test,
                mean_test,
                threshold,
                emb_test_scores,
                emb_threshold,
                int(config["protocol"]["bootstrap_replicates"]),
                seeds[0],
            )
            comparison.update(
                {
                    "dataset": dataset,
                    "method": method,
                    "correction_family": (
                        "headline-v1" if dataset in headline else "secondary-exploratory"
                    ),
                }
            )
            if dataset in headline:
                comparisons.append(comparison)
            table_rows.append(
                {
                    "dataset": dataset,
                    "role": "headline" if dataset in headline else "secondary",
                    "method": method,
                    "f1_mean": float(np.mean(per_seed_f1)),
                    "f1_std": float(np.std(per_seed_f1, ddof=1)),
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "average_precision": result["average_precision"],
                    "roc_auc": result["roc_auc"],
                    "brier": result["brier"],
                    "ece": result["ece"],
                }
            )

        if dataset in headline:
            cross_metrics = [
                read_json(
                    root
                    / "artifacts"
                    / "exp01_cross_encoder"
                    / dataset
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                for seed in seeds
            ]
            table_rows.append(
                {
                    "dataset": dataset,
                    "role": "headline",
                    "method": "tuned_cross_encoder",
                    "f1_mean": float(np.mean([item["test"]["f1"] for item in cross_metrics])),
                    "f1_std": float(np.std([item["test"]["f1"] for item in cross_metrics], ddof=1)),
                    "precision": float(np.mean([item["test"]["precision"] for item in cross_metrics])),
                    "recall": float(np.mean([item["test"]["recall"] for item in cross_metrics])),
                    "average_precision": float(
                        np.mean([item["test"]["average_precision"] for item in cross_metrics])
                    ),
                    "roc_auc": float(np.mean([item["test"]["roc_auc"] for item in cross_metrics])),
                    "brier": float(np.mean([item["test"]["brier"] for item in cross_metrics])),
                    "ece": float(np.mean([item["test"]["ece"] for item in cross_metrics])),
                }
            )

    adjusted = holm_adjust([item["two_sided_p"] for item in comparisons])
    for item, value in zip(comparisons, adjusted):
        item["holm_adjusted_p"] = value
        item["significant_at_0_05"] = value < 0.05

    complete = all(
        item["fusion_metrics"]
        and item["three_fusion_seeds"]
        and (item["cross_encoder_three_seeds"] is not False)
        for item in completeness.values()
    )
    output = root / "artifacts" / "exp01_final" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(table_rows)
    table.to_csv(output / "experiment_1_results.csv", index=False)
    summary = {
        "experiment": "Experiment 1: standard public pair classification",
        "scope": {
            "headline": headline,
            "secondary": secondary,
            "excluded": ["link_lives_release2", "private_genealogy"],
        },
        "primary_backbone": {"model_id": model_id, "revision": revision},
        "seeds": seeds,
        "correction_family": "headline-v1",
        "complete": complete,
        "completeness": completeness,
        "paired_comparisons": comparisons,
        "status": "paper_eligible" if complete else "incomplete",
    }
    (output / "experiment_1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    if complete:
        for dataset in datasets:
            manifest_path = (
                root
                / "artifacts"
                / "exp01_final"
                / dataset
                / model_dir
                / revision
                / "run_manifest.json"
            )
            manifest = read_json(manifest_path)
            manifest["paper_eligible"] = True
            manifest["paper_eligibility_blocker"] = None
            manifest["final_audit"] = str(
                (output / "experiment_1_summary.json").relative_to(root)
            )
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Experiment 1 results",
        "",
        f"Status: **{'complete' if complete else 'incomplete'}**.",
        "",
        "The table below reports the official held-out test results. HEF values",
        "are means across the three locked seeds.",
        "",
        "| Dataset | Role | Method | F1 | Precision | Recall | AP | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['method']} | "
            f"{row['f1_mean']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['average_precision']:.3f} | {row['roc_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Link-Lives and private genealogy are not included in this table.",
            "",
        ]
    )
    (output / "EXPERIMENT_1_RESULTS_SIMPLE.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
