#!/usr/bin/env python3
"""Strictly validate and aggregate the 6-dataset x 3-seed Jina matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = (
    "abt_buy",
    "amazon_google",
    "walmart_amazon",
    "wdc_80_medium_seen",
    "wdc_80_medium_unseen",
    "link_lives_release2",
)
SEEDS = (20260725, 20260726, 20260727)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.project_root / "artifacts" / "exp01_jina_finetuned" / "v1"
    rows = []
    missing = []
    for dataset in DATASETS:
        for seed in SEEDS:
            root = base / dataset / f"seed_{seed}"
            required = [
                root / name
                for name in (
                    "metrics.json",
                    "scores.npz",
                    "run_manifest.json",
                    "COMPLETED.json",
                )
            ]
            absent = [
                str(path)
                for path in required
                if not path.is_file() or path.stat().st_size == 0
            ]
            if absent:
                missing.extend(absent)
                continue
            metrics = json.loads((root / "metrics.json").read_text())
            complete = json.loads((root / "COMPLETED.json").read_text())
            scores = np.load(root / "scores.npz", allow_pickle=False)
            if metrics["status"] != "complete" or not metrics["paper_eligible"]:
                raise RuntimeError(f"Noneligible metrics: {root}")
            if complete["status"] != "complete":
                raise RuntimeError(f"Noncomplete marker: {root}")
            for split in ("valid", "test"):
                expected = int(metrics["coverage"][split]["expected"])
                observed = len(scores[f"{split}_score"])
                if (
                    observed != expected
                    or not np.isfinite(scores[f"{split}_score"]).all()
                ):
                    raise RuntimeError(f"Coverage/finite check failed: {root} {split}")
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "f1": metrics["test"]["f1"],
                    "precision": metrics["test"]["precision"],
                    "recall": metrics["test"]["recall"],
                    "average_precision": metrics["test"]["average_precision"],
                    "roc_auc": metrics["test"]["roc_auc"],
                    "threshold": metrics["threshold"],
                    "runtime_seconds": metrics["runtime_seconds"],
                }
            )
    if missing:
        raise SystemExit("Matrix is incomplete; missing:\n" + "\n".join(missing))
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("dataset", sort=False)
        .agg(
            runs=("seed", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            average_precision_mean=("average_precision", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    if not (summary["runs"] == 3).all():
        raise RuntimeError("Every dataset must contain exactly three seeded runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    frame.to_csv(args.output.with_name(args.output.stem + "_runs.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
