#!/usr/bin/env python3
"""Strict 6-dataset x 3-seed Exp2 Jina/HEF result aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
METHODS = ("jina_finetuned", "hef_gbdt_e5_plus_tuned_jina")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.project_root / "artifacts" / "exp02_jina_finetuned_oof" / "v1"
    rows, missing = [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            root = base / dataset / f"seed_{seed}"
            required = [
                root / n
                for n in (
                    "metrics.json",
                    "scores.csv.gz",
                    "run_manifest.json",
                    "COMPLETED.json",
                )
            ]
            absent = [
                str(p) for p in required if not p.is_file() or p.stat().st_size == 0
            ]
            if absent:
                missing.extend(absent)
                continue
            metrics = json.loads((root / "metrics.json").read_text())
            if not metrics.get("paper_eligible") or any(
                f["query_overlap"] for f in metrics["folds"]
            ):
                raise RuntimeError(f"Protocol assertion failed: {root}")
            for method in METHODS:
                test = metrics["methods"][method]["test"]
                rows.append(
                    {"dataset": dataset, "seed": seed, "method": method, **test}
                )
    if missing:
        raise SystemExit("Incomplete matrix:\n" + "\n".join(missing))
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["dataset", "method"], sort=False)
        .agg(
            runs=("seed", "count"),
            mrr_mean=("mrr_conditional", "mean"),
            mrr_std=("mrr_conditional", "std"),
            hits1_mean=("hits_at_1_conditional", "mean"),
            hits1_std=("hits_at_1_conditional", "std"),
            hits100_e2e=("hits_at_100_end_to_end", "mean"),
        )
        .reset_index()
    )
    if not (summary["runs"] == 3).all():
        raise RuntimeError("Expected three seeds per cell")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    runs.to_csv(args.output.with_name(args.output.stem + "_runs.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
