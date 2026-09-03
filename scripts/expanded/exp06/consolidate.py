#!/usr/bin/env python3
"""Strictly validate and consolidate the 162-cell expanded Experiment 6 matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp06_common import CONDITIONS, DATASETS, SEED

MODELS = ("hef_gbdt_e5", "hef_gbdt_e5_tuned_roberta_oof", "joint_neural_hef_roberta")


def metric(payload: dict, model: str) -> tuple[float, float]:
    tree = payload["ranking_test"] if model == "joint_neural_hef_roberta" else payload["test"]
    return float(tree["100"]["conditional"]["mrr"]), float(tree["1"]["conditional"]["hits"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root / "artifacts" / "exp06_rerun_v2"
    rows, missing = [], []
    for model in MODELS:
        for condition in CONDITIONS:
            for dataset in DATASETS:
                leaf = base / model / condition / dataset / f"seed_{SEED}"
                required = [leaf / name for name in ("SUCCESS.json", "metrics.json", "manifest.json", "scores.npz")]
                if any(not path.exists() or path.stat().st_size == 0 for path in required):
                    missing.append((model, condition, dataset)); continue
                payload = json.loads((leaf / "metrics.json").read_text())
                if payload["model"] != model or payload["condition"] != condition or payload["dataset"] != dataset:
                    raise ValueError(f"Identity mismatch: {leaf}")
                mrr, hits1 = metric(payload, model)
                if not np.isfinite([mrr, hits1]).all(): raise ValueError(f"Nonfinite metric: {leaf}")
                rows.append({"model": model, "condition": condition, "dataset": dataset,
                             "seed": SEED, "mrr_at_100": mrr, "hits_at_1": hits1})
    if missing:
        raise RuntimeError(f"Matrix incomplete: {len(rows)}/162 complete; first missing={missing[:10]}")
    frame = pd.DataFrame(rows).sort_values(["model", "condition", "dataset"])
    output = base / "consolidated"; output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "results.csv", index=False)
    summary = frame.groupby(["model", "condition"], as_index=False).agg(
        mean_mrr_at_100=("mrr_at_100", "mean"), mean_hits_at_1=("hits_at_1", "mean")
    )
    summary.to_csv(output / "summary.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "experiment": "exp06_public_evidence_ablation_rerun_v2", "expected_cells": 162,
        "completed_cells": len(frame), "models": list(MODELS), "conditions": list(CONDITIONS),
        "datasets": list(DATASETS), "seed": SEED, "variance_claim": False, "status": "complete",
    }, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__": main()

