#!/usr/bin/env python3
"""Validate and aggregate three-seed Exp2 propagated result cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (20260725, 20260726, 20260727)


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            output.update(flatten_metrics(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    payloads = []
    for seed in SEEDS:
        seed_root = args.cell_root / f"seed_{seed}"
        for name in ("metrics.json", "run_manifest.json", "scores.npz"):
            path = seed_root / name
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        payload = json.loads((seed_root / "metrics.json").read_text())
        if payload.get("dataset") != args.dataset or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Misaligned seed metric: {seed_root}")
        payloads.append(payload)
    flat = [flatten_metrics(value) for value in payloads]
    common = sorted(set.intersection(*(set(value) for value in flat)))
    aggregate: dict[str, Any] = {}
    for key in common:
        if not (
            ".validation." in f".{key}." or ".test." in f".{key}."
        ):
            continue
        if key.endswith((".queries", ".total_queries", ".pool_hit_queries")):
            continue
        values = np.asarray([value[key] for value in flat], dtype=float)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    result = {
        "dataset": args.dataset,
        "method": payloads[0]["method"],
        "seeds": list(SEEDS),
        "runs": 3,
        "aggregate": aggregate,
        "paper_eligible": True,
    }
    path = args.cell_root / "metrics.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
