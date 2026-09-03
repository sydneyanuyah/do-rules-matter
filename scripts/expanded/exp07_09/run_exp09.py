#!/usr/bin/env python3
"""Paired query bootstrap for finalized revised Experiment 7 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_bootstrap(
    left: np.ndarray, right: np.ndarray, *, seed: int, repetitions: int,
) -> dict[str, float]:
    if left.shape != right.shape or not len(left):
        raise ValueError("Paired arrays must be aligned and nonempty")
    difference = left - right
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 250):
        count = min(250, repetitions - start)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        samples[start:start + count] = difference[indices].mean(axis=1)
    observed = float(difference.mean())
    standard = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    lower, upper = np.quantile(samples, [0.025, 0.975])
    # Centered-bootstrap two-sided test of a zero paired mean.
    centered = samples - observed
    p_value = min(1.0, 2.0 * min(float((centered <= -abs(observed)).mean()),
                                  float((centered >= abs(observed)).mean())))
    return {
        "left_mean": float(left.mean()), "right_mean": float(right.mean()),
        "paired_difference": observed, "ci95_lower": float(lower), "ci95_upper": float(upper),
        "effect_size_paired_dz": observed / standard if standard else 0.0,
        "probability_of_superiority": float((difference > 0).mean() + 0.5 * (difference == 0).mean()),
        "raw_p_value": p_value, "queries": int(len(difference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp07-root", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True,
                        help="CSV columns: family,dataset,mask_probability,mask_seed,left_model,right_model,metric")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    manifest = args.exp07_root / "FINALIZED.json"
    if not manifest.exists() or not json.loads(manifest.read_text()).get("validated"):
        raise SystemExit("Experiment 9 is gated: finalized Experiment 7 manifest is absent")
    comparisons = pd.read_csv(args.comparisons)
    required = {"family", "dataset", "mask_probability", "mask_seed", "left_model", "right_model", "metric"}
    if required - set(comparisons):
        raise ValueError(f"Comparison registry missing {sorted(required - set(comparisons))}")
    results = []
    for index, row in comparisons.iterrows():
        suffix = f"p{int(round(100 * float(row.mask_probability))):02d}_seed{int(row.mask_seed)}"
        frames = []
        for model in (row.left_model, row.right_model):
            path = args.exp07_root / str(model) / str(row.dataset) / suffix / "per_query.parquet"
            frame = pd.read_parquet(path)[["query_id", str(row.metric)]].rename(columns={str(row.metric): model})
            frames.append(frame)
        paired = frames[0].merge(frames[1], on="query_id", validate="one_to_one")
        if len(paired) != len(frames[0]) or len(paired) != len(frames[1]):
            raise ValueError(f"Query alignment failure: {row.to_dict()}")
        stats = paired_bootstrap(
            paired[str(row.left_model)].to_numpy(float), paired[str(row.right_model)].to_numpy(float),
            seed=20260725 + index, repetitions=args.bootstrap_repetitions,
        )
        results.append({**row.to_dict(), **stats})
    output = pd.DataFrame(results)
    output["holm_adjusted_p_value"] = np.nan
    for _, indices in output.groupby("family").groups.items():
        output.loc[indices, "holm_adjusted_p_value"] = holm(output.loc[indices, "raw_p_value"].tolist())
    output["statistically_supported_0_05"] = output["holm_adjusted_p_value"].le(0.05)
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "paired_bootstrap_results.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "complete", "experiment": 9, "bootstrap_repetitions": args.bootstrap_repetitions,
        "hypothesis_families": sorted(output["family"].unique().tolist()),
        "comparisons": len(output), "holm_within_family": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
