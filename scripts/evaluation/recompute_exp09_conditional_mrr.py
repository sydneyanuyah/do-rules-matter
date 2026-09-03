#!/usr/bin/env python3
"""Primary Exp9 inference: paired query bootstrap on conditional MRR@100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("abt_buy", "amazon_google", "walmart_amazon", "wdc_80_medium_seen", "wdc_80_medium_unseen", "link_lives_release2")
KEYS = ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
COMPARATORS = ("rules", "embedding", "ditto_style_roberta_mixda", "tuned_cross_encoder", "jina_cross_encoder", "anymatch_official")


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * float(p[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def score_frame(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "test_scored.csv.gz", dtype={"query_id": str, "candidate_id": str})
    sources = [
        ("anymatch_scores.csv.gz", "anymatch_score", "anymatch_official"),
        ("ditto_style_mixda_scores.csv.gz", "ditto_style_mixda_score", "ditto_style_roberta_mixda"),
        ("jina_cross_encoder_scores.csv.gz", "jina_score", "jina_cross_encoder"),
    ]
    roberta = "roberta_cross_encoder_scores.csv.gz" if (directory / "roberta_cross_encoder_scores.csv.gz").exists() else "cross_encoder_scores.csv.gz"
    sources.append((roberta, "mean_score", "tuned_cross_encoder"))
    for filename, source, target in sources:
        extra = pd.read_csv(directory / filename, dtype={"query_id": str, "candidate_id": str})
        if len(extra) != len(frame):
            raise ValueError(f"{directory.name}/{filename}: incomplete coverage")
        frame = frame.merge(extra[KEYS + [source]].rename(columns={source: target}), on=KEYS, how="left", validate="one_to_one")
        values = frame[target].to_numpy(float)
        if not np.isfinite(values).all() or values.std() <= 1e-8:
            raise ValueError(f"{directory.name}/{target}: degenerate score vector")
    return frame


def conditional_mrr(frame: pd.DataFrame, score: str) -> pd.Series:
    values = {}
    for query_id, group in frame.groupby("query_id", sort=False):
        # Conditional means only queries whose candidate pool contains gold.
        if not group["label"].eq(1).any():
            continue
        ordered = group.sort_values([score, "retrieval_rank"], ascending=[False, True])
        rank = int(np.flatnonzero(ordered["label"].to_numpy(int) == 1)[0]) + 1
        values[str(query_id)] = 1.0 / rank
    return pd.Series(values, name=score).sort_index()


def bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, reps: int) -> dict[str, float]:
    d = a - b
    n = len(d)
    draws = np.empty(reps)
    for start in range(0, reps, 200):
        width = min(200, reps - start)
        draws[start:start + width] = d[rng.integers(0, n, size=(width, n))].mean(axis=1)
    p = min(1.0, 2 * min((np.count_nonzero(draws <= 0) + 1) / (reps + 1), (np.count_nonzero(draws >= 0) + 1) / (reps + 1)))
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    observed = float(d.mean())
    return {"hef_conditional_mrr_at_100": float(a.mean()), "comparator_conditional_mrr_at_100": float(b.mean()), "mean_paired_difference": observed, "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)), "two_sided_p": p, "cohen_dz": observed / sd if sd else None, "queries": n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    rows = []
    for d_index, dataset in enumerate(DATASETS):
        frame = score_frame(args.ranking_root / dataset)
        methods = {name: conditional_mrr(frame, name) for name in ("hef_rank", *COMPARATORS)}
        common = methods["hef_rank"].index
        for name, values in methods.items():
            if not values.index.equals(common):
                raise ValueError(f"{dataset}/{name}: conditional query mismatch")
        for c_index, comparator in enumerate(COMPARATORS):
            result = bootstrap(methods["hef_rank"].to_numpy(), methods[comparator].to_numpy(), np.random.default_rng(args.seed + d_index * 100 + c_index), args.replicates)
            rows.append({"dataset": dataset, "metric": "conditional_mrr_at_100", "proposed": "hef_rank", "comparator": comparator, "holm_family": "primary_ranking:conditional_mrr_at_100:all_retained_dataset_comparisons", **result})
    adjusted = holm(np.asarray([row["two_sided_p"] for row in rows]))
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "conditional_mrr_at_100_paired_bootstrap.csv", index=False)
    (args.output_dir / "conditional_mrr_at_100_manifest.json").write_text(json.dumps({"status": "complete", "metric": "conditional_mrr_at_100", "bootstrap_unit": "query with gold present in the locked top-100 candidate pool", "replicates": args.replicates, "seed": args.seed, "holm_family": rows[0]["holm_family"], "hypotheses": len(rows), "datasets": list(DATASETS)}, indent=2) + "\n")
    print(f"completed {len(rows)} retained primary hypotheses")


if __name__ == "__main__":
    main()
