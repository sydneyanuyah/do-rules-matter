#!/usr/bin/env python3
"""Validate the 60-lane Exp7 matrix and register Exp9 public hypotheses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = (
    "jina_finetuned", "roberta_finetuned", "official_ditto_plain", "official_ditto_mixda",
    "hef_linear_bert", "hef_linear_bge", "hef_gbdt_e5_roberta_oof",
    "hef_gbdt_e5_jina_oof", "joint_neural_hef_roberta", "joint_neural_hef_jina",
)
DATASETS = (
    "abt_buy", "amazon_google", "walmart_amazon", "wdc_80_medium_seen",
    "wdc_80_medium_unseen", "link_lives_release2",
)
SCENARIOS = ((0.0, 20260725),) + tuple(
    (probability, seed) for probability in (0.1, 0.3, 0.5, 0.7)
    for seed in (20260725, 20260726, 20260727)
)
COMPARISONS = (
    ("jina_finetuned", "hef_linear_bert", "cross_family"),
    ("jina_finetuned", "hef_gbdt_e5_roberta_oof", "cross_family"),
    ("jina_finetuned", "joint_neural_hef_roberta", "cross_family"),
    ("hef_linear_bert", "hef_gbdt_e5_roberta_oof", "cross_family"),
    ("hef_linear_bert", "joint_neural_hef_roberta", "cross_family"),
    ("hef_gbdt_e5_roberta_oof", "joint_neural_hef_roberta", "cross_family"),
    ("jina_finetuned", "roberta_finetuned", "within_standalone"),
    ("official_ditto_plain", "official_ditto_mixda", "within_standalone"),
    ("hef_linear_bert", "hef_linear_bge", "within_linear"),
    ("hef_gbdt_e5_roberta_oof", "hef_gbdt_e5_jina_oof", "within_gbdt"),
    ("joint_neural_hef_roberta", "joint_neural_hef_jina", "within_joint"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    args = parser.parse_args()
    reference: dict[tuple[str, str], tuple[str, ...]] = {}
    cells = 0
    for model in MODELS:
        for dataset in DATASETS:
            if not (args.root / model / dataset / "SUCCESS.json").is_file():
                raise ValueError(f"Missing lane completion: {model}/{dataset}")
            for probability, seed in SCENARIOS:
                suffix = f"p{round(probability * 100):02d}_seed{seed}"
                path = args.root / model / dataset / suffix / "per_query.parquet"
                frame = pd.read_parquet(path)
                required = {"query_id", "reciprocal_rank", "hit_at_1", "hit_at_100", "rank"}
                if required - set(frame) or frame["query_id"].duplicated().any() or frame.empty:
                    raise ValueError(f"Invalid per-query artifact: {path}")
                ids = tuple(frame["query_id"].astype(str))
                key = dataset, suffix
                if key in reference and reference[key] != ids:
                    raise ValueError(f"Query alignment mismatch: {model}/{dataset}/{suffix}")
                reference[key] = ids; cells += 1
    rows = []
    for dataset in DATASETS:
        for probability, seed in SCENARIOS:
            for left, right, comparison_type in COMPARISONS:
                for metric in ("reciprocal_rank", "hit_at_1"):
                    rows.append({
                        "family": f"public__{dataset}__p{round(probability*100):02d}__seed{seed}__{metric}",
                        "dataset": dataset, "mask_probability": probability, "mask_seed": seed,
                        "left_model": left, "right_model": right, "metric": metric,
                        "comparison_type": comparison_type,
                    })
    pd.DataFrame(rows).to_csv(args.comparisons, index=False)
    (args.root / "FINALIZED.json").write_text(json.dumps({
        "validated": True, "model_dataset_lanes": 60, "model_scenarios": cells,
        "models": len(MODELS), "datasets": len(DATASETS), "scenarios_per_lane": len(SCENARIOS),
        "registered_hypotheses": len(rows), "holm_family": "dataset x scenario x metric",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
