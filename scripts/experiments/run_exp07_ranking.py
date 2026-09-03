#!/usr/bin/env python3
"""Experiment 7 reranking robustness under deterministic evidence loss.

The dense top-100 candidate pool is frozen. Evidence is removed consistently at
record level, reranking features are rebuilt, and every system is evaluated on
the same masked pool without retuning on test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from paper1_hef.exp02 import _evaluate, _fields_for_frame, _ranking_metrics, _source
from paper1_hef.features import FIELDS, GENEALOGY_FIELDS, serialize, structured_features

from run_exp07_classification import (
    GENEALOGY_MASK_GROUPS,
    MODEL_ID,
    MODEL_REVISION,
    PRODUCT_MASK_GROUPS,
    OfficialDittoModel,
    choose_ditto,
    ditto_scores,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable_uniform(value: str) -> float:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") / float(2**64)


def pair_candidate_pool(root: Path, config: dict[str, Any], dataset: str) -> pd.DataFrame:
    pool = pd.read_csv(root / "artifacts/exp02/candidate_pools" / dataset / "top100.csv.gz")
    pool = pool.loc[pool["split"].eq("test"), [
        "split", "query_id", "candidate_id", "retrieval_rank", "label"
    ]].reset_index(drop=True)
    pool["query_id"] = pool["query_id"].astype(str)
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    source = _source(root, config, dataset)
    queries, candidates = source["queries"], source["candidates"]
    qfields = [field for field in _fields_for_frame(queries) if field in queries]
    cfields = [field for field in _fields_for_frame(candidates) if field in candidates]
    left = queries[["id", *qfields]].drop_duplicates("id").rename(
        columns={"id": "query_id", **{field: f"left_{field}" for field in qfields}}
    )
    right = candidates[["id", *cfields]].drop_duplicates("id").rename(
        columns={"id": "candidate_id", **{field: f"right_{field}" for field in cfields}}
    )
    left["query_id"] = left["query_id"].astype(str)
    right["candidate_id"] = right["candidate_id"].astype(str)
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    if paired.filter(regex=r"^(left|right)_").isna().all(axis=1).any():
        raise ValueError("Candidate pool contains unresolved record identifiers")
    return paired


def controlled_mask(
    frame: pd.DataFrame, fields: list[str], probability: float, seed: int
) -> tuple[pd.DataFrame, str, int]:
    out = frame.copy()
    selected: list[str] = []
    changed = 0
    for field in fields:
        for side, id_column in (("left", "query_id"), ("right", "candidate_id")):
            column = f"{side}_{field}"
            if column not in out:
                continue
            record_ids = out[id_column].astype(str).to_numpy()
            mask = np.asarray([
                stable_uniform(f"{seed}|{record_id}|{side}|{field}") < probability
                for record_id in record_ids
            ])
            nonempty = out[column].notna().to_numpy() & out[column].astype(str).ne("").to_numpy()
            actual = mask & nonempty
            if np.any(actual):
                out.loc[actual, column] = ""
                changed += int(actual.sum())
                selected.extend(
                    f"{record_id}|{side}|{field}" for record_id, flag in zip(record_ids, actual) if flag
                )
    return out, hashlib.sha256("\n".join(sorted(set(selected))).encode()).hexdigest(), changed


def choose_hef_ranking(root: Path, dataset: str) -> dict[str, Any]:
    ranking = root / "artifacts/exp02/ranking" / dataset
    metrics = json.loads((ranking / "metrics.json").read_text())
    feature_names = list(metrics["features"])
    pool = pd.read_csv(root / "artifacts/exp02/candidate_pools" / dataset / "top100.csv.gz")
    valid = pool.loc[pool["split"].eq("valid")].copy()
    hit_ids = set(valid.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
    conditional = valid.loc[valid["query_id"].isin(hit_ids)]
    choices = []
    for family in ("hef_linear", "hef_gbdt", "hef_rank"):
        models = []
        for seed in (20260725, 20260726, 20260727):
            models.append(joblib.load(ranking / f"{family}_seed_{seed}.joblib"))
        scores = []
        for model in models:
            if hasattr(model, "predict_proba"):
                scores.append(model.predict_proba(valid[feature_names])[:, 1])
            else:
                scores.append(model.predict(valid[feature_names]))
        ensemble = np.mean(scores, axis=0)
        trial = valid.assign(_score=ensemble)
        value = _ranking_metrics(trial.loc[trial["query_id"].isin(hit_ids)], "_score")["mrr"]
        choices.append((float(value), family == "hef_rank", family, models))
    value, _, family, models = max(choices)
    return {
        "family": family, "validation_mrr": value,
        "seeds": [20260725, 20260726, 20260727],
        "models": models, "feature_names": feature_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--ditto-batch-size", type=int, default=256)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    probabilities = [float(x) for x in config["experiments"]["exp07_controlled_field_masking"]["probabilities"]]
    seeds = [int(x) for x in config["protocol"]["seeds"]]
    frame = pair_candidate_pool(root, config, args.dataset)
    genealogy = "left_name" in frame
    all_fields = list(GENEALOGY_FIELDS if genealogy else FIELDS)
    groups = GENEALOGY_MASK_GROUPS if genealogy else PRODUCT_MASK_GROUPS
    scenarios = [("random_fields", 0.0, seeds[0], all_fields)]
    for probability in probabilities:
        if probability == 0:
            continue
        for seed in seeds:
            scenarios.append(("random_fields", probability, seed, all_fields))
            scenarios.extend((name, probability, seed, fields) for name, fields in groups.items())

    device = torch.device(args.device)
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=args.device)
    prefix = next(x["symmetric_prefix"] for x in config["frozen_backbones"] if x["id"] == MODEL_ID)
    hef = choose_hef_ranking(root, args.dataset)
    ditto = choose_ditto(root, args.dataset)
    cross = config["cross_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(cross["id"], revision=cross["revision"])
    ditto_model = OfficialDittoModel(cross["id"], cross["revision"], device).to(device)
    ditto_model.load_state_dict(ditto["checkpoint"]["model"])

    output = root / "artifacts/exp07/ranking" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "metrics.partial.json"
    completed = json.loads(partial.read_text()).get("scenarios", {}) if partial.exists() else {}
    baseline: dict[str, float] | None = None
    started = time.time()
    sizes = [int(x) for x in config["experiments"]["exp02_candidate_ranking"]["k"]]
    for mask_type, probability, seed, fields in scenarios:
        key = f"{mask_type}__p{int(round(probability*100)):02d}__seed{seed}"
        score_path = output / f"scores_{key}.npz"
        if key in completed and score_path.exists() and score_path.stat().st_size:
            if probability == 0:
                baseline = {name: float(value["100"]["conditional"]["mrr"])
                            for name, value in completed[key]["methods"].items()}
            continue
        masked, mask_sha256, cells_masked = controlled_mask(frame, fields, probability, seed)
        structured = structured_features(masked)
        left_text = [prefix + x for x in serialize(masked, "left").tolist()]
        right_text = [prefix + x for x in serialize(masked, "right").tolist()]
        left_vec = embedding_model.encode(left_text, batch_size=args.embedding_batch_size,
                                          normalize_embeddings=True, show_progress_bar=False)
        right_vec = embedding_model.encode(right_text, batch_size=args.embedding_batch_size,
                                           normalize_embeddings=True, show_progress_bar=False)
        embedding_score = np.sum(left_vec * right_vec, axis=1)
        feature_frame = structured.copy()
        feature_frame["embedding_score"] = embedding_score
        hef_scores = []
        for model in hef["models"]:
            if hasattr(model, "predict_proba"):
                hef_scores.append(model.predict_proba(feature_frame[hef["feature_names"]])[:, 1])
            else:
                hef_scores.append(model.predict(feature_frame[hef["feature_names"]]))
        best_hef = np.mean(hef_scores, axis=0)
        official_ditto = ditto_scores(masked, ditto_model, tokenizer, device,
                                      int(cross["max_length"]), args.ditto_batch_size)
        scored = masked[["query_id", "candidate_id", "retrieval_rank", "label"]].copy()
        raw_scores = {
            "rules": structured["rule_score"].to_numpy(dtype=float),
            "frozen_embedding": embedding_score,
            "official_ditto": official_ditto,
            "best_hef": best_hef,
        }
        methods = {name: _evaluate(scored.assign(_score=values), "_score", sizes)
                   for name, values in raw_scores.items()}
        if probability == 0:
            baseline = {name: float(value["100"]["conditional"]["mrr"])
                        for name, value in methods.items()}
        if baseline is None:
            raise RuntimeError("Unmasked baseline must complete first")
        for name, value in methods.items():
            value["mrr_retention_at_100"] = (
                float(value["100"]["conditional"]["mrr"] / baseline[name]) if baseline[name] else None
            )
        completed[key] = {
            "mask_type": mask_type, "probability": probability, "seed": seed,
            "fields": fields, "mask_sha256": mask_sha256, "cells_masked": cells_masked,
            "rows": len(masked), "queries": int(masked["query_id"].nunique()), "methods": methods,
        }
        np.savez_compressed(score_path, query_id=masked["query_id"].astype(str).to_numpy(),
                            candidate_id=masked["candidate_id"].astype(str).to_numpy(),
                            label=masked["label"].to_numpy(dtype=np.int8),
                            **{name: values.astype(np.float32) for name, values in raw_scores.items()})
        write_json(partial, {"status": "running", "dataset": args.dataset, "scenarios": completed})

    result = {
        "experiment": "exp07_controlled_evidence_loss", "component": "ranking",
        "status": "complete", "dataset": args.dataset,
        "protocol": {
            "candidate_pool": "frozen_unmasked_top100", "probabilities": probabilities,
            "seeds": seeds, "matched_record_level_mask_across_systems": True,
            "selection": "clean_validation_only", "test_policy": "no retuning after masking",
            "causal_claim": False,
        },
        "mask_groups": groups,
        "selected_hef": {k: v for k, v in hef.items() if k not in {"models", "feature_names"}},
        "selected_ditto": {k: v for k, v in ditto.items() if k != "checkpoint"},
        "scenarios": completed, "runtime_seconds": time.time() - started,
    }
    write_json(output / "metrics.json", result)
    write_json(output / "manifest.json", {
        "status": "verified_local", "dataset": args.dataset,
        "expected_scenarios": len(scenarios), "completed_scenarios": len(completed),
        "score_files": len(list(output.glob("scores_*.npz"))),
    })


if __name__ == "__main__":
    main()
