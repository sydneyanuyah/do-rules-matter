#!/usr/bin/env python3
"""One-seed Experiment 6 ablation for classic and RoBERTa-OOF GBDT HEF."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from exp06_common import CONDITIONS, DATASETS, SEED, selected_hybrid_features, structured_groups
from paper1_hef.exp02 import _evaluate


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool(root: Path, dataset: str) -> tuple[pd.DataFrame, Path]:
    directory = root / "artifacts" / "exp02" / "candidate_pools" / dataset
    path, manifest = directory / "top100.csv.gz", directory / "manifest.json"
    if not path.exists() or not manifest.exists():
        raise FileNotFoundError(f"Locked candidate pool incomplete: {directory}")
    frame = pd.read_csv(path)
    frame["query_id"] = frame["query_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.duplicated(["split", "query_id", "candidate_id"]).any():
        raise ValueError("Duplicate candidate keys")
    return frame, manifest


def attach_roberta_oof(root: Path, dataset: str, pool: pd.DataFrame) -> pd.DataFrame:
    source = (
        root / "artifacts" / "exp02_hef_cross_evidence" / "v1" / dataset
        / "e5_plus_tuned_roberta" / f"seed_{SEED}"
    )
    required = [source / "SUCCESS.json", source / "oof_train_scores.npz", source / "scores.npz"]
    if any(not path.exists() or path.stat().st_size == 0 for path in required):
        raise FileNotFoundError(f"RoBERTa OOF prerequisite incomplete: {source}")
    train_hit = set(
        pool.loc[pool["split"].eq("train")].groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index
    )
    fit_train = pool.loc[pool["split"].eq("train") & pool["query_id"].isin(train_hit)].copy()
    # Canonical OOF bundles store string identifiers as NumPy object arrays.
    # They are internal, checksum-tracked artifacts; labels and row identity are
    # still revalidated below before any score is used.
    train_npz = np.load(source / "oof_train_scores.npz", allow_pickle=True)
    train_scores = pd.DataFrame({
        "query_id": train_npz["query_id"].astype(str),
        "candidate_id": train_npz["candidate_id"].astype(str),
        "label_check": train_npz["label"].astype(np.int8),
        "tuned_roberta_score": train_npz["score"].astype(float),
    })
    if train_scores.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate RoBERTa OOF keys")
    joined_train = fit_train.merge(train_scores, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
    if joined_train["tuned_roberta_score"].isna().any() or not np.array_equal(
        joined_train["label"].to_numpy(np.int8), joined_train["label_check"].to_numpy(np.int8)
    ):
        raise ValueError("RoBERTa OOF train alignment failed")
    scored = np.load(source / "scores.npz", allow_pickle=True)
    parts = []
    for split in ("valid", "test"):
        values = pd.DataFrame({
            "query_id": scored[f"{split}_query_id"].astype(str),
            "candidate_id": scored[f"{split}_candidate_id"].astype(str),
            "label_check": scored[f"{split}_label"].astype(np.int8),
            "tuned_roberta_score": scored[f"{split}_roberta_score"].astype(float),
        })
        base = pool.loc[pool["split"].eq(split)].copy()
        merged = base.merge(values, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
        if merged["tuned_roberta_score"].isna().any() or not np.array_equal(
            merged["label"].to_numpy(np.int8), merged["label_check"].to_numpy(np.int8)
        ):
            raise ValueError(f"RoBERTa {split} alignment failed")
        parts.append(merged.drop(columns="label_check"))
    joined_train = joined_train.drop(columns="label_check")
    return pd.concat([joined_train, *parts], ignore_index=True)


def run(args: argparse.Namespace) -> Path:
    root = args.project_root.resolve()
    if args.dataset not in DATASETS or args.condition not in CONDITIONS:
        raise ValueError("Dataset or condition is outside the locked Experiment 6 matrix")
    if args.model not in {"hef_gbdt_e5", "hef_gbdt_e5_tuned_roberta_oof"}:
        raise ValueError("Unsupported hybrid model")
    output = root / "artifacts" / "exp06_rerun_v2" / args.model / args.condition / args.dataset / f"seed_{SEED}"
    if (output / "SUCCESS.json").exists():
        return output
    lock = output.parent / f".{output.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"Collision lock exists: {lock}") from exc
    stage = output.parent / f".{output.name}.partial-{os.getpid()}"
    started = time.time()
    try:
        stage.mkdir(parents=True, exist_ok=False)
        pool, pool_manifest = load_pool(root, args.dataset)
        if args.model.endswith("tuned_roberta_oof"):
            pool = attach_roberta_oof(root, args.dataset, pool)
        excluded = {
            "split", "query_id", "candidate_id", "retrieval_rank", "label",
            "left_text", "right_text", "tuned_roberta_score",
        }
        structured = [column for column in pool.columns if column not in excluded]
        groups = structured_groups([column for column in structured if column != "embedding_score"])
        structured_order = [column for names in groups.values() for column in names]
        semantics = ["embedding_score"]
        if args.model.endswith("tuned_roberta_oof"):
            semantics.append("tuned_roberta_score")
        features = selected_hybrid_features(structured_order, args.condition, semantics)
        train = pool.loc[pool["split"].eq("train")].copy()
        valid = pool.loc[pool["split"].eq("valid")].copy()
        test = pool.loc[pool["split"].eq("test")].copy()
        train_hit = set(train.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
        fit_train = train.loc[train["query_id"].isin(train_hit)].copy()
        if any(frame[features].isna().any().any() for frame in (fit_train, valid, test)):
            raise ValueError("Ablation feature matrix contains nulls")
        config = yaml.safe_load((root / "configs" / "experiment.yaml").read_text())
        sizes = [int(k) for k in config["experiments"]["exp02_candidate_ranking"]["k"]]
        candidates = []
        for leaves in (7, 15, 31):
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=leaves,
                l2_regularization=1.0, early_stopping=True, random_state=SEED,
            )
            model.fit(fit_train[features], fit_train["label"])
            score = model.predict_proba(valid[features])[:, 1]
            metrics = _evaluate(valid.assign(_score=score), "_score", sizes)
            selection = float(metrics["100"]["conditional"]["mrr"])
            candidates.append((selection, -leaves, leaves, model, score, metrics))
        _, _, leaves, model, valid_score, valid_metrics = max(candidates, key=lambda x: (x[0], x[1]))
        test_score = model.predict_proba(test[features])[:, 1]
        if not np.isfinite(test_score).all():
            raise ValueError("Nonfinite test scores")
        test_metrics = _evaluate(test.assign(_score=test_score), "_score", sizes)
        joblib.dump(model, stage / "model.joblib")
        np.savez_compressed(
            stage / "scores.npz",
            valid_query_id=valid["query_id"].astype(str).to_numpy(),
            valid_candidate_id=valid["candidate_id"].astype(str).to_numpy(),
            valid_label=valid["label"].to_numpy(np.int8), valid_score=valid_score.astype(np.float32),
            test_query_id=test["query_id"].astype(str).to_numpy(),
            test_candidate_id=test["candidate_id"].astype(str).to_numpy(),
            test_label=test["label"].to_numpy(np.int8), test_score=test_score.astype(np.float32),
        )
        write_json(stage / "metrics.json", {
            "experiment": "exp06_public_evidence_ablation_rerun_v2",
            "model": args.model, "condition": args.condition, "dataset": args.dataset,
            "seed": SEED, "features": features, "selected_max_leaf_nodes": leaves,
            "selection_metric": "validation_conditional_mrr_at_100",
            "validation": valid_metrics, "test": test_metrics,
            "score_integrity": {
                "valid_unique_values": int(np.unique(valid_score).size),
                "test_unique_values": int(np.unique(test_score).size),
                "degenerate_test_scores": bool(np.unique(test_score).size < 2),
            },
            "test_policy": "untouched until validation capacity selection was locked",
            "runtime_seconds": time.time() - started, "status": "complete",
        })
        write_json(stage / "manifest.json", {
            "pool_sha256": sha256(pool_manifest), "rows": {"train": len(fit_train), "valid": len(valid), "test": len(test)},
            "paper_eligible": True, "seed_policy": "single locked screening seed; no variance claim",
        })
        write_json(stage / "SUCCESS.json", {"validated": True})
        if output.exists():
            raise FileExistsError(output)
        os.replace(stage, output)
        return output
    finally:
        if stage.exists(): shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse()))
