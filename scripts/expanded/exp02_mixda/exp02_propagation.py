#!/usr/bin/env python3
"""Propagate completed Exp1 models to fixed E5 Experiment-2 top-100 pools.

Two scientifically distinct modes are supported:

* ``mixda-hef`` applies a completed HEF-GBDT + training-only MixDA model.
  The E5 candidate pool stays fixed; candidate-pair cosine evidence is
  recomputed with the requested HEF backbone before applying the Exp1 model.
* ``official-ditto`` applies a completed Official Ditto checkpoint and, when
  present, its leakage-safe Exp1 HEF-GBDT E5 + Ditto OOF fusion model.

Neither mode fits or selects anything on Exp2 test queries.  Model/checkpoint
selection occurred on the locked Exp1 validation split.  Ranking is reported
on the fixed label-blind E5 pool, including conditional and end-to-end Hits@100.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from paper1_hef.exp02 import _evaluate, _fields_for_frame, _source
from paper1_hef.features import serialize, structured_features


E5_ID = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
DITTO_COMMIT = "52985564a93fb11308439516d3e17a033d43ec8f"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pool_and_pairs(
    root: Path, config: dict[str, Any], dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    pool_dir = root / "artifacts" / "exp02" / "candidate_pools" / dataset
    pool_path = pool_dir / "top100.csv.gz"
    manifest_path = pool_dir / "manifest.json"
    if not pool_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Fixed E5 pool is absent: {pool_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_id") != E5_ID or manifest.get("revision") != E5_REVISION:
        raise ValueError("Candidate pool is not the locked E5 retrieval pool")
    if not manifest.get("labels_joined_after_top100_frozen"):
        raise ValueError("Candidate pool was not frozen label-blind")
    pool = pd.read_csv(pool_path)
    required = {
        "split", "query_id", "candidate_id", "retrieval_rank",
        "embedding_score", "label",
    }
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Candidate pool missing columns: {sorted(missing)}")
    if set(pool["split"].astype(str)) != {"train", "valid", "test"}:
        raise ValueError("Candidate pool must contain train/valid/test")
    if pool.duplicated(["split", "query_id", "candidate_id"]).any():
        raise ValueError("Duplicate pool candidate rows")
    if not set(pool["label"].unique()) <= {0, 1}:
        raise ValueError("Nonbinary pool labels")
    for (_split, _query), group in pool.groupby(["split", "query_id"], sort=False):
        ranks = group["retrieval_rank"].to_numpy(dtype=int)
        if not np.array_equal(ranks, np.arange(1, len(group) + 1)):
            raise ValueError("Pool retrieval ranks are not contiguous")

    source = _source(root, config, dataset)
    queries, candidates = source["queries"], source["candidates"]
    query_fields = [column for column in _fields_for_frame(queries) if column in queries]
    candidate_fields = [
        column for column in _fields_for_frame(candidates) if column in candidates
    ]
    left = queries[["id", *query_fields]].rename(
        columns={"id": "query_id", **{field: f"left_{field}" for field in query_fields}}
    )
    right = candidates[["id", *candidate_fields]].rename(
        columns={
            "id": "candidate_id",
            **{field: f"right_{field}" for field in candidate_fields},
        }
    )
    pool["query_id"] = pool["query_id"].astype(str)
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    left["query_id"] = left["query_id"].astype(str)
    right["candidate_id"] = right["candidate_id"].astype(str)
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    if len(paired) != len(pool):
        raise AssertionError("Pool-to-record merge changed row coverage")
    return pool, paired, pool_path, manifest_path


def adaptive_encode(model: Any, texts: list[str], batch_size: int) -> tuple[np.ndarray, int]:
    import torch

    current = batch_size
    while True:
        try:
            values = model.encode(
                texts,
                batch_size=current,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            return np.asarray(values), current
        except torch.cuda.OutOfMemoryError:
            if current <= 4:
                raise
            current = max(4, current // 2)
            torch.cuda.empty_cache()


def serialize_pairs(paired: pd.DataFrame, side: str) -> list[str]:
    """Serialize product or genealogy evidence without dropping domain fields."""
    values = serialize(paired, side).fillna("").astype(str).tolist()
    if any(value.strip() for value in values):
        return values
    prefix = f"{side}_"
    fields = [column[len(prefix):] for column in paired.columns if column.startswith(prefix)]
    return [
        " ".join(
            f"COL {field} VAL {row.get(prefix + field)}"
            for field in fields
            if pd.notna(row.get(prefix + field)) and str(row.get(prefix + field)).strip()
        )
        for _, row in paired.iterrows()
    ]


def backbone_pair_scores(
    paired: pd.DataFrame,
    model_id: str,
    revision: str,
    model_spec: dict[str, Any],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_id == E5_ID:
        # The fixed pool already contains the exact normalized E5 dot product.
        values = paired["embedding_score"].to_numpy(dtype=float)
        return values, {"source": "fixed_e5_candidate_pool", "batch_size": None}
    os.environ.setdefault("USE_TF", "0")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_id,
        revision=revision,
        device=device,
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
    )
    prefix = str(model_spec["symmetric_prefix"])
    left_text = [prefix + value for value in serialize_pairs(paired, "left")]
    right_text = [prefix + value for value in serialize_pairs(paired, "right")]
    left, used_left = adaptive_encode(model, left_text, batch_size)
    right, used_right = adaptive_encode(model, right_text, batch_size)
    scores = np.sum(left * right, axis=1).astype(float)
    return scores, {
        "source": "candidate_pairs_encoded_on_fixed_e5_pool",
        "batch_size_left": used_left,
        "batch_size_right": used_right,
    }


def score_record(
    pool: pd.DataFrame, scores: np.ndarray, sizes: list[int]
) -> dict[str, Any]:
    if len(scores) != len(pool) or not np.isfinite(scores).all():
        raise ValueError("Score coverage is incomplete or nonfinite")
    if np.ptp(scores) <= 1e-8:
        raise ValueError("Scores are degenerate")
    result: dict[str, Any] = {}
    for split in ("valid", "test"):
        frame = pool.loc[pool["split"].eq(split)].copy()
        frame["model_score"] = scores[pool["split"].eq(split)]
        result[split] = _evaluate(frame, "model_score", sizes)
    return result


def save_scores(path: Path, pool: pd.DataFrame, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        split=pool["split"].astype(str).to_numpy(dtype=str),
        query_id=pool["query_id"].astype(str).to_numpy(dtype=str),
        candidate_id=pool["candidate_id"].astype(str).to_numpy(dtype=str),
        retrieval_rank=pool["retrieval_rank"].to_numpy(dtype=np.int16),
        label=pool["label"].to_numpy(dtype=np.int8),
        score=scores.astype(np.float32),
    )


def run_mixda_hef(args: argparse.Namespace, root: Path, config: dict[str, Any]) -> Path:
    started = time.time()
    model_spec = next(
        item for item in config["frozen_backbones"] if item["id"] == args.model_id
    )
    if str(model_spec["revision"]) != args.revision:
        raise ValueError("Backbone revision differs from the locked configuration")
    pool, paired, pool_path, manifest_path = load_pool_and_pairs(root, config, args.dataset)
    evidence, encode_report = backbone_pair_scores(
        paired, args.model_id, args.revision, model_spec, args.device, args.batch_size
    )
    features = structured_features(paired)
    features["embedding_score"] = evidence
    model_dir = (
        root / "artifacts" / "exp01_hef_mixda" / args.dataset
        / args.model_id.replace("/", "__") / args.revision / f"seed_{args.seed}"
    )
    model_path = model_dir / "model.joblib"
    metric_path = model_dir / "metrics.json"
    if not model_path.exists() or not metric_path.exists():
        raise FileNotFoundError(f"Completed Exp1 MixDA HEF model absent: {model_dir}")
    exp1 = json.loads(metric_path.read_text())
    if exp1.get("seed") != args.seed or exp1.get("dataset") != args.dataset:
        raise ValueError("Exp1 MixDA model metadata is misaligned")
    if exp1.get("augmentation", {}).get("scope") != "training_only":
        raise ValueError("Exp1 MixDA model does not have training-only augmentation")
    model = joblib.load(model_path)
    if list(getattr(model, "feature_names_in_", [])) != list(features.columns):
        raise ValueError("Exp1 HEF feature schema does not match candidate-pair features")
    scores = model.predict_proba(features)[:, 1]
    sizes = [int(value) for value in config["experiments"]["exp02_candidate_ranking"]["k"]]
    metrics = score_record(pool, scores, sizes)
    output = (
        root / "artifacts" / "exp02_propagated" / "hef_gbdt_mixda"
        / args.dataset / args.model_id.replace("/", "__") / args.revision
        / f"seed_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    save_scores(output / "scores.npz", pool, scores)
    write_json(
        output / "metrics.json",
        {
            "experiment": "exp02_candidate_ranking_propagated",
            "method": "hef_gbdt_mixda",
            "label": f"HEF-GBDT + MixDA-style field augmentation with {args.model_id}",
            "dataset": args.dataset,
            "seed": args.seed,
            "model_id": args.model_id,
            "revision": args.revision,
            "retrieval_pool": "fixed_label_blind_e5_top100",
            "candidate_recall_reported_separately": True,
            "model_selection": "completed Exp1 model; no Exp2 test selection or fitting",
            "validation": metrics["valid"],
            "test": metrics["test"],
            "encoding": encode_report,
            "runtime_seconds": time.time() - started,
        },
    )
    write_json(
        output / "run_manifest.json",
        {
            "paper_eligible": True,
            "dataset": args.dataset,
            "seed": args.seed,
            "model_id": args.model_id,
            "revision": args.revision,
            "candidate_pool": str(pool_path.relative_to(root)),
            "candidate_pool_sha256": sha256(pool_path),
            "candidate_pool_manifest_sha256": sha256(manifest_path),
            "source_model": str(model_path.relative_to(root)),
            "source_model_sha256": sha256(model_path),
            "selection_split": "Exp1 validation only",
            "test_policy": "fixed model applied once to fixed Exp2 test pool",
        },
    )
    return output


def import_official_compat(root: Path) -> Any:
    candidates = [
        root / "scripts" / "run_ditto_official_compat.py",
        root / "run_ditto_official_compat.py",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError("Pinned Official Ditto compatibility runner absent")
    spec = importlib.util.spec_from_file_location("official_ditto_compat_exp02", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.UPSTREAM_COMMIT != DITTO_COMMIT:
        raise ValueError("Official Ditto upstream commit differs from lock")
    return module


def ditto_scores(
    paired: pd.DataFrame,
    checkpoint: Path,
    compat: Any,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("upstream_commit") != DITTO_COMMIT:
        raise ValueError("Checkpoint is not the locked Official Ditto reproduction")
    model_id, revision = str(payload["model_id"]), str(payload["revision"])
    target = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = compat.OfficialDittoModel(model_id, revision, target, 0.8).to(target)
    model.load_state_dict(payload["model"])
    model.eval()
    left = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize_pairs(paired, "left")]
    right = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize_pairs(paired, "right")]
    output = np.empty(len(paired), dtype=np.float32)
    current = batch_size
    start = 0
    while start < len(paired):
        end = min(len(paired), start + current)
        try:
            # Match official DittoDataset + pad exactly: tokenizer.encode for
            # each pair, followed by pad_sequence's default padding value 0.
            sequences = [
                torch.tensor(
                    tokenizer.encode(a, text_pair=b, max_length=max_length, truncation=True),
                    dtype=torch.long,
                )
                for a, b in zip(left[start:end], right[start:end], strict=True)
            ]
            encoded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            with torch.inference_mode(), torch.autocast(
                device_type=target.type, dtype=torch.float16, enabled=target.type == "cuda"
            ):
                logits = model(encoded)
            output[start:end] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            start = end
        except torch.cuda.OutOfMemoryError:
            if current <= 1:
                raise
            current = max(1, current // 2)
            torch.cuda.empty_cache()
    return output, {
        "model_id": model_id,
        "revision": revision,
        "selected_epoch": int(payload["selected_epoch"]),
        "batch_size_used": current,
        "max_length": max_length,
    }


def run_official_ditto(
    args: argparse.Namespace, root: Path, config: dict[str, Any]
) -> Path:
    started = time.time()
    pool, paired, pool_path, manifest_path = load_pool_and_pairs(root, config, args.dataset)
    source_dir = (
        root / "artifacts" / "exp01_ditto_official" / args.variant
        / args.dataset / f"seed_{args.seed}"
    )
    checkpoint = source_dir / "model.pt"
    metrics_path = source_dir / "metrics.json"
    if not checkpoint.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Completed Official Ditto source absent: {source_dir}")
    source_metrics = json.loads(metrics_path.read_text())
    if not source_metrics.get("official_ditto_reproduction"):
        raise ValueError("Source checkpoint is not an Official Ditto reproduction")
    if source_metrics.get("upstream_commit") != DITTO_COMMIT:
        raise ValueError("Source checkpoint has wrong Official Ditto commit")
    compat = import_official_compat(root)
    standalone, inference = ditto_scores(
        paired,
        checkpoint,
        compat,
        args.device,
        args.batch_size,
        int(source_metrics["max_length"]),
    )
    sizes = [int(value) for value in config["experiments"]["exp02_candidate_ranking"]["k"]]
    standalone_metrics = score_record(pool, standalone, sizes)
    method = f"official_ditto_{args.variant}"
    output = (
        root / "artifacts" / "exp02_propagated" / method
        / args.dataset / f"seed_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    save_scores(output / "scores.npz", pool, standalone)
    result: dict[str, Any] = {
        "experiment": "exp02_candidate_ranking_propagated",
        "method": method,
        "official_ditto_reproduction": True,
        "upstream_commit": DITTO_COMMIT,
        "dataset": args.dataset,
        "seed": args.seed,
        "variant": args.variant,
        "retrieval_pool": "fixed_label_blind_e5_top100",
        "candidate_recall_reported_separately": True,
        "model_selection": "completed Exp1 validation-selected checkpoint; no Exp2 test selection or fitting",
        "standalone": {
            "validation": standalone_metrics["valid"],
            "test": standalone_metrics["test"],
        },
        "inference": inference,
        "runtime_seconds": time.time() - started,
    }

    fusion_dir = (
        root / "artifacts" / "exp01_hef_cross_evidence" / "v1" / args.dataset
        / E5_ID.replace("/", "__") / E5_REVISION / method / f"seed_{args.seed}"
    )
    fusion_model_path = fusion_dir / "hef_gbdt.joblib"
    if fusion_model_path.exists():
        fusion_model = joblib.load(fusion_model_path)
        features = structured_features(paired)
        features["embedding_score"] = pool["embedding_score"].to_numpy(dtype=float)
        features["ditto_score"] = standalone
        if list(getattr(fusion_model, "feature_names_in_", [])) != list(features.columns):
            raise ValueError("Exp1 Official Ditto fusion feature schema mismatch")
        fusion_score = fusion_model.predict_proba(features)[:, 1]
        fusion_metrics = score_record(pool, fusion_score, sizes)
        save_scores(output / "fusion_scores.npz", pool, fusion_score)
        result["hef_gbdt_e5_fusion"] = {
            "method": f"hef_gbdt_e5_plus_{method}",
            "source_training_protocol": "record-grouped 3-fold OOF Ditto training scores",
            "validation": fusion_metrics["valid"],
            "test": fusion_metrics["test"],
        }
    else:
        result["hef_gbdt_e5_fusion"] = {
            "status": "not_scored",
            "blocker": "leakage-safe completed Exp1 OOF fusion model absent",
            "expected": str(fusion_model_path.relative_to(root)),
        }
    write_json(output / "metrics.json", result)
    write_json(
        output / "run_manifest.json",
        {
            "paper_eligible": True,
            "dataset": args.dataset,
            "seed": args.seed,
            "variant": args.variant,
            "candidate_pool": str(pool_path.relative_to(root)),
            "candidate_pool_sha256": sha256(pool_path),
            "candidate_pool_manifest_sha256": sha256(manifest_path),
            "source_checkpoint": str(checkpoint.relative_to(root)),
            "source_checkpoint_sha256": sha256(checkpoint),
            "source_fusion_model": (
                str(fusion_model_path.relative_to(root)) if fusion_model_path.exists() else None
            ),
            "selection_split": "Exp1 validation only",
            "test_policy": "fixed checkpoints applied once to fixed Exp2 test pool",
            "python": sys.version,
            "platform": platform.platform(),
        },
    )
    return output


def validate_output(root: Path, output: Path, dataset: str) -> None:
    metric = output / "metrics.json"
    manifest = output / "run_manifest.json"
    scores = output / "scores.npz"
    for path in (metric, manifest, scores):
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty artifact: {path}")
    payload = np.load(scores, allow_pickle=False)
    pool = pd.read_csv(root / "artifacts" / "exp02" / "candidate_pools" / dataset / "top100.csv.gz")
    for key, expected in {
        "split": pool["split"].astype(str).to_numpy(),
        "query_id": pool["query_id"].astype(str).to_numpy(),
        "candidate_id": pool["candidate_id"].astype(str).to_numpy(),
        "retrieval_rank": pool["retrieval_rank"].to_numpy(dtype=np.int16),
        "label": pool["label"].to_numpy(dtype=np.int8),
    }.items():
        if not np.array_equal(payload[key].astype(expected.dtype), expected):
            raise AssertionError(f"{key} is not exactly pool-aligned")
    values = payload["score"].astype(float)
    if len(values) != len(pool) or not np.isfinite(values).all() or np.ptp(values) <= 1e-8:
        raise AssertionError("Primary scores fail coverage/nondegeneracy")
    metrics = json.loads(metric.read_text())
    if metrics.get("dataset") != dataset:
        raise AssertionError("Metric dataset mismatch")
    fusion = metrics.get("hef_gbdt_e5_fusion")
    if isinstance(fusion, dict) and "method" in fusion:
        fusion_path = output / "fusion_scores.npz"
        if not fusion_path.exists() or fusion_path.stat().st_size == 0:
            raise AssertionError("Fusion metric exists without fusion scores")
        fusion_payload = np.load(fusion_path, allow_pickle=False)
        for key in ("split", "query_id", "candidate_id", "retrieval_rank", "label"):
            if not np.array_equal(fusion_payload[key], payload[key]):
                raise AssertionError(f"Fusion {key} is not pool-aligned")
        fusion_values = fusion_payload["score"].astype(float)
        if (
            len(fusion_values) != len(pool)
            or not np.isfinite(fusion_values).all()
            or np.ptp(fusion_values) <= 1e-8
        ):
            raise AssertionError("Fusion scores fail coverage/nondegeneracy")
    metric_text = metric.read_text()
    if '"hits_at_100"' not in metric_text or '"hits_at_100_end_to_end"' not in metric_text:
        raise AssertionError("Hits@100 is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("mixda-hef", "official-ditto"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--variant", choices=("plain", "mixda_all"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs" / "experiment.yaml").read_text())
    if args.dataset not in (set(config["dataset_groups"]["exp01_all"]) | {"link_lives_release2"}):
        raise ValueError("Dataset is outside the locked Experiment-1/2 scope")
    if args.seed not in {int(value) for value in config["protocol"]["seeds"]}:
        raise ValueError("Seed is outside the locked protocol")
    if args.mode == "mixda-hef":
        if not args.model_id or not args.revision:
            raise ValueError("mixda-hef requires --model-id and --revision")
        output = run_mixda_hef(args, root, config)
    else:
        if not args.variant:
            raise ValueError("official-ditto requires --variant")
        output = run_official_ditto(args, root, config)
    validate_output(root, output, args.dataset)
    print(output)


if __name__ == "__main__":
    main()
