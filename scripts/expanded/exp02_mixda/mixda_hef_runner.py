#!/usr/bin/env python3
"""Training-only MixDA-style augmentation for HEF-GBDT.

This is an HEF adaptation of Ditto's MixDA text operators, not an official
Ditto reproduction.  Each protocol seed creates exactly one augmented copy of
every training pair.  Structured and dense features are recomputed from that
copy.  Validation and test records are never augmented or used for fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from paper1_hef.data import load_dataset, validate_splits
from paper1_hef.ditto_style import _seeded_int, ditto_text, mixda_augment
from paper1_hef.evaluate import classification_metrics, select_threshold
from paper1_hef.exp01 import _load_embedding_scores
from paper1_hef.features import FIELDS, GENEALOGY_FIELDS, serialize, structured_features


METHOD = "hef_gbdt_mixda"
METHOD_LABEL = "HEF-GBDT + MixDA-style field augmentation"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_col_val(text: str, allowed_fields: set[str]) -> tuple[dict[str, str], dict[str, int]]:
    """Parse Ditto COL/VAL text and retain only schema-compatible attributes."""
    tokens = text.split()
    values: dict[str, str] = {}
    diagnostics = {"unknown_fields": 0, "malformed_chunks": 0, "duplicate_fields": 0}
    cursor = 0
    while cursor < len(tokens):
        try:
            start = tokens.index("[COL]", cursor)
        except ValueError:
            break
        if start + 2 >= len(tokens) or tokens[start + 2] != "[VAL]":
            diagnostics["malformed_chunks"] += 1
            cursor = start + 1
            continue
        field = tokens[start + 1]
        end = start + 3
        while end < len(tokens) and tokens[end] != "[COL]":
            end += 1
        value = " ".join(tokens[start + 3 : end]).strip()
        if field not in allowed_fields:
            diagnostics["unknown_fields"] += 1
        elif value:
            if field in values:
                diagnostics["duplicate_fields"] += 1
                values[field] = f"{values[field]} {value}".strip()
            else:
                values[field] = value
        cursor = end
    return values, diagnostics


def augment_training_frame(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one deterministic MixDA copy per training row.

    A malformed augmented side falls back to its original side, keeping row
    coverage exact while making the issue count auditable.
    """
    fields = GENEALOGY_FIELDS if "left_name" in frame else FIELDS
    allowed = set(fields)
    left_original = ditto_text(frame, "left")
    right_original = ditto_text(frame, "right")
    rows: list[pd.Series] = []
    diagnostics: dict[str, Any] = {
        "rows": int(len(frame)),
        "side_fallbacks": 0,
        "swapped_rows": 0,
        "unknown_fields": 0,
        "malformed_chunks": 0,
        "duplicate_fields": 0,
        "affected_pair_id_sample": [],
    }
    for position, (_, source) in enumerate(frame.iterrows()):
        pair_id = str(source["pair_id"])
        augmentation_seed = _seeded_int(seed, pair_id, position)
        augmented_left, augmented_right = mixda_augment(
            left_original[position], right_original[position], augmentation_seed
        )
        # mixda_augment's first random draw decides whether records are swapped.
        swapped = random.Random(augmentation_seed).randint(0, 1) == 0
        if swapped:
            diagnostics["swapped_rows"] += 1
        left_values, left_diag = _parse_col_val(augmented_left, allowed)
        right_values, right_diag = _parse_col_val(augmented_right, allowed)
        affected = False
        for side_diag in (left_diag, right_diag):
            for key in ("unknown_fields", "malformed_chunks", "duplicate_fields"):
                diagnostics[key] += side_diag[key]
                affected = affected or side_diag[key] > 0
        if not left_values:
            fallback, _ = _parse_col_val(left_original[position], allowed)
            left_values = fallback
            diagnostics["side_fallbacks"] += 1
            affected = True
        if not right_values:
            fallback, _ = _parse_col_val(right_original[position], allowed)
            right_values = fallback
            diagnostics["side_fallbacks"] += 1
            affected = True

        row = source.copy()
        for field in fields:
            if f"left_{field}" in row.index:
                row[f"left_{field}"] = left_values.get(field, "")
            if f"right_{field}" in row.index:
                row[f"right_{field}"] = right_values.get(field, "")
        if swapped:
            for left_name, right_name in (("left_id", "right_id"), ("source1", "source2")):
                if left_name in row.index and right_name in row.index:
                    row[left_name], row[right_name] = source[right_name], source[left_name]
        row["pair_id"] = f"{pair_id}#mixda#{seed}"
        rows.append(row)
        if affected and len(diagnostics["affected_pair_id_sample"]) < 25:
            diagnostics["affected_pair_id_sample"].append(pair_id)

    augmented = pd.DataFrame(rows).reset_index(drop=True)
    if len(augmented) != len(frame):
        raise AssertionError("MixDA row coverage changed")
    if not np.array_equal(augmented["label"].to_numpy(), frame["label"].to_numpy()):
        raise AssertionError("MixDA labels changed")
    if augmented["pair_id"].duplicated().any():
        raise AssertionError("MixDA pair IDs are not unique")
    return augmented, diagnostics


def _adaptive_encode(
    model: Any,
    texts: list[str],
    batch_size: int,
    minimum_batch_size: int = 4,
) -> tuple[np.ndarray, int]:
    import torch

    current = batch_size
    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            return np.asarray(embeddings), current
        except torch.cuda.OutOfMemoryError:
            if current <= minimum_batch_size:
                raise
            current = max(minimum_batch_size, current // 2)
            torch.cuda.empty_cache()


def _output_root(
    project_root: Path, dataset: str, model_id: str, revision: str
) -> Path:
    return (
        project_root
        / "artifacts"
        / "exp01_hef_mixda"
        / dataset
        / model_id.replace("/", "__")
        / revision
    )


def _training_signature(frame: pd.DataFrame) -> str:
    """Identify identical training inputs across dataset aliases (notably WDC)."""
    digest = hashlib.sha256()
    for pair_id, label, left, right in zip(
        frame["pair_id"].astype(str),
        frame["label"].astype(int),
        serialize(frame, "left"),
        serialize(frame, "right"),
    ):
        digest.update(f"{pair_id}\t{label}\t{left}\t{right}\n".encode())
    return digest.hexdigest()


def run_seed(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
    seed: int,
    batch_size: int,
    device: str,
    loaded_model: Any | None = None,
) -> Path:
    started = time.time()
    if seed not in {int(value) for value in config["protocol"]["seeds"]}:
        raise ValueError(f"Seed {seed} is not in the locked protocol")
    allowed_datasets = set(config["dataset_groups"]["exp01_all"]) | {
        "link_lives_release2"
    }
    if dataset not in allowed_datasets:
        raise ValueError(f"Dataset {dataset} is not in Experiment 1")
    model_spec = next(
        item for item in config["frozen_backbones"] if item["id"] == model_id
    )
    if str(model_spec["revision"]) != revision:
        raise ValueError("Requested model revision does not match locked config")

    spec = config["datasets"][dataset]
    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    split_validation = validate_splits(
        splits,
        enforce_offer_disjoint=dataset == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
        enforce_record_disjoint=spec["adapter"] == "link_lives",
    )
    original_scores, embedding_manifest = _load_embedding_scores(
        project_root, dataset, model_id, revision, splits
    )
    augmented, augmentation_diagnostics = augment_training_frame(splits["train"], seed)

    signature = _training_signature(splits["train"])
    cache = (
        project_root
        / "artifacts"
        / "exp01_hef_mixda_aug_cache"
        / signature
        / model_id.replace("/", "__")
        / revision
        / f"seed_{seed}.npz"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = False
    used_left_batch = batch_size
    used_right_batch = batch_size
    if cache.exists() and cache.stat().st_size > 0:
        cached = np.load(cache, allow_pickle=False)
        if np.array_equal(
            cached["pair_id"].astype(str), augmented["pair_id"].astype(str).to_numpy()
        ) and np.array_equal(
            cached["label"].astype(np.int8), augmented["label"].to_numpy(dtype=np.int8)
        ):
            augmented_scores = cached["embedding_score"].astype(np.float64)
            cache_hit = True
        else:
            raise AssertionError(f"Misaligned augmented embedding cache: {cache}")
    else:
        os.environ.setdefault("USE_TF", "0")
        if loaded_model is None:
            from sentence_transformers import SentenceTransformer

            loaded_model = SentenceTransformer(
                model_id,
                revision=revision,
                device=device,
                trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
            )
        prefix = str(model_spec["symmetric_prefix"])
        left_text = [prefix + value for value in serialize(augmented, "left").tolist()]
        right_text = [prefix + value for value in serialize(augmented, "right").tolist()]
        left, used_left_batch = _adaptive_encode(loaded_model, left_text, batch_size)
        right, used_right_batch = _adaptive_encode(loaded_model, right_text, batch_size)
        augmented_scores = np.sum(left * right, axis=1).astype(np.float64)
        temporary = cache.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            pair_id=augmented["pair_id"].astype(str).to_numpy(dtype=str),
            label=augmented["label"].to_numpy(dtype=np.int8),
            embedding_score=augmented_scores.astype(np.float32),
        )
        temporary.replace(cache)
        del left, right

    original_features = {
        split: structured_features(frame) for split, frame in splits.items()
    }
    for split in original_features:
        original_features[split]["embedding_score"] = original_scores[split]
    augmented_features = structured_features(augmented)
    augmented_features["embedding_score"] = augmented_scores
    train_features = pd.concat(
        [original_features["train"], augmented_features], ignore_index=True
    )
    train_labels = np.concatenate(
        [
            splits["train"]["label"].to_numpy(dtype=int),
            augmented["label"].to_numpy(dtype=int),
        ]
    )
    if list(train_features.columns) != list(original_features["valid"].columns):
        raise AssertionError("Feature schema changed between augmented train and validation")
    if not np.isfinite(train_features.to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite augmented training features")

    estimator = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    estimator.fit(train_features, train_labels)
    valid_score = estimator.predict_proba(original_features["valid"])[:, 1]
    test_score = estimator.predict_proba(original_features["test"])[:, 1]
    if not np.isfinite(valid_score).all() or not np.isfinite(test_score).all():
        raise AssertionError("Non-finite model scores")
    if np.ptp(valid_score) <= 1e-8 or np.ptp(test_score) <= 1e-8:
        raise AssertionError("Degenerate model scores")
    threshold = select_threshold(
        splits["valid"]["label"].to_numpy(dtype=int), valid_score
    )

    root = _output_root(project_root, dataset, model_id, revision)
    output = root / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, output / "model.joblib")
    np.savez_compressed(
        output / "scores.npz",
        valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(dtype=str),
        valid_label=splits["valid"]["label"].to_numpy(dtype=np.int8),
        valid_score=valid_score.astype(np.float32),
        test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(dtype=str),
        test_label=splits["test"]["label"].to_numpy(dtype=np.int8),
        test_score=test_score.astype(np.float32),
    )
    np.savez_compressed(
        output / "augmented_train_embedding_scores.npz",
        pair_id=augmented["pair_id"].astype(str).to_numpy(dtype=str),
        label=augmented["label"].to_numpy(dtype=np.int8),
        embedding_score=augmented_scores.astype(np.float32),
    )
    metrics = {
        "experiment": "exp01_hef_gbdt_mixda_training_only",
        "method": METHOD,
        "label": METHOD_LABEL,
        "official_ditto_claim": False,
        "dataset": dataset,
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "augmentation": {
            "family": "MixDA-style",
            "source_implementation": "paper1_hef.ditto_style.mixda_augment",
            "scope": "training_only",
            "operators": ["del", "swap", "drop_col", "append_col"],
            "operators_per_example": 3,
            "one_augmented_copy_per_original": True,
            "structured_and_dense_features_recomputed": True,
            "diagnostics": augmentation_diagnostics,
        },
        "selection": "fixed HEF-GBDT hyperparameters; threshold selected on validation only",
        "test_access_policy": "untouched until model fit and validation threshold lock",
        "threshold": float(threshold),
        "validation": classification_metrics(
            splits["valid"]["label"].to_numpy(dtype=int), valid_score, threshold
        ),
        "test": classification_metrics(
            splits["test"]["label"].to_numpy(dtype=int), test_score, threshold
        ),
        "split_validation": split_validation,
        "rows": {
            "original_train": int(len(splits["train"])),
            "augmented_train": int(len(augmented)),
            "fit_train": int(len(train_labels)),
            "valid": int(len(splits["valid"])),
            "test": int(len(splits["test"])),
        },
        "runtime_seconds": time.time() - started,
        "augmented_embedding_cache": {
            "training_signature_sha256": signature,
            "cache_hit": cache_hit,
            "path": str(cache.relative_to(project_root)),
        },
    }
    _write_json(output / "metrics.json", metrics)
    _write_json(
        output / "run_manifest.json",
        {
            "paper_eligible": True,
            "official_ditto_claim": False,
            "dataset": dataset,
            "seed": seed,
            "model_id": model_id,
            "model_revision": revision,
            "embedding_manifest": str(embedding_manifest.relative_to(project_root)),
            "embedding_manifest_sha256": _sha256(embedding_manifest),
            "feature_columns": list(train_features.columns),
            "batch_size_requested": batch_size,
            "batch_size_used_left": used_left_batch,
            "batch_size_used_right": used_right_batch,
            "device": device,
            "augmented_embedding_cache_hit": cache_hit,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "created_unix": time.time(),
        },
    )
    return output


def run_all_seeds(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
    batch_size: int,
    device: str,
) -> Path:
    """Run missing seeds while retaining one backbone instance in GPU memory."""
    from sentence_transformers import SentenceTransformer

    model_spec = next(
        item for item in config["frozen_backbones"] if item["id"] == model_id
    )
    model = SentenceTransformer(
        model_id,
        revision=revision,
        device=device,
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
    )
    root = _output_root(project_root, dataset, model_id, revision)
    for seed in [int(value) for value in config["protocol"]["seeds"]]:
        seed_root = root / f"seed_{seed}"
        required = [
            seed_root / "metrics.json",
            seed_root / "model.joblib",
            seed_root / "scores.npz",
            seed_root / "run_manifest.json",
        ]
        if all(path.exists() and path.stat().st_size > 0 for path in required):
            continue
        run_seed(
            project_root,
            config,
            dataset,
            model_id,
            revision,
            seed,
            batch_size,
            device,
            loaded_model=model,
        )
    return aggregate(project_root, config, dataset, model_id, revision)


def aggregate(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
) -> Path:
    root = _output_root(project_root, dataset, model_id, revision)
    runs = []
    for seed in [int(value) for value in config["protocol"]["seeds"]]:
        path = root / f"seed_{seed}" / "metrics.json"
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing seed metric: {path}")
        payload = json.loads(path.read_text())
        if int(payload["seed"]) != seed or payload["dataset"] != dataset:
            raise ValueError(f"Misaligned metric: {path}")
        runs.append(payload)
    f1 = np.asarray([run["test"]["f1"] for run in runs], dtype=float)
    summary = {
        "experiment": "exp01_hef_gbdt_mixda_training_only",
        "method": METHOD,
        "label": METHOD_LABEL,
        "official_ditto_claim": False,
        "dataset": dataset,
        "model_id": model_id,
        "revision": revision,
        "seeds": [run["seed"] for run in runs],
        "runs": len(runs),
        "test_f1_mean": float(f1.mean()),
        "test_f1_std": float(f1.std(ddof=1)),
        "seed_metrics": [
            {
                "seed": run["seed"],
                "threshold": run["threshold"],
                "validation": run["validation"],
                "test": run["test"],
            }
            for run in runs
        ],
        "paper_eligible": True,
    }
    _write_json(root / "metrics.json", summary)
    return root / "metrics.json"


def validate_output(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
) -> None:
    splits = load_dataset(
        project_root / config["project"]["data_root"], config["datasets"][dataset]
    )
    root = _output_root(project_root, dataset, model_id, revision)
    summary = json.loads((root / "metrics.json").read_text())
    if summary["runs"] != 3 or len(summary["seed_metrics"]) != 3:
        raise AssertionError("Expected exactly three seed runs")
    for seed in [int(value) for value in config["protocol"]["seeds"]]:
        output = root / f"seed_{seed}"
        for name in ("model.joblib", "scores.npz", "metrics.json", "run_manifest.json"):
            path = output / name
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing or empty artifact: {path}")
        scores = np.load(output / "scores.npz", allow_pickle=False)
        for split in ("valid", "test"):
            expected = splits[split]
            if not np.array_equal(
                scores[f"{split}_pair_id"].astype(str),
                expected["pair_id"].astype(str).to_numpy(),
            ):
                raise AssertionError(f"{dataset}/{seed}/{split}: pair ID mismatch")
            if not np.array_equal(
                scores[f"{split}_label"].astype(np.int8),
                expected["label"].to_numpy(dtype=np.int8),
            ):
                raise AssertionError(f"{dataset}/{seed}/{split}: label mismatch")
            values = scores[f"{split}_score"].astype(float)
            if len(values) != len(expected) or not np.isfinite(values).all():
                raise AssertionError(f"{dataset}/{seed}/{split}: score coverage failure")
            if np.ptp(values) <= 1e-8:
                raise AssertionError(f"{dataset}/{seed}/{split}: degenerate scores")


def load_config(project_root: Path) -> dict[str, Any]:
    return yaml.safe_load((project_root / "configs" / "experiment.yaml").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "run-all", "aggregate", "validate"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.project_root)
    if args.command == "run":
        if args.seed is None:
            raise ValueError("--seed is required for run")
        path = run_seed(
            args.project_root,
            config,
            args.dataset,
            args.model_id,
            args.revision,
            args.seed,
            args.batch_size,
            args.device,
        )
        print(path)
    elif args.command == "run-all":
        print(
            run_all_seeds(
                args.project_root,
                config,
                args.dataset,
                args.model_id,
                args.revision,
                args.batch_size,
                args.device,
            )
        )
    elif args.command == "aggregate":
        print(aggregate(args.project_root, config, args.dataset, args.model_id, args.revision))
    else:
        validate_output(args.project_root, config, args.dataset, args.model_id, args.revision)
        print("VALID")


if __name__ == "__main__":
    main()
