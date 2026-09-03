from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import load_dataset, validate_splits
from .evaluate import classification_metrics, paired_bootstrap_f1, select_threshold
from .features import structured_features


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tree_hash(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve().relative_to(base.resolve())).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _scale_from_train(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    low = float(np.min(train))
    high = float(np.max(train))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _best_convex(
    y_valid: np.ndarray,
    embedding_valid: np.ndarray,
    rule_valid: np.ndarray,
) -> tuple[float, float]:
    best = (-1.0, 0.5, 0.5)
    for weight in np.linspace(0.0, 1.0, 101):
        scores = weight * embedding_valid + (1.0 - weight) * rule_valid
        threshold = select_threshold(y_valid, scores)
        f1 = classification_metrics(y_valid, scores, threshold)["f1"]
        candidate = (f1, -abs(weight - 0.5), weight, threshold)
        if candidate[:3] > (best[0], -abs(best[1] - 0.5), best[1]):
            best = (f1, weight, threshold)
    return float(best[1]), float(best[2])


def _load_embedding_scores(
    root: Path,
    dataset_name: str,
    model_id: str,
    revision: str,
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, np.ndarray], Path]:
    directory = root / "artifacts" / "embeddings" / dataset_name / model_id.replace("/", "__") / revision
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen embeddings: {manifest_path}")
    scores: dict[str, np.ndarray] = {}
    for split, frame in splits.items():
        # These are project-generated artifacts. Older embedding files stored
        # string pair IDs as NumPy object arrays, which require pickle support.
        payload = np.load(directory / f"{split}.npz", allow_pickle=True)
        pair_ids = payload["pair_id"].astype(str)
        labels = payload["label"].astype(np.int8)
        if not np.array_equal(pair_ids, frame["pair_id"].astype(str).to_numpy()):
            raise ValueError(f"{dataset_name}/{split}: embedding pair IDs are not row-aligned")
        if not np.array_equal(labels, frame["label"].to_numpy(dtype=np.int8)):
            raise ValueError(f"{dataset_name}/{split}: embedding labels are not row-aligned")
        scores[split] = payload["embedding_score"].astype(float)
    return scores, manifest_path


def run_exp01_dataset(
    project_root: Path,
    config: dict[str, Any],
    dataset_name: str,
    model_id: str,
    revision: str,
    bootstrap_replicates: int | None = None,
) -> Path:
    started = time.time()
    spec = config["datasets"][dataset_name]
    if spec.get("experiment") == "public_genealogy_transfer":
        raise ValueError(f"{dataset_name} is a supplemental genealogy study, not Experiment 1")

    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    validation = validate_splits(
        splits,
        enforce_offer_disjoint=dataset_name == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
    )
    embedding, embedding_manifest_path = _load_embedding_scores(
        project_root, dataset_name, model_id, revision, splits
    )
    features = {split: structured_features(frame) for split, frame in splits.items()}
    for split in features:
        features[split]["embedding_score"] = embedding[split]
    y = {split: splits[split]["label"].to_numpy(dtype=int) for split in splits}

    rule_scaled = {
        split: _scale_from_train(
            features["train"]["rule_score"].to_numpy(),
            features[split]["rule_score"].to_numpy(),
        )
        for split in splits
    }
    embedding_scaled = {
        split: _scale_from_train(embedding["train"], embedding[split])
        for split in splits
    }

    methods: dict[str, Any] = {}
    score_cache: dict[str, dict[str, np.ndarray]] = {}

    for name, scores in {
        "rules": rule_scaled,
        "frozen_embedding": embedding_scaled,
        "equal_fusion": {
            split: 0.5 * embedding_scaled[split] + 0.5 * rule_scaled[split]
            for split in splits
        },
    }.items():
        threshold = select_threshold(y["valid"], scores["valid"])
        score_cache[name] = scores
        methods[name] = {
            "selection": "validation_only",
            "threshold": threshold,
            "validation": classification_metrics(y["valid"], scores["valid"], threshold),
            "test": classification_metrics(y["test"], scores["test"], threshold),
        }

    convex_weight, convex_threshold = _best_convex(
        y["valid"], embedding_scaled["valid"], rule_scaled["valid"]
    )
    convex_scores = {
        split: convex_weight * embedding_scaled[split] + (1.0 - convex_weight) * rule_scaled[split]
        for split in splits
    }
    score_cache["convex_fusion"] = convex_scores
    methods["convex_fusion"] = {
        "selection": "weight_and_threshold_on_validation_only",
        "embedding_weight": convex_weight,
        "threshold": convex_threshold,
        "validation": classification_metrics(y["valid"], convex_scores["valid"], convex_threshold),
        "test": classification_metrics(y["test"], convex_scores["test"], convex_threshold),
    }

    seeds = [int(seed) for seed in config["protocol"]["seeds"]]
    repetitions: dict[str, list[dict[str, Any]]] = {"hef_linear": [], "hef_gbdt": []}
    output = (
        project_root
        / config["project"]["output_root"]
        / "exp01_final"
        / dataset_name
        / model_id.replace("/", "__")
        / revision
    )
    output.mkdir(parents=True, exist_ok=True)
    model_features = list(features["train"].columns)
    reps = bootstrap_replicates or int(config["protocol"]["bootstrap_replicates"])

    for seed in seeds:
        models = {
            "hef_linear": LogisticRegression(
                class_weight="balanced", max_iter=3000, random_state=seed, solver="lbfgs"
            ),
            "hef_gbdt": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=seed,
            ),
        }
        for name, model in models.items():
            model.fit(features["train"], y["train"])
            valid_scores = model.predict_proba(features["valid"])[:, 1]
            test_scores = model.predict_proba(features["test"])[:, 1]
            threshold = select_threshold(y["valid"], valid_scores)
            result = {
                "seed": seed,
                "threshold": threshold,
                "validation": classification_metrics(y["valid"], valid_scores, threshold),
                "test": classification_metrics(y["test"], test_scores, threshold),
                "paired_bootstrap_minus_embedding": paired_bootstrap_f1(
                    y["test"],
                    test_scores,
                    threshold,
                    embedding_scaled["test"],
                    methods["frozen_embedding"]["threshold"],
                    reps,
                    seed,
                ),
            }
            repetitions[name].append(result)
            joblib.dump(model, output / f"{name}_seed_{seed}.joblib")
            np.savez_compressed(
                output / f"{name}_scores_seed_{seed}.npz",
                valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(),
                valid_label=y["valid"].astype(np.int8),
                valid_score=valid_scores.astype(np.float32),
                test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),
                test_label=y["test"].astype(np.int8),
                test_score=test_scores.astype(np.float32),
            )

    for name, runs in repetitions.items():
        test_f1 = np.asarray([run["test"]["f1"] for run in runs], dtype=float)
        methods[name] = {
            "features": model_features,
            "repetitions": runs,
            "test_f1_mean": float(test_f1.mean()),
            "test_f1_std": float(test_f1.std(ddof=1)) if len(test_f1) > 1 else 0.0,
        }

    metrics = {
        "experiment": "exp01_standard_pair_classification",
        "dataset": dataset_name,
        "status": "paper_candidate_pending_cross_encoder_and_final_audit",
        "primary_metric": "match_class_f1",
        "model_id": model_id,
        "revision": revision,
        "split_validation": validation,
        "methods": methods,
        "runtime_seconds": time.time() - started,
        "test_access_policy": "evaluated_after validation-only thresholds and hyperparameters were locked",
    }
    _write_json(output / "metrics.json", metrics)

    source_files = [
        path
        for path in (project_root / config["project"]["data_root"] / spec["directory"]).rglob("*")
        if path.is_file()
    ]
    code_files = list((project_root / "src" / "paper1_hef").glob("*.py"))
    manifest = {
        "experiment": "exp01_standard_pair_classification",
        "dataset": dataset_name,
        "dataset_sha256": _tree_hash(source_files, project_root),
        "embedding_manifest_sha256": _tree_hash([embedding_manifest_path], project_root),
        "code_sha256": _tree_hash(code_files, project_root),
        "config_sha256": _tree_hash([project_root / "configs" / "experiment.yaml"], project_root),
        "model_id": model_id,
        "model_revision": revision,
        "seeds": seeds,
        "bootstrap_replicates": reps,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "created_unix": time.time(),
        "paper_eligible": False,
        "paper_eligibility_blocker": "Cross-encoder baseline and final multi-dataset audit are not yet attached.",
    }
    _write_json(output / "run_manifest.json", manifest)
    return output
