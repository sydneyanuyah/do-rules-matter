from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import load_dataset
from .evaluate import classification_metrics, select_threshold
from .exp03 import _f1, _paired_bootstrap_f1_predictions
from .features import structured_features


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _pair_id_hash(pair_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for pair_id in pair_ids.astype(str):
        digest.update(pair_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _nested_group_order(
    left_ids: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> list[str]:
    """Create one deterministic, approximately stratified ordering of left records."""
    group_label: dict[str, tuple[bool, bool]] = {}
    for group, label in zip(left_ids.astype(str), labels.astype(int)):
        positive, negative = group_label.get(group, (False, False))
        group_label[group] = (positive or label == 1, negative or label == 0)
    strata: dict[str, list[str]] = {}
    for group, (positive, negative) in group_label.items():
        key = f"positive_{int(positive)}_negative_{int(negative)}"
        strata.setdefault(key, []).append(group)
    rng = np.random.default_rng(seed)
    ranked: list[tuple[float, str, str]] = []
    for stratum, groups in sorted(strata.items()):
        values = np.asarray(sorted(groups), dtype=object)
        rng.shuffle(values)
        denominator = len(values)
        for index, group in enumerate(values):
            ranked.append(((index + 0.5) / denominator, stratum, str(group)))
    ranked.sort()
    return [group for _, _, group in ranked]


def _nested_subsets(
    left_ids: np.ndarray,
    labels: np.ndarray,
    fractions: list[float],
    seed: int,
) -> tuple[list[str], dict[float, np.ndarray]]:
    order = _nested_group_order(left_ids, labels, seed)
    group_rows = {
        group: np.flatnonzero(left_ids.astype(str) == group)
        for group in order
    }
    subsets: dict[float, np.ndarray] = {}
    selected: list[int] = []
    next_group = 0
    total = len(labels)
    for fraction in fractions:
        target = total if fraction >= 1.0 else max(2, int(np.ceil(fraction * total)))
        while (
            len(selected) < target
            or len(np.unique(labels[np.asarray(selected, dtype=int)])) < 2
        ):
            if next_group >= len(order):
                break
            selected.extend(group_rows[order[next_group]].tolist())
            next_group += 1
        subsets[fraction] = np.asarray(sorted(selected), dtype=int)
    return order, subsets


def run_exp05_dataset(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
    bootstrap_replicates: int | None = None,
) -> Path:
    if dataset not in config["dataset_groups"]["exp01_all"]:
        raise ValueError("Experiment 5 is locked to the five public Exp1 datasets")
    spec = config["experiments"]["exp05_label_efficiency"]
    fractions = sorted({float(value) for value in spec["fractions"]})
    if not fractions or fractions[-1] != 1.0:
        raise ValueError("Experiment 5 requires a 100% full-data anchor")
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    subset_seed = int(spec["subset_seed"])
    replicates = bootstrap_replicates or int(
        config["protocol"]["bootstrap_replicates"]
    )
    splits = load_dataset(
        project_root / config["project"]["data_root"],
        config["datasets"][dataset],
    )
    features = {
        split: structured_features(frame)
        for split, frame in splits.items()
    }
    embedding_dir = (
        project_root
        / "artifacts"
        / "embeddings"
        / dataset
        / model_id.replace("/", "__")
        / revision
    )
    for split in features:
        payload = np.load(
            embedding_dir / f"{split}.npz",
            allow_pickle=True,
        )
        if not np.array_equal(
            payload["pair_id"].astype(str),
            splits[split]["pair_id"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{dataset}/{split}: embedding pairs are not aligned")
        features[split]["embedding_score"] = payload["embedding_score"].astype(
            float
        )
    labels = {
        split: splits[split]["label"].to_numpy(dtype=np.int8)
        for split in splits
    }
    left_ids = splits["train"]["left_id"].astype(str).to_numpy()
    group_order, subsets = _nested_subsets(
        left_ids,
        labels["train"],
        fractions,
        subset_seed,
    )
    if model_id == "intfloat/e5-base-v2":
        output = project_root / "artifacts" / "exp05" / dataset
    else:
        output = (
            project_root
            / "artifacts"
            / "exp05_by_backbone"
            / model_id.replace("/", "__")
            / revision
            / dataset
        )
    output.mkdir(parents=True, exist_ok=True)
    fraction_results: dict[str, Any] = {}
    ensemble_predictions: dict[str, dict[str, np.ndarray]] = {}

    for fraction in fractions:
        key = f"{int(round(fraction * 100)):03d}"
        indices = subsets[fraction]
        pair_ids = splits["train"].iloc[indices]["pair_id"].astype(str).to_numpy()
        method_runs: dict[str, list[dict[str, Any]]] = {
            "hef_linear": [],
            "hef_gbdt": [],
        }
        predictions: dict[str, list[np.ndarray]] = {
            "hef_linear": [],
            "hef_gbdt": [],
        }
        scores: dict[str, list[np.ndarray]] = {
            "hef_linear": [],
            "hef_gbdt": [],
        }
        for seed in seeds:
            models = {
                "hef_linear": LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                    solver="lbfgs",
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
            for method, model in models.items():
                model.fit(features["train"].iloc[indices], labels["train"][indices])
                valid_score = model.predict_proba(features["valid"])[:, 1]
                test_score = model.predict_proba(features["test"])[:, 1]
                threshold = select_threshold(labels["valid"], valid_score)
                predictions[method].append(test_score >= threshold)
                scores[method].append(test_score.astype(np.float32))
                method_runs[method].append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "validation": classification_metrics(
                            labels["valid"], valid_score, threshold
                        ),
                        "test": classification_metrics(
                            labels["test"], test_score, threshold
                        ),
                    }
                )
                model_path = (
                    output
                    / "models"
                    / f"fraction_{key}"
                    / f"{method}_seed_{seed}.joblib"
                )
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path, compress=3)

        ensemble_predictions[key] = {}
        methods: dict[str, Any] = {}
        score_payload: dict[str, np.ndarray] = {
            "test_pair_id": splits["test"]["pair_id"].astype(str).to_numpy(),
            "test_label": labels["test"],
        }
        for method in ("hef_linear", "hef_gbdt"):
            majority = np.mean(np.vstack(predictions[method]), axis=0) >= 0.5
            ensemble_predictions[key][method] = majority
            test_values = np.asarray(
                [run["test"]["f1"] for run in method_runs[method]],
                dtype=float,
            )
            methods[method] = {
                "runs": method_runs[method],
                "test_f1_mean": float(test_values.mean()),
                "test_f1_std": float(test_values.std(ddof=1)),
                "majority_vote_test_f1": _f1(labels["test"], majority),
                "majority_vote_f1_interval": _paired_bootstrap_f1_predictions(
                    labels["test"],
                    majority,
                    np.zeros_like(majority),
                    replicates,
                    subset_seed,
                ),
            }
            score_payload[f"{method}_mean_score"] = np.mean(
                np.vstack(scores[method]), axis=0
            ).astype(np.float32)
            score_payload[f"{method}_majority_prediction"] = majority.astype(
                np.int8
            )
        np.savez_compressed(output / f"fraction_{key}_scores.npz", **score_payload)
        fraction_results[key] = {
            "requested_fraction": fraction,
            "selected_pairs": int(len(indices)),
            "total_train_pairs": int(len(labels["train"])),
            "actual_fraction": float(len(indices) / len(labels["train"])),
            "selected_left_groups": int(len(np.unique(left_ids[indices]))),
            "train_positive": int(labels["train"][indices].sum()),
            "train_negative": int(len(indices) - labels["train"][indices].sum()),
            "pair_id_sha256": _pair_id_hash(pair_ids),
            "methods": methods,
        }

    full_key = "100"
    for key, fraction_result in fraction_results.items():
        for method in ("hef_linear", "hef_gbdt"):
            fraction_result["methods"][method][
                "paired_f1_minus_full_data"
            ] = _paired_bootstrap_f1_predictions(
                labels["test"],
                ensemble_predictions[key][method],
                ensemble_predictions[full_key][method],
                replicates,
                subset_seed,
            )

    _write_json(
        output / "metrics.json",
        {
            "experiment": "exp05_label_efficiency",
            "dataset": dataset,
            "status": "complete_fusion_learners",
            "scope": "public_data_only",
            "model_id": model_id,
            "revision": revision,
            "fractions": fractions,
            "subset_policy": spec["subset_policy"],
            "subset_seed": subset_seed,
            "identical_subsets_across_methods_and_seeds": True,
            "selection": "thresholds selected on the unchanged full validation split",
            "bootstrap_replicates": replicates,
            "fractions_results": fraction_results,
        },
    )
    _write_json(
        output / "subset_manifest.json",
        {
            "dataset": dataset,
            "group_column": "left_id",
            "group_order": group_order,
            "fractions": {
                f"{int(round(fraction * 100)):03d}": {
                    "requested_fraction": fraction,
                    "selected_group_count": int(
                        len(np.unique(left_ids[subsets[fraction]]))
                    ),
                    "selected_pair_count": int(len(subsets[fraction])),
                    "pair_id_sha256": _pair_id_hash(
                        splits["train"]
                        .iloc[subsets[fraction]]["pair_id"]
                        .astype(str)
                        .to_numpy()
                    ),
                }
                for fraction in fractions
            },
        },
    )
    return output
