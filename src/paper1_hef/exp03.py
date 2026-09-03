from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_dataset
from .features import structured_features


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Paired bootstrap for aligned per-pair or per-query values."""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("Paired values must be aligned one-dimensional arrays")
    rng = np.random.default_rng(seed)
    difference = values_a - values_b
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(difference), len(difference))
        draws[index] = float(difference[sample].mean())
    return {
        "point_difference": float(difference.mean()),
        "bootstrap_mean_difference": float(draws.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "two_sided_p": float(
            min(1.0, 2.0 * min((draws <= 0).mean(), (draws >= 0).mean()))
        ),
        "replicates": int(replicates),
        "units": int(len(difference)),
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down family-wise-error adjusted p-values."""
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float), kind="stable")
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[int(original_index)]))
        running = max(running, candidate)
        adjusted[int(original_index)] = running
    return adjusted.tolist()


def _f1(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    true_positive = int(np.sum(y_true & prediction))
    denominator = int(np.sum(y_true) + np.sum(prediction))
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def _paired_bootstrap_f1_predictions(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    prediction_a = np.asarray(prediction_a, dtype=bool)
    prediction_b = np.asarray(prediction_b, dtype=bool)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(y_true), len(y_true))
        draws[index] = _f1(y_true[sample], prediction_a[sample]) - _f1(
            y_true[sample], prediction_b[sample]
        )
    return {
        "point_difference": float(
            _f1(y_true, prediction_a) - _f1(y_true, prediction_b)
        ),
        "bootstrap_mean_difference": float(draws.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "two_sided_p": float(
            min(1.0, 2.0 * min((draws <= 0).mean(), (draws >= 0).mean()))
        ),
        "replicates": int(replicates),
        "units": int(len(y_true)),
    }


def _scale(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    low = float(np.min(train))
    high = float(np.max(train))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _classification_ablation(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
    replicates: int,
) -> dict[str, Any]:
    splits = load_dataset(
        project_root / config["project"]["data_root"],
        config["datasets"][dataset],
    )
    y_test = splits["test"]["label"].to_numpy(dtype=np.int8)
    embedding_dir = (
        project_root
        / "artifacts"
        / "embeddings"
        / dataset
        / model_id.replace("/", "__")
        / revision
    )
    embedding = {
        split: np.load(embedding_dir / f"{split}.npz", allow_pickle=True)[
            "embedding_score"
        ].astype(float)
        for split in ("train", "test")
    }
    feature = {
        split: structured_features(splits[split])
        for split in ("train", "test")
    }
    rule = {
        split: _scale(
            feature["train"]["rule_score"].to_numpy(),
            feature[split]["rule_score"].to_numpy(),
        )
        for split in ("train", "test")
    }
    embedding_scaled = {
        split: _scale(embedding["train"], embedding[split])
        for split in ("train", "test")
    }
    result_dir = (
        project_root
        / "artifacts"
        / "exp01_final"
        / dataset
        / model_id.replace("/", "__")
        / revision
    )
    metrics = json.loads((result_dir / "metrics.json").read_text())
    methods = metrics["methods"]
    scores = {
        "frozen_embedding": embedding_scaled["test"],
        "rules": rule["test"],
        "equal_fusion": 0.5 * (embedding_scaled["test"] + rule["test"]),
    }
    convex_weight = float(methods["convex_fusion"]["embedding_weight"])
    scores["convex_fusion"] = (
        convex_weight * embedding_scaled["test"]
        + (1.0 - convex_weight) * rule["test"]
    )
    predictions = {
        name: score >= float(methods[name]["threshold"])
        for name, score in scores.items()
    }
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    for name in ("hef_linear", "hef_gbdt"):
        per_seed: list[np.ndarray] = []
        runs = {int(run["seed"]): run for run in methods[name]["repetitions"]}
        for seed in seeds:
            payload = np.load(
                result_dir / f"{name}_scores_seed_{seed}.npz",
                allow_pickle=True,
            )
            if not np.array_equal(
                payload["test_pair_id"].astype(str),
                splits["test"]["pair_id"].astype(str).to_numpy(),
            ):
                raise ValueError(f"{dataset}/{name}/{seed}: test pairs are not aligned")
            per_seed.append(
                payload["test_score"].astype(float)
                >= float(runs[seed]["threshold"])
            )
        predictions[name] = np.mean(np.vstack(per_seed), axis=0) >= 0.5

    comparisons = [
        ("equal_minus_embedding", "equal_fusion", "frozen_embedding"),
        ("convex_minus_equal", "convex_fusion", "equal_fusion"),
        ("linear_minus_equal", "hef_linear", "equal_fusion"),
        ("gbdt_minus_equal", "hef_gbdt", "equal_fusion"),
        ("gbdt_minus_linear", "hef_gbdt", "hef_linear"),
    ]
    bootstrap_seed = int(config["protocol"]["seeds"][0])
    return {
        "unit": "test_pair",
        "learned_seed_aggregation": (
            "three-seed majority vote using each seed's validation-selected threshold"
        ),
        "method_f1": {
            name: _f1(y_test, prediction)
            for name, prediction in predictions.items()
        },
        "paired_differences": {
            label: {
                "metric": "f1",
                "method_a": method_a,
                "method_b": method_b,
                **_paired_bootstrap_f1_predictions(
                    y_test,
                    predictions[method_a],
                    predictions[method_b],
                    replicates,
                    bootstrap_seed,
                ),
            }
            for label, method_a, method_b in comparisons
        },
    }


def _query_reciprocal_ranks(frame: pd.DataFrame, score: str) -> pd.Series:
    values: dict[str, float] = {}
    for query_id, group in frame.groupby("query_id", sort=False):
        labels = group["label"].to_numpy(dtype=np.int8)
        if not np.any(labels == 1):
            continue
        order = np.argsort(-group[score].to_numpy(dtype=float), kind="stable")
        positive = np.flatnonzero(labels[order] == 1)
        values[str(query_id)] = 1.0 / (int(positive[0]) + 1)
    return pd.Series(values, dtype=float)


def _ranking_ablation(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    replicates: int,
) -> dict[str, Any]:
    path = (
        project_root
        / "artifacts"
        / "exp02"
        / "ranking"
        / dataset
        / "test_scored.csv.gz"
    )
    frame = pd.read_csv(path)
    method_names = [
        "embedding",
        "rules",
        "equal_fusion",
        "convex_fusion",
        "hef_linear",
        "hef_gbdt",
        "hef_rank",
    ]
    reciprocal = {
        name: _query_reciprocal_ranks(frame, name)
        for name in method_names
    }
    query_ids = reciprocal["embedding"].index
    for name in method_names[1:]:
        if not reciprocal[name].index.equals(query_ids):
            raise ValueError(f"{dataset}/{name}: conditional query sets are not aligned")
    comparisons = [
        ("equal_minus_embedding", "equal_fusion", "embedding"),
        ("convex_minus_equal", "convex_fusion", "equal_fusion"),
        ("linear_minus_equal", "hef_linear", "equal_fusion"),
        ("gbdt_minus_equal", "hef_gbdt", "equal_fusion"),
        ("rank_minus_equal", "hef_rank", "equal_fusion"),
        ("gbdt_minus_linear", "hef_gbdt", "hef_linear"),
        ("rank_minus_gbdt", "hef_rank", "hef_gbdt"),
    ]
    bootstrap_seed = int(config["protocol"]["seeds"][0])
    return {
        "unit": "test_query_with_true_match_in_top_100_pool",
        "method_mrr_at_100": {
            name: float(values.mean())
            for name, values in reciprocal.items()
        },
        "paired_differences": {
            label: {
                "metric": "mrr_at_100",
                "method_a": method_a,
                "method_b": method_b,
                **_paired_bootstrap(
                    reciprocal[method_a].to_numpy(),
                    reciprocal[method_b].to_numpy(),
                    replicates,
                    bootstrap_seed,
                ),
            }
            for label, method_a, method_b in comparisons
        },
    }


def run_exp03(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
    bootstrap_replicates: int | None = None,
) -> Path:
    if dataset not in config["dataset_groups"]["exp01_all"]:
        raise ValueError("Experiment 3 is locked to the five public Exp1/2 datasets")
    replicates = bootstrap_replicates or int(
        config["protocol"]["bootstrap_replicates"]
    )
    output = project_root / "artifacts" / "exp03" / dataset
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "exp03_fusion_estimator_ablation",
        "dataset": dataset,
        "status": "complete",
        "scope": "public_data_only",
        "backbone": {
            "model_id": model_id,
            "revision": revision,
            "policy": "fixed primary E5 evidence reused from Experiments 1 and 2",
        },
        "selection_policy": (
            "No estimator, feature, threshold, or split was changed for Experiment 3"
        ),
        "classification": _classification_ablation(
            project_root,
            config,
            dataset,
            model_id,
            revision,
            replicates,
        ),
        "ranking": _ranking_ablation(
            project_root,
            config,
            dataset,
            replicates,
        ),
    }
    _write_json(output / "metrics.json", result)
    return output


def finalize_exp03(
    project_root: Path,
    datasets: list[str],
) -> Path:
    """Attach locked Holm corrections and write a compact all-dataset summary."""
    output_root = project_root / "artifacts" / "exp03"
    results = {
        dataset: json.loads((output_root / dataset / "metrics.json").read_text())
        for dataset in datasets
    }
    family_sizes: dict[str, int] = {}
    for section in ("classification", "ranking"):
        members: list[tuple[str, str, dict[str, Any]]] = []
        for dataset in datasets:
            for comparison, value in results[dataset][section][
                "paired_differences"
            ].items():
                members.append((dataset, comparison, value))
        adjusted = _holm_adjust(
            [float(value["two_sided_p"]) for _, _, value in members]
        )
        family = f"exp03_{section}_all_public_datasets"
        family_sizes[family] = len(members)
        for (_, _, value), p_adjusted in zip(members, adjusted):
            value["holm_adjusted_p"] = p_adjusted
            value["holm_family"] = family
        for dataset in datasets:
            _write_json(output_root / dataset / "metrics.json", results[dataset])

    summary = {
        "experiment": "exp03_fusion_estimator_ablation",
        "status": "complete",
        "datasets": datasets,
        "multiple_testing": "Holm correction, one locked family per metric",
        "holm_family_sizes": family_sizes,
        "results": {
            dataset: {
                "classification_f1": results[dataset]["classification"][
                    "method_f1"
                ],
                "classification_differences": results[dataset]["classification"][
                    "paired_differences"
                ],
                "ranking_mrr_at_100": results[dataset]["ranking"][
                    "method_mrr_at_100"
                ],
                "ranking_differences": results[dataset]["ranking"][
                    "paired_differences"
                ],
            }
            for dataset in datasets
        },
    }
    _write_json(output_root / "summary.json", summary)
    return output_root / "summary.json"
