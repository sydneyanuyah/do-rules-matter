#!/usr/bin/env python3
"""Experiment 6: evidence-group ablation for classification and ranking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from paper1_hef.data import load_dataset
from paper1_hef.evaluate import classification_metrics, select_threshold
from paper1_hef.exp02 import _evaluate
from paper1_hef.features import structured_features


PRODUCT_GROUPS = {
    "lexical": ["title_jaccard", "title_char_ratio", "description_jaccard", "all_token_jaccard"],
    "numerical": ["price_similarity", "numeric_token_jaccard"],
    "categorical": ["brand_exact", "manufacturer_exact", "currency_exact"],
    "relational": ["model_exact"],
    "incumbent_rule": ["rule_score"],
    "embedding": ["embedding_score"],
    "availability_mask": ["left_field_fraction", "right_field_fraction", "shared_field_fraction"],
}

GENEALOGY_GROUPS = {
    "lexical": [
        "name_jaccard", "name_char_ratio", "birth_place_jaccard",
        "residence_information_jaccard", "occupation_jaccard", "all_token_jaccard",
    ],
    "numerical": ["birth_year_similarity"],
    "categorical": ["sex_exact", "marital_status_exact"],
    "relational": ["age_progression_similarity", "residence_parish_exact", "residence_county_exact"],
    "incumbent_rule": ["rule_score"],
    "embedding": ["embedding_score"],
    "availability_mask": ["left_field_fraction", "right_field_fraction", "shared_field_fraction"],
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    template = GENEALOGY_GROUPS if "name_jaccard" in columns else PRODUCT_GROUPS
    groups = {name: [column for column in values if column in columns] for name, values in template.items()}
    assigned = [column for values in groups.values() for column in values]
    missing = sorted(set(columns) - set(assigned))
    if missing:
        raise ValueError(f"Unassigned evidence columns: {missing}")
    if any(not values for values in groups.values()):
        empty = [name for name, values in groups.items() if not values]
        raise ValueError(f"Empty domain evidence groups: {empty}")
    return groups


def conditions(columns: list[str], groups: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {"full_hef": list(columns)}
    for name, values in groups.items():
        result[f"drop_{name}"] = [column for column in columns if column not in values]
    raw = groups["lexical"] + groups["numerical"] + groups["categorical"] + groups["relational"] + groups["availability_mask"]
    result["raw_field_features_only"] = raw
    result["aggregate_rule_score_only"] = groups["incumbent_rule"]
    return result


def classifier(seed: int, name: str):
    if name == "hef_linear":
        return LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed, solver="lbfgs")
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=True, random_state=seed,
    )


def ranking_model(seed: int, name: str):
    if name == "hef_gbdt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=seed,
        )
    from lightgbm import LGBMRanker
    return LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=500,
        learning_rate=0.05, num_leaves=15, min_child_samples=20,
        reg_lambda=1.0, random_state=seed, verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--revision", default="f52bf8ec8c7124536f0efb74aca902b2995e5bcd")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    if args.dataset not in config["dataset_groups"]["exp01_all"]:
        raise ValueError("Experiment 6 is locked to the six public datasets")
    seeds = [int(seed) for seed in config["protocol"]["seeds"]]
    sizes = [int(value) for value in config["experiments"]["exp02_candidate_ranking"]["k"]]
    out = root / "artifacts" / "exp06" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    splits = load_dataset(root / config["project"]["data_root"], config["datasets"][args.dataset])
    emb_dir = root / "artifacts" / "embeddings" / args.dataset / args.model.replace("/", "__") / args.revision
    frames: dict[str, pd.DataFrame] = {}
    for split, frame in splits.items():
        values = structured_features(frame)
        values["embedding_score"] = np.load(emb_dir / f"{split}.npz", allow_pickle=True)["embedding_score"].astype(float)
        frames[split] = values
    groups = feature_groups(list(frames["train"].columns))
    variants = conditions(list(frames["train"].columns), groups)
    y = {split: splits[split]["label"].to_numpy(dtype=np.int8) for split in splits}
    classification: dict[str, Any] = {}
    classification_arrays: dict[str, np.ndarray] = {
        "test_pair_id": splits["test"]["pair_id"].astype(str).to_numpy(),
        "test_label": y["test"],
    }
    for variant, names in variants.items():
        classification[variant] = {"features": names, "methods": {}}
        for method in ("hef_linear", "hef_gbdt"):
            runs = []
            for seed in seeds:
                model = classifier(seed, method)
                model.fit(frames["train"][names], y["train"])
                valid_score = model.predict_proba(frames["valid"][names])[:, 1]
                test_score = model.predict_proba(frames["test"][names])[:, 1]
                threshold = select_threshold(y["valid"], valid_score)
                runs.append({
                    "seed": seed, "threshold": threshold,
                    "validation": classification_metrics(y["valid"], valid_score, threshold),
                    "test": classification_metrics(y["test"], test_score, threshold),
                })
                classification_arrays[f"{variant}__{method}__seed_{seed}"] = test_score.astype(np.float32)
                joblib.dump(model, out / f"classification_{variant}_{method}_seed_{seed}.joblib")
            f1 = np.asarray([run["test"]["f1"] for run in runs], dtype=float)
            classification[variant]["methods"][method] = {
                "per_seed": runs, "test_f1_mean": float(f1.mean()),
                "test_f1_std": float(f1.std(ddof=1)),
            }
    np.savez_compressed(out / "classification_scores.npz", **classification_arrays)

    pool = pd.read_csv(root / "artifacts" / "exp02" / "candidate_pools" / args.dataset / "top100.csv.gz")
    pool_columns = [column for column in pool.columns if column not in {"split", "query_id", "candidate_id", "retrieval_rank", "label"}]
    rank_groups = feature_groups(pool_columns)
    rank_variants = conditions(pool_columns, rank_groups)
    train = pool.loc[pool["split"].eq("train")].copy()
    valid = pool.loc[pool["split"].eq("valid")].copy()
    test = pool.loc[pool["split"].eq("test")].copy()
    train_hit = set(train.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
    valid_hit = set(valid.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
    fit_train = train.loc[train["query_id"].isin(train_hit)].sort_values(["query_id", "retrieval_rank"])
    fit_valid = valid.loc[valid["query_id"].isin(valid_hit)].sort_values(["query_id", "retrieval_rank"])
    train_groups = fit_train.groupby("query_id", sort=False).size().to_list()
    valid_groups = fit_valid.groupby("query_id", sort=False).size().to_list()
    ranking: dict[str, Any] = {}
    ranking_arrays: dict[str, np.ndarray] = {
        "query_id": test["query_id"].astype(str).to_numpy(),
        "candidate_id": test["candidate_id"].astype(str).to_numpy(),
        "label": test["label"].to_numpy(dtype=np.int8),
    }
    from lightgbm import early_stopping, log_evaluation
    for variant, names in rank_variants.items():
        ranking[variant] = {"features": names, "methods": {}}
        for method in ("hef_gbdt", "hef_rank"):
            per_seed = []
            scores = []
            for seed in seeds:
                model = ranking_model(seed, method)
                if method == "hef_rank":
                    model.fit(
                        fit_train[names], fit_train["label"], group=train_groups,
                        eval_set=[(fit_valid[names], fit_valid["label"])], eval_group=[valid_groups],
                        eval_at=[1, 5, 10, 100], callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
                    )
                    score = model.predict(test[names])
                else:
                    model.fit(fit_train[names], fit_train["label"])
                    score = model.predict_proba(test[names])[:, 1]
                score = np.asarray(score, dtype=float)
                scores.append(score)
                per_seed.append({"seed": seed, "metrics": _evaluate(test.assign(_score=score), "_score", sizes)})
                ranking_arrays[f"{variant}__{method}__seed_{seed}"] = score.astype(np.float32)
                joblib.dump(model, out / f"ranking_{variant}_{method}_seed_{seed}.joblib")
            ensemble = np.mean(np.vstack(scores), axis=0)
            ranking[variant]["methods"][method] = {
                "per_seed": per_seed,
                "three_seed_ensemble": _evaluate(test.assign(_score=ensemble), "_score", sizes),
            }
    np.savez_compressed(out / "ranking_scores.npz", **ranking_arrays)

    result = {
        "experiment": "exp06_evidence_feature_ablation",
        "dataset": args.dataset,
        "status": "complete",
        "protocol": {
            "selection": "validation_only", "test": "untouched until locked selection",
            "seeds": seeds, "ranking_pool": "fixed top100 from Experiment 2",
            "ranking_reports_hits_at_100": True,
        },
        "feature_groups": groups,
        "classification": classification,
        "ranking": ranking,
        "runtime_seconds": time.time() - started,
    }
    write_json(out / "metrics.json", result)
    write_json(out / "manifest.json", {
        "dataset": args.dataset, "expected_conditions": sorted(variants),
        "classification_score_file": "classification_scores.npz",
        "ranking_score_file": "ranking_scores.npz", "status": "verified_local",
    })
    print(out)


if __name__ == "__main__":
    main()
