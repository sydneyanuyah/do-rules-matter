from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import load_dataset
from .features import FIELDS, GENEALOGY_FIELDS, clean, serialize, structured_features


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash_split(dataset: str, query_id: str) -> str:
    value = int(hashlib.sha256(f"{dataset}:{query_id}".encode()).hexdigest()[:16], 16) % 10000
    if value < 7000:
        return "train"
    if value < 8500:
        return "valid"
    return "test"


def _normal_name(name: str) -> str:
    return {
        "name": "title",
        "model_no": "modelno",
        "model_number": "modelno",
        "currency": "priceCurrency",
    }.get(name, name)


def _fields_for_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    """Select the public schema that is actually present in a record frame."""
    return GENEALOGY_FIELDS if "name" in frame.columns else FIELDS


def _catalog(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={column: _normal_name(column) for column in frame.columns})
    frame["id"] = frame["id"].astype(str)
    return frame.drop_duplicates("id").reset_index(drop=True)


def _deepmatcher_source(
    project_root: Path, config: dict[str, Any], dataset: str
) -> dict[str, Any]:
    spec = config["datasets"][dataset]
    directory = project_root / config["project"]["data_root"] / spec["directory"]
    left = _catalog(directory / "tableA.csv")
    right = _catalog(directory / "tableB.csv")
    pairs = pd.concat(
        [pd.read_csv(directory / f"{split}.csv") for split in ("train", "valid", "test")],
        ignore_index=True,
    )
    positives = pairs.loc[pairs["label"].eq(1), ["ltable_id", "rtable_id"]].copy()
    positives["query_id"] = positives["ltable_id"].astype(str)
    positives["candidate_id"] = positives["rtable_id"].astype(str)
    positives = positives[["query_id", "candidate_id"]].drop_duplicates()
    gold = positives.groupby("query_id")["candidate_id"].agg(lambda values: set(values))
    queries = left[left["id"].isin(gold.index)].copy()
    queries["split"] = queries["id"].map(lambda value: _hash_split(dataset, str(value)))
    return {
        "queries": queries,
        "candidates": right,
        "gold": gold.to_dict(),
        "split_policy": "sha256_query_id_70_15_15",
    }


def _wdc_source(
    project_root: Path, config: dict[str, Any], dataset: str
) -> dict[str, Any]:
    spec = config["datasets"][dataset]
    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    query_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    gold: dict[str, set[str]] = {}
    for split, pairs in splits.items():
        left_columns = ["left_id", "left_cluster_id", *[f"left_{field}" for field in FIELDS]]
        right_columns = ["right_id", "right_cluster_id", *[f"right_{field}" for field in FIELDS]]
        left_columns = [column for column in left_columns if column in pairs]
        right_columns = [column for column in right_columns if column in pairs]
        left = pairs[left_columns].drop_duplicates("left_id").copy()
        right = pairs[right_columns].drop_duplicates("right_id").copy()
        left = left.rename(
            columns={
                "left_id": "id",
                "left_cluster_id": "cluster_id",
                **{f"left_{field}": field for field in FIELDS},
            }
        )
        right = right.rename(
            columns={
                "right_id": "id",
                "right_cluster_id": "cluster_id",
                **{f"right_{field}": field for field in FIELDS},
            }
        )
        left["id"] = left["id"].astype(str)
        right["id"] = right["id"].astype(str)
        positive_queries = set(pairs.loc[pairs["label"].eq(1), "left_id"].astype(str))
        left = left[left["id"].isin(positive_queries)].copy()
        left["split"] = split
        right["split"] = split
        by_cluster = right.groupby("cluster_id")["id"].agg(lambda values: set(values)).to_dict()
        for row in left[["id", "cluster_id"]].itertuples(index=False):
            gold[f"{split}:{row.id}"] = set(by_cluster.get(row.cluster_id, set()))
        left["original_id"] = left["id"]
        left["id"] = split + ":" + left["id"]
        right["original_id"] = right["id"]
        right["id"] = split + ":" + right["id"]
        for query_id in left["id"]:
            raw = query_id.split(":", 1)[1]
            gold[query_id] = {f"{split}:{value}" for value in gold.pop(f"{split}:{raw}")}
        query_parts.append(left)
        candidate_parts.append(right)
    return {
        "queries": pd.concat(query_parts, ignore_index=True),
        "candidates": pd.concat(candidate_parts, ignore_index=True),
        "gold": gold,
        "split_policy": "official_wdc_files",
        "candidate_partition_column": "split",
    }


def _link_lives_source(
    project_root: Path, config: dict[str, Any], dataset: str
) -> dict[str, Any]:
    """Build a ranking source directly from Link-Lives pair files.

    Link-Lives is pair-labelled genealogical data, not a DeepMatcher catalogue.
    IDs are namespace-qualified by split so repeated record identifiers cannot
    leak candidates across its official train/validation/test partitions.
    """
    spec = config["datasets"][dataset]
    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    fields = set(GENEALOGY_FIELDS) | set(FIELDS)
    query_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    gold: dict[str, set[str]] = {}
    for split, pairs in splits.items():
        if not {"left_id", "right_id", "label"}.issubset(pairs.columns):
            raise ValueError(f"Link-Lives {split} lacks required pair columns")
        left_columns = ["left_id", *[f"left_{field}" for field in fields if f"left_{field}" in pairs]]
        right_columns = ["right_id", *[f"right_{field}" for field in fields if f"right_{field}" in pairs]]
        left = pairs[left_columns].drop_duplicates("left_id").rename(
            columns={"left_id": "id", **{f"left_{field}": field for field in fields}}
        )
        right = pairs[right_columns].drop_duplicates("right_id").rename(
            columns={"right_id": "id", **{f"right_{field}": field for field in fields}}
        )
        left["id"] = left["id"].astype(str)
        right["id"] = right["id"].astype(str)
        positive = pairs.loc[pairs["label"].eq(1), ["left_id", "right_id"]].copy()
        positive["left_id"] = positive["left_id"].astype(str)
        positive["right_id"] = positive["right_id"].astype(str)
        positive_map = positive.groupby("left_id")["right_id"].agg(lambda values: set(values)).to_dict()
        left = left[left["id"].isin(positive_map)].copy()
        left["split"] = split
        right["split"] = split
        left["original_id"] = left["id"]
        right["original_id"] = right["id"]
        left["id"] = split + ":" + left["id"]
        right["id"] = split + ":" + right["id"]
        for raw_id, candidates in positive_map.items():
            gold[f"{split}:{raw_id}"] = {f"{split}:{candidate}" for candidate in candidates}
        query_parts.append(left)
        candidate_parts.append(right)
    return {
        "queries": pd.concat(query_parts, ignore_index=True),
        "candidates": pd.concat(candidate_parts, ignore_index=True),
        "gold": gold,
        "split_policy": "official_link_lives_files",
        "candidate_partition_column": "split",
    }


def _ranking_source(project_root: Path, config: dict[str, Any], dataset: str) -> dict[str, Any]:
    adapter = config["datasets"][dataset]["adapter"]
    if adapter == "wdc":
        return _wdc_source(project_root, config, dataset)
    if adapter == "link_lives":
        return _link_lives_source(project_root, config, dataset)
    return _deepmatcher_source(project_root, config, dataset)


def _record_text(frame: pd.DataFrame, prefix: str) -> list[str]:
    fields = set(FIELDS) | set(GENEALOGY_FIELDS)
    renamed = frame.rename(columns={field: f"left_{field}" for field in fields if field in frame})
    return [prefix + value for value in serialize(renamed, "left").tolist()]


def _exp02_backbone_directory(model_id: str) -> str | None:
    if model_id == "intfloat/e5-base-v2":
        return None
    return model_id.replace("/", "__")


def _topk(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
    device: str,
    batch_size: int = 256,
    query_ids: np.ndarray | None = None,
    candidate_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    target = torch.device(device)
    candidates = torch.as_tensor(candidate_embeddings, device=target)
    all_scores: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for start in range(0, len(query_embeddings), batch_size):
        query = torch.as_tensor(query_embeddings[start : start + batch_size], device=target)
        similarities = query @ candidates.T
        if query_ids is not None and candidate_ids is not None:
            current_ids = query_ids[start : start + batch_size]
            for row, query_id in enumerate(current_ids):
                same = np.flatnonzero(candidate_ids == query_id)
                if len(same):
                    similarities[row, torch.as_tensor(same, device=target)] = -torch.inf
        scores, indices = torch.topk(
            similarities, k=min(k, len(candidates) - 1), dim=1
        )
        all_scores.append(scores.float().cpu().numpy())
        all_indices.append(indices.cpu().numpy())
    return np.vstack(all_scores), np.vstack(all_indices)


def build_candidate_pool(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
    device: str = "cuda",
    batch_size: int = 128,
) -> Path:
    os.environ.setdefault("USE_TF", "0")
    from sentence_transformers import SentenceTransformer

    started = time.time()
    backbone_directory = _exp02_backbone_directory(model_id)
    output = project_root / "artifacts" / "exp02"
    if backbone_directory:
        output = output / "candidate_pools_by_backbone" / backbone_directory / dataset
    else:
        output = output / "candidate_pools" / dataset
    pool_path = output / "top100.csv.gz"
    if pool_path.exists() and (output / "manifest.json").exists():
        return output
    source = _ranking_source(project_root, config, dataset)
    queries: pd.DataFrame = source["queries"]
    candidates: pd.DataFrame = source["candidates"]
    model_spec = next(item for item in config["frozen_backbones"] if item["id"] == model_id)
    model_spec = next(
        item for item in config["frozen_backbones"] if item["id"] == model_id
    )
    model = SentenceTransformer(
        model_id,
        revision=revision,
        device=device,
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
    )
    prefix = model_spec["symmetric_prefix"]
    query_embeddings = model.encode(
        _record_text(queries, prefix),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    rows: list[pd.DataFrame] = []
    if source.get("candidate_partition_column"):
        for split in ("train", "valid", "test"):
            query_mask = queries["split"].eq(split).to_numpy()
            candidate_mask = candidates["split"].eq(split).to_numpy()
            query_subset = queries.loc[query_mask].reset_index(drop=True)
            candidate_subset = candidates.loc[candidate_mask].reset_index(drop=True)
            candidate_embeddings = model.encode(
                _record_text(candidate_subset, prefix),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)
            scores, indices = _topk(
                query_embeddings[query_mask],
                candidate_embeddings,
                100,
                device,
                query_ids=query_subset["original_id"].astype(str).to_numpy(),
                candidate_ids=candidate_subset["original_id"].astype(str).to_numpy(),
            )
            rows.append(_pool_rows(query_subset, candidate_subset, scores, indices, source["gold"]))
    else:
        candidate_embeddings = model.encode(
            _record_text(candidates, prefix),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        scores, indices = _topk(query_embeddings, candidate_embeddings, 100, device)
        rows.append(_pool_rows(queries, candidates, scores, indices, source["gold"]))
    pool = pd.concat(rows, ignore_index=True)
    pool = _attach_features(pool, queries, candidates)
    output.mkdir(parents=True, exist_ok=True)
    pool.to_csv(pool_path, index=False, compression="gzip")
    split_counts = {
        split: {
            "queries": int(pool.loc[pool["split"].eq(split), "query_id"].nunique()),
            "rows": int(pool["split"].eq(split).sum()),
        }
        for split in ("train", "valid", "test")
    }
    _write_json(
        output / "manifest.json",
        {
            "dataset": dataset,
            "model_id": model_id,
            "revision": revision,
            "candidate_generation": "label_blind_dense_retrieval",
            "labels_joined_after_top100_frozen": True,
            "split_policy": source["split_policy"],
            "splits": split_counts,
            "runtime_seconds": time.time() - started,
        },
    )
    return output


def _pool_rows(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: np.ndarray,
    indices: np.ndarray,
    gold: dict[str, set[str]],
) -> pd.DataFrame:
    count = indices.shape[1]
    query_ids = np.repeat(queries["id"].astype(str).to_numpy(), count)
    candidate_ids = candidates["id"].astype(str).to_numpy()[indices.reshape(-1)]
    return pd.DataFrame(
        {
            "split": np.repeat(queries["split"].astype(str).to_numpy(), count),
            "query_id": query_ids,
            "candidate_id": candidate_ids,
            "retrieval_rank": np.tile(np.arange(1, count + 1), len(queries)),
            "embedding_score": scores.reshape(-1),
            "label": [
                int(candidate_id in gold.get(query_id, set()))
                for query_id, candidate_id in zip(query_ids, candidate_ids)
            ],
        }
    )


def _attach_features(
    pool: pd.DataFrame, queries: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    query_fields = [
        column for column in _fields_for_frame(queries) if column in queries
    ]
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
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    features = structured_features(paired)
    return pd.concat([pool.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def _ranking_metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    reciprocal: list[float] = []
    ndcg: list[float] = []
    hits = {1: [], 5: [], 10: [], 100: []}
    for _, group in frame.groupby("query_id", sort=False):
        ordered = group.sort_values(
            [score, "retrieval_rank"],
            ascending=[False, True],
            kind="stable",
        )
        labels = ordered["label"].to_numpy(dtype=int)
        positives = np.flatnonzero(labels == 1)
        reciprocal.append(1.0 / (int(positives[0]) + 1))
        for k in hits:
            hits[k].append(float(positives[0] < k))
        discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
        dcg = float(np.sum(labels * discounts))
        ideal = np.sort(labels)[::-1]
        idcg = float(np.sum(ideal * discounts))
        ndcg.append(dcg / idcg)
    return {
        "mrr": float(np.mean(reciprocal)),
        "hits_at_1": float(np.mean(hits[1])),
        "hits_at_5": float(np.mean(hits[5])),
        "hits_at_10": float(np.mean(hits[10])),
        "hits_at_100": float(np.mean(hits[100])),
        "ndcg": float(np.mean(ndcg)),
        "queries": len(reciprocal),
    }


def _evaluate(frame: pd.DataFrame, score: str, sizes: list[int]) -> dict[str, Any]:
    total_queries = int(frame["query_id"].nunique())
    output: dict[str, Any] = {}
    for size in sizes:
        pool = frame[frame["retrieval_rank"].le(size)].copy()
        hits = pool.groupby("query_id")["label"].sum().gt(0)
        hit_ids = set(hits[hits].index)
        conditional = pool[pool["query_id"].isin(hit_ids)]
        pool_hit = float(len(hit_ids) / total_queries)
        output[str(size)] = {
            "pool_hit": pool_hit,
            f"hits_at_{size}_end_to_end": pool_hit,
            "pool_hit_queries": len(hit_ids),
            "total_queries": total_queries,
            "conditional": _ranking_metrics(conditional, score) if hit_ids else None,
        }
    return output


def _ensure_hits_at_100(value: Any) -> Any:
    """Backfill Hits@100 aliases into previously completed metric trees."""
    if isinstance(value, list):
        return [_ensure_hits_at_100(item) for item in value]
    if not isinstance(value, dict):
        return value
    updated = {key: _ensure_hits_at_100(item) for key, item in value.items()}
    for size in ("20", "50", "100"):
        record = updated.get(size)
        if not isinstance(record, dict) or "pool_hit" not in record:
            continue
        record[f"hits_at_{size}_end_to_end"] = float(record["pool_hit"])
        conditional = record.get("conditional")
        if isinstance(conditional, dict) and int(conditional.get("queries", 0)) > 0:
            conditional["hits_at_100"] = 1.0
    return updated


def _convex_weight(valid: pd.DataFrame) -> float:
    best = (-1.0, -1.0, 0.5)
    for weight in np.linspace(0, 1, 101):
        scores = weight * valid["embedding_score"] + (1 - weight) * valid["rule_score"]
        trial = valid.assign(_score=scores)
        hit_ids = set(trial.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
        value = _ranking_metrics(trial[trial["query_id"].isin(hit_ids)], "_score")["mrr"]
        candidate = (value, -abs(float(weight) - 0.5), float(weight))
        if candidate > best:
            best = candidate
    return best[2]


def run_exp02_dataset(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    revision: str,
) -> Path:
    from lightgbm import LGBMRanker, early_stopping, log_evaluation

    started = time.time()
    backbone_directory = _exp02_backbone_directory(model_id)
    exp02_root = project_root / "artifacts" / "exp02"
    if backbone_directory:
        pool_dir = (
            exp02_root
            / "candidate_pools_by_backbone"
            / backbone_directory
            / dataset
        )
    else:
        pool_dir = exp02_root / "candidate_pools" / dataset
    frame = pd.read_csv(pool_dir / "top100.csv.gz")
    feature_names = [
        column
        for column in frame.columns
        if column
        not in {"split", "query_id", "candidate_id", "retrieval_rank", "label"}
    ]
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    sizes = [int(value) for value in config["experiments"]["exp02_candidate_ranking"]["k"]]
    train = frame[frame["split"].eq("train")].copy()
    valid = frame[frame["split"].eq("valid")].copy()
    test = frame[frame["split"].eq("test")].copy()
    train_hit = set(train.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
    valid_hit = set(valid.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
    fit_train = train[train["query_id"].isin(train_hit)].sort_values(
        ["query_id", "retrieval_rank"]
    )
    fit_valid = valid[valid["query_id"].isin(valid_hit)].sort_values(
        ["query_id", "retrieval_rank"]
    )
    if backbone_directory:
        output = exp02_root / "ranking_by_backbone" / backbone_directory / dataset
    else:
        output = exp02_root / "ranking" / dataset
    output.mkdir(parents=True, exist_ok=True)
    baseline_scores = {
        "embedding": test["embedding_score"].to_numpy(),
        "rules": test["rule_score"].to_numpy(),
        "equal_fusion": 0.5
        * (test["embedding_score"].to_numpy() + test["rule_score"].to_numpy()),
    }
    weight = _convex_weight(valid)
    baseline_scores["convex_fusion"] = (
        weight * test["embedding_score"].to_numpy()
        + (1 - weight) * test["rule_score"].to_numpy()
    )
    preserved_methods: dict[str, Any] = {}
    existing_metrics_path = output / "metrics.json"
    if existing_metrics_path.exists():
        existing = json.loads(existing_metrics_path.read_text())
        for name in ("tuned_cross_encoder", "jina_cross_encoder"):
            if name in existing.get("methods", {}):
                preserved_methods[name] = _ensure_hits_at_100(
                    existing["methods"][name]
                )
    methods: dict[str, Any] = {}
    scored_test = test[
        ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
    ].copy()
    for name, scores in baseline_scores.items():
        scored_test[name] = scores
        methods[name] = _evaluate(scored_test.assign(_score=scores), "_score", sizes)
    methods["convex_fusion"]["validation_embedding_weight"] = weight

    seed_scores: dict[str, list[np.ndarray]] = {
        "hef_linear": [],
        "hef_gbdt": [],
        "hef_rank": [],
    }
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in seed_scores}
    train_groups = fit_train.groupby("query_id", sort=False).size().to_list()
    valid_groups = fit_valid.groupby("query_id", sort=False).size().to_list()
    for seed in seeds:
        models: dict[str, Any] = {
            "hef_linear": LogisticRegression(
                class_weight="balanced", max_iter=3000, random_state=seed
            ),
            "hef_gbdt": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=seed,
            ),
            "hef_rank": LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=20,
                reg_lambda=1.0,
                random_state=seed,
                verbosity=-1,
            ),
        }
        for name, model in models.items():
            if name == "hef_rank":
                model.fit(
                    fit_train[feature_names],
                    fit_train["label"],
                    group=train_groups,
                    eval_set=[(fit_valid[feature_names], fit_valid["label"])],
                    eval_group=[valid_groups],
                    eval_at=[1, 5, 10],
                    callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
                )
            else:
                model.fit(fit_train[feature_names], fit_train["label"])
            scores = (
                model.predict(test[feature_names])
                if name == "hef_rank"
                else model.predict_proba(test[feature_names])[:, 1]
            )
            seed_scores[name].append(np.asarray(scores, dtype=float))
            per_seed[name].append(
                {"seed": seed, "metrics": _evaluate(test.assign(_score=scores), "_score", sizes)}
            )
            joblib.dump(model, output / f"{name}_seed_{seed}.joblib")

    for name, values in seed_scores.items():
        mean_scores = np.mean(values, axis=0)
        scored_test[name] = mean_scores
        methods[name] = {
            "three_seed_ensemble": _evaluate(
                test.assign(_score=mean_scores), "_score", sizes
            ),
            "per_seed": per_seed[name],
        }
    methods.update(preserved_methods)
    scored_test.to_csv(output / "test_scored.csv.gz", index=False, compression="gzip")
    result = {
        "experiment": "exp02_candidate_ranking",
        "dataset": dataset,
        "status": "complete",
        "model_id": model_id,
        "revision": revision,
        "pool_sizes": sizes,
        "candidate_recall_separate": True,
        "conditional_metrics_exclude_pool_misses": True,
        "features": feature_names,
        "methods": methods,
        "runtime_seconds": time.time() - started,
    }
    _write_json(output / "metrics.json", result)
    _write_json(
        output / "manifest.json",
        {
            "dataset": dataset,
            "candidate_pool_manifest": str(pool_dir / "manifest.json"),
            "seeds": seeds,
            "objective": "lambdarank",
            "selection_split": "valid",
            "test_policy": "evaluated_after fixed retrieval and validation selection",
            "paper_eligible": True,
        },
    )
    return output


def score_exp02_cross_encoder(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    device: str = "cuda",
    batch_size: int = 0,
    target_memory_gib: float = 20.0,
    max_batch_size: int = 4096,
) -> Path:
    import subprocess

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pool_dir = project_root / "artifacts" / "exp02" / "candidate_pools" / dataset
    ranking_dir = project_root / "artifacts" / "exp02" / "ranking" / dataset
    pool = pd.read_csv(pool_dir / "top100.csv.gz")
    pool["query_id"] = pool["query_id"].astype(str)
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    source = _ranking_source(project_root, config, dataset)
    queries = source["queries"]
    candidates = source["candidates"]
    query_fields = [column for column in FIELDS if column in queries]
    candidate_fields = [column for column in FIELDS if column in candidates]
    left = queries[["id", *query_fields]].rename(
        columns={"id": "query_id", **{field: f"left_{field}" for field in query_fields}}
    )
    right = candidates[["id", *candidate_fields]].rename(
        columns={
            "id": "candidate_id",
            **{field: f"right_{field}" for field in candidate_fields},
        }
    )
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    test_mask = paired["split"].eq("test")
    test = pool.loc[test_mask].copy()
    left_text = serialize(paired.loc[test_mask], "left").tolist()
    right_text = serialize(paired.loc[test_mask], "right").tolist()
    torch_device = torch.device(device)
    sizes = [int(value) for value in config["experiments"]["exp02_candidate_ranking"]["k"]]
    spec = config["exp02_jina_cross_encoder"]
    model_id = str(spec["id"])
    revision = str(spec["revision"])
    max_length = int(spec["max_length"])
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,
    ).to(torch_device)
    model.eval()

    lengths = np.asarray(
        [len(left) + len(right) for left, right in zip(left_text, right_text)],
        dtype=np.int64,
    )
    order = np.argsort(-lengths, kind="stable")
    gib = 1024**3
    total_memory = (
        int(torch.cuda.get_device_properties(torch_device).total_memory)
        if torch_device.type == "cuda"
        else 0
    )
    target_bytes = int(target_memory_gib * gib)
    safe_limit = int(total_memory * 0.92) if total_memory else 0
    calibration: list[dict[str, Any]] = []

    def forward(indices: np.ndarray) -> tuple[np.ndarray, int]:
        encoded = tokenizer(
            [left_text[int(index)] for index in indices],
            [right_text[int(index)] for index in indices],
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded = {key: value.to(torch_device, non_blocking=True) for key, value in encoded.items()}
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                logits = model(**encoded).logits
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
            peak = int(torch.cuda.max_memory_allocated(torch_device))
        else:
            peak = 0
        if logits.ndim == 1 or logits.shape[-1] == 1:
            values = logits.float().reshape(-1)
        else:
            values = torch.softmax(logits.float(), dim=-1)[:, 1]
        return values.cpu().numpy(), peak

    if batch_size > 0:
        selected_batch = min(batch_size, len(order))
        candidates = [selected_batch]
    else:
        candidates = []
        value = min(32, len(order))
        while value and value <= min(max_batch_size, len(order)):
            candidates.append(value)
            next_value = min(max_batch_size, len(order), max(value + 8, int(value * 1.5) // 8 * 8))
            if next_value == value:
                break
            value = next_value

    selected_batch = 1
    for candidate in candidates:
        try:
            _, peak = forward(order[:candidate])
            calibration.append(
                {
                    "batch_size": int(candidate),
                    "peak_allocated_gib": peak / gib,
                    "status": "ok",
                }
            )
            if not safe_limit or peak <= safe_limit:
                selected_batch = candidate
            if target_bytes and peak >= target_bytes:
                break
        except torch.OutOfMemoryError:
            calibration.append({"batch_size": int(candidate), "status": "out_of_memory"})
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            break

    scores = np.empty(len(test), dtype=np.float32)
    batch_history: list[dict[str, Any]] = []
    cursor = 0
    current_batch = selected_batch
    while cursor < len(order):
        count = min(current_batch, len(order) - cursor)
        indices = order[cursor : cursor + count]
        try:
            values, peak = forward(indices)
        except torch.OutOfMemoryError:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            if current_batch <= 1:
                raise
            current_batch = max(1, current_batch // 2)
            continue
        scores[indices] = values
        cursor += count
        record: dict[str, Any] = {
            "rows_complete": int(cursor),
            "rows_total": int(len(order)),
            "batch_size": int(count),
            "peak_allocated_gib": peak / gib,
        }
        if torch_device.type == "cuda":
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            record["nvidia_smi"] = gpu.stdout.strip().splitlines()
        batch_history.append(record)
        print(json.dumps({"dataset": dataset, **record}), flush=True)
        if target_bytes and peak > 0 and peak < int(target_bytes * 0.85):
            scale = min(2.0, max(1.05, target_bytes / peak * 0.92))
            proposed = max(current_batch + 8, int(current_batch * scale) // 8 * 8)
            current_batch = min(max_batch_size, proposed)

    score_frame = test[
        ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
    ].copy()
    score_frame["jina_score"] = scores
    score_frame.to_csv(
        ranking_dir / "jina_cross_encoder_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics_path = ranking_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["methods"]["jina_cross_encoder"] = {
        "metrics": _evaluate(test.assign(_score=scores), "_score", sizes),
        "source": "public pretrained Jina reranker used without task-specific fine-tuning",
        "model_id": model_id,
        "revision": revision,
        "max_length": max_length,
        "initial_batch_size": int(selected_batch),
        "maximum_batch_size_used": int(
            max((item["batch_size"] for item in batch_history), default=selected_batch)
        ),
        "target_memory_gib": target_memory_gib,
        "calibration": calibration,
        "batch_history": batch_history,
        "runtime_seconds": time.time() - started,
    }
    _write_json(metrics_path, metrics)
    return ranking_dir


def score_exp02_roberta(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    device: str = "cuda",
    batch_size: int = 0,
    target_memory_gib: float = 20.0,
    max_batch_size: int = 4096,
) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pool_dir = project_root / "artifacts" / "exp02" / "candidate_pools" / dataset
    ranking_dir = project_root / "artifacts" / "exp02" / "ranking" / dataset
    pool = pd.read_csv(pool_dir / "top100.csv.gz")
    pool["query_id"] = pool["query_id"].astype(str)
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    source = _ranking_source(project_root, config, dataset)
    queries = source["queries"]
    candidates = source["candidates"]
    query_fields = [
        column for column in _fields_for_frame(queries) if column in queries
    ]
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
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    test_mask = paired["split"].eq("test")
    test = pool.loc[test_mask].copy()
    left_text = serialize(paired.loc[test_mask], "left").tolist()
    right_text = serialize(paired.loc[test_mask], "right").tolist()
    order = np.argsort(
        -np.asarray(
            [len(left_value) + len(right_value) for left_value, right_value in zip(left_text, right_text)],
            dtype=np.int64,
        ),
        kind="stable",
    )
    torch_device = torch.device(device)
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    sizes = [int(value) for value in config["experiments"]["exp02_candidate_ranking"]["k"]]
    max_length = int(config["cross_encoder"]["max_length"])
    gib = 1024**3
    target_bytes = int(target_memory_gib * gib)
    total_memory = (
        int(torch.cuda.get_device_properties(torch_device).total_memory)
        if torch_device.type == "cuda"
        else 0
    )
    safe_limit = int(total_memory * 0.92) if total_memory else 0
    all_scores: list[np.ndarray] = []
    per_seed: list[dict[str, Any]] = []
    runtime_started = time.time()

    for seed in seeds:
        model_dir = (
            project_root
            / "artifacts"
            / "exp01_cross_encoder"
            / dataset
            / f"seed_{seed}"
            / "model"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(torch_device)
        model.eval()

        def forward(indices: np.ndarray) -> tuple[np.ndarray, int]:
            encoded = tokenizer(
                [left_text[int(index)] for index in indices],
                [right_text[int(index)] for index in indices],
                padding=True,
                truncation=True,
                max_length=max_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(torch_device, non_blocking=True)
                for key, value in encoded.items()
            }
            if torch_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(torch_device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=torch.float16,
                    enabled=torch_device.type == "cuda",
                ):
                    logits = model(**encoded).logits
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
                peak = int(torch.cuda.max_memory_allocated(torch_device))
            else:
                peak = 0
            values = torch.softmax(logits.float(), dim=-1)[:, 1]
            return values.cpu().numpy(), peak

        if batch_size > 0:
            batch_candidates = [min(batch_size, len(order))]
        else:
            batch_candidates = []
            candidate = min(64, len(order))
            while candidate:
                batch_candidates.append(candidate)
                next_candidate = min(
                    max_batch_size,
                    len(order),
                    max(candidate + 8, int(candidate * 1.5) // 8 * 8),
                )
                if next_candidate == candidate:
                    break
                candidate = next_candidate

        calibration: list[dict[str, Any]] = []
        selected_batch = 1
        for candidate in batch_candidates:
            try:
                _, peak = forward(order[:candidate])
                calibration.append(
                    {
                        "batch_size": int(candidate),
                        "peak_allocated_gib": peak / gib,
                        "status": "ok",
                    }
                )
                if not safe_limit or peak <= safe_limit:
                    selected_batch = candidate
                if target_bytes and peak >= target_bytes:
                    break
            except torch.OutOfMemoryError:
                calibration.append(
                    {"batch_size": int(candidate), "status": "out_of_memory"}
                )
                if torch_device.type == "cuda":
                    torch.cuda.empty_cache()
                break

        scores = np.empty(len(order), dtype=np.float32)
        cursor = 0
        current_batch = selected_batch
        maximum_batch_used = selected_batch
        while cursor < len(order):
            count = min(current_batch, len(order) - cursor)
            indices = order[cursor : cursor + count]
            try:
                values, peak = forward(indices)
            except torch.OutOfMemoryError:
                if torch_device.type == "cuda":
                    torch.cuda.empty_cache()
                if current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                continue
            scores[indices] = values
            cursor += count
            maximum_batch_used = max(maximum_batch_used, count)
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "rows_complete": cursor,
                        "rows_total": len(order),
                        "batch_size": count,
                        "peak_allocated_gib": peak / gib,
                    }
                ),
                flush=True,
            )
            if target_bytes and peak > 0 and peak < int(target_bytes * 0.85):
                scale = min(2.0, max(1.05, target_bytes / peak * 0.92))
                proposed = max(current_batch + 8, int(current_batch * scale) // 8 * 8)
                current_batch = min(max_batch_size, proposed)

        all_scores.append(scores)
        per_seed.append(
            {
                "seed": seed,
                "metrics": _evaluate(test.assign(_score=scores), "_score", sizes),
                "initial_batch_size": selected_batch,
                "maximum_batch_size_used": maximum_batch_used,
                "calibration": calibration,
            }
        )
        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    mean_scores = np.mean(all_scores, axis=0)
    score_frame = test[
        ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
    ].copy()
    for seed, scores in zip(seeds, all_scores):
        score_frame[f"seed_{seed}"] = scores
    score_frame["mean_score"] = mean_scores
    score_frame.to_csv(
        ranking_dir / "roberta_cross_encoder_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics_path = ranking_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["methods"]["tuned_cross_encoder"] = {
        "three_seed_ensemble": _evaluate(
            test.assign(_score=mean_scores), "_score", sizes
        ),
        "per_seed": per_seed,
        "source": "Experiment 1 validation-selected final checkpoints",
        "model_id": str(config["cross_encoder"]["id"]),
        "revision": str(config["cross_encoder"]["revision"]),
        "target_memory_gib": target_memory_gib,
        "runtime_seconds": time.time() - runtime_started,
    }
    _write_json(metrics_path, metrics)
    return ranking_dir
