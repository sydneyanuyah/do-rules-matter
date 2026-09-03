#!/usr/bin/env python3
"""Exp2 tuned-Jina reranking and leakage-safe HEF-GBDT score fusion.

The train-pool Jina evidence is produced out-of-fold by query-record group.
Validation selects Jina learning rate and epoch; test is scored once after all
choices and the fusion model are locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_ID = "jinaai/jina-reranker-v2-base-multilingual"
MODEL_REVISION = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fold_for(dataset: str, query_id: str, folds: int, seed: int) -> int:
    value = hashlib.sha256(f"{dataset}:{seed}:{query_id}".encode()).digest()
    return int.from_bytes(value[:8], "big") % folds


def attach_text(
    root: Path, config: dict[str, Any], dataset: str, pool: pd.DataFrame
) -> pd.DataFrame:
    from paper1_hef.exp02 import _fields_for_frame, _ranking_source as _source
    from paper1_hef.features import serialize

    source = _source(root, config, dataset)
    queries, candidates = source["queries"], source["candidates"]
    query_fields = [c for c in _fields_for_frame(queries) if c in queries]
    candidate_fields = [c for c in _fields_for_frame(candidates) if c in candidates]
    left = queries[["id", *query_fields]].rename(
        columns={"id": "query_id", **{f: f"left_{f}" for f in query_fields}}
    )
    right = candidates[["id", *candidate_fields]].rename(
        columns={"id": "candidate_id", **{f: f"right_{f}" for f in candidate_fields}}
    )
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    paired["left_text"] = serialize(paired, "left")
    paired["right_text"] = serialize(paired, "right")
    # The shared serializer is product-schema oriented. Genealogical frames use
    # domain-specific columns, so fall back to the exact fields selected above.
    if paired["left_text"].str.len().eq(0).all():
        paired["left_text"] = paired.apply(
            lambda row: " ".join(
                f"COL {field} VAL {row.get(f'left_{field}')}"
                for field in query_fields
                if pd.notna(row.get(f"left_{field}")) and str(row.get(f"left_{field}")).strip()
            ), axis=1,
        )
    if paired["right_text"].str.len().eq(0).all():
        paired["right_text"] = paired.apply(
            lambda row: " ".join(
                f"COL {field} VAL {row.get(f'right_{field}')}"
                for field in candidate_fields
                if pd.notna(row.get(f"right_{field}")) and str(row.get(f"right_{field}")).strip()
            ), axis=1,
        )
    if (
        paired["left_text"].str.len().eq(0).all()
        or paired["right_text"].str.len().eq(0).all()
    ):
        raise RuntimeError(
            f"{dataset}: empty serialized side (domain-field selection failure)"
        )
    return paired


def training_subset(frame: pd.DataFrame, negatives_per_query: int) -> pd.DataFrame:
    ordered = frame.sort_values(["query_id", "retrieval_rank"], kind="stable")
    parts = []
    for _, group in ordered.groupby("query_id", sort=False):
        positives = group[group["label"].eq(1)]
        if positives.empty:
            continue
        parts.append(positives)
        parts.append(group[group["label"].eq(0)].head(negatives_per_query))
    if not parts:
        raise RuntimeError("No positive-containing training query groups")
    return pd.concat(parts, ignore_index=True)


def ranking_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    reciprocal, h1, h5, h10, hit_queries = [], [], [], [], 0
    for _, group in frame.groupby("query_id", sort=False):
        if not group["label"].any():
            continue
        hit_queries += 1
        ordered = group.sort_values(
            [score_column, "candidate_id"], ascending=[False, True], kind="stable"
        )
        ranks = np.flatnonzero(ordered["label"].to_numpy(dtype=bool)) + 1
        best = int(ranks.min())
        reciprocal.append(1.0 / best)
        h1.append(best <= 1)
        h5.append(best <= 5)
        h10.append(best <= 10)
    total_queries = int(frame["query_id"].nunique())
    pool_hit = hit_queries / total_queries if total_queries else 0.0
    return {
        "queries": total_queries,
        "pool_hit_queries": hit_queries,
        "pool_hit": pool_hit,
        "mrr_conditional": float(np.mean(reciprocal)) if reciprocal else 0.0,
        "hits_at_1_conditional": float(np.mean(h1)) if h1 else 0.0,
        "hits_at_5_conditional": float(np.mean(h5)) if h5 else 0.0,
        "hits_at_10_conditional": float(np.mean(h10)) if h10 else 0.0,
        "hits_at_100_conditional": 1.0 if reciprocal else 0.0,
        "hits_at_100_end_to_end": pool_hit,
    }


class PairRows:
    def __init__(self, frame: pd.DataFrame):
        self.left = frame["left_text"].fillna("").astype(str).tolist()
        self.right = frame["right_text"].fillna("").astype(str).tolist()
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.lengths = np.asarray(
            [len(a) + len(b) for a, b in zip(self.left, self.right)]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, str, float, int]:
        return self.left[index], self.right[index], float(self.labels[index]), index


def collator(tokenizer: Any, max_length: int):
    import torch

    def collect(rows: list[tuple[str, str, float, int]]) -> dict[str, Any]:
        left, right, labels, indices = zip(*rows)
        encoded = tokenizer(
            list(left),
            list(right),
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
        encoded["row_indices"] = torch.tensor(indices, dtype=torch.int64)
        return encoded

    return collect


def load_model(device: Any):
    import torch
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)


def logits1(logits: Any) -> Any:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise RuntimeError(f"Expected single relevance logit, got {tuple(logits.shape)}")


def loaders(
    rows: PairRows, tokenizer: Any, batch_size: int, max_length: int, shuffle: bool
):
    from torch.utils.data import DataLoader, Sampler

    class BatchSampler(Sampler[list[int]]):
        def __iter__(self):
            order = np.argsort(rows.lengths, kind="stable")
            batches = [
                order[i : i + batch_size].tolist()
                for i in range(0, len(order), batch_size)
            ]
            if shuffle:
                random.shuffle(batches)
            yield from batches

        def __len__(self):
            return int(np.ceil(len(rows) / batch_size))

    sampler = BatchSampler()
    return DataLoader(
        rows,
        batch_sampler=sampler,
        collate_fn=collator(tokenizer, max_length),
        num_workers=2,
        pin_memory=True,
        persistent_workers=len(sampler) > 1,
    )


def infer(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    batch_size: int,
    max_length: int,
    device: Any,
) -> np.ndarray:
    import torch

    rows = PairRows(frame)
    result = np.empty(len(rows), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for batch in loaders(rows, tokenizer, batch_size, max_length, False):
            indices = batch.pop("row_indices").numpy()
            batch.pop("labels")
            encoded = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                values = torch.sigmoid(logits1(model(**encoded).logits).float())
            result[indices] = values.cpu().numpy()
    if not np.isfinite(result).all() or float(np.std(result)) <= 1e-8:
        raise RuntimeError("Non-finite or degenerate Jina ranking scores")
    return result


def fit_epochs(
    train: pd.DataFrame,
    tokenizer: Any,
    device: Any,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    max_length: int,
    seed: int,
) -> Any:
    import torch
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    model = load_model(device)
    rows = PairRows(train)
    loader = loaders(rows, tokenizer, batch_size, max_length, True)
    positives = float(rows.labels.sum())
    pos_weight = torch.tensor((len(rows) - positives) / positives, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    total = max(1, epochs * len(loader))
    schedule = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total * 0.05)), num_training_steps=total
    )
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch.pop("row_indices")
            labels = batch.pop("labels").to(device, non_blocking=True)
            encoded = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(logits1(model(**encoded).logits).float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
        print(
            json.dumps({"epoch": epoch, "epochs": epochs, "train_rows": len(train)}),
            flush=True,
        )
    return model


def calibrate_training_batch(
    frame: pd.DataFrame,
    tokenizer: Any,
    device: Any,
    max_length: int,
    maximum: int,
    target_gib: float,
) -> tuple[int, list[dict[str, Any]]]:
    import torch

    rows = PairRows(frame)
    longest = np.argsort(-rows.lengths, kind="stable")
    positive = float(rows.labels.sum())
    pos_weight = torch.tensor((len(rows) - positive) / positive, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = load_model(device)
    audit: list[dict[str, Any]] = []
    chosen = 1
    for size in (4, 8, 16, 32, 64, 128):
        if size > maximum or size > len(rows):
            break
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            batch = collator(tokenizer, max_length)(
                [rows[int(i)] for i in longest[:size]]
            )
            batch.pop("row_indices")
            labels = batch.pop("labels").to(device)
            encoded = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(logits1(model(**encoded).logits).float(), labels)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            audit.append(
                {"batch_size": size, "peak_allocated_gib": peak, "status": "ok"}
            )
            if peak <= 21.5:
                chosen = size
            if peak >= target_gib:
                break
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            audit.append({"batch_size": size, "status": "out_of_memory"})
            break
    del model
    torch.cuda.empty_cache()
    return chosen, audit


def select_final(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    tokenizer: Any,
    device: Any,
    learning_rates: list[float],
    max_epochs: int,
    patience: int,
    batch_size: int,
    eval_batch_size: int,
    max_length: int,
    seed: int,
    staging: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    best = None
    trials = []
    selected = staging / "model"
    for lr in learning_rates:
        set_seed(seed)
        model = load_model(device)
        rows = PairRows(train)
        loader = loaders(rows, tokenizer, batch_size, max_length, True)
        positives = float(rows.labels.sum())
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((len(rows) - positives) / positives, device=device)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total = max(1, max_epochs * len(loader))
        schedule = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total * 0.05)),
            num_training_steps=total,
        )
        scaler = torch.amp.GradScaler("cuda")
        stale, trial_best = 0, -1.0
        epochs = []
        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch in loader:
                batch.pop("row_indices")
                labels = batch.pop("labels").to(device, non_blocking=True)
                encoded = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = loss_fn(logits1(model(**encoded).logits).float(), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
            scores = infer(model, tokenizer, valid, eval_batch_size, max_length, device)
            metric = ranking_metrics(valid.assign(_score=scores), "_score")
            record = {"learning_rate": lr, "epoch": epoch, "validation": metric}
            epochs.append(record)
            value = metric["mrr_conditional"]
            if value > trial_best + 1e-12:
                trial_best, stale = value, 0
            else:
                stale += 1
            if best is None or value > best["validation"]["mrr_conditional"] + 1e-12:
                best = dict(record)
                if selected.exists():
                    shutil.rmtree(selected)
                model.save_pretrained(selected)
                tokenizer.save_pretrained(selected)
            if stale >= patience:
                break
        trials.append({"learning_rate": lr, "epochs": epochs})
        del model, loader, optimizer, schedule, scaler
        torch.cuda.empty_cache()
    assert best is not None
    best["trials"] = trials
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        selected, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    valid_scores = infer(model, tokenizer, valid, eval_batch_size, max_length, device)
    del model
    torch.cuda.empty_cache()
    return best, valid_scores


def main() -> None:
    import torch
    import yaml
    from sklearn.ensemble import HistGradientBoostingClassifier
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--learning-rates", default="1e-5,2e-5")
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--target-memory-gib", type=float, default=20.0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.seed not in [int(v) for v in config["protocol"]["seeds"]]:
        raise ValueError("Seed is outside locked protocol")
    output = (
        root
        / "artifacts"
        / "exp02_jina_finetuned_oof"
        / "v1"
        / args.dataset
        / f"seed_{args.seed}"
    )
    lock = output.parent / f".{output.name}.lock"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    staging = output.parent / f".{output.name}.partial.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    started = time.time()
    try:
        pool_path = (
            root
            / "artifacts"
            / "exp02"
            / "candidate_pools"
            / args.dataset
            / "top100.csv.gz"
        )
        pool = pd.read_csv(pool_path, dtype={"query_id": str, "candidate_id": str})
        paired = attach_text(root, config, args.dataset, pool)
        train = paired[paired["split"].eq("train")].reset_index(drop=True)
        valid = paired[paired["split"].eq("valid")].reset_index(drop=True)
        test = paired[paired["split"].eq("test")].reset_index(drop=True)
        device = torch.device(args.device)
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        )
        train_sample = training_subset(train, args.negatives_per_query)
        if args.batch_size > 0:
            train_batch_size = args.batch_size
            batch_calibration = [
                {"status": "user_locked", "batch_size": args.batch_size}
            ]
        else:
            train_batch_size, batch_calibration = calibrate_training_batch(
                train_sample,
                tokenizer,
                device,
                args.max_length,
                args.max_batch_size,
                args.target_memory_gib,
            )
        best, valid_jina = select_final(
            train_sample,
            valid,
            tokenizer,
            device,
            [float(v) for v in args.learning_rates.split(",")],
            args.max_epochs,
            args.patience,
            train_batch_size,
            args.eval_batch_size,
            args.max_length,
            args.seed,
            staging,
        )

        # Query records, rather than pair rows, are the indivisible OOF unit.
        query_fold = {
            q: fold_for(args.dataset, q, args.folds, args.seed)
            for q in train["query_id"].drop_duplicates()
        }
        train_jina = np.empty(len(train), dtype=np.float32)
        fold_audit = []
        for fold in range(args.folds):
            held_queries = {q for q, f in query_fold.items() if f == fold}
            fit = train_sample[~train_sample["query_id"].isin(held_queries)]
            held = train[train["query_id"].isin(held_queries)]
            if fit.empty or held.empty:
                raise RuntimeError(f"Empty OOF fold {fold}")
            model = fit_epochs(
                fit,
                tokenizer,
                device,
                float(best["learning_rate"]),
                int(best["epoch"]),
                train_batch_size,
                args.max_length,
                args.seed + fold + 1,
            )
            values = infer(
                model, tokenizer, held, args.eval_batch_size, args.max_length, device
            )
            train_jina[held.index.to_numpy()] = values
            fold_audit.append(
                {
                    "fold": fold,
                    "fit_queries": int(fit["query_id"].nunique()),
                    "held_queries": len(held_queries),
                    "held_rows": len(held),
                    "query_overlap": len(set(fit["query_id"]) & held_queries),
                }
            )
            del model
            torch.cuda.empty_cache()
        if not np.isfinite(train_jina).all() or float(np.std(train_jina)) <= 1e-8:
            raise RuntimeError("OOF train score coverage/variance failure")

        final_model = AutoModelForSequenceClassification.from_pretrained(
            staging / "model", trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)
        # First and only test access after Jina selection and all OOF work.
        test_jina = infer(
            final_model, tokenizer, test, args.eval_batch_size, args.max_length, device
        )
        del final_model
        torch.cuda.empty_cache()

        excluded = {
            "split",
            "query_id",
            "candidate_id",
            "retrieval_rank",
            "label",
            "left_text",
            "right_text",
        }
        base_features = [c for c in pool.columns if c not in excluded]
        feature_names = [*base_features, "tuned_jina_score"]
        train_fusion = train.copy()
        valid_fusion = valid.copy()
        test_fusion = test.copy()
        train_fusion["tuned_jina_score"] = train_jina
        valid_fusion["tuned_jina_score"] = valid_jina
        test_fusion["tuned_jina_score"] = test_jina
        train_hit_queries = set(
            train_fusion.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index
        )
        fit_fusion = train_fusion[train_fusion["query_id"].isin(train_hit_queries)]
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=args.seed,
            class_weight="balanced",
        )
        model.fit(fit_fusion[feature_names], fit_fusion["label"])
        valid_hef = model.predict_proba(valid_fusion[feature_names])[:, 1]
        test_hef = model.predict_proba(test_fusion[feature_names])[:, 1]
        joblib.dump(model, staging / "hef_gbdt_e5_plus_tuned_jina.joblib")

        score_frame = pd.concat(
            [
                train[
                    ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
                ].assign(tuned_jina_score=train_jina),
                valid[
                    ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
                ].assign(
                    tuned_jina_score=valid_jina,
                    hef_gbdt_e5_plus_tuned_jina_score=valid_hef,
                ),
                test[
                    ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
                ].assign(
                    tuned_jina_score=test_jina,
                    hef_gbdt_e5_plus_tuned_jina_score=test_hef,
                ),
            ],
            ignore_index=True,
        )
        score_frame.to_csv(staging / "scores.csv.gz", index=False, compression="gzip")
        metrics = {
            "experiment": "exp02_candidate_ranking",
            "status": "complete",
            "paper_eligible": True,
            "dataset": args.dataset,
            "seed": args.seed,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "oof_policy": "five-fold SHA256 query-record-grouped; zero query overlap",
            "folds": fold_audit,
            "selection": best,
            "batch_calibration": batch_calibration,
            "train_batch_size": train_batch_size,
            "methods": {
                "jina_finetuned": {
                    "validation": ranking_metrics(
                        valid.assign(_score=valid_jina), "_score"
                    ),
                    "test": ranking_metrics(test.assign(_score=test_jina), "_score"),
                },
                "hef_gbdt_e5_plus_tuned_jina": {
                    "features": feature_names,
                    "train_jina_evidence": "query-grouped out-of-fold only",
                    "validation": ranking_metrics(
                        valid.assign(_score=valid_hef), "_score"
                    ),
                    "test": ranking_metrics(test.assign(_score=test_hef), "_score"),
                },
            },
            "coverage": {
                "train": {"expected": len(train), "scored": len(train_jina)},
                "valid": {"expected": len(valid), "scored": len(valid_jina)},
                "test": {"expected": len(test), "scored": len(test_jina)},
            },
            "test_policy": "scored once after validation selection and OOF generation",
            "runtime_seconds": time.time() - started,
        }
        write_json(staging / "metrics.json", metrics)
        write_json(
            staging / "run_manifest.json",
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "config_sha256": sha256(config_path),
                "runner_sha256": sha256(Path(__file__)),
                "candidate_pool_sha256": sha256(pool_path),
                "hostname": socket.gethostname(),
                "paper_eligible": True,
            },
        )
        write_json(
            staging / "COMPLETED.json",
            {
                "status": "complete",
                "metrics_sha256": sha256(staging / "metrics.json"),
                "scores_sha256": sha256(staging / "scores.csv.gz"),
            },
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
        print(f"COMPLETE {output}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
