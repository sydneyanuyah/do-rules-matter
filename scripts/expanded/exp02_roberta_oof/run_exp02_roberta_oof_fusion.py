#!/usr/bin/env python3
"""Leakage-safe Exp2 HEF-GBDT with tuned-RoBERTa evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from paper1_hef.features import serialize


def _load_exp02_helpers() -> tuple[Any, Any, Any]:
    """Load repaired ranking helpers without mutating a host's shared package."""
    try:
        from paper1_hef.exp02 import _evaluate, _fields_for_frame, _ranking_source

        return _evaluate, _fields_for_frame, _ranking_source
    except ImportError:
        path = Path(__file__).with_name("exp02_compat.py")
        spec = importlib.util.spec_from_file_location("paper1_hef._roberta_exp02_compat", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load private Exp2 compatibility module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._evaluate, module._fields_for_frame, module._ranking_source


_evaluate, _fields_for_frame, _source = _load_exp02_helpers()


FOLDS = 3
MODEL_SLUG = "FacebookAI__roberta-base"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_pool(root: Path, dataset: str) -> tuple[pd.DataFrame, Path]:
    directory = root / "artifacts" / "exp02" / "candidate_pools" / dataset
    path = directory / "top100.csv.gz"
    manifest = directory / "manifest.json"
    if not path.exists() or not manifest.exists():
        raise FileNotFoundError(f"Locked E5 candidate pool is incomplete: {directory}")
    frame = pd.read_csv(path)
    required = {"split", "query_id", "candidate_id", "retrieval_rank", "embedding_score", "label"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Candidate pool missing columns: {sorted(missing)}")
    frame["query_id"] = frame["query_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.duplicated(["split", "query_id", "candidate_id"]).any():
        raise ValueError("Duplicate query-candidate keys in candidate pool")
    if not set(frame["split"].unique()) <= {"train", "valid", "test"}:
        raise ValueError("Unexpected split in candidate pool")
    if not set(frame["label"].unique()) <= {0, 1}:
        raise ValueError("Nonbinary labels in candidate pool")
    for split in ("train", "valid", "test"):
        ranks = frame.loc[frame["split"].eq(split)].groupby("query_id")["retrieval_rank"]
        if ranks.size().eq(0).any() or ranks.max().gt(100).any():
            raise ValueError(f"{split}: malformed top-100 pool")
    return frame, manifest


def attach_text(root: Path, config: dict[str, Any], dataset: str, pool: pd.DataFrame) -> pd.DataFrame:
    source = _source(root, config, dataset)
    queries, candidates = source["queries"], source["candidates"]
    qfields = [field for field in _fields_for_frame(queries) if field in queries]
    cfields = [field for field in _fields_for_frame(candidates) if field in candidates]
    left = queries[["id", *qfields]].copy()
    left["id"] = left["id"].astype(str)
    left = left.rename(columns={"id": "query_id", **{f: f"left_{f}" for f in qfields}})
    right = candidates[["id", *cfields]].copy()
    right["id"] = right["id"].astype(str)
    right = right.rename(columns={"id": "candidate_id", **{f: f"right_{f}" for f in cfields}})
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    paired["left_text"] = serialize(paired, "left")
    paired["right_text"] = serialize(paired, "right")
    if paired["left_text"].str.len().eq(0).all():
        paired["left_text"] = paired.apply(
            lambda row: " ".join(
                f"COL {field} VAL {row.get(f'left_{field}')}"
                for field in qfields
                if pd.notna(row.get(f"left_{field}")) and str(row.get(f"left_{field}")).strip()
            ), axis=1,
        )
    if paired["right_text"].str.len().eq(0).all():
        paired["right_text"] = paired.apply(
            lambda row: " ".join(
                f"COL {field} VAL {row.get(f'right_{field}')}"
                for field in cfields
                if pd.notna(row.get(f"right_{field}")) and str(row.get(f"right_{field}")).strip()
            ), axis=1,
        )
    if paired[["left_text", "right_text"]].isna().any().any():
        raise ValueError("Candidate pool text join produced null serialization")
    return paired


def query_folds(frame: pd.DataFrame, seed: int = 20260725) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically balance complete query records across three folds."""
    groups = frame.groupby("query_id", sort=True).agg(rows=("label", "size"), positives=("label", "sum"))
    if len(groups) < FOLDS:
        raise ValueError("Insufficient query groups for OOF")
    rng = random.Random(seed)
    items = [(str(q), int(row.rows), int(row.positives), rng.random()) for q, row in groups.iterrows()]
    items.sort(key=lambda x: (-x[1], -x[2], x[3], x[0]))
    buckets: list[list[str]] = [[] for _ in range(FOLDS)]
    rows = [0] * FOLDS
    positives = [0] * FOLDS
    target_rows = len(frame) / FOLDS
    target_pos = max(1.0, float(frame["label"].sum()) / FOLDS)
    for query, count, pos, _ in items:
        chosen = min(
            range(FOLDS),
            key=lambda f: ((rows[f] + count) / target_rows + (positives[f] + pos) / target_pos, rows[f], positives[f], f),
        )
        buckets[chosen].append(query)
        rows[chosen] += count
        positives[chosen] += pos
    assignment = np.full(len(frame), -1, dtype=np.int8)
    report: dict[str, Any] = {"group_unit": "query_record", "fold_assignment_seed": seed, "folds": []}
    query_values = frame["query_id"].astype(str)
    for fold, queries in enumerate(buckets):
        holdout = query_values.isin(queries).to_numpy()
        train_queries = set(query_values[~holdout])
        holdout_queries = set(query_values[holdout])
        overlap = train_queries & holdout_queries
        if overlap:
            raise AssertionError(f"Fold {fold}: query leakage")
        if len(np.unique(frame.loc[~holdout, "label"])) != 2:
            raise ValueError(f"Fold {fold}: training complement lacks a class")
        assignment[holdout] = fold
        report["folds"].append(
            {"fold": fold, "train_rows": int((~holdout).sum()), "holdout_rows": int(holdout.sum()),
             "train_queries": len(train_queries), "holdout_queries": len(holdout_queries),
             "holdout_positives": int(frame.loc[holdout, "label"].sum()), "query_overlap": 0}
        )
    if np.any(assignment < 0):
        raise AssertionError("OOF assignment incomplete")
    return assignment, report


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.left = frame["left_text"].tolist()
        self.right = frame["right_text"].tolist()
        self.labels = frame["label"].to_numpy(dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.tokenizer(self.left[index], self.right[index], truncation=True, max_length=self.max_length)
        item["labels"] = int(self.labels[index])
        return item


def fit_and_score(
    train: pd.DataFrame,
    score_frames: list[pd.DataFrame],
    model_id: str,
    revision: str,
    max_length: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    checkpoint: Path | None = None,
) -> list[np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(PairDataset(train, tokenizer, max_length), batch_size=batch_size,
                              shuffle=True, generator=generator, collate_fn=collator,
                              num_workers=0, pin_memory=device.type == "cuda")
    loaders = [DataLoader(PairDataset(frame, tokenizer, max_length), batch_size=batch_size * 4,
                          shuffle=False, collate_fn=collator, num_workers=0,
                          pin_memory=device.type == "cuda") for frame in score_frames]
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, num_labels=2).to(device)
    pos = float(train["label"].sum())
    neg = float(len(train) - pos)
    if pos <= 0 or neg <= 0:
        raise ValueError("Cross-encoder training requires both classes")
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg / pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * 0.1)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"event": "epoch", "epoch": epoch, "epochs": epochs, "mean_loss": float(np.mean(losses))}), flush=True)
    results: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for loader in loaders:
            values = []
            for batch in loader:
                batch.pop("labels")
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(**batch).logits
                # Ranking only requires a monotone score.  The positive-class
                # probability can round to a constant when both logits are far
                # into the saturated softmax regime (observed on Link-Lives).
                # The logit margin preserves the exact ordering and remains
                # finite/non-degenerate, so use it for OOF fusion and ranking.
                values.append((logits.float()[:, 1] - logits.float()[:, 0]).cpu().numpy())
            results.append(np.concatenate(values))
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
    del model
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / args.config).read_text())
    allowed = set(config["dataset_groups"]["exp01_all"]) | {"link_lives_release2"}
    if args.dataset not in allowed or args.seed not in {int(s) for s in config["protocol"]["seeds"]}:
        raise ValueError("Outside locked dataset/seed protocol")
    pool, pool_manifest = candidate_pool(root, args.dataset)
    paired = attach_text(root, config, args.dataset, pool)
    train = paired[paired["split"].eq("train")].reset_index(drop=True)
    valid = paired[paired["split"].eq("valid")].reset_index(drop=True)
    test = paired[paired["split"].eq("test")].reset_index(drop=True)
    # HEF learns only on queries whose retrieved pool contains a positive.
    hit_train = set(train.groupby("query_id")["label"].sum().loc[lambda x: x.gt(0)].index)
    fit_train = train[train["query_id"].isin(hit_train)].reset_index(drop=True)
    assignment, fold_report = query_folds(fit_train)

    exp01 = root / "artifacts" / "exp01_cross_encoder" / args.dataset / f"seed_{args.seed}"
    exp01_metrics_path = exp01 / "metrics.json"
    if not exp01_metrics_path.exists():
        raise FileNotFoundError(f"Exp1 tuned-RoBERTa selection absent: {exp01_metrics_path}")
    exp01_metrics = json.loads(exp01_metrics_path.read_text())
    learning_rate = float(exp01_metrics["selected_learning_rate"])
    epochs = int(exp01_metrics["selected_epoch"])
    spec = config["cross_encoder"]
    model_id, revision = str(spec["id"]), str(spec["revision"])
    max_length = int(spec["max_length"])

    output = root / "artifacts" / "exp02_hef_cross_evidence" / "v1" / args.dataset / "e5_plus_tuned_roberta" / f"seed_{args.seed}"
    if (output / "SUCCESS.json").exists():
        print(f"Already complete: {output}")
        return
    lock = output.parent / f".{output.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"Collision lock exists: {lock}") from exc
    stage = output.parent / f".{output.name}.partial-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        device = torch.device(args.device)
        oof = np.full(len(fit_train), np.nan, dtype=float)
        for fold in range(FOLDS):
            hold = np.flatnonzero(assignment == fold)
            complement = np.flatnonzero(assignment != fold)
            scores = fit_and_score(
                fit_train.iloc[complement].reset_index(drop=True),
                [fit_train.iloc[hold].reset_index(drop=True)],
                model_id, revision, max_length, learning_rate, epochs,
                args.seed * 10 + fold, args.batch_size, device,
            )[0]
            if len(scores) != len(hold):
                raise ValueError(f"Fold {fold}: OOF coverage mismatch")
            oof[hold] = scores
        if not np.isfinite(oof).all() or float(np.ptp(oof)) <= 1e-8:
            raise ValueError("OOF scores incomplete or degenerate")

        valid_score, test_score = fit_and_score(
            fit_train, [valid, test], model_id, revision, max_length,
            learning_rate, epochs, args.seed, args.batch_size, device,
            checkpoint=stage / "full_roberta_model",
        )
        if any(not np.isfinite(x).all() or float(np.ptp(x)) <= 1e-8 for x in (valid_score, test_score)):
            raise ValueError("Validation/test RoBERTa scores incomplete or degenerate")

        excluded = {"split", "query_id", "candidate_id", "retrieval_rank", "label", "left_text", "right_text"}
        features = [column for column in pool.columns if column not in excluded]
        if "embedding_score" not in features:
            raise ValueError("E5 embedding score absent from HEF features")
        train_x = fit_train[features].copy()
        valid_x = valid[features].copy()
        test_x = test[features].copy()
        train_x["tuned_roberta_score"] = oof
        valid_x["tuned_roberta_score"] = valid_score
        test_x["tuned_roberta_score"] = test_score
        if train_x.isna().any().any() or valid_x.isna().any().any() or test_x.isna().any().any():
            raise ValueError("HEF feature matrix contains null values")

        sizes = [int(k) for k in config["experiments"]["exp02_candidate_ranking"]["k"]]
        candidates = []
        for leaves in (7, 15, 31):
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=leaves,
                l2_regularization=1.0, early_stopping=True, random_state=args.seed,
            )
            model.fit(train_x, fit_train["label"])
            score = model.predict_proba(valid_x)[:, 1]
            metrics = _evaluate(valid.assign(_score=score), "_score", sizes)
            conditional = metrics["100"].get("conditional")
            selection = float(conditional.get("mrr", 0.0) if conditional else 0.0)
            candidates.append((selection, -leaves, leaves, model, score, metrics))
        _selection, _tie, selected_leaves, model, selected_valid_score, validation_metrics = max(candidates, key=lambda x: (x[0], x[1]))
        # The untouched test pool is accessed only after model selection.
        hef_test_score = model.predict_proba(test_x)[:, 1]
        test_metrics = _evaluate(test.assign(_score=hef_test_score), "_score", sizes)
        roberta_test_metrics = _evaluate(test.assign(_score=test_score), "_score", sizes)
        np.savez_compressed(
            stage / "oof_train_scores.npz", query_id=fit_train["query_id"].astype(str).to_numpy(),
            candidate_id=fit_train["candidate_id"].astype(str).to_numpy(), label=fit_train["label"].to_numpy(dtype=np.int8),
            fold=assignment, score=oof.astype(np.float32),
        )
        np.savez_compressed(
            stage / "scores.npz",
            valid_query_id=valid["query_id"].astype(str).to_numpy(), valid_candidate_id=valid["candidate_id"].astype(str).to_numpy(),
            valid_label=valid["label"].to_numpy(dtype=np.int8), valid_roberta_score=valid_score.astype(np.float32),
            valid_hef_score=selected_valid_score.astype(np.float32),
            test_query_id=test["query_id"].astype(str).to_numpy(), test_candidate_id=test["candidate_id"].astype(str).to_numpy(),
            test_label=test["label"].to_numpy(dtype=np.int8), test_roberta_score=test_score.astype(np.float32),
            test_hef_score=hef_test_score.astype(np.float32),
        )
        joblib.dump(model, stage / "hef_gbdt.joblib")
        metrics = {
            "experiment": "exp02_candidate_ranking_cross_evidence",
            "method": "hef_gbdt_with_e5_plus_tuned_roberta",
            "dataset": args.dataset, "seed": args.seed,
            "candidate_pool": "locked_e5_top100", "candidate_recall_separate": True,
            "features": list(train_x.columns),
            "roberta_hyperparameters_source": "matching Exp1 dataset/seed validation selection",
            "roberta_score_type": "positive_minus_negative_logit_margin",
            "selected_learning_rate": learning_rate, "selected_epoch": epochs,
            "oof_protocol": fold_report,
            "gbdt_candidates_max_leaf_nodes": [7, 15, 31], "selected_max_leaf_nodes": selected_leaves,
            "selection_metric": "validation_conditional_mrr_at_100",
            "validation": validation_metrics, "test": test_metrics,
            "tuned_roberta_test": roberta_test_metrics,
            "test_policy": "scored once after RoBERTa hyperparameters and HEF capacity were locked",
            "runtime_seconds": time.time() - started,
        }
        write_json(stage / "metrics.json", metrics)
        manifest = {
            "paper_eligible": True, "dataset": args.dataset, "seed": args.seed,
            "fit_unit": "query_record_group", "oof_rows": len(oof), "oof_folds": FOLDS,
            "candidate_pool_sha256": sha256(pool_manifest), "exp01_selection_sha256": sha256(exp01_metrics_path),
            "model_id": model_id, "model_revision": revision, "runner_sha256": sha256(Path(__file__)),
        }
        write_json(stage / "run_manifest.json", manifest)
        expected = [stage / name for name in ("metrics.json", "run_manifest.json", "scores.npz", "oof_train_scores.npz", "hef_gbdt.joblib")]
        if any(not path.exists() or path.stat().st_size == 0 for path in expected):
            raise RuntimeError("Expected artifact absent or empty")
        write_json(stage / "SUCCESS.json", {"validated": True, "completed_unix": time.time()})
        if output.exists():
            raise FileExistsError(output)
        os.replace(stage, output)
        print(json.dumps({"event": "complete", "output": str(output), "test": test_metrics}), flush=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
