#!/usr/bin/env python3
"""Record-component OOF HEF fusion for tuned RoBERTa and fine-tuned Jina."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
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

from paper1_hef.data import load_dataset, validate_splits
from paper1_hef.evaluate import classification_metrics, select_threshold
from paper1_hef.features import serialize, structured_features


E5_ID = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
JINA_ID = "jinaai/jina-reranker-v2-base-multilingual"
JINA_REVISION = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"
FOLDS = 3


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: str, b: str) -> None:
        left, right = self.find(a), self.find(b)
        if left != right:
            self.parent[right] = left


@dataclass(frozen=True)
class FoldPlan:
    assignment: np.ndarray
    report: dict[str, Any]


def record_nodes(frame: pd.DataFrame, adapter: str) -> tuple[np.ndarray, np.ndarray]:
    left = frame["left_id"].astype(str).to_numpy(dtype=str)
    right = frame["right_id"].astype(str).to_numpy(dtype=str)
    if adapter == "deepmatcher":
        left = np.char.add("L:", left)
        right = np.char.add("R:", right)
    return left, right


def component_folds(frame: pd.DataFrame, adapter: str, seed: int = 20260725) -> FoldPlan:
    left, right = record_nodes(frame, adapter)
    dsu = DSU()
    for a, b in zip(left, right, strict=True):
        dsu.union(str(a), str(b))
    components: dict[str, list[int]] = {}
    for index, node in enumerate(left):
        components.setdefault(dsu.find(str(node)), []).append(index)
    labels = frame["label"].to_numpy(dtype=np.int8)
    # WDC's crawl graph can collapse into one connected component through shared
    # offers.  In that case, a both-side component split is mathematically
    # impossible.  Fall back only to query-side groups: every query is held out
    # wholesale, so no query's pairs leak from train into its OOF score.  The
    # manifest records this weaker but still leakage-safe-for-ranking protocol.
    if len(components) < FOLDS:
        if adapter != "wdc":
            raise ValueError(
                f"Strict both-side record-component OOF requires >=3 components; found {len(components)}"
            )
        groups: dict[str, list[int]] = {}
        for index, query_id in enumerate(left):
            groups.setdefault(str(query_id), []).append(index)
        if len(groups) < FOLDS:
            raise ValueError(
                f"WDC query-grouped OOF requires >=3 distinct queries; found {len(groups)}"
            )
        components = groups
        grouping = "query_side_groups_fallback_after_single_connected_component"
    else:
        grouping = "connected_components_of_both_record_sides"
    rng = random.Random(seed)
    items = [(root, rows, int(labels[rows].sum()), rng.random()) for root, rows in components.items()]
    items.sort(key=lambda x: (-len(x[1]), -x[2], x[3], x[0]))
    buckets: list[list[int]] = [[] for _ in range(FOLDS)]
    positives = [0] * FOLDS
    target_rows = len(frame) / FOLDS
    target_pos = max(1.0, float(labels.sum()) / FOLDS)
    for _root, rows, pos, _jitter in items:
        chosen = min(
            range(FOLDS),
            key=lambda f: ((len(buckets[f]) + len(rows)) / target_rows + (positives[f] + pos) / target_pos,
                           len(buckets[f]), positives[f], f),
        )
        buckets[chosen].extend(rows)
        positives[chosen] += pos
    assignment = np.full(len(frame), -1, dtype=np.int8)
    report: dict[str, Any] = {
        "grouping": grouping,
        "fold_assignment_seed": seed,
        "component_count": len(components),
        "folds": [],
    }
    all_rows = set(range(len(frame)))
    for fold, rows in enumerate(buckets):
        hold = np.asarray(sorted(rows), dtype=int)
        train = np.asarray(sorted(all_rows - set(rows)), dtype=int)
        if not len(hold) or len(np.unique(labels[train])) != 2:
            raise ValueError(f"Fold {fold} is empty or its complement lacks a class")
        train_records = set(left[train]) | set(right[train])
        hold_records = set(left[hold]) | set(right[hold])
        overlap = train_records & hold_records
        if grouping == "connected_components_of_both_record_sides" and overlap:
            raise AssertionError(f"Fold {fold}: {len(overlap)} record IDs leak")
        query_overlap = set(left[train]) & set(left[hold])
        if query_overlap:
            raise AssertionError(f"Fold {fold}: {len(query_overlap)} query IDs leak")
        assignment[hold] = fold
        report["folds"].append(
            {"fold": fold, "train_rows": len(train), "holdout_rows": len(hold),
             "train_positives": int(labels[train].sum()), "holdout_positives": int(labels[hold].sum()),
             "record_overlap": 0 if grouping == "connected_components_of_both_record_sides" else len(overlap),
             "query_overlap": 0}
        )
    if np.any(assignment < 0):
        raise AssertionError("OOF assignment does not cover every train row")
    return FoldPlan(assignment, report)


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.left = serialize(frame, "left").fillna("").astype(str).tolist()
        self.right = serialize(frame, "right").fillna("").astype(str).tolist()
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.tokenizer(self.left[index], self.right[index], truncation=True, max_length=self.max_length)
        item["labels"] = float(self.labels[index])
        return item


def train_fold(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    family: str,
    model_id: str,
    revision: str,
    max_length: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    trust = family == "jina"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PairDataset(train, tokenizer, max_length), batch_size=batch_size, shuffle=True,
        generator=generator, collate_fn=collator, num_workers=0, pin_memory=True,
    )
    score_loader = DataLoader(
        PairDataset(holdout, tokenizer, max_length), batch_size=batch_size * 2,
        shuffle=False, collate_fn=collator, num_workers=0, pin_memory=True,
    )
    kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": trust}
    if family == "roberta":
        kwargs["num_labels"] = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs).to(device)
    pos = float(train["label"].sum()); neg = float(len(train) - pos)
    if pos <= 0 or neg <= 0:
        raise ValueError("Fold training data lacks a class")
    if family == "roberta":
        loss_fn: Any = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg / pos], device=device))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    weight_decay = 0.01 if family == "jina" else 0.0
    warmup_ratio = 0.05 if family == "jina" else 0.10
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total * warmup_ratio)), total)
    # Jina's remote model may keep BF16 parameters.  CUDA GradScaler cannot
    # unscale BF16 gradients; disabled scaling retains ordinary FP32 optimizer
    # updates while the explicit float16 autocast remains in effect.
    scaler = torch.amp.GradScaler("cuda", enabled=family == "roberta")
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            encoded = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encoded).logits
                if family == "roberta":
                    loss = loss_fn(logits, labels.long())
                else:
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        logits = logits[:, 0]
                    if logits.ndim != 1:
                        raise RuntimeError(f"Jina expected one logit, got {tuple(logits.shape)}")
                    loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"event": "fold_epoch", "family": family, "epoch": epoch,
                          "epochs": epochs, "mean_loss": float(np.mean(losses))}), flush=True)
    values = []
    model.eval()
    with torch.inference_mode():
        for batch in score_loader:
            batch.pop("labels")
            encoded = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encoded).logits
            if family == "roberta":
                score = torch.softmax(logits.float(), dim=1)[:, 1]
            else:
                score = torch.sigmoid(logits.float().reshape(-1))
            values.append(score.cpu().numpy())
    result = np.concatenate(values)
    del model
    torch.cuda.empty_cache()
    return result


def aligned_scores(path: Path, splits: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    output = []
    for split in ("valid", "test"):
        if not np.array_equal(payload[f"{split}_pair_id"].astype(str), splits[split]["pair_id"].astype(str).to_numpy()):
            raise ValueError(f"{split}: neural pair IDs misaligned")
        if not np.array_equal(payload[f"{split}_label"].astype(np.int8), splits[split]["label"].to_numpy(dtype=np.int8)):
            raise ValueError(f"{split}: neural labels misaligned")
        scores = payload[f"{split}_score"].astype(float)
        if not np.isfinite(scores).all() or float(np.ptp(scores)) <= 1e-8:
            raise ValueError(f"{split}: neural scores nonfinite or degenerate")
        output.append(scores)
    return output[0], output[1]


def aligned_e5(root: Path, dataset: str, splits: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    directory = root / "artifacts" / "embeddings" / dataset / E5_ID.replace("/", "__") / E5_REVISION
    output = {}
    for split, frame in splits.items():
        payload = np.load(directory / f"{split}.npz", allow_pickle=True)
        source_ids = payload["pair_id"].astype(str)
        target_ids = frame["pair_id"].astype(str).to_numpy()
        if np.array_equal(source_ids, target_ids):
            if not np.array_equal(payload["label"].astype(np.int8), frame["label"].to_numpy(dtype=np.int8)):
                raise ValueError(f"{split}: E5 labels misaligned")
            output[split] = payload["embedding_score"].astype(float)
        else:
            if len(set(source_ids)) != len(source_ids):
                raise ValueError(f"{split}: duplicate E5 pair IDs")
            lookup = {pair_id: (float(score), int(label)) for pair_id, score, label in zip(source_ids, payload["embedding_score"], payload["label"])}
            if any(pair_id not in lookup for pair_id in target_ids):
                raise ValueError(f"{split}: selected pair absent from E5 scores")
            expected_labels = frame["label"].to_numpy(dtype=np.int8)
            mapped = [lookup[pair_id] for pair_id in target_ids]
            if not np.array_equal(np.asarray([x[1] for x in mapped], dtype=np.int8), expected_labels):
                raise ValueError(f"{split}: mapped E5 labels misaligned")
            output[split] = np.asarray([x[0] for x in mapped], dtype=float)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--family", choices=("roberta", "jina"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exp05-fraction", type=float)
    args = parser.parse_args()
    started = time.time()
    root = args.project_root.resolve()
    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text())
    allowed = set(config["dataset_groups"]["exp01_all"]) | {"link_lives_release2"}
    if args.dataset not in allowed or args.seed not in {int(x) for x in config["protocol"]["seeds"]}:
        raise ValueError("Dataset or seed outside locked protocol")
    spec = config["datasets"][args.dataset]
    splits = load_dataset(root / config["project"]["data_root"], spec)
    split_validation = validate_splits(
        splits, enforce_offer_disjoint=args.dataset == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
        enforce_record_disjoint=spec["adapter"] == "link_lives",
    )
    subset_metadata = None
    if args.exp05_fraction is not None:
        from paper1_hef.exp05 import _nested_subsets, _pair_id_hash
        exp05 = config["experiments"]["exp05_label_efficiency"]
        fraction = float(args.exp05_fraction)
        locked = sorted({float(x) for x in exp05["fractions"]})
        if fraction not in locked:
            raise ValueError(f"fraction {fraction} is outside locked Experiment 5 schedule")
        train = splits["train"].reset_index(drop=True)
        _, subsets = _nested_subsets(train["left_id"].astype(str).to_numpy(), train["label"].to_numpy(dtype=np.int64), locked, int(exp05["subset_seed"]))
        selected = train.iloc[subsets[fraction]].reset_index(drop=True)
        subset_metadata = {"requested_fraction": fraction, "selected_pair_count": int(len(selected)), "selected_group_count": int(selected["left_id"].astype(str).nunique()), "pair_id_sha256": _pair_id_hash(selected["pair_id"].astype(str).to_numpy()), "subset_seed": int(exp05["subset_seed"])}
        splits = dict(splits)
        splits["train"] = selected
    plan = component_folds(splits["train"], spec["adapter"])
    if args.family == "roberta":
        if args.exp05_fraction is None:
            full_dir = root / "artifacts" / "exp01_cross_encoder" / args.dataset / f"seed_{args.seed}"
        else:
            fraction_key = f"{int(round(float(args.exp05_fraction) * 100)):03d}"
            full_dir = root / "artifacts" / "exp05_roberta" / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
        model_id = str(config["cross_encoder"]["id"]); revision = str(config["cross_encoder"]["revision"])
        max_length = int(config["cross_encoder"]["max_length"])
    else:
        if args.exp05_fraction is None:
            full_dir = root / "artifacts" / "exp01_jina_finetuned" / "v1" / args.dataset / f"seed_{args.seed}"
        else:
            fraction_key = f"{int(round(float(args.exp05_fraction) * 100)):03d}"
            full_dir = root / "artifacts" / "exp05_expanded" / "v1" / "jina_finetuned" / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
        model_id = JINA_ID; revision = JINA_REVISION
        max_length = 1024
    metrics_path, scores_path = full_dir / "metrics.json", full_dir / "scores.npz"
    if not metrics_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"Completed full-fit {args.family} artifact missing: {full_dir}")
    full_metrics = json.loads(metrics_path.read_text())
    learning_rate = float(full_metrics["selected_learning_rate"])
    epochs = int(full_metrics["selected_epoch"])
    full_valid, full_test = aligned_scores(scores_path, splits)
    e5 = aligned_e5(root, args.dataset, splits)

    if args.exp05_fraction is None:
        output = root / "artifacts" / "exp01_hef_neural_oof" / "v1" / args.dataset / f"e5_plus_{args.family}" / f"seed_{args.seed}"
        experiment_name = "exp01_hef_gbdt_neural_oof"
    else:
        family_name = f"hef_gbdt_e5_finetuned_{args.family}_oof"
        output = root / "artifacts" / "exp05_expanded" / "v1" / family_name / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
        experiment_name = "exp05_label_efficiency"
    if (output / "SUCCESS.json").exists():
        print(f"Already complete: {output}"); return
    lock = output.parent / f".{output.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"Collision lock exists: {lock}") from exc
    stage = output.parent / f".{output.name}.partial-{os.getpid()}"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        train = splits["train"].reset_index(drop=True)
        oof = np.full(len(train), np.nan, dtype=float)
        for fold in range(FOLDS):
            hold = np.flatnonzero(plan.assignment == fold); fit = np.flatnonzero(plan.assignment != fold)
            oof[hold] = train_fold(
                train.iloc[fit].reset_index(drop=True), train.iloc[hold].reset_index(drop=True),
                args.family, model_id, revision, max_length, learning_rate, epochs,
                args.seed * 10 + fold, args.batch_size, torch.device(args.device),
            )
        if not np.isfinite(oof).all() or float(np.ptp(oof)) <= 1e-8:
            raise ValueError("OOF neural scores incomplete or degenerate")
        neural = {"train": oof, "valid": full_valid, "test": full_test}
        feature_frames = {}
        for split in ("train", "valid", "test"):
            frame = structured_features(splits[split])
            frame["embedding_score"] = e5[split]
            frame[f"{args.family}_score"] = neural[split]
            if frame.isna().any().any(): raise ValueError(f"{split}: HEF features contain nulls")
            feature_frames[split] = frame
        labels = {split: splits[split]["label"].to_numpy(dtype=int) for split in splits}
        candidates = []
        for leaves in (7, 15, 31):
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=leaves,
                l2_regularization=1.0, early_stopping=True, random_state=args.seed,
            )
            model.fit(feature_frames["train"], labels["train"])
            score = model.predict_proba(feature_frames["valid"])[:, 1]
            threshold = select_threshold(labels["valid"], score)
            metrics = classification_metrics(labels["valid"], score, threshold)
            candidates.append((float(metrics["f1"]), -leaves, leaves, threshold, model, score, metrics))
        _f1, _tie, leaves, threshold, model, valid_score, valid_metrics = max(candidates, key=lambda x: (x[0], x[1]))
        test_score = model.predict_proba(feature_frames["test"])[:, 1]
        test_metrics = classification_metrics(labels["test"], test_score, threshold)
        np.savez_compressed(stage / "oof_train_scores.npz", pair_id=train["pair_id"].astype(str).to_numpy(),
                            label=labels["train"].astype(np.int8), fold=plan.assignment, score=oof.astype(np.float32))
        np.savez_compressed(stage / "scores.npz",
                            valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(), valid_label=labels["valid"].astype(np.int8),
                            valid_score=valid_score.astype(np.float32), test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),
                            test_label=labels["test"].astype(np.int8), test_score=test_score.astype(np.float32))
        joblib.dump(model, stage / "hef_gbdt.joblib")
        metrics = {
            "experiment": experiment_name, "method": f"hef_gbdt_e5_plus_finetuned_{args.family}",
            "dataset": args.dataset, "seed": args.seed, "family": args.family,
            "training_subset": subset_metadata or {"requested_fraction": 1.0},
            "features": list(feature_frames["train"].columns), "oof_protocol": plan.report,
            "neural_hyperparameters_source": str(metrics_path.relative_to(root)),
            "selected_learning_rate": learning_rate, "selected_epoch": epochs,
            "gbdt_candidates_max_leaf_nodes": [7, 15, 31], "selected_max_leaf_nodes": leaves,
            "selection": "GBDT capacity and threshold selected on validation F1 only",
            "threshold": threshold, "validation": valid_metrics, "test": test_metrics,
            "test_policy": "untouched until neural and HEF selection lock; scored once",
            "split_validation": split_validation, "runtime_seconds": time.time() - started,
        }
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "run_manifest.json", {
            "paper_eligible": True, "dataset": args.dataset, "seed": args.seed, "family": args.family,
            "oof_folds": FOLDS, "oof_rows": len(oof), "grouping": "both_side_record_components",
            "full_neural_metrics_sha256": sha256(metrics_path), "full_neural_scores_sha256": sha256(scores_path),
            "e5_revision": E5_REVISION, "model_id": model_id, "model_revision": revision,
            "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)),
        })
        expected = [stage / name for name in ("metrics.json", "run_manifest.json", "scores.npz", "oof_train_scores.npz", "hef_gbdt.joblib")]
        if any(not path.exists() or path.stat().st_size == 0 for path in expected):
            raise RuntimeError("Expected artifact absent or empty")
        write_json(stage / "SUCCESS.json", {"validated": True, "completed_unix": time.time()})
        if output.exists(): raise FileExistsError(output)
        os.replace(stage, output)
        print(json.dumps({"event": "complete", "output": str(output), "test_f1": test_metrics["f1"]}), flush=True)
    finally:
        if stage.exists(): shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
