#!/usr/bin/env python3
"""Leakage-safe Jina reranker fine-tuning for Experiment 1.

One invocation owns exactly one (dataset, seed) cell. Hyperparameters, epoch,
and classification threshold are selected on the official validation split.
The test split is scored exactly once after the selected checkpoint is locked.
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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "jinaai/jina-reranker-v2-base-multilingual"
MODEL_REVISION = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"
METHOD = "jina_finetuned"


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_single_logit(logits: Any) -> Any:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise RuntimeError(
        f"Pinned Jina relevance model must return one logit per pair; got {tuple(logits.shape)}"
    )


def build_texts(
    splits: dict[str, Any],
) -> dict[str, tuple[list[str], list[str], np.ndarray]]:
    from paper1_hef.features import serialize

    result = {}
    for split, frame in splits.items():
        result[split] = (
            serialize(frame, "left").fillna("").astype(str).tolist(),
            serialize(frame, "right").fillna("").astype(str).tolist(),
            frame["label"].to_numpy(dtype=np.float32),
        )
    return result


def make_collate(tokenizer: Any, max_length: int):
    import torch

    def collate(rows: list[tuple[str, str, float]]) -> dict[str, Any]:
        left, right, labels = zip(*rows)
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
        return encoded

    return collate


class RawPairDataset:
    def __init__(self, left: list[str], right: list[str], labels: np.ndarray):
        self.left = left
        self.right = right
        self.labels = labels
        self.lengths = np.asarray(
            [min(len(a) + len(b), 16000) for a, b in zip(left, right)], dtype=np.int32
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, str, float]:
        return self.left[index], self.right[index], float(self.labels[index])


class BucketBatchSampler:
    """Length-bucketed batches with deterministic per-epoch batch shuffling."""

    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int, shuffle: bool):
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        order = np.argsort(lengths, kind="stable")
        self.batches = [
            order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for index in order:
            yield self.batches[index]


def load_model(device: Any):
    import torch
    from transformers import AutoModelForSequenceClassification

    # Keep master weights in fp32 and use autocast. This is safer for fine-tuning
    # the custom single-logit head than loading trainable weights directly in fp16.
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)


def calibrate_batch_size(
    model: Any,
    tokenizer: Any,
    train: RawPairDataset,
    device: Any,
    max_length: int,
    requested: int,
    maximum: int,
    target_gib: float,
    pos_weight: Any,
) -> tuple[int, list[dict[str, Any]]]:
    import torch

    if requested > 0:
        return requested, [{"batch_size": requested, "status": "user_locked"}]
    longest = np.argsort(-train.lengths, kind="stable")
    candidates: list[int] = []
    value = min(8, len(train))
    while value:
        candidates.append(value)
        nxt = min(maximum, len(train), value * 2)
        if nxt == value:
            break
        value = nxt
    chosen = 1
    audit: list[dict[str, Any]] = []
    target_bytes = int(target_gib * 1024**3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    collate = make_collate(tokenizer, max_length)
    model.train()
    for candidate in candidates:
        rows = [train[int(i)] for i in longest[:candidate]]
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            batch = collate(rows)
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = extract_single_logit(model(**batch).logits)
                loss = loss_fn(logits.float(), labels)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
            audit.append(
                {
                    "batch_size": candidate,
                    "peak_allocated_gib": peak / 1024**3,
                    "status": "ok",
                }
            )
            if peak <= int(21.5 * 1024**3):
                chosen = candidate
            if peak >= target_bytes:
                break
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            audit.append({"batch_size": candidate, "status": "out_of_memory"})
            break
    return chosen, audit


def score(
    model: Any,
    tokenizer: Any,
    dataset: RawPairDataset,
    batch_size: int,
    max_length: int,
    device: Any,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    sampler = BucketBatchSampler(dataset.lengths, batch_size, 0, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=make_collate(tokenizer, max_length),
        num_workers=2,
        pin_memory=True,
        persistent_workers=len(sampler) > 1,
    )
    values = np.empty(len(dataset), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for indices, batch in zip(sampler, loader):
            batch.pop("labels")
            encoded = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = extract_single_logit(model(**encoded).logits)
            values[np.asarray(indices)] = torch.sigmoid(logits.float()).cpu().numpy()
    return values


def assert_scores(
    name: str, labels: np.ndarray, scores: np.ndarray, threshold: float
) -> None:
    if len(labels) != len(scores) or not len(scores):
        raise RuntimeError(f"{name}: label/score coverage mismatch")
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{name}: non-finite scores")
    if float(np.std(scores)) <= 1e-8:
        raise RuntimeError(f"{name}: degenerate scores")
    predictions = scores >= threshold
    if predictions.all() or (~predictions).all():
        raise RuntimeError(
            f"{name}: validation-selected threshold collapses predictions "
            f"(positive_rate={predictions.mean():.8f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp05-fraction", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rates", default="1e-5,2e-5")
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--target-memory-gib", type=float, default=20.0)
    parser.add_argument("--eval-batch-multiplier", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    import yaml
    from paper1_hef.data import load_dataset, validate_splits
    from paper1_hef.evaluate import classification_metrics, select_threshold
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    args = parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.seed not in [int(x) for x in config["protocol"]["seeds"]]:
        raise ValueError(f"Seed {args.seed} is not in the locked protocol seed list")
    configured = config["exp02_jina_cross_encoder"]
    if configured["id"] != MODEL_ID or configured["revision"] != MODEL_REVISION:
        raise ValueError("Pinned Jina model/revision disagrees with experiment config")

    if args.exp05_fraction is None:
        final = root / "artifacts" / "exp01_jina_finetuned" / "v1" / args.dataset / f"seed_{args.seed}"
    else:
        fraction_key = f"{int(round(float(args.exp05_fraction) * 100)):03d}"
        final = root / "artifacts" / "exp05_expanded" / "v1" / "jina_finetuned" / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
    lock = final.parent / f".{final.name}.lock"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and (final / "COMPLETED.json").exists() and not args.overwrite:
        print(f"Already complete: {final}", flush=True)
        return
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another worker owns {args.dataset}/seed_{args.seed}: {lock}"
        ) from exc
    os.write(descriptor, f"host={socket.gethostname()} pid={os.getpid()}\n".encode())
    os.close(descriptor)
    staging = final.parent / f".{final.name}.partial.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.time()

    try:
        set_seed(args.seed)
        torch_device = torch.device(args.device)
        if torch_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This paper run requires a CUDA GPU")
        splits = load_dataset(
            root / config["project"]["data_root"], config["datasets"][args.dataset]
        )
        split_validation = validate_splits(
            splits,
            enforce_offer_disjoint=args.dataset == "wdc_80_medium_unseen",
            report_offer_overlap=config["datasets"][args.dataset]["adapter"] == "wdc",
        )
        subset_metadata = None
        if args.exp05_fraction is not None:
            from paper1_hef.exp05 import _nested_subsets, _pair_id_hash
            spec = config["experiments"]["exp05_label_efficiency"]
            fraction = float(args.exp05_fraction)
            locked = sorted({float(x) for x in spec["fractions"]})
            if fraction not in locked:
                raise ValueError(f"fraction {fraction} is outside locked Experiment 5 schedule")
            train = splits["train"].reset_index(drop=True)
            _, subsets = _nested_subsets(
                train["left_id"].astype(str).to_numpy(),
                train["label"].to_numpy(dtype=np.int64),
                locked,
                int(spec["subset_seed"]),
            )
            selected = train.iloc[subsets[fraction]].reset_index(drop=True)
            subset_metadata = {
                "requested_fraction": fraction,
                "selected_pair_count": int(len(selected)),
                "selected_group_count": int(selected["left_id"].astype(str).nunique()),
                "pair_id_sha256": _pair_id_hash(selected["pair_id"].astype(str).to_numpy()),
                "subset_seed": int(spec["subset_seed"]),
            }
            splits = dict(splits)
            splits["train"] = selected
        texts = build_texts(splits)
        datasets = {name: RawPairDataset(*value) for name, value in texts.items()}
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        )
        model = load_model(torch_device)
        positive = float(texts["train"][2].sum())
        negative = float(len(texts["train"][2]) - positive)
        if positive <= 0 or negative <= 0:
            raise RuntimeError("Training split must contain both classes")
        pos_weight = torch.tensor(
            negative / positive, dtype=torch.float32, device=torch_device
        )
        train_batch_size, calibration = calibrate_batch_size(
            model,
            tokenizer,
            datasets["train"],
            torch_device,
            args.max_length,
            args.batch_size,
            args.max_batch_size,
            args.target_memory_gib,
            pos_weight,
        )
        del model
        torch.cuda.empty_cache()
        eval_batch_size = min(
            args.max_batch_size * 2, train_batch_size * args.eval_batch_multiplier
        )
        learning_rates = [float(x) for x in args.learning_rates.split(",") if x.strip()]
        trials: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        selected_dir = staging / "model"

        for trial_index, learning_rate in enumerate(learning_rates):
            set_seed(args.seed)
            model = load_model(torch_device)
            sampler = BucketBatchSampler(
                datasets["train"].lengths,
                train_batch_size,
                args.seed + trial_index * 1000,
                shuffle=True,
            )
            loader = DataLoader(
                datasets["train"],
                batch_sampler=sampler,
                collate_fn=make_collate(tokenizer, args.max_length),
                num_workers=2,
                pin_memory=True,
                persistent_workers=len(sampler) > 1,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
            )
            total_steps = args.max_epochs * len(loader)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
                num_training_steps=total_steps,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            trial_best = -1.0
            stale = 0
            epoch_records = []
            candidate_dir = staging / f"candidate_lr_{learning_rate:g}"

            for epoch in range(1, args.max_epochs + 1):
                sampler.set_epoch(epoch)
                model.train()
                losses = []
                for batch in loader:
                    labels = batch.pop("labels").to(torch_device, non_blocking=True)
                    encoded = {
                        k: v.to(torch_device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = extract_single_logit(model(**encoded).logits)
                        loss = loss_fn(logits.float(), labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    losses.append(float(loss.detach().cpu()))

                valid_scores = score(
                    model,
                    tokenizer,
                    datasets["valid"],
                    eval_batch_size,
                    args.max_length,
                    torch_device,
                )
                threshold = select_threshold(texts["valid"][2], valid_scores)
                metrics = classification_metrics(
                    texts["valid"][2], valid_scores, threshold
                )
                epoch_records.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "threshold": threshold,
                        "validation": metrics,
                    }
                )
                print(
                    json.dumps(
                        {
                            "dataset": args.dataset,
                            "seed": args.seed,
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "validation_f1": metrics["f1"],
                            "gpu_peak_gib": torch.cuda.max_memory_allocated(
                                torch_device
                            )
                            / 1024**3,
                        }
                    ),
                    flush=True,
                )
                if metrics["f1"] > trial_best + 1e-12:
                    trial_best = metrics["f1"]
                    stale = 0
                    if candidate_dir.exists():
                        shutil.rmtree(candidate_dir)
                    model.save_pretrained(candidate_dir)
                    tokenizer.save_pretrained(candidate_dir)
                    json_write(
                        candidate_dir / "selection.json",
                        {
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "threshold": threshold,
                            "validation": metrics,
                        },
                    )
                else:
                    stale += 1
                if stale >= args.patience:
                    break

            selection = json.loads((candidate_dir / "selection.json").read_text())
            trial_record = {
                "learning_rate": learning_rate,
                "epochs_executed": len(epoch_records),
                "selected_epoch": selection["epoch"],
                "threshold": selection["threshold"],
                "validation": selection["validation"],
                "epochs": epoch_records,
            }
            trials.append(trial_record)
            if (
                best is None
                or selection["validation"]["f1"] > best["validation"]["f1"] + 1e-12
            ):
                best = trial_record
                if selected_dir.exists():
                    shutil.rmtree(selected_dir)
                shutil.copytree(candidate_dir, selected_dir)
            shutil.rmtree(candidate_dir)
            del model, optimizer, scheduler, scaler, loader
            torch.cuda.empty_cache()

        assert best is not None
        # The chosen hyperparameters/checkpoint/threshold are now locked. The
        # test split is touched exactly once below.
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            selected_dir, trust_remote_code=True, torch_dtype=torch.float32
        ).to(torch_device)
        valid_scores = score(
            model,
            tokenizer,
            datasets["valid"],
            eval_batch_size,
            args.max_length,
            torch_device,
        )
        test_scores = score(
            model,
            tokenizer,
            datasets["test"],
            eval_batch_size,
            args.max_length,
            torch_device,
        )
        threshold = float(best["threshold"])
        assert_scores("valid", texts["valid"][2], valid_scores, threshold)
        assert_scores("test", texts["test"][2], test_scores, threshold)
        metrics = {
            "experiment": "exp05_label_efficiency" if args.exp05_fraction is not None else "exp01_standard_pair_classification",
            "method": METHOD,
            "status": "complete",
            "paper_eligible": True,
            "dataset": args.dataset,
            "seed": args.seed,
            "training_subset": subset_metadata or {"requested_fraction": 1.0},
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "objective": "BCEWithLogitsLoss on one relevance logit",
            "selection": "learning rate, epoch, and threshold selected on validation only",
            "test_policy": "untouched until checkpoint and threshold lock; scored once",
            "search_budget": {
                "learning_rates": learning_rates,
                "max_epochs": args.max_epochs,
                "early_stopping_patience": args.patience,
                "max_length": args.max_length,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "batch_calibration": calibration,
            },
            "trials": trials,
            "selected_learning_rate": best["learning_rate"],
            "selected_epoch": best["selected_epoch"],
            "threshold": threshold,
            "validation": classification_metrics(
                texts["valid"][2], valid_scores, threshold
            ),
            "test": classification_metrics(texts["test"][2], test_scores, threshold),
            "coverage": {
                "train": {
                    "expected": len(splits["train"]),
                    "loaded": len(texts["train"][2]),
                },
                "valid": {
                    "expected": len(splits["valid"]),
                    "scored": len(valid_scores),
                },
                "test": {"expected": len(splits["test"]), "scored": len(test_scores)},
            },
            "score_assertions": {
                "validation_std": float(np.std(valid_scores)),
                "test_std": float(np.std(test_scores)),
                "test_predicted_positive_rate": float(
                    (test_scores >= threshold).mean()
                ),
            },
            "split_validation": split_validation,
            "runtime_seconds": time.time() - started,
        }
        json_write(staging / "metrics.json", metrics)
        np.savez_compressed(
            staging / "scores.npz",
            valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(),
            valid_label=texts["valid"][2].astype(np.int8),
            valid_score=valid_scores,
            test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),
            test_label=texts["test"][2].astype(np.int8),
            test_score=test_scores,
        )
        manifest = {
            "dataset": args.dataset,
            "seed": args.seed,
            "method": METHOD,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__)),
            "fit_split": "train",
            "selection_split": "valid",
            "test_policy": "scored_once_after_selection_lock",
            "hostname": socket.gethostname(),
            "cuda_device": str(torch_device),
            "pytorch_version": torch.__version__,
            "paper_eligible": True,
        }
        json_write(staging / "run_manifest.json", manifest)
        json_write(
            staging / "COMPLETED.json",
            {
                "status": "complete",
                "metrics_sha256": sha256(staging / "metrics.json"),
                "scores_sha256": sha256(staging / "scores.npz"),
                "manifest_sha256": sha256(staging / "run_manifest.json"),
                "completed_at_unix": time.time(),
            },
        )
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
        print(f"COMPLETE {final}", flush=True)
    except Exception:
        json_write(
            staging / "FAILED.json",
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "host": socket.gethostname(),
                "pid": os.getpid(),
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
