#!/usr/bin/env python3
"""Retrain Joint Neural HEF + RoBERTa under one Experiment 6 evidence ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from exp06_common import CONDITIONS, DATASETS, SEED, selected_structured_features
from paper1_hef.data import load_dataset, validate_splits
from paper1_hef.evaluate import classification_metrics, select_threshold
from paper1_hef.features import serialize, structured_features


def load_joint_modules(source_dir: Path):
    sys.path.insert(0, str(source_dir))
    from run_joint_hef import BACKBONES, EvidenceCollator, JointHEF, PairEvidenceDataset
    from run_joint_hef_ranking import _evaluate, attach_text, candidate_pool
    return BACKBONES, EvidenceCollator, JointHEF, PairEvidenceDataset, _evaluate, attach_text, candidate_pool


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StructuredOnlyHEF(nn.Module):
    def __init__(self, structured_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(structured_dim), nn.Linear(structured_dim, 128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2),
        )

    def forward(self, structured: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.network(structured)


def run(args: argparse.Namespace) -> Path:
    root, source_dir = args.project_root.resolve(), args.joint_source.resolve()
    if args.dataset not in DATASETS or args.condition not in CONDITIONS:
        raise ValueError("Dataset or condition outside locked Experiment 6 matrix")
    modules = load_joint_modules(source_dir)
    BACKBONES, EvidenceCollator, JointHEF, PairEvidenceDataset, evaluate, attach_text, candidate_pool = modules
    spec = BACKBONES["roberta"]
    output = (
        root / "artifacts" / "exp06_rerun_v2" / "joint_neural_hef_roberta"
        / args.condition / args.dataset / f"seed_{SEED}"
    )
    if (output / "SUCCESS.json").exists(): return output
    lock = output.parent / f".{output.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"Collision lock exists: {lock}") from exc
    stage = output.parent / f".{output.name}.partial-{os.getpid()}"
    started = time.time()
    try:
        stage.mkdir(parents=True, exist_ok=False)
        config_path = root / "configs" / "experiment.yaml"
        config = yaml.safe_load(config_path.read_text())
        splits = load_dataset(root / config["project"]["data_root"], config["datasets"][args.dataset])
        split_report = validate_splits(
            splits,
            enforce_offer_disjoint=args.dataset == "wdc_80_medium_unseen",
            report_offer_overlap=config["datasets"][args.dataset]["adapter"] == "wdc",
            enforce_record_disjoint=config["datasets"][args.dataset]["adapter"] == "link_lives",
        )
        feature_frames = {name: structured_features(frame) for name, frame in splits.items()}
        features, semantic_enabled = selected_structured_features(feature_frames["train"].columns, args.condition)
        train_x = feature_frames["train"][features].to_numpy(np.float32)
        mean, std = train_x.mean(0), train_x.std(0); std[std < 1e-6] = 1.0
        arrays = {
            name: ((frame[features].to_numpy(np.float32) - mean) / std).astype(np.float32)
            for name, frame in feature_frames.items()
        }
        labels = {name: frame["label"].to_numpy(np.int64) for name, frame in splits.items()}
        texts = {name: (serialize(frame, "left").tolist(), serialize(frame, "right").tolist()) for name, frame in splits.items()}
        tokenizer = None
        if semantic_enabled:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id, revision=spec.revision, trust_remote_code=spec.trust_remote_code
            )
            collator = EvidenceCollator(tokenizer)
            loaders = {
                name: DataLoader(
                    PairEvidenceDataset(*texts[name], arrays[name], labels[name], tokenizer, spec.max_length),
                    batch_size=args.batch_size if name == "train" else args.eval_batch_size,
                    shuffle=name == "train", collate_fn=collator, num_workers=0, pin_memory=True,
                ) for name in splits
            }
        else:
            loaders = {
                name: DataLoader(
                    TensorDataset(torch.from_numpy(arrays[name]), torch.from_numpy(labels[name])),
                    batch_size=args.structured_batch_size, shuffle=name == "train", pin_memory=True,
                ) for name in splits
            }
        device = torch.device(args.device)
        positive = float(labels["train"].sum()); negative = len(labels["train"]) - positive
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, negative / max(positive, 1.0)], device=device))

        def build_model() -> nn.Module:
            return JointHEF(spec, len(features)).to(device) if semantic_enabled else StructuredOnlyHEF(len(features)).to(device)

        def unpack(batch: Any) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            if semantic_enabled:
                y = batch.pop("labels").to(device)
                structured = batch.pop("structured").to(device)
                tokens = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                return tokens, structured, y
            structured, y = batch
            return {}, structured.to(device), y.to(device)

        def score_loader(model: nn.Module, loader: DataLoader) -> np.ndarray:
            model.eval(); chunks = []
            with torch.inference_mode():
                for batch in loader:
                    tokens, structured, _ = unpack(batch)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(structured, **tokens)
                    chunks.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            return np.concatenate(chunks)

        learning_rates = args.learning_rates if semantic_enabled else [args.structured_learning_rate]
        max_epochs = args.max_epochs if semantic_enabled else args.structured_epochs
        patience = args.patience if semantic_enabled else args.structured_patience
        trials, best = [], None
        for learning_rate in learning_rates:
            seed_all(SEED); model = build_model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            total = max_epochs * len(loaders["train"])
            scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.1 * total)), total)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            stale, trial_best = 0, None
            for epoch in range(1, max_epochs + 1):
                model.train(); losses = []
                for batch in loaders["train"]:
                    tokens, structured, y = unpack(batch); optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(structured, **tokens); loss = loss_fn(logits, y)
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); scheduler.step(); losses.append(float(loss.detach()))
                valid = score_loader(model, loaders["valid"])
                threshold = select_threshold(labels["valid"], valid)
                metrics = classification_metrics(labels["valid"], valid, threshold)
                event = {"dataset": args.dataset, "condition": args.condition, "epoch": epoch,
                         "learning_rate": learning_rate, "loss": float(np.mean(losses)), "validation_f1": metrics["f1"]}
                print(json.dumps(event), flush=True)
                if trial_best is None or metrics["f1"] > trial_best["validation_f1"]:
                    trial_best = {"epoch": epoch, "learning_rate": learning_rate, "validation_f1": metrics["f1"],
                                  "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                    stale = 0
                else: stale += 1
                if stale >= patience: break
            assert trial_best is not None
            trials.append({k: v for k, v in trial_best.items() if k != "state"})
            if best is None or trial_best["validation_f1"] > best["validation_f1"]: best = trial_best
            del model; torch.cuda.empty_cache()
        assert best is not None
        model = build_model(); model.load_state_dict(best["state"])

        pool, pool_manifest = candidate_pool(root, args.dataset)
        paired = attach_text(root, config, args.dataset, pool)
        pool_features = structured_features(paired)
        missing = set(features) - set(pool_features)
        if missing: raise ValueError(f"Candidate pool missing features: {sorted(missing)}")
        pool_x = ((pool_features[features].to_numpy(np.float32) - mean) / std).astype(np.float32)
        selected = paired[paired["split"].isin(["valid", "test"])].copy()
        index = selected.index.to_numpy()
        if semantic_enabled:
            dataset = PairEvidenceDataset(
                selected["left_text"].tolist(), selected["right_text"].tolist(), pool_x[index],
                selected["label"].to_numpy(np.int64), tokenizer, spec.max_length,
            )
            pool_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                                     collate_fn=EvidenceCollator(tokenizer), num_workers=0, pin_memory=True)
        else:
            pool_loader = DataLoader(
                TensorDataset(torch.from_numpy(pool_x[index]), torch.from_numpy(selected["label"].to_numpy(np.int64))),
                batch_size=args.structured_batch_size, shuffle=False, pin_memory=True,
            )
        selected["score"] = score_loader(model, pool_loader)
        if not np.isfinite(selected["score"]).all():
            raise ValueError("Joint ablation candidate scores are incomplete or nonfinite")
        sizes = [int(k) for k in config["experiments"]["exp02_candidate_ranking"]["k"]]
        ranking = {
            split: evaluate(selected.loc[selected["split"].eq(split)].copy(), "score", sizes)
            for split in ("valid", "test")
        }
        np.savez_compressed(
            stage / "scores.npz", split=selected["split"].astype(str).to_numpy(),
            query_id=selected["query_id"].astype(str).to_numpy(),
            candidate_id=selected["candidate_id"].astype(str).to_numpy(),
            label=selected["label"].to_numpy(np.int8), score=selected["score"].to_numpy(np.float32),
        )
        torch.save({"state_dict": model.state_dict(), "feature_names": features, "feature_mean": mean,
                    "feature_std": std, "semantic_enabled": semantic_enabled}, stage / "model.pt")
        if tokenizer is not None: tokenizer.save_pretrained(stage / "tokenizer")
        write_json(stage / "metrics.json", {
            "experiment": "exp06_public_evidence_ablation_rerun_v2", "model": "joint_neural_hef_roberta",
            "condition": args.condition, "dataset": args.dataset, "seed": SEED,
            "features": features, "semantic_enabled": semantic_enabled, "trials": trials,
            "selection": "learning rate and epoch selected on classification validation only, matching the locked joint-HEF protocol",
            "ranking_validation": ranking["valid"], "ranking_test": ranking["test"],
            "score_integrity": {
                "valid_unique_values": int(selected.loc[selected["split"].eq("valid"), "score"].nunique()),
                "test_unique_values": int(selected.loc[selected["split"].eq("test"), "score"].nunique()),
                "degenerate_test_scores": bool(
                    selected.loc[selected["split"].eq("test"), "score"].nunique() < 2
                ),
            },
            "test_policy": "fixed Experiment 2 top-100 test pool scored once after selection lock",
            "split_validation": split_report, "runtime_seconds": time.time() - started, "status": "complete",
        })
        write_json(stage / "manifest.json", {
            "paper_eligible": True, "seed_policy": "single locked screening seed; no variance claim",
            "pool_sha256": sha256(pool_manifest), "config_sha256": sha256(config_path),
        })
        write_json(stage / "SUCCESS.json", {"validated": True, "rows": len(selected)})
        if output.exists(): raise FileExistsError(output)
        os.replace(stage, output)
        return output
    finally:
        if stage.exists(): shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--joint-source", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--structured-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-5, 2e-5])
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--structured-learning-rate", type=float, default=1e-3)
    parser.add_argument("--structured-epochs", type=int, default=80)
    parser.add_argument("--structured-patience", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse()))
