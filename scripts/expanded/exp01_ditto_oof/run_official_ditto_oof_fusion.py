#!/usr/bin/env python3
"""Leakage-safe Official Ditto OOF scoring and HEF-GBDT fusion.

This runner imports the pinned Official Ditto compatibility implementation
rather than maintaining a second architecture. Fold models never train on a
record present in their scored OOF fold. The OOF labels are used only after the
scores have been produced, to fit the downstream HEF model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from paper1_hef.data import load_dataset, validate_splits
from paper1_hef.evaluate import classification_metrics, select_threshold
from paper1_hef.features import structured_features


E5_ID = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
UPSTREAM_COMMIT = "52985564a93fb11308439516d3e17a033d43ec8f"
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

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def entity_nodes(frame: pd.DataFrame, adapter: str) -> tuple[np.ndarray, np.ndarray]:
    left = frame["left_id"].astype(str).to_numpy(dtype=str)
    right = frame["right_id"].astype(str).to_numpy(dtype=str)
    # DeepMatcher table A and table B identifiers are separate namespaces.
    if adapter == "deepmatcher":
        left = np.char.add("L:", left)
        right = np.char.add("R:", right)
    return left, right


@dataclass(frozen=True)
class FoldPlan:
    assignment: np.ndarray
    report: dict[str, Any]


def make_component_folds(
    frame: pd.DataFrame, adapter: str, seed: int, n_folds: int = FOLDS
) -> FoldPlan:
    """Assign entire pair-graph components to deterministic balanced folds."""
    if len(frame) < n_folds:
        raise ValueError("Fewer train rows than requested OOF folds")
    left, right = entity_nodes(frame, adapter)
    dsu = DSU()
    for a, b in zip(left, right, strict=True):
        dsu.union(str(a), str(b))
    components: dict[str, list[int]] = {}
    for index, node in enumerate(left):
        components.setdefault(dsu.find(str(node)), []).append(index)
    grouping = "connected_components_of_both_record_sides"
    if len(components) < n_folds:
        # Dense candidate sets can make the all-edge bipartite graph one giant
        # component through arbitrary negative pairs.  In that case, use the
        # retrieval query/left record as the leakage-safe OOF group.  This
        # guarantees that no left/query record occurs in both train and
        # holdout and still scores every training pair exactly once.
        components = {}
        for index, node in enumerate(left):
            components.setdefault(str(node), []).append(index)
        grouping = "left_record_grouped"
    if len(components) < n_folds:
        raise ValueError(f"Only {len(components)} left-record groups; cannot form {n_folds} folds")

    labels = frame["label"].to_numpy(dtype=np.int8)
    rng = random.Random(seed)
    items: list[tuple[str, list[int], int, float]] = []
    for root, rows in components.items():
        items.append((root, rows, int(labels[rows].sum()), rng.random()))
    items.sort(key=lambda item: (-len(item[1]), -item[2], item[3], item[0]))

    fold_rows: list[list[int]] = [[] for _ in range(n_folds)]
    fold_pos = np.zeros(n_folds, dtype=int)
    target_rows = len(frame) / n_folds
    target_pos = max(1.0, float(labels.sum()) / n_folds)
    for _root, rows, positives, _jitter in items:
        costs = []
        for fold in range(n_folds):
            row_cost = (len(fold_rows[fold]) + len(rows)) / target_rows
            pos_cost = (fold_pos[fold] + positives) / target_pos
            costs.append((row_cost + pos_cost, len(fold_rows[fold]), fold_pos[fold], fold))
        chosen = min(costs)[-1]
        fold_rows[chosen].extend(rows)
        fold_pos[chosen] += positives

    assignment = np.full(len(frame), -1, dtype=np.int8)
    report: dict[str, Any] = {
        "n_folds": n_folds,
        "component_count": len(components),
        "grouping": grouping,
        "folds": [],
    }
    all_indices = set(range(len(frame)))
    for fold, rows in enumerate(fold_rows):
        holdout = np.asarray(sorted(rows), dtype=int)
        train = np.asarray(sorted(all_indices - set(rows)), dtype=int)
        if not len(holdout) or not len(train):
            raise ValueError(f"Fold {fold} is empty")
        if len(np.unique(labels[train])) != 2:
            raise ValueError(f"Fold {fold} training complement lacks a class")
        train_entities = set(left[train]) | set(right[train])
        holdout_entities = set(left[holdout]) | set(right[holdout])
        overlap = train_entities & holdout_entities
        left_overlap = set(left[train]) & set(left[holdout])
        if left_overlap:
            raise AssertionError(f"Fold {fold}: {len(left_overlap)} leaked left/query records")
        if grouping == "connected_components_of_both_record_sides" and overlap:
            raise AssertionError(f"Fold {fold}: {len(overlap)} leaked records")
        assignment[holdout] = fold
        report["folds"].append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "train_positives": int(labels[train].sum()),
                "holdout_rows": int(len(holdout)),
                "holdout_positives": int(labels[holdout].sum()),
                "left_entity_overlap": 0,
                "both_side_entity_overlap": int(len(overlap)),
                "entity_overlap": int(len(overlap)),
            }
        )
    if np.any(assignment < 0) or not np.array_equal(np.sort(np.flatnonzero(assignment >= 0)), np.arange(len(frame))):
        raise AssertionError("OOF assignment does not cover every train row exactly once")
    return FoldPlan(assignment=assignment, report=report)


def import_compat(root: Path) -> Any:
    path = root / "scripts" / "run_ditto_official_compat.py"
    if not path.exists():
        path = root / "run_ditto_official_compat.py"
    if not path.exists():
        raise FileNotFoundError("Pinned run_ditto_official_compat.py is absent")
    spec = importlib.util.spec_from_file_location("official_ditto_compat", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.UPSTREAM_COMMIT != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected Ditto commit {module.UPSTREAM_COMMIT}")
    return module


def assert_aligned(
    payload: Any, split: str, frame: pd.DataFrame, score_key: str
) -> np.ndarray:
    pair_key = f"{split}_pair_id" if f"{split}_pair_id" in payload else "pair_id"
    label_key = f"{split}_label" if f"{split}_label" in payload else "label"
    expected_pair = frame["pair_id"].astype(str).to_numpy()
    expected_label = frame["label"].to_numpy(dtype=np.int8)
    if not np.array_equal(payload[pair_key].astype(str), expected_pair):
        raise ValueError(f"{split}: pair IDs are not exactly aligned")
    if not np.array_equal(payload[label_key].astype(np.int8), expected_label):
        raise ValueError(f"{split}: labels are not exactly aligned")
    scores = payload[score_key].astype(float)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError(f"{split}: scores are incomplete or nonfinite")
    if len(scores) > 1 and float(np.ptp(scores)) <= 1e-8:
        raise ValueError(f"{split}: neural scores are degenerate")
    return scores


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def train_and_score_fold(
    root: Path,
    compat: Any,
    config: dict[str, Any],
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    variant: str,
    seed: int,
    fixed_epochs: int,
    batch_size: int,
    fold_dir: Path,
    device: torch.device,
) -> np.ndarray:
    upstream = root / "external" / "ditto_official"
    sys.path.insert(0, str(upstream))
    from ditto_light import dataset as official_dataset  # type: ignore

    model_spec = config["cross_encoder"]
    model_id, revision = str(model_spec["id"]), str(model_spec["revision"])
    max_length = int(model_spec["max_length"])
    official_dataset.get_tokenizer = lambda _lm: AutoTokenizer.from_pretrained(
        model_id, revision=revision
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    all_lines = compat._lines(frame)
    train_path, holdout_path = fold_dir / "train.txt", fold_dir / "holdout.txt"
    write_lines(train_path, (all_lines[i] for i in train_idx))
    write_lines(holdout_path, (all_lines[i] for i in holdout_idx))
    da = "all" if variant == "mixda_all" else None
    trainset = official_dataset.DittoDataset(train_path, lm=model_id, max_len=max_length, da=da)
    holdoutset = official_dataset.DittoDataset(holdout_path, lm=model_id, max_len=max_length)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=0, collate_fn=trainset.pad, pin_memory=True,
    )
    holdout_loader = DataLoader(
        holdoutset, batch_size=batch_size * 8, shuffle=False, num_workers=0,
        collate_fn=holdoutset.pad, pin_memory=True,
    )
    model = compat.OfficialDittoModel(model_id, revision, device, 0.8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    steps = max(1, len(train_loader) * fixed_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, fixed_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if len(batch) == 2:
                x, y = batch
                x_aug = None
            else:
                x, x_aug, y = batch
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, x_aug)
                loss = torch.nn.functional.cross_entropy(logits, y.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "mean_train_loss": float(np.mean(losses))})
        print(json.dumps({"event": "fold_epoch", "epoch": epoch, "epochs": fixed_epochs, "mean_loss": history[-1]["mean_train_loss"]}), flush=True)

    model.eval()
    scores: list[np.ndarray] = []
    observed_labels: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y in holdout_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            observed_labels.append(y.numpy())
    result = np.concatenate(scores)
    if not np.array_equal(np.concatenate(observed_labels).astype(np.int8), frame.iloc[holdout_idx]["label"].to_numpy(dtype=np.int8)):
        raise ValueError("Official Ditto loader changed holdout row/label order")
    torch.save(
        {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "seed": seed, "variant": variant, "fixed_epochs": fixed_epochs,
         "upstream_commit": UPSTREAM_COMMIT},
        fold_dir / "model.pt",
    )
    write_json(fold_dir / "history.json", history)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=("plain", "mixda_all"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exp05-fraction", type=float)
    args = parser.parse_args()
    started = time.time()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / args.config).read_text())
    allowed = set(config["dataset_groups"]["exp01_all"]) | {"link_lives_release2"}
    seeds = {int(seed) for seed in config["protocol"]["seeds"]}
    if args.dataset not in allowed or args.seed not in seeds:
        raise ValueError("Dataset or seed is outside the locked protocol")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    spec = config["datasets"][args.dataset]
    splits = load_dataset(root / config["project"]["data_root"], spec)
    validation = validate_splits(
        splits,
        enforce_offer_disjoint=args.dataset == "wdc_80_medium_unseen",
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
    slug = E5_ID.replace("/", "__")
    method = f"official_ditto_{args.variant}"
    if args.exp05_fraction is None:
        final_dir = root / config["project"]["output_root"] / "exp01_hef_cross_evidence" / "v1" / args.dataset / slug / E5_REVISION / method / f"seed_{args.seed}"
        experiment_name = "exp01_hef_gbdt_cross_evidence"
    else:
        fraction_key = f"{int(round(float(args.exp05_fraction) * 100)):03d}"
        family = "hef_gbdt_e5_official_ditto_mixda" if args.variant == "mixda_all" else "hef_gbdt_e5_official_ditto"
        final_dir = root / config["project"]["output_root"] / "exp05_expanded" / "v1" / family / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
        experiment_name = "exp05_label_efficiency"
    success = final_dir / "SUCCESS.json"
    if success.exists():
        print(f"Already complete: {success}", flush=True)
        return
    lock = final_dir.parent / f".{final_dir.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"Collision lock exists: {lock}") from exc
    stage = final_dir.parent / f".{final_dir.name}.partial-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        # Keep the OOF partition identical across model seeds so the reported
        # variance isolates optimization randomness rather than fold changes.
        plan = make_component_folds(splits["train"], spec["adapter"], 20260725)
        compat = import_compat(root)
        if args.exp05_fraction is None:
            full_dir = root / config["project"]["output_root"] / "exp01_ditto_official" / args.variant / args.dataset / f"seed_{args.seed}"
        else:
            standalone_family = "official_ditto_mixda" if args.variant == "mixda_all" else "official_ditto"
            full_dir = root / config["project"]["output_root"] / "exp05_expanded" / "v1" / standalone_family / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
        full_metrics_path, full_scores_path = full_dir / "metrics.json", full_dir / "scores.npz"
        if not full_metrics_path.exists() or not full_scores_path.exists():
            raise FileNotFoundError(f"Completed full Ditto outputs absent: {full_dir}")
        full_metrics = json.loads(full_metrics_path.read_text())
        if full_metrics.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError("Full Ditto result does not use the locked upstream commit")
        fixed_epochs = int(full_metrics["selected_epoch"])
        full_payload = np.load(full_scores_path, allow_pickle=True)
        ditto_valid = assert_aligned(full_payload, "valid", splits["valid"], "valid_score")
        ditto_test = assert_aligned(full_payload, "test", splits["test"], "test_score")

        oof = np.full(len(splits["train"]), np.nan, dtype=float)
        fold_ids = plan.assignment
        for fold in range(FOLDS):
            holdout_idx = np.flatnonzero(fold_ids == fold)
            train_idx = np.flatnonzero(fold_ids != fold)
            fold_seed = args.seed * 10 + fold
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            scores = train_and_score_fold(
                root, compat, config, splits["train"], train_idx, holdout_idx,
                args.variant, fold_seed, fixed_epochs, args.batch_size,
                stage / "oof" / f"fold_{fold}", torch.device(args.device),
            )
            if len(scores) != len(holdout_idx):
                raise ValueError(f"Fold {fold}: score count mismatch")
            oof[holdout_idx] = scores
        if not np.isfinite(oof).all() or float(np.ptp(oof)) <= 1e-8:
            raise ValueError("OOF scores are incomplete, nonfinite, or degenerate")
        np.savez_compressed(
            stage / "oof_train_scores.npz",
            pair_id=splits["train"]["pair_id"].astype(str).to_numpy(),
            label=splits["train"]["label"].to_numpy(dtype=np.int8),
            fold=fold_ids,
            score=oof.astype(np.float32),
        )

        emb_dir = root / config["project"]["output_root"] / "embeddings" / args.dataset / slug / E5_REVISION
        e5: dict[str, np.ndarray] = {}
        for split in ("train", "valid", "test"):
            payload = np.load(emb_dir / f"{split}.npz", allow_pickle=True)
            if split == "train" and args.exp05_fraction is not None:
                source_ids = payload["pair_id"].astype(str)
                source_scores = payload["embedding_score"].astype(float)
                if len(set(source_ids)) != len(source_ids):
                    raise ValueError("duplicate pair IDs in E5 training scores")
                lookup = dict(zip(source_ids, source_scores))
                target_ids = splits[split]["pair_id"].astype(str).to_numpy()
                if any(pair_id not in lookup for pair_id in target_ids):
                    raise ValueError("selected Experiment 5 pair absent from E5 scores")
                e5[split] = np.asarray([lookup[pair_id] for pair_id in target_ids], dtype=float)
            else:
                e5[split] = assert_aligned(payload, split, splits[split], "embedding_score")
        ditto = {"train": oof, "valid": ditto_valid, "test": ditto_test}
        feature_frames: dict[str, pd.DataFrame] = {}
        for split in ("train", "valid", "test"):
            features = structured_features(splits[split])
            features["embedding_score"] = e5[split]
            features["ditto_score"] = ditto[split]
            feature_frames[split] = features
        labels = {split: splits[split]["label"].to_numpy(dtype=int) for split in splits}
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=args.seed,
        )
        model.fit(feature_frames["train"], labels["train"])
        valid_score = model.predict_proba(feature_frames["valid"])[:, 1]
        threshold = select_threshold(labels["valid"], valid_score)
        # Test is accessed only after fitting and validation threshold selection.
        test_score = model.predict_proba(feature_frames["test"])[:, 1]
        metrics = {
            "experiment": experiment_name,
            "method": f"hef_gbdt_e5_plus_{method}",
            "dataset": args.dataset,
            "seed": args.seed,
            "training_subset": subset_metadata or {"requested_fraction": 1.0},
            "features": list(feature_frames["train"].columns),
            "ditto_train_score_protocol": "3_fold_record_component_oof",
            "fold_report": plan.report,
            "fixed_fold_epochs_from_completed_full_run": fixed_epochs,
            "threshold": threshold,
            "validation": classification_metrics(labels["valid"], valid_score, threshold),
            "test": classification_metrics(labels["test"], test_score, threshold),
            "split_validation": validation,
            "test_access_policy": "scored once after HEF fit and validation threshold selection",
            "runtime_seconds": time.time() - started,
        }
        write_json(stage / "metrics.json", metrics)
        joblib.dump(model, stage / "hef_gbdt.joblib")
        np.savez_compressed(
            stage / "scores.npz",
            valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(),
            valid_label=labels["valid"].astype(np.int8),
            valid_score=valid_score.astype(np.float32),
            test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),
            test_label=labels["test"].astype(np.int8),
            test_score=test_score.astype(np.float32),
        )
        manifest = {
            "paper_eligible": True,
            "dataset": args.dataset,
            "seed": args.seed,
            "variant": args.variant,
            "folds": FOLDS,
            "fold_assignment_seed": 20260725,
            "grouping": plan.report["grouping"],
            "oof_coverage_rows": len(oof),
            "oof_nonfinite": int((~np.isfinite(oof)).sum()),
            "upstream_commit": UPSTREAM_COMMIT,
            "full_ditto_metrics_sha256": sha256(full_metrics_path),
            "full_ditto_scores_sha256": sha256(full_scores_path),
            "e5_revision": E5_REVISION,
            "runner_sha256": sha256(Path(__file__)),
        }
        write_json(stage / "run_manifest.json", manifest)
        expected = [stage / "metrics.json", stage / "run_manifest.json", stage / "scores.npz", stage / "oof_train_scores.npz", stage / "hef_gbdt.joblib"]
        if any(not path.exists() or path.stat().st_size == 0 for path in expected):
            raise RuntimeError("Expected output is absent or empty")
        write_json(stage / "SUCCESS.json", {"completed_unix": time.time(), "validated": True})
        if final_dir.exists():
            raise FileExistsError(final_dir)
        os.replace(stage, final_dir)
        print(json.dumps({"event": "complete", "output": str(final_dir), "test_f1": metrics["test"]["f1"]}), flush=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
