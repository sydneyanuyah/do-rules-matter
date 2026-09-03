#!/usr/bin/env python3
"""Propagate completed Exp1 joint-neural HEF checkpoints to locked Exp2 pools."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from paper1_hef.features import FIELDS, GENEALOGY_FIELDS, serialize, structured_features
from run_joint_hef import BACKBONES, EvidenceCollator, JointHEF, PairEvidenceDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _load_exp02_helpers():
    """Load helpers without modifying the app's shared paper1_hef source tree.

    Some long-lived SageMaker apps contain a pre-ranking-repair exp02 module
    without ``_ranking_source``.  The private compatibility copy travels with
    this runner, so active RoBERTa OOF and Exp1 processes keep their original
    imported module unchanged.
    """
    try:
        from paper1_hef.exp02 import _evaluate, _ranking_source

        return _evaluate, _ranking_source
    except ImportError:
        path = Path(__file__).with_name("exp02_compat.py")
        spec = importlib.util.spec_from_file_location(
            "paper1_hef._joint_exp02_compat", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Exp2 compatibility module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._evaluate, module._ranking_source


_evaluate, _ranking_source = _load_exp02_helpers()

def _fields_for_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    """Choose fields from the actual schema (the live bundle predates this fix)."""
    return GENEALOGY_FIELDS if "name" in frame.columns else FIELDS


def candidate_pool(root: Path, dataset: str) -> tuple[pd.DataFrame, Path]:
    directory = root / "artifacts" / "exp02" / "candidate_pools" / dataset
    path, manifest = directory / "top100.csv.gz", directory / "manifest.json"
    if not path.exists() or not manifest.exists():
        raise FileNotFoundError(f"Locked candidate pool incomplete: {directory}")
    frame = pd.read_csv(path)
    required = {"split", "query_id", "candidate_id", "retrieval_rank", "embedding_score", "label"}
    if required - set(frame):
        raise ValueError(f"Candidate-pool columns missing: {sorted(required - set(frame))}")
    frame["query_id"] = frame["query_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.duplicated(["split", "query_id", "candidate_id"]).any():
        raise ValueError("Duplicate candidate-pool keys")
    return frame, manifest


def attach_text(root: Path, config: dict, dataset: str, pool: pd.DataFrame) -> pd.DataFrame:
    source = _ranking_source(root, config, dataset)
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
        paired["left_text"] = paired.apply(lambda row: " ".join(
            f"COL {field} VAL {row.get(f'left_{field}')}" for field in qfields
            if pd.notna(row.get(f"left_{field}")) and str(row.get(f"left_{field}")).strip()), axis=1)
    if paired["right_text"].str.len().eq(0).all():
        paired["right_text"] = paired.apply(lambda row: " ".join(
            f"COL {field} VAL {row.get(f'right_{field}')}" for field in cfields
            if pd.notna(row.get(f"right_{field}")) and str(row.get(f"right_{field}")).strip()), axis=1)
    if paired[["left_text", "right_text"]].isna().any().any():
        raise ValueError("Null text after candidate join")
    return paired


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    spec = BACKBONES[args.backbone]
    source = root / "artifacts" / "exp01_hef_joint_finetuned" / "v1" / args.dataset / spec.key / spec.revision / f"seed_{args.seed}"
    if not (source / "COMPLETED.json").exists():
        raise FileNotFoundError(f"Completed Exp1 joint checkpoint absent: {source}")
    output = root / "artifacts" / "exp02_joint_hef_ranking" / "v1" / args.dataset / spec.key / spec.revision / f"seed_{args.seed}"
    if (output / "SUCCESS.json").exists():
        print(f"Already complete: {output}")
        return
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
        stage.mkdir(parents=True)
        pool, pool_manifest = candidate_pool(root, args.dataset)
        paired = attach_text(root, config, args.dataset, pool)
        checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
        names = list(checkpoint["feature_names"])
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        frame_features = structured_features(paired)
        missing = set(names) - set(frame_features)
        if missing:
            raise ValueError(f"Candidate pairs lack joint-HEF features: {sorted(missing)}")
        x = ((frame_features[names].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
        tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", trust_remote_code=spec.trust_remote_code)
        device = torch.device(args.device)
        model = JointHEF(spec, len(names)).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
        scores = np.full(len(paired), np.nan, dtype=np.float32)
        for split in ("valid", "test"):
            idx = np.flatnonzero(paired["split"].eq(split).to_numpy())
            part = paired.iloc[idx]
            ds = PairEvidenceDataset(part["left_text"].tolist(), part["right_text"].tolist(), x[idx], part["label"].to_numpy(dtype=np.int64), tokenizer, spec.max_length)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=EvidenceCollator(tokenizer), num_workers=0, pin_memory=True)
            chunks = []
            with torch.inference_mode():
                for batch in loader:
                    batch.pop("labels"); structured = batch.pop("structured").to(device)
                    tokens = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(structured, **tokens)
                    chunks.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            scores[idx] = np.concatenate(chunks)
        selected = paired[paired["split"].isin(["valid", "test"])].copy()
        selected["joint_hef_score"] = scores[selected.index]
        if not np.isfinite(selected["joint_hef_score"]).all() or selected["joint_hef_score"].nunique() < 2:
            raise ValueError("Joint HEF ranking scores are incomplete or degenerate")
        sizes = [int(k) for k in config["experiments"]["exp02_candidate_ranking"]["k"]]
        metrics = {}
        for split in ("valid", "test"):
            part = selected[selected["split"].eq(split)].copy()
            metrics[split] = _evaluate(part, "joint_hef_score", sizes)
        np.savez_compressed(stage / "scores.npz", split=selected["split"].astype(str).to_numpy(), query_id=selected["query_id"].astype(str).to_numpy(), candidate_id=selected["candidate_id"].astype(str).to_numpy(), label=selected["label"].to_numpy(dtype=np.int8), score=selected["joint_hef_score"].to_numpy(dtype=np.float32))
        write_json(stage / "metrics.json", {"experiment":"exp02_joint_hef_ranking","method":"joint_neural_hef","dataset":args.dataset,"backbone":spec.key,"seed":args.seed,"source_checkpoint":str(source),"validation":metrics["valid"],"test":metrics["test"],"runtime_seconds":time.time()-started,"test_policy":"direct propagation of validation-selected Exp1 model; no test-time fitting"})
        write_json(stage / "run_manifest.json", {"paper_eligible":True,"pool_sha256":sha256(pool_manifest),"source_model_sha256":sha256(source/"model.pt"),"runner_sha256":sha256(Path(__file__))})
        write_json(stage / "SUCCESS.json", {"validated":True,"rows":len(selected)})
        if output.exists(): raise FileExistsError(output)
        os.replace(stage, output)
        print(json.dumps({"event":"complete","output":str(output)}), flush=True)
    finally:
        if stage.exists(): shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
