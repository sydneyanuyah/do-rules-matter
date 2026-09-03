#!/usr/bin/env python3
"""Experiment 7 classification robustness under controlled evidence loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from paper1_hef.data import load_dataset
from paper1_hef.evaluate import classification_metrics
from paper1_hef.features import FIELDS, GENEALOGY_FIELDS, serialize, structured_features


MODEL_ID = "intfloat/e5-base-v2"
MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"

PRODUCT_MASK_GROUPS = {
    "names_titles": ["title", "description"],
    "dates_numerics": ["price"],
    "places_categories": ["brand", "manufacturer", "priceCurrency"],
    "relationships": ["modelno"],
}
GENEALOGY_MASK_GROUPS = {
    "names_titles": ["name"],
    "dates_numerics": ["birth_year", "age"],
    "places_categories": ["birth_place", "sex", "occupation", "marital_status"],
    "relationships": ["residence_parish", "residence_county", "residence_information"],
}


class OfficialDittoModel(torch.nn.Module):
    def __init__(self, model_id: str, revision: str, device: torch.device) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id, revision=revision)
        self.device = device
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.bert(values.to(self.device, non_blocking=True))[0][:, 0, :]
        return self.fc(encoded)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable_uniform(value: str) -> float:
    raw = hashlib.sha256(value.encode()).digest()[:8]
    return int.from_bytes(raw, "big") / float(2**64)


def controlled_mask(
    frame: pd.DataFrame,
    fields: list[str],
    probability: float,
    seed: int,
) -> tuple[pd.DataFrame, str, int]:
    out = frame.copy()
    selected: list[str] = []
    changed = 0
    ids = frame["pair_id"].astype(str).tolist()
    for field in fields:
        for side in ("left", "right"):
            column = f"{side}_{field}"
            if column not in out:
                continue
            mask = np.asarray([
                stable_uniform(f"{seed}|{pair_id}|{side}|{field}") < probability
                for pair_id in ids
            ])
            nonempty = out[column].notna() & out[column].astype(str).ne("")
            actual = mask & nonempty.to_numpy()
            if np.any(actual):
                out.loc[actual, column] = ""
                changed += int(actual.sum())
                selected.extend(f"{pair_id}|{side}|{field}" for pair_id, flag in zip(ids, actual) if flag)
    digest = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    return out, digest, changed


def scale(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    low, high = float(np.min(train)), float(np.max(train))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def choose_hef(root: Path, dataset: str, revision: str) -> dict[str, Any]:
    directory = root / "artifacts" / "exp01_final" / dataset / MODEL_ID.replace("/", "__") / revision
    metrics = json.loads((directory / "metrics.json").read_text())
    choices = []
    for family in ("hef_linear", "hef_gbdt"):
        for run in metrics["methods"][family]["repetitions"]:
            choices.append((float(run["validation"]["f1"]), family == "hef_linear", -int(run["seed"]), family, run))
    _, _, _, family, run = max(choices)
    model = joblib.load(directory / f"{family}_seed_{run['seed']}.joblib")
    return {
        "family": family, "seed": int(run["seed"]), "threshold": float(run["threshold"]),
        "validation_f1": float(run["validation"]["f1"]), "model": model,
        "feature_names": list(metrics["methods"][family]["features"]),
        "rule_threshold": float(metrics["methods"]["rules"]["threshold"]),
        "embedding_threshold": float(metrics["methods"]["frozen_embedding"]["threshold"]),
    }


def choose_ditto(root: Path, dataset: str) -> dict[str, Any]:
    candidates = []
    for variant in ("plain", "mixda_all"):
        for seed_dir in sorted((root / "artifacts" / "exp01_ditto_official" / variant / dataset).glob("seed_*")):
            metrics = json.loads((seed_dir / "metrics.json").read_text())
            candidates.append((float(metrics["validation"]["f1"]), variant == "plain", -int(metrics["seed"]), variant, seed_dir, metrics))
    if not candidates:
        raise FileNotFoundError(f"No official Ditto checkpoints for {dataset}")
    _, _, _, variant, directory, metrics = max(candidates)
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    return {
        "variant": variant, "seed": int(metrics["seed"]),
        "threshold": float(checkpoint["threshold"]),
        "validation_f1": float(metrics["validation"]["f1"]),
        "checkpoint": checkpoint,
    }


def ditto_scores(
    frame: pd.DataFrame,
    model: OfficialDittoModel,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    left = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize(frame, "left").tolist()]
    right = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize(frame, "right").tolist()]
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = [
                tokenizer.encode(text=a, text_pair=b, max_length=max_length, truncation=True)
                for a, b in zip(left[start : start + batch_size], right[start : start + batch_size])
            ]
            width = max(map(len, batch))
            # Upstream Ditto pads with token id 0; retain that behavior exactly.
            values = torch.tensor([row + [0] * (width - len(row)) for row in batch], dtype=torch.long)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(values)
            rows.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--ditto-batch-size", type=int, default=512)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    probabilities = [float(value) for value in config["experiments"]["exp07_controlled_field_masking"]["probabilities"]]
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    splits = load_dataset(root / config["project"]["data_root"], config["datasets"][args.dataset])
    y_test = splits["test"]["label"].to_numpy(dtype=np.int8)
    genealogy = "left_name" in splits["test"]
    all_fields = list(GENEALOGY_FIELDS if genealogy else FIELDS)
    groups = GENEALOGY_MASK_GROUPS if genealogy else PRODUCT_MASK_GROUPS
    scenarios = [("random_fields", 0.0, seeds[0], all_fields)]
    for probability in probabilities:
        if probability == 0:
            continue
        for seed in seeds:
            scenarios.append(("random_fields", probability, seed, all_fields))
            scenarios.extend((name, probability, seed, fields) for name, fields in groups.items())

    device = torch.device(args.device)
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=args.device)
    prefix = next(item for item in config["frozen_backbones"] if item["id"] == MODEL_ID)["symmetric_prefix"]
    hef = choose_hef(root, args.dataset, MODEL_REVISION)
    ditto = choose_ditto(root, args.dataset)
    cross = config["cross_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(cross["id"], revision=cross["revision"])
    ditto_model = OfficialDittoModel(cross["id"], cross["revision"], device).to(device)
    ditto_model.load_state_dict(ditto["checkpoint"]["model"])

    train_structured = structured_features(splits["train"])
    train_embedding = np.load(
        root / "artifacts" / "embeddings" / args.dataset / MODEL_ID.replace("/", "__") / MODEL_REVISION / "train.npz",
        allow_pickle=True,
    )["embedding_score"].astype(float)
    output = root / "artifacts" / "exp07" / "classification" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "metrics.partial.json"
    completed: dict[str, Any] = {}
    if partial_path.exists():
        completed = json.loads(partial_path.read_text()).get("scenarios", {})
    started = time.time()
    baseline: dict[str, float] | None = None
    for mask_type, probability, seed, fields in scenarios:
        key = f"{mask_type}__p{int(round(probability*100)):02d}__seed{seed}"
        score_path = output / f"scores_{key}.npz"
        if key in completed and score_path.exists() and score_path.stat().st_size > 0:
            if probability == 0:
                baseline = {name: float(value["f1"]) for name, value in completed[key]["methods"].items()}
            continue
        masked, mask_sha256, cells_masked = controlled_mask(splits["test"], fields, probability, seed)
        structured = structured_features(masked)
        left_text = [prefix + value for value in serialize(masked, "left").tolist()]
        right_text = [prefix + value for value in serialize(masked, "right").tolist()]
        left_vec = embedding_model.encode(left_text, batch_size=args.embedding_batch_size, normalize_embeddings=True, show_progress_bar=False)
        right_vec = embedding_model.encode(right_text, batch_size=args.embedding_batch_size, normalize_embeddings=True, show_progress_bar=False)
        raw_embedding = np.sum(left_vec * right_vec, axis=1)
        embedding_score = scale(train_embedding, raw_embedding)
        rule_score = scale(train_structured["rule_score"].to_numpy(), structured["rule_score"].to_numpy())
        hef_frame = structured.copy()
        hef_frame["embedding_score"] = raw_embedding
        hef_score = hef["model"].predict_proba(hef_frame[hef["feature_names"]])[:, 1]
        ditto_score = ditto_scores(masked, ditto_model, tokenizer, device, int(cross["max_length"]), args.ditto_batch_size)
        methods = {
            "rules": classification_metrics(y_test, rule_score, hef["rule_threshold"]),
            "frozen_embedding": classification_metrics(y_test, embedding_score, hef["embedding_threshold"]),
            "official_ditto": classification_metrics(y_test, ditto_score, ditto["threshold"]),
            "best_hef": classification_metrics(y_test, hef_score, hef["threshold"]),
        }
        if probability == 0:
            baseline = {name: float(value["f1"]) for name, value in methods.items()}
        if baseline is None:
            raise RuntimeError("Unmasked baseline must be evaluated first")
        for name, value in methods.items():
            value["f1_retention"] = float(value["f1"] / baseline[name]) if baseline[name] > 0 else None
        completed[key] = {
            "mask_type": mask_type, "probability": probability, "seed": seed,
            "fields": fields, "mask_sha256": mask_sha256, "cells_masked": cells_masked,
            "rows": len(masked), "methods": methods,
        }
        np.savez_compressed(
            score_path,
            pair_id=splits["test"]["pair_id"].astype(str).to_numpy(), label=y_test,
            rules=rule_score.astype(np.float32), frozen_embedding=embedding_score.astype(np.float32),
            official_ditto=ditto_score.astype(np.float32), best_hef=hef_score.astype(np.float32),
        )
        write_json(partial_path, {"status": "running", "dataset": args.dataset, "scenarios": completed})

    result = {
        "experiment": "exp07_controlled_evidence_loss",
        "component": "classification",
        "status": "complete",
        "dataset": args.dataset,
        "protocol": {
            "probabilities": probabilities, "seeds": seeds,
            "matched_mask_across_systems": True,
            "selection": "unmasked_validation_only",
            "test_policy": "fixed test rows; no retuning after masking",
            "causal_claim": False,
        },
        "mask_groups": groups,
        "selected_hef": {key: value for key, value in hef.items() if key != "model"},
        "selected_ditto": {key: value for key, value in ditto.items() if key != "checkpoint"},
        "scenarios": completed,
        "runtime_seconds": time.time() - started,
    }
    write_json(output / "metrics.json", result)
    write_json(output / "manifest.json", {
        "status": "verified_local", "dataset": args.dataset,
        "expected_scenarios": len(scenarios), "completed_scenarios": len(completed),
        "score_files": len(list(output.glob("scores_*.npz"))),
    })
    print(output)


if __name__ == "__main__":
    main()
