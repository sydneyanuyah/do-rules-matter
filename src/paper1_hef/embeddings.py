from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_dataset, validate_splits
from .features import serialize


def encode_dataset(
    project_root: Path,
    config: dict[str, Any],
    dataset_name: str,
    model_id: str,
    revision: str,
    batch_size: int,
    device: str,
) -> Path:
    if revision == "LOCK_BEFORE_GPU_RUN":
        raise ValueError("A model commit/revision must be locked before a GPU run.")
    os.environ.setdefault("USE_TF", "0")
    from sentence_transformers import SentenceTransformer

    model_spec = next(item for item in config["frozen_backbones"] if item["id"] == model_id)
    splits = load_dataset(project_root / config["project"]["data_root"], config["datasets"][dataset_name])
    validate_splits(
        splits,
        enforce_offer_disjoint=dataset_name == "wdc_80_medium_unseen",
        report_offer_overlap=config["datasets"][dataset_name]["adapter"] == "wdc",
    )
    output = (
        project_root
        / config["project"]["output_root"]
        / "embeddings"
        / dataset_name
        / model_id.replace("/", "__")
        / revision
    )
    output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        model_id,
        revision=revision,
        device=device,
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
    )
    started = time.time()
    manifest: dict[str, Any] = {
        "dataset": dataset_name,
        "model_id": model_id,
        "revision": revision,
        "device": device,
        "batch_size": batch_size,
        "normalize": True,
        "prefix": model_spec["symmetric_prefix"],
        "trust_remote_code": bool(model_spec.get("trust_remote_code", False)),
        "splits": {},
    }
    for split, frame in splits.items():
        prefix = model_spec["symmetric_prefix"]
        left_text = [prefix + value for value in serialize(frame, "left").tolist()]
        right_text = [prefix + value for value in serialize(frame, "right").tolist()]
        left = model.encode(left_text, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        right = model.encode(right_text, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        scores = np.sum(left * right, axis=1)
        np.savez_compressed(
            output / f"{split}.npz",
            pair_id=frame["pair_id"].astype(str).to_numpy(dtype=str),
            label=frame["label"].to_numpy(dtype=np.int8),
            embedding_score=scores.astype(np.float32),
        )
        manifest["splits"][split] = {"rows": len(frame)}
    manifest["runtime_seconds"] = time.time() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output
