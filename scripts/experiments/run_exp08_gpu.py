#!/usr/bin/env python3
"""Experiment 8 GPU efficiency for the frozen encoder and official Ditto."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from paper1_hef.data import load_dataset
from paper1_hef.features import serialize


MODEL_ID = "intfloat/e5-base-v2"
MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


class OfficialDittoModel(torch.nn.Module):
    def __init__(self, model_id: str, revision: str, device: torch.device) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id, revision=revision)
        self.device = device
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.bert(values.to(self.device, non_blocking=True))[0][:, 0, :]
        return self.fc(encoded)


def choose_ditto(root: Path, dataset: str) -> dict[str, Any]:
    candidates = []
    for variant in ("plain", "mixda_all"):
        directory = root / "artifacts" / "exp01_ditto_official" / variant / dataset
        for seed_dir in sorted(directory.glob("seed_*")):
            metrics = json.loads((seed_dir / "metrics.json").read_text())
            candidates.append(
                (
                    float(metrics["validation"]["f1"]),
                    variant == "plain",
                    -int(metrics["seed"]),
                    variant,
                    seed_dir,
                    metrics,
                )
            )
    if not candidates:
        raise FileNotFoundError(f"No official Ditto checkpoints for {dataset}")
    _, _, _, variant, directory, metrics = max(candidates)
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    return {
        "variant": variant,
        "seed": int(metrics["seed"]),
        "validation_f1": float(metrics["validation"]["f1"]),
        "checkpoint": checkpoint,
    }


def model_size(model: torch.nn.Module) -> dict[str, int]:
    parameters = sum(value.numel() for value in model.parameters())
    buffers = sum(value.numel() for value in model.buffers())
    bytes_total = sum(value.numel() * value.element_size() for value in model.parameters())
    bytes_total += sum(value.numel() * value.element_size() for value in model.buffers())
    return {"parameters": int(parameters), "buffers": int(buffers), "bytes": int(bytes_total)}


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(call: Callable[[], Any], rows: int, repeats: int) -> dict[str, Any]:
    call()
    synchronize()
    durations = []
    peaks_allocated = []
    peaks_reserved = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        call()
        synchronize()
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        peaks_allocated.append(torch.cuda.max_memory_allocated())
        peaks_reserved.append(torch.cuda.max_memory_reserved())
    median = statistics.median(durations)
    return {
        "rows": rows,
        "repeats": repeats,
        "seconds": durations,
        "median_seconds": median,
        "pairs_per_second": rows / median,
        "microseconds_per_pair": median * 1e6 / rows,
        "peak_allocated_bytes": max(peaks_allocated),
        "peak_reserved_bytes": max(peaks_reserved),
    }


def ditto_score(
    frame: pd.DataFrame,
    model: OfficialDittoModel,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    left = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize(frame, "left")]
    right = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize(frame, "right")]
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            token_rows = [
                tokenizer.encode(text=a, text_pair=b, max_length=max_length, truncation=True)
                for a, b in zip(left[start : start + batch_size], right[start : start + batch_size])
            ]
            width = max(map(len, token_rows))
            values = torch.tensor(
                [row + [0] * (width - len(row)) for row in token_rows], dtype=torch.long
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(values)
            output.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--ditto-batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--online-pairs", type=int, default=100)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs" / "experiment.yaml").read_text())
    frame = load_dataset(
        root / config["project"]["data_root"], config["datasets"][args.dataset]
    )["test"].sort_values("pair_id").reset_index(drop=True)
    online = frame.head(min(args.online_pairs, len(frame)))
    device = torch.device("cuda")
    hardware = {
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }

    from sentence_transformers import SentenceTransformer

    prefix = next(
        item["symmetric_prefix"]
        for item in config["frozen_backbones"]
        if item["id"] == MODEL_ID
    )
    embedding_model = SentenceTransformer(
        MODEL_ID, revision=MODEL_REVISION, device="cuda"
    )

    def embedding_call(values: pd.DataFrame = frame) -> np.ndarray:
        left = [prefix + value for value in serialize(values, "left")]
        right = [prefix + value for value in serialize(values, "right")]
        left_vectors = embedding_model.encode(
            left,
            batch_size=args.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        right_vectors = embedding_model.encode(
            right,
            batch_size=args.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.sum(left_vectors * right_vectors, axis=1)

    embedding_batch = benchmark(embedding_call, len(frame), args.repeats)
    embedding_online = benchmark(
        lambda: embedding_call(online), len(online), args.repeats
    )
    embedding_size = model_size(embedding_model)
    del embedding_model
    torch.cuda.empty_cache()

    selection = choose_ditto(root, args.dataset)
    cross = config["cross_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(cross["id"], revision=cross["revision"])
    ditto_model = OfficialDittoModel(cross["id"], cross["revision"], device).to(device)
    ditto_model.load_state_dict(selection["checkpoint"]["model"])
    ditto_batch = benchmark(
        lambda: ditto_score(
            frame,
            ditto_model,
            tokenizer,
            device,
            int(cross["max_length"]),
            args.ditto_batch_size,
        ),
        len(frame),
        args.repeats,
    )
    ditto_online = benchmark(
        lambda: ditto_score(
            online,
            ditto_model,
            tokenizer,
            device,
            int(cross["max_length"]),
            min(args.ditto_batch_size, len(online)),
        ),
        len(online),
        args.repeats,
    )

    result = {
        "experiment": "exp08_efficiency_deployment",
        "component": "gpu_encoder_and_cross_encoder",
        "status": "component_complete",
        "dataset": args.dataset,
        "rows": len(frame),
        "hardware": hardware,
        "protocol": {
            "warmup_before_timing": True,
            "cuda_synchronize": True,
            "repeats": args.repeats,
            "online_pairs": len(online),
            "batch_reranking_and_online_reported_separately": True,
        },
        "frozen_embedding": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "batch_size": args.embedding_batch_size,
            "model_size": embedding_size,
            "batch": embedding_batch,
            "online": embedding_online,
        },
        "official_ditto": {
            "model_id": cross["id"],
            "revision": cross["revision"],
            "variant": selection["variant"],
            "seed": selection["seed"],
            "batch_size": args.ditto_batch_size,
            "model_size": model_size(ditto_model),
            "batch": ditto_batch,
            "online": ditto_online,
        },
    }
    output = root / "artifacts" / "exp08" / "gpu" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
