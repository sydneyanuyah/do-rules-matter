#!/usr/bin/env python3
"""Experiment 8 inference-efficiency measurements for the Jina reranker."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paper1_hef.data import load_dataset
from paper1_hef.features import serialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--online-pairs", type=int, default=100)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs" / "experiment.yaml").read_text())
    frame = load_dataset(
        root / config["project"]["data_root"], config["datasets"][args.dataset]
    )["test"].sort_values("pair_id").reset_index(drop=True)
    spec = config["exp02_jina_cross_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        trust_remote_code=True,
        dtype=torch.float16,
    ).cuda().eval()

    left = serialize(frame, "left").tolist()
    right = serialize(frame, "right").tolist()

    def score(count: int, batch_size: int) -> None:
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                encoded = tokenizer(
                    left[start : start + batch_size],
                    right[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=int(spec["max_length"]),
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                encoded = {key: value.cuda(non_blocking=True) for key, value in encoded.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(**encoded)

    def benchmark(count: int, batch_size: int) -> dict:
        score(count, batch_size)
        torch.cuda.synchronize()
        durations = []
        allocated = []
        reserved = []
        for _ in range(args.repeats):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            score(count, batch_size)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            allocated.append(torch.cuda.max_memory_allocated())
            reserved.append(torch.cuda.max_memory_reserved())
        median = statistics.median(durations)
        return {
            "rows": count,
            "repeats": args.repeats,
            "seconds": durations,
            "median_seconds": median,
            "pairs_per_second": count / median,
            "microseconds_per_pair": median * 1e6 / count,
            "peak_allocated_bytes": max(allocated),
            "peak_reserved_bytes": max(reserved),
        }

    online_count = min(args.online_pairs, len(frame))
    size_bytes = sum(x.numel() * x.element_size() for x in model.parameters())
    result = {
        "experiment": "exp08_efficiency_deployment",
        "component": "jina_cross_encoder",
        "status": "component_complete",
        "dataset": args.dataset,
        "rows": len(frame),
        "model_id": spec["id"],
        "revision": spec["revision"],
        "batch_size": args.batch_size,
        "model_size_bytes": size_bytes,
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "protocol": {
            "warmup_before_timing": True,
            "cuda_synchronize": True,
            "repeats": args.repeats,
            "batch_reranking_and_online_reported_separately": True,
        },
        "batch": benchmark(len(frame), args.batch_size),
        "online": benchmark(online_count, min(args.batch_size, online_count)),
    }
    output = root / "artifacts" / "exp08" / "jina" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
