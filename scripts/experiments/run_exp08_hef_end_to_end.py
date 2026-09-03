#!/usr/bin/env python3
"""Experiment 8: reproducible end-to-end HEF efficiency benchmark.

The measured cold path begins with an already-loaded raw pair DataFrame and
includes pair serialization, two-sided E5 encoding plus cosine similarity,
structured feature construction, and prediction by the frozen HEF model.  The
warm/cached path reuses the materialized HEF feature matrix and measures frozen
HEF prediction.  Retrieval/candidate generation is intentionally out of scope
and is disclosed in every result file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from paper1_hef.data import load_dataset
from paper1_hef.features import serialize, structured_features


DEFAULT_MODEL_ID = "intfloat/e5-base-v2"
DEFAULT_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
DATASETS = (
    "abt_buy",
    "amazon_google",
    "link_lives_release2",
    "walmart_amazon",
    "wdc_80_medium_seen",
    "wdc_80_medium_unseen",
)
STAGES = ("serialization", "semantic_encoding_cosine", "structured_features", "hef_prediction")
SCOPE_NOTE = (
    "Measures pair scoring after a candidate pair has been supplied. Dataset download, "
    "S3/network transfer, raw-file parsing, candidate generation/dense retrieval, index "
    "construction, and retrieval recall are excluded. These excluded costs must be reported "
    "separately and must not be described as end-to-end retrieval latency."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        # macOS reports bytes; Linux reports KiB.
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024


@dataclass
class MemoryResult:
    value: Any
    elapsed_seconds: float
    rss_start_bytes: int
    rss_peak_bytes: int
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int


def measured_call(call: Callable[[], Any], sample_interval: float = 0.01) -> MemoryResult:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start_rss = rss_bytes()
    peak = [start_rss]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(sample_interval):
            peak[0] = max(peak[0], rss_bytes())

    monitor = threading.Thread(target=sample, name="rss-monitor", daemon=True)
    monitor.start()
    synchronize()
    started = time.perf_counter()
    try:
        value = call()
        synchronize()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        monitor.join(timeout=1.0)
        peak[0] = max(peak[0], rss_bytes())
    return MemoryResult(
        value=value,
        elapsed_seconds=float(elapsed),
        rss_start_bytes=start_rss,
        rss_peak_bytes=peak[0],
        gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
    )


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Timing distribution is empty or non-finite")
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def model_footprint(model: torch.nn.Module) -> dict[str, int]:
    tensors = list(model.parameters()) + list(model.buffers())
    return {
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "buffers": int(sum(value.numel() for value in model.buffers())),
        "tensor_bytes": int(sum(value.numel() * value.element_size() for value in tensors)),
    }


def select_frozen_hef(root: Path, dataset: str, model_id: str, revision: str) -> dict[str, Any]:
    directory = root / "artifacts" / "exp01_final" / dataset / model_id.replace("/", "__") / revision
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing HEF metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    candidates: list[tuple[float, str, int, Path]] = []
    for method in ("hef_gbdt", "hef_linear"):
        for run in metrics["methods"][method]["repetitions"]:
            seed = int(run["seed"])
            path = directory / f"{method}_seed_{seed}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Missing frozen HEF model: {path}")
            candidates.append((float(run["validation"]["f1"]), method, seed, path))
    validation_f1, method, seed, path = max(candidates, key=lambda row: (row[0], row[1] == "hef_gbdt", -row[2]))
    started = time.perf_counter()
    model = joblib.load(path)
    load_seconds = time.perf_counter() - started
    return {
        "model": model,
        "method": method,
        "seed": seed,
        "validation_f1": validation_f1,
        "path": path,
        "sha256": sha256(path),
        "serialized_bytes": path.stat().st_size,
        "load_seconds": load_seconds,
        "selection": "maximum validation F1 across saved HEF learner/seed runs; test metrics unused",
    }


class Pipeline:
    def __init__(self, encoder: Any, prefix: str, batch_size: int, hef_model: Any) -> None:
        self.encoder = encoder
        self.prefix = prefix
        self.batch_size = batch_size
        self.hef_model = hef_model

    def cold(self, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        left = [self.prefix + value for value in serialize(frame, "left").tolist()]
        right = [self.prefix + value for value in serialize(frame, "right").tolist()]
        timings["serialization"] = time.perf_counter() - started

        synchronize()
        started = time.perf_counter()
        left_vectors = self.encoder.encode(
            left,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        right_vectors = self.encoder.encode(
            right,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embedding_score = np.einsum("ij,ij->i", left_vectors, right_vectors, optimize=True)
        synchronize()
        timings["semantic_encoding_cosine"] = time.perf_counter() - started

        started = time.perf_counter()
        features = structured_features(frame)
        features["embedding_score"] = embedding_score.astype(float)
        expected = list(getattr(self.hef_model, "feature_names_in_", features.columns))
        missing = set(expected) - set(features.columns)
        if missing:
            raise ValueError(f"Frozen HEF expects missing features: {sorted(missing)}")
        features = features.loc[:, expected]
        timings["structured_features"] = time.perf_counter() - started

        started = time.perf_counter()
        scores = self.hef_model.predict_proba(features)[:, 1]
        timings["hef_prediction"] = time.perf_counter() - started
        if len(scores) != len(frame) or not np.isfinite(scores).all():
            raise ValueError("HEF produced missing or non-finite scores")
        return scores, features, timings

    def warm(self, features: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
        started = time.perf_counter()
        scores = self.hef_model.predict_proba(features)[:, 1]
        elapsed = time.perf_counter() - started
        if len(scores) != len(features) or not np.isfinite(scores).all():
            raise ValueError("Cached HEF produced missing or non-finite scores")
        return scores, {"hef_prediction": elapsed}


def benchmark_batch(
    call: Callable[[], tuple[Any, dict[str, float]]], rows: int, warmups: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmups):
        call()
        synchronize()
    totals: list[float] = []
    stage_values: dict[str, list[float]] = {stage: [] for stage in STAGES}
    memories: list[MemoryResult] = []
    for _ in range(repeats):
        measured = measured_call(call)
        _, timings = measured.value
        totals.append(measured.elapsed_seconds)
        for stage in STAGES:
            if stage in timings:
                stage_values[stage].append(float(timings[stage]))
        memories.append(measured)
    total = distribution(totals)
    return {
        "rows": rows,
        "warmups": warmups,
        "repeats": repeats,
        "total_seconds": total,
        "pairs_per_second_from_median": float(rows / total["median"]),
        "milliseconds_per_pair_from_median": float(total["median"] * 1000.0 / rows),
        "stage_seconds": {key: distribution(value) for key, value in stage_values.items() if value},
        "memory": {
            "rss_peak_bytes": max(value.rss_peak_bytes for value in memories),
            "rss_peak_delta_bytes": max(value.rss_peak_bytes - value.rss_start_bytes for value in memories),
            "gpu_peak_allocated_bytes": max(value.gpu_peak_allocated_bytes for value in memories),
            "gpu_peak_reserved_bytes": max(value.gpu_peak_reserved_bytes for value in memories),
        },
    }


def benchmark_online(
    frames: list[pd.DataFrame],
    call: Callable[[pd.DataFrame], tuple[Any, dict[str, float]]],
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("Online benchmark has no pairs")
    for frame in frames[:warmups]:
        call(frame)
        synchronize()
    totals: list[float] = []
    stage_values: dict[str, list[float]] = {stage: [] for stage in STAGES}

    def workload() -> None:
        for _ in range(repeats):
            for frame in frames:
                synchronize()
                started = time.perf_counter()
                _, timings = call(frame)
                synchronize()
                totals.append(time.perf_counter() - started)
                for stage in STAGES:
                    if stage in timings:
                        stage_values[stage].append(float(timings[stage]))

    measured = measured_call(workload)
    latency = distribution([value * 1000.0 for value in totals])
    return {
        "pairs": len(frames),
        "warmups": min(warmups, len(frames)),
        "repeats": repeats,
        "observations": len(totals),
        "latency_milliseconds": latency,
        "pairs_per_second_from_median_latency": float(1000.0 / latency["median"]),
        "stage_latency_milliseconds": {
            key: distribution([item * 1000.0 for item in value])
            for key, value in stage_values.items()
            if value
        },
        "memory": {
            "rss_peak_bytes": measured.rss_peak_bytes,
            "rss_peak_delta_bytes": measured.rss_peak_bytes - measured.rss_start_bytes,
            "gpu_peak_allocated_bytes": measured.gpu_peak_allocated_bytes,
            "gpu_peak_reserved_bytes": measured.gpu_peak_reserved_bytes,
        },
    }


def benchmark_dataset(
    root: Path,
    config: dict[str, Any],
    dataset: str,
    encoder: Any,
    encoder_load_seconds: float,
    model_id: str,
    revision: str,
    batch_size: int,
    repeats: int,
    warmups: int,
    online_pairs: int,
) -> dict[str, Any]:
    splits = load_dataset(root / config["project"]["data_root"], config["datasets"][dataset])
    frame = splits["test"].sort_values("pair_id").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{dataset}: empty test split")
    selection = select_frozen_hef(root, dataset, model_id, revision)
    pipeline = Pipeline(encoder, next(item["symmetric_prefix"] for item in config["frozen_backbones"] if item["id"] == model_id), batch_size, selection["model"])

    # A single untimed materialization supplies the cached path. It is never
    # counted as warm latency; the full cost is already measured by cold mode.
    _, cached_features, _ = pipeline.cold(frame)
    online_count = min(online_pairs, len(frame))
    online_frames = [frame.iloc[index : index + 1] for index in range(online_count)]
    cached_online = [cached_features.iloc[index : index + 1] for index in range(online_count)]

    batch_cold = benchmark_batch(
        lambda: ((result := pipeline.cold(frame))[0], result[2]), len(frame), warmups, repeats
    )
    batch_cached = benchmark_batch(
        lambda: pipeline.warm(cached_features), len(frame), warmups, repeats
    )
    online_cold = benchmark_online(
        online_frames,
        lambda value: ((result := pipeline.cold(value))[0], result[2]),
        warmups,
        repeats,
    )
    online_cached = benchmark_online(
        cached_online, lambda value: pipeline.warm(value), warmups, repeats
    )

    model = selection.pop("model")
    payload = {
        "experiment": "exp08_efficiency_deployment",
        "component": "hef_raw_pair_to_score_end_to_end",
        "status": "complete",
        "dataset": dataset,
        "test_pairs": len(frame),
        "protocol": {
            "cold_boundary": "loaded raw pair DataFrame -> serialization -> E5 encoding/cosine -> structured features -> frozen HEF predict_proba",
            "warm_cached_boundary": "materialized structured+embedding feature row -> frozen HEF predict_proba",
            "model_loading_timed_separately": True,
            "batch_warmups": warmups,
            "batch_repeats": repeats,
            "online_pairs_deterministic_pair_id_order": online_count,
            "online_repeats": repeats,
            "cuda_synchronize_around_measurements": True,
            "retrieval_exclusion_disclosure": SCOPE_NOTE,
        },
        "encoder": {
            "model_id": model_id,
            "revision": revision,
            "prefix": pipeline.prefix,
            "normalize_embeddings": True,
            "cosine_implementation": "dot product of L2-normalized vectors",
            "batch_size": batch_size,
            "max_sequence_length": int(encoder.max_seq_length),
            "load_seconds_once_per_run": encoder_load_seconds,
            "footprint": model_footprint(encoder),
        },
        "frozen_hef": {
            key: str(value) if isinstance(value, Path) else value for key, value in selection.items()
        }
        | {
            "sklearn_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "feature_names": list(getattr(model, "feature_names_in_", cached_features.columns)),
            "feature_count": int(cached_features.shape[1]),
        },
        "cache": {
            "rows": len(cached_features),
            "columns": int(cached_features.shape[1]),
            "feature_matrix_bytes_deep": int(cached_features.memory_usage(index=True, deep=True).sum()),
            "persistence": "in-process memory only; no disk read is included in warm latency",
        },
        "measurements": {
            "batch_cold_uncached": batch_cold,
            "batch_warm_cached": batch_cached,
            "online_cold_uncached": online_cold,
            "online_warm_cached": online_cached,
        },
    }
    return payload


def csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, metrics in payload["measurements"].items():
        online = mode.startswith("online")
        latency = metrics["latency_milliseconds"] if online else metrics["total_seconds"]
        rows.append(
            {
                "dataset": payload["dataset"],
                "mode": mode,
                "test_pairs": payload["test_pairs"],
                "observations": metrics.get("observations", metrics["repeats"]),
                "median_seconds": latency["median"] / 1000.0 if online else latency["median"],
                "p95_seconds": latency["p95"] / 1000.0 if online else latency["p95"],
                "pairs_per_second": metrics.get("pairs_per_second_from_median_latency", metrics.get("pairs_per_second_from_median")),
                "rss_peak_bytes": metrics["memory"]["rss_peak_bytes"],
                "gpu_peak_allocated_bytes": metrics["memory"]["gpu_peak_allocated_bytes"],
                "gpu_peak_reserved_bytes": metrics["memory"]["gpu_peak_reserved_bytes"],
                "hef_method": payload["frozen_hef"]["method"],
                "hef_seed": payload["frozen_hef"]["seed"],
                "encoder_id": payload["encoder"]["model_id"],
                "encoder_revision": payload["encoder"]["revision"],
                "retrieval_included": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--online-pairs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1 or args.online_pairs < 1 or args.batch_size < 1:
        parser.error("repeats, warmups, online-pairs, and batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")

    root = args.project_root.resolve()
    config_path = root / "configs" / "experiment.yaml"
    config = yaml.safe_load(config_path.read_text())
    model_spec = next(
        (item for item in config["frozen_backbones"] if item["id"] == args.model_id), None
    )
    if model_spec is None:
        raise ValueError(f"Encoder is not locked in experiment.yaml: {args.model_id}")
    if model_spec["revision"] != args.revision:
        raise ValueError(
            f"Revision mismatch: command={args.revision}, config={model_spec['revision']}"
        )

    os.environ.setdefault("USE_TF", "0")
    from sentence_transformers import SentenceTransformer

    load_started = time.perf_counter()
    encoder = SentenceTransformer(
        args.model_id,
        revision=args.revision,
        device=args.device,
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
    )
    encoder_load_seconds = time.perf_counter() - load_started
    output = (args.output or root / "artifacts" / "exp08" / "hef_end_to_end").resolve()
    output.mkdir(parents=True, exist_ok=True)

    hardware = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    results = []
    rows = []
    for dataset in args.datasets:
        payload = benchmark_dataset(
            root, config, dataset, encoder, encoder_load_seconds, args.model_id,
            args.revision, args.batch_size, args.repeats, args.warmups, args.online_pairs,
        )
        payload["hardware"] = hardware
        payload["provenance"] = {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "created_unix": time.time(),
        }
        path = output / dataset / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        # Parse-back validation is part of the completion contract.
        parsed = json.loads(path.read_text())
        if parsed["status"] != "complete" or not parsed["measurements"] or path.stat().st_size == 0:
            raise RuntimeError(f"Invalid output written for {dataset}")
        results.append(payload)
        rows.extend(csv_rows(payload))
        print(f"complete {dataset}: {path}", flush=True)

    frame = pd.DataFrame(rows)
    csv_path = output / "summary.csv"
    frame.to_csv(csv_path, index=False)
    parsed_csv = pd.read_csv(csv_path)
    if parsed_csv.empty or len(parsed_csv) != 4 * len(args.datasets):
        raise RuntimeError("summary.csv failed parse/count validation")
    manifest = {
        "experiment": "exp08_efficiency_deployment",
        "component": "hef_raw_pair_to_score_end_to_end",
        "status": "complete",
        "datasets": list(args.datasets),
        "dataset_count": len(args.datasets),
        "summary_rows": len(parsed_csv),
        "json_results": [str(output / item["dataset"] / "metrics.json") for item in results],
        "summary_csv": str(csv_path),
        "retrieval_exclusion_disclosure": SCOPE_NOTE,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not json.loads(manifest_path.read_text())["status"] == "complete":
        raise RuntimeError("manifest parse validation failed")
    print(f"complete manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
