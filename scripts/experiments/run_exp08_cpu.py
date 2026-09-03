#!/usr/bin/env python3
"""Experiment 8 CPU feature/fusion efficiency component."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from paper1_hef.data import load_dataset
from paper1_hef.features import structured_features


def timed_scores(model, frame: pd.DataFrame, repeats: int = 5) -> dict[str, float]:
    model.predict_proba(frame.iloc[: min(32, len(frame))])
    batch_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(frame)
        batch_times.append(time.perf_counter() - started)
    seconds = float(np.median(batch_times))
    online = []
    for index in range(min(1000, len(frame))):
        started = time.perf_counter_ns()
        model.predict_proba(frame.iloc[index : index + 1])
        online.append(time.perf_counter_ns() - started)
    return {
        "batch_rows": len(frame),
        "batch_seconds_median": seconds,
        "batch_pairs_per_second": float(len(frame) / seconds),
        "online_pairs": len(online),
        "online_latency_ms_median": float(np.median(online) / 1e6),
        "online_latency_ms_p95": float(np.quantile(online, 0.95) / 1e6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--revision", default="f52bf8ec8c7124536f0efb74aca902b2995e5bcd")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    splits = load_dataset(root / config["project"]["data_root"], config["datasets"][args.dataset])
    started = time.perf_counter()
    feature_started = time.perf_counter()
    features = {split: structured_features(frame) for split, frame in splits.items()}
    structured_seconds = time.perf_counter() - feature_started
    emb_dir = root / "artifacts" / "embeddings" / args.dataset / args.model.replace("/", "__") / args.revision
    for split in features:
        features[split]["embedding_score"] = np.load(emb_dir / f"{split}.npz", allow_pickle=True)["embedding_score"].astype(float)
    y = {split: splits[split]["label"].to_numpy(dtype=np.int8) for split in splits}
    out = root / "artifacts" / "exp08" / "cpu_fusion" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    methods = {}
    for name, model in {
        "hef_linear": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=20260725),
        "hef_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=20260725,
        ),
    }.items():
        train_started = time.perf_counter()
        model.fit(features["train"], y["train"])
        train_seconds = time.perf_counter() - train_started
        path = out / f"{name}.joblib"
        joblib.dump(model, path)
        methods[name] = {
            "fusion_training_seconds": train_seconds,
            "model_size_bytes": path.stat().st_size,
            "gpu_required": False,
            "batch_reranking": timed_scores(model, features["test"]),
        }
    payload = {
        "experiment": "exp08_efficiency_deployment",
        "component": "cpu_structured_feature_and_fusion",
        "status": "component_complete",
        "dataset": args.dataset,
        "hardware": {
            "cpu_count": os.cpu_count(),
            "peak_rss_mib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
        "offline_feature_construction": {
            "pairs": int(sum(len(frame) for frame in splits.values())),
            "seconds": structured_seconds,
            "pairs_per_second": float(sum(len(frame) for frame in splits.values()) / structured_seconds),
        },
        "methods": methods,
        "runtime_seconds": time.perf_counter() - started,
        "scope_note": "GPU encoders, cross-encoders, retrieval, and LLM cost are separate Exp8 components.",
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
