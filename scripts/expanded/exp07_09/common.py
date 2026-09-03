"""Shared, locked utilities for revised Experiments 7 and 8."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


MASK_PROBABILITIES = (0.0, 0.1, 0.3, 0.5, 0.7)
MASK_SEEDS = (20260725, 20260726, 20260727)


def scenarios() -> list[tuple[float, int]]:
    """One clean baseline plus three repeats at each nonzero mask level."""
    return [(0.0, MASK_SEEDS[0])] + [
        (probability, seed)
        for probability in MASK_PROBABILITIES[1:]
        for seed in MASK_SEEDS
    ]


def stable_uniform(*parts: object) -> float:
    key = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)


def controlled_record_mask(
    frame: pd.DataFrame, fields: list[str], probability: float, seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mask fields deterministically at record level with one mask shared by all models."""
    if probability not in MASK_PROBABILITIES or seed not in MASK_SEEDS:
        raise ValueError("Mask probability/seed is outside the locked protocol")
    out = frame.copy()
    selected: set[str] = set()
    affected_rows = 0
    for side, id_column in (("left", "query_id"), ("right", "candidate_id")):
        ids = out[id_column].astype(str).to_numpy()
        for field in fields:
            column = f"{side}_{field}"
            if column not in out:
                continue
            decisions = np.asarray([
                stable_uniform("exp07-v2", seed, record_id, side, field) < probability
                for record_id in ids
            ])
            nonempty = out[column].notna().to_numpy() & out[column].astype(str).str.strip().ne("").to_numpy()
            actual = decisions & nonempty
            if actual.any():
                out.loc[actual, column] = ""
                affected_rows += int(actual.sum())
                selected.update(
                    f"{record_id}|{side}|{field}" for record_id, flag in zip(ids, actual) if flag
                )
    digest = hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()
    return out, {
        "probability": probability, "seed": seed, "mask_sha256": digest,
        "unique_record_fields_masked": len(selected), "affected_pair_cells": affected_rows,
    }


def query_metrics(frame: pd.DataFrame, score_column: str) -> tuple[dict[str, float], pd.DataFrame]:
    """Stable top-100 query metrics and per-query contributions for paired bootstrap."""
    required = {"query_id", "candidate_id", "retrieval_rank", "label", score_column}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate query/candidate pairs")
    if not np.isfinite(frame[score_column]).all() or frame[score_column].nunique() < 2:
        raise ValueError("Scores are nonfinite or degenerate")
    rows = []
    for query_id, group in frame.groupby("query_id", sort=False):
        ordered = group.sort_values(
            [score_column, "retrieval_rank"], ascending=[False, True], kind="stable"
        ).head(100)
        positions = np.flatnonzero(ordered["label"].to_numpy(dtype=int) == 1)
        rank = int(positions[0] + 1) if len(positions) else 0
        rows.append({
            "query_id": str(query_id), "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "hit_at_1": float(rank == 1), "hit_at_100": float(rank > 0), "rank": rank,
        })
    per_query = pd.DataFrame(rows)
    return {
        "queries": int(len(per_query)), "mrr_at_100": float(per_query["reciprocal_rank"].mean()),
        "hits_at_1": float(per_query["hit_at_1"].mean()),
        "hits_at_100": float(per_query["hit_at_100"].mean()),
    }, per_query


def rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


@contextmanager
def memory_timer(device: str = "cuda") -> Iterator[dict[str, Any]]:
    import torch
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    result: dict[str, Any] = {"rss_start_bytes": rss_bytes()}
    peak = [result["rss_start_bytes"]]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.01):
            peak[0] = max(peak[0], rss_bytes())

    monitor = threading.Thread(target=sample, daemon=True); monitor.start()
    started = time.perf_counter()
    try:
        yield result
    finally:
        if use_cuda: torch.cuda.synchronize()
        result["seconds"] = float(time.perf_counter() - started)
        stop.set(); monitor.join(timeout=1)
        peak[0] = max(peak[0], rss_bytes())
        result["rss_peak_bytes"] = peak[0]
        result["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if use_cuda else 0
        result["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved()) if use_cuda else 0


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
