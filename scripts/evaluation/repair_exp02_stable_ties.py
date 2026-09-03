#!/usr/bin/env python3
"""Metrics-only repair for Exp2/Exp4 HEF ranking tie handling.

The scored rows and every non-target metrics payload are immutable inputs.  Only
``methods.{hef_linear,hef_gbdt,hef_rank}.three_seed_ensemble`` is recomputed.
Candidates are ordered by score descending and retrieval rank ascending using a
stable sort, matching the declared ranking protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_METHODS = ("hef_linear", "hef_gbdt", "hef_rank")
REQUIRED_COLUMNS = {
    "query_id", "candidate_id", "retrieval_rank", "label", *TARGET_METHODS
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def without_targets(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a projection used to prove unrelated payloads did not change."""
    projected = copy.deepcopy(metrics)
    methods = projected.get("methods", {})
    for method in TARGET_METHODS:
        payload = methods.get(method)
        if isinstance(payload, dict):
            payload.pop("three_seed_ensemble", None)
    return projected


def ranking_metrics(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    reciprocal: list[float] = []
    ndcg: list[float] = []
    hits = {1: [], 5: [], 10: [], 100: []}
    for _, group in frame.groupby("query_id", sort=False):
        # mergesort is explicitly stable. retrieval_rank is the declared tie-break.
        ordered = group.sort_values(
            [score, "retrieval_rank"], ascending=[False, True], kind="mergesort"
        )
        labels = ordered["label"].to_numpy(dtype=int)
        positives = np.flatnonzero(labels == 1)
        if len(positives) == 0:
            raise ValueError("conditional ranking group unexpectedly has no positive")
        first = int(positives[0])
        reciprocal.append(1.0 / (first + 1))
        for k in hits:
            hits[k].append(float(first < k))
        discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
        dcg = float(np.sum(labels * discounts))
        ideal = np.sort(labels)[::-1]
        idcg = float(np.sum(ideal * discounts))
        ndcg.append(dcg / idcg)
    return {
        "mrr": float(np.mean(reciprocal)),
        "hits_at_1": float(np.mean(hits[1])),
        "hits_at_5": float(np.mean(hits[5])),
        "hits_at_10": float(np.mean(hits[10])),
        "hits_at_100": float(np.mean(hits[100])),
        "ndcg": float(np.mean(ndcg)),
        "queries": len(reciprocal),
    }


def evaluate(frame: pd.DataFrame, score: str, sizes: list[int]) -> dict[str, Any]:
    total_queries = int(frame["query_id"].nunique())
    if total_queries == 0:
        raise ValueError("test_scored contains no queries")
    output: dict[str, Any] = {}
    for size in sizes:
        pool = frame[frame["retrieval_rank"].le(size)].copy()
        hit_counts = pool.groupby("query_id", sort=False)["label"].sum()
        hit_ids = set(hit_counts[hit_counts.gt(0)].index)
        conditional = pool[pool["query_id"].isin(hit_ids)]
        pool_hit = float(len(hit_ids) / total_queries)
        output[str(size)] = {
            "pool_hit": pool_hit,
            f"hits_at_{size}_end_to_end": pool_hit,
            "pool_hit_queries": len(hit_ids),
            "total_queries": total_queries,
            "conditional": ranking_metrics(conditional, score) if hit_ids else None,
        }
    return output


def infer_sizes(metrics: dict[str, Any]) -> list[int]:
    declared = metrics.get("pool_sizes")
    if isinstance(declared, list) and declared:
        return sorted({int(value) for value in declared})
    for method in TARGET_METHODS:
        ensemble = metrics.get("methods", {}).get(method, {}).get("three_seed_ensemble")
        if isinstance(ensemble, dict):
            sizes = [int(key) for key in ensemble if str(key).isdigit()]
            if sizes:
                return sorted(set(sizes))
    raise ValueError("cannot infer pool sizes from metrics.json")


def tie_summary(frame: pd.DataFrame, method: str) -> dict[str, int]:
    grouped = frame.groupby(["query_id", method], dropna=False, sort=False).size()
    tied = grouped[grouped.gt(1)]
    tied_queries = (
        int(tied.reset_index()["query_id"].nunique()) if len(tied) else 0
    )
    return {
        "tied_score_groups": int(len(tied)),
        "rows_in_tied_score_groups": int(tied.sum()) if len(tied) else 0,
        "queries_with_score_ties": tied_queries,
    }


def discover(roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in root.rglob("metrics.json"):
            if (metrics_path.parent / "test_scored.csv.gz").is_file():
                found.add(metrics_path.resolve())
    return sorted(found)


def repair_one(metrics_path: Path, apply: bool) -> dict[str, Any]:
    scores_path = metrics_path.parent / "test_scored.csv.gz"
    before = json.loads(metrics_path.read_text())
    frame = pd.read_csv(scores_path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{scores_path}: missing columns {missing}")
    if frame[list(TARGET_METHODS)].isna().any().any():
        raise ValueError(f"{scores_path}: target score columns contain nulls")
    if not np.isfinite(frame[list(TARGET_METHODS)].to_numpy(dtype=float)).all():
        raise ValueError(f"{scores_path}: target score columns contain non-finite values")
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError(f"{scores_path}: duplicate query/candidate pairs")
    if frame.duplicated(["query_id", "retrieval_rank"]).any():
        raise ValueError(f"{scores_path}: duplicate retrieval ranks within a query")

    sizes = infer_sizes(before)
    after = copy.deepcopy(before)
    method_audit: dict[str, Any] = {}
    for method in TARGET_METHODS:
        payload = after.get("methods", {}).get(method)
        if not isinstance(payload, dict) or "three_seed_ensemble" not in payload:
            raise ValueError(f"{metrics_path}: missing methods.{method}.three_seed_ensemble")
        old = copy.deepcopy(payload["three_seed_ensemble"])
        new = evaluate(frame, method, sizes)
        payload["three_seed_ensemble"] = new
        method_audit[method] = {
            "before_sha256": object_sha256(old),
            "after_sha256": object_sha256(new),
            "changed": old != new,
            "ties": tie_summary(frame, method),
            "before": old,
            "after": new,
        }

    unrelated_before = object_sha256(without_targets(before))
    unrelated_after = object_sha256(without_targets(after))
    if unrelated_before != unrelated_after:
        raise AssertionError("non-target metrics payload changed")

    record = {
        "metrics_path": str(metrics_path),
        "scores_path": str(scores_path),
        "dataset": before.get("dataset", metrics_path.parent.name),
        "model_id": before.get("model_id"),
        "rows": int(len(frame)),
        "queries": int(frame["query_id"].nunique()),
        "pool_sizes": sizes,
        "scores_sha256": file_sha256(scores_path),
        "metrics_before_sha256": file_sha256(metrics_path),
        "unrelated_payload_sha256": unrelated_before,
        "methods": method_audit,
        "applied": apply,
    }
    if apply:
        backup = metrics_path.with_name("metrics.before_stable_tie_repair.json")
        if not backup.exists():
            backup.write_bytes(metrics_path.read_bytes())
        encoded = json.dumps(after, indent=2, sort_keys=True, allow_nan=False) + "\n"
        fd, temp_name = tempfile.mkstemp(prefix=".metrics.", suffix=".json", dir=metrics_path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, metrics_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        record["metrics_after_sha256"] = file_sha256(metrics_path)
        record["backup_path"] = str(backup)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", action="append", type=Path, required=True,
        help="Ranking root to scan recursively; repeat for canonical and backbone roots.",
    )
    parser.add_argument("--apply", action="store_true", help="Atomically update metrics.json files.")
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    paths = discover(args.root)
    if not paths:
        raise SystemExit("no metrics.json + test_scored.csv.gz pairs discovered")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            records.append(repair_one(path, args.apply))
        except Exception as exc:  # retain a complete audit, then fail closed
            failures.append({"metrics_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "schema": "paper1.exp02_stable_tie_repair.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "score_desc_then_retrieval_rank_asc_stable_mergesort",
        "target_methods": list(TARGET_METHODS),
        "mode": "apply" if args.apply else "dry_run",
        "discovered": len(paths),
        "succeeded": len(records),
        "failed": len(failures),
        "records": records,
        "failures": failures,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: manifest[key] for key in ("mode", "discovered", "succeeded", "failed")}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
