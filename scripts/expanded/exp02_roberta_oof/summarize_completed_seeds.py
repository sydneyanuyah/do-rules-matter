#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

SEEDS = (20260725, 20260726, 20260727)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--s3-root")
    args = parser.parse_args()
    base = args.project_root / "artifacts" / "exp02_hef_cross_evidence" / "v1" / args.dataset / "e5_plus_tuned_roberta"
    if not all((base / f"seed_{s}" / "SUCCESS.json").exists() for s in SEEDS):
        return
    lock = base / ".summary.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return
    try:
        runs = [json.loads((base / f"seed_{s}" / "metrics.json").read_text()) for s in SEEDS]
        keys = ["mrr", "hits_at_1", "hits_at_5", "hits_at_10", "hits_at_100", "ndcg"]
        summary: dict[str, object] = {
            "dataset": args.dataset, "method": "hef_gbdt_with_e5_plus_tuned_roberta",
            "seeds": list(SEEDS), "runs": 3,
            "retrieval_hits_at_100_end_to_end": runs[0]["test"]["100"]["hits_at_100_end_to_end"],
        }
        for key in keys:
            values = np.asarray([run["test"]["100"]["conditional"][key] for run in runs], dtype=float)
            summary[f"conditional_{key}_mean"] = float(values.mean())
            summary[f"conditional_{key}_std_sample"] = float(values.std(ddof=1))
        path = base / "summary.json"
        temp = base / f".summary.{os.getpid()}.json"
        temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
        if args.s3_root:
            rel = base.relative_to(args.project_root)
            subprocess.run(["aws", "s3", "cp", str(path), f"{args.s3_root}/{rel}/summary.json", "--only-show-errors"], check=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

