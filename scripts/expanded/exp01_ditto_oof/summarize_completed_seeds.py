#!/usr/bin/env python3
"""Atomically aggregate the three completed HEF fusion seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


SEEDS = (20260725, 20260726, 20260727)
REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=("plain", "mixda_all"), required=True)
    parser.add_argument("--s3-root")
    args = parser.parse_args()
    base = (
        args.project_root / "artifacts" / "exp01_hef_cross_evidence" / "v1"
        / args.dataset / "intfloat__e5-base-v2" / REVISION
        / f"official_ditto_{args.variant}"
    )
    if not all((base / f"seed_{seed}" / "SUCCESS.json").exists() for seed in SEEDS):
        return
    lock = base / ".summary.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    try:
        runs = [json.loads((base / f"seed_{seed}" / "metrics.json").read_text()) for seed in SEEDS]
        values = np.asarray([run["test"]["f1"] for run in runs], dtype=float)
        summary = {
            "dataset": args.dataset,
            "method": f"hef_gbdt_e5_plus_official_ditto_{args.variant}",
            "seeds": list(SEEDS),
            "runs": len(SEEDS),
            "test_f1_mean": float(values.mean()),
            "test_f1_std_sample": float(values.std(ddof=1)),
            "test_f1_by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values, strict=True)},
            "all_seed_outputs_validated": True,
        }
        temporary = base / f".summary.{os.getpid()}.json"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, base / "summary.json")
        if args.s3_root:
            relative = base.relative_to(args.project_root)
            subprocess.run(
                ["aws", "s3", "cp", str(base / "summary.json"), f"{args.s3_root}/{relative}/summary.json", "--only-show-errors"],
                check=True,
            )
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

