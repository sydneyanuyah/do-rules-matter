#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
import numpy as np

SEEDS = (20260725, 20260726, 20260727)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--dataset", required=True); p.add_argument("--family", choices=("roberta", "jina"), required=True)
    p.add_argument("--s3-root"); a = p.parse_args()
    base = a.project_root / "artifacts" / "exp01_hef_neural_oof" / "v1" / a.dataset / f"e5_plus_{a.family}"
    if not all((base / f"seed_{s}" / "SUCCESS.json").exists() for s in SEEDS): return
    lock = base / ".summary.lock"
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError: return
    try:
        runs = [json.loads((base / f"seed_{s}" / "metrics.json").read_text()) for s in SEEDS]
        values = np.asarray([run["test"]["f1"] for run in runs], dtype=float)
        summary = {"dataset": a.dataset, "family": a.family, "method": f"hef_gbdt_e5_plus_finetuned_{a.family}",
                   "seeds": list(SEEDS), "runs": 3, "test_f1_mean": float(values.mean()),
                   "test_f1_std_sample": float(values.std(ddof=1)),
                   "test_f1_by_seed": {str(s): float(v) for s, v in zip(SEEDS, values, strict=True)}}
        tmp = base / f".summary.{os.getpid()}.json"; tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        out = base / "summary.json"; os.replace(tmp, out)
        if a.s3_root:
            rel = base.relative_to(a.project_root)
            subprocess.run(["aws", "s3", "cp", str(out), f"{a.s3_root}/{rel}/summary.json", "--only-show-errors"], check=True)
    finally: lock.unlink(missing_ok=True)

if __name__ == "__main__": main()

