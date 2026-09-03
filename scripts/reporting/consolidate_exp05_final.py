#!/usr/bin/env python3
"""Consolidate the verified 5,616-cell Experiment 5 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_FAMILIES = 13
EXPECTED_DATASETS = 6
EXPECTED_FRACTIONS = 24
EXPECTED_SEEDS = 3
METRICS = ("f1", "precision", "recall", "roc_auc", "average_precision", "brier", "ece")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run: list[dict] = []
    errors: list[str] = []
    for path in sorted(args.input_root.glob("*/*/fraction_*/seed_*/metrics.json")):
        relative = path.relative_to(args.input_root).parts
        family, dataset, fraction_part, seed_part = relative[:4]
        try:
            payload = json.loads(path.read_text())
            test = payload["test"]
            row = {
                "model_family": family, "dataset": dataset,
                "fraction_percent": int(fraction_part.removeprefix("fraction_")),
                "seed": int(seed_part.removeprefix("seed_")),
                "source_metrics": str(path),
            }
            for metric in METRICS:
                value = test.get(metric, "")
                if value != "" and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f"invalid {metric}: {value}")
                row[metric] = value
            per_run.append(row)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    expected = EXPECTED_FAMILIES * EXPECTED_DATASETS * EXPECTED_FRACTIONS * EXPECTED_SEEDS
    logical = {(r["model_family"], r["dataset"], r["fraction_percent"], r["seed"]) for r in per_run}
    families = sorted({r["model_family"] for r in per_run})
    datasets = sorted({r["dataset"] for r in per_run})
    fractions = sorted({r["fraction_percent"] for r in per_run})
    seeds = sorted({r["seed"] for r in per_run})
    if errors or len(per_run) != expected or len(logical) != expected:
        raise RuntimeError(f"incomplete/invalid matrix: rows={len(per_run)}, logical={len(logical)}, errors={errors[:3]}")
    if not (len(families) == EXPECTED_FAMILIES and len(datasets) == EXPECTED_DATASETS and len(fractions) == EXPECTED_FRACTIONS and len(seeds) == EXPECTED_SEEDS):
        raise RuntimeError("matrix dimension mismatch")

    fields = ["model_family", "dataset", "fraction_percent", "seed", *METRICS, "source_metrics"]
    per_run_path = args.output_dir / "exp05_per_run_metrics.csv"
    write_csv(per_run_path, per_run, fields)

    grouped: dict[tuple, list[dict]] = {}
    for row in per_run:
        grouped.setdefault((row["model_family"], row["dataset"], row["fraction_percent"]), []).append(row)
    summary = []
    for (family, dataset, fraction), rows in sorted(grouped.items()):
        out = {"model_family": family, "dataset": dataset, "fraction_percent": fraction, "seed_count": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric] != ""]
            out[f"{metric}_mean"] = statistics.mean(values) if values else ""
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(out)
    summary_fields = ["model_family", "dataset", "fraction_percent", "seed_count"] + [item for metric in METRICS for item in (f"{metric}_mean", f"{metric}_std")]
    summary_path = args.output_dir / "exp05_results_by_dataset_fraction.csv"
    write_csv(summary_path, summary, summary_fields)

    overall = []
    for family in families:
        for fraction in fractions:
            rows = [row for row in per_run if row["model_family"] == family and row["fraction_percent"] == fraction]
            out = {"model_family": family, "fraction_percent": fraction, "dataset_count": len({r['dataset'] for r in rows}), "run_count": len(rows)}
            for metric in METRICS:
                values = [float(row[metric]) for row in rows if row[metric] != ""]
                out[f"{metric}_macro_mean"] = statistics.mean(values) if values else ""
                out[f"{metric}_macro_std"] = statistics.stdev(values) if len(values) > 1 else ""
            overall.append(out)
    overall_fields = ["model_family", "fraction_percent", "dataset_count", "run_count"] + [item for metric in METRICS for item in (f"{metric}_macro_mean", f"{metric}_macro_std")]
    overall_path = args.output_dir / "exp05_results_macro_all_datasets.csv"
    write_csv(overall_path, overall, overall_fields)

    manifest = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_cells": expected, "verified_cells": len(logical), "families": families,
        "datasets": datasets, "fractions_percent": fractions, "seeds": seeds,
        "sample_standard_deviation": True, "validation_only_selection": True,
        "untouched_test_evaluation": True, "leakage_safe_oof_required_and_preserved": True,
        "files": {},
    }
    for path in (per_run_path, summary_path, overall_path):
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme = args.output_dir / "README.md"
    readme.write_text(
        "# Experiment 5 — Final Consolidated Results\n\n"
        "Status: **complete**. The strict matrix contains 13 model families × 6 datasets × "
        "24 label fractions × 3 fixed seeds = **5,616 verified cells**.\n\n"
        "Means and sample standard deviations are reported from the three fixed seeds. "
        "Model selection used validation data only; test evaluation remained untouched; "
        "task-trained stacking inputs use leakage-safe out-of-fold predictions.\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
