#!/usr/bin/env python3
"""CPU-only admission check for all 36 neural-fusion cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from paper1_hef.data import load_dataset
from run_exp01_neural_oof_fusion import component_folds

SEEDS = (20260725, 20260726, 20260727)
DATASETS = (
    "abt_buy", "amazon_google", "walmart_amazon", "wdc_80_medium_seen",
    "wdc_80_medium_unseen", "link_lives_release2",
)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(); root = args.project_root.resolve()
    config = yaml.safe_load((root / "configs" / "experiment.yaml").read_text())
    report = {"admitted": True, "datasets": {}}
    for dataset in DATASETS:
        spec = config["datasets"][dataset]
        splits = load_dataset(root / config["project"]["data_root"], spec)
        plan = component_folds(splits["train"], spec["adapter"])
        item = {"train_rows": len(splits["train"]), "component_count": plan.report["component_count"],
                "folds": plan.report["folds"], "cells": {}}
        for seed in SEEDS:
            paths = {
                "roberta": root / "artifacts" / "exp01_cross_encoder" / dataset / f"seed_{seed}",
                "jina": root / "artifacts" / "exp01_jina_finetuned" / "v1" / dataset / f"seed_{seed}",
            }
            item["cells"][str(seed)] = {}
            for family, path in paths.items():
                required = [path / "metrics.json", path / "scores.npz"]
                present = all(file.exists() and file.stat().st_size > 0 for file in required)
                item["cells"][str(seed)][family] = {"full_fit_prerequisite": str(path), "present": present}
                report["admitted"] = bool(report["admitted"] and present)
        report["datasets"][dataset] = item
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["admitted"]:
        raise SystemExit(2)


if __name__ == "__main__": main()
