#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paper1_hef.link_lives import (
    assign_parish_splits,
    build_hard_negative_pairs,
    build_primary_positive_pairs,
    enforce_record_disjoint_negatives,
    load_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("data/public_genealogy/link_lives/extracted/release_2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/public_genealogy/link_lives/processed/exp01/"
            "primary_pairs.csv.gz"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/public_genealogy/link_lives/manifests/"
            "exp01_primary_positive_join.json"
        ),
    )
    args = parser.parse_args()

    benchmark = load_benchmark(
        args.extracted_root
        / "Data"
        / "CSV"
        / "links_and_lifecourses"
        / "benchmark__v1.xlsx"
    )
    positives, report = build_primary_positive_pairs(benchmark, args.extracted_root)
    positives, split_report = assign_parish_splits(positives)
    negatives, negative_report = build_hard_negative_pairs(
        positives, args.extracted_root
    )
    negatives, disjoint_report = enforce_record_disjoint_negatives(
        positives, negatives
    )
    pairs = (
        pd.concat([positives.assign(negative_rank=0), negatives], ignore_index=True)
        .sort_values(["split", "left_id", "label", "negative_rank"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    report["split_audit"] = split_report
    report["negative_construction"] = negative_report
    report["record_disjoint_filter"] = disjoint_report
    report["final_pair_rows"] = int(len(pairs))
    report["final_positive_rows"] = int(pairs["label"].sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False, compression="gzip")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
