#!/usr/bin/env python3
"""Produce row-level score diagnostics; do not discard a dataset for bad rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["split", "query_id", "candidate_id", "retrieval_rank", "label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-dir", type=Path, required=True)
    parser.add_argument("--min-std", type=float, default=1e-8)
    args = parser.parse_args()
    root = args.ranking_dir
    base = pd.read_csv(root / "test_scored.csv.gz", dtype={"query_id": str, "candidate_id": str})
    report_rows: list[pd.DataFrame] = []
    methods: dict[str, dict[str, object]] = {}
    vectors: dict[str, np.ndarray] = {}
    for filename, score, label in (
        ("anymatch_scores.csv.gz", "anymatch_score", "AnyMatch"),
        ("roberta_cross_encoder_scores.csv.gz", "mean_score", "tuned_RoBERTa"),
    ):
        path = root / filename
        outcome: dict[str, object] = {"file": filename, "expected_rows": int(len(base))}
        if not path.exists():
            outcome.update({"observed_rows": 0, "state": "repair_required", "reason": "score_file_missing"})
            methods[label] = outcome
            continue
        scored = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str})
        outcome["observed_rows"] = int(len(scored))
        duplicate = scored.duplicated(KEYS, keep=False)
        if duplicate.any():
            bad = scored.loc[duplicate, KEYS].copy()
            bad["method"] = label
            bad["issue"] = "duplicate_identifier"
            report_rows.append(bad)
        keep = scored.drop_duplicates(KEYS, keep="last")
        merged = base[KEYS].merge(keep[KEYS + [score]], on=KEYS, how="left", validate="one_to_one")
        values = merged[score].to_numpy(float)
        nonfinite = ~np.isfinite(values)
        if nonfinite.any():
            bad = merged.loc[nonfinite, KEYS].copy()
            bad["method"] = label
            bad["issue"] = "missing_or_nonfinite_score"
            report_rows.append(bad)
        finite = values[np.isfinite(values)]
        std = float(finite.std()) if len(finite) else 0.0
        outcome.update({"aligned_rows": int((~nonfinite).sum()), "nonfinite_rows": int(nonfinite.sum()), "duplicate_rows": int(duplicate.sum()), "score_std": std})
        if nonfinite.any() or duplicate.any():
            outcome.update({"state": "row_repair_required", "reason": "identifier_or_score_anomaly"})
        elif std <= args.min_std:
            outcome.update({"state": "systemic_repair_required", "reason": "degenerate_score_vector"})
        else:
            outcome["state"] = "ready"
            vectors[label] = values
        methods[label] = outcome
    relationship: dict[str, object] = {"state": "not_checked"}
    if len(vectors) == 2:
        labels = list(vectors)
        rho = float(pd.Series(vectors[labels[0]]).corr(pd.Series(vectors[labels[1]]), method="spearman"))
        same = bool(np.array_equal(vectors[labels[0]].view(np.uint8), vectors[labels[1]].view(np.uint8)))
        relationship = {"spearman_rho": rho, "byte_identical": same, "state": "ready" if not same and abs(rho) <= 0.999 else "systemic_repair_required"}
    anomalies = pd.concat(report_rows, ignore_index=True) if report_rows else pd.DataFrame(columns=[*KEYS, "method", "issue"])
    anomalies.to_csv(root / "score_integrity_row_report.csv", index=False)
    publishable = all(x.get("state") == "ready" for x in methods.values()) and relationship["state"] == "ready"
    report = {"dataset": root.name, "expected_rows": int(len(base)), "publishable": publishable, "methods": methods, "cross_comparator": relationship, "row_diagnostic_file": "score_integrity_row_report.csv", "row_diagnostic_count": int(len(anomalies)), "next_action": "continue_with_statistics" if publishable else "inspect_report_and_repair_only_the_reported_rows_or_rescore_the_affected_method"}
    (root / "score_integrity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
