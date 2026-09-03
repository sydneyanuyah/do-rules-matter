from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import classification_metrics, select_threshold


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json" or path.name.endswith(".json.gz"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def validate_group_splits(frame: pd.DataFrame, split_col: str, query_col: str) -> dict[str, Any]:
    expected = {"train", "valid", "test"}
    observed = set(frame[split_col].dropna().astype(str))
    if observed != expected:
        raise ValueError(f"{split_col} must contain exactly {sorted(expected)}; got {sorted(observed)}")
    queries = {
        split: set(frame.loc[frame[split_col] == split, query_col].astype(str))
        for split in sorted(expected)
    }
    overlap = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        count = len(queries[left] & queries[right])
        overlap[f"{left}_{right}"] = count
        if count:
            raise ValueError(f"Query leakage between {left} and {right}: {count}")
    return {
        "queries": {split: len(values) for split, values in queries.items()},
        "query_overlap": overlap,
        "rows": {
            split: int((frame[split_col] == split).sum())
            for split in sorted(expected)
        },
    }


def ranking_metrics(
    frame: pd.DataFrame, query_col: str, label_col: str, score_col: str
) -> dict[str, float]:
    reciprocal_ranks = []
    hits = {1: [], 5: [], 10: []}
    for _, group in frame.groupby(query_col, sort=False):
        ordered = group.sort_values(score_col, ascending=False)
        positives = np.flatnonzero(ordered[label_col].to_numpy(dtype=int) == 1)
        reciprocal_ranks.append(0.0 if not len(positives) else 1.0 / (int(positives[0]) + 1))
        for k in hits:
            hits[k].append(float(bool(len(positives)) and positives[0] < k))
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "hits_at_1": float(np.mean(hits[1])),
        "hits_at_5": float(np.mean(hits[5])),
        "hits_at_10": float(np.mean(hits[10])),
        "queries": len(reciprocal_ranks),
    }


def run_pointwise_stacking(
    input_path: Path,
    output_dir: Path,
    split_col: str,
    query_col: str,
    candidate_col: str,
    label_col: str,
    features: list[str],
    seed: int,
) -> Path:
    started = time.time()
    frame = read_frame(input_path)
    missing = [column for column in [split_col, query_col, candidate_col, label_col, *features] if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not set(frame[label_col].dropna().unique()) <= {0, 1}:
        raise ValueError("Labels must be binary.")
    split_report = validate_group_splits(frame, split_col, query_col)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = frame[frame[split_col] == "train"].copy()
    valid = frame[frame[split_col] == "valid"].copy()
    test = frame[frame[split_col] == "test"].copy()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train[features], train[label_col].astype(int))
    for split_frame in (valid, test):
        split_frame["fusion_score"] = model.predict_proba(split_frame[features])[:, 1]
    threshold = select_threshold(valid[label_col].to_numpy(), valid["fusion_score"].to_numpy())
    metrics = {
        "method": "pointwise_score_stacking",
        "not_a_pairwise_or_listwise_ranker": True,
        "split_validation": split_report,
        "features": features,
        "validation": {
            "classification": classification_metrics(
                valid[label_col].to_numpy(), valid["fusion_score"].to_numpy(), threshold
            ),
            "ranking": ranking_metrics(valid, query_col, label_col, "fusion_score"),
        },
        "test": {
            "classification": classification_metrics(
                test[label_col].to_numpy(), test["fusion_score"].to_numpy(), threshold
            ),
            "ranking": ranking_metrics(test, query_col, label_col, "fusion_score"),
        },
        "training_metrics_reported": False,
        "runtime_seconds": time.time() - started,
    }
    valid.to_csv(output_dir / "validation_scored.csv.gz", index=False, compression="gzip")
    test.to_csv(output_dir / "test_scored.csv.gz", index=False, compression="gzip")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    joblib.dump(model, output_dir / "pointwise_stacker.joblib")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": str(input_path.resolve()),
                "split_col": split_col,
                "query_col": query_col,
                "candidate_col": candidate_col,
                "label_col": label_col,
                "features": features,
                "seed": seed,
                "fit_split": "train",
                "selection_split": "valid",
                "final_split": "test",
                "paper_eligible": False,
                "paper_eligibility_blocker": "Must be run from immutable paper inputs under the repeated-seed protocol.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output_dir

