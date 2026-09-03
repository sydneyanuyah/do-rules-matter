from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import load_dataset, validate_splits
from .evaluate import classification_metrics, paired_bootstrap_f1, select_threshold
from .features import mask_fields, serialize, structured_features


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tfidf_scores(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer]:
    train_left, train_right = serialize(train, "left"), serialize(train, "right")
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True
    )
    vectorizer.fit(pd.concat([train_left, train_right], ignore_index=True))

    def score(frame: pd.DataFrame) -> np.ndarray:
        left = vectorizer.transform(serialize(frame, "left"))
        right = vectorizer.transform(serialize(frame, "right"))
        return np.asarray(left.multiply(right).sum(axis=1)).ravel()

    return score(train), score(valid), score(test), vectorizer


def run_dataset(
    project_root: Path,
    config: dict[str, Any],
    dataset_name: str,
    bootstrap_replicates: int | None = None,
) -> Path:
    started = time.time()
    data_root = project_root / config["project"]["data_root"]
    spec = config["datasets"][dataset_name]
    splits = load_dataset(data_root, spec)
    validation = validate_splits(
        splits,
        enforce_offer_disjoint=dataset_name == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
        enforce_record_disjoint=spec["adapter"] == "link_lives",
    )
    seed = int(config["project"]["seed"])
    output = project_root / config["project"]["output_root"] / "exp01_pair_classification" / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    tfidf_train, tfidf_valid, tfidf_test, vectorizer = _tfidf_scores(
        splits["train"], splits["valid"], splits["test"]
    )
    feature_frames: dict[str, pd.DataFrame] = {}
    for split, tfidf in zip(("train", "valid", "test"), (tfidf_train, tfidf_valid, tfidf_test)):
        features = structured_features(splits[split])
        features["lexical_tfidf_score"] = tfidf
        feature_frames[split] = features
        export = pd.concat(
            [splits[split][["pair_id", "label"]].reset_index(drop=True), features.reset_index(drop=True)],
            axis=1,
        )
        export.to_csv(output / f"{split}_features.csv.gz", index=False, compression="gzip")

    y = {split: splits[split]["label"].to_numpy() for split in splits}
    models = {
        "linear": LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=seed, solver="lbfgs"
        ),
        "gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=seed,
        ),
    }
    results: dict[str, Any] = {
        "dataset": dataset_name,
        "status": "pilot_not_paper_claim",
        "warning": "TF-IDF is a CPU pipeline smoke-test proxy, not the frozen-embedding paper condition.",
        "validation": validation,
        "models": {},
    }
    score_cache: dict[str, dict[str, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(feature_frames["train"], y["train"])
        score_cache[name] = {}
        for split in ("valid", "test"):
            score_cache[name][split] = model.predict_proba(feature_frames[split])[:, 1]
        threshold = select_threshold(y["valid"], score_cache[name]["valid"])
        results["models"][name] = {
            "features": list(feature_frames["train"].columns),
            "validation": classification_metrics(y["valid"], score_cache[name]["valid"], threshold),
            "test": classification_metrics(y["test"], score_cache[name]["test"], threshold),
        }
        joblib.dump(model, output / f"{name}.joblib")

    rule_valid = feature_frames["valid"]["rule_score"].to_numpy()
    rule_test = feature_frames["test"]["rule_score"].to_numpy()
    rule_threshold = select_threshold(y["valid"], rule_valid)
    tfidf_threshold = select_threshold(y["valid"], tfidf_valid)
    results["baselines"] = {
        "rules": {
            "validation": classification_metrics(y["valid"], rule_valid, rule_threshold),
            "test": classification_metrics(y["test"], rule_test, rule_threshold),
        },
        "tfidf_proxy": {
            "validation": classification_metrics(y["valid"], tfidf_valid, tfidf_threshold),
            "test": classification_metrics(y["test"], tfidf_test, tfidf_threshold),
        },
    }
    reps = bootstrap_replicates or int(config["protocol"]["bootstrap_replicates"])
    results["paired_bootstrap_gbdt_minus_tfidf"] = paired_bootstrap_f1(
        y["test"],
        score_cache["gbdt"]["test"],
        results["models"]["gbdt"]["validation"]["threshold"],
        tfidf_test,
        tfidf_threshold,
        reps,
        seed,
    )
    results["runtime_seconds"] = time.time() - started
    _json(output / "metrics.json", results)
    joblib.dump(vectorizer, output / "tfidf_vectorizer.joblib")
    _json(
        output / "run_manifest.json",
        {
            "dataset": dataset_name,
            "seed": seed,
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "config_sha256": sha256(project_root / "configs" / "experiment.yaml"),
            "created_unix": time.time(),
            "paper_eligible": False,
            "reason": "CPU smoke test uses TF-IDF proxy and one seed.",
        },
    )
    return output


def run_masking(
    project_root: Path, config: dict[str, Any], dataset_name: str, probabilities: list[float]
) -> Path:
    spec = config["datasets"][dataset_name]
    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    seed = int(config["project"]["seed"])
    output = project_root / config["project"]["output_root"] / "exp07_controlled_field_masking" / dataset_name
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for probability in probabilities:
        masked = mask_fields(splits["test"], probability, seed)
        features = structured_features(masked)
        rows.append(
            {
                "probability": probability,
                "mean_rule_score": float(features["rule_score"].mean()),
                "mean_shared_field_fraction": float(features["shared_field_fraction"].mean()),
                "rows": len(masked),
                "status": "diagnostic_not_paper_claim",
            }
        )
    pd.DataFrame(rows).to_csv(output / "masking_diagnostic.csv", index=False)
    return output
