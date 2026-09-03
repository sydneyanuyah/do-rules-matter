from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.r_[0.0, scores, 1.0])

    # Evaluate every exact candidate in O(n log n), rather than repeatedly
    # rescanning the full validation set. np.argmax preserves the original
    # lowest-threshold tie break because candidates are sorted ascending.
    order = np.argsort(scores, kind="stable")
    sorted_y = y_true[order]
    sorted_scores = scores[order]
    positives_before = np.r_[0, np.cumsum(sorted_y)]
    starts = np.searchsorted(sorted_scores, candidates, side="left")
    total_positive = int(sorted_y.sum())
    true_positive = total_positive - positives_before[starts]
    predicted_positive = len(scores) - starts
    denominator = predicted_positive + total_positive
    values = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(candidates, dtype=float),
        where=denominator != 0,
    )
    return float(candidates[int(np.argmax(values))])


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (scores >= low) & (scores < high if high < 1 else scores <= high)
        if mask.any():
            total += mask.mean() * abs(y_true[mask].mean() - scores[mask].mean())
    return float(total)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    pred = scores >= threshold
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "ece": expected_calibration_error(y_true, scores),
    }


def paired_bootstrap_f1(
    y_true: np.ndarray,
    score_a: np.ndarray,
    threshold_a: float,
    score_b: np.ndarray,
    threshold_b: float,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    differences = np.empty(replicates)
    for i in range(replicates):
        idx = rng.integers(0, n, n)
        differences[i] = f1_score(y_true[idx], score_a[idx] >= threshold_a, zero_division=0) - f1_score(
            y_true[idx], score_b[idx] >= threshold_b, zero_division=0
        )
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "two_sided_p": float(2 * min((differences <= 0).mean(), (differences >= 0).mean())),
    }
