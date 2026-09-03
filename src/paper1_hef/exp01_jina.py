from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_dataset, validate_splits
from .evaluate import classification_metrics, select_threshold
from .features import serialize


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_exp01_jina(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    device: str = "cuda",
    batch_size: int = 0,
    target_memory_gib: float = 20.0,
    max_batch_size: int = 4096,
) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    spec = config["datasets"][dataset]
    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    validation = validate_splits(
        splits,
        enforce_offer_disjoint=dataset == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
    )
    jina = config["exp02_jina_cross_encoder"]
    model_id = str(jina["id"])
    revision = str(jina["revision"])
    max_length = int(jina["max_length"])
    torch_device = torch.device(device)
    started = time.time()

    split_names = ("valid", "test")
    split_sizes = {name: len(splits[name]) for name in split_names}
    left_text: list[str] = []
    right_text: list[str] = []
    labels: list[np.ndarray] = []
    for name in split_names:
        left_text.extend(serialize(splits[name], "left").tolist())
        right_text.extend(serialize(splits[name], "right").tolist())
        labels.append(splits[name]["label"].to_numpy(dtype=np.int64))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,
    ).to(torch_device)
    model.eval()
    lengths = np.asarray(
        [len(left) + len(right) for left, right in zip(left_text, right_text)],
        dtype=np.int64,
    )
    order = np.argsort(-lengths, kind="stable")
    gib = 1024**3
    target_bytes = int(target_memory_gib * gib)
    total_memory = (
        int(torch.cuda.get_device_properties(torch_device).total_memory)
        if torch_device.type == "cuda"
        else 0
    )
    safe_limit = int(total_memory * 0.92) if total_memory else 0
    calibration: list[dict[str, Any]] = []

    def forward(indices: np.ndarray) -> tuple[np.ndarray, int]:
        encoded = tokenizer(
            [left_text[int(index)] for index in indices],
            [right_text[int(index)] for index in indices],
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded = {key: value.to(torch_device, non_blocking=True) for key, value in encoded.items()}
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                logits = model(**encoded).logits
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
            peak = int(torch.cuda.max_memory_allocated(torch_device))
        else:
            peak = 0
        # Jina returns a single relevance logit. Convert it to a probability
        # before validation threshold selection and reporting. Previously the
        # metrics helper clipped negative logits to zero while retaining a raw
        # negative threshold, which made every pair positive.
        values = (
            torch.sigmoid(logits.float().reshape(-1))
            if logits.ndim == 1 or logits.shape[-1] == 1
            else torch.softmax(logits.float(), dim=-1)[:, 1]
        )
        return values.cpu().numpy(), peak

    if batch_size > 0:
        candidates = [min(batch_size, len(order))]
    else:
        candidates = []
        candidate = min(32, len(order))
        while candidate:
            candidates.append(candidate)
            next_candidate = min(
                max_batch_size,
                len(order),
                max(candidate + 8, int(candidate * 1.5) // 8 * 8),
            )
            if next_candidate == candidate:
                break
            candidate = next_candidate

    selected_batch = 1
    for candidate in candidates:
        try:
            _, peak = forward(order[:candidate])
            calibration.append(
                {
                    "batch_size": int(candidate),
                    "peak_allocated_gib": peak / gib,
                    "status": "ok",
                }
            )
            if not safe_limit or peak <= safe_limit:
                selected_batch = candidate
            if target_bytes and peak >= target_bytes:
                break
        except torch.OutOfMemoryError:
            calibration.append({"batch_size": int(candidate), "status": "out_of_memory"})
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            break

    scores = np.empty(len(order), dtype=np.float32)
    cursor = 0
    current_batch = selected_batch
    maximum_batch_used = selected_batch
    while cursor < len(order):
        count = min(current_batch, len(order) - cursor)
        indices = order[cursor : cursor + count]
        try:
            values, peak = forward(indices)
        except torch.OutOfMemoryError:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            if current_batch <= 1:
                raise
            current_batch = max(1, current_batch // 2)
            continue
        scores[indices] = values
        cursor += count
        maximum_batch_used = max(maximum_batch_used, count)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "rows_complete": cursor,
                    "rows_total": len(order),
                    "batch_size": count,
                    "peak_allocated_gib": peak / gib,
                }
            ),
            flush=True,
        )
        if target_bytes and peak > 0 and peak < int(target_bytes * 0.85):
            scale = min(2.0, max(1.05, target_bytes / peak * 0.92))
            proposed = max(current_batch + 8, int(current_batch * scale) // 8 * 8)
            current_batch = min(max_batch_size, proposed)

    valid_count = split_sizes["valid"]
    valid_scores = scores[:valid_count]
    test_scores = scores[valid_count:]
    valid_labels, test_labels = labels
    threshold = select_threshold(valid_labels, valid_scores)
    test_predictions = test_scores >= threshold
    if not np.isfinite(test_scores).all() or float(np.std(test_scores)) <= 1e-8:
        raise ValueError(f"{dataset}: Jina test scores are degenerate")
    if bool(test_predictions.all()) or bool((~test_predictions).all()):
        raise ValueError(
            f"{dataset}: validation-selected Jina threshold collapses test predictions "
            f"(positive_rate={float(test_predictions.mean()):.8f})"
        )
    output = project_root / "artifacts" / "exp01_jina" / dataset
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "metrics.json",
        {
            "experiment": "exp01_standard_pair_classification",
            "method": "jina_cross_encoder",
            "dataset": dataset,
            "status": "complete",
            "selection": "classification threshold selected on validation only",
            "task_fine_tuning": "none",
            "model_id": model_id,
            "revision": revision,
            "max_length": max_length,
            "initial_batch_size": selected_batch,
            "maximum_batch_size_used": maximum_batch_used,
            "target_memory_gib": target_memory_gib,
            "calibration": calibration,
            "validation": classification_metrics(valid_labels, valid_scores, threshold),
            "test": classification_metrics(test_labels, test_scores, threshold),
            "threshold": threshold,
            "split_validation": validation,
            "runtime_seconds": time.time() - started,
        },
    )
    np.savez_compressed(
        output / "scores.npz",
        valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(),
        valid_label=valid_labels,
        valid_score=valid_scores,
        test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),
        test_label=test_labels,
        test_score=test_scores,
    )
    return output
