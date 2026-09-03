#!/usr/bin/env python3
"""Run pinned AnyMatch leave-one-dataset-out on the Paper-1 public suite."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(np.int8)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-batch-size", type=int, default=40)
    parser.add_argument("--valid-batch-size", type=int, default=80)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load((root / args.config).read_text())
    all_datasets = list(config["dataset_groups"]["exp01_all"])
    if args.target not in all_datasets:
        raise ValueError(f"{args.target} is not registered for Experiment 1")
    if args.seed not in {int(value) for value in config["protocol"]["seeds"]}:
        raise ValueError(f"{args.seed} is not a locked protocol seed")

    output = (
        root
        / config["project"]["output_root"]
        / "exp01_anymatch_official"
        / args.target
        / f"seed_{args.seed}"
    )
    if (output / "metrics.json").exists():
        print(f"Skipping completed AnyMatch run: {args.target} seed={args.seed}")
        return
    output.mkdir(parents=True, exist_ok=True)

    upstream = root / "external/anymatch_official"
    sys.path.insert(0, str(upstream))
    from data import GPTDataset  # type: ignore
    from model import load_model  # type: ignore
    from utils.data_utils import (  # type: ignore
        read_multi_attr_data,
        read_multi_row_data,
        read_single_row_data,
    )
    from utils.train_eval import train  # type: ignore

    prepared = upstream / "data" / "prepared"
    # The seen and unseen WDC conditions come from the same benchmark family.
    # Allowing one WDC condition to train the model evaluated on the other would
    # no longer be a defensible zero-shot leave-one-dataset-out comparison.
    wdc_family = {"wdc_80_medium_seen", "wdc_80_medium_unseen"}
    excluded_sources = (
        wdc_family if args.target in wdc_family else {args.target}
    )
    source_dirs = [
        str(prepared / name)
        for name in all_datasets
        if name not in excluded_sources
    ]
    for source in source_dirs:
        automl = upstream / "data" / "automl" / Path(source).name / "train_preds.csv"
        if not automl.exists():
            raise FileNotFoundError(f"Missing published AnyMatch AutoML input: {automl}")

    started = time.time()
    model, tokenizer = load_model("gpt2")
    train_attr, _, _ = read_multi_attr_data(source_dirs, "mode1")
    train_row, valid_row, _ = read_multi_row_data(
        source_dirs, "mode1", "automl_filter"
    )
    train_frame = (
        pd.concat([train_attr, train_row], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    train_dataset = GPTDataset(tokenizer, train_frame, max_len=350)
    valid_dataset = GPTDataset(tokenizer, valid_row, max_len=350)
    best_model = train(
        tokenizer,
        model,
        train_dataset,
        valid_dataset,
        epochs=50,
        lr=2e-5,
        seed=args.seed,
        patient=True,
        save_model=False,
        save_freq=50,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        save_model_path="",
        save_result_prefix=str(output / "training"),
        patience=6,
        patience_start=20,
        base_model="gpt2",
    )
    best_model = copy.deepcopy(best_model)
    best_model.save_pretrained(output / "model")
    tokenizer.save_pretrained(output / "model")

    _, _, target_test = read_single_row_data(
        str(prepared / args.target), "mode1", print_info=False
    )
    target_dataset = GPTDataset(tokenizer, target_test, max_len=350)
    loader = DataLoader(
        target_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=target_dataset.collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device).eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            target = batch["labels"].numpy()
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = best_model(**inputs).logits
            scores.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            labels.append(target)
    score = np.concatenate(scores)
    label = np.concatenate(labels).astype(np.int8)
    pair_ids = pd.read_csv(prepared / args.target / "test_pair_ids.csv")[
        "pair_id"
    ].astype(str).to_numpy()
    if not (len(pair_ids) == len(label) == len(target_dataset)):
        raise ValueError("AnyMatch filtered target examples; paper result is invalid")
    np.savez_compressed(
        output / "scores.npz",
        test_pair_id=pair_ids,
        test_label=label,
        test_score=score.astype(np.float32),
    )
    result = {
        "experiment": "exp01_anymatch_official_leave_one_dataset_out",
        "method": "AnyMatch",
        "implementation": "official pinned source with external Paper-1 data adapter",
        "upstream_commit": "4d49549233f75719972164c54ebaa13286dc0cdb",
        "target_dataset": args.target,
        "source_datasets": [Path(value).name for value in source_dirs],
        "excluded_source_datasets": sorted(excluded_sources),
        "source_exclusion_policy": (
            "exclude entire WDC benchmark family when either WDC condition is "
            "the target; otherwise exclude target dataset only"
        ),
        "seed": args.seed,
        "base_model": "gpt2",
        "train_batch_size": args.train_batch_size,
        "valid_batch_size": args.valid_batch_size,
        "serialization_mode": "mode1",
        "train_data": "attr+row",
        "row_sample_func": "automl_filter",
        "target_label_use": "none",
        "decision_rule": "argmax, equivalent to class-1 probability >= 0.5",
        "test": _metrics(label, score),
        "train_examples_after_filtering": len(train_dataset),
        "source_validation_examples": len(valid_dataset),
        "target_test_examples": len(target_dataset),
        "runtime_seconds": time.time() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
