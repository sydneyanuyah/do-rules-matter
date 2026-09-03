#!/usr/bin/env python3
"""Reproduce official Ditto and Ditto+MixDA on the locked Exp. 1 splits.

The upstream Ditto light data interface and augmentation operators are vendored
from megagonlabs/ditto commit 52985564a93fb11308439516d3e17a033d43ec8f.
This adapter replaces only obsolete Apex/Transformers plumbing with native
PyTorch AMP and pins the Hugging Face revision for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from paper1_hef.data import load_dataset, validate_splits
from paper1_hef.features import serialize


UPSTREAM_COMMIT = "52985564a93fb11308439516d3e17a033d43ec8f"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _lines(frame: Any) -> list[str]:
    left = [
        value.replace("[COL]", "COL").replace("[VAL]", "VAL")
        for value in serialize(frame, "left").tolist()
    ]
    right = [
        value.replace("[COL]", "COL").replace("[VAL]", "VAL")
        for value in serialize(frame, "right").tolist()
    ]
    return [
        f"{a}\t{b}\t{int(label)}"
        for a, b, label in zip(left, right, frame["label"], strict=True)
    ]


class OfficialDittoModel(torch.nn.Module):
    """The official Ditto light RoBERTa+linear architecture."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        device: torch.device,
        alpha_aug: float,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id, revision=revision)
        self.device = device
        self.alpha_aug = alpha_aug
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x1 = x1.to(self.device, non_blocking=True)
        if x2 is not None:
            x2 = x2.to(self.device, non_blocking=True)
            encoded = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            encoded_original = encoded[:batch_size]
            encoded_augmented = encoded[batch_size:]
            weight = np.random.beta(self.alpha_aug, self.alpha_aug)
            encoded = (
                encoded_original * weight
                + encoded_augmented * (1.0 - weight)
            )
        else:
            encoded = self.bert(x1)[0][:, 0, :]
        return self.fc(encoded)


def _threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.0, 1.0, 0.05):
        predictions = (scores > threshold).astype(np.int8)
        value = float(f1_score(labels, predictions, zero_division=0))
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    return best_f1, best_threshold


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (scores > threshold).astype(np.int8)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("plain", "mixda_all"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exp05-fraction", type=float)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load((root / args.config).read_text())
    datasets = list(config["dataset_groups"]["exp01_all"])
    if args.dataset not in datasets:
        raise ValueError(f"{args.dataset} is not registered in exp01_all")
    if args.seed not in {int(seed) for seed in config["protocol"]["seeds"]}:
        raise ValueError(f"{args.seed} is not a locked seed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    spec = config["datasets"][args.dataset]
    splits = load_dataset(root / config["project"]["data_root"], spec)
    split_validation = validate_splits(
        splits,
        enforce_offer_disjoint=args.dataset == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
        enforce_record_disjoint=spec["adapter"] == "link_lives",
    )
    subset_metadata = None
    if args.exp05_fraction is not None:
        from paper1_hef.exp05 import _nested_subsets, _pair_id_hash
        exp05 = config["experiments"]["exp05_label_efficiency"]
        fraction = float(args.exp05_fraction)
        locked = sorted({float(x) for x in exp05["fractions"]})
        if fraction not in locked:
            raise ValueError(f"fraction {fraction} is outside locked Experiment 5 schedule")
        train = splits["train"].reset_index(drop=True)
        _, subsets = _nested_subsets(
            train["left_id"].astype(str).to_numpy(),
            train["label"].to_numpy(dtype=np.int64),
            locked,
            int(exp05["subset_seed"]),
        )
        selected = train.iloc[subsets[fraction]].reset_index(drop=True)
        subset_metadata = {
            "requested_fraction": fraction,
            "selected_pair_count": int(len(selected)),
            "selected_group_count": int(selected["left_id"].astype(str).nunique()),
            "pair_id_sha256": _pair_id_hash(selected["pair_id"].astype(str).to_numpy()),
            "subset_seed": int(exp05["subset_seed"]),
        }
        splits = dict(splits)
        splits["train"] = selected

    upstream = root / "external" / "ditto_official"
    import sys

    sys.path.insert(0, str(upstream))
    from ditto_light import dataset as official_dataset  # type: ignore

    model_spec = config["cross_encoder"]
    model_id = str(model_spec["id"])
    revision = str(model_spec["revision"])
    max_length = int(model_spec["max_length"])
    epochs = int(model_spec["epochs"])
    learning_rate = 3e-5
    alpha_aug = 0.8
    da = "all" if args.variant == "mixda_all" else None

    # The upstream class does not accept a revision. This preserves its data
    # path while ensuring the tokenizer matches the pinned model checkpoint.
    official_dataset.get_tokenizer = lambda _lm: AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
    )

    prepared_suffix = "full" if args.exp05_fraction is None else f"fraction_{int(round(float(args.exp05_fraction) * 100)):03d}_seed_{args.seed}"
    prepared = root / "artifacts" / "exp01_ditto_official" / "prepared" / args.dataset / prepared_suffix
    prepared.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "valid", "test"):
        path = prepared / f"{split_name}.txt"
        path.write_text("\n".join(_lines(splits[split_name])) + "\n")

    trainset = official_dataset.DittoDataset(
        prepared / "train.txt",
        lm=model_id,
        max_len=max_length,
        da=da,
    )
    validset = official_dataset.DittoDataset(
        prepared / "valid.txt",
        lm=model_id,
        max_len=max_length,
    )
    testset = official_dataset.DittoDataset(
        prepared / "test.txt",
        lm=model_id,
        max_len=max_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=trainset.pad,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=args.batch_size * 8,
        shuffle=False,
        num_workers=0,
        collate_fn=validset.pad,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size * 8,
        shuffle=False,
        num_workers=0,
        collate_fn=testset.pad,
        pin_memory=True,
    )

    device = torch.device(args.device)

    def build_model() -> OfficialDittoModel:
        return OfficialDittoModel(
            model_id,
            revision,
            device,
            alpha_aug,
        ).to(device)

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def score(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        with torch.inference_mode():
            for x, y in loader:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(x)
                probabilities.append(
                    torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                )
                labels.append(y.numpy())
        return np.concatenate(labels), np.concatenate(probabilities)

    started = time.time()
    best_validation_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if len(batch) == 2:
                x, y = batch
                x_aug = None
            else:
                x, x_aug, y = batch
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(x, x_aug)
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    y.to(device, non_blocking=True),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        valid_labels, valid_scores = score(valid_loader)
        valid_f1, threshold = _threshold(valid_labels, valid_scores)
        history.append(
            {
                "epoch": epoch,
                "mean_train_loss": float(np.mean(losses)),
                "validation_f1": valid_f1,
                "threshold": threshold,
            }
        )
        if valid_f1 > best_validation_f1:
            best_validation_f1 = valid_f1
            best_epoch = epoch
            best_threshold = threshold
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No valid Ditto checkpoint was selected")
    model.load_state_dict(best_state)
    valid_labels, valid_scores = score(valid_loader)
    test_labels, test_scores = score(test_loader)

    if args.exp05_fraction is None:
        output = root / config["project"]["output_root"] / "exp01_ditto_official" / args.variant / args.dataset / f"seed_{args.seed}"
        experiment_name = "exp01_official_ditto_reproduction"
    else:
        fraction_key = f"{int(round(float(args.exp05_fraction) * 100)):03d}"
        family = "official_ditto_mixda" if args.variant == "mixda_all" else "official_ditto"
        output = root / config["project"]["output_root"] / "exp05_expanded" / "v1" / family / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
        experiment_name = "exp05_label_efficiency"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "model_id": model_id,
            "revision": revision,
            "variant": args.variant,
            "upstream_commit": UPSTREAM_COMMIT,
            "selected_epoch": best_epoch,
            "threshold": best_threshold,
        },
        output / "model.pt",
    )
    np.savez_compressed(
        output / "scores.npz",
        valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(),
        valid_label=valid_labels.astype(np.int8),
        valid_score=valid_scores.astype(np.float32),
        test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),
        test_label=test_labels.astype(np.int8),
        test_score=test_scores.astype(np.float32),
    )
    result = {
        "experiment": experiment_name,
        "method": (
            "official_ditto_mixda_all"
            if args.variant == "mixda_all"
            else "official_ditto_plain"
        ),
        "official_ditto_reproduction": True,
        "upstream_repository": "https://github.com/megagonlabs/ditto",
        "upstream_commit": UPSTREAM_COMMIT,
        "compatibility_modifications": [
            "native torch.amp replaces obsolete NVIDIA Apex",
            "torch.optim.AdamW replaces removed transformers.AdamW alias",
            "Hugging Face model/tokenizer revision is pinned",
            "malformed MixDA samples that remove the pair separator fall back "
            "to the original pair; valid official augmentations are unchanged",
            "test set is scored once after validation checkpoint selection",
        ],
        "dataset": args.dataset,
        "seed": args.seed,
        "training_subset": subset_metadata or {"requested_fraction": 1.0},
        "variant": args.variant,
        "data_augmentation": da,
        "model_id": model_id,
        "revision": revision,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "threshold": best_threshold,
        "validation": _metrics(valid_labels, valid_scores, best_threshold),
        "test": _metrics(test_labels, test_scores, best_threshold),
        "history": history,
        "split_validation": split_validation,
        "runtime_seconds": time.time() - started,
    }
    _json(output / "metrics.json", result)
    _json(
        output / "run_manifest.json",
        {
            "paper_eligible": True,
            "official_ditto_reproduction": True,
            "upstream_commit": UPSTREAM_COMMIT,
            "dataset": args.dataset,
            "seed": args.seed,
            "variant": args.variant,
            "checkpoint": str((output / "model.pt").relative_to(root)),
        },
    )


if __name__ == "__main__":
    main()
