from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_dataset, validate_splits
from .evaluate import classification_metrics, select_threshold
from .features import serialize


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cross_encoder(
    project_root: Path,
    config: dict[str, Any],
    dataset_name: str,
    seed: int,
    batch_size: int = 32,
    device: str = "cuda",
) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    class PairDataset(Dataset):
        def __init__(self, left: list[str], right: list[str], labels: np.ndarray) -> None:
            self.left = left
            self.right = right
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = tokenizer(
                self.left[index],
                self.right[index],
                truncation=True,
                max_length=max_length,
            )
            item["labels"] = int(self.labels[index])
            return item

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

    spec = config["datasets"][dataset_name]
    if dataset_name not in config["dataset_groups"]["exp01_all"]:
        raise ValueError("The tuned cross-encoder is locked to the five public Exp01 datasets.")
    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    split_validation = validate_splits(
        splits,
        enforce_offer_disjoint=dataset_name == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
    )

    model_spec = config["cross_encoder"]
    model_id = model_spec["id"]
    revision = model_spec["revision"]
    max_length = int(model_spec["max_length"])
    learning_rates = [float(value) for value in model_spec["learning_rates"]]
    max_epochs = int(model_spec["epochs"])
    patience = int(model_spec["early_stopping_patience"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    texts = {
        split: (
            serialize(frame, "left").tolist(),
            serialize(frame, "right").tolist(),
            frame["label"].to_numpy(dtype=np.int64),
        )
        for split, frame in splits.items()
    }
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    loaders = {
        split: DataLoader(
            PairDataset(*texts[split]),
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=collator,
            num_workers=0,
            pin_memory=device.startswith("cuda"),
        )
        for split in splits
    }
    torch_device = torch.device(device)
    positive = float(texts["train"][2].sum())
    negative = float(len(texts["train"][2]) - positive)
    class_weights = torch.tensor([1.0, negative / positive], device=torch_device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    def score(model: Any, split: str) -> np.ndarray:
        model.eval()
        result: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loaders[split]:
                labels = batch.pop("labels")
                batch = {key: value.to(torch_device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=torch.float16,
                    enabled=torch_device.type == "cuda",
                ):
                    logits = model(**batch).logits
                result.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(result)

    output = (
        project_root
        / config["project"]["output_root"]
        / "exp01_cross_encoder"
        / dataset_name
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    started = time.time()

    for learning_rate in learning_rates:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, num_labels=2
        ).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = max_epochs * len(loaders["train"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * 0.1)),
            num_training_steps=total_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=torch_device.type == "cuda")
        trial_best_f1 = -1.0
        trial_best_epoch = 0
        trial_best_threshold = 0.5
        trial_best_scores: np.ndarray | None = None
        trial_best_state: dict[str, Any] | None = None
        stale_epochs = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch in loaders["train"]:
                labels = batch.pop("labels").to(torch_device)
                batch = {key: value.to(torch_device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=torch.float16,
                    enabled=torch_device.type == "cuda",
                ):
                    logits = model(**batch).logits
                    loss = loss_function(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            valid_scores = score(model, "valid")
            threshold = select_threshold(texts["valid"][2], valid_scores)
            valid_metrics = classification_metrics(texts["valid"][2], valid_scores, threshold)
            if valid_metrics["f1"] > trial_best_f1:
                trial_best_f1 = valid_metrics["f1"]
                trial_best_epoch = epoch
                trial_best_threshold = threshold
                trial_best_scores = valid_scores.copy()
                trial_best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break

        assert trial_best_state is not None and trial_best_scores is not None
        trial = {
            "learning_rate": learning_rate,
            "best_epoch": trial_best_epoch,
            "threshold": trial_best_threshold,
            "validation": classification_metrics(
                texts["valid"][2], trial_best_scores, trial_best_threshold
            ),
        }
        trials.append(trial)
        if best_trial is None or trial["validation"]["f1"] > best_trial["validation"]["f1"]:
            best_trial = trial
            best_state = trial_best_state
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert best_trial is not None and best_state is not None
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=2
    ).to(torch_device)
    model.load_state_dict(best_state)
    valid_scores = score(model, "valid")
    test_scores = score(model, "test")
    threshold = float(best_trial["threshold"])
    metrics = {
        "experiment": "exp01_standard_pair_classification",
        "method": "tuned_cross_encoder",
        "dataset": dataset_name,
        "seed": seed,
        "selection": "learning rate, epoch, and threshold selected on validation only",
        "search_budget": {
            "learning_rates": learning_rates,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "batch_size": batch_size,
        },
        "trials": trials,
        "selected_learning_rate": best_trial["learning_rate"],
        "selected_epoch": best_trial["best_epoch"],
        "validation": classification_metrics(texts["valid"][2], valid_scores, threshold),
        "test": classification_metrics(texts["test"][2], test_scores, threshold),
        "split_validation": split_validation,
        "runtime_seconds": time.time() - started,
    }
    _json(output / "metrics.json", metrics)
    np.savez_compressed(
        output / "scores.npz",
        valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(),
        valid_label=texts["valid"][2],
        valid_score=valid_scores.astype(np.float32),
        test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),
        test_label=texts["test"][2],
        test_score=test_scores.astype(np.float32),
    )
    model.save_pretrained(output / "model")
    tokenizer.save_pretrained(output / "model")
    _json(
        output / "run_manifest.json",
        {
            "dataset": dataset_name,
            "method": "tuned_cross_encoder",
            "model_id": model_id,
            "model_revision": revision,
            "seed": seed,
            "config_sha256": _sha256(project_root / "configs" / "experiment.yaml"),
            "fit_split": "train",
            "selection_split": "valid",
            "test_policy": "scored once after selection",
            "paper_eligible": False,
            "paper_eligibility_blocker": "Final multi-dataset audit and result assembly pending.",
        },
    )
    return output
