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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _seeded_int(*values: object) -> int:
    digest = hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()
    return int(digest[:16], 16)


def ditto_text(frame: Any, side: str) -> list[str]:
    """Return Ditto-compatible COL/VAL serialization without changing fields."""
    return [
        value.replace("[COL]", "COL").replace("[VAL]", "VAL")
        for value in serialize(frame, side).tolist()
    ]


def _labels(tokens: list[str]) -> list[str]:
    return ["HD" if token in {"COL", "VAL"} else "O" for token in tokens]


def _span(tokens: list[str], labels: list[str], length: int, rng: random.Random) -> tuple[int, int]:
    candidates = [
        (index, index + length - 1)
        for index in range(len(tokens) - length + 1)
        if all(label == "O" for label in labels[index : index + length])
    ]
    return rng.choice(candidates) if candidates else (-1, -1)


def _attribute_ranges(tokens: list[str]) -> list[tuple[int, int]]:
    starts = [index for index, token in enumerate(tokens) if token == "COL"]
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(tokens))
        for index, start in enumerate(starts)
    ]


def _apply_operator(
    tokens: list[str], labels: list[str], operator: str, rng: random.Random
) -> tuple[list[str], list[str]]:
    if operator == "del":
        start, end = _span(tokens, labels, rng.randint(1, 2), rng)
        if start >= 0:
            return tokens[:start] + tokens[end + 1 :], labels[:start] + labels[end + 1 :]
    elif operator == "swap":
        start, end = _span(tokens, labels, rng.randint(2, 4), rng)
        if start >= 0:
            order = list(range(start, end + 1))
            rng.shuffle(order)
            return (
                tokens[:start] + [tokens[index] for index in order] + tokens[end + 1 :],
                labels[:start] + ["O"] * len(order) + labels[end + 1 :],
            )
    elif operator == "drop_col":
        ranges = [
            value
            for value in _attribute_ranges(tokens)
            if value[1] - value[0] <= 10
        ]
        if ranges:
            start, end = rng.choice(ranges)
            return tokens[:start] + tokens[end:], labels[:start] + labels[end:]
    elif operator == "append_col":
        ranges = _attribute_ranges(tokens)
        if len(ranges) >= 2:
            source, target = rng.sample(range(len(ranges)), 2)
            source_start, source_end = ranges[source]
            target_start, target_end = ranges[target]
            chunk = tokens[source_start:source_end]
            try:
                value_start = chunk.index("VAL") + 1
            except ValueError:
                value_start = 2
            appended = chunk[value_start:]
            kept_tokens = tokens[:source_start] + tokens[source_end:]
            kept_labels = labels[:source_start] + labels[source_end:]
            removed = source_end - source_start
            if source_start < target_end:
                target_end -= removed
            return (
                kept_tokens[:target_end] + appended + kept_tokens[target_end:],
                kept_labels[:target_end] + ["O"] * len(appended) + kept_labels[target_end:],
            )
    return tokens, labels


def mixda_augment(left: str, right: str, seed: int) -> tuple[str, str]:
    """Apply Ditto's all-operator MixDA family deterministically per example."""
    rng = random.Random(seed)
    if rng.randint(0, 1) == 0:
        left, right = right, left
    combined = f"{left} [SEP] {right}"
    tokens = combined.split()
    labels = _labels(tokens)
    for operator in rng.choices(("del", "swap", "drop_col", "append_col"), k=3):
        tokens, labels = _apply_operator(tokens, labels, operator, rng)
    try:
        separator = tokens.index("[SEP]")
    except ValueError:
        return left, right
    return " ".join(tokens[:separator]), " ".join(tokens[separator + 1 :])


def _model_factory(model_id: str, revision: str, device: str, dropout: float = 0.1) -> Any:
    import torch
    from transformers import AutoModel

    class DittoStyleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_id, revision=revision)
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 2)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            batch_size: int | None = None,
            mix_lambda: float | None = None,
        ) -> torch.Tensor:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0]
            if batch_size is not None:
                original = hidden[:batch_size]
                augmented = hidden[batch_size:]
                weight = 0.5 if mix_lambda is None else float(mix_lambda)
                hidden = original * weight + augmented * (1.0 - weight)
            return self.classifier(self.dropout(hidden))

    return DittoStyleModel().to(device)


def run_ditto_style(
    project_root: Path,
    config: dict[str, Any],
    dataset_name: str,
    seed: int,
    batch_size: int = 64,
    device: str = "cuda",
    train_indices: np.ndarray | None = None,
    output: Path | None = None,
    experiment: str = "exp01_ditto_style_mixda",
    subset_metadata: dict[str, Any] | None = None,
) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    spec = config["datasets"][dataset_name]
    if dataset_name not in config["dataset_groups"]["exp01_all"]:
        raise ValueError(f"{dataset_name} is not registered for Experiment 1")
    splits = load_dataset(project_root / config["project"]["data_root"], spec)
    split_validation = validate_splits(
        splits,
        enforce_offer_disjoint=dataset_name == "wdc_80_medium_unseen",
        report_offer_overlap=spec["adapter"] == "wdc",
        enforce_record_disjoint=spec["adapter"] == "link_lives",
    )
    frames = dict(splits)
    if train_indices is not None:
        frames["train"] = splits["train"].iloc[train_indices].reset_index(drop=True)

    texts = {
        split: (
            ditto_text(frame, "left"),
            ditto_text(frame, "right"),
            frame["label"].to_numpy(dtype=np.int64),
            frame["pair_id"].astype(str).to_numpy(),
        )
        for split, frame in frames.items()
    }
    model_spec = config["cross_encoder"]
    model_id = str(model_spec["id"])
    revision = str(model_spec["revision"])
    max_length = int(model_spec["max_length"])
    learning_rates = [float(value) for value in model_spec["learning_rates"]]
    max_epochs = int(model_spec["epochs"])
    patience = int(model_spec["early_stopping_patience"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    target = torch.device(device)

    class PairDataset(Dataset):
        def __init__(self, split: str) -> None:
            self.split = split
            self.left, self.right, self.labels, self.pair_ids = texts[split]
            self.epoch = 0

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> tuple[str, str, int, str, int]:
            return (
                self.left[index],
                self.right[index],
                int(self.labels[index]),
                str(self.pair_ids[index]),
                index,
            )

    datasets = {split: PairDataset(split) for split in frames}

    def collate_train(batch: list[tuple[str, str, int, str, int]]) -> dict[str, Any]:
        original_left = [item[0] for item in batch]
        original_right = [item[1] for item in batch]
        augmented = [
            mixda_augment(
                item[0],
                item[1],
                _seeded_int(seed, datasets["train"].epoch, item[3], item[4]),
            )
            for item in batch
        ]
        encoded = tokenizer(
            original_left + [item[0] for item in augmented],
            original_right + [item[1] for item in augmented],
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([item[2] for item in batch], dtype=torch.long),
        }

    def collate_eval(batch: list[tuple[str, str, int, str, int]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([item[2] for item in batch], dtype=torch.long),
        }

    eval_loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size * 4,
            shuffle=False,
            collate_fn=collate_eval,
            num_workers=0,
            pin_memory=target.type == "cuda",
        )
        for split in ("valid", "test")
    }

    def score(model: Any, split: str) -> np.ndarray:
        model.eval()
        result: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in eval_loaders[split]:
                inputs = {
                    "input_ids": batch["input_ids"].to(target, non_blocking=True),
                    "attention_mask": batch["attention_mask"].to(
                        target, non_blocking=True
                    ),
                }
                with torch.autocast(
                    device_type=target.type,
                    dtype=torch.float16,
                    enabled=target.type == "cuda",
                ):
                    logits = model(**inputs)
                result.append(
                    torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                )
        return np.concatenate(result)

    if output is None:
        output = (
            project_root
            / config["project"]["output_root"]
            / "exp01_ditto_style"
            / dataset_name
            / f"seed_{seed}"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    trials: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    for trial_index, learning_rate in enumerate(learning_rates):
        trial_seed = _seeded_int(seed, trial_index, learning_rate) % (2**31)
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)
        generator = torch.Generator().manual_seed(trial_seed)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_train,
            num_workers=0,
            pin_memory=target.type == "cuda",
        )
        beta_rng = np.random.default_rng(trial_seed)
        model = _model_factory(model_id, revision, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = max_epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * 0.1)),
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler(
            target.type, enabled=target.type == "cuda"
        )
        trial_best_f1 = -1.0
        trial_best_epoch = 0
        trial_best_threshold = 0.5
        trial_best_state: dict[str, Any] | None = None
        stale = 0
        epoch_history: list[dict[str, Any]] = []

        for epoch in range(1, max_epochs + 1):
            datasets["train"].epoch = epoch
            model.train()
            losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                labels = batch["labels"].to(target, non_blocking=True)
                original_batch = len(labels)
                mix_lambda = float(beta_rng.beta(0.8, 0.8))
                with torch.autocast(
                    device_type=target.type,
                    dtype=torch.float16,
                    enabled=target.type == "cuda",
                ):
                    logits = model(
                        batch["input_ids"].to(target, non_blocking=True),
                        batch["attention_mask"].to(target, non_blocking=True),
                        batch_size=original_batch,
                        mix_lambda=mix_lambda,
                    )
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))

            valid_score = score(model, "valid")
            threshold = select_threshold(texts["valid"][2], valid_score)
            valid_metrics = classification_metrics(
                texts["valid"][2], valid_score, threshold
            )
            epoch_history.append(
                {
                    "epoch": epoch,
                    "mean_train_loss": float(np.mean(losses)),
                    "validation": valid_metrics,
                    "threshold": threshold,
                }
            )
            if valid_metrics["f1"] > trial_best_f1:
                trial_best_f1 = float(valid_metrics["f1"])
                trial_best_epoch = epoch
                trial_best_threshold = float(threshold)
                trial_best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        assert trial_best_state is not None
        trial = {
            "learning_rate": learning_rate,
            "trial_seed": trial_seed,
            "best_epoch": trial_best_epoch,
            "threshold": trial_best_threshold,
            "validation_f1": trial_best_f1,
            "epoch_history": epoch_history,
        }
        trials.append(trial)
        if best_trial is None or trial_best_f1 > float(best_trial["validation_f1"]):
            best_trial = trial
            best_state = trial_best_state
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert best_trial is not None and best_state is not None
    model = _model_factory(model_id, revision, device)
    model.load_state_dict(best_state)
    valid_score = score(model, "valid")
    test_score = score(model, "test")
    threshold = float(best_trial["threshold"])
    checkpoint = output / "model" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_id": model_id,
            "revision": revision,
            "max_length": max_length,
            "augmentation": "MixDA all: del, swap, drop_col, append_col",
            "alpha_aug": 0.8,
        },
        checkpoint,
    )
    tokenizer.save_pretrained(output / "model" / "tokenizer")
    np.savez_compressed(
        output / "scores.npz",
        valid_pair_id=texts["valid"][3],
        valid_label=texts["valid"][2].astype(np.int8),
        valid_score=valid_score.astype(np.float32),
        test_pair_id=texts["test"][3],
        test_label=texts["test"][2].astype(np.int8),
        test_score=test_score.astype(np.float32),
    )
    metrics: dict[str, Any] = {
        "experiment": experiment,
        "method": "ditto_style_roberta_mixda",
        "label": "Ditto-style RoBERTa + MixDA",
        "official_ditto_claim": False,
        "dataset": dataset_name,
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "serialization": "Ditto COL/VAL field serialization",
        "augmentation": {
            "family": "MixDA",
            "operator": "all",
            "operators": ["del", "swap", "drop_col", "append_col"],
            "operators_per_example": 3,
            "record_order_flip_probability": 0.5,
            "representation_mix_beta_alpha": 0.8,
        },
        "selection": "learning rate, epoch, and threshold selected on validation only",
        "selected_learning_rate": best_trial["learning_rate"],
        "selected_epoch": best_trial["best_epoch"],
        "threshold": threshold,
        "validation": classification_metrics(
            texts["valid"][2], valid_score, threshold
        ),
        "test": classification_metrics(texts["test"][2], test_score, threshold),
        "trials": trials,
        "split_validation": split_validation,
        "runtime_seconds": time.time() - started,
    }
    if subset_metadata is not None:
        metrics["training_subset"] = subset_metadata
    _write_json(output / "metrics.json", metrics)
    _write_json(
        output / "run_manifest.json",
        {
            "method": "ditto_style_roberta_mixda",
            "official_ditto_claim": False,
            "dataset": dataset_name,
            "seed": seed,
            "model_id": model_id,
            "revision": revision,
            "train_rows": int(len(texts["train"][2])),
            "valid_rows": int(len(texts["valid"][2])),
            "test_rows": int(len(texts["test"][2])),
            "batch_size": batch_size,
            "device": device,
            "checkpoint": str(checkpoint.relative_to(project_root)),
            "paper_eligible": True,
        },
    )
    return output


def run_exp05_ditto_style(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    fraction: float,
    seed: int,
    batch_size: int = 64,
    device: str = "cuda",
) -> Path:
    from .exp05 import _nested_subsets, _pair_id_hash

    if dataset not in config["dataset_groups"]["exp01_all"]:
        raise ValueError("Experiment 5 is locked to the registered public Exp1 datasets")
    spec = config["experiments"]["exp05_label_efficiency"]
    fractions = sorted({float(value) for value in spec["fractions"]})
    if fraction not in fractions:
        raise ValueError(f"Fraction {fraction} is not in the locked schedule")
    if seed not in {int(value) for value in config["protocol"]["seeds"]}:
        raise ValueError(f"Seed {seed} is not in the locked protocol")
    splits = load_dataset(
        project_root / config["project"]["data_root"],
        config["datasets"][dataset],
    )
    labels = splits["train"]["label"].to_numpy()
    left_ids = splits["train"]["left_id"].astype(str).to_numpy()
    _, subsets = _nested_subsets(
        left_ids, labels, fractions, int(spec["subset_seed"])
    )
    indices = subsets[fraction]
    pair_ids = splits["train"].iloc[indices]["pair_id"].astype(str).to_numpy()
    subset_hash = _pair_id_hash(pair_ids)
    key = f"{int(round(fraction * 100)):03d}"
    manifest_path = (
        project_root
        / config["project"]["output_root"]
        / "exp05"
        / dataset
        / "subset_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["fractions"][key]
    if subset_hash != expected["pair_id_sha256"]:
        raise ValueError(f"{dataset}/{key}: Ditto-style subset hash mismatch")
    output = (
        project_root
        / config["project"]["output_root"]
        / "exp05_ditto_style"
        / dataset
        / f"fraction_{key}"
        / f"seed_{seed}"
    )
    metadata = {
        "requested_fraction": fraction,
        "fraction_key": key,
        "selected_pair_count": int(len(indices)),
        "pair_id_sha256": subset_hash,
        "reference_manifest": str(manifest_path.relative_to(project_root)),
        "subset_policy": spec["subset_policy"],
        "subset_seed": int(spec["subset_seed"]),
    }
    return run_ditto_style(
        project_root,
        config,
        dataset,
        seed,
        batch_size,
        device,
        train_indices=indices,
        output=output,
        experiment="exp05_label_efficiency_ditto_style_mixda",
        subset_metadata=metadata,
    )


def score_exp02_ditto_style(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    batch_size: int = 512,
    device: str = "cuda",
) -> Path:
    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    from .exp02 import _evaluate, _fields_for_frame, _source

    pool_dir = project_root / "artifacts" / "exp02" / "candidate_pools" / dataset
    ranking_dir = project_root / "artifacts" / "exp02" / "ranking" / dataset
    pool = pd.read_csv(pool_dir / "top100.csv.gz")
    pool["query_id"] = pool["query_id"].astype(str)
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    source = _source(project_root, config, dataset)
    queries = source["queries"]
    candidates = source["candidates"]
    query_fields = [
        column for column in _fields_for_frame(queries) if column in queries
    ]
    candidate_fields = [
        column for column in _fields_for_frame(candidates) if column in candidates
    ]
    left = queries[["id", *query_fields]].rename(
        columns={
            "id": "query_id",
            **{field: f"left_{field}" for field in query_fields},
        }
    )
    right = candidates[["id", *candidate_fields]].rename(
        columns={
            "id": "candidate_id",
            **{field: f"right_{field}" for field in candidate_fields},
        }
    )
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    test_mask = paired["split"].eq("test")
    test = pool.loc[test_mask].copy()
    left_text = ditto_text(paired.loc[test_mask], "left")
    right_text = ditto_text(paired.loc[test_mask], "right")
    target = torch.device(device)
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    sizes = [
        int(value)
        for value in config["experiments"]["exp02_candidate_ranking"]["k"]
    ]
    per_seed: list[dict[str, Any]] = []
    all_scores: list[np.ndarray] = []
    started = time.time()

    for seed in seeds:
        model_dir = (
            project_root
            / "artifacts"
            / "exp01_ditto_style"
            / dataset
            / f"seed_{seed}"
            / "model"
        )
        checkpoint = torch.load(
            model_dir / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir / "tokenizer", local_files_only=True
        )
        model = _model_factory(
            str(checkpoint["model_id"]), str(checkpoint["revision"]), device
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scores: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(test), batch_size):
                stop = min(len(test), start + batch_size)
                encoded = tokenizer(
                    left_text[start:stop],
                    right_text[start:stop],
                    padding=True,
                    truncation=True,
                    max_length=int(checkpoint["max_length"]),
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                with torch.autocast(
                    device_type=target.type,
                    dtype=torch.float16,
                    enabled=target.type == "cuda",
                ):
                    logits = model(
                        encoded["input_ids"].to(target, non_blocking=True),
                        encoded["attention_mask"].to(target, non_blocking=True),
                    )
                scores.append(
                    torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                )
        seed_scores = np.concatenate(scores)
        all_scores.append(seed_scores)
        per_seed.append(
            {
                "seed": seed,
                "metrics": _evaluate(
                    test.assign(_score=seed_scores), "_score", sizes
                ),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_score = np.mean(np.vstack(all_scores), axis=0)
    score_frame = test[
        ["split", "query_id", "candidate_id", "retrieval_rank", "label"]
    ].copy()
    score_frame["ditto_style_mixda_score"] = mean_score
    score_frame.to_csv(
        ranking_dir / "ditto_style_mixda_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics_path = ranking_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["methods"]["ditto_style_roberta_mixda"] = {
        "label": "Ditto-style RoBERTa + MixDA",
        "official_ditto_claim": False,
        "three_seed_ensemble": _evaluate(
            test.assign(_score=mean_score), "_score", sizes
        ),
        "per_seed": per_seed,
        "batch_size": batch_size,
        "runtime_seconds": time.time() - started,
    }
    _write_json(metrics_path, metrics)
    return ranking_dir
