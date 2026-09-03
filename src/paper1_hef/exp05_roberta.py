from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cross_encoder import run_cross_encoder
from .data import load_dataset
from .exp05 import _nested_subsets, _pair_id_hash


def run_exp05_roberta(
    project_root: Path,
    config: dict[str, Any],
    dataset: str,
    fraction: float,
    seed: int,
    batch_size: int = 32,
    device: str = "cuda",
) -> Path:
    """Fine-tune RoBERTa on one locked Experiment 5 training subset."""
    if dataset not in config["dataset_groups"]["exp01_all"]:
        raise ValueError("Experiment 5 is locked to the registered public Exp1 datasets")
    spec = config["experiments"]["exp05_label_efficiency"]
    fractions = sorted({float(value) for value in spec["fractions"]})
    if fraction not in fractions:
        raise ValueError(f"Fraction {fraction} is not in the locked Experiment 5 schedule")
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    if seed not in seeds:
        raise ValueError(f"Seed {seed} is not in the locked protocol")

    splits = load_dataset(
        project_root / config["project"]["data_root"],
        config["datasets"][dataset],
    )
    labels = splits["train"]["label"].to_numpy()
    left_ids = splits["train"]["left_id"].astype(str).to_numpy()
    _, subsets = _nested_subsets(
        left_ids,
        labels,
        fractions,
        int(spec["subset_seed"]),
    )
    indices = subsets[fraction]
    pair_ids = splits["train"].iloc[indices]["pair_id"].astype(str).to_numpy()
    subset_hash = _pair_id_hash(pair_ids)
    key = f"{int(round(fraction * 100)):03d}"

    manifest_path = (
        project_root / config["project"]["output_root"] / "exp05" / dataset
        / "subset_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["fractions"][key]
    if subset_hash != expected["pair_id_sha256"]:
        raise ValueError(
            f"{dataset}/{key}: RoBERTa subset hash does not match fusion subset"
        )
    if len(indices) != int(expected["selected_pair_count"]):
        raise ValueError(f"{dataset}/{key}: selected pair count does not match manifest")

    output = (
        project_root / config["project"]["output_root"] / "exp05_roberta"
        / dataset / f"fraction_{key}" / f"seed_{seed}"
    )
    metadata = {
        "requested_fraction": fraction,
        "fraction_key": key,
        "selected_pair_count": len(indices),
        "pair_id_sha256": subset_hash,
        "reference_manifest": str(manifest_path.relative_to(project_root)),
        "subset_policy": spec["subset_policy"],
        "subset_seed": int(spec["subset_seed"]),
    }
    return run_cross_encoder(
        project_root,
        config,
        dataset,
        seed,
        batch_size,
        device,
        train_indices=indices,
        output=output,
        experiment="exp05_label_efficiency_roberta",
        subset_metadata=metadata,
    )
