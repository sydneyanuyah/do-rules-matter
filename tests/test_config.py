from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_locked_public_scope_and_seeds() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    datasets = config["dataset_groups"]["exp01_all"]
    assert len(datasets) == 6
    assert "link_lives_release2" in datasets
    assert config["protocol"]["seeds"] == [20260725, 20260726, 20260727]
    assert config["project"]["evidence_scope"] == "public_benchmarks_only"


def test_backbones_are_revision_pinned() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    backbones = {item["id"]: item["revision"] for item in config["frozen_backbones"]}
    assert "intfloat/e5-base-v2" in backbones
    assert "sentence-transformers/all-roberta-large-v1" in backbones
    assert all(len(revision) == 40 for revision in backbones.values())
