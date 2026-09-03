#!/usr/bin/env python3
"""Run one revised Exp7/8 model × public-dataset lane."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from common import atomic_json, controlled_record_mask, memory_timer, query_metrics, scenarios
from paper1_hef.features import FIELDS, GENEALOGY_FIELDS, serialize, structured_features


def load_ranking_source():
    try:
        from paper1_hef.exp02 import _ranking_source
        return _ranking_source
    except ImportError:
        path = Path(__file__).with_name("exp02_compat.py")
        spec = importlib.util.spec_from_file_location("paper1_hef._exp78_exp02_compat", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load ranking compatibility module: {path}")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module._ranking_source


_ranking_source = load_ranking_source()


SEEDS = (20260725, 20260726, 20260727)
E5_ID = "intfloat/e5-base-v2"
E5_REV = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


def load_joblib(path: Path) -> Any:
    """Load NumPy-2-created sklearn artifacts under NumPy 1.26 images."""
    import numpy.random._pickle as random_pickle
    original = random_pickle.__bit_generator_ctor
    def compatible(value: Any = "MT19937") -> Any:
        if isinstance(value, type):
            return value()
        return original(value)
    random_pickle.__bit_generator_ctor = compatible
    try:
        return joblib.load(path)
    finally:
        random_pickle.__bit_generator_ctor = original


def fields_for(frame: pd.DataFrame) -> tuple[str, ...]:
    return GENEALOGY_FIELDS if "name" in frame.columns else FIELDS


def pair_pool(root: Path, config: dict[str, Any], dataset: str) -> tuple[pd.DataFrame, list[str]]:
    pool = pd.read_csv(root / "artifacts/exp02/candidate_pools" / dataset / "top100.csv.gz")
    pool = pool.loc[pool["split"].eq("test")].copy()
    pool["query_id"] = pool["query_id"].astype(str); pool["candidate_id"] = pool["candidate_id"].astype(str)
    source = _ranking_source(root, config, dataset)
    queries, candidates = source["queries"], source["candidates"]
    fields = sorted(set(fields_for(queries)) | set(fields_for(candidates)))
    qfields = [field for field in fields if field in queries]
    cfields = [field for field in fields if field in candidates]
    left = queries[["id", *qfields]].drop_duplicates("id").rename(
        columns={"id": "query_id", **{field: f"left_{field}" for field in qfields}}
    )
    right = candidates[["id", *cfields]].drop_duplicates("id").rename(
        columns={"id": "candidate_id", **{field: f"right_{field}" for field in cfields}}
    )
    left["query_id"] = left["query_id"].astype(str); right["candidate_id"] = right["candidate_id"].astype(str)
    paired = pool.merge(left, on="query_id", how="left", validate="many_to_one")
    paired = paired.merge(right, on="candidate_id", how="left", validate="many_to_one")
    if paired.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate candidate-pool keys")
    return paired, fields


def logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.reshape(-1).float())
    return torch.softmax(logits.float(), dim=-1)[:, 1]


def pair_text(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    return serialize(frame, "left").tolist(), serialize(frame, "right").tolist()


def score_hf_pair(model: Any, tokenizer: Any, frame: pd.DataFrame, device: torch.device,
                  batch_size: int, max_length: int) -> np.ndarray:
    left, right = pair_text(frame); chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            encoded = tokenizer(
                left[start:start + batch_size], right[start:start + batch_size], padding=True,
                truncation=True, max_length=max_length, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                scores = logits_to_scores(model(**encoded).logits)
            chunks.append(scores.cpu().numpy())
    return np.concatenate(chunks)


def directory_size(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


class Scorer:
    def __init__(self, root: Path, dataset: str, model_id: str, device: str, batch_size: int,
                 joint_source: Path | None) -> None:
        self.root, self.dataset, self.model_id = root, dataset, model_id
        self.device, self.batch_size = torch.device(device), batch_size
        self.joint_source = joint_source
        self.models: list[Any] = []
        self.tokenizers: list[Any] = []
        self.encoder: Any = None
        self.feature_names: list[str] | None = None
        self.model_bytes = 0
        self.load_seconds = 0.0
        self._load()

    def _load_hf(self, directories: list[Path], trust_remote_code: bool) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        for directory in directories:
            self.tokenizers.append(AutoTokenizer.from_pretrained(directory, trust_remote_code=trust_remote_code))
            model = AutoModelForSequenceClassification.from_pretrained(
                directory, trust_remote_code=trust_remote_code
            ).to(self.device).eval()
            self.models.append(model); self.model_bytes += directory_size(directory)

    def _load(self) -> None:
        started = time.perf_counter(); r, d = self.root, self.dataset
        if self.model_id in {"jina_finetuned", "hef_gbdt_e5_jina_oof"}:
            dirs = [r / "artifacts/exp02_jina_finetuned_oof/v1" / d / f"seed_{s}" / "model" for s in SEEDS]
            self._load_hf(dirs, True)
            if self.model_id.startswith("hef_gbdt"):
                self.fusion = [load_joblib(dirs[i].parent / "hef_gbdt_e5_plus_tuned_jina.joblib") for i in range(3)]
                self.model_bytes += sum((dirs[i].parent / "hef_gbdt_e5_plus_tuned_jina.joblib").stat().st_size for i in range(3))
        elif self.model_id in {"roberta_finetuned", "hef_gbdt_e5_roberta_oof"}:
            dirs = [r / "artifacts/exp02_hef_cross_evidence/v1" / d / "e5_plus_tuned_roberta" / f"seed_{s}" / "full_roberta_model" for s in SEEDS]
            self._load_hf(dirs, False)
            if self.model_id.startswith("hef_gbdt"):
                self.fusion = [load_joblib(dirs[i].parent / "hef_gbdt.joblib") for i in range(3)]
                self.model_bytes += sum((dirs[i].parent / "hef_gbdt.joblib").stat().st_size for i in range(3))
        elif self.model_id.startswith("official_ditto"):
            from transformers import AutoModel, AutoTokenizer
            class OfficialDittoModel(torch.nn.Module):
                def __init__(self, model_id: str, revision: str) -> None:
                    super().__init__(); self.bert = AutoModel.from_pretrained(model_id, revision=revision)
                    self.fc = torch.nn.Linear(self.bert.config.hidden_size, 2)
                def forward(self, values: torch.Tensor) -> torch.Tensor:
                    return self.fc(self.bert(values)[0][:, 0, :])
            augmentation = "mixda_all" if self.model_id.endswith("mixda") else "plain"
            for seed in SEEDS:
                directory = r / "artifacts/exp01_ditto_official" / augmentation / d / f"seed_{seed}"
                checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
                metrics = json.loads((directory / "metrics.json").read_text())
                model_id = metrics["model_id"]; revision = metrics["revision"]
                model = OfficialDittoModel(model_id, revision).to(self.device)
                model.load_state_dict(checkpoint["model"])
                self.models.append(model.eval())
                self.tokenizers.append(AutoTokenizer.from_pretrained(model_id, revision=revision))
                self.model_bytes += (directory / "model.pt").stat().st_size
        elif self.model_id.startswith("hef_linear"):
            from sentence_transformers import SentenceTransformer
            if self.model_id.endswith("bert"):
                slug, hf_id, revision = "sentence-transformers__bert-base-nli-mean-tokens", "sentence-transformers/bert-base-nli-mean-tokens", "160a52b38a51ae87295ec3eabcf11755e5d27a8d"
            else:
                slug, hf_id, revision = "BAAI__bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
            self.encoder = SentenceTransformer(hf_id, revision=revision, device=str(self.device))
            directory = r / "artifacts/exp02/ranking_by_backbone" / slug / d
            self.models = [load_joblib(directory / f"hef_linear_seed_{seed}.joblib") for seed in SEEDS]
            self.model_bytes = sum((directory / f"hef_linear_seed_{seed}.joblib").stat().st_size for seed in SEEDS)
        elif self.model_id.startswith("joint_neural_hef"):
            if not self.joint_source:
                raise ValueError("--joint-source is required for joint HEF")
            sys.path.insert(0, str(self.joint_source))
            from run_joint_hef import BACKBONES, JointHEF
            from transformers import AutoTokenizer
            key = "roberta" if self.model_id.endswith("roberta") else "jina"
            spec = BACKBONES[key]
            for seed in SEEDS:
                directory = r / "artifacts/exp01_hef_joint_finetuned/v1" / d / key / spec.revision / f"seed_{seed}"
                checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
                model = JointHEF(spec, len(checkpoint["feature_names"])).to(self.device)
                model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
                self.models.append((model, checkpoint))
                self.tokenizers.append(AutoTokenizer.from_pretrained(directory / "tokenizer", trust_remote_code=spec.trust_remote_code))
                self.model_bytes += (directory / "model.pt").stat().st_size + directory_size(directory / "tokenizer")
        else:
            raise ValueError(f"Unsupported registry model: {self.model_id}")
        self.load_seconds = float(time.perf_counter() - started)

    def _standalone(self, frame: pd.DataFrame) -> np.ndarray:
        scores = [score_hf_pair(m, t, frame, self.device, self.batch_size, 512)
                  for m, t in zip(self.models, self.tokenizers)]
        return np.mean(scores, axis=0)

    def _official_ditto(self, frame: pd.DataFrame) -> np.ndarray:
        left = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize(frame, "left")]
        right = [value.replace("[COL]", "COL").replace("[VAL]", "VAL") for value in serialize(frame, "right")]
        ensemble = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(frame), self.batch_size):
                    batch = [tokenizer.encode(a, b, max_length=256, truncation=True)
                             for a, b in zip(left[start:start+self.batch_size], right[start:start+self.batch_size])]
                    width = max(map(len, batch))
                    values = torch.tensor([row + [0] * (width-len(row)) for row in batch], dtype=torch.long, device=self.device)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                        chunks.append(torch.softmax(model(values).float(), 1)[:, 1].cpu().numpy())
            ensemble.append(np.concatenate(chunks))
        return np.mean(ensemble, axis=0)

    def _semantic_e5(self, frame: pd.DataFrame) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if self.encoder is None:
            self.encoder = SentenceTransformer(E5_ID, revision=E5_REV, device=str(self.device))
        left, right = pair_text(frame)
        left = ["query: " + value for value in left]; right = ["passage: " + value for value in right]
        lv = self.encoder.encode(left, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        rv = self.encoder.encode(right, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.einsum("ij,ij->i", lv, rv, optimize=True)

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
        stage: dict[str, float] = {}
        if self.model_id in {"jina_finetuned", "roberta_finetuned"}:
            started = time.perf_counter(); score = self._standalone(frame)
            stage["neural_inference"] = time.perf_counter() - started
            return score, stage
        if self.model_id.startswith("official_ditto"):
            started = time.perf_counter(); score = self._official_ditto(frame)
            stage["neural_inference"] = time.perf_counter() - started
            return score, stage
        if self.model_id.startswith("hef_linear"):
            started = time.perf_counter(); structured = structured_features(frame)
            stage["structured_features"] = time.perf_counter() - started
            started = time.perf_counter(); left, right = pair_text(frame)
            lv = self.encoder.encode(left, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
            rv = self.encoder.encode(right, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
            structured["embedding_score"] = np.einsum("ij,ij->i", lv, rv, optimize=True)
            stage["semantic_encoding"] = time.perf_counter() - started
            started = time.perf_counter(); values = []
            for model in self.models:
                names = list(getattr(model, "feature_names_in_", structured.columns))
                values.append(model.predict_proba(structured[names])[:, 1])
            stage["cached_fusion"] = time.perf_counter() - started
            return np.mean(values, axis=0), stage
        if self.model_id.startswith("hef_gbdt"):
            started = time.perf_counter(); structured = structured_features(frame)
            stage["structured_features"] = time.perf_counter() - started
            started = time.perf_counter(); structured["embedding_score"] = self._semantic_e5(frame)
            stage["e5_encoding"] = time.perf_counter() - started
            started = time.perf_counter(); neural = self._standalone(frame)
            stage["task_neural_inference"] = time.perf_counter() - started
            neural_name = "tuned_jina_score" if self.model_id.endswith("jina_oof") else "tuned_roberta_score"
            structured[neural_name] = neural
            started = time.perf_counter(); values = []
            for model in self.fusion:
                names = list(getattr(model, "feature_names_in_", structured.columns))
                values.append(model.predict_proba(structured[names])[:, 1])
            stage["cached_fusion"] = time.perf_counter() - started
            return np.mean(values, axis=0), stage
        if self.model_id.startswith("joint_neural_hef"):
            from run_joint_hef import EvidenceCollator, PairEvidenceDataset
            from torch.utils.data import DataLoader
            values = []
            for (model, checkpoint), tokenizer in zip(self.models, self.tokenizers):
                names = list(checkpoint["feature_names"])
                started = time.perf_counter(); sf = structured_features(frame)
                x = ((sf[names].to_numpy(np.float32) - np.asarray(checkpoint["feature_mean"])) /
                     np.asarray(checkpoint["feature_std"])).astype(np.float32)
                stage["structured_features"] = stage.get("structured_features", 0.0) + time.perf_counter() - started
                left, right = pair_text(frame)
                ds = PairEvidenceDataset(left, right, x, frame["label"].to_numpy(np.int64), tokenizer, 512)
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=EvidenceCollator(tokenizer))
                chunks = []
                with torch.inference_mode():
                    for batch in loader:
                        batch.pop("labels"); structured = batch.pop("structured").to(self.device)
                        tokens = {k: v.to(self.device) for k, v in batch.items()}
                        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                            chunks.append(torch.softmax(model(structured, **tokens).float(), 1)[:, 1].cpu().numpy())
                values.append(np.concatenate(chunks))
            stage["joint_neural_inference"] = sum(stage.values())
            return np.mean(values, axis=0), stage
        raise AssertionError(self.model_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--joint-source", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    root = args.project_root.resolve(); registry = json.loads(args.registry.read_text())
    if args.model not in {item["id"] for item in registry["models"]} or args.dataset not in registry["datasets"]:
        raise ValueError("Model/dataset outside locked registry")
    output = root / "artifacts/exp07_08_rerankers_v2" / args.model / args.dataset
    if (output / "SUCCESS.json").exists():
        print(output); return
    lock = output.parent / f".{args.dataset}.lock"; lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
    try:
        config = yaml.safe_load((root / "configs/experiment.yaml").read_text())
        frame, fields = pair_pool(root, config, args.dataset)
        scorer = Scorer(root, args.dataset, args.model, args.device, args.batch_size, args.joint_source)
        rows = []; baseline: dict[str, float] | None = None
        for probability, mask_seed in scenarios():
            key = f"p{int(round(probability * 100)):02d}_seed{mask_seed}"
            masked, mask = controlled_record_mask(frame, fields, probability, mask_seed)
            with memory_timer(args.device) as timing:
                scores, stages = scorer.score(masked)
            scored = masked[["query_id", "candidate_id", "retrieval_rank", "label"]].copy()
            scored["score"] = scores
            metrics, per_query = query_metrics(scored, "score")
            if probability == 0:
                baseline = metrics
            assert baseline is not None
            metrics["mrr_retention"] = metrics["mrr_at_100"] / baseline["mrr_at_100"]
            metrics["hits_at_1_retention"] = metrics["hits_at_1"] / baseline["hits_at_1"] if baseline["hits_at_1"] else None
            scenario_dir = output / key; scenario_dir.mkdir(parents=True, exist_ok=True)
            per_query.to_parquet(scenario_dir / "per_query.parquet", index=False)
            np.savez_compressed(scenario_dir / "scores.npz", query_id=scored.query_id.astype(str),
                                candidate_id=scored.candidate_id.astype(str), label=scored.label.to_numpy(np.int8),
                                score=np.asarray(scores, dtype=np.float32))
            payload = {"model": args.model, "dataset": args.dataset, "scenario": key,
                       "mask": mask, "metrics": metrics, "timing": timing, "stages": stages}
            atomic_json(scenario_dir / "metrics.json", payload); rows.append(payload)
        atomic_json(output / "metrics.json", {
            "status": "complete", "model": args.model, "dataset": args.dataset,
            "scenarios": rows, "model_load_seconds": scorer.load_seconds,
            "cold_end_to_end_seconds": scorer.load_seconds + rows[0]["timing"]["seconds"],
            "serialized_model_bytes": scorer.model_bytes,
            "training_seed_policy": "three-checkpoint score ensemble",
            "mask_seed_policy": "one clean baseline and three matched record-level masks per nonzero level",
        })
        atomic_json(output / "SUCCESS.json", {"validated": True, "scenarios": 13})
        print(output)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
