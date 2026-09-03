#!/usr/bin/env python3
"""End-to-end fine-tuned HEF: trainable text backbone + structured evidence.

Unlike the historical frozen-score HEF implementation, gradients from the
fusion loss update the neural backbone and the structured-evidence head jointly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from paper1_hef.data import load_dataset, validate_splits
from paper1_hef.evaluate import classification_metrics, select_threshold
from paper1_hef.features import serialize, structured_features
from paper1_hef.exp05 import _nested_subsets, _pair_id_hash

SEEDS = (20260725, 20260726, 20260727)
LOCKED_DATASETS = {
    "abt_buy",
    "amazon_google",
    "walmart_amazon",
    "wdc_80_medium_seen",
    "wdc_80_medium_unseen",
    "link_lives_release2",
}

@dataclass(frozen=True)
class Backbone:
    key: str
    model_id: str
    revision: str
    kind: str = "encoder"
    max_length: int = 256
    trust_remote_code: bool = False

BACKBONES = {
    "e5": Backbone("e5", "intfloat/e5-base-v2", "f52bf8ec8c7124536f0efb74aca902b2995e5bcd", max_length=384),
    "minilm": Backbone("minilm", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "bge": Backbone("bge", "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", max_length=384),
    "gte": Backbone("gte", "Alibaba-NLP/gte-base-en-v1.5", "a829fd0e060bb84554da0dfd354d0de0f7712b7f", max_length=384, trust_remote_code=True),
    "roberta": Backbone("roberta", "sentence-transformers/all-roberta-large-v1", "cf74d8acd4f198de950bf004b262e6accfed5d2c"),
    "bert": Backbone("bert", "sentence-transformers/bert-base-nli-mean-tokens", "160a52b38a51ae87295ec3eabcf11755e5d27a8d"),
    "jina": Backbone("jina", "jinaai/jina-reranker-v2-base-multilingual", "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9", kind="reranker", max_length=512, trust_remote_code=True),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class PairEvidenceDataset(Dataset):
    def __init__(self, left: list[str], right: list[str], structured: np.ndarray, labels: np.ndarray, tokenizer: Any, max_length: int) -> None:
        self.left, self.right, self.structured, self.labels = left, right, structured, labels
        self.tokenizer, self.max_length = tokenizer, max_length
    def __len__(self) -> int: return len(self.labels)
    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.tokenizer(self.left[i], self.right[i], truncation=True, max_length=self.max_length)
        item["structured"] = self.structured[i]
        item["labels"] = int(self.labels[i])
        return item

class EvidenceCollator:
    def __init__(self, tokenizer: Any) -> None: self.base = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        structured = torch.as_tensor(np.stack([x.pop("structured") for x in features]), dtype=torch.float32)
        batch = self.base(features); batch["structured"] = structured
        return batch

class JointHEF(nn.Module):
    def __init__(self, spec: Backbone, structured_dim: int) -> None:
        super().__init__(); self.kind = spec.kind
        kwargs = {"revision": spec.revision, "trust_remote_code": spec.trust_remote_code}
        if spec.kind == "reranker":
            # Jina's checkpoint is stored with BF16 parameters. GradScaler cannot
            # unscale BF16 gradients, so keep FP32 master weights and let autocast
            # select mixed-precision kernels during forward/backward.
            self.backbone = AutoModelForSequenceClassification.from_pretrained(spec.model_id, **kwargs).float()
            neural_dim = int(getattr(self.backbone.config, "num_labels", 1))
        else:
            self.backbone = AutoModel.from_pretrained(spec.model_id, **kwargs)
            neural_dim = int(self.backbone.config.hidden_size)
        self.structured_head = nn.Sequential(nn.LayerNorm(structured_dim), nn.Linear(structured_dim, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(neural_dim + 64, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2))
    def forward(self, structured: torch.Tensor, **tokens: torch.Tensor) -> torch.Tensor:
        out = self.backbone(**tokens)
        if self.kind == "reranker":
            neural = out.logits.float()
            if neural.ndim == 1:
                neural = neural.unsqueeze(1)
        else:
            hidden = out.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            neural = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(torch.cat([neural, self.structured_head(structured)], dim=1))

def run(args: argparse.Namespace) -> Path:
    root = args.project_root.resolve(); config_path = root / "configs" / "experiment.yaml"
    config = yaml.safe_load(config_path.read_text()); spec = BACKBONES[args.backbone]
    if args.seed not in SEEDS: raise ValueError(f"seed must be one of {SEEDS}")
    if args.dataset not in LOCKED_DATASETS:
        raise ValueError("dataset is outside locked Exp1 scope")
    if args.dataset not in config["datasets"]:
        raise ValueError(f"locked dataset lacks a config definition: {args.dataset}")
    seed_all(args.seed)
    splits = load_dataset(root / config["project"]["data_root"], config["datasets"][args.dataset])
    split_report = validate_splits(splits, enforce_offer_disjoint=args.dataset == "wdc_80_medium_unseen", report_offer_overlap=config["datasets"][args.dataset]["adapter"] == "wdc", enforce_record_disjoint=config["datasets"][args.dataset]["adapter"] == "link_lives")
    subset_metadata = None
    if args.exp05_fraction is not None:
        fraction = float(args.exp05_fraction)
        exp05_spec = config["experiments"]["exp05_label_efficiency"]
        locked = sorted({float(x) for x in exp05_spec["fractions"]})
        if fraction not in locked:
            raise ValueError(f"fraction {fraction} is outside locked Experiment 5 schedule")
        train = splits["train"].reset_index(drop=True)
        group_column = exp05_spec.get("group_column", "left_id")
        if group_column not in train.columns:
            raise ValueError(f"missing Experiment 5 grouping column: {group_column}")
        _, subsets = _nested_subsets(
            train[group_column].astype(str).to_numpy(),
            train["label"].to_numpy(dtype=np.int64),
            locked,
            int(exp05_spec["subset_seed"]),
        )
        indices = subsets[fraction]
        selected = train.iloc[indices].reset_index(drop=True)
        subset_metadata = {
            "requested_fraction": fraction,
            "selected_pair_count": int(len(selected)),
            "selected_group_count": int(selected[group_column].astype(str).nunique()),
            "pair_id_sha256": _pair_id_hash(selected["pair_id"]),
            "group_column": group_column,
            "subset_seed": int(exp05_spec["subset_seed"]),
        }
        splits = dict(splits)
        splits["train"] = selected
    feature_frames = {k: structured_features(v) for k, v in splits.items()}
    feature_names = list(feature_frames["train"].columns)
    train_x = feature_frames["train"].to_numpy(dtype=np.float32)
    mean, std = train_x.mean(0), train_x.std(0); std[std < 1e-6] = 1.0
    arrays = {k: ((v.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32) for k, v in feature_frames.items()}
    labels = {k: v["label"].to_numpy(dtype=np.int64) for k, v in splits.items()}
    texts = {k: (serialize(v, "left").tolist(), serialize(v, "right").tolist()) for k, v in splits.items()}
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, trust_remote_code=spec.trust_remote_code)
    collator = EvidenceCollator(tokenizer)
    loaders = {k: DataLoader(PairEvidenceDataset(*texts[k], arrays[k], labels[k], tokenizer, spec.max_length), batch_size=args.batch_size if k == "train" else args.eval_batch_size, shuffle=k == "train", collate_fn=collator, num_workers=0, pin_memory=True) for k in splits}
    device = torch.device(args.device); positive = float(labels["train"].sum()); negative = len(labels["train"]) - positive
    weights = torch.tensor([1.0, negative / max(positive, 1.0)], device=device); loss_fn = nn.CrossEntropyLoss(weight=weights)

    def score(model: JointHEF, split: str) -> np.ndarray:
        model.eval(); chunks=[]
        with torch.no_grad():
            for batch in loaders[split]:
                batch.pop("labels"); structured=batch.pop("structured").to(device); tokens={k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"): logits=model(structured, **tokens)
                chunks.append(torch.softmax(logits.float(),1)[:,1].cpu().numpy())
        return np.concatenate(chunks)

    if args.exp05_fraction is None:
        output = root / "artifacts" / "exp01_hef_joint_finetuned" / "v1" / args.dataset / spec.key / spec.revision / f"seed_{args.seed}"
        experiment_name = "exp01_hef_joint_finetuned"
    else:
        fraction_key = f"{int(round(float(args.exp05_fraction) * 100)):03d}"
        output = root / "artifacts" / "exp05_expanded" / "v1" / f"joint_neural_hef_{spec.key}_finetuned" / args.dataset / f"fraction_{fraction_key}" / f"seed_{args.seed}"
        experiment_name = "exp05_label_efficiency"
    if output.exists() and (output / "COMPLETED.json").exists(): return output
    stage = output.with_name(output.name + f".staging.{os.getpid()}")
    stage.mkdir(parents=True, exist_ok=False); started=time.time(); trials=[]; best=None
    for lr in args.learning_rates:
        seed_all(args.seed); model=JointHEF(spec, len(feature_names)).to(device)
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total=args.max_epochs*len(loaders["train"]); scheduler=get_linear_schedule_with_warmup(optimizer,max(1,int(.1*total)),total)
        scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); stale=0; trial_best=None
        for epoch in range(1,args.max_epochs+1):
            model.train(); losses=[]
            for batch in loaders["train"]:
                y=batch.pop("labels").to(device); structured=batch.pop("structured").to(device); tokens={k:v.to(device) for k,v in batch.items()}; optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"): logits=model(structured,**tokens); loss=loss_fn(logits,y)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update(); scheduler.step(); losses.append(float(loss.detach()))
            valid=score(model,"valid"); threshold=select_threshold(labels["valid"],valid); vm=classification_metrics(labels["valid"],valid,threshold)
            event={"epoch":epoch,"lr":lr,"mean_loss":float(np.mean(losses)),"validation_f1":vm["f1"]}; print(json.dumps(event),flush=True)
            if trial_best is None or vm["f1"]>trial_best["validation"]["f1"]:
                trial_best={"learning_rate":lr,"epoch":epoch,"threshold":float(threshold),"validation":vm,"state":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}; stale=0
            else: stale+=1
            if stale>=args.patience: break
        assert trial_best is not None; trials.append({k:v for k,v in trial_best.items() if k!="state"})
        if best is None or trial_best["validation"]["f1"]>best["validation"]["f1"]: best=trial_best
        del model; torch.cuda.empty_cache()
    assert best is not None; model=JointHEF(spec,len(feature_names)).to(device); model.load_state_dict(best["state"])
    valid=score(model,"valid"); test=score(model,"test"); threshold=float(best["threshold"])
    if not np.isfinite(test).all() or float(np.std(test))<1e-8: raise ValueError("degenerate/nonfinite test scores")
    metrics={"experiment":experiment_name,"method":"hef_joint_finetuned","definition":"backbone and structured HEF fusion head optimized jointly end-to-end","frozen_backbone":False,"dataset":args.dataset,"seed":args.seed,"backbone":asdict(spec),"training_subset":subset_metadata or {"requested_fraction":1.0},"features":feature_names,"normalization":"selected_train_mean_std_only","selection":"learning rate, epoch, and threshold on validation only","test_policy":"test scored once after lock","trials":trials,"selected_learning_rate":best["learning_rate"],"selected_epoch":best["epoch"],"threshold":threshold,"validation":classification_metrics(labels["valid"],valid,threshold),"test":classification_metrics(labels["test"],test,threshold),"score_std":float(np.std(test)),"split_validation":split_report,"runtime_seconds":time.time()-started}
    write_json(stage/"metrics.json",metrics); np.savez_compressed(stage/"scores.npz",valid_pair_id=splits["valid"]["pair_id"].astype(str).to_numpy(),valid_label=labels["valid"],valid_score=valid.astype(np.float32),test_pair_id=splits["test"]["pair_id"].astype(str).to_numpy(),test_label=labels["test"],test_score=test.astype(np.float32))
    torch.save({"state_dict":model.state_dict(),"feature_mean":mean,"feature_std":std,"feature_names":feature_names,"backbone":asdict(spec)},stage/"model.pt"); tokenizer.save_pretrained(stage/"tokenizer")
    write_json(stage/"run_manifest.json",{"paper_eligible":True,"dataset":args.dataset,"seed":args.seed,"backbone":asdict(spec),"config_sha256":sha256(config_path),"files":{p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in stage.iterdir() if p.is_file()}})
    write_json(stage/"COMPLETED.json",{"status":"complete","test_rows":len(labels["test"]),"test_score_std":float(np.std(test))})
    output.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(stage),str(output)); return output

def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,required=True); p.add_argument("--dataset",required=True); p.add_argument("--backbone",choices=sorted(BACKBONES),required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--exp05-fraction",type=float); p.add_argument("--device",default="cuda"); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--eval-batch-size",type=int,default=64); p.add_argument("--learning-rates",type=float,nargs="+",default=[1e-5,2e-5]); p.add_argument("--max-epochs",type=int,default=12); p.add_argument("--patience",type=int,default=3); return p.parse_args()
if __name__=="__main__": print(run(parse()))
