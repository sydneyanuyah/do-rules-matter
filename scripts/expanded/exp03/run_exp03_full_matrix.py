#!/usr/bin/env python3
"""Consolidate the locked 39-configuration Experiment 3 evaluation matrix.

This evaluator never retrains or reselects on test.  It consumes the canonical
validation-selected per-seed outputs produced by Experiments 1 and 2, validates
the six-dataset/three-seed coverage, and emits long, summary, and wide tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DATASETS = ["abt_buy", "amazon_google", "walmart_amazon", "wdc_80_medium_seen", "wdc_80_medium_unseen", "link_lives_release2"]
SEEDS = [20260725, 20260726, 20260727]
BACKBONES = {
    "e5": ("E5", "intfloat__e5-base-v2", "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"),
    "minilm": ("MiniLM", "sentence-transformers__all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "bge": ("BGE", "BAAI__bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
    "gte": ("GTE", "Alibaba-NLP__gte-base-en-v1.5", "a829fd0e060bb84554da0dfd354d0de0f7712b7f"),
    "roberta": ("RoBERTa embedding", "sentence-transformers__all-roberta-large-v1", "cf74d8acd4f198de950bf004b262e6accfed5d2c"),
    "bert": ("BERT embedding", "sentence-transformers__bert-base-nli-mean-tokens", "160a52b38a51ae87295ec3eabcf11755e5d27a8d"),
    "jina": ("Jina embedding", "jinaai__jina-embeddings-v3", "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"),
}

@dataclass(frozen=True)
class Config:
    order: int
    config_id: str
    label: str
    family: str
    classification: bool = True
    ranking: bool = True

def registry() -> list[Config]:
    out = [Config(1, "rules", "Rules", "deterministic")]
    n = 2
    for key, (label, _, _) in BACKBONES.items():
        out.append(Config(n, f"embedding_{key}", label, "frozen_embedding")); n += 1
    out += [Config(n, "equal_fusion_e5", "Equal fusion + E5", "scalar_fusion"), Config(n + 1, "convex_fusion_e5", "Convex fusion + E5", "scalar_fusion")]; n += 2
    for key, (label, _, _) in BACKBONES.items():
        out.append(Config(n, f"hef_linear_{key}", f"HEF-linear + {label}", "hef_linear")); n += 1
    for key, (label, _, _) in BACKBONES.items():
        out.append(Config(n, f"hef_gbdt_{key}", f"HEF-GBDT + {label}", "hef_gbdt")); n += 1
    neural = [
        ("jina_zero_shot", "Jina zero-shot"), ("jina_finetuned", "Jina fine-tuned"),
        ("tuned_roberta", "Tuned RoBERTa"), ("ditto_style_mixda", "Ditto-style RoBERTa + MixDA"),
        ("official_ditto_plain", "Official Ditto"), ("official_ditto_mixda", "Official Ditto + MixDA"),
        ("anymatch", "AnyMatch"),
    ]
    for key, label in neural:
        out.append(Config(n, key, label, "neural_em")); n += 1
    for key in ("e5", "jina", "roberta", "bert"):
        label = BACKBONES[key][0]
        out.append(Config(n, f"hef_gbdt_mixda_{key}", f"HEF-GBDT + MixDA-style augmentation + {label}", "mixda_hef")); n += 1
    out += [
        Config(n, "hef_e5_jina_zero_shot", "HEF-GBDT + E5 + Jina zero-shot", "cross_evidence", True, False),
        Config(n + 1, "hef_e5_tuned_roberta", "HEF-GBDT + E5 + tuned RoBERTa", "cross_evidence", False, True),
        Config(n + 2, "hef_e5_official_ditto", "HEF-GBDT + E5 + Official Ditto", "cross_evidence"),
        Config(n + 3, "hef_e5_official_ditto_mixda", "HEF-GBDT + E5 + Official Ditto + MixDA", "cross_evidence"),
    ]
    assert len(out) == 39 and [x.order for x in out] == list(range(1, 40))
    return out

def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0: return None
    with path.open() as f: return json.load(f)

def finite(value: Any) -> float | None:
    try: v = float(value)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) else None

def core_path(root: Path, dataset: str, key: str) -> Path:
    _, slug, rev = BACKBONES[key]
    return root / "exp01_final" / dataset / slug / rev / "metrics.json"

def rank_core_path(root: Path, dataset: str, key: str) -> Path:
    if key == "e5": return root / "exp02" / "ranking" / dataset / "metrics.json"
    return root / "exp02" / "ranking_by_backbone" / BACKBONES[key][1] / dataset / "metrics.json"

def add(rows: list[dict[str, Any]], cfg: Config, ds: str, task: str, seed: int | None, metric: str, value: Any, source: Path) -> None:
    v = finite(value)
    if v is None: return
    parts = source.parts
    start = next((i for i, part in enumerate(parts) if part.startswith("exp0")), 0)
    provenance = str(Path(*parts[start:]))
    rows.append({"order": cfg.order, "config_id": cfg.config_id, "model": cfg.label, "family": cfg.family, "dataset": ds, "task": task, "seed": seed if seed is not None else "deterministic", "metric": metric, "value": v, "source": provenance})

def seeded_test(rows: list[dict[str, Any]], cfg: Config, ds: str, task: str, paths: Iterable[Path], extractor) -> None:
    for p in paths:
        x = load(p)
        if not x: continue
        seed = x.get("seed")
        for metric, value in extractor(x).items(): add(rows, cfg, ds, task, seed, metric, value, p)

def class_metrics(x: dict[str, Any]) -> dict[str, Any]:
    t = x.get("test", {})
    return {k: t.get(k) for k in ("f1", "precision", "recall", "roc_auc", "average_precision")}

def rank_metrics(node: dict[str, Any]) -> dict[str, Any]:
    if "100" in node: node = node["100"]
    c = node.get("conditional", node)
    return {"mrr_at_100": c.get("mrr", c.get("mrr_conditional")), "hits_at_1": c.get("hits_at_1", c.get("hits_at_1_conditional")), "hits_at_100": c.get("hits_at_100", c.get("hits_at_100_conditional")), "ndcg_at_100": c.get("ndcg"), "hits_at_100_end_to_end": node.get("hits_at_100_end_to_end")}

def extract(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []; missing: list[dict[str, Any]] = []
    regs = registry(); byid = {x.config_id: x for x in regs}
    for ds in DATASETS:
        # Core deterministic and learned models across seven backbones.
        e5 = load(core_path(root, ds, "e5"))
        for cid, method in (("rules", "rules"), ("equal_fusion_e5", "equal_fusion"), ("convex_fusion_e5", "convex_fusion")):
            cfg = byid[cid]
            if e5 and method in e5.get("methods", {}): add(rows, cfg, ds, "classification", None, "f1", e5["methods"][method]["test"]["f1"], core_path(root, ds, "e5"))
        for key in BACKBONES:
            p = core_path(root, ds, key); x = load(p)
            for cid, method in ((f"embedding_{key}", "frozen_embedding"), (f"hef_linear_{key}", "hef_linear"), (f"hef_gbdt_{key}", "hef_gbdt")):
                cfg = byid[cid]
                if not x or method not in x.get("methods", {}): continue
                m = x["methods"][method]
                reps = m.get("repetitions", m.get("runs", []))
                if reps:
                    for rep in reps: add(rows, cfg, ds, "classification", rep.get("seed"), "f1", rep.get("test", {}).get("f1"), p)
                else: add(rows, cfg, ds, "classification", None, "f1", m.get("test", {}).get("f1"), p)
            rp = rank_core_path(root, ds, key); rx = load(rp)
            for cid, method in ((f"embedding_{key}", "embedding"), (f"hef_linear_{key}", "hef_linear"), (f"hef_gbdt_{key}", "hef_gbdt")):
                cfg = byid[cid]
                if not rx or method not in rx.get("methods", {}): continue
                m = rx["methods"][method]; per = m.get("per_seed", [])
                if per:
                    for rep in per:
                        for metric, value in rank_metrics(rep.get("metrics", rep)).items(): add(rows, cfg, ds, "ranking", rep.get("seed"), metric, value, rp)
                else:
                    for metric, value in rank_metrics(m.get("metrics", m)).items(): add(rows, cfg, ds, "ranking", None, metric, value, rp)
        rp = rank_core_path(root, ds, "e5"); rx = load(rp)
        for cid, method in (("rules", "rules"), ("equal_fusion_e5", "equal_fusion"), ("convex_fusion_e5", "convex_fusion")):
            cfg = byid[cid]
            if rx and method in rx.get("methods", {}):
                for metric, value in rank_metrics(rx["methods"][method]).items(): add(rows, cfg, ds, "ranking", None, metric, value, rp)

        # Standalone classification neural/EM systems.
        class_sources = {
            "jina_zero_shot": [(root / "exp01_jina" / ds / "metrics.json", lambda x: x.get("test", x.get("methods", {}).get("jina_cross_encoder", {}).get("test", {})))],
            "jina_finetuned": [(root / "exp01_jina_finetuned" / "v1" / ds / f"seed_{s}" / "metrics.json", lambda x: x.get("test", {})) for s in SEEDS],
            "tuned_roberta": [(root / "exp01_cross_encoder" / ds / f"seed_{s}" / "metrics.json", lambda x: x.get("test", {})) for s in SEEDS],
            "ditto_style_mixda": [(root / "exp01_ditto_style" / ds / f"seed_{s}" / "metrics.json", lambda x: x.get("test", {})) for s in SEEDS],
            "official_ditto_plain": [(root / "exp01_ditto_official" / "plain" / ds / f"seed_{s}" / "metrics.json", lambda x: x.get("test", {})) for s in SEEDS],
            "official_ditto_mixda": [(root / "exp01_ditto_official" / "mixda_all" / ds / f"seed_{s}" / "metrics.json", lambda x: x.get("test", {})) for s in SEEDS],
            "anymatch": [(root / "exp01_anymatch_official" / ds / f"seed_{s}" / "metrics.json", lambda x: x.get("test", {})) for s in SEEDS],
        }
        # WDC AnyMatch lives in full-coverage compatibility outputs.
        if ds.startswith("wdc_"): class_sources["anymatch"] = [(root / "exp01_anymatch_full_coverage" / ds / f"seed_{s}" / "metrics.json", lambda x: x.get("test", {})) for s in SEEDS]
        for cid, sources in class_sources.items():
            cfg = byid[cid]
            for p, pick in sources:
                x = load(p)
                if not x: continue
                t = pick(x)
                for metric in ("f1", "precision", "recall", "roc_auc", "average_precision"): add(rows, cfg, ds, "classification", x.get("seed"), metric, t.get(metric), p)

        # Standalone ranking neural/EM systems stored in the E5 locked pool.
        if rx:
            rank_map = {"jina_zero_shot":"jina_cross_encoder", "tuned_roberta":"tuned_cross_encoder", "ditto_style_mixda":"ditto_style_roberta_mixda", "anymatch":"anymatch_official"}
            for cid, method in rank_map.items():
                m = rx.get("methods", {}).get(method); cfg = byid[cid]
                if not m: continue
                per = m.get("per_seed", [])
                if per:
                    for rep in per:
                        for metric, value in rank_metrics(rep.get("metrics", rep)).items(): add(rows, cfg, ds, "ranking", rep.get("seed"), metric, value, rp)
                else:
                    for metric, value in rank_metrics(m.get("metrics", m)).items(): add(rows, cfg, ds, "ranking", None, metric, value, rp)
        # Fine-tuned Jina ranking and its HEF fusion.
        for s in SEEDS:
            p = root / "exp02_jina_finetuned_oof" / "v1" / ds / f"seed_{s}" / "metrics.json"; x = load(p)
            if x:
                for cid, method in (("jina_finetuned", "jina_finetuned"), ("hef_e5_jina_zero_shot", "__not_applicable__")):
                    if method in x.get("methods", {}):
                        for metric, value in rank_metrics(x["methods"][method].get("test", {})).items(): add(rows, byid[cid], ds, "ranking", s, metric, value, p)
        # Official Ditto standalone ranking.
        for cid, variant in (("official_ditto_plain", "official_ditto_plain"), ("official_ditto_mixda", "official_ditto_mixda_all")):
            for s in SEEDS:
                p = root / "exp02_propagated" / variant / ds / f"seed_{s}" / "metrics.json"; x = load(p)
                if x:
                    node = x.get("standalone", {}).get("test", x.get("test", x.get("metrics", x)))
                    for metric, value in rank_metrics(node).items(): add(rows, byid[cid], ds, "ranking", s, metric, value, p)

        # HEF + MixDA classification/ranking for four explicit backbones.
        for key in ("e5", "jina", "roberta", "bert"):
            _, slug, rev = BACKBONES[key]; cfg = byid[f"hef_gbdt_mixda_{key}"]
            for s in SEEDS:
                cp = root / "exp01_hef_mixda" / ds / slug / rev / f"seed_{s}" / "metrics.json"; x = load(cp)
                if x:
                    for metric, value in class_metrics(x).items(): add(rows, cfg, ds, "classification", s, metric, value, cp)
                rp2 = root / "exp02_propagated" / "hef_gbdt_mixda" / ds / slug / rev / f"seed_{s}" / "metrics.json"; y = load(rp2)
                if y:
                    node = y.get("test", y.get("metrics", y))
                    for metric, value in rank_metrics(node).items(): add(rows, cfg, ds, "ranking", s, metric, value, rp2)

        # Cross-evidence fusions.
        jp = root / "exp01_hef_neural" / "jina_zero_shot" / ds / "metrics.json"; jx = load(jp)
        if jx:
            for rep in jx.get("methods", {}).get("hef_gbdt", {}).get("runs", []): add(rows, byid["hef_e5_jina_zero_shot"], ds, "classification", rep.get("seed"), "f1", rep.get("test", {}).get("f1"), jp)
        for cid, variant in (("hef_e5_official_ditto", "official_ditto_plain"), ("hef_e5_official_ditto_mixda", "official_ditto_mixda_all")):
            for s in SEEDS:
                p = root / "exp01_hef_cross_evidence" / "v1" / ds / BACKBONES["e5"][1] / BACKBONES["e5"][2] / variant / f"seed_{s}" / "metrics.json"; x = load(p)
                if x:
                    for metric, value in class_metrics(x).items(): add(rows, byid[cid], ds, "classification", s, metric, value, p)
                rp3 = root / "exp02_propagated" / variant / ds / f"seed_{s}" / "metrics.json"; y = load(rp3)
                if y:
                    node = y.get("hef_gbdt_e5_fusion", {}).get("test", {})
                    for metric, value in rank_metrics(node).items(): add(rows, byid[cid], ds, "ranking", s, metric, value, rp3)
        for s in SEEDS:
            p = root / "exp02_hef_cross_evidence" / "v1" / ds / "e5_plus_tuned_roberta" / f"seed_{s}" / "metrics.json"; x = load(p)
            if x:
                for metric, value in rank_metrics(x.get("test", {})).items(): add(rows, byid["hef_e5_tuned_roberta"], ds, "ranking", s, metric, value, p)

    # Explicit cell-level coverage audit (primary metric only).
    for cfg in regs:
        for ds in DATASETS:
            for task, applicable, primary in (("classification", cfg.classification, "f1"), ("ranking", cfg.ranking, "mrr_at_100")):
                if not applicable: continue
                found = [r for r in rows if r["config_id"] == cfg.config_id and r["dataset"] == ds and r["task"] == task and r["metric"] == primary]
                deterministic = cfg.config_id in {"rules", "equal_fusion_e5", "convex_fusion_e5", "jina_zero_shot"} or cfg.config_id.startswith("embedding_")
                expected = 1 if deterministic else 3
                actual = len({str(r["seed"]) for r in found})
                if actual < expected:
                    missing.append({"config_id":cfg.config_id,"model":cfg.label,"dataset":ds,"task":task,"reason":f"expected_{expected}_runs_found_{actual}"})
    return rows, missing

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--artifacts-root", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--strict", action="store_true"); args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    rows, missing = extract(args.artifacts_root)
    fields = ["order","config_id","model","family","dataset","task","seed","metric","value","source"]
    write_csv(out / "per_run_metrics.csv", rows, fields)
    grouped: dict[tuple, list[float]] = {}
    for r in rows: grouped.setdefault((r["order"],r["config_id"],r["model"],r["family"],r["dataset"],r["task"],r["metric"]), []).append(r["value"])
    summary=[]
    for k, vals in sorted(grouped.items()):
        summary.append(dict(zip(["order","config_id","model","family","dataset","task","metric"],k)) | {"n_runs":len(vals),"mean":statistics.fmean(vals),"std":statistics.stdev(vals) if len(vals)>1 else "","min":min(vals),"max":max(vals)})
    write_csv(out / "summary_long.csv", summary, ["order","config_id","model","family","dataset","task","metric","n_runs","mean","std","min","max"])
    for task, metric, filename in (("classification","f1","classification_f1_wide.csv"),("ranking","mrr_at_100","ranking_mrr_at_100_wide.csv"),("ranking","hits_at_100","ranking_hits_at_100_wide.csv")):
        wide=[]
        for cfg in registry():
            rec={"order":cfg.order,"config_id":cfg.config_id,"model":cfg.label}
            for ds in DATASETS:
                hit=next((x for x in summary if x["config_id"]==cfg.config_id and x["dataset"]==ds and x["task"]==task and x["metric"]==metric),None)
                rec[f"{ds}_mean"]=hit["mean"] if hit else ""; rec[f"{ds}_std"]=hit["std"] if hit else ""; rec[f"{ds}_n"]=hit["n_runs"] if hit else 0
            wide.append(rec)
        wf=["order","config_id","model"]+[f"{ds}_{suffix}" for ds in DATASETS for suffix in ("mean","std","n")]
        write_csv(out/filename,wide,wf)
    write_csv(out/"missing_cells.csv",missing,["config_id","model","dataset","task","reason"])
    with (out/"configuration_registry.json").open("w") as f: json.dump([c.__dict__ for c in registry()],f,indent=2)
    manifest={"experiment":"exp03_full_39_configuration_matrix","datasets":DATASETS,"seeds":SEEDS,"configuration_count":39,"per_run_metric_rows":len(rows),"summary_rows":len(summary),"missing_applicable_cells":len(missing),"status":"complete" if not missing else "partial_waiting_dependencies","test_policy":"reuse validation-selected locked predictions; no test-time selection","std_policy":"sample SD (n-1); blank for deterministic single-run methods"}
    with (out/"manifest.json").open("w") as f: json.dump(manifest,f,indent=2)
    if args.strict and missing: raise SystemExit(f"incomplete: {len(missing)} applicable cells missing")
    print(json.dumps(manifest,indent=2))

if __name__ == "__main__": main()
