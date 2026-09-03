from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_column(name: str) -> str:
    aliases = {
        "name": "title",
        "model_no": "modelno",
        "model_number": "modelno",
        "currency": "priceCurrency",
    }
    return aliases.get(name, name)


def _standardize_prefixed(frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for col in frame.columns:
        if col.endswith("_left"):
            renamed[col] = f"left_{_normalize_column(col[:-5])}"
        elif col.endswith("_right"):
            renamed[col] = f"right_{_normalize_column(col[:-6])}"
    out = frame.rename(columns=renamed).copy()
    out["label"] = out["label"].astype(int)
    if "pair_id" not in out:
        left = out.get("left_id", pd.Series(out.index, index=out.index))
        right = out.get("right_id", pd.Series(out.index, index=out.index))
        out["pair_id"] = left.astype(str) + "#" + right.astype(str)
    return out


def load_wdc(root: Path, spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    directory = root / spec["directory"]
    return {
        split: _standardize_prefixed(
            pd.read_json(directory / spec[split], compression="gzip", lines=True)
        )
        for split in ("train", "valid", "test")
    }


def _join_deepmatcher(directory: Path, split: str) -> pd.DataFrame:
    pairs = pd.read_csv(directory / f"{split}.csv")
    left = pd.read_csv(directory / "tableA.csv")
    right = pd.read_csv(directory / "tableB.csv")
    left_id = "id"
    right_id = "id"
    left = left.rename(
        columns={c: f"left_{_normalize_column(c)}" for c in left.columns if c != left_id}
    )
    right = right.rename(
        columns={c: f"right_{_normalize_column(c)}" for c in right.columns if c != right_id}
    )
    left = left.rename(columns={left_id: "ltable_id"})
    right = right.rename(columns={right_id: "rtable_id"})
    out = pairs.merge(left, on="ltable_id", how="left", validate="many_to_one")
    out = out.merge(right, on="rtable_id", how="left", validate="many_to_one")
    out["left_id"] = out["ltable_id"]
    out["right_id"] = out["rtable_id"]
    out["pair_id"] = out["left_id"].astype(str) + "#" + out["right_id"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def load_deepmatcher(root: Path, spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    directory = root / spec["directory"]
    return {split: _join_deepmatcher(directory, split) for split in ("train", "valid", "test")}


def load_link_lives(root: Path, spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    path = root / spec["directory"] / spec["file"]
    frame = pd.read_csv(path)
    required = {"split", "label", "pair_id", "left_id", "right_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Link-Lives processed pairs missing columns: {sorted(missing)}")
    unknown = set(frame["split"].dropna().unique()) - {"train", "valid", "test"}
    if unknown:
        raise ValueError(f"Link-Lives unknown split values: {sorted(unknown)}")
    return {
        split: frame.loc[frame["split"].eq(split)].copy()
        for split in ("train", "valid", "test")
    }


def load_dataset(root: Path, spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if spec["adapter"] == "wdc":
        return load_wdc(root, spec)
    if spec["adapter"] == "deepmatcher":
        return load_deepmatcher(root, spec)
    if spec["adapter"] == "link_lives":
        return load_link_lives(root, spec)
    raise ValueError(f"Unknown adapter: {spec['adapter']}")


def validate_splits(
    splits: dict[str, pd.DataFrame],
    enforce_offer_disjoint: bool = False,
    report_offer_overlap: bool = False,
    enforce_record_disjoint: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}, "pair_overlap": {}}
    for split, frame in splits.items():
        if frame["label"].isna().any():
            raise ValueError(f"{split}: null labels")
        values = set(frame["label"].unique())
        if not values <= {0, 1}:
            raise ValueError(f"{split}: labels outside {{0,1}}: {values}")
        report["splits"][split] = {
            "rows": int(len(frame)),
            "positives": int(frame["label"].sum()),
            "positive_rate": float(frame["label"].mean()),
            "duplicate_pair_ids": int(frame["pair_id"].duplicated().sum()),
        }
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = set(splits[a]["pair_id"]) & set(splits[b]["pair_id"])
        report["pair_overlap"][f"{a}_{b}"] = len(overlap)
        if overlap:
            raise ValueError(f"Pair leakage between {a} and {b}: {len(overlap)} pairs")
    if enforce_offer_disjoint or report_offer_overlap:
        report["offer_overlap"] = {}
        offers = {
            split: set(frame["left_id"].astype(str)) | set(frame["right_id"].astype(str))
            for split, frame in splits.items()
        }
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = offers[a] & offers[b]
            report["offer_overlap"][f"{a}_{b}"] = len(overlap)
            if overlap and enforce_offer_disjoint:
                raise ValueError(f"Offer leakage between {a} and {b}: {len(overlap)} offers")
    if enforce_record_disjoint:
        report["record_overlap"] = {}
        records = {
            split: set(frame["left_id"].astype(str))
            | set(frame["right_id"].astype(str))
            for split, frame in splits.items()
        }
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = records[a] & records[b]
            report["record_overlap"][f"{a}_{b}"] = len(overlap)
            if overlap:
                raise ValueError(
                    f"Record leakage between {a} and {b}: {len(overlap)} records"
                )
    return report
