from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

import numpy as np
import pandas as pd

FIELDS = (
    "title",
    "brand",
    "manufacturer",
    "modelno",
    "price",
    "priceCurrency",
    "description",
)
GENEALOGY_FIELDS = (
    "name",
    "sex",
    "birth_year",
    "birth_place",
    "residence_parish",
    "residence_county",
    "residence_information",
    "occupation",
    "marital_status",
    "age",
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def tokens(value: object) -> set[str]:
    return set(TOKEN_RE.findall(clean(value)))


def jaccard(a: object, b: object) -> float:
    left, right = tokens(a), tokens(b)
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def exact(a: object, b: object) -> float:
    left, right = clean(a), clean(b)
    return float(bool(left) and bool(right) and left == right)


def char_ratio(a: object, b: object) -> float:
    left, right = clean(a), clean(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def numeric_tokens(value: object) -> set[str]:
    return set(NUMBER_RE.findall(clean(value)))


def numeric_jaccard(a: object, b: object) -> float:
    left, right = numeric_tokens(a), numeric_tokens(b)
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def price_similarity(a: object, b: object) -> float:
    try:
        left, right = float(a), float(b)
        if not np.isfinite(left) or not np.isfinite(right):
            return 0.0
        return float(np.exp(-abs(np.log1p(max(left, 0)) - np.log1p(max(right, 0)))))
    except (TypeError, ValueError):
        return 0.0


def serialize(frame: pd.DataFrame, side: str, max_chars: int = 2000) -> pd.Series:
    fields = (
        GENEALOGY_FIELDS
        if f"{side}_name" in frame
        else FIELDS
    )
    columns = [(field, f"{side}_{field}") for field in fields if f"{side}_{field}" in frame]

    def row_text(row: pd.Series) -> str:
        chunks = []
        for field, col in columns:
            value = clean(row[col])
            if value:
                chunks.append(f"[COL] {field} [VAL] {value[:max_chars]}")
        return " ".join(chunks)

    return frame.apply(row_text, axis=1)


def structured_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "left_name" in frame and "right_name" in frame:
        return genealogy_structured_features(frame)

    def col(side: str, field: str) -> pd.Series:
        name = f"{side}_{field}"
        return frame[name] if name in frame else pd.Series("", index=frame.index)

    features = pd.DataFrame(index=frame.index)
    features["title_jaccard"] = [jaccard(a, b) for a, b in zip(col("left", "title"), col("right", "title"))]
    features["title_char_ratio"] = [char_ratio(a, b) for a, b in zip(col("left", "title"), col("right", "title"))]
    features["description_jaccard"] = [jaccard(a, b) for a, b in zip(col("left", "description"), col("right", "description"))]
    left_text, right_text = serialize(frame, "left"), serialize(frame, "right")
    features["all_token_jaccard"] = [jaccard(a, b) for a, b in zip(left_text, right_text)]
    features["price_similarity"] = [price_similarity(a, b) for a, b in zip(col("left", "price"), col("right", "price"))]
    features["numeric_token_jaccard"] = [numeric_jaccard(a, b) for a, b in zip(left_text, right_text)]
    features["brand_exact"] = [exact(a, b) for a, b in zip(col("left", "brand"), col("right", "brand"))]
    features["manufacturer_exact"] = [exact(a, b) for a, b in zip(col("left", "manufacturer"), col("right", "manufacturer"))]
    features["currency_exact"] = [exact(a, b) for a, b in zip(col("left", "priceCurrency"), col("right", "priceCurrency"))]
    features["model_exact"] = [exact(a, b) for a, b in zip(col("left", "modelno"), col("right", "modelno"))]
    left_available = np.column_stack([[bool(clean(v)) for v in col("left", f)] for f in FIELDS])
    right_available = np.column_stack([[bool(clean(v)) for v in col("right", f)] for f in FIELDS])
    features["left_field_fraction"] = left_available.mean(axis=1)
    features["right_field_fraction"] = right_available.mean(axis=1)
    features["shared_field_fraction"] = (left_available & right_available).mean(axis=1)
    features["rule_score"] = features[
        ["title_jaccard", "title_char_ratio", "all_token_jaccard", "price_similarity", "brand_exact", "model_exact"]
    ].mean(axis=1)
    return features.astype(float)


def _numeric_similarity(a: object, b: object, scale: float) -> float:
    try:
        left, right = float(a), float(b)
        if not np.isfinite(left) or not np.isfinite(right):
            return 0.0
        return float(np.exp(-abs(left - right) / scale))
    except (TypeError, ValueError):
        return 0.0


def _age_progression_similarity(
    left_age: object,
    right_age: object,
    left_year: object,
    right_year: object,
) -> float:
    try:
        observed = float(left_age) - float(right_age)
        expected = float(left_year) - float(right_year)
        if not np.isfinite(observed) or not np.isfinite(expected):
            return 0.0
        return float(np.exp(-abs(observed - expected) / 2.0))
    except (TypeError, ValueError):
        return 0.0


def genealogy_structured_features(frame: pd.DataFrame) -> pd.DataFrame:
    def col(side: str, field: str) -> pd.Series:
        name = f"{side}_{field}"
        return frame[name] if name in frame else pd.Series("", index=frame.index)

    features = pd.DataFrame(index=frame.index)
    features["name_jaccard"] = [
        jaccard(a, b) for a, b in zip(col("left", "name"), col("right", "name"))
    ]
    features["name_char_ratio"] = [
        char_ratio(a, b) for a, b in zip(col("left", "name"), col("right", "name"))
    ]
    features["birth_year_similarity"] = [
        _numeric_similarity(a, b, 2.0)
        for a, b in zip(col("left", "birth_year"), col("right", "birth_year"))
    ]
    features["age_progression_similarity"] = [
        _age_progression_similarity(a, b, left_year, right_year)
        for a, b, left_year, right_year in zip(
            col("left", "age"),
            col("right", "age"),
            frame.get("source1", pd.Series("", index=frame.index)),
            frame.get("source2", pd.Series("", index=frame.index)),
        )
    ]
    features["sex_exact"] = [
        exact(a, b) for a, b in zip(col("left", "sex"), col("right", "sex"))
    ]
    features["birth_place_jaccard"] = [
        jaccard(a, b)
        for a, b in zip(col("left", "birth_place"), col("right", "birth_place"))
    ]
    features["residence_parish_exact"] = [
        exact(a, b)
        for a, b in zip(
            col("left", "residence_parish"), col("right", "residence_parish")
        )
    ]
    features["residence_county_exact"] = [
        exact(a, b)
        for a, b in zip(
            col("left", "residence_county"), col("right", "residence_county")
        )
    ]
    features["residence_information_jaccard"] = [
        jaccard(a, b)
        for a, b in zip(
            col("left", "residence_information"),
            col("right", "residence_information"),
        )
    ]
    features["occupation_jaccard"] = [
        jaccard(a, b)
        for a, b in zip(col("left", "occupation"), col("right", "occupation"))
    ]
    features["marital_status_exact"] = [
        exact(a, b)
        for a, b in zip(
            col("left", "marital_status"), col("right", "marital_status")
        )
    ]
    left_text, right_text = serialize(frame, "left"), serialize(frame, "right")
    features["all_token_jaccard"] = [
        jaccard(a, b) for a, b in zip(left_text, right_text)
    ]
    left_available = np.column_stack(
        [[bool(clean(v)) for v in col("left", field)] for field in GENEALOGY_FIELDS]
    )
    right_available = np.column_stack(
        [[bool(clean(v)) for v in col("right", field)] for field in GENEALOGY_FIELDS]
    )
    features["left_field_fraction"] = left_available.mean(axis=1)
    features["right_field_fraction"] = right_available.mean(axis=1)
    features["shared_field_fraction"] = (left_available & right_available).mean(axis=1)
    features["rule_score"] = features[
        [
            "name_jaccard",
            "name_char_ratio",
            "birth_year_similarity",
            "age_progression_similarity",
            "sex_exact",
            "birth_place_jaccard",
            "residence_parish_exact",
            "occupation_jaccard",
        ]
    ].mean(axis=1)
    return features.astype(float)


def mask_fields(frame: pd.DataFrame, probability: float, seed: int, fields: Iterable[str] = FIELDS) -> pd.DataFrame:
    out = frame.copy()
    rng = np.random.default_rng(seed)
    selected_fields = (
        GENEALOGY_FIELDS
        if fields == FIELDS and "left_name" in frame
        else fields
    )
    for field in selected_fields:
        for side in ("left", "right"):
            name = f"{side}_{field}"
            if name in out:
                mask = rng.random(len(out)) < probability
                out.loc[mask, name] = np.nan
    return out
