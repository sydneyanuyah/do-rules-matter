"""Shared definitions for the expanded Experiment 6 ranking ablation."""
from __future__ import annotations

from collections.abc import Iterable

PRODUCT_GROUPS = {
    "lexical": ["title_jaccard", "title_char_ratio", "description_jaccard", "all_token_jaccard"],
    "numerical": ["price_similarity", "numeric_token_jaccard"],
    "categorical": ["brand_exact", "manufacturer_exact", "currency_exact"],
    "relational": ["model_exact"],
    "incumbent_rule": ["rule_score"],
    "availability_mask": ["left_field_fraction", "right_field_fraction", "shared_field_fraction"],
}

GENEALOGY_GROUPS = {
    "lexical": [
        "name_jaccard", "name_char_ratio", "birth_place_jaccard",
        "residence_information_jaccard", "occupation_jaccard", "all_token_jaccard",
    ],
    "numerical": ["birth_year_similarity"],
    "categorical": ["sex_exact", "marital_status_exact"],
    "relational": ["age_progression_similarity", "residence_parish_exact", "residence_county_exact"],
    "incumbent_rule": ["rule_score"],
    "availability_mask": ["left_field_fraction", "right_field_fraction", "shared_field_fraction"],
}

CONDITIONS = (
    "drop_lexical",
    "drop_numerical",
    "drop_categorical",
    "drop_relational",
    "drop_incumbent_rule",
    "drop_semantic",
    "drop_availability_mask",
    "raw_field_features_only",
    "aggregate_rule_score_only",
)

DATASETS = (
    "abt_buy",
    "amazon_google",
    "walmart_amazon",
    "wdc_80_medium_seen",
    "wdc_80_medium_unseen",
    "link_lives_release2",
)

SEED = 20260725


def structured_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    columns = list(columns)
    template = GENEALOGY_GROUPS if "name_jaccard" in columns else PRODUCT_GROUPS
    groups = {name: [column for column in names if column in columns] for name, names in template.items()}
    assigned = {column for names in groups.values() for column in names}
    missing = sorted(set(columns) - assigned)
    if missing:
        raise ValueError(f"Unassigned structured evidence columns: {missing}")
    empty = sorted(name for name, names in groups.items() if not names)
    if empty:
        raise ValueError(f"Empty structured evidence groups: {empty}")
    return groups


def selected_structured_features(columns: Iterable[str], condition: str) -> tuple[list[str], bool]:
    """Return structured columns and whether the neural semantic branch remains enabled."""
    columns = list(columns)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    groups = structured_groups(columns)
    semantic_enabled = condition not in {
        "drop_semantic", "raw_field_features_only", "aggregate_rule_score_only"
    }
    if condition.startswith("drop_") and condition != "drop_semantic":
        group = condition.removeprefix("drop_")
        names = [column for column in columns if column not in groups[group]]
    elif condition == "drop_semantic":
        names = columns
    elif condition == "raw_field_features_only":
        names = (
            groups["lexical"] + groups["numerical"] + groups["categorical"]
            + groups["relational"] + groups["availability_mask"]
        )
    else:
        names = groups["incumbent_rule"]
    if not names:
        raise ValueError(f"Condition {condition} produced an empty feature set")
    return names, semantic_enabled


def selected_hybrid_features(
    structured_columns: Iterable[str], condition: str, semantic_columns: Iterable[str]
) -> list[str]:
    structured_columns = list(structured_columns)
    semantic_columns = list(semantic_columns)
    names, _ = selected_structured_features(structured_columns, condition)
    if condition not in {"drop_semantic", "raw_field_features_only", "aggregate_rule_score_only"}:
        names += semantic_columns
    return names
