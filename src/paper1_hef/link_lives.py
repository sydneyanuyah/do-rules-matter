from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import re

import pandas as pd

from paper1_hef.features import clean, jaccard

BENCHMARK_REQUIRED = {
    "source1_type",
    "source1",
    "source2",
    "parish",
    "id1",
    "type",
    "id2",
    "type_resolution",
    "status",
    "period",
    "linking_unit",
    "method_type",
}

ALA_COLUMNS = {
    "id": "record_id",
    "Place": "place",
    "Res. parish": "residence_parish",
    "Res. county": "residence_county",
    "Res. information": "residence_information",
    "Occ. information": "occupation",
    "Marital st.": "marital_status",
    "Sex": "sex",
    "Name": "name",
    "Birth place": "birth_place",
    "Age": "age",
    "Birth year": "birth_year",
    "Birth month": "birth_month",
    "Birth day": "birth_day",
    "R.no.": "relationship_number",
    "HH. id": "household_id",
}

PAIR_FIELDS = tuple(column for column in ALA_COLUMNS.values() if column != "record_id")
YEAR_RE = re.compile(r"^-?\d+$")


def load_benchmark(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    missing = BENCHMARK_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Link-Lives benchmark missing columns: {sorted(missing)}")
    return frame


def select_primary_census_links(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select conservative, explicit gold links for the 1845–1901 core.

    Uncertain, contested, secondary-information, experiment-method, and
    permission-controlled source rows remain available for diagnostics but do
    not enter the primary positive set.
    """

    filters = {
        "census_source": frame["source1_type"].eq("Census"),
        "primary_period": frame["period"].eq("1845-1901"),
        "production_method": frame["method_type"].eq("production"),
        "explicit_link": frame["type"].eq("link"),
        "uncontested": frame["type_resolution"].eq("uncontested"),
        "binary_status_link": frame["status"].eq("link"),
        "valid_target_id": pd.to_numeric(frame["id2"], errors="coerce").ge(0),
    }
    mask = pd.Series(True, index=frame.index)
    retained: dict[str, int] = {}
    for name, current in filters.items():
        mask &= current.fillna(False)
        retained[name] = int(mask.sum())

    selected = frame.loc[mask].copy()
    selected["id1"] = pd.to_numeric(selected["id1"], errors="raise").astype("int64")
    selected["id2"] = pd.to_numeric(selected["id2"], errors="raise").astype("int64")
    selected["source1"] = pd.to_numeric(selected["source1"], errors="raise").astype("int64")
    selected["source2"] = pd.to_numeric(selected["source2"], errors="raise").astype("int64")

    duplicate_keys = int(
        selected.duplicated(["source1", "source2", "id1", "id2"]).sum()
    )
    if duplicate_keys:
        raise ValueError(f"Duplicate primary gold links: {duplicate_keys}")

    report = {
        "input_rows": int(len(frame)),
        "retained_after_each_gate": retained,
        "primary_rows": int(len(selected)),
        "parishes": int(selected["parish"].nunique()),
        "linking_units": int(selected["linking_unit"].nunique()),
        "source_pair_rows": {
            f"{int(source1)}-{int(source2)}": int(len(group))
            for (source1, source2), group in selected.groupby(["source1", "source2"])
        },
    }
    return selected, report


def ala_path(extracted_root: Path, year: int) -> Path:
    return (
        extracted_root
        / "Data"
        / "CSV"
        / "main_datasets"
        / "ALA"
        / f"ALA_census_{year}.csv"
    )


def load_ala_census(
    extracted_root: Path, year: int, ids: Iterable[int] | None = None
) -> pd.DataFrame:
    path = ala_path(extracted_root, year)
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = set(ALA_COLUMNS) - available
    if missing:
        raise ValueError(f"{path.name} missing ALA columns: {sorted(missing)}")

    frame = pd.read_csv(path, usecols=list(ALA_COLUMNS), low_memory=False).rename(
        columns=ALA_COLUMNS
    )
    frame["record_id"] = pd.to_numeric(frame["record_id"], errors="raise").astype(
        "int64"
    )
    if frame["record_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate record ids")
    if ids is not None:
        wanted = {int(value) for value in ids}
        frame = frame[frame["record_id"].isin(wanted)].copy()
        missing_ids = wanted - set(frame["record_id"])
        if missing_ids:
            sample = sorted(missing_ids)[:10]
            raise ValueError(
                f"{path.name}: {len(missing_ids)} benchmark ids absent; sample={sample}"
            )
    frame["source_year"] = int(year)
    return frame


def _side_records(frame: pd.DataFrame, side: str, key: str) -> pd.DataFrame:
    renamed = {
        column: f"{side}_{column}"
        for column in PAIR_FIELDS
        if column in frame.columns
    }
    return frame.rename(columns={"record_id": key, **renamed}).drop(
        columns=["source_year"]
    )


def build_primary_positive_pairs(
    benchmark: pd.DataFrame, extracted_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected, report = select_primary_census_links(benchmark)
    batches: list[pd.DataFrame] = []

    for (source1, source2), links in selected.groupby(["source1", "source2"]):
        left = load_ala_census(extracted_root, int(source1), links["id1"])
        right = load_ala_census(extracted_root, int(source2), links["id2"])
        batch = links.merge(
            _side_records(left, "left", "id1"),
            on="id1",
            how="left",
            validate="many_to_one",
        )
        batch = batch.merge(
            _side_records(right, "right", "id2"),
            on="id2",
            how="left",
            validate="many_to_one",
        )
        batches.append(batch)

    pairs = pd.concat(batches, ignore_index=True)
    pairs["left_id"] = pairs["source1"].astype(str) + ":" + pairs["id1"].astype(str)
    pairs["right_id"] = pairs["source2"].astype(str) + ":" + pairs["id2"].astype(str)
    pairs["pair_id"] = pairs["left_id"] + "#" + pairs["right_id"]
    pairs["label"] = 1
    pairs["split_group"] = pairs["parish"].astype(str)

    if pairs["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair ids after Link-Lives joins")
    report["joined_rows"] = int(len(pairs))
    report["left_missing_name"] = int(pairs["left_name"].isna().sum())
    report["right_missing_name"] = int(pairs["right_name"].isna().sum())
    return pairs, report


def assign_parish_splits(
    pairs: pd.DataFrame,
    seed: int = 20260725,
    ratios: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Greedily balance source-pair counts while keeping parishes disjoint."""

    ratios = ratios or {"train": 0.70, "valid": 0.15, "test": 0.15}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")

    frame = pairs.copy()
    frame["_source_pair"] = (
        frame["source1"].astype(str) + "-" + frame["source2"].astype(str)
    )
    table = frame.pivot_table(
        index="split_group",
        columns="_source_pair",
        values="pair_id",
        aggfunc="count",
        fill_value=0,
    )
    targets = {split: table.sum() * ratio for split, ratio in ratios.items()}
    totals = {
        split: pd.Series(0.0, index=table.columns) for split in ratios
    }
    assignment: dict[str, str] = {}
    groups = sorted(
        table.index,
        key=lambda group: (
            -int(table.loc[group].sum()),
            hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
        ),
    )

    for group in groups:
        contribution = table.loc[group]

        def objective(candidate: str) -> float:
            error = 0.0
            for split in ratios:
                value = totals[split] + contribution if split == candidate else totals[split]
                denominator = targets[split].clip(lower=1)
                error += float((((value - targets[split]) / denominator) ** 2).sum())
            return error

        chosen = min(ratios, key=lambda split: (objective(split), split))
        assignment[str(group)] = chosen
        totals[chosen] += contribution

    frame["split"] = frame["split_group"].astype(str).map(assignment)
    if frame["split"].isna().any():
        raise ValueError("Unassigned Link-Lives split groups")
    overlap: dict[str, int] = {}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        left = set(frame.loc[frame["split"].eq(a), "split_group"].astype(str))
        right = set(frame.loc[frame["split"].eq(b), "split_group"].astype(str))
        overlap[f"{a}_{b}"] = len(left & right)
        if left & right:
            raise ValueError(f"Parish leakage between {a} and {b}")

    report = {
        "seed": seed,
        "ratios": ratios,
        "group_counts": {
            str(split): int(sum(value == split for value in assignment.values()))
            for split in ratios
        },
        "row_counts": {
            str(split): int(frame["split"].eq(split).sum()) for split in ratios
        },
        "source_pair_rows": {
            str(split): {
                str(pair): int(count)
                for pair, count in totals[split].astype(int).items()
            }
            for split in ratios
        },
        "group_overlap": overlap,
    }
    return frame.drop(columns=["_source_pair"]), report


def _block_value(value: object) -> str:
    return clean(value)


def _year_value(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not YEAR_RE.match(text):
        return None
    return int(text)


def _candidate_index(
    target: pd.DataFrame, geography: str
) -> dict[tuple[str, str, int], list[int]]:
    index: dict[tuple[str, str, int], list[int]] = {}
    for row_index, row in target.iterrows():
        year = _year_value(row["birth_year"])
        place = _block_value(row[geography])
        sex = _block_value(row["sex"])
        if year is None or not place:
            continue
        index.setdefault((place, sex, year), []).append(int(row_index))
    return index


def build_hard_negative_pairs(
    positive_pairs: pd.DataFrame,
    extracted_root: Path,
    negatives_per_positive: int = 5,
    birth_year_tolerance: int = 2,
    allow_county_fallback: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build deterministic, plausible ALA alternatives without test outcomes."""

    batches: list[pd.DataFrame] = []
    candidate_counts: list[int] = []
    short_queries = 0

    for (source1, source2), positives in positive_pairs.groupby(["source1", "source2"]):
        target = load_ala_census(extracted_root, int(source2)).reset_index(drop=True)
        parish_index = _candidate_index(target, "residence_parish")
        county_index = (
            _candidate_index(target, "residence_county")
            if allow_county_fallback
            else {}
        )
        target_by_index = target.to_dict("index")
        rows: list[dict[str, Any]] = []

        for positive in positives.to_dict("records"):
            birth_year = _year_value(positive.get("left_birth_year"))
            sex = _block_value(positive.get("left_sex"))
            parish = _block_value(positive.get("left_residence_parish"))
            county = _block_value(positive.get("left_residence_county"))
            candidate_indices: list[int] = []
            if birth_year is not None:
                for year in range(
                    birth_year - birth_year_tolerance,
                    birth_year + birth_year_tolerance + 1,
                ):
                    candidate_indices.extend(parish_index.get((parish, sex, year), []))
                if allow_county_fallback and len(candidate_indices) < negatives_per_positive:
                    for year in range(
                        birth_year - birth_year_tolerance,
                        birth_year + birth_year_tolerance + 1,
                    ):
                        candidate_indices.extend(county_index.get((county, sex, year), []))

            gold_id = int(positive["id2"])
            unique_indices = sorted(
                {
                    index
                    for index in candidate_indices
                    if int(target_by_index[index]["record_id"]) != gold_id
                }
            )
            scored = []
            for index in unique_indices:
                candidate = target_by_index[index]
                score = (
                    2.0 * jaccard(positive.get("left_name"), candidate.get("name"))
                    + 0.5
                    * float(
                        _block_value(positive.get("left_birth_place"))
                        == _block_value(candidate.get("birth_place"))
                        and bool(_block_value(candidate.get("birth_place")))
                    )
                    + 0.25
                    * jaccard(
                        positive.get("left_occupation"), candidate.get("occupation")
                    )
                )
                scored.append((score, int(candidate["record_id"]), index))
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = scored[:negatives_per_positive]
            candidate_counts.append(len(unique_indices))
            if len(chosen) < negatives_per_positive:
                short_queries += 1

            for rank, (_, record_id, index) in enumerate(chosen, start=1):
                candidate = target_by_index[index]
                row = dict(positive)
                for field in PAIR_FIELDS:
                    row[f"right_{field}"] = candidate.get(field)
                row["id2"] = record_id
                row["right_id"] = f"{int(source2)}:{record_id}"
                row["pair_id"] = f"{row['left_id']}#{row['right_id']}"
                row["label"] = 0
                row["negative_rank"] = rank
                rows.append(row)
        batches.append(pd.DataFrame(rows))

    negatives = pd.concat(batches, ignore_index=True)
    if negatives["pair_id"].duplicated().any():
        raise ValueError("Duplicate hard-negative pair ids")
    report = {
        "negatives_per_positive_requested": negatives_per_positive,
        "birth_year_tolerance": birth_year_tolerance,
        "allow_county_fallback": allow_county_fallback,
        "negative_rows": int(len(negatives)),
        "queries_with_fewer_than_requested": int(short_queries),
        "candidate_pool_min": int(min(candidate_counts, default=0)),
        "candidate_pool_median": float(pd.Series(candidate_counts).median()),
        "candidate_pool_max": int(max(candidate_counts, default=0)),
    }
    return negatives, report


def enforce_record_disjoint_negatives(
    positives: pd.DataFrame, negatives: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Give every negative target record one split owner and drop cross-split uses."""

    positive_owner: dict[str, str] = {}
    for row in positives[["split", "left_id", "right_id"]].itertuples(index=False):
        for record_id in (str(row.left_id), str(row.right_id)):
            previous = positive_owner.setdefault(record_id, str(row.split))
            if previous != str(row.split):
                raise ValueError(
                    f"Positive record crosses splits: {record_id} in {previous}/{row.split}"
                )

    split_order = {"train": 0, "valid": 1, "test": 2}
    negative_owner: dict[str, str] = {}
    counts = (
        negatives.groupby(["right_id", "split"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    for record_id, group in counts.groupby("right_id", sort=False):
        key = str(record_id)
        if key in positive_owner:
            negative_owner[key] = positive_owner[key]
            continue
        ranked = sorted(
            (
                (-int(row.rows), split_order.get(str(row.split), 999), str(row.split))
                for row in group.itertuples(index=False)
            )
        )
        negative_owner[key] = ranked[0][2]

    owner = negatives["right_id"].astype(str).map(negative_owner)
    kept = negatives.loc[owner.eq(negatives["split"].astype(str))].copy()
    dropped = int(len(negatives) - len(kept))

    records = {
        split: set(frame["left_id"].astype(str)) | set(frame["right_id"].astype(str))
        for split, frame in pd.concat([positives, kept]).groupby("split")
    }
    overlap: dict[str, int] = {}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        current = records.get(a, set()) & records.get(b, set())
        overlap[f"{a}_{b}"] = len(current)
        if current:
            raise ValueError(f"Record leakage after ownership filter: {a}/{b}")

    report = {
        "input_negative_rows": int(len(negatives)),
        "kept_negative_rows": int(len(kept)),
        "dropped_cross_split_rows": dropped,
        "positive_owned_records": int(len(positive_owner)),
        "negative_target_owners": int(len(negative_owner)),
        "record_overlap": overlap,
    }
    return kept, report
