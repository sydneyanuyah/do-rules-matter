#!/usr/bin/env python3
"""Inventory Link-Lives Release 2 without assigning experimental labels."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd


def normalized_basename(member: str) -> str:
    return " ".join(PurePosixPath(member).name.lower().replace("_", " ").split())


def classify_member(member: str) -> str | None:
    name = normalized_basename(member)
    if name == "benchmark v1.xlsx":
        return "benchmark"
    if name.startswith("census ") and name.endswith(" std.csv"):
        return "census_harmonized"
    if name.startswith("ala census ") and name.endswith(".csv"):
        return "census_ala"
    if name.startswith("sc ") and name.endswith(".csv"):
        return "auxiliary_catalogue"
    return None


def safe_destination(root: Path, member: str) -> Path:
    relative = PurePosixPath(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {member}")
    destination = (root / Path(*relative.parts)).resolve()
    resolved_root = root.resolve()
    if resolved_root not in destination.parents and destination != resolved_root:
        raise ValueError(f"Archive member escapes output root: {member}")
    return destination


def audit_archive(archive: Path, output_root: Path) -> dict[str, object]:
    selected: dict[str, list[str]] = {}
    entries: list[dict[str, object]] = []

    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in archive member: {bad}")

        for info in bundle.infolist():
            entries.append(
                {
                    "path": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )
            category = classify_member(info.filename)
            if category:
                selected.setdefault(category, []).append(info.filename)

        benchmark_members = selected.get("benchmark", [])
        if len(benchmark_members) != 1:
            raise ValueError(
                f"Expected exactly one benchmark v1.xlsx; found {len(benchmark_members)}"
            )

        for members in selected.values():
            for member in members:
                destination = safe_destination(output_root, member)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)

    benchmark_path = safe_destination(output_root, benchmark_members[0])
    workbook = pd.ExcelFile(benchmark_path)
    benchmark_sheets: dict[str, object] = {}
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(benchmark_path, sheet_name=sheet)
        sheet_report: dict[str, object] = {
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
        }
        for decision_column in (
            "type",
            "type_linker1",
            "type_linker2",
            "type_agreement",
            "type_resolution",
            "status",
            "period",
            "method_type",
            "source1_type",
        ):
            if decision_column in frame:
                values = Counter(
                    "<NA>" if pd.isna(value) else str(value)
                    for value in frame[decision_column]
                )
                sheet_report.setdefault("decision_values", {})[decision_column] = dict(
                    values.most_common()
                )
        benchmark_sheets[sheet] = sheet_report
        if {"type", "id2"} <= set(frame.columns):
            nulls = frame["id2"].isna()
            sheet_report["id2_null_by_type"] = {
                str(label): {
                    "rows": int(len(group)),
                    "id2_null": int(nulls.loc[group.index].sum()),
                }
                for label, group in frame.groupby("type", dropna=False)
            }
        if {"source1_type", "source1", "source2"} <= set(frame.columns):
            source_pairs = (
                frame.groupby(["source1_type", "source1", "source2"], dropna=False)
                .size()
                .sort_values(ascending=False)
            )
            sheet_report["source_pair_counts"] = [
                {
                    "source1_type": str(key[0]),
                    "source1": str(key[1]),
                    "source2": str(key[2]),
                    "rows": int(value),
                }
                for key, value in source_pairs.items()
            ]

    return {
        "archive": str(archive),
        "archive_entries": entries,
        "selected_members": selected,
        "benchmark_sheets": benchmark_sheets,
        "label_mapping_status": "not_defined_pending_human_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/public_genealogy/link_lives/extracted/release_2"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/public_genealogy/link_lives/manifests/release2_schema_audit.json"
        ),
    )
    args = parser.parse_args()

    report = audit_archive(args.archive, args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.report)


if __name__ == "__main__":
    main()
