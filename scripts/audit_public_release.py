#!/usr/bin/env python3
"""Fail when a release contains internal identifiers, secrets, or oversized files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {
    ".cfg", ".cff", ".csv", ".env", ".ini", ".json", ".md", ".py",
    ".sh", ".tex", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
FORBIDDEN = {
    "company identifiers": re.compile(r"ancestry|rarecompare|rarebias|genco|redshift", re.I),
    "cloud account identifiers": re.compile(r"251465955583|613686586219|l3-datascience|u-datascience|d-dqny48k3gf6i", re.I),
    "internal storage": re.compile(r"scratch_sydney|ds-dense-emb-train|datascience-recordlinking", re.I),
    "absolute user paths": re.compile(r"/Users/[^/\s]+/|/home/sagemaker-user/|/mnt/sagemaker-nvme/", re.I),
    "AWS access keys": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private keys": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "presigned tokens": re.compile(r"(?:token=eyJ|X-Amz-Signature=)", re.I),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--max-mib", type=int, default=95)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(root)
        if rel.as_posix() == "scripts/audit_public_release.py":
            continue
        if any(part.lower() in {"experiment_10", "experiment_11"} for part in rel.parts):
            failures.append(f"excluded experiment present: {rel}")
        if path.stat().st_size > args.max_mib * 1024 * 1024:
            failures.append(f"oversized file: {rel} ({path.stat().st_size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{label}: {rel}")

    if failures:
        print("PUBLIC RELEASE AUDIT FAILED")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("PUBLIC RELEASE AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
