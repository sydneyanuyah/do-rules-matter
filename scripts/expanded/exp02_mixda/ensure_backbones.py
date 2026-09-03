#!/usr/bin/env python3
"""Idempotently ensure the four requested locked backbones are configured."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REQUIRED = [
    {
        "id": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "pooling": "mean",
        "normalize": True,
        "symmetric_prefix": "query: ",
        "max_length": 512,
    },
    {
        "id": "jinaai/jina-embeddings-v3",
        "revision": "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
        "pooling": "mean",
        "normalize": True,
        "symmetric_prefix": "",
        "max_length": 8192,
        "trust_remote_code": True,
    },
    {
        "id": "sentence-transformers/all-roberta-large-v1",
        "revision": "cf74d8acd4f198de950bf004b262e6accfed5d2c",
        "pooling": "mean",
        "normalize": True,
        "symmetric_prefix": "",
        "max_length": 256,
    },
    {
        "id": "sentence-transformers/bert-base-nli-mean-tokens",
        "revision": "160a52b38a51ae87295ec3eabcf11755e5d27a8d",
        "pooling": "mean",
        "normalize": True,
        "symmetric_prefix": "",
        "max_length": 256,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text())
    configured = {item["id"]: item for item in payload["frozen_backbones"]}
    for required in REQUIRED:
        existing = configured.get(required["id"])
        if existing is not None and existing.get("revision") != required["revision"]:
            raise ValueError(
                f"Revision conflict for {required['id']}: "
                f"{existing.get('revision')} != {required['revision']}"
            )
        if existing is None:
            payload["frozen_backbones"].append(required)
    args.config.write_text(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
