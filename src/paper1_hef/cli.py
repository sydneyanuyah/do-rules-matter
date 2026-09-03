from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_dataset, validate_splits
from .cross_encoder import run_cross_encoder
from .embeddings import encode_dataset
from .exp01 import run_exp01_dataset
from .exp01_jina import run_exp01_jina
from .exp02 import (
    build_candidate_pool,
    run_exp02_dataset,
    score_exp02_cross_encoder,
    score_exp02_roberta,
)
from .exp03 import finalize_exp03, run_exp03
from .exp05 import run_exp05_dataset
from .pipeline import load_config, run_dataset, run_masking
from .stacking import run_pointwise_stacking


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(prog="paper1-hef")
    out.add_argument("--project-root", type=Path, default=Path.cwd())
    out.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    sub = out.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--dataset", default="all")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--dataset", default="all")
    pilot.add_argument("--bootstrap-replicates", type=int)
    masking = sub.add_parser("mask")
    masking.add_argument("--dataset", required=True)
    masking.add_argument("--probabilities", default="0,0.1,0.3,0.5,0.7")
    encode = sub.add_parser("encode")
    encode.add_argument("--dataset", required=True)
    encode.add_argument("--model", required=True)
    encode.add_argument("--revision")
    encode.add_argument("--batch-size", type=int, default=128)
    encode.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    exp01 = sub.add_parser("exp01")
    exp01.add_argument("--dataset", default="exp01_all")
    exp01.add_argument("--model", default="intfloat/e5-base-v2")
    exp01.add_argument("--revision")
    exp01.add_argument("--bootstrap-replicates", type=int)
    cross_encoder = sub.add_parser("cross-encoder")
    cross_encoder.add_argument("--dataset", required=True)
    cross_encoder.add_argument("--seed", type=int, required=True)
    cross_encoder.add_argument("--batch-size", type=int, default=32)
    cross_encoder.add_argument("--device", default="cuda")
    exp01_jina = sub.add_parser("exp01-jina")
    exp01_jina.add_argument("--dataset", required=True)
    exp01_jina.add_argument("--batch-size", type=int, default=0)
    exp01_jina.add_argument("--target-memory-gib", type=float, default=20.0)
    exp01_jina.add_argument("--max-batch-size", type=int, default=4096)
    exp01_jina.add_argument("--device", default="cuda")
    exp02_pool = sub.add_parser("exp02-pool")
    exp02_pool.add_argument("--dataset", required=True)
    exp02_pool.add_argument("--model", default="intfloat/e5-base-v2")
    exp02_pool.add_argument("--revision")
    exp02_pool.add_argument("--batch-size", type=int, default=128)
    exp02_pool.add_argument("--device", default="cuda")
    exp02_rank = sub.add_parser("exp02-rank")
    exp02_rank.add_argument("--dataset", required=True)
    exp02_rank.add_argument("--model", default="intfloat/e5-base-v2")
    exp02_rank.add_argument("--revision")
    exp02_cross = sub.add_parser("exp02-cross")
    exp02_cross.add_argument("--dataset", required=True)
    exp02_cross.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 enables automatic GPU-memory calibration",
    )
    exp02_cross.add_argument("--target-memory-gib", type=float, default=20.0)
    exp02_cross.add_argument("--max-batch-size", type=int, default=4096)
    exp02_cross.add_argument("--device", default="cuda")
    exp02_roberta = sub.add_parser("exp02-roberta")
    exp02_roberta.add_argument("--dataset", required=True)
    exp02_roberta.add_argument("--batch-size", type=int, default=0)
    exp02_roberta.add_argument("--target-memory-gib", type=float, default=20.0)
    exp02_roberta.add_argument("--max-batch-size", type=int, default=4096)
    exp02_roberta.add_argument("--device", default="cuda")
    exp03 = sub.add_parser("exp03")
    exp03.add_argument("--dataset", default="exp01_all")
    exp03.add_argument("--model", default="intfloat/e5-base-v2")
    exp03.add_argument("--revision")
    exp03.add_argument("--bootstrap-replicates", type=int)
    exp05 = sub.add_parser("exp05")
    exp05.add_argument("--dataset", default="exp01_all")
    exp05.add_argument("--model", default="intfloat/e5-base-v2")
    exp05.add_argument("--revision")
    exp05.add_argument("--bootstrap-replicates", type=int)
    stacking = sub.add_parser("stack")
    stacking.add_argument("--input", type=Path, required=True)
    stacking.add_argument("--output", type=Path, required=True)
    stacking.add_argument("--split-col", required=True)
    stacking.add_argument("--query-col", required=True)
    stacking.add_argument("--candidate-col", required=True)
    stacking.add_argument("--label-col", default="label")
    stacking.add_argument("--features", nargs="+", required=True)
    stacking.add_argument("--seed", type=int, default=20260725)
    return out


def main() -> None:
    args = parser().parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    requested = getattr(args, "dataset", "all")
    if requested == "all":
        names = list(config["datasets"])
    elif requested in config.get("dataset_groups", {}):
        names = list(config["dataset_groups"][requested])
    else:
        names = [requested]
    if args.command == "validate":
        reports = {}
        for name in names:
            reports[name] = validate_splits(
                load_dataset(root / config["project"]["data_root"], config["datasets"][name]),
                enforce_offer_disjoint=name == "wdc_80_medium_unseen",
                report_offer_overlap=config["datasets"][name]["adapter"] == "wdc",
                enforce_record_disjoint=config["datasets"][name]["adapter"] == "link_lives",
            )
        print(json.dumps(reports, indent=2, sort_keys=True))
    elif args.command == "pilot":
        for name in names:
            print(run_dataset(root, config, name, args.bootstrap_replicates))
    elif args.command == "mask":
        probabilities = [float(value) for value in args.probabilities.split(",")]
        print(run_masking(root, config, args.dataset, probabilities))
    elif args.command == "encode":
        revision = args.revision or next(
            item["revision"] for item in config["frozen_backbones"] if item["id"] == args.model
        )
        print(
            encode_dataset(
                root, config, args.dataset, args.model, revision, args.batch_size, args.device
            )
        )
    elif args.command == "exp01":
        revision = args.revision or next(
            item["revision"] for item in config["frozen_backbones"] if item["id"] == args.model
        )
        for name in names:
            print(
                run_exp01_dataset(
                    root,
                    config,
                    name,
                    args.model,
                    revision,
                    args.bootstrap_replicates,
                )
            )
    elif args.command == "cross-encoder":
        print(
            run_cross_encoder(
                root,
                config,
                args.dataset,
                args.seed,
                args.batch_size,
                args.device,
            )
        )
    elif args.command == "exp01-jina":
        print(
            run_exp01_jina(
                root,
                config,
                args.dataset,
                args.device,
                args.batch_size,
                args.target_memory_gib,
                args.max_batch_size,
            )
        )
    elif args.command == "exp02-pool":
        revision = args.revision or next(
            item["revision"] for item in config["frozen_backbones"] if item["id"] == args.model
        )
        print(
            build_candidate_pool(
                root, config, args.dataset, args.model, revision, args.device, args.batch_size
            )
        )
    elif args.command == "exp02-rank":
        revision = args.revision or next(
            item["revision"] for item in config["frozen_backbones"] if item["id"] == args.model
        )
        print(run_exp02_dataset(root, config, args.dataset, args.model, revision))
    elif args.command == "exp02-cross":
        print(
            score_exp02_cross_encoder(
                root,
                config,
                args.dataset,
                args.device,
                args.batch_size,
                args.target_memory_gib,
                args.max_batch_size,
            )
        )
    elif args.command == "exp02-roberta":
        print(
            score_exp02_roberta(
                root,
                config,
                args.dataset,
                args.device,
                args.batch_size,
                args.target_memory_gib,
                args.max_batch_size,
            )
        )
    elif args.command == "exp03":
        revision = args.revision or next(
            item["revision"]
            for item in config["frozen_backbones"]
            if item["id"] == args.model
        )
        for name in names:
            print(
                run_exp03(
                    root,
                    config,
                    name,
                    args.model,
                    revision,
                    args.bootstrap_replicates,
                )
            )
        print(finalize_exp03(root, names))
    elif args.command == "exp05":
        revision = args.revision or next(
            item["revision"]
            for item in config["frozen_backbones"]
            if item["id"] == args.model
        )
        for name in names:
            print(
                run_exp05_dataset(
                    root,
                    config,
                    name,
                    args.model,
                    revision,
                    args.bootstrap_replicates,
                )
            )
    elif args.command == "stack":
        input_path = args.input if args.input.is_absolute() else root / args.input
        output_path = args.output if args.output.is_absolute() else root / args.output
        print(
            run_pointwise_stacking(
                input_path,
                output_path,
                args.split_col,
                args.query_col,
                args.candidate_col,
                args.label_col,
                args.features,
                args.seed,
            )
        )


if __name__ == "__main__":
    main()
