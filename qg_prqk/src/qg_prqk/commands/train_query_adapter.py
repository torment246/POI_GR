"""Train a query adapter from a prepared bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qg_prqk.adapters.training import AdapterTrainingError, train_adapter_bundle
from qg_prqk.config import QGPRQKConfigError, load_config
from qg_prqk.adapters.model import QueryAdapterError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练 QG-PRQK Query-only residual Adapter。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="实验根目录。")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", "--sample-size", dest="limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(
            args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            resume=True if args.resume else None,
            overwrite=True if args.overwrite else None,
        )
        manifest = train_adapter_bundle(
            config,
            training_data=args.training_data,
            output_dir=config.paths.output_dir / "query_adapter",
            limit=args.limit,
            device=args.device,
            batch_size=args.batch_size,
        )
    except (
        AdapterTrainingError,
        QGPRQKConfigError,
        QueryAdapterError,
        OSError,
        ValueError,
    ) as error:
        print(f"QG-PRQK Adapter failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
