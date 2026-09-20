"""Build the shared SFT vocabulary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from qg_prqk.sft.vocabulary import SftVocabError, prepare_shared_sft_vocab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为 A4 GID-parent 与 NoGID 构造共同的 Qwen3 普通原子 Token 词表。"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gid-tokens", type=Path, required=True)
    parser.add_argument("--nogid-tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = prepare_shared_sft_vocab(
            model_dir=args.model_dir,
            gid_tokens_path=args.gid_tokens,
            nogid_tokens_path=args.nogid_tokens,
            output_dir=args.output_dir,
        )
    except (SftVocabError, OSError, ValueError) as error:
        print(f"QG-PRQK 扩词表失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "original_vocab_size": payload["original_vocab_size"],
                "new_vocab_size": payload["new_vocab_size"],
                "added_token_count": payload["added_token_count"],
                "shared_by_variants": payload["shared_by_variants"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
