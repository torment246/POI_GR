"""Run or validate the fixed query-adapter engineering gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qg_prqk.adapters.retrieval import QueryAdapterRetrievalError
from qg_prqk.adapters.config import QueryAdapterConfigError, load_query_adapter_config
from qg_prqk.adapters.data import QueryAdapterDataError
from qg_prqk.adapters.gate import QueryAdapterGateError, run_query_adapter_gate, validate_query_adapter_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "运行历史 50k Query Adapter 工程 Gate：D3 Query cache、精确 Top-100、"
            "false-negative mask、Adapter 训练及 raw/adapted Gate。"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--gate",
        choices=("sample", "medium"),
        required=True,
        help="sample=1,000 条链路验收；medium=最多 50,000 条正式 Gate。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只复核已有产物的状态、隔离证据、shape/dtype 与 SHA256。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_query_adapter_config(args.config)
        manifest = (
            validate_query_adapter_gate(config, gate=args.gate)
            if args.validate_only
            else run_query_adapter_gate(config, gate=args.gate)
        )
    except (
        QueryAdapterRetrievalError,
        QueryAdapterConfigError,
        QueryAdapterDataError,
        QueryAdapterGateError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"QG-PRQK Query Adapter Gate 失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "validated" if args.validate_only else "completed",
                "phase": "P3A",
                "gate": args.gate,
                "selected_rows": manifest["selection"]["selected_rows"],
                "gate_passed": manifest["decision"]["gate_passed"],
                "query_view": manifest["decision"]["query_view"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
