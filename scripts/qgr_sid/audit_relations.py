#!/usr/bin/env python3
"""Audit pure-numeric QGR-SID relation coverage in TIGER collision buckets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.audit import (  # noqa: E402
    QgrSidAuditError,
    audit_numeric_relations,
    validate_audit_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "严格对齐 POI 原始 JSONL 与 TIGER mapping，只对碰撞 POI 抽取"
            "稳定的纯数值关系，并统计普通/Train Query 加权覆盖率。"
        )
    )
    parser.add_argument("--poi-dir", type=Path, help="POI part-*.json 目录。")
    parser.add_argument(
        "--identifier-dir", type=Path, help="TIGER identifier epoch 输出目录。"
    )
    parser.add_argument("--output-dir", type=Path, help="审计输出目录。")
    parser.add_argument(
        "--query-aggregate-dir",
        type=Path,
        default=None,
        help="可选 Train-only Query-POI aggregate 目录。",
    )
    parser.add_argument("--value-max", type=int, default=1055)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--examples-per-relation", type=int, default=20)
    parser.add_argument(
        "--max-rows", type=int, default=None, help="仅用于 smoke 的前 N 行上限。"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证 --output-dir 的正式产物 SHA256。",
    )
    return parser.parse_args()


def _from_project_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _from_project_root(args.output_dir)
    if output_dir is None:
        print("必须提供 --output-dir", file=sys.stderr)
        return 2
    try:
        if args.validate_only:
            payload = validate_audit_output(output_dir)
        else:
            if args.poi_dir is None or args.identifier_dir is None:
                raise QgrSidAuditError(
                    "正式/Smoke 审计必须提供 --poi-dir 和 --identifier-dir"
                )
            result = audit_numeric_relations(
                poi_dir=_from_project_root(args.poi_dir),
                identifier_dir=_from_project_root(args.identifier_dir),
                output_dir=output_dir,
                query_aggregate_dir=_from_project_root(args.query_aggregate_dir),
                value_max=args.value_max,
                batch_rows=args.batch_rows,
                examples_per_relation=args.examples_per_relation,
                max_rows=args.max_rows,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            payload = {
                "status": result.metrics["status"],
                "colliding_poi_count": result.metrics["colliding_poi_count"],
                "in_vocab_relation_poi_count": result.metrics[
                    "in_vocab_relation_poi_count"
                ],
                "in_vocab_relation_poi_ratio": result.metrics[
                    "in_vocab_relation_poi_ratio"
                ],
                "output_dir": str(result.output_dir),
            }
    except (QgrSidAuditError, OSError, ValueError) as error:
        print(f"QGR-SID 关系审计失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
