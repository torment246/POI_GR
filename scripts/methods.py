#!/usr/bin/env python3
"""Inspect and validate current, baseline, and innovation method contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.catalog import (  # noqa: E402
    MethodCatalogError,
    load_method_catalog,
    method_spec_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查看并校验当前主线、论文 baseline 和后续创新方法契约。",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/methods"),
        help="方法配置根目录；相对路径相对于仓库根目录。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="列出全部方法及四阶段准备状态。")

    show_parser = subparsers.add_parser("show", help="显示一个方法的完整契约。")
    show_parser.add_argument("method_id", help="方法 ID，例如 tiger。")

    validate_parser = subparsers.add_parser("validate", help="校验方法配置和路径。")
    validate_parser.add_argument(
        "method_id",
        nargs="?",
        help="可选方法 ID；不指定时校验全部方法。",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(args: argparse.Namespace):
    return load_method_catalog(_resolve(args.config_root), PROJECT_ROOT)


def _get_method(catalog, method_id: str):
    try:
        return catalog[method_id]
    except KeyError as error:
        available = "、".join(catalog)
        raise MethodCatalogError(
            f"未知方法 {method_id}；可选方法：{available}"
        ) from error


def _list_methods(catalog) -> None:
    header = (
        "method_id",
        "group",
        "status",
        "data",
        "identifier",
        "train",
        "evaluate",
    )
    rows = [
        (
            spec.method_id,
            spec.group,
            spec.status,
            *(spec.stage(name).status for name in ("data", "identifier", "train", "evaluate")),
        )
        for spec in catalog.values()
    ]
    widths = [
        max(len(header[index]), *(len(row[index]) for row in rows))
        for index in range(len(header))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(header)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    args = parse_args()
    try:
        catalog = _load(args)
        if args.command == "list":
            _list_methods(catalog)
        elif args.command == "show":
            spec = _get_method(catalog, args.method_id)
            print(
                json.dumps(
                    method_spec_payload(spec, PROJECT_ROOT),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "validate":
            if args.method_id:
                spec = _get_method(catalog, args.method_id)
                print(f"方法配置校验通过：{spec.method_id}")
            else:
                print("全部方法配置校验通过：{}".format("、".join(catalog)))
        else:
            raise AssertionError(args.command)
    except MethodCatalogError as error:
        print(f"方法配置错误：{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
