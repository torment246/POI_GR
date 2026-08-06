#!/usr/bin/env python3
"""Build compact query/proximity-label data for the GenPOI SSP estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.genpoi_proximity import (  # noqa: E402
    GenPoiProximityError,
    ProximityExample,
    extract_proximity_example,
)
from poi_gr.pid_trie import sha256_file  # noqa: E402


SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("query", pa.string()),
        ("user_gid", pa.string()),
        ("target_gid", pa.string()),
        ("label", pa.int8()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 GenPOI SFT JSONL 提取当前 Query，并以用户/目标 GID 最长公共前缀"
            "长度构造 0..6 地理邻近分类标签。"
        )
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def iter_examples(
    path: Path,
    *,
    split: str,
    limit: int | None,
    source_digest: Any,
) -> Iterator[ProximityExample]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                break
            source_digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise GenPoiProximityError(
                    f"{path.name}:{line_number} JSON 非法"
                ) from error
            if not isinstance(record, dict):
                raise GenPoiProximityError(f"{path.name}:{line_number} 不是 object")
            try:
                yield extract_proximity_example(record, expected_split=split)
            except GenPoiProximityError as error:
                raise GenPoiProximityError(
                    f"{path.name}:{line_number}：{error}"
                ) from error


def write_split(
    source: Path,
    destination: Path,
    *,
    split: str,
    batch_rows: int,
    limit: int | None,
) -> dict[str, Any]:
    temporary = destination.with_name(f".{destination.name}.tmp")
    writer = pq.ParquetWriter(
        temporary,
        SCHEMA,
        compression="zstd",
        use_dictionary=("query", "user_gid", "target_gid"),
    )
    buffers: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
    counts: Counter[int] = Counter()
    rows = 0
    source_digest = hashlib.sha256()

    def flush() -> None:
        if not buffers["label"]:
            return
        writer.write_table(pa.Table.from_pydict(buffers, schema=SCHEMA))
        for values in buffers.values():
            values.clear()

    try:
        for example in iter_examples(
            source,
            split=split,
            limit=limit,
            source_digest=source_digest,
        ):
            buffers["sample_id"].append(example.sample_id)
            buffers["query"].append(example.query)
            buffers["user_gid"].append("".join(example.user_gid))
            buffers["target_gid"].append("".join(example.target_gid))
            buffers["label"].append(example.label)
            counts[example.label] += 1
            rows += 1
            if rows % batch_rows == 0:
                flush()
                print(f"[{split}] {rows:,} rows", file=sys.stderr, flush=True)
        flush()
    finally:
        writer.close()
    if rows == 0:
        temporary.unlink(missing_ok=True)
        raise GenPoiProximityError(f"{split} 没有可写样本")
    os.replace(temporary, destination)
    return {
        "source_file": str(source),
        "source_sha256": source_digest.hexdigest(),
        "source_sha256_scope": "full_file" if limit is None else f"first_{rows}_rows",
        "rows": rows,
        "label_counts": {str(index): counts[index] for index in range(7)},
        "output_file": str(destination),
        "output_sha256": sha256_file(destination),
        "output_bytes": destination.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    if args.batch_rows <= 0:
        print("构建失败：--batch-rows 必须为正整数", file=sys.stderr)
        return 2
    for name in ("train_limit", "valid_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            print(f"构建失败：--{name.replace('_', '-')} 必须为正整数", file=sys.stderr)
            return 2
    train_file = resolve(args.train_file)
    valid_file = resolve(args.valid_file)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (output_dir / "train.parquet", output_dir / "valid.parquet")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() or any(path.exists() for path in outputs):
        print("构建失败：输出已存在，请使用新的输出目录", file=sys.stderr)
        return 2
    try:
        train = write_split(
            train_file,
            outputs[0],
            split="train",
            batch_rows=args.batch_rows,
            limit=args.train_limit,
        )
        valid = write_split(
            valid_file,
            outputs[1],
            split="valid",
            batch_rows=args.batch_rows,
            limit=args.valid_limit,
        )
    except (GenPoiProximityError, OSError, ValueError) as error:
        print(f"构建失败：{error}", file=sys.stderr)
        return 2
    manifest = {
        "schema_version": "genpoi-proximity-data-v1",
        "status": "completed",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "label_definition": (
            "longest common prefix length of current user GID and clicked target POI GID"
        ),
        "classes": list(range(7)),
        "query_source": "CURRENT/QUERY",
        "gid_length": 6,
        "train_limit": args.train_limit,
        "valid_limit": args.valid_limit,
        "outputs": {"train": train, "valid": valid},
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
