#!/usr/bin/env python3
"""Stream raw GNPR POI features into bounded Beijing SID inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr.sid_input import (  # noqa: E402
    DEFAULT_USER_HASH_BUCKETS,
    USER_HASH_PERSON,
    GnprSidInput,
    build_feature_dimensions,
    prepare_gnpr_sid_input,
)


SCHEMA_VERSION = "gnpr-sid-input-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "流式读取 004 GNPR 原始 POI 特征，仅保留有交互 POI，并将 Top10 "
            "passenger_id 固定哈希到有限桶中。"
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="004 本地输出目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="不存在的输出目录。")
    parser.add_argument(
        "--user-hash-buckets",
        type=int,
        default=DEFAULT_USER_HASH_BUCKETS,
        help="用户 Feature Hash 桶数，默认 8192。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32768,
        help="单次读取的 Parquet 行数。",
    )
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=None,
        help="仅用于 smoke，最多扫描的原始行数。",
    )
    parser.add_argument(
        "--expected-source-pois",
        type=int,
        default=None,
        help="可选的原始 POI 总行数门禁。",
    )
    parser.add_argument(
        "--expected-output-pois",
        type=int,
        default=None,
        help="可选的有交互 POI 输出行数门禁。",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parquet_files(path: Path) -> list[Path]:
    files = sorted(item for item in path.rglob("*.parquet") if item.is_file())
    if not files:
        raise ValueError(f"没有找到 Parquet 分片：{path}")
    return files


def _parquet_row_count(files: Iterable[Path]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def _arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("poi_id", pa.string(), nullable=False),
            pa.field("category_index", pa.int64(), nullable=False),
            pa.field("region_index", pa.int64(), nullable=False),
            pa.field("top_visit_hours", pa.list_(pa.int32()), nullable=False),
            pa.field("user_hash_indices", pa.list_(pa.int32()), nullable=False),
            pa.field("interaction_count", pa.int64(), nullable=False),
        ]
    )


def _as_dict(row: GnprSidInput) -> dict[str, object]:
    return {
        "poi_id": row.poi_id,
        "category_index": row.category_index,
        "region_index": row.region_index,
        "top_visit_hours": list(row.top_visit_hours),
        "user_hash_indices": list(row.user_hash_indices),
        "interaction_count": row.interaction_count,
    }


def prepare_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    user_hash_buckets: int,
    batch_size: int,
    max_input_rows: int | None = None,
    expected_source_pois: int | None = None,
    expected_output_pois: int | None = None,
) -> dict[str, object]:
    """Stream and atomically write one bounded GNPR SID-input dataset."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    input_dir = _resolve(input_dir)
    output_dir = _resolve(output_dir)
    staging_dir = output_dir.with_name(output_dir.name + ".building")
    if batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    if max_input_rows is not None and max_input_rows <= 0:
        raise ValueError("max-input-rows 必须大于 0")
    if output_dir.exists():
        raise ValueError(f"输出目录已存在：{output_dir}")
    if staging_dir.exists():
        raise ValueError(f"临时目录已存在：{staging_dir}")

    feature_files = _parquet_files(input_dir / "poi_features.parquet")
    category_files = _parquet_files(input_dir / "category_vocab.parquet")
    region_files = _parquet_files(input_dir / "region_vocab.parquet")
    source_poi_count = _parquet_row_count(feature_files)
    if expected_source_pois is not None and source_poi_count != expected_source_pois:
        raise ValueError(
            f"原始 POI 行数不一致：{source_poi_count} != {expected_source_pois}"
        )

    dimensions = build_feature_dimensions(
        category_count=_parquet_row_count(category_files),
        region_count=_parquet_row_count(region_files),
        user_hash_buckets=user_hash_buckets,
    )
    output_parts_dir = staging_dir / "poi_sid_inputs.parquet"
    output_parts_dir.mkdir(parents=True)
    schema = _arrow_schema()
    columns = [
        "poi_id",
        "category_index",
        "region_index",
        "top_visit_hours",
        "top_visitor_ids",
        "interaction_count",
    ]
    scanned_count = 0
    output_count = 0
    filtered_cold_count = 0
    raw_visitor_count = 0
    hashed_visitor_count = 0
    output_part_count = 0

    for source_path in feature_files:
        if max_input_rows is not None and scanned_count >= max_input_rows:
            break
        writer = None
        try:
            parquet_file = pq.ParquetFile(source_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                remaining = (
                    None
                    if max_input_rows is None
                    else max_input_rows - scanned_count
                )
                if remaining is not None and remaining <= 0:
                    break
                if remaining is not None and batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                scanned_count += batch.num_rows

                output_rows: list[dict[str, object]] = []
                for raw_row in batch.to_pylist():
                    prepared = prepare_gnpr_sid_input(raw_row, dimensions=dimensions)
                    if prepared is None:
                        filtered_cold_count += 1
                        continue
                    raw_visitor_count += len(raw_row["top_visitor_ids"])
                    hashed_visitor_count += len(prepared.user_hash_indices)
                    output_rows.append(_as_dict(prepared))
                if not output_rows:
                    continue
                if writer is None:
                    output_path = output_parts_dir / f"part-{output_part_count:05d}.parquet"
                    writer = pq.ParquetWriter(output_path, schema=schema, compression="snappy")
                    output_part_count += 1
                table = pa.Table.from_pylist(output_rows, schema=schema)
                writer.write_table(table)
                output_count += table.num_rows
        finally:
            if writer is not None:
                writer.close()

    if output_count + filtered_cold_count != scanned_count:
        raise RuntimeError("输出行数与冷 POI 过滤数无法还原扫描行数")
    if expected_output_pois is not None and output_count != expected_output_pois:
        raise ValueError(
            f"有交互 POI 行数不一致：{output_count} != {expected_output_pois}"
        )
    if output_count == 0:
        raise ValueError("过滤后没有可用于 SID 构建的 POI")

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "method": "GNPR-SID Beijing adapted",
        "input_dir": str(input_dir),
        "output": "poi_sid_inputs.parquet",
        "catalog_filter": "interaction_count > 0",
        "user_encoding": {
            "type": "blake2b64_mod",
            "hash_person": USER_HASH_PERSON.decode("ascii"),
            "buckets": user_hash_buckets,
            "deduplicate_within_poi": True,
        },
        "dimensions": {
            "category": dimensions.category_dim,
            "region": dimensions.region_dim,
            "time": dimensions.time_dim,
            "user_hash": dimensions.user_hash_dim,
            "total": dimensions.total_dim,
        },
        "stats": {
            "source_poi_count": source_poi_count,
            "scanned_poi_count": scanned_count,
            "output_behavior_poi_count": output_count,
            "filtered_cold_poi_count": filtered_cold_count,
            "output_part_count": output_part_count,
            "raw_top_visitor_count": raw_visitor_count,
            "hashed_active_visitor_count": hashed_visitor_count,
            "within_poi_hash_collision_count": raw_visitor_count
            - hashed_visitor_count,
        },
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging_dir / "_SUCCESS").touch()
    staging_dir.rename(output_dir)
    return manifest


def main() -> int:
    args = parse_args()
    try:
        manifest = prepare_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            user_hash_buckets=args.user_hash_buckets,
            batch_size=args.batch_size,
            max_input_rows=args.max_input_rows,
            expected_source_pois=args.expected_source_pois,
            expected_output_pois=args.expected_output_pois,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"GNPR SID 输入构建失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
