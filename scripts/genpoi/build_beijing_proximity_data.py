#!/usr/bin/env python3
"""Build leakage-free calibration data for Beijing-adapted GenPOI SSP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.genpoi.beijing_ssp import USEFUL_PREFIX_DEPTHS  # noqa: E402
from poi_gr.methods.genpoi.proximity import (  # noqa: E402
    GenPoiProximityError,
    extract_proximity_example,
)
from poi_gr.pid.trie import sha256_file  # noqa: E402


SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("query", pa.string()),
        ("user_gid", pa.string()),
        ("target_gid", pa.string()),
        ("label", pa.int8()),
    ]
)


class BeijingProximityDataError(RuntimeError):
    """Raised when the leakage-free calibration contract is violated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "复用既有 GenPOI Train Parquet，并从完整 Validation 中排除固定评测集，"
            "构建北京 SSP 独立校准数据。"
        )
    )
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--reserved-eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise BeijingProximityDataError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BeijingProximityDataError(f"{name} JSON 非法") from error
    if not isinstance(value, dict):
        raise BeijingProximityDataError(f"{name} 必须是 object")
    return value


def load_base_train(base_data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = base_data_dir / "manifest.json"
    manifest = load_json(manifest_path, "基础邻近数据 manifest")
    if manifest.get("schema_version") != "genpoi-proximity-data-v1":
        raise BeijingProximityDataError("基础邻近数据 schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise BeijingProximityDataError("基础邻近数据状态不是 completed")
    train = manifest.get("outputs", {}).get("train")
    if not isinstance(train, dict):
        raise BeijingProximityDataError("基础邻近数据缺少 train")
    train_path = Path(train.get("output_file", "")).resolve()
    if not train_path.is_file() or sha256_file(train_path) != train.get("output_sha256"):
        raise BeijingProximityDataError("基础 Train Parquet 不存在或 SHA256 不一致")
    if pq.ParquetFile(train_path).metadata.num_rows != train.get("rows"):
        raise BeijingProximityDataError("基础 Train Parquet 行数不一致")
    return manifest, dict(train)


def load_reserved_sample_ids(path: Path) -> tuple[set[str], str, int]:
    sample_ids: set[str] = set()
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise BeijingProximityDataError(
                    f"固定评测集 {path.name}:{line_number} JSON 非法"
                ) from error
            sample_id = record.get("sample_id") if isinstance(record, dict) else None
            if not isinstance(sample_id, str) or not sample_id:
                raise BeijingProximityDataError(
                    f"固定评测集 {path.name}:{line_number} 缺少 sample_id"
                )
            if sample_id in sample_ids:
                raise BeijingProximityDataError(
                    f"固定评测集 sample_id 重复：{sample_id}"
                )
            sample_ids.add(sample_id)
            rows += 1
    if rows == 0:
        raise BeijingProximityDataError("固定评测集为空")
    return sample_ids, digest.hexdigest(), rows


def write_calibration(
    source: Path,
    destination: Path,
    *,
    reserved_ids: set[str],
    batch_rows: int,
) -> tuple[dict[str, Any], set[str]]:
    temporary = destination.with_name(f".{destination.name}.tmp")
    writer = pq.ParquetWriter(
        temporary,
        SCHEMA,
        compression="zstd",
        use_dictionary=("query", "user_gid", "target_gid"),
    )
    buffers: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
    source_digest = hashlib.sha256()
    label_counts: Counter[int] = Counter()
    found_reserved: set[str] = set()
    reserved_occurrences = 0
    source_rows = 0
    rows = 0

    def flush() -> None:
        if not buffers["label"]:
            return
        writer.write_table(pa.Table.from_pydict(buffers, schema=SCHEMA))
        for values in buffers.values():
            values.clear()

    try:
        with source.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                source_digest.update(raw_line)
                source_rows += 1
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise BeijingProximityDataError(
                        f"{source.name}:{line_number} JSON 非法"
                    ) from error
                if not isinstance(record, dict):
                    raise BeijingProximityDataError(
                        f"{source.name}:{line_number} 不是 object"
                    )
                try:
                    example = extract_proximity_example(record, expected_split="valid")
                except GenPoiProximityError as error:
                    raise BeijingProximityDataError(
                        f"{source.name}:{line_number}：{error}"
                    ) from error
                if example.sample_id in reserved_ids:
                    found_reserved.add(example.sample_id)
                    reserved_occurrences += 1
                    continue
                buffers["sample_id"].append(example.sample_id)
                buffers["query"].append(example.query)
                buffers["user_gid"].append("".join(example.user_gid))
                buffers["target_gid"].append("".join(example.target_gid))
                buffers["label"].append(example.label)
                label_counts[example.label] += 1
                rows += 1
                if rows % batch_rows == 0:
                    flush()
                    print(f"[calibration] {rows:,} rows", file=sys.stderr, flush=True)
        flush()
    finally:
        writer.close()
    if found_reserved != reserved_ids or reserved_occurrences != len(reserved_ids):
        temporary.unlink(missing_ok=True)
        missing = len(reserved_ids - found_reserved)
        raise BeijingProximityDataError(
            "固定评测集未能从完整 Validation 中被精确排除："
            f"missing={missing}, occurrences={reserved_occurrences}"
        )
    if rows == 0 or source_rows != rows + reserved_occurrences:
        temporary.unlink(missing_ok=True)
        raise BeijingProximityDataError("Calibration 行数守恒失败")
    os.replace(temporary, destination)
    return (
        {
            "source_file": str(source),
            "source_sha256": source_digest.hexdigest(),
            "source_rows": source_rows,
            "rows": rows,
            "label_counts": {
                str(index): label_counts[index] for index in range(7)
            },
            "output_file": str(destination),
            "output_sha256": sha256_file(destination),
            "output_bytes": destination.stat().st_size,
        },
        found_reserved,
    )


def main() -> int:
    args = parse_args()
    try:
        if args.batch_rows <= 0:
            raise BeijingProximityDataError("--batch-rows 必须为正整数")
        base_data_dir = resolve(args.base_data_dir)
        source = resolve(args.calibration_source)
        reserved_file = resolve(args.reserved_eval_file)
        output_dir = resolve(args.output_dir)
        if not source.is_file() or not reserved_file.is_file():
            raise BeijingProximityDataError("Calibration Source 或固定评测集不存在")
        output_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = output_dir / "calibration.parquet"
        manifest_path = output_dir / "manifest.json"
        if calibration_path.exists() or manifest_path.exists():
            raise BeijingProximityDataError("输出已存在，请使用新的输出目录")

        base_manifest, train = load_base_train(base_data_dir)
        reserved_ids, reserved_sha256, reserved_rows = load_reserved_sample_ids(
            reserved_file
        )
        calibration, found_reserved = write_calibration(
            source,
            calibration_path,
            reserved_ids=reserved_ids,
            batch_rows=args.batch_rows,
        )
        manifest = {
            "schema_version": "genpoi-beijing-proximity-data-v1",
            "status": "completed",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "method": "ordinal_safe_gid_prefix",
            "useful_prefix_depths": list(USEFUL_PREFIX_DEPTHS),
            "label_definition": (
                "longest common prefix length of current user GID and target POI GID"
            ),
            "train": {
                **train,
                "reused_from_manifest": str(base_data_dir / "manifest.json"),
                "reused_from_manifest_sha256": sha256_file(
                    base_data_dir / "manifest.json"
                ),
            },
            "calibration": calibration,
            "reserved_evaluation": {
                "file": str(reserved_file),
                "sha256": reserved_sha256,
                "rows": reserved_rows,
                "unique_sample_ids": len(reserved_ids),
                "matched_and_excluded_sample_ids": len(found_reserved),
                "calibration_overlap_count": 0,
            },
            "base_label_definition": base_manifest.get("label_definition"),
        }
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (BeijingProximityDataError, OSError, ValueError, KeyError) as error:
        print(f"构建失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
