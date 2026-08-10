"""Shared order splits, stable IDs, and Final-PID lookup helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq

from ..pid.dedup import sha256_file
from ..pid.geohash import GeohashPidError, tokens_to_geohash


MAPPING_COLUMNS = (
    "poi_id",
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "s1",
    "s2",
    "s3",
    "requires_dedup",
    "dedup_code",
    "final_pid_length",
    "final_pid_key",
)


class SftDataError(ValueError):
    """Raised when a shared SFT data contract is violated."""


@dataclass(frozen=True)
class TimeSplit:
    train_start: date
    train_end: date
    valid_date: date
    test_date: date

    def __post_init__(self) -> None:
        if self.train_start > self.train_end:
            raise SftDataError("train_start 不能晚于 train_end")
        if not self.train_end < self.valid_date < self.test_date:
            raise SftDataError(
                "时间切分必须满足 train_end < valid_date < test_date"
            )

    @property
    def allowed_start(self) -> date:
        return self.train_start

    @property
    def allowed_end(self) -> date:
        return self.test_date

    def split_for(self, value: date) -> str:
        if self.train_start <= value <= self.train_end:
            return "train"
        if value == self.valid_date:
            return "valid"
        if value == self.test_date:
            return "test"
        raise SftDataError(f"日期 {value.isoformat()} 不属于任何切分")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "field": "create_time",
            "train": {
                "start": self.train_start.isoformat(),
                "end": self.train_end.isoformat(),
            },
            "valid": self.valid_date.isoformat(),
            "test": self.test_date.isoformat(),
            "source_dt_usage": "verification_only",
        }


@dataclass(frozen=True)
class PidLookup:
    row_by_poi_id: dict[str, int]
    codes: np.ndarray
    dedup_codes: np.ndarray
    requires_dedup: np.ndarray
    poi_count: int


def parse_iso_date(value: str, name: str) -> date:
    """Parse one CLI date in strict YYYY-MM-DD form."""

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as error:
        raise SftDataError(f"{name} 必须为 YYYY-MM-DD：{value}") from error
    if parsed.isoformat() != value:
        raise SftDataError(f"{name} 必须为 YYYY-MM-DD：{value}")
    return parsed


def discover_order_files(orders_dir: Path) -> tuple[Path, ...]:
    """Return sorted Spark JSON part files and reject an ambiguous input."""

    orders_dir = orders_dir.resolve()
    if not orders_dir.is_dir():
        raise SftDataError(f"订单目录不存在：{orders_dir}")
    files = tuple(
        sorted(
            path
            for path in orders_dir.iterdir()
            if path.is_file() and path.name.startswith("part-")
        )
    )
    if not files:
        raise SftDataError(f"订单目录中没有 part-* 分片：{orders_dir}")
    unexpected = {
        path.name
        for path in orders_dir.iterdir()
        if path.is_file() and path.name != "_SUCCESS" and path not in files
    }
    if unexpected:
        raise SftDataError(
            "订单目录包含无法判定的数据文件："
            + ", ".join(sorted(unexpected))
        )
    return files


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise SftDataError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SftDataError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(value, dict):
        raise SftDataError(f"{name} 必须是 JSON object：{path}")
    return value


def _stable_base_pid_key(codes: Sequence[int]) -> str:
    gid = "-".join(str(int(value)) for value in codes[:6])
    sid = "-".join(str(int(value)) for value in codes[6:])
    return f"{gid}|{sid}"


def stable_final_pid_key(codes: Sequence[int], dedup_code: int) -> str:
    """Return the canonical key persisted by PID-002."""

    base_key = _stable_base_pid_key(codes)
    return base_key if dedup_code == -1 else f"{base_key}|d{dedup_code}"


def assistant_pid_content(codes: Sequence[int], dedup_code: int) -> str:
    """Format one mapped Final PID using the SFT token contract."""

    if len(codes) != 9:
        raise SftDataError("Final PID 映射必须包含九层 base PID")
    try:
        geohash = tokens_to_geohash(codes[:6])
    except GeohashPidError as error:
        raise SftDataError(str(error)) from error
    sid_values = [int(value) for value in codes[6:]]
    if any(not 0 <= value < 1024 for value in sid_values):
        raise SftDataError("SID Token 超出 [0,1024)")
    if dedup_code < -1 or dedup_code >= 512:
        raise SftDataError("Dedup Code 超出 {-1} 或 [0,512)")
    content = "".join(f"<G_{character}>" for character in geohash)
    content += "".join(
        f"<S{level}_{value}>"
        for level, value in enumerate(sid_values, start=1)
    )
    if dedup_code >= 0:
        content += f"<D_{dedup_code}>"
    return content


def stable_sample_id(relative_path: str, line_number: int) -> str:
    """Hash a unique source location into a deterministic sample ID."""

    source = f"{relative_path}:{line_number}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def load_pid_lookup(
    pid_mapping_path: Path, pid_manifest_path: Path
) -> tuple[PidLookup, dict[str, Any], str, str]:
    """Load and verify the formal unique POI-PID mapping."""

    pid_mapping_path = pid_mapping_path.resolve()
    pid_manifest_path = pid_manifest_path.resolve()
    manifest = _load_json_object(pid_manifest_path, "Final PID manifest")
    if manifest.get("schema_version") != "dedup-pid-v1":
        raise SftDataError("PID manifest schema_version 必须是 dedup-pid-v1")
    if manifest.get("status") != "completed":
        raise SftDataError("PID manifest 状态不是 completed")
    poi_count = manifest.get("poi_count")
    if isinstance(poi_count, bool) or not isinstance(poi_count, int):
        raise SftDataError("PID manifest poi_count 无效")
    mapping_spec = manifest.get("outputs", {}).get("poi_pid_mapping.parquet")
    if not isinstance(mapping_spec, dict):
        raise SftDataError("PID manifest 缺少 Parquet 映射定义")
    mapping_sha256 = sha256_file(pid_mapping_path)
    if mapping_spec.get("sha256") != mapping_sha256:
        raise SftDataError("PID 映射 SHA256 与 manifest 不一致")

    parquet_file = pq.ParquetFile(pid_mapping_path)
    if parquet_file.metadata.num_rows != poi_count:
        raise SftDataError("PID 映射行数与 manifest poi_count 不一致")
    missing_columns = set(MAPPING_COLUMNS) - set(parquet_file.schema_arrow.names)
    if missing_columns:
        raise SftDataError(
            "PID 映射缺少字段：" + ", ".join(sorted(missing_columns))
        )

    codes = np.empty((poi_count, 9), dtype=np.int32)
    dedup_codes = np.empty(poi_count, dtype=np.int16)
    requires_dedup = np.empty(poi_count, dtype=np.bool_)
    row_by_poi_id: dict[str, int] = {}
    offset = 0
    token_columns = [f"g{index}" for index in range(1, 7)] + [
        f"s{index}" for index in range(1, 4)
    ]
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_index, columns=list(MAPPING_COLUMNS)
        )
        data = table.to_pydict()
        row_count = table.num_rows
        end = offset + row_count
        group_codes = np.column_stack(
            [
                np.asarray(data[column], dtype=np.int32)
                for column in token_columns
            ]
        )
        codes[offset:end] = group_codes
        group_requires = np.asarray(data["requires_dedup"], dtype=np.bool_)
        requires_dedup[offset:end] = group_requires
        group_dedup = np.asarray(
            [-1 if value is None else value for value in data["dedup_code"]],
            dtype=np.int16,
        )
        dedup_codes[offset:end] = group_dedup
        expected_lengths = np.where(group_requires, 10, 9)
        if not np.array_equal(
            np.asarray(data["final_pid_length"]), expected_lengths
        ):
            raise SftDataError("PID 映射 final_pid_length 与 Dedup 状态不一致")
        if not np.array_equal(group_requires, group_dedup >= 0):
            raise SftDataError("PID 映射 requires_dedup 与 dedup_code 不一致")

        for local_index, poi_id in enumerate(data["poi_id"]):
            if not isinstance(poi_id, str) or not poi_id:
                raise SftDataError("PID 映射包含非法 poi_id")
            if poi_id in row_by_poi_id:
                raise SftDataError(f"PID 映射包含重复 poi_id：{poi_id}")
            global_index = offset + local_index
            row_by_poi_id[poi_id] = global_index
            expected_key = stable_final_pid_key(
                group_codes[local_index], int(group_dedup[local_index])
            )
            if data["final_pid_key"][local_index] != expected_key:
                raise SftDataError("PID 映射 final_pid_key 与 Token 不一致")
        offset = end
    if offset != poi_count or len(row_by_poi_id) != poi_count:
        raise SftDataError("PID 映射行数或 POI 唯一数不完整")
    return (
        PidLookup(
            row_by_poi_id=row_by_poi_id,
            codes=codes,
            dedup_codes=dedup_codes,
            requires_dedup=requires_dedup,
            poi_count=poi_count,
        ),
        manifest,
        mapping_sha256,
        sha256_file(pid_manifest_path),
    )
