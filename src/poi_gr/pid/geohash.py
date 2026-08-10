"""Build and evaluate Geohash-prefixed Semantic IDs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..sid.evaluation import (
    SidEvaluationError,
    compute_basic_metrics,
    load_sid_input,
)


GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
PID_SCHEMA_VERSION = "geohash-pid-v1"
PID_METRICS_SCHEMA_VERSION = "geohash-pid-evaluation-v1"
PID_COMPARISON_SCHEMA_VERSION = "geohash-pid-comparison-v1"
EXPECTED_SID_CODEBOOK_SIZES = (1024, 1024, 1024)
EXPECTED_SID_CHECKPOINT_EPOCH = 20
OUTPUT_FILENAMES = (
    "gid_codes.npy",
    "pid_codes.npy",
    "pid_manifest.json",
    "metrics.json",
    "residual_collision_cases.jsonl",
    "comparison.json",
)


class GeohashPidError(ValueError):
    pass


@dataclass(frozen=True)
class GeohashPidResult:
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    comparison: dict[str, Any]
    residual_cases: tuple[dict[str, Any], ...]


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GeohashPidError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GeohashPidError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(value, dict):
        raise GeohashPidError(f"{name} 必须是 JSON object：{path}")
    return value


def resolve_poi_data_path(
    sid_manifest_path: Path,
    project_root: Path,
    explicit_path: Path | None = None,
) -> tuple[Path, str]:
    """Resolve POI metadata from an override, prior metrics, or resolved config."""

    if explicit_path is not None:
        path = explicit_path if explicit_path.is_absolute() else project_root / explicit_path
        return path.resolve(), "cli"

    sid_manifest_path = sid_manifest_path.resolve()
    metrics_path = sid_manifest_path.parent / "metrics.json"
    if metrics_path.is_file():
        metrics = _load_json_object(metrics_path, "现有 SID metrics")
        value = metrics.get("inputs", {}).get("poi_data")
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if not path.is_absolute():
                path = project_root / path
            return path.resolve(), "sid_metrics"

    sid_manifest = _load_json_object(sid_manifest_path, "SID manifest")
    checkpoint = sid_manifest.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint_value = checkpoint.get("path")
        if isinstance(checkpoint_value, str) and checkpoint_value.strip():
            checkpoint_path = Path(checkpoint_value)
            if not checkpoint_path.is_absolute():
                checkpoint_path = sid_manifest_path.parent / checkpoint_path
            resolved_config_path = checkpoint_path.resolve().parent / "resolved_config.json"
            if resolved_config_path.is_file():
                resolved = _load_json_object(
                    resolved_config_path, "RQ-VAE resolved config"
                )
                value = resolved.get("config", {}).get("poi_data_path")
                if isinstance(value, str) and value.strip():
                    path = Path(value)
                    if not path.is_absolute():
                        path = project_root / path
                    return path.resolve(), "rqvae_resolved_config"

    raise GeohashPidError(
        "无法从 SID metrics 或 RQ-VAE resolved_config 解析 POI 数据路径；"
        "请显式传入 --poi-data"
    )


def geohash_to_tokens(value: str) -> np.ndarray:
    """Convert a Geohash string to integer Base32 tokens."""

    if not isinstance(value, str) or not value:
        raise GeohashPidError("Geohash 必须是非空字符串")
    token_by_character = {character: index for index, character in enumerate(GEOHASH_ALPHABET)}
    try:
        return np.asarray([token_by_character[character] for character in value], dtype=np.uint8)
    except KeyError as error:
        raise GeohashPidError(f"非法 Geohash 字符：{error.args[0]}") from error


def tokens_to_geohash(tokens: Sequence[int]) -> str:
    """Convert integer Base32 tokens back to a Geohash string."""

    characters: list[str] = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
            raise GeohashPidError("Geohash Token 必须是整数")
        value = int(token)
        if not 0 <= value < len(GEOHASH_ALPHABET):
            raise GeohashPidError(f"Geohash Token 超出范围 [0, 32)：{value}")
        characters.append(GEOHASH_ALPHABET[value])
    if not characters:
        raise GeohashPidError("Geohash Token 序列不能为空")
    return "".join(characters)


def encode_geohash(longitude: float, latitude: float, length: int = 6) -> str:
    """Encode one coordinate with the standard longitude-first Geohash algorithm."""

    if length <= 0:
        raise GeohashPidError("Geohash 长度必须大于 0")
    if (
        isinstance(longitude, bool)
        or not isinstance(longitude, (int, float, np.integer, np.floating))
        or not math.isfinite(float(longitude))
        or not -180.0 <= float(longitude) <= 180.0
    ):
        raise GeohashPidError(f"longitude 非法：{longitude}")
    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, (int, float, np.integer, np.floating))
        or not math.isfinite(float(latitude))
        or not -90.0 <= float(latitude) <= 90.0
    ):
        raise GeohashPidError(f"latitude 非法：{latitude}")

    longitude_range = [-180.0, 180.0]
    latitude_range = [-90.0, 90.0]
    characters: list[str] = []
    character_value = 0
    bit_count = 0
    longitude_turn = True
    while len(characters) < length:
        active_range = longitude_range if longitude_turn else latitude_range
        coordinate = float(longitude) if longitude_turn else float(latitude)
        midpoint = (active_range[0] + active_range[1]) / 2.0
        character_value <<= 1
        if coordinate >= midpoint:
            character_value |= 1
            active_range[0] = midpoint
        else:
            active_range[1] = midpoint
        longitude_turn = not longitude_turn
        bit_count += 1
        if bit_count == 5:
            characters.append(GEOHASH_ALPHABET[character_value])
            character_value = 0
            bit_count = 0
    return "".join(characters)


def encode_geohash_tokens(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    length: int = 6,
    chunk_rows: int = 65_536,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Vectorize standard Geohash encoding into one uint8 token per character."""

    longitudes = np.asarray(longitudes)
    latitudes = np.asarray(latitudes)
    if longitudes.ndim != 1 or latitudes.ndim != 1:
        raise GeohashPidError("longitude 和 latitude 必须是一维数组")
    if len(longitudes) != len(latitudes):
        raise GeohashPidError("longitude 和 latitude 行数不一致")
    if length <= 0:
        raise GeohashPidError("Geohash 长度必须大于 0")
    if chunk_rows <= 0:
        raise GeohashPidError("chunk_rows 必须大于 0")
    if not np.isfinite(longitudes).all() or np.any(
        (longitudes < -180.0) | (longitudes > 180.0)
    ):
        raise GeohashPidError("longitude 包含非 finite 或超出 [-180, 180] 的值")
    if not np.isfinite(latitudes).all() or np.any(
        (latitudes < -90.0) | (latitudes > 90.0)
    ):
        raise GeohashPidError("latitude 包含非 finite 或超出 [-90, 90] 的值")

    output = np.empty((len(longitudes), length), dtype=np.uint8)
    total_rows = len(longitudes)
    for start in range(0, total_rows, chunk_rows):
        end = min(start + chunk_rows, total_rows)
        longitude = np.asarray(longitudes[start:end], dtype=np.float64)
        latitude = np.asarray(latitudes[start:end], dtype=np.float64)
        longitude_low = np.full(len(longitude), -180.0)
        longitude_high = np.full(len(longitude), 180.0)
        latitude_low = np.full(len(latitude), -90.0)
        latitude_high = np.full(len(latitude), 90.0)
        character_values = np.zeros(len(longitude), dtype=np.uint8)
        output_column = 0
        for bit_index in range(length * 5):
            longitude_turn = bit_index % 2 == 0
            if longitude_turn:
                midpoint = (longitude_low + longitude_high) / 2.0
                upper = longitude >= midpoint
                longitude_low = np.where(upper, midpoint, longitude_low)
                longitude_high = np.where(upper, longitude_high, midpoint)
            else:
                midpoint = (latitude_low + latitude_high) / 2.0
                upper = latitude >= midpoint
                latitude_low = np.where(upper, midpoint, latitude_low)
                latitude_high = np.where(upper, latitude_high, midpoint)
            character_values = (character_values << 1) | upper.astype(np.uint8)
            if bit_index % 5 == 4:
                output[start:end, output_column] = character_values
                output_column += 1
                character_values.fill(0)
        if progress is not None:
            progress(f"Geohash 编码：{end:,}/{total_rows:,}")
    return output


def compose_pid(gid_codes: np.ndarray, sid_codes: np.ndarray) -> np.ndarray:
    """Compose [GID tokens, SID tokens] as an int32 PID matrix."""

    gid_codes = np.asarray(gid_codes)
    sid_codes = np.asarray(sid_codes)
    if gid_codes.ndim != 2 or sid_codes.ndim != 2:
        raise GeohashPidError("GID 和 SID 必须是二维数组")
    if gid_codes.shape[0] != sid_codes.shape[0]:
        raise GeohashPidError("GID 和 SID 行数不一致")
    if np.any(gid_codes < 0) or np.any(gid_codes >= 32):
        raise GeohashPidError("GID Token 必须位于 [0, 32)")
    pid_codes = np.empty(
        (gid_codes.shape[0], gid_codes.shape[1] + sid_codes.shape[1]),
        dtype=np.int32,
    )
    pid_codes[:, : gid_codes.shape[1]] = gid_codes
    pid_codes[:, gid_codes.shape[1] :] = sid_codes
    return pid_codes


def _discover_poi_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise GeohashPidError(f"POI 数据不存在：{path}")
    files = tuple(
        sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.name.startswith("part-")
        )
    )
    if not files:
        raise GeohashPidError(f"POI 目录中没有 part-* 分片：{path}")
    return files


def _coordinate_value(
    record: dict[str, Any],
    field: str,
    minimum: float,
    maximum: float,
    location: str,
) -> float:
    value = record.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise GeohashPidError(
            f"{location} 的 {field} 非法：要求 finite 且位于 [{minimum}, {maximum}]"
        )
    return float(value)


def load_aligned_coordinates(
    poi_data_path: Path,
    poi_ids: Sequence[str],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load coordinates while proving POI metadata order equals SID row order."""

    poi_count = len(poi_ids)
    longitudes = np.empty(poi_count, dtype=np.float64)
    latitudes = np.empty(poi_count, dtype=np.float64)
    row = 0
    files = _discover_poi_files(poi_data_path)
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                location = f"{path.name}:{line_number}"
                if not line.strip():
                    raise GeohashPidError(f"{location} 是空行")
                if row >= poi_count:
                    raise GeohashPidError("POI 元数据行数多于 SID/POI ID 行数")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise GeohashPidError(f"{location} JSON 解析失败") from error
                if not isinstance(record, dict):
                    raise GeohashPidError(f"{location} 不是 JSON object")
                poi_id = record.get("poi_id")
                if not isinstance(poi_id, str) or not poi_id.strip():
                    raise GeohashPidError(f"{location} poi_id 无效")
                if poi_id != poi_ids[row]:
                    raise GeohashPidError(
                        f"POI 与 SID 行映射不一致：row={row}，"
                        f"POI 数据为 {poi_id}，SID 映射为 {poi_ids[row]}"
                    )
                longitudes[row] = _coordinate_value(
                    record, "lng", -180.0, 180.0, location
                )
                latitudes[row] = _coordinate_value(
                    record, "lat", -90.0, 90.0, location
                )
                row += 1
        if progress is not None:
            progress(f"坐标与行映射校验：{row:,}/{poi_count:,}（{path.name}）")
    if row != poi_count:
        raise GeohashPidError(
            f"POI 元数据行数 {row} 与 SID/POI ID 行数 {poi_count} 不一致"
        )
    return longitudes, latitudes, {
        "poi_metadata_rows": row,
        "poi_id_order_matches_sid": True,
        "longitude_all_finite": True,
        "latitude_all_finite": True,
        "longitude_min": float(longitudes.min()),
        "longitude_max": float(longitudes.max()),
        "latitude_min": float(latitudes.min()),
        "latitude_max": float(latitudes.max()),
        "coordinate_conversion": "none",
    }


def _identifier_metrics(
    basic: dict[str, Any], identifier: str
) -> dict[str, Any]:
    result = dict(basic)
    result[f"distinct_{identifier}_count"] = result.pop("distinct_sid_count")
    result[f"distinct_{identifier}_ratio"] = result.pop("distinct_sid_ratio")
    result[f"singleton_{identifier}_count"] = result.pop("singleton_sid_count")
    return result


def _validate_sid_baseline(
    recomputed: dict[str, Any], metrics_path: Path
) -> tuple[dict[str, Any], str]:
    existing = _load_json_object(metrics_path, "现有 SID metrics")
    expected = existing.get("basic")
    if not isinstance(expected, dict):
        raise GeohashPidError("现有 SID metrics 缺少 basic 指标")
    differences = {
        key: {"recomputed": recomputed.get(key), "existing": expected.get(key)}
        for key in sorted(set(recomputed) | set(expected))
        if recomputed.get(key) != expected.get(key)
    }
    if differences:
        preview = dict(list(differences.items())[:5])
        raise GeohashPidError(
            "重新计算的 SID-only 指标与现有 epoch 20 结果不一致："
            + json.dumps(preview, ensure_ascii=False)
        )
    return existing, _sha256_file(metrics_path)


def _select_top_buckets(
    unique_codes: np.ndarray,
    bucket_sizes: np.ndarray,
    *,
    max_cases: int,
) -> list[int]:
    colliding = np.flatnonzero(bucket_sizes > 1)
    return sorted(
        (int(index) for index in colliding),
        key=lambda index: (
            -int(bucket_sizes[index]),
            tuple(int(token) for token in unique_codes[index]),
        ),
    )[:max_cases]


def _top_gid_buckets(
    unique_gid_codes: np.ndarray,
    gid_bucket_sizes: np.ndarray,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    order = sorted(
        range(len(unique_gid_codes)),
        key=lambda index: (
            -int(gid_bucket_sizes[index]),
            tuple(int(token) for token in unique_gid_codes[index]),
        ),
    )[:limit]
    poi_count = int(gid_bucket_sizes.sum())
    return [
        {
            "geohash6": tokens_to_geohash(unique_gid_codes[index]),
            "gid_tokens": [int(token) for token in unique_gid_codes[index]],
            "bucket_size": int(gid_bucket_sizes[index]),
            "poi_ratio": int(gid_bucket_sizes[index]) / poi_count,
        }
        for index in order
    ]


def compute_collision_resolution(
    sid_inverse: np.ndarray,
    sid_bucket_sizes: np.ndarray,
    pid_inverse: np.ndarray,
    pid_bucket_sizes: np.ndarray,
) -> dict[str, Any]:
    """Measure how many original SID collisions are resolved by a GID prefix."""

    sid_inverse = np.asarray(sid_inverse)
    pid_inverse = np.asarray(pid_inverse)
    if sid_inverse.ndim != 1 or pid_inverse.ndim != 1:
        raise GeohashPidError("SID/PID inverse 必须是一维数组")
    if len(sid_inverse) != len(pid_inverse):
        raise GeohashPidError("SID/PID inverse 行数不一致")
    sid_row_sizes = np.asarray(sid_bucket_sizes)[sid_inverse]
    pid_row_sizes = np.asarray(pid_bucket_sizes)[pid_inverse]
    sid_colliding_mask = sid_row_sizes > 1
    pid_colliding_mask = pid_row_sizes > 1
    if np.any(pid_colliding_mask & ~sid_colliding_mask):
        raise GeohashPidError("PID 出现跨 SID 合并，PID 顺序或构建逻辑错误")

    sid_colliding_poi_count = int(sid_colliding_mask.sum())
    resolved_poi_count = int((sid_colliding_mask & ~pid_colliding_mask).sum())
    residual_poi_count = int(pid_colliding_mask.sum())
    if resolved_poi_count + residual_poi_count != sid_colliding_poi_count:
        raise GeohashPidError("SID 碰撞拆分计数不守恒")

    sid_bucket_has_residual = np.zeros(len(sid_bucket_sizes), dtype=np.bool_)
    sid_bucket_has_residual[sid_inverse[pid_colliding_mask]] = True
    sid_colliding_buckets = np.asarray(sid_bucket_sizes) > 1
    fully_resolved_sid_buckets = sid_colliding_buckets & ~sid_bucket_has_residual
    residual_pid_buckets = np.asarray(pid_bucket_sizes) > 1
    poi_count = len(sid_inverse)
    sid_colliding_bucket_count = int(sid_colliding_buckets.sum())
    fully_resolved_bucket_count = int(fully_resolved_sid_buckets.sum())
    return {
        "original_sid_singleton_poi_count": int((sid_row_sizes == 1).sum()),
        "sid_colliding_poi_count": sid_colliding_poi_count,
        "sid_colliding_bucket_count": sid_colliding_bucket_count,
        "sid_colliding_poi_became_unique_count": resolved_poi_count,
        "fully_resolved_sid_colliding_bucket_count": fully_resolved_bucket_count,
        "sid_colliding_poi_resolution_ratio": (
            resolved_poi_count / sid_colliding_poi_count
            if sid_colliding_poi_count
            else 0.0
        ),
        "sid_colliding_bucket_resolution_ratio": (
            fully_resolved_bucket_count / sid_colliding_bucket_count
            if sid_colliding_bucket_count
            else 0.0
        ),
        "residual_colliding_poi_count": residual_poi_count,
        "residual_colliding_poi_ratio_of_sid_colliding": (
            residual_poi_count / sid_colliding_poi_count
            if sid_colliding_poi_count
            else 0.0
        ),
        "residual_colliding_poi_ratio_of_all_pois": residual_poi_count / poi_count,
        "residual_pid_bucket_count": int(residual_pid_buckets.sum()),
        "sid_colliding_buckets_with_residual_count": int(
            (sid_colliding_buckets & sid_bucket_has_residual).sum()
        ),
        "maximum_residual_pid_bucket_size": (
            int(np.asarray(pid_bucket_sizes)[residual_pid_buckets].max())
            if residual_pid_buckets.any()
            else 1
        ),
        "dedup_token_poi_count_for_strict_one_to_one": residual_poi_count,
        "maximum_required_dedup_capacity": (
            int(np.asarray(pid_bucket_sizes)[residual_pid_buckets].max())
            if residual_pid_buckets.any()
            else 1
        ),
        "dedup_code_required_for_strict_one_to_one": bool(residual_poi_count),
    }


def _selected_rows_by_bucket(
    inverse: np.ndarray, selected_buckets: Sequence[int]
) -> dict[int, list[int]]:
    selected = {int(bucket): [] for bucket in selected_buckets}
    for row, bucket_value in enumerate(inverse):
        bucket = int(bucket_value)
        if bucket in selected:
            selected[bucket].append(row)
    return selected


def _load_selected_metadata(
    poi_data_path: Path,
    poi_ids: Sequence[str],
    selected_rows: set[int],
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    row = 0
    for path in _discover_poi_files(poi_data_path):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if row in selected_rows:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise GeohashPidError(
                            f"{path.name}:{line_number} JSON 解析失败"
                        ) from error
                    if not isinstance(record, dict):
                        raise GeohashPidError(
                            f"{path.name}:{line_number} 不是 JSON object"
                        )
                    if record.get("poi_id") != poi_ids[row]:
                        raise GeohashPidError(
                            f"残余 Case 回表行映射不一致：row={row}"
                        )
                    records[row] = record
                row += 1
    if len(records) != len(selected_rows):
        raise GeohashPidError("部分残余碰撞 POI 未在主表中找到")
    return records


def _normalize_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "N"}
    )
    return normalized or None


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _case_poi(record: dict[str, Any]) -> dict[str, Any]:
    parent_value = None
    for field in ("parent_poi_id", "parent_id", "father_poi_id"):
        if field in record:
            parent_value = record.get(field)
            break
    return {
        "poi_id": record.get("poi_id"),
        "displayname": (
            record.get("displayname")
            if isinstance(record.get("displayname"), str)
            else None
        ),
        "address": (
            record.get("address")
            if isinstance(record.get("address"), str)
            else None
        ),
        "category": (
            record.get("category")
            if isinstance(record.get("category"), str)
            else None
        ),
        "category_code": (
            record.get("category_code")
            if isinstance(record.get("category_code"), str)
            else None
        ),
        "lng": _safe_number(record.get("lng")),
        "lat": _safe_number(record.get("lat")),
        "layer": _safe_number(record.get("layer")),
        "parent_poi_id": parent_value,
        "click_score": _safe_number(record.get("click_score")),
    }


def _case_sort_key(poi: dict[str, Any]) -> tuple[int, float, str]:
    click_score = poi.get("click_score")
    if isinstance(click_score, (int, float)) and not isinstance(click_score, bool):
        return 0, -float(click_score), str(poi.get("poi_id"))
    return 1, 0.0, str(poi.get("poi_id"))


def _maximum_haversine_km(pois: Sequence[dict[str, Any]]) -> float:
    points = np.asarray(
        [
            (float(poi["lng"]), float(poi["lat"]))
            for poi in pois
            if isinstance(poi.get("lng"), (int, float))
            and isinstance(poi.get("lat"), (int, float))
        ],
        dtype=np.float64,
    )
    if len(points) < 2:
        return 0.0
    longitude = np.radians(points[:, 0])
    latitude = np.radians(points[:, 1])
    maximum = 0.0
    for index in range(len(points) - 1):
        delta_longitude = longitude[index + 1 :] - longitude[index]
        delta_latitude = latitude[index + 1 :] - latitude[index]
        value = (
            np.sin(delta_latitude / 2.0) ** 2
            + np.cos(latitude[index])
            * np.cos(latitude[index + 1 :])
            * np.sin(delta_longitude / 2.0) ** 2
        )
        distance = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
        if len(distance):
            maximum = max(maximum, float(distance.max()))
    return maximum


def _collision_reason(
    pois: Sequence[dict[str, Any]],
    *,
    top_name_ratio: float,
    coordinate_unique_count: int,
    maximum_span_km: float,
) -> tuple[str, str]:
    names = [
        poi.get("displayname")
        for poi in pois
        if isinstance(poi.get("displayname"), str)
    ]
    categories = [
        poi.get("category")
        for poi in pois
        if isinstance(poi.get("category"), str)
    ]
    entrance_ratio = (
        sum(
            any(keyword in name for keyword in ("入口", "出口", "东门", "西门", "南门", "北门"))
            for name in names
        )
        / len(names)
        if names
        else 0.0
    )
    property_ratio = (
        sum(
            any(keyword in category for keyword in ("房产小区", "楼栋号", "商务楼宇"))
            for category in categories
        )
        / len(categories)
        if categories
        else 0.0
    )
    parent_category_ratio = (
        sum(
            any(keyword in category for keyword in ("附属", "丧葬"))
            for category in categories
        )
        / len(categories)
        if categories
        else 0.0
    )
    numeric_ratio = (
        sum(any(character.isdigit() for character in name) for name in names)
        / len(names)
        if names
        else 0.0
    )
    coordinate_ratio = coordinate_unique_count / len(pois)

    if top_name_ratio >= 0.8 and coordinate_ratio <= 0.5:
        return (
            "数据重复或近重复",
            "高频标准化名称占比很高，且大量记录共享相同或近似坐标，疑似重复或近重复数据。",
        )
    if entrance_ratio >= 0.3:
        return (
            "父子POI或入口",
            "名称中入口/出口方向词占比较高，同一 Geohash6 内父子点或入口未被语义 SID 完全拆分。",
        )
    if parent_category_ratio >= 0.3 and maximum_span_km <= 1.0:
        return (
            "父子POI或入口",
            "同一机构或场所的分区、附属设施和内部子点集中在一个 Geohash6，属于父子实体残余碰撞。",
        )
    if property_ratio >= 0.5 and maximum_span_km <= 2.0:
        return (
            "同一小区或商业体内部的楼栋/子POI",
            "房产或楼栋类 POI 高度集中在同一 Geohash6，主要是同一小区/商业体内部子实体。",
        )
    if numeric_ratio >= 0.7 and maximum_span_km <= 2.0:
        return (
            "数字门牌差异",
            "多数名称仅在楼号、单元或门牌数字上有差异，Geohash6 与语义 SID 均未区分。",
        )
    if maximum_span_km <= 0.05 or coordinate_ratio <= 0.35:
        return (
            "坐标相同或高度接近",
            "桶内坐标相同或高度接近，空间前缀无法继续区分这些 POI。",
        )
    if top_name_ratio >= 0.5:
        return (
            "数据重复或近重复",
            "同一标准化名称在同一 Geohash6 内重复出现，需进一步核验重复数据或同名子点。",
        )
    return (
        "其他有害碰撞",
        "同一 Geohash6 与语义 SID 下仍包含多个不同 POI，未呈现单一的父子、门牌或重复模式。",
    )


def build_residual_cases(
    unique_pid_codes: np.ndarray,
    pid_bucket_sizes: np.ndarray,
    rows_by_bucket: dict[int, list[int]],
    records_by_row: dict[int, dict[str, Any]],
    *,
    geohash_length: int,
) -> list[dict[str, Any]]:
    """Aggregate every member of the largest residual PID collision buckets."""

    cases: list[dict[str, Any]] = []
    for bucket, rows in rows_by_bucket.items():
        pois = sorted(
            (_case_poi(records_by_row[row]) for row in rows),
            key=_case_sort_key,
        )
        name_counts = Counter(
            normalized
            for poi in pois
            if (normalized := _normalize_name(poi.get("displayname"))) is not None
        )
        category_counts = Counter(
            str(poi["category_code"]).strip()
            for poi in pois
            if isinstance(poi.get("category_code"), str)
            and str(poi["category_code"]).strip()
        )
        top_category_code = None
        top_category_count = 0
        if category_counts:
            top_category_code, top_category_count = sorted(
                category_counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
        top_name_count = max(name_counts.values()) if name_counts else 0
        coordinates = {
            (float(poi["lng"]), float(poi["lat"]))
            for poi in pois
            if isinstance(poi.get("lng"), (int, float))
            and isinstance(poi.get("lat"), (int, float))
        }
        maximum_span_km = _maximum_haversine_km(pois)
        reason_type, reason = _collision_reason(
            pois,
            top_name_ratio=top_name_count / len(pois),
            coordinate_unique_count=len(coordinates),
            maximum_span_km=maximum_span_km,
        )
        pid = [int(token) for token in unique_pid_codes[bucket]]
        gid = pid[:geohash_length]
        sid = pid[geohash_length:]
        labeled_category_count = sum(category_counts.values())
        cases.append(
            {
                "geohash6": tokens_to_geohash(gid),
                "semantic_sid": sid,
                "full_pid": pid,
                "bucket_size": int(pid_bucket_sizes[bucket]),
                "output_poi_count": len(pois),
                "truncated": False,
                "normalized_name_count": len(name_counts),
                "category_code_count": len(category_counts),
                "labeled_category_count": labeled_category_count,
                "missing_category_count": len(pois) - labeled_category_count,
                "top_category_code": top_category_code,
                "top_category_ratio": (
                    top_category_count / labeled_category_count
                    if labeled_category_count
                    else None
                ),
                "coordinate_unique_count": len(coordinates),
                "maximum_geographic_span_km": maximum_span_km,
                "collision_reason_type": reason_type,
                "collision_reason": reason,
                "pois": pois,
            }
        )
    return cases


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in output_dir.iterdir() if path.name not in OUTPUT_FILENAMES
    )
    if unexpected:
        raise GeohashPidError(
            "输出目录包含非 PID-001 文件，拒绝覆盖或删除：" + ", ".join(unexpected)
        )
    for filename in OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.exists() and not path.is_file():
            raise GeohashPidError(f"输出目标不是普通文件：{path}")


def _write_outputs(
    output_dir: Path,
    gid_codes: np.ndarray,
    pid_codes: np.ndarray,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    _prepare_output_dir(output_dir)
    temporary_paths = {
        filename: output_dir / f".{filename}.tmp" for filename in OUTPUT_FILENAMES
    }
    try:
        with temporary_paths["gid_codes.npy"].open("wb") as handle:
            np.save(handle, gid_codes, allow_pickle=False)
        with temporary_paths["pid_codes.npy"].open("wb") as handle:
            np.save(handle, pid_codes, allow_pickle=False)
        for filename, payload in (
            ("pid_manifest.json", manifest),
            ("metrics.json", metrics),
            ("comparison.json", comparison),
        ):
            temporary_paths[filename].write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        with temporary_paths["residual_collision_cases.jsonl"].open(
            "w", encoding="utf-8"
        ) as handle:
            for case in cases:
                handle.write(
                    json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n"
                )
        for filename in OUTPUT_FILENAMES:
            os.replace(temporary_paths[filename], output_dir / filename)
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)


def build_geohash_pid(
    sid_manifest_path: Path,
    poi_data_path: Path,
    output_dir: Path,
    *,
    geohash_length: int = 6,
    order: str = "gid_sid",
    max_cases: int = 20,
    progress: Callable[[str], None] | None = None,
) -> GeohashPidResult:
    """Build full GID/PID arrays, evaluate collisions, and write six artifacts."""

    started = time.perf_counter()
    sid_manifest_path = sid_manifest_path.resolve()
    poi_data_path = poi_data_path.resolve()
    output_dir = output_dir.resolve()
    if geohash_length != 6:
        raise GeohashPidError("PID-001 只允许 geohash_length=6")
    if order != "gid_sid":
        raise GeohashPidError("PID-001 只允许 order=gid_sid")
    if max_cases <= 0:
        raise GeohashPidError("max_cases 必须大于 0")

    if progress is not None:
        progress("读取并校验 SID manifest、SID Token 与 POI ID 映射")
    try:
        sid_input = load_sid_input(sid_manifest_path)
    except SidEvaluationError as error:
        raise GeohashPidError(str(error)) from error
    if sid_input.codes.shape[1] != 3:
        raise GeohashPidError(
            f"PID-001 要求 SID shape=[N,3]，实际 {list(sid_input.codes.shape)}"
        )
    if sid_input.codebook_sizes != EXPECTED_SID_CODEBOOK_SIZES:
        raise GeohashPidError(
            "PID-001 要求 SID codebook_sizes=[1024,1024,1024]，实际 "
            f"{list(sid_input.codebook_sizes)}"
        )
    sid_manifest = _load_json_object(sid_manifest_path, "SID manifest")
    experiment_id = sid_manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise GeohashPidError(
            "PID-001 要求 SID manifest 包含非空 experiment_id"
        )
    checkpoint = sid_manifest.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("epoch") != EXPECTED_SID_CHECKPOINT_EPOCH
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint["path"].strip()
        or not isinstance(checkpoint.get("sha256"), str)
        or not checkpoint["sha256"].strip()
    ):
        raise GeohashPidError(
            "PID-001 要求 SID manifest 完整记录 epoch 20 checkpoint 路径和 SHA256"
        )
    expected_poi_hash = sid_manifest.get("poi_ids", {}).get("sha256")
    actual_poi_hash = _sha256_file(sid_input.poi_ids_path)
    if expected_poi_hash != actual_poi_hash:
        raise GeohashPidError(
            f"POI ID SHA256 {actual_poi_hash} 与 SID manifest {expected_poi_hash} 不一致"
        )

    if progress is not None:
        progress("重新计算 SID-only 指标并与现有 epoch 20 结果逐字段核对")
    sid_basic, unique_sids, sid_inverse, sid_bucket_sizes = compute_basic_metrics(
        sid_input.codes
    )
    baseline_metrics_path = sid_manifest_path.parent / "metrics.json"
    _, baseline_metrics_sha256 = _validate_sid_baseline(
        sid_basic, baseline_metrics_path
    )

    longitudes, latitudes, coordinate_validation = load_aligned_coordinates(
        poi_data_path, sid_input.poi_ids, progress=progress
    )
    gid_codes = encode_geohash_tokens(
        longitudes,
        latitudes,
        length=geohash_length,
        progress=progress,
    )
    pid_codes = compose_pid(gid_codes, sid_input.codes)
    del longitudes, latitudes

    if progress is not None:
        progress("计算 GID6-only 全量桶统计")
    gid_basic_raw, unique_gids, _, gid_bucket_sizes = compute_basic_metrics(gid_codes)
    gid_metrics = _identifier_metrics(gid_basic_raw, "gid")
    gid_metrics["top20_geohash6"] = _top_gid_buckets(
        unique_gids, gid_bucket_sizes
    )

    if progress is not None:
        progress("计算 GID6+SID 完整 PID 全量桶统计")
    pid_basic_raw, unique_pids, pid_inverse, pid_bucket_sizes = compute_basic_metrics(
        pid_codes
    )
    pid_metrics = _identifier_metrics(pid_basic_raw, "pid")
    collision_resolution = compute_collision_resolution(
        sid_inverse,
        sid_bucket_sizes,
        pid_inverse,
        pid_bucket_sizes,
    )

    selected_buckets = _select_top_buckets(
        unique_pids, pid_bucket_sizes, max_cases=max_cases
    )
    rows_by_bucket = _selected_rows_by_bucket(pid_inverse, selected_buckets)
    selected_rows = {row for rows in rows_by_bucket.values() for row in rows}
    if progress is not None:
        progress(
            f"回表聚合残余 Top {len(selected_buckets)} 完整碰撞桶，"
            f"涉及 {len(selected_rows):,} 条 POI"
        )
    records_by_row = _load_selected_metadata(
        poi_data_path, sid_input.poi_ids, selected_rows
    )
    residual_cases = build_residual_cases(
        unique_pids,
        pid_bucket_sizes,
        rows_by_bucket,
        records_by_row,
        geohash_length=geohash_length,
    )

    built_at = datetime.now(timezone.utc).isoformat()
    sid_manifest_sha256 = _sha256_file(sid_manifest_path)
    build_seconds = time.perf_counter() - started
    manifest = {
        "schema_version": PID_SCHEMA_VERSION,
        "status": "completed",
        "built_at": built_at,
        "build_seconds": build_seconds,
        "poi_count": int(pid_codes.shape[0]),
        "poi_ids": {
            "path": str(sid_input.poi_ids_path),
            "sha256": actual_poi_hash,
            "rows": len(sid_input.poi_ids),
            "unique": True,
        },
        "sid_source": {
            "manifest": str(sid_manifest_path),
            "manifest_sha256": sid_manifest_sha256,
            "sid_codes": str(sid_input.codes_path),
            "shape": [int(value) for value in sid_input.codes.shape],
            "dtype": str(sid_input.codes.dtype),
            "codebook_sizes": list(sid_input.codebook_sizes),
            "experiment_id": sid_manifest.get("experiment_id"),
            "method": sid_manifest.get("method"),
            "checkpoint": checkpoint,
            "checkpoint_epoch": (
                checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
            ),
        },
        "geohash": {
            "algorithm": "standard_binary_interval_subdivision",
            "longitude_first": True,
            "upper_half_rule": "coordinate_greater_than_or_equal_to_midpoint",
            "alphabet": GEOHASH_ALPHABET,
            "length": geohash_length,
            "coordinate_fields": {"longitude": "lng", "latitude": "lat"},
            "coordinate_conversion": "none",
        },
        "gid_codes": {
            "path": "gid_codes.npy",
            "shape": [int(value) for value in gid_codes.shape],
            "dtype": str(gid_codes.dtype),
            "codebook_sizes": [32] * geohash_length,
        },
        "pid_codes": {
            "path": "pid_codes.npy",
            "shape": [int(value) for value in pid_codes.shape],
            "dtype": str(pid_codes.dtype),
            "order": order,
            "token_order": [
                *[f"G{index}" for index in range(1, geohash_length + 1)],
                "S1",
                "S2",
                "S3",
            ],
            "codebook_sizes": [
                *([32] * geohash_length),
                *sid_input.codebook_sizes,
            ],
        },
    }
    metrics = {
        "schema_version": PID_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "built_at": built_at,
        "build_seconds": build_seconds,
        "inputs": {
            "sid_manifest": str(sid_manifest_path),
            "sid_manifest_sha256": sid_manifest_sha256,
            "sid_metrics": str(baseline_metrics_path.resolve()),
            "sid_metrics_sha256": baseline_metrics_sha256,
            "poi_ids": str(sid_input.poi_ids_path),
            "poi_ids_sha256": actual_poi_hash,
            "poi_data": str(poi_data_path),
        },
        "validation": {
            "sid_shape": [int(value) for value in sid_input.codes.shape],
            "sid_codebook_sizes": list(sid_input.codebook_sizes),
            "poi_id_rows": len(sid_input.poi_ids),
            "poi_ids_unique": True,
            "poi_id_hash_matches_manifest": True,
            "sid_baseline_matches_existing_metrics": True,
            **coordinate_validation,
        },
        "sid_only": sid_basic,
        "gid6_only": gid_metrics,
        "gid6_sid_pid": pid_metrics,
        "collision_resolution": collision_resolution,
        "residual_case_selection": {
            "order": "bucket_size_desc_then_full_pid_lexicographic",
            "max_cases": max_cases,
            "available_residual_collision_bucket_count": collision_resolution[
                "residual_pid_bucket_count"
            ],
            "output_case_count": len(residual_cases),
            "case_contains_all_bucket_pois": True,
        },
    }
    comparison = {
        "schema_version": PID_COMPARISON_SCHEMA_VERSION,
        "status": "completed",
        "id_spaces": [
            {
                "name": "semantic_sid",
                "layers": 3,
                "distinct_count": sid_basic["distinct_sid_count"],
                "distinct_ratio": sid_basic["distinct_sid_ratio"],
                "collision_excess_count": sid_basic["collision_excess_count"],
                "collision_excess_ratio": sid_basic["collision_excess_ratio"],
                "colliding_poi_count": sid_basic["colliding_poi_count"],
                "colliding_poi_ratio": sid_basic["colliding_poi_ratio"],
                "bucket_size_p99": sid_basic["bucket_size_p99"],
                "bucket_size_max": sid_basic["bucket_size_max"],
            },
            {
                "name": "geohash6_gid",
                "layers": 6,
                "distinct_count": gid_metrics["distinct_gid_count"],
                "distinct_ratio": gid_metrics["distinct_gid_ratio"],
                "collision_excess_count": gid_metrics["collision_excess_count"],
                "collision_excess_ratio": gid_metrics["collision_excess_ratio"],
                "colliding_poi_count": gid_metrics["colliding_poi_count"],
                "colliding_poi_ratio": gid_metrics["colliding_poi_ratio"],
                "bucket_size_p99": gid_metrics["bucket_size_p99"],
                "bucket_size_max": gid_metrics["bucket_size_max"],
            },
            {
                "name": "geohash6_semantic_pid",
                "layers": 9,
                "distinct_count": pid_metrics["distinct_pid_count"],
                "distinct_ratio": pid_metrics["distinct_pid_ratio"],
                "collision_excess_count": pid_metrics["collision_excess_count"],
                "collision_excess_ratio": pid_metrics["collision_excess_ratio"],
                "colliding_poi_count": pid_metrics["colliding_poi_count"],
                "colliding_poi_ratio": pid_metrics["colliding_poi_ratio"],
                "bucket_size_p99": pid_metrics["bucket_size_p99"],
                "bucket_size_max": pid_metrics["bucket_size_max"],
            },
        ],
        "collision_resolution": collision_resolution,
        "pid_minus_sid": {
            "distinct_count_increase": (
                pid_metrics["distinct_pid_count"] - sid_basic["distinct_sid_count"]
            ),
            "distinct_ratio_increase": (
                pid_metrics["distinct_pid_ratio"] - sid_basic["distinct_sid_ratio"]
            ),
            "colliding_poi_count_decrease": (
                sid_basic["colliding_poi_count"]
                - pid_metrics["colliding_poi_count"]
            ),
            "colliding_poi_ratio_decrease": (
                sid_basic["colliding_poi_ratio"]
                - pid_metrics["colliding_poi_ratio"]
            ),
        },
        "dedup_assessment": {
            "strict_one_to_one_pid_achieved": (
                collision_resolution["residual_colliding_poi_count"] == 0
            ),
            "dedup_code_required_for_strict_one_to_one": collision_resolution[
                "dedup_code_required_for_strict_one_to_one"
            ],
            "poi_count_requiring_dedup_token": collision_resolution[
                "dedup_token_poi_count_for_strict_one_to_one"
            ],
            "maximum_required_dedup_capacity": collision_resolution[
                "maximum_required_dedup_capacity"
            ],
            "dedup_code_constructed": False,
        },
    }
    _write_outputs(
        output_dir,
        gid_codes,
        pid_codes,
        manifest,
        metrics,
        residual_cases,
        comparison,
    )
    if progress is not None:
        progress(f"PID-001 完成，输出目录：{output_dir}")
    return GeohashPidResult(
        manifest=manifest,
        metrics=metrics,
        comparison=comparison,
        residual_cases=tuple(residual_cases),
    )
