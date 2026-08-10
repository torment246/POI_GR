"""Evaluate aligned Semantic ID codes with deterministic metric definitions."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SID_INPUT_SCHEMA_VERSION = "sid-input-v1"
SID_EVALUATION_SCHEMA_VERSION = "sid-evaluation-v1"


class SidEvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class SidInput:
    manifest_path: Path
    codes_path: Path
    poi_ids_path: Path
    codes: np.ndarray
    poi_ids: tuple[str, ...]
    codebook_sizes: tuple[int, ...]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SidEvaluationError(f"{name} 必须是 JSON object")
    return value


def _resolve_manifest_path(value: Any, manifest_path: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SidEvaluationError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _positive_int_list(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SidEvaluationError(f"{name} 必须是非空正整数数组")
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value):
        raise SidEvaluationError(f"{name} 必须是非空正整数数组")
    return tuple(value)


def _load_poi_ids(path: Path, expected_rows: int) -> tuple[str, ...]:
    if not path.is_file():
        raise SidEvaluationError(f"POI ID 文件不存在：{path}")
    poi_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SidEvaluationError(f"POI ID 文件第 {line_number} 行为空")
            try:
                poi_id = json.loads(line)
            except json.JSONDecodeError as error:
                raise SidEvaluationError(
                    f"POI ID 文件第 {line_number} 行 JSON 解析失败"
                ) from error
            if not isinstance(poi_id, str) or not poi_id.strip():
                raise SidEvaluationError(
                    f"POI ID 文件第 {line_number} 行必须是非空 JSON 字符串"
                )
            if poi_id in seen:
                raise SidEvaluationError(f"POI ID 重复：{poi_id}")
            seen.add(poi_id)
            poi_ids.append(poi_id)
    if len(poi_ids) != expected_rows:
        raise SidEvaluationError(
            f"POI ID 行数 {len(poi_ids)} 与 SID 行数 {expected_rows} 不一致"
        )
    return tuple(poi_ids)


def load_sid_input(manifest_path: Path) -> SidInput:
    """Load and validate an aligned SID array and its declared codebooks."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise SidEvaluationError(f"SID manifest 不存在：{manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = _require_mapping(json.load(handle), "SID manifest")
    except json.JSONDecodeError as error:
        raise SidEvaluationError("SID manifest JSON 解析失败") from error

    if manifest.get("schema_version") != SID_INPUT_SCHEMA_VERSION:
        raise SidEvaluationError(
            f"schema_version 必须是 {SID_INPUT_SCHEMA_VERSION}"
        )
    codes_spec = _require_mapping(manifest.get("sid_codes"), "sid_codes")
    ids_spec = _require_mapping(manifest.get("poi_ids"), "poi_ids")
    declared_shape = _positive_int_list(codes_spec.get("shape"), "sid_codes.shape")
    if len(declared_shape) != 2:
        raise SidEvaluationError("sid_codes.shape 必须是 [N, L]")
    codebook_sizes = _positive_int_list(
        manifest.get("codebook_sizes"), "codebook_sizes"
    )
    if len(codebook_sizes) != declared_shape[1]:
        raise SidEvaluationError(
            "codebook_sizes 长度必须等于 sid_codes.shape 的层数 L"
        )

    dtype_value = codes_spec.get("dtype")
    if not isinstance(dtype_value, str) or not dtype_value.strip():
        raise SidEvaluationError("sid_codes.dtype 必须是非空字符串")
    try:
        declared_dtype = np.dtype(dtype_value)
    except TypeError as error:
        raise SidEvaluationError(f"不支持的 sid_codes.dtype：{dtype_value}") from error
    if declared_dtype.kind not in {"i", "u"}:
        raise SidEvaluationError("sid_codes.dtype 必须是整数类型")

    codes_path = _resolve_manifest_path(
        codes_spec.get("path"), manifest_path, "sid_codes.path"
    )
    poi_ids_path = _resolve_manifest_path(
        ids_spec.get("path"), manifest_path, "poi_ids.path"
    )
    if not codes_path.is_file():
        raise SidEvaluationError(f"SID codes 文件不存在：{codes_path}")
    try:
        codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SidEvaluationError(f"SID codes NPY 读取失败：{codes_path}") from error
    if codes.ndim != 2:
        raise SidEvaluationError(f"SID codes 必须是二维数组，实际 ndim={codes.ndim}")
    if tuple(codes.shape) != declared_shape:
        raise SidEvaluationError(
            f"SID shape {tuple(codes.shape)} 与 manifest {declared_shape} 不一致"
        )
    if codes.dtype != declared_dtype:
        raise SidEvaluationError(
            f"SID dtype {codes.dtype} 与 manifest {declared_dtype} 不一致"
        )

    for level_index, codebook_size in enumerate(codebook_sizes):
        level_tokens = codes[:, level_index]
        minimum = int(level_tokens.min())
        maximum = int(level_tokens.max())
        if minimum < 0:
            raise SidEvaluationError(
                f"SID 第 {level_index + 1} 层存在负 Token：{minimum}"
            )
        if maximum >= codebook_size:
            raise SidEvaluationError(
                f"SID 第 {level_index + 1} 层 Token {maximum} 超出码本范围 "
                f"[0, {codebook_size})"
            )

    poi_ids = _load_poi_ids(poi_ids_path, declared_shape[0])
    return SidInput(
        manifest_path=manifest_path,
        codes_path=codes_path.resolve(),
        poi_ids_path=poi_ids_path.resolve(),
        codes=codes,
        poi_ids=poi_ids,
        codebook_sizes=codebook_sizes,
    )


def _nearest_rank(values: np.ndarray, quantile: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.int64))
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return int(ordered[index])


def _bucket_statistics(bucket_sizes: np.ndarray) -> dict[str, Any]:
    histogram_values, histogram_counts = np.unique(bucket_sizes, return_counts=True)
    return {
        "bucket_size_p50": _nearest_rank(bucket_sizes, 0.50),
        "bucket_size_p90": _nearest_rank(bucket_sizes, 0.90),
        "bucket_size_p95": _nearest_rank(bucket_sizes, 0.95),
        "bucket_size_p99": _nearest_rank(bucket_sizes, 0.99),
        "bucket_size_max": int(bucket_sizes.max()),
        "bucket_size_histogram": {
            str(int(size)): int(count)
            for size, count in zip(histogram_values, histogram_counts, strict=True)
        },
    }


def compute_basic_metrics(codes: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Compute full-SID collision metrics with separate excess and POI ratios."""

    unique_sids, inverse, bucket_sizes = np.unique(
        np.asarray(codes), axis=0, return_inverse=True, return_counts=True
    )
    poi_count = int(codes.shape[0])
    distinct_sid_count = int(len(unique_sids))
    collision_excess_count = poi_count - distinct_sid_count
    colliding_mask = bucket_sizes > 1
    colliding_bucket_count = int(colliding_mask.sum())
    colliding_poi_count = int(bucket_sizes[colliding_mask].sum())
    singleton_sid_count = int((bucket_sizes == 1).sum())
    singleton_poi_count = singleton_sid_count
    metrics = {
        "poi_count": poi_count,
        "distinct_sid_count": distinct_sid_count,
        "distinct_sid_ratio": distinct_sid_count / poi_count,
        "collision_excess_count": collision_excess_count,
        "collision_excess_ratio": collision_excess_count / poi_count,
        "colliding_bucket_count": colliding_bucket_count,
        "colliding_poi_count": colliding_poi_count,
        "colliding_poi_ratio": colliding_poi_count / poi_count,
        "singleton_sid_count": singleton_sid_count,
        "singleton_poi_count": singleton_poi_count,
        "singleton_poi_ratio": singleton_poi_count / poi_count,
        **_bucket_statistics(bucket_sizes),
    }
    return metrics, unique_sids, inverse, bucket_sizes


def compute_layer_metrics(
    codes: np.ndarray, codebook_sizes: Sequence[int]
) -> list[dict[str, Any]]:
    """Compute exact token usage for every declared SID level."""

    layers: list[dict[str, Any]] = []
    for level_index, codebook_size in enumerate(codebook_sizes):
        token_counts = np.bincount(
            np.asarray(codes[:, level_index], dtype=np.int64),
            minlength=codebook_size,
        )
        used_token_count = int(np.count_nonzero(token_counts))
        probabilities = token_counts[token_counts > 0] / len(codes)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = (
            entropy / math.log(codebook_size) if codebook_size > 1 else 0.0
        )
        layers.append(
            {
                "level": level_index + 1,
                "codebook_size": int(codebook_size),
                "used_token_count": used_token_count,
                "codebook_utilization_ratio": used_token_count / codebook_size,
                "unused_token_count": int(codebook_size - used_token_count),
                "normalized_entropy": normalized_entropy,
                "token_counts": [int(value) for value in token_counts],
            }
        )
    return layers


def _discover_poi_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise SidEvaluationError(f"POI 数据不存在：{path}")
    files = tuple(
        sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.name.startswith("part-")
        )
    )
    if not files:
        raise SidEvaluationError(f"POI 目录中没有 part-* 分片：{path}")
    return files


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _case_poi(record: dict[str, Any], poi_id: str) -> dict[str, Any]:
    return {
        "poi_id": poi_id,
        "displayname": record.get("displayname")
        if isinstance(record.get("displayname"), str)
        else None,
        "address": record.get("address")
        if isinstance(record.get("address"), str)
        else None,
        "alias": record.get("alias") if isinstance(record.get("alias"), str) else None,
        "category": record.get("category")
        if isinstance(record.get("category"), str)
        else None,
        "category_code": record.get("category_code")
        if isinstance(record.get("category_code"), str)
        else None,
        "lng": _safe_number(record.get("lng")),
        "lat": _safe_number(record.get("lat")),
        "layer": _safe_number(record.get("layer")),
        "click_score": _safe_number(record.get("click_score")),
    }


def load_poi_metadata(
    poi_data_path: Path,
    poi_ids: Sequence[str],
    selected_rows: set[int],
) -> tuple[np.ndarray, dict[int, dict[str, Any]], int]:
    """Join full category labels and selected Case metadata by POI ID."""

    row_by_id = {poi_id: row for row, poi_id in enumerate(poi_ids)}
    found = np.zeros(len(poi_ids), dtype=np.bool_)
    category_ids = np.full(len(poi_ids), -1, dtype=np.int32)
    category_index: dict[str, int] = {}
    case_records: dict[int, dict[str, Any]] = {}
    remaining_rows = len(poi_ids)

    for path in _discover_poi_files(poi_data_path):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SidEvaluationError(f"{path.name}:{line_number} 是空行")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SidEvaluationError(
                        f"{path.name}:{line_number} JSON 解析失败"
                    ) from error
                if not isinstance(record, dict):
                    raise SidEvaluationError(f"{path.name}:{line_number} 不是 JSON object")
                poi_id = record.get("poi_id")
                if not isinstance(poi_id, str) or not poi_id.strip():
                    raise SidEvaluationError(f"{path.name}:{line_number} poi_id 无效")
                row = row_by_id.get(poi_id)
                if row is None:
                    continue
                if found[row]:
                    raise SidEvaluationError(f"POI 元数据存在重复 poi_id：{poi_id}")
                found[row] = True
                remaining_rows -= 1

                category_code = record.get("category_code")
                if category_code is not None and not isinstance(category_code, str):
                    raise SidEvaluationError(
                        f"{path.name}:{line_number} category_code 必须是字符串或 null"
                    )
                if isinstance(category_code, str) and category_code.strip():
                    label = category_code.strip()
                    if label not in category_index:
                        category_index[label] = len(category_index)
                    category_ids[row] = category_index[label]
                if row in selected_rows:
                    case_records[row] = _case_poi(record, poi_id)
                if remaining_rows == 0:
                    break
        if remaining_rows == 0:
            break

    missing_rows = np.flatnonzero(~found)
    if len(missing_rows):
        raise SidEvaluationError(
            f"有 {len(missing_rows)} 个 SID POI 未在 POI 元数据中找到"
        )
    if len(case_records) != len(selected_rows):
        raise SidEvaluationError("部分热点碰撞 POI 缺少 Case 元数据")
    return category_ids, case_records, len(category_index)


def compute_category_purity(
    prefix_inverse: np.ndarray,
    prefix_bucket_count: int,
    category_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute micro/macro purity while excluding missing category labels."""

    valid_mask = category_ids >= 0
    labeled_poi_count = int(valid_mask.sum())
    missing_category_count = int(len(category_ids) - labeled_poi_count)
    if labeled_poi_count == 0:
        return {
            "label_field": "category_code",
            "micro_purity": None,
            "macro_purity": None,
            "labeled_poi_count": 0,
            "missing_category_count": missing_category_count,
            "category_coverage": 0.0,
            "macro_valid_bucket_count": 0,
        }

    pairs = np.column_stack(
        (prefix_inverse[valid_mask], category_ids[valid_mask])
    )
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    labeled_per_bucket = np.bincount(
        prefix_inverse[valid_mask], minlength=prefix_bucket_count
    )
    majority_per_bucket = np.zeros(prefix_bucket_count, dtype=np.int64)
    np.maximum.at(majority_per_bucket, unique_pairs[:, 0], pair_counts)
    eligible = labeled_per_bucket > 0
    micro_purity = float(majority_per_bucket.sum() / labeled_poi_count)
    macro_purity = float(
        np.mean(majority_per_bucket[eligible] / labeled_per_bucket[eligible])
    )
    return {
        "label_field": "category_code",
        "micro_purity": micro_purity,
        "macro_purity": macro_purity,
        "labeled_poi_count": labeled_poi_count,
        "missing_category_count": missing_category_count,
        "category_coverage": labeled_poi_count / len(category_ids),
        "macro_valid_bucket_count": int(eligible.sum()),
    }


def compute_prefix_metrics(
    codes: np.ndarray,
    category_ids: np.ndarray,
    *,
    full_inverse: np.ndarray,
    full_bucket_sizes: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute bucket statistics and category purity for every prefix depth."""

    prefixes: list[dict[str, Any]] = []
    level_count = int(codes.shape[1])
    for depth in range(1, level_count + 1):
        if depth == level_count:
            prefix_inverse = full_inverse
            bucket_sizes = full_bucket_sizes
        else:
            _, prefix_inverse, bucket_sizes = np.unique(
                np.asarray(codes[:, :depth]),
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
        prefix_bucket_count = int(len(bucket_sizes))
        prefixes.append(
            {
                "depth": depth,
                "prefix_bucket_count": prefix_bucket_count,
                **_bucket_statistics(bucket_sizes),
                "category_purity": compute_category_purity(
                    prefix_inverse,
                    prefix_bucket_count,
                    category_ids,
                ),
            }
        )
    return prefixes


def _case_sort_key(poi: dict[str, Any]) -> tuple[int, float, str]:
    click_score = poi["click_score"]
    if isinstance(click_score, (int, float)) and not isinstance(click_score, bool):
        return (0, -float(click_score), str(poi["poi_id"]))
    return (1, 0.0, str(poi["poi_id"]))


def _select_case_buckets(
    unique_sids: np.ndarray,
    bucket_sizes: np.ndarray,
    max_cases: int,
) -> list[int]:
    colliding = np.flatnonzero(bucket_sizes > 1)
    return sorted(
        (int(index) for index in colliding),
        key=lambda index: (
            -int(bucket_sizes[index]),
            tuple(int(token) for token in unique_sids[index]),
        ),
    )[:max_cases]


def _selected_rows_by_bucket(
    full_inverse: np.ndarray, selected_buckets: Sequence[int]
) -> dict[int, list[int]]:
    selected = {bucket: [] for bucket in selected_buckets}
    for row, bucket_value in enumerate(full_inverse):
        bucket = int(bucket_value)
        if bucket in selected:
            selected[bucket].append(row)
    return selected


def build_collision_cases(
    unique_sids: np.ndarray,
    bucket_sizes: np.ndarray,
    rows_by_bucket: dict[int, list[int]],
    case_records: dict[int, dict[str, Any]],
    max_pois_per_case: int,
) -> list[dict[str, Any]]:
    """Build deterministic hotspot collision Cases."""

    cases: list[dict[str, Any]] = []
    for bucket, rows in rows_by_bucket.items():
        pois = sorted((case_records[row] for row in rows), key=_case_sort_key)
        output_pois = pois[:max_pois_per_case]
        cases.append(
            {
                "semantic_sid": [int(token) for token in unique_sids[bucket]],
                "bucket_size": int(bucket_sizes[bucket]),
                "output_poi_count": len(output_pois),
                "truncated": len(output_pois) < len(pois),
                "pois": output_pois,
            }
        )
    return cases


def evaluate_sid(
    manifest_path: Path,
    poi_data_path: Path,
    *,
    max_cases: int = 20,
    max_pois_per_case: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one SID artifact without writing output files."""

    if max_cases <= 0:
        raise SidEvaluationError("max_cases 必须大于 0")
    if max_pois_per_case <= 0:
        raise SidEvaluationError("max_pois_per_case 必须大于 0")

    sid_input = load_sid_input(manifest_path)
    basic, unique_sids, full_inverse, full_bucket_sizes = compute_basic_metrics(
        sid_input.codes
    )
    selected_buckets = _select_case_buckets(
        unique_sids, full_bucket_sizes, max_cases
    )
    rows_by_bucket = _selected_rows_by_bucket(full_inverse, selected_buckets)
    selected_rows = {
        row for rows in rows_by_bucket.values() for row in rows
    }
    category_ids, case_records, category_count = load_poi_metadata(
        poi_data_path.resolve(), sid_input.poi_ids, selected_rows
    )
    prefixes = compute_prefix_metrics(
        sid_input.codes,
        category_ids,
        full_inverse=full_inverse,
        full_bucket_sizes=full_bucket_sizes,
    )
    cases = build_collision_cases(
        unique_sids,
        full_bucket_sizes,
        rows_by_bucket,
        case_records,
        max_pois_per_case,
    )
    metrics = {
        "schema_version": SID_EVALUATION_SCHEMA_VERSION,
        "status": "completed",
        "inputs": {
            "manifest": str(sid_input.manifest_path),
            "sid_codes": str(sid_input.codes_path),
            "poi_ids": str(sid_input.poi_ids_path),
            "poi_data": str(poi_data_path.resolve()),
            "sid_shape": [int(value) for value in sid_input.codes.shape],
            "sid_dtype": str(sid_input.codes.dtype),
            "codebook_sizes": list(sid_input.codebook_sizes),
        },
        "validation": {
            "poi_id_rows": len(sid_input.poi_ids),
            "poi_ids_unique": True,
            "poi_metadata_rows_matched": len(sid_input.poi_ids),
            "token_ranges_valid": True,
            "category_label_field": "category_code",
            "distinct_category_count": category_count,
        },
        "basic": basic,
        "layers": compute_layer_metrics(sid_input.codes, sid_input.codebook_sizes),
        "prefixes": prefixes,
        "case_selection": {
            "order": "bucket_size_desc_then_semantic_sid_lexicographic",
            "poi_order": "click_score_desc_then_poi_id_asc",
            "max_cases": max_cases,
            "max_pois_per_case": max_pois_per_case,
            "available_colliding_bucket_count": basic["colliding_bucket_count"],
            "output_case_count": len(cases),
        },
    }
    return metrics, cases


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("metrics.json", "collision_cases.jsonl"):
        target = output_dir / filename
        if target.exists() and not target.is_file():
            raise SidEvaluationError(f"输出目标不是普通文件：{target}")


def write_evaluation_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    cases: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    """Atomically write exactly one metrics file and one Case file."""

    output_dir = output_dir.resolve()
    _prepare_output_dir(output_dir)
    metrics_path = output_dir / "metrics.json"
    cases_path = output_dir / "collision_cases.jsonl"
    metrics_temp = output_dir / ".metrics.json.tmp"
    cases_temp = output_dir / ".collision_cases.jsonl.tmp"
    try:
        with metrics_temp.open("w", encoding="utf-8") as handle:
            json.dump(
                metrics,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        with cases_temp.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(
                    json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n"
                )
        os.replace(metrics_temp, metrics_path)
        os.replace(cases_temp, cases_path)
    finally:
        metrics_temp.unlink(missing_ok=True)
        cases_temp.unlink(missing_ok=True)
    return metrics_path, cases_path


def run_sid_evaluation(
    manifest_path: Path,
    poi_data_path: Path,
    output_dir: Path,
    *,
    max_cases: int = 20,
    max_pois_per_case: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a SID artifact and write the two allowed output files."""

    metrics, cases = evaluate_sid(
        manifest_path,
        poi_data_path,
        max_cases=max_cases,
        max_pois_per_case=max_pois_per_case,
    )
    write_evaluation_outputs(output_dir, metrics, cases)
    return metrics, cases
