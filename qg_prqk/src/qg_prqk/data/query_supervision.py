"""Build and validate Train-only category-aware query supervision."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.data.query_supervision_config import (
    CategoryBuildConfig,
    DepthThreshold,
    SensitivitySetting,
)
from qg_prqk.data.query_statistics import QUERY_POI_SCHEMA, QUERY_STATS_SCHEMA

try:
    import orjson
except ImportError:  # pragma: no cover - production environment includes orjson
    orjson = None


SCHEMA_VERSION = "qg-prqk-category-query-depth-v1"
GATE_SCHEMA_VERSION = "qg-prqk-p2-5-cat-gate-v1"
DEPTH_LABELS = ("D0_CONTEXTUAL", "D1_COARSE", "D2_FINE", "D3_EXACT")
ASCII_CATEGORY_CODE = re.compile(r"[0-9]{6}")
_SHOW_PROGRESS = True

CATEGORY_MAPPING_SCHEMA = pa.schema(
    [
        pa.field("poi_row_index", pa.int64(), nullable=False),
        pa.field("poi_id", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("fine_category_id", pa.string(), nullable=False),
        pa.field("fine_category_index", pa.int32(), nullable=False),
        pa.field("coarse_category_id", pa.string(), nullable=False),
        pa.field("coarse_category_index", pa.int16(), nullable=False),
    ]
)

QUERY_CATEGORY_STATS_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("query_shard_id", pa.int32(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("representative_raw_query", pa.string(), nullable=False),
        pa.field("query_count", pa.int64(), nullable=False),
        pa.field("distinct_poi_count", pa.int32(), nullable=False),
        pa.field("top1_poi_id", pa.string(), nullable=False),
        pa.field("top1_count", pa.int64(), nullable=False),
        pa.field("top1_share", pa.float64(), nullable=False),
        pa.field("exact_normalized_entropy", pa.float64(), nullable=False),
        pa.field("is_p2_exact_core", pa.bool_(), nullable=False),
        pa.field("fine_distinct_category_count", pa.int16(), nullable=False),
        pa.field("dominant_fine_category_id", pa.string(), nullable=False),
        pa.field("dominant_fine_count", pa.int64(), nullable=False),
        pa.field("fine_concentration", pa.float64(), nullable=False),
        pa.field("fine_entropy", pa.float64(), nullable=False),
        pa.field("fine_normalized_entropy", pa.float64(), nullable=False),
        pa.field("coarse_distinct_category_count", pa.int16(), nullable=False),
        pa.field("dominant_coarse_category_id", pa.string(), nullable=False),
        pa.field("dominant_coarse_count", pa.int64(), nullable=False),
        pa.field("coarse_concentration", pa.float64(), nullable=False),
        pa.field("coarse_entropy", pa.float64(), nullable=False),
        pa.field("coarse_normalized_entropy", pa.float64(), nullable=False),
    ]
)

QUERY_CATEGORY_DEPTH_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("query_shard_id", pa.int32(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("representative_raw_query", pa.string(), nullable=False),
        pa.field("query_count", pa.int64(), nullable=False),
        pa.field("supervision_depth", pa.int8(), nullable=False),
        pa.field("supervision_label", pa.string(), nullable=False),
        pa.field("supervise_s1", pa.bool_(), nullable=False),
        pa.field("supervise_s2", pa.bool_(), nullable=False),
        pa.field("supervise_s3", pa.bool_(), nullable=False),
        pa.field("dominant_fine_category_id", pa.string(), nullable=False),
        pa.field("dominant_coarse_category_id", pa.string(), nullable=False),
        pa.field("support_score", pa.float64(), nullable=False),
        pa.field("coarse_reliability", pa.float64(), nullable=False),
        pa.field("fine_reliability", pa.float64(), nullable=False),
        pa.field("exact_reliability", pa.float64(), nullable=False),
        pa.field("s1_reliability", pa.float64(), nullable=False),
        pa.field("s2_reliability", pa.float64(), nullable=False),
        pa.field("s3_reliability", pa.float64(), nullable=False),
        pa.field("retained_poi_edge_count", pa.int32(), nullable=False),
        pa.field("retained_layer_edge_count", pa.int32(), nullable=False),
    ]
)

QUERY_POI_LAYER_EDGE_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("query_shard_id", pa.int32(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
        pa.field("pair_count", pa.int64(), nullable=False),
        pa.field("pair_share", pa.float64(), nullable=False),
        pa.field("supervision_depth", pa.int8(), nullable=False),
        pa.field("supervision_label", pa.string(), nullable=False),
        pa.field("layer", pa.int8(), nullable=False),
        pa.field("hierarchy_source", pa.string(), nullable=False),
        pa.field("fine_category_id", pa.string(), nullable=False),
        pa.field("coarse_category_id", pa.string(), nullable=False),
        pa.field("edge_probability", pa.float64(), nullable=False),
        pa.field("query_reliability", pa.float64(), nullable=False),
        pa.field("edge_weight", pa.float64(), nullable=False),
    ]
)


class CategoryDepthError(RuntimeError):
    """Raised when P2.5-CAT input or output invariants fail."""


@dataclass(frozen=True)
class PairObservation:
    poi_id: str
    pair_count: int
    pair_share: float
    is_top1: bool


@dataclass(frozen=True)
class DistributionSummary:
    distinct_count: int
    dominant_id: str
    dominant_count: int
    concentration: float
    entropy: float
    normalized_entropy: float


@dataclass(frozen=True)
class QueryComputation:
    depth: int
    support_score: float
    fine: DistributionSummary
    coarse: DistributionSummary
    coarse_reliability: float
    fine_reliability: float
    exact_reliability: float
    retained_pairs: tuple[PairObservation, ...]
    layers: tuple[int, ...]
    layer_reliabilities: tuple[float, ...]
    hierarchy_source: str


@dataclass(frozen=True)
class CatalogSource:
    path: Path
    expected_rows: int
    expected_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class FrozenInputBundle:
    p2_manifest: Mapping[str, Any]
    p2_manifest_path: Path
    p2_num_shards: int
    query_stats_paths: tuple[Path, ...]
    query_pair_paths: tuple[Path, ...]
    category_indices: np.ndarray
    fine_codes: tuple[str, ...]
    coarse_codes: tuple[str, ...]
    coarse_index_by_id: Mapping[str, int]
    vocab_data_path: Path
    bge_manifest: Mapping[str, Any]
    category_source_manifest: Mapping[str, Any]
    catalog_sources: tuple[CatalogSource, ...]


@dataclass(frozen=True)
class CategoryDepthBuildResult:
    output_dir: Path
    manifest_path: Path
    category_rows: int
    unique_queries: int
    layer_edges: int
    depth_counts: Mapping[str, int]
    reused: bool


def set_progress_enabled(enabled: bool) -> None:
    """Enable or disable CLI progress bars."""

    global _SHOW_PROGRESS
    _SHOW_PROGRESS = enabled


def normalized_entropy(counts: Sequence[int]) -> tuple[float, float]:
    """Return natural-log entropy and its category-count normalization."""

    total = sum(counts)
    if total <= 0 or any(count <= 0 for count in counts):
        raise CategoryDepthError("类别计数必须全部为正且总数大于 0")
    probabilities = [count / total for count in counts]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return entropy, normalized


def summarize_distribution(counts: Mapping[str, int]) -> DistributionSummary:
    """Summarize a categorical target distribution with deterministic ties."""

    if not counts or any(count <= 0 for count in counts.values()):
        raise CategoryDepthError("类别分布必须包含正计数")
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counts.values())
    entropy, normalized = normalized_entropy([count for _, count in ranked])
    return DistributionSummary(
        distinct_count=len(ranked),
        dominant_id=ranked[0][0],
        dominant_count=ranked[0][1],
        concentration=ranked[0][1] / total,
        entropy=entropy,
        normalized_entropy=normalized,
    )


def capped_log_support(query_count: int, cap: int) -> float:
    """Map Query support to [0,1] using normalized capped log1p."""

    if query_count <= 0 or cap <= 0:
        raise CategoryDepthError("query_count 与 support cap 必须大于 0")
    return math.log1p(min(query_count, cap)) / math.log1p(cap)


def layer_reliability(
    support_score: float,
    concentration: float,
    normalized_distribution_entropy: float,
    gamma: float,
) -> float:
    """Compute the frozen hierarchy reliability for one Query layer."""

    if not (
        0.0 <= support_score <= 1.0
        and 0.0 <= concentration <= 1.0
        and 0.0 <= normalized_distribution_entropy <= 1.0 + 1e-12
        and gamma > 0.0
    ):
        raise CategoryDepthError("reliability 输入越界")
    value = (
        support_score
        * concentration**gamma
        * max(0.0, 1.0 - normalized_distribution_entropy)
    )
    return min(1.0, max(0.0, value))


def passes_threshold(
    query_count: int,
    distribution: DistributionSummary,
    threshold: DepthThreshold,
) -> bool:
    return (
        query_count >= threshold.min_query_count
        and distribution.concentration >= threshold.min_concentration
        and distribution.normalized_entropy <= threshold.max_normalized_entropy
    )


def classify_depth(
    *,
    is_p2_exact_core: bool,
    query_count: int,
    fine: DistributionSummary,
    coarse: DistributionSummary,
    d2_fine: DepthThreshold,
    d1_coarse: DepthThreshold,
) -> int:
    """Apply the frozen D3, then D2, then D1, then D0 priority."""

    if is_p2_exact_core:
        return 3
    if passes_threshold(query_count, fine, d2_fine):
        return 2
    if passes_threshold(query_count, coarse, d1_coarse):
        return 1
    return 0


def compute_query_supervision(
    stats: Mapping[str, Any],
    pairs: Sequence[PairObservation],
    poi_categories: Mapping[str, tuple[int, int, int]],
    fine_codes: Sequence[str],
    coarse_codes: Sequence[str],
    config: CategoryBuildConfig,
) -> QueryComputation:
    """Compute category distributions, depth, retained pairs, and reliabilities."""

    query_count = int(stats["query_count"])
    if not pairs:
        raise CategoryDepthError(f"Query {stats['query_id']} 没有 Query–POI pair")
    if sum(pair.pair_count for pair in pairs) != query_count:
        raise CategoryDepthError(f"Query {stats['query_id']} pair_count 不守恒")
    if abs(sum(pair.pair_share for pair in pairs) - 1.0) > 1e-9:
        raise CategoryDepthError(f"Query {stats['query_id']} pair_share 不归一")

    fine_counts: Counter[str] = Counter()
    coarse_counts: Counter[str] = Counter()
    enriched: list[tuple[PairObservation, str, str]] = []
    for pair in pairs:
        try:
            _, fine_index, coarse_index = poi_categories[pair.poi_id]
        except KeyError as error:
            raise CategoryDepthError(f"P2 POI 缺少类别映射：{pair.poi_id}") from error
        try:
            fine_id = fine_codes[fine_index]
            coarse_id = coarse_codes[coarse_index]
        except IndexError as error:
            raise CategoryDepthError("POI 类别 index 越界") from error
        fine_counts[fine_id] += pair.pair_count
        coarse_counts[coarse_id] += pair.pair_count
        enriched.append((pair, fine_id, coarse_id))

    fine = summarize_distribution(fine_counts)
    coarse = summarize_distribution(coarse_counts)
    depth = classify_depth(
        is_p2_exact_core=bool(stats["is_high_confidence"]),
        query_count=query_count,
        fine=fine,
        coarse=coarse,
        d2_fine=config.query_depth.d2_fine,
        d1_coarse=config.query_depth.d1_coarse,
    )
    support_score = capped_log_support(query_count, config.query_depth.support_cap)
    coarse_reliability = layer_reliability(
        support_score,
        coarse.concentration,
        coarse.normalized_entropy,
        config.query_depth.reliability_gamma,
    )
    fine_reliability = layer_reliability(
        support_score,
        fine.concentration,
        fine.normalized_entropy,
        config.query_depth.reliability_gamma,
    )
    exact_reliability = layer_reliability(
        support_score,
        float(stats["top1_share"]),
        float(stats["normalized_entropy"]),
        config.query_depth.reliability_gamma,
    )

    if depth == 3:
        retained = tuple(pair for pair in pairs if pair.poi_id == stats["top1_poi_id"])
        if len(retained) != 1 or not retained[0].is_top1:
            raise CategoryDepthError(f"D3 Query {stats['query_id']} top1 边不唯一")
        layers = (1, 2, 3)
        reliabilities = (coarse_reliability, fine_reliability, exact_reliability)
        source = "p2_exact_core_top1"
    elif depth == 2:
        retained = tuple(
            pair for pair, fine_id, _ in enriched if fine_id == fine.dominant_id
        )
        layers = (1, 2)
        reliabilities = (coarse_reliability, fine_reliability)
        source = "dominant_fine_category"
    elif depth == 1:
        retained = tuple(
            pair for pair, _, coarse_id in enriched if coarse_id == coarse.dominant_id
        )
        layers = (1,)
        reliabilities = (coarse_reliability,)
        source = "dominant_coarse_category"
    else:
        retained = ()
        layers = ()
        reliabilities = ()
        source = "none"
    if depth > 0 and not retained:
        raise CategoryDepthError(f"Query {stats['query_id']} 没有保留监督边")
    return QueryComputation(
        depth=depth,
        support_score=support_score,
        fine=fine,
        coarse=coarse,
        coarse_reliability=coarse_reliability,
        fine_reliability=fine_reliability,
        exact_reliability=exact_reliability,
        retained_pairs=tuple(sorted(retained, key=lambda pair: pair.poi_id)),
        layers=layers,
        layer_reliabilities=reliabilities,
        hierarchy_source=source,
    )


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CategoryDepthError(f"{label} 读取失败：{path}") from error
    if not isinstance(value, Mapping):
        raise CategoryDepthError(f"{label} 必须是 JSON object")
    return value


def _parse_json_line(raw_line: bytes, source: str) -> Any:
    try:
        return orjson.loads(raw_line) if orjson is not None else json.loads(raw_line)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise CategoryDepthError(f"{source} JSON 非法") from error


def _file_info(path: Path, *, shard_id: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "file": path.name,
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if shard_id is not None:
        info["shard_id"] = shard_id
    return info


def _write_rows(path: Path, schema: pa.Schema, rows: Sequence[Mapping[str, Any]]) -> None:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=max(1, min(65_536, len(table))),
    )


def _validate_declared_group(
    p2_dir: Path,
    raw: Any,
    expected_schema: pa.Schema,
    expected_parts: int,
    label: str,
) -> tuple[tuple[Path, ...], int]:
    group = raw if isinstance(raw, Mapping) else None
    if group is None or not isinstance(group.get("files"), list):
        raise CategoryDepthError(f"P2 {label} 输出清单非法")
    files = group["files"]
    if len(files) != expected_parts:
        raise CategoryDepthError(f"P2 {label} 分片数不等于 {expected_parts}")
    directory = p2_dir / str(group.get("dir", ""))
    paths: list[Path] = []
    rows = 0
    for expected_shard, entry in enumerate(files):
        if not isinstance(entry, Mapping) or int(entry.get("shard_id", -1)) != expected_shard:
            raise CategoryDepthError(f"P2 {label} shard 清单顺序非法")
        file_name = str(entry.get("file", ""))
        if not file_name or Path(file_name).name != file_name:
            raise CategoryDepthError(f"P2 {label} 文件名非法")
        path = directory / file_name
        if not path.is_file():
            raise CategoryDepthError(f"P2 {label} 文件不存在：{path}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise CategoryDepthError(f"P2 {label} 文件大小变化：{path}")
        if sha256_file(path) != entry.get("sha256"):
            raise CategoryDepthError(f"P2 {label} SHA256 变化：{path}")
        schema = pq.read_schema(path)
        if not schema.equals(expected_schema, check_metadata=False):
            raise CategoryDepthError(f"P2 {label} schema 不一致：{path}")
        file_rows = pq.ParquetFile(path).metadata.num_rows
        if file_rows != int(entry.get("rows", -1)):
            raise CategoryDepthError(f"P2 {label} 行数变化：{path}")
        paths.append(path)
        rows += file_rows
    if rows != int(group.get("rows", -1)):
        raise CategoryDepthError(f"P2 {label} 总行数与 manifest 不一致")
    return tuple(paths), rows


def _validate_p2(config: CategoryBuildConfig) -> tuple[
    Mapping[str, Any], int, tuple[Path, ...], tuple[Path, ...]
]:
    p2_dir = config.paths.p2_query_stats
    manifest_path = p2_dir / "manifest.json"
    if not manifest_path.is_file() or not (p2_dir / "_SUCCESS").is_file():
        raise CategoryDepthError("冻结 P2 缺少 manifest.json 或 _SUCCESS")
    if sha256_file(manifest_path) != config.frozen.p2_manifest_sha256:
        raise CategoryDepthError("冻结 P2 manifest SHA256 不一致")
    manifest = _load_json(manifest_path, "P2 manifest")
    if (
        manifest.get("schema_version") != "qg-prqk-query-stats-v1"
        or manifest.get("status") != "completed"
    ):
        raise CategoryDepthError("冻结 P2 manifest 版本或状态非法")
    source = manifest.get("source")
    stats = manifest.get("stats")
    settings = manifest.get("settings")
    outputs = manifest.get("outputs")
    if not all(isinstance(item, Mapping) for item in (source, stats, settings, outputs)):
        raise CategoryDepthError("冻结 P2 manifest 缺少 source/stats/settings/outputs")
    assert isinstance(source, Mapping)
    assert isinstance(stats, Mapping)
    assert isinstance(settings, Mapping)
    assert isinstance(outputs, Mapping)
    if (
        source.get("split") != "train"
        or source.get("limit") is not None
        or source.get("is_prefix_sample") is not False
        or source.get("valid_and_test_read") is not False
    ):
        raise CategoryDepthError("P2.5-CAT 只接受完整 Train-only P2")
    if int(source.get("scanned_rows", -1)) != int(stats.get("source_rows", -2)):
        raise CategoryDepthError("P2 source_rows 不守恒")
    num_shards = int(settings.get("num_shards", 0))
    if num_shards <= 0:
        raise CategoryDepthError("P2 num_shards 非法")
    query_paths, query_rows = _validate_declared_group(
        p2_dir,
        outputs.get("query_stats"),
        QUERY_STATS_SCHEMA,
        num_shards,
        "query_stats",
    )
    pair_paths, pair_rows = _validate_declared_group(
        p2_dir,
        outputs.get("query_poi_pairs"),
        QUERY_POI_SCHEMA,
        num_shards,
        "query_poi_pairs",
    )
    if query_rows != int(stats.get("unique_queries", -1)):
        raise CategoryDepthError("P2 unique_queries 与 Parquet 不一致")
    if pair_rows != int(stats.get("unique_query_poi_pairs", -1)):
        raise CategoryDepthError("P2 unique_query_poi_pairs 与 Parquet 不一致")
    return manifest, num_shards, query_paths, pair_paths


def _single_vocab_file(path: Path) -> Path:
    if path.is_file() and path.suffix == ".parquet":
        return path
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else []
    if len(files) != 1:
        raise CategoryDepthError(f"category vocab 必须恰有一个 Parquet 数据文件：{path}")
    return files[0]


def _validate_category_assets(
    config: CategoryBuildConfig,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], dict[str, int], Path, Mapping[str, Any]]:
    indices_path = config.paths.category_indices
    if not indices_path.is_file():
        raise CategoryDepthError(f"category_indices 不存在：{indices_path}")
    if sha256_file(indices_path) != config.frozen.category_indices_sha256:
        raise CategoryDepthError("category_indices SHA256 不一致")
    try:
        indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CategoryDepthError("category_indices NPY 无法读取") from error
    if indices.shape != (config.frozen.poi_rows,) or indices.dtype != np.int32:
        raise CategoryDepthError(
            f"category_indices 期望 int32[{config.frozen.poi_rows}]，实际 {indices.dtype}{indices.shape}"
        )

    vocab_path = _single_vocab_file(config.paths.category_vocab)
    if sha256_file(vocab_path) != config.frozen.category_vocab_data_sha256:
        raise CategoryDepthError("category vocab 数据 SHA256 不一致")
    vocab_schema = pq.read_schema(vocab_path)
    required = {"category_code": pa.string(), "category_index": pa.int64()}
    for name, expected_type in required.items():
        if name not in vocab_schema.names or vocab_schema.field(name).type != expected_type:
            raise CategoryDepthError(f"category vocab 字段 {name} 缺失或类型错误")
    rows = pq.read_table(vocab_path, columns=list(required)).to_pylist()
    if len(rows) != config.category.fine_expected_count:
        raise CategoryDepthError("category vocab 行数不是冻结 fine category 数")
    by_index: dict[int, str] = {}
    for row in rows:
        code = row["category_code"]
        index = row["category_index"]
        if (
            not isinstance(code, str)
            or ASCII_CATEGORY_CODE.fullmatch(code) is None
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index in by_index
        ):
            raise CategoryDepthError("category vocab code/index 非法或重复")
        by_index[index] = code
    expected_indices = set(range(config.category.fine_expected_count))
    if set(by_index) != expected_indices or len(set(by_index.values())) != len(by_index):
        raise CategoryDepthError("category vocab index 必须连续且 code 必须唯一")
    fine_codes = tuple(by_index[index] for index in range(len(by_index)))
    coarse_codes = tuple(sorted({code[:2] for code in fine_codes}))
    if len(coarse_codes) != config.category.coarse_expected_count:
        raise CategoryDepthError("category vocab coarse category 数不一致")
    coarse_index = {code: index for index, code in enumerate(coarse_codes)}
    unique_indices = np.unique(indices)
    if (
        len(unique_indices) == 0
        or int(unique_indices[0]) < 0
        or int(unique_indices[-1]) >= config.category.fine_expected_count
    ):
        raise CategoryDepthError("category_indices 含冻结 fine vocab 范围外的 index")

    source_manifest_path = config.paths.category_source_manifest
    if sha256_file(source_manifest_path) != config.frozen.category_source_manifest_sha256:
        raise CategoryDepthError("category source manifest SHA256 不一致")
    source_manifest = _load_json(source_manifest_path, "category source manifest")
    source_stats = source_manifest.get("stats")
    source_sha = source_manifest.get("sha256")
    if (
        source_manifest.get("status") != "completed"
        or source_manifest.get("row_order") != config.category.mapping_row_order
        or not isinstance(source_stats, Mapping)
        or not isinstance(source_sha, Mapping)
        or int(source_stats.get("output_rows", -1)) != config.frozen.poi_rows
        or int(source_stats.get("source_embedding_rows", -1)) != config.frozen.poi_rows
        or int(source_stats.get("source_feature_rows", -1)) != config.frozen.poi_rows
        or int(source_stats.get("category_vocab_count", -1))
        != config.category.fine_expected_count
        or int(source_stats.get("invalid_category_count", -1)) != 0
        or source_sha.get("category_indices")
        != config.frozen.category_indices_sha256
    ):
        raise CategoryDepthError("category source manifest 合同不一致")
    return (
        indices,
        fine_codes,
        coarse_codes,
        coarse_index,
        vocab_path,
        source_manifest,
    )


def _validate_bge_and_catalog(
    config: CategoryBuildConfig,
) -> tuple[Mapping[str, Any], tuple[CatalogSource, ...]]:
    manifest_path = config.paths.embedding_manifest
    if sha256_file(manifest_path) != config.frozen.poi_embedding_manifest_sha256:
        raise CategoryDepthError("BGE manifest SHA256 不一致")
    manifest = _load_json(manifest_path, "BGE manifest")
    output = manifest.get("output")
    model = manifest.get("model")
    input_info = manifest.get("input")
    if not all(isinstance(item, Mapping) for item in (output, model, input_info)):
        raise CategoryDepthError("BGE manifest 缺少 output/model/input")
    assert isinstance(output, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(input_info, Mapping)
    if (
        manifest.get("status") != "completed"
        or output.get("shape")
        != [config.frozen.poi_rows, config.frozen.poi_embedding_dim]
        or output.get("dtype") != config.frozen.poi_embedding_dtype
        or model.get("normalize_embeddings")
        is not config.frozen.poi_embedding_normalized
        or int(input_info.get("total_rows", -1)) != config.frozen.poi_rows
        or input_info.get("id_field") != "poi_id"
    ):
        raise CategoryDepthError("BGE manifest shape/dtype/normalization 合同不一致")
    try:
        embeddings = np.load(config.paths.poi_embeddings, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CategoryDepthError("BGE embedding NPY header 无法读取") from error
    if embeddings.shape != (
        config.frozen.poi_rows,
        config.frozen.poi_embedding_dim,
    ) or embeddings.dtype != np.dtype(config.frozen.poi_embedding_dtype):
        raise CategoryDepthError("BGE embedding NPY shape/dtype 与冻结配置不一致")
    del embeddings
    if not config.paths.poi_ids.is_file():
        raise CategoryDepthError(f"BGE poi_ids 不存在：{config.paths.poi_ids}")

    raw_sources = input_info.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise CategoryDepthError("BGE manifest 未登记 POI catalog sources")
    sources: list[CatalogSource] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise CategoryDepthError("BGE catalog source 清单非法")
        name = str(raw.get("name", ""))
        if not name or Path(name).name != name:
            raise CategoryDepthError("BGE catalog source 文件名非法")
        path = config.paths.poi_catalog / name
        expected_bytes = int(raw.get("bytes_scanned", -1))
        expected_rows = int(raw.get("rows_scanned", -1))
        expected_sha = str(raw.get("sha256_scanned", ""))
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or expected_rows <= 0
            or len(expected_sha) != 64
        ):
            raise CategoryDepthError(f"POI catalog source 缺失或大小变化：{path}")
        sources.append(
            CatalogSource(
                path=path,
                expected_rows=expected_rows,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha,
            )
        )
    if sum(source.expected_rows for source in sources) != config.frozen.poi_rows:
        raise CategoryDepthError("BGE catalog source 行数之和不一致")
    return manifest, tuple(sources)


def preflight_category_depth(config: CategoryBuildConfig) -> FrozenInputBundle:
    """Validate every frozen P2.5 input without reading raw order splits."""

    p2_manifest, num_shards, query_paths, pair_paths = _validate_p2(config)
    (
        indices,
        fine_codes,
        coarse_codes,
        coarse_index,
        vocab_path,
        source_manifest,
    ) = _validate_category_assets(config)
    bge_manifest, catalog_sources = _validate_bge_and_catalog(config)
    return FrozenInputBundle(
        p2_manifest=p2_manifest,
        p2_manifest_path=config.paths.p2_query_stats / "manifest.json",
        p2_num_shards=num_shards,
        query_stats_paths=query_paths,
        query_pair_paths=pair_paths,
        category_indices=indices,
        fine_codes=fine_codes,
        coarse_codes=coarse_codes,
        coarse_index_by_id=coarse_index,
        vocab_data_path=vocab_path,
        bge_manifest=bge_manifest,
        category_source_manifest=source_manifest,
        catalog_sources=catalog_sources,
    )


def implementation_signature() -> dict[str, Any]:
    """Fingerprint the query-supervision implementation used by each build."""

    source_dir = Path(__file__).resolve().parent
    package_dir = source_dir.parent
    files = [
        source_dir / "query_supervision_config.py",
        source_dir / "query_supervision.py",
        package_dir / "commands/query_supervision.py",
    ]
    if not all(path.is_file() for path in files):
        raise CategoryDepthError("Query supervision implementation 文件不完整")
    entries = [
        {
            "file": path.relative_to(package_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"signature": digest, "files": entries}


def _pair_observation(row: Mapping[str, Any]) -> PairObservation:
    poi_id = row.get("target_poi_id")
    pair_count = row.get("pair_count")
    pair_share = row.get("pair_share")
    is_top1 = row.get("is_top1")
    if (
        not isinstance(poi_id, str)
        or not poi_id
        or isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or pair_count <= 0
        or isinstance(pair_share, bool)
        or not isinstance(pair_share, (int, float))
        or not 0.0 < float(pair_share) <= 1.0
        or not isinstance(is_top1, bool)
    ):
        raise CategoryDepthError("P2 Query–POI pair 字段非法")
    return PairObservation(
        poi_id=poi_id,
        pair_count=pair_count,
        pair_share=float(pair_share),
        is_top1=is_top1,
    )


def _load_query_prefix(
    bundle: FrozenInputBundle,
    limit: int,
) -> tuple[list[Mapping[str, Any]], dict[int, list[PairObservation]]]:
    total_queries = int(bundle.p2_manifest["stats"]["unique_queries"])
    if limit <= 0 or limit > total_queries:
        raise CategoryDepthError(f"gate query limit 必须位于 [1,{total_queries}]")
    stats_rows: list[Mapping[str, Any]] = []
    pairs_by_query: dict[int, list[PairObservation]] = defaultdict(list)
    expected_query_id = 0
    for stats_path, pair_path in zip(
        bundle.query_stats_paths, bundle.query_pair_paths, strict=True
    ):
        remaining = limit - len(stats_rows)
        if remaining <= 0:
            break
        shard_rows = pq.read_table(stats_path).to_pylist()
        selected = shard_rows[:remaining]
        selected_ids: set[int] = set()
        query_text: dict[int, str] = {}
        for row in selected:
            query_id = int(row["query_id"])
            if query_id != expected_query_id:
                raise CategoryDepthError("P2 query_id 不是从 0 开始连续递增")
            expected_query_id += 1
            selected_ids.add(query_id)
            query_text[query_id] = str(row["normalized_query"])
            stats_rows.append(row)
        if selected_ids:
            for row in pq.read_table(pair_path).to_pylist():
                query_id = int(row["query_id"])
                if query_id not in selected_ids:
                    continue
                if row["normalized_query"] != query_text[query_id]:
                    raise CategoryDepthError("P2 Query stats/pair normalized_query 不一致")
                pairs_by_query[query_id].append(_pair_observation(row))
    if len(stats_rows) != limit or len(pairs_by_query) != limit:
        raise CategoryDepthError("P2 gate Query 或 pair 数不足")
    return stats_rows, pairs_by_query


def _scan_poi_ids_for_targets(
    config: CategoryBuildConfig,
    bundle: FrozenInputBundle,
    targets: set[str],
) -> dict[str, tuple[int, int, int]]:
    mapped: dict[str, tuple[int, int, int]] = {}
    digest = hashlib.sha256()
    row_count = 0
    with config.paths.poi_ids.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            digest.update(raw_line)
            poi_id = _parse_json_line(raw_line, f"poi_ids:{row_index + 1}")
            if not isinstance(poi_id, str) or not poi_id:
                raise CategoryDepthError(f"poi_ids 第 {row_index + 1} 行不是非空字符串")
            if row_index >= config.frozen.poi_rows:
                raise CategoryDepthError("poi_ids 行数超过冻结 POI 数")
            if poi_id in targets:
                if poi_id in mapped:
                    raise CategoryDepthError(f"P2 target POI 在 BGE 行映射中重复：{poi_id}")
                fine_index = int(bundle.category_indices[row_index])
                fine_id = bundle.fine_codes[fine_index]
                coarse_index = bundle.coarse_index_by_id[fine_id[:2]]
                mapped[poi_id] = (len(mapped), fine_index, coarse_index)
            row_count += 1
    if row_count != config.frozen.poi_rows:
        raise CategoryDepthError(
            f"poi_ids 行数 {row_count} != {config.frozen.poi_rows}"
        )
    if digest.hexdigest() != config.frozen.poi_ids_sha256:
        raise CategoryDepthError("poi_ids SHA256 不一致")
    missing = targets.difference(mapped)
    if missing:
        sample = sorted(missing)[:5]
        raise CategoryDepthError(f"P2 target POI 不在 BGE 行映射：{sample}")
    return mapped


def _depth_counts_for_rows(
    stats_rows: Sequence[Mapping[str, Any]],
    pairs_by_query: Mapping[int, Sequence[PairObservation]],
    poi_categories: Mapping[str, tuple[int, int, int]],
    bundle: FrozenInputBundle,
    config: CategoryBuildConfig,
) -> dict[str, Any]:
    counts = Counter()
    edge_queries = 0
    for stats in stats_rows:
        query_id = int(stats["query_id"])
        computation = compute_query_supervision(
            stats,
            pairs_by_query[query_id],
            poi_categories,
            bundle.fine_codes,
            bundle.coarse_codes,
            config,
        )
        counts[DEPTH_LABELS[computation.depth]] += 1
        edge_queries += int(bool(computation.retained_pairs))
    total = len(stats_rows)
    return {
        "query_count": total,
        "depth_counts": {label: counts[label] for label in DEPTH_LABELS},
        "depth_ratios": {
            label: counts[label] / total if total else 0.0 for label in DEPTH_LABELS
        },
        "queries_with_retained_edges": edge_queries,
    }


def preview_category_depth(
    config: CategoryBuildConfig,
    *,
    query_limit: int,
) -> dict[str, Any]:
    """Run a bounded no-write P2.5 preview over frozen P2 Query IDs."""

    bundle = preflight_category_depth(config)
    stats_rows, pairs_by_query = _load_query_prefix(bundle, query_limit)
    targets = {
        pair.poi_id
        for pairs in pairs_by_query.values()
        for pair in pairs
    }
    poi_categories = _scan_poi_ids_for_targets(config, bundle, targets)
    summary = _depth_counts_for_rows(
        stats_rows, pairs_by_query, poi_categories, bundle, config
    )
    return {
        "status": "dry_run_passed",
        "writes": False,
        "query_limit": query_limit,
        "target_poi_count": len(targets),
        "p2_manifest_sha256": config.frozen.p2_manifest_sha256,
        "category_indices_sha256": config.frozen.category_indices_sha256,
        "poi_ids_sha256": config.frozen.poi_ids_sha256,
        "source_access": {
            "p2_query_stats_read": True,
            "p2_query_poi_pairs_read": True,
            "raw_train_read": False,
            "validation_read": False,
            "test_read": False,
            "poi_catalog_content_read": False,
        },
        **summary,
    }


def _gate_path(config: CategoryBuildConfig, level: str) -> Path:
    if level not in {"sample", "medium"}:
        raise CategoryDepthError("gate level 只能是 sample 或 medium")
    return config.gate_dir / f"{level}.json"


def _gate_limit(config: CategoryBuildConfig, level: str) -> int:
    return (
        config.runtime.sample_query_limit
        if level == "sample"
        else config.runtime.medium_query_limit
    )


def _validate_gate_payload(
    payload: Mapping[str, Any],
    config: CategoryBuildConfig,
    level: str,
    code: Mapping[str, Any],
) -> None:
    expected_limit = _gate_limit(config, level)
    if (
        payload.get("schema_version") != GATE_SCHEMA_VERSION
        or payload.get("status") != "passed"
        or payload.get("level") != level
        or int(payload.get("query_limit", -1)) != expected_limit
        or payload.get("config_signature") != config.signature()
        or payload.get("p2_manifest_sha256") != config.frozen.p2_manifest_sha256
        or payload.get("category_indices_sha256")
        != config.frozen.category_indices_sha256
        or payload.get("poi_ids_sha256") != config.frozen.poi_ids_sha256
        or payload.get("implementation_signature") != code.get("signature")
    ):
        raise CategoryDepthError(f"{level} gate 与当前配置/代码/输入不一致")
    access = payload.get("source_access")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in ("raw_train_read", "validation_read", "test_read")
    ):
        raise CategoryDepthError(f"{level} gate 缺少 Train-only 隔离证据")


def run_category_depth_gate(
    config: CategoryBuildConfig,
    *,
    level: str,
) -> dict[str, Any]:
    """Run and persist one deterministic sample/medium P2.5 gate."""

    path = _gate_path(config, level)
    code = implementation_signature()
    if path.exists():
        payload = _load_json(path, f"{level} gate")
        _validate_gate_payload(payload, config, level, code)
        return dict(payload) | {"reused": True}
    preview = preview_category_depth(
        config,
        query_limit=_gate_limit(config, level),
    )
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "passed",
        "built_at": utc_now(),
        "level": level,
        "query_limit": preview["query_limit"],
        "target_poi_count": preview["target_poi_count"],
        "query_count": preview["query_count"],
        "depth_counts": preview["depth_counts"],
        "depth_ratios": preview["depth_ratios"],
        "queries_with_retained_edges": preview["queries_with_retained_edges"],
        "config_path": str(config.source_path),
        "config_sha256": config.source_sha256,
        "config_signature": config.signature(),
        "p2_manifest_sha256": config.frozen.p2_manifest_sha256,
        "category_indices_sha256": config.frozen.category_indices_sha256,
        "poi_ids_sha256": config.frozen.poi_ids_sha256,
        "implementation_signature": code["signature"],
        "implementation_files": code["files"],
        "source_access": preview["source_access"],
        "reused": False,
    }
    write_json_atomic(path, payload)
    return payload


def require_gate_evidence(
    config: CategoryBuildConfig,
) -> dict[str, dict[str, Any]]:
    """Validate both required gate artifacts against current code and inputs."""

    if not config.runtime.require_sample_and_medium_gate_before_full:
        return {}
    code = implementation_signature()
    evidence: dict[str, dict[str, Any]] = {}
    for level in ("sample", "medium"):
        path = _gate_path(config, level)
        if not path.is_file():
            raise CategoryDepthError(f"full 前缺少 {level} gate：{path}")
        payload = _load_json(path, f"{level} gate")
        _validate_gate_payload(payload, config, level, code)
        evidence[level] = {
            "file": str(path),
            "sha256": sha256_file(path),
            "query_limit": int(payload["query_limit"]),
            "depth_counts": payload["depth_counts"],
        }
    return evidence


def _collect_p2_target_pois(bundle: FrozenInputBundle) -> set[str]:
    targets: set[str] = set()
    rows = 0
    for path in tqdm(
        bundle.query_pair_paths,
        desc="Collect P2 target POIs",
        unit="shard",
        disable=not _SHOW_PROGRESS,
    ):
        column = pq.read_table(path, columns=["target_poi_id"])["target_poi_id"]
        values = column.to_pylist()
        if any(not isinstance(value, str) or not value for value in values):
            raise CategoryDepthError(f"P2 target_poi_id 非法：{path}")
        targets.update(values)
        rows += len(values)
    expected = int(bundle.p2_manifest["stats"]["unique_query_poi_pairs"])
    if rows != expected:
        raise CategoryDepthError("收集 P2 target POI 时 pair 行数不守恒")
    expected_unique = int(bundle.p2_manifest["stats"]["covered_pois"])
    if len(targets) != expected_unique:
        raise CategoryDepthError("P2 target POI 去重数与 manifest 不一致")
    return targets


def _mapping_part_name(index: int, total: int) -> str:
    width = max(5, len(str(total)))
    return f"part-{index:0{width}d}-of-{total:0{width}d}.parquet"


def _write_category_mapping_part(
    path: Path,
    rows: Mapping[str, Sequence[Any]],
) -> None:
    table = pa.Table.from_pydict(rows, schema=CATEGORY_MAPPING_SCHEMA)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=max(1, min(65_536, len(table))),
    )


def _build_category_mapping(
    config: CategoryBuildConfig,
    bundle: FrozenInputBundle,
    output_dir: Path,
    target_pois: set[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[int, int, int]],
    dict[str, Any],
    dict[str, set[str]],
]:
    output_dir.mkdir()
    target_categories: dict[str, tuple[int, int, int]] = {}
    fine_paths: dict[str, set[str]] = defaultdict(set)
    fine_row_counts = np.zeros(len(bundle.fine_codes), dtype=np.int64)
    coarse_row_counts = np.zeros(len(bundle.coarse_codes), dtype=np.int64)
    mapping_files: list[dict[str, Any]] = []
    ids_digest = hashlib.sha256()
    global_row = 0

    with config.paths.poi_ids.open("rb") as ids_handle:
        for source_index, source in enumerate(
            tqdm(
                bundle.catalog_sources,
                desc="Build category mapping",
                unit="part",
                disable=not _SHOW_PROGRESS,
            )
        ):
            columns: dict[str, list[Any]] = {
                field.name: [] for field in CATEGORY_MAPPING_SCHEMA
            }
            source_digest = hashlib.sha256()
            source_rows = 0
            source_bytes = 0
            with source.path.open("rb") as catalog_handle:
                for line_number, raw_line in enumerate(catalog_handle, start=1):
                    source_digest.update(raw_line)
                    source_bytes += len(raw_line)
                    record = _parse_json_line(
                        raw_line, f"{source.path}:{line_number}"
                    )
                    if not isinstance(record, Mapping):
                        raise CategoryDepthError(
                            f"POI catalog 行必须是 object：{source.path}:{line_number}"
                        )
                    id_line = ids_handle.readline()
                    if not id_line:
                        raise CategoryDepthError("poi_ids 早于 POI catalog 结束")
                    ids_digest.update(id_line)
                    bge_poi_id = _parse_json_line(
                        id_line, f"poi_ids:{global_row + 1}"
                    )
                    poi_id = record.get("poi_id")
                    category_path = record.get(config.category.readable_path_column)
                    category_code = record.get(config.category.fine_source_column)
                    if not isinstance(poi_id, str) or not poi_id:
                        raise CategoryDepthError("POI catalog poi_id 必须是非空字符串")
                    if poi_id != bge_poi_id:
                        raise CategoryDepthError(
                            f"POI catalog 与 BGE 行序错位：row={global_row}"
                        )
                    if not isinstance(category_path, str) or not category_path.strip():
                        raise CategoryDepthError(f"POI {poi_id} category 为空")
                    if (
                        not isinstance(category_code, str)
                        or ASCII_CATEGORY_CODE.fullmatch(category_code) is None
                    ):
                        raise CategoryDepthError(f"POI {poi_id} category_code 不是 6 位数字")
                    if global_row >= config.frozen.poi_rows:
                        raise CategoryDepthError("POI catalog 行数超过冻结 POI 数")
                    fine_index = int(bundle.category_indices[global_row])
                    expected_code = bundle.fine_codes[fine_index]
                    if category_code != expected_code:
                        raise CategoryDepthError(
                            f"POI {poi_id} category index/code 不一致："
                            f"{fine_index}->{expected_code} != {category_code}"
                        )
                    coarse_id = category_code[:2]
                    coarse_index = bundle.coarse_index_by_id[coarse_id]
                    columns["poi_row_index"].append(global_row)
                    columns["poi_id"].append(poi_id)
                    columns["category"].append(category_path)
                    columns["fine_category_id"].append(category_code)
                    columns["fine_category_index"].append(fine_index)
                    columns["coarse_category_id"].append(coarse_id)
                    columns["coarse_category_index"].append(coarse_index)
                    fine_row_counts[fine_index] += 1
                    coarse_row_counts[coarse_index] += 1
                    fine_paths[category_code].add(category_path)
                    if poi_id in target_pois:
                        if poi_id in target_categories:
                            raise CategoryDepthError(f"P2 target POI 在目录中重复：{poi_id}")
                        target_categories[poi_id] = (
                            len(target_categories),
                            fine_index,
                            coarse_index,
                        )
                    global_row += 1
                    source_rows += 1
            if (
                source_rows != source.expected_rows
                or source_bytes != source.expected_bytes
                or source_digest.hexdigest() != source.expected_sha256
            ):
                raise CategoryDepthError(f"POI catalog source 指纹不一致：{source.path}")
            part_path = output_dir / _mapping_part_name(
                source_index, len(bundle.catalog_sources)
            )
            _write_category_mapping_part(part_path, columns)
            info = _file_info(part_path, shard_id=source_index)
            info["source_file"] = source.path.name
            mapping_files.append(info)
        extra_id_line = ids_handle.readline()
        if extra_id_line:
            raise CategoryDepthError("poi_ids 在 POI catalog 结束后仍有额外行")

    if global_row != config.frozen.poi_rows:
        raise CategoryDepthError(
            f"POI mapping 行数 {global_row} != {config.frozen.poi_rows}"
        )
    if ids_digest.hexdigest() != config.frozen.poi_ids_sha256:
        raise CategoryDepthError("POI mapping 扫描所得 poi_ids SHA256 不一致")
    missing_targets = target_pois.difference(target_categories)
    if missing_targets:
        raise CategoryDepthError(
            f"P2 target POI 缺少目录类别：{sorted(missing_targets)[:5]}"
        )
    if np.any(coarse_row_counts <= 0):
        raise CategoryDepthError("active POI 存在零覆盖 coarse category")
    observed_fine_mask = fine_row_counts > 0
    observed_fine_codes = [
        code
        for code, observed in zip(
            bundle.fine_codes, observed_fine_mask, strict=True
        )
        if observed
    ]
    missing_fine_codes = [
        code
        for code, observed in zip(
            bundle.fine_codes, observed_fine_mask, strict=True
        )
        if not observed
    ]
    path_variant_counts = {code: len(paths) for code, paths in fine_paths.items()}
    metrics = {
        "rows": global_row,
        "missing_category_rows": 0,
        "invalid_category_code_rows": 0,
        "row_order_mismatches": 0,
        "fine_category_count": len(bundle.fine_codes),
        "observed_fine_category_count": len(observed_fine_codes),
        "missing_fine_category_codes": missing_fine_codes,
        "coarse_category_count": len(bundle.coarse_codes),
        "fine_to_coarse_consistent": all(
            code[:2] in bundle.coarse_index_by_id for code in bundle.fine_codes
        ),
        "fine_category_min_poi_rows": int(
            fine_row_counts[observed_fine_mask].min()
        ),
        "fine_category_max_poi_rows": int(fine_row_counts.max()),
        "coarse_category_min_poi_rows": int(coarse_row_counts.min()),
        "coarse_category_max_poi_rows": int(coarse_row_counts.max()),
        "fine_codes_with_multiple_readable_paths": sum(
            count > 1 for count in path_variant_counts.values()
        ),
        "p2_target_poi_count": len(target_categories),
        "poi_ids_scanned_sha256": ids_digest.hexdigest(),
    }
    return mapping_files, target_categories, metrics, fine_paths


def _push_sample(
    heaps: dict[int, list[tuple[int, int, dict[str, Any]]]],
    depth: int,
    sample: dict[str, Any],
    limit: int = 20,
) -> None:
    key = (int(sample["query_count"]), -int(sample["query_id"]), sample)
    heap = heaps[depth]
    if len(heap) < limit:
        heapq.heappush(heap, key)
    elif key[:2] > heap[0][:2]:
        heapq.heapreplace(heap, key)


def _distribution_metrics(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        raise CategoryDepthError("不能对空数组计算分布")
    numeric = values.astype(np.float64, copy=False)
    return {
        "count": int(numeric.size),
        "min": float(numeric.min()),
        "p50": float(np.quantile(numeric, 0.50)),
        "p90": float(np.quantile(numeric, 0.90)),
        "p99": float(np.quantile(numeric, 0.99)),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def _sensitivity_counts(
    query_counts: np.ndarray,
    is_exact: np.ndarray,
    fine_concentration: np.ndarray,
    fine_entropy: np.ndarray,
    coarse_concentration: np.ndarray,
    coarse_entropy: np.ndarray,
    settings: Sequence[SensitivitySetting],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = int(query_counts.size)
    for setting in settings:
        d3 = is_exact
        d2 = (
            ~d3
            & (query_counts >= setting.d2_fine.min_query_count)
            & (fine_concentration >= setting.d2_fine.min_concentration)
            & (fine_entropy <= setting.d2_fine.max_normalized_entropy)
        )
        d1 = (
            ~d3
            & ~d2
            & (query_counts >= setting.d1_coarse.min_query_count)
            & (coarse_concentration >= setting.d1_coarse.min_concentration)
            & (coarse_entropy <= setting.d1_coarse.max_normalized_entropy)
        )
        counts = {
            "D0_CONTEXTUAL": int(np.count_nonzero(~d3 & ~d2 & ~d1)),
            "D1_COARSE": int(np.count_nonzero(d1)),
            "D2_FINE": int(np.count_nonzero(d2)),
            "D3_EXACT": int(np.count_nonzero(d3)),
        }
        results.append(
            {
                "name": setting.name,
                "d2_fine_category": vars(setting.d2_fine),
                "d1_coarse_category": vars(setting.d1_coarse),
                "depth_counts": counts,
                "depth_ratios": {
                    label: value / total for label, value in counts.items()
                },
            }
        )
    return results


def _read_shard_inputs(
    stats_path: Path,
    pair_path: Path,
    shard_id: int,
    expected_first_query_id: int,
) -> tuple[list[Mapping[str, Any]], dict[int, list[PairObservation]]]:
    stats_rows = pq.read_table(stats_path).to_pylist()
    query_text: dict[int, str] = {}
    for offset, row in enumerate(stats_rows):
        query_id = int(row["query_id"])
        if query_id != expected_first_query_id + offset:
            raise CategoryDepthError(f"P2 shard {shard_id} query_id 不连续")
        if int(row["query_shard_id"]) != shard_id:
            raise CategoryDepthError(f"P2 shard {shard_id} query_shard_id 不一致")
        query_text[query_id] = str(row["normalized_query"])

    pairs_by_query: dict[int, list[PairObservation]] = defaultdict(list)
    for row in pq.read_table(pair_path).to_pylist():
        query_id = int(row["query_id"])
        if query_id not in query_text:
            raise CategoryDepthError(f"P2 shard {shard_id} 出现未知 query_id={query_id}")
        if row["normalized_query"] != query_text[query_id]:
            raise CategoryDepthError(f"P2 shard {shard_id} Query 文本不一致")
        pairs_by_query[query_id].append(_pair_observation(row))
    if len(pairs_by_query) != len(stats_rows):
        raise CategoryDepthError(f"P2 shard {shard_id} Query 与 pair 未一一覆盖")
    return stats_rows, pairs_by_query


def _process_query_shards(
    config: CategoryBuildConfig,
    bundle: FrozenInputBundle,
    target_categories: Mapping[str, tuple[int, int, int]],
    fine_paths: Mapping[str, set[str]],
    staging: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    stats_dir = staging / "query_category_stats.parquet"
    depth_dir = staging / "query_category_depth.parquet"
    edge_dir = staging / "query_poi_layer_edges.parquet"
    for directory in (stats_dir, depth_dir, edge_dir):
        directory.mkdir()

    total_queries = int(bundle.p2_manifest["stats"]["unique_queries"])
    query_counts = np.zeros(total_queries, dtype=np.int64)
    fine_distinct = np.zeros(total_queries, dtype=np.int16)
    coarse_distinct = np.zeros(total_queries, dtype=np.int16)
    fine_concentrations = np.zeros(total_queries, dtype=np.float64)
    fine_entropies = np.zeros(total_queries, dtype=np.float64)
    coarse_concentrations = np.zeros(total_queries, dtype=np.float64)
    coarse_entropies = np.zeros(total_queries, dtype=np.float64)
    depths = np.full(total_queries, -1, dtype=np.int8)
    is_exact = np.zeros(total_queries, dtype=np.bool_)

    target_count = len(target_categories)
    depth_coverage = {
        depth: np.zeros(target_count, dtype=np.bool_) for depth in range(4)
    }
    layer_coverage = {
        layer: np.zeros(target_count, dtype=np.bool_) for layer in (1, 2, 3)
    }
    depth_counts: Counter[int] = Counter()
    layer_edge_counts: Counter[int] = Counter()
    retained_poi_edges = 0
    total_layer_edges = 0
    query_count_sum = 0
    max_probability_error = 0.0
    max_weight_error = 0.0
    sample_heaps: dict[int, list[tuple[int, int, dict[str, Any]]]] = {
        depth: [] for depth in range(4)
    }
    output_files: dict[str, list[dict[str, Any]]] = {
        "query_category_stats": [],
        "query_category_depth": [],
        "query_poi_layer_edges": [],
    }
    next_query_id = 0

    for shard_id, (stats_path, pair_path) in enumerate(
        tqdm(
            zip(bundle.query_stats_paths, bundle.query_pair_paths, strict=True),
            total=bundle.p2_num_shards,
            desc="Aggregate Query category depth",
            unit="shard",
            disable=not _SHOW_PROGRESS,
        )
    ):
        stats_rows, pairs_by_query = _read_shard_inputs(
            stats_path, pair_path, shard_id, next_query_id
        )
        shard_stats_output: list[dict[str, Any]] = []
        shard_depth_output: list[dict[str, Any]] = []
        shard_edge_output: list[dict[str, Any]] = []
        for stats in stats_rows:
            query_id = int(stats["query_id"])
            pairs = pairs_by_query[query_id]
            if len(pairs) != int(stats["distinct_poi_count"]):
                raise CategoryDepthError(f"Query {query_id} distinct_poi_count 不一致")
            top1_pairs = [pair for pair in pairs if pair.is_top1]
            if (
                len(top1_pairs) != 1
                or top1_pairs[0].poi_id != stats["top1_poi_id"]
                or top1_pairs[0].pair_count != int(stats["top1_count"])
            ):
                raise CategoryDepthError(f"Query {query_id} P2 top1 字段不一致")
            computation = compute_query_supervision(
                stats,
                pairs,
                target_categories,
                bundle.fine_codes,
                bundle.coarse_codes,
                config,
            )
            query_count = int(stats["query_count"])
            query_count_sum += query_count
            query_counts[query_id] = query_count
            fine_distinct[query_id] = computation.fine.distinct_count
            coarse_distinct[query_id] = computation.coarse.distinct_count
            fine_concentrations[query_id] = computation.fine.concentration
            fine_entropies[query_id] = computation.fine.normalized_entropy
            coarse_concentrations[query_id] = computation.coarse.concentration
            coarse_entropies[query_id] = computation.coarse.normalized_entropy
            depths[query_id] = computation.depth
            is_exact[query_id] = bool(stats["is_high_confidence"])
            depth_counts[computation.depth] += 1

            shard_stats_output.append(
                {
                    "query_id": query_id,
                    "query_shard_id": shard_id,
                    "normalized_query": stats["normalized_query"],
                    "representative_raw_query": stats["representative_raw_query"],
                    "query_count": query_count,
                    "distinct_poi_count": int(stats["distinct_poi_count"]),
                    "top1_poi_id": stats["top1_poi_id"],
                    "top1_count": int(stats["top1_count"]),
                    "top1_share": float(stats["top1_share"]),
                    "exact_normalized_entropy": float(stats["normalized_entropy"]),
                    "is_p2_exact_core": bool(stats["is_high_confidence"]),
                    "fine_distinct_category_count": computation.fine.distinct_count,
                    "dominant_fine_category_id": computation.fine.dominant_id,
                    "dominant_fine_count": computation.fine.dominant_count,
                    "fine_concentration": computation.fine.concentration,
                    "fine_entropy": computation.fine.entropy,
                    "fine_normalized_entropy": computation.fine.normalized_entropy,
                    "coarse_distinct_category_count": computation.coarse.distinct_count,
                    "dominant_coarse_category_id": computation.coarse.dominant_id,
                    "dominant_coarse_count": computation.coarse.dominant_count,
                    "coarse_concentration": computation.coarse.concentration,
                    "coarse_entropy": computation.coarse.entropy,
                    "coarse_normalized_entropy": computation.coarse.normalized_entropy,
                }
            )
            layer_reliability_by_id = dict(
                zip(
                    computation.layers,
                    computation.layer_reliabilities,
                    strict=True,
                )
            )
            retained_total = sum(
                pair.pair_count for pair in computation.retained_pairs
            )
            probability_sums: Counter[int] = Counter()
            weight_sums: Counter[int] = Counter()
            for layer in computation.layers:
                reliability = layer_reliability_by_id[layer]
                for pair in computation.retained_pairs:
                    coverage_index, fine_index, coarse_index = target_categories[
                        pair.poi_id
                    ]
                    probability = pair.pair_count / retained_total
                    weight = probability * reliability
                    probability_sums[layer] += probability
                    weight_sums[layer] += weight
                    depth_coverage[computation.depth][coverage_index] = True
                    layer_coverage[layer][coverage_index] = True
                    layer_edge_counts[layer] += 1
                    total_layer_edges += 1
                    shard_edge_output.append(
                        {
                            "query_id": query_id,
                            "query_shard_id": shard_id,
                            "normalized_query": stats["normalized_query"],
                            "target_poi_id": pair.poi_id,
                            "pair_count": pair.pair_count,
                            "pair_share": pair.pair_share,
                            "supervision_depth": computation.depth,
                            "supervision_label": DEPTH_LABELS[computation.depth],
                            "layer": layer,
                            "hierarchy_source": computation.hierarchy_source,
                            "fine_category_id": bundle.fine_codes[fine_index],
                            "coarse_category_id": bundle.coarse_codes[coarse_index],
                            "edge_probability": probability,
                            "query_reliability": reliability,
                            "edge_weight": weight,
                        }
                    )
            for layer in computation.layers:
                max_probability_error = max(
                    max_probability_error, abs(probability_sums[layer] - 1.0)
                )
                max_weight_error = max(
                    max_weight_error,
                    abs(weight_sums[layer] - layer_reliability_by_id[layer]),
                )
            retained_poi_edges += len(computation.retained_pairs)
            layer_mask = tuple(computation.depth >= layer for layer in (1, 2, 3))
            shard_depth_output.append(
                {
                    "query_id": query_id,
                    "query_shard_id": shard_id,
                    "normalized_query": stats["normalized_query"],
                    "representative_raw_query": stats["representative_raw_query"],
                    "query_count": query_count,
                    "supervision_depth": computation.depth,
                    "supervision_label": DEPTH_LABELS[computation.depth],
                    "supervise_s1": layer_mask[0],
                    "supervise_s2": layer_mask[1],
                    "supervise_s3": layer_mask[2],
                    "dominant_fine_category_id": computation.fine.dominant_id,
                    "dominant_coarse_category_id": computation.coarse.dominant_id,
                    "support_score": computation.support_score,
                    "coarse_reliability": computation.coarse_reliability,
                    "fine_reliability": computation.fine_reliability,
                    "exact_reliability": computation.exact_reliability,
                    "s1_reliability": layer_reliability_by_id.get(1, 0.0),
                    "s2_reliability": layer_reliability_by_id.get(2, 0.0),
                    "s3_reliability": layer_reliability_by_id.get(3, 0.0),
                    "retained_poi_edge_count": len(computation.retained_pairs),
                    "retained_layer_edge_count": (
                        len(computation.retained_pairs) * len(computation.layers)
                    ),
                }
            )
            readable_paths = sorted(fine_paths[computation.fine.dominant_id])
            _push_sample(
                sample_heaps,
                computation.depth,
                {
                    "query_id": query_id,
                    "representative_raw_query": stats["representative_raw_query"],
                    "normalized_query": stats["normalized_query"],
                    "query_count": query_count,
                    "distinct_poi_count": int(stats["distinct_poi_count"]),
                    "top1_poi_id": stats["top1_poi_id"],
                    "top1_share": float(stats["top1_share"]),
                    "dominant_fine_category_id": computation.fine.dominant_id,
                    "dominant_fine_category_path": readable_paths[0],
                    "fine_readable_path_variants": len(readable_paths),
                    "fine_concentration": computation.fine.concentration,
                    "fine_normalized_entropy": computation.fine.normalized_entropy,
                    "dominant_coarse_category_id": computation.coarse.dominant_id,
                    "coarse_concentration": computation.coarse.concentration,
                    "coarse_normalized_entropy": computation.coarse.normalized_entropy,
                    "retained_poi_edge_count": len(computation.retained_pairs),
                },
            )
        next_query_id += len(stats_rows)
        part_name = _mapping_part_name(shard_id, bundle.p2_num_shards)
        stats_output_path = stats_dir / part_name
        depth_output_path = depth_dir / part_name
        edge_output_path = edge_dir / part_name
        _write_rows(
            stats_output_path, QUERY_CATEGORY_STATS_SCHEMA, shard_stats_output
        )
        _write_rows(
            depth_output_path, QUERY_CATEGORY_DEPTH_SCHEMA, shard_depth_output
        )
        _write_rows(
            edge_output_path, QUERY_POI_LAYER_EDGE_SCHEMA, shard_edge_output
        )
        output_files["query_category_stats"].append(
            _file_info(stats_output_path, shard_id=shard_id)
        )
        output_files["query_category_depth"].append(
            _file_info(depth_output_path, shard_id=shard_id)
        )
        output_files["query_poi_layer_edges"].append(
            _file_info(edge_output_path, shard_id=shard_id)
        )

    if next_query_id != total_queries or np.any(depths < 0) or np.any(query_counts <= 0):
        raise CategoryDepthError("全量 Query 未完整聚合")
    expected_source_rows = int(bundle.p2_manifest["stats"]["source_rows"])
    if query_count_sum != expected_source_rows:
        raise CategoryDepthError("Query count 总和与 P2 Train 行数不守恒")
    expected_d3 = int(bundle.p2_manifest["stats"]["retained_queries"])
    if depth_counts[3] != expected_d3 or not np.array_equal(is_exact, depths == 3):
        raise CategoryDepthError("D3 未原样复用 P2 Exact-Core")
    if max_probability_error > 1e-9 or max_weight_error > 1e-9:
        raise CategoryDepthError("Query 层级边归一化失败")

    depth_count_payload = {
        DEPTH_LABELS[depth]: depth_counts[depth] for depth in range(4)
    }
    samples = {
        DEPTH_LABELS[depth]: [
            item[2]
            for item in sorted(sample_heaps[depth], key=lambda item: item[:2], reverse=True)
        ]
        for depth in range(4)
    }
    if any(
        len(samples[DEPTH_LABELS[depth]]) != min(20, depth_counts[depth])
        for depth in range(4)
    ):
        raise CategoryDepthError("supervision depth 样例数量不完整")
    sensitivity = _sensitivity_counts(
        query_counts,
        is_exact,
        fine_concentrations,
        fine_entropies,
        coarse_concentrations,
        coarse_entropies,
        config.query_depth.sensitivity,
    )
    baseline = next(item for item in sensitivity if item["name"] == "baseline")
    if baseline["depth_counts"] != depth_count_payload:
        raise CategoryDepthError("baseline 敏感性结果与正式 depth 不一致")

    p2_covered_pois = int(bundle.p2_manifest["stats"]["covered_pois"])
    catalog_rows = config.frozen.poi_rows
    coverage_by_depth = {
        DEPTH_LABELS[depth]: {
            "poi_count": int(np.count_nonzero(depth_coverage[depth])),
            "ratio_of_p2_covered_pois": float(
                np.count_nonzero(depth_coverage[depth]) / p2_covered_pois
            ),
            "ratio_of_catalog": float(
                np.count_nonzero(depth_coverage[depth]) / catalog_rows
            ),
        }
        for depth in range(4)
    }
    coverage_by_layer = {
        f"S{layer}": {
            "poi_count": int(np.count_nonzero(layer_coverage[layer])),
            "ratio_of_p2_covered_pois": float(
                np.count_nonzero(layer_coverage[layer]) / p2_covered_pois
            ),
            "ratio_of_catalog": float(
                np.count_nonzero(layer_coverage[layer]) / catalog_rows
            ),
            "edge_rows": layer_edge_counts[layer],
        }
        for layer in (1, 2, 3)
    }
    metrics = {
        "queries": {
            "total": total_queries,
            "query_count_sum": query_count_sum,
            "depth_counts": depth_count_payload,
            "depth_ratios": {
                label: count / total_queries
                for label, count in depth_count_payload.items()
            },
        },
        "distributions": {
            "query_count": _distribution_metrics(query_counts),
            "fine_distinct_category_count": _distribution_metrics(fine_distinct),
            "fine_concentration": _distribution_metrics(fine_concentrations),
            "fine_normalized_entropy": _distribution_metrics(fine_entropies),
            "coarse_distinct_category_count": _distribution_metrics(coarse_distinct),
            "coarse_concentration": _distribution_metrics(coarse_concentrations),
            "coarse_normalized_entropy": _distribution_metrics(coarse_entropies),
        },
        "edges": {
            "retained_query_poi_edges": retained_poi_edges,
            "layer_edge_rows": total_layer_edges,
            "rows_by_layer": {
                f"S{layer}": layer_edge_counts[layer] for layer in (1, 2, 3)
            },
            "max_query_layer_probability_sum_error": max_probability_error,
            "max_query_layer_weight_sum_error": max_weight_error,
            "normalization_tolerance": 1e-9,
        },
        "poi_coverage": {
            "catalog_pois": catalog_rows,
            "p2_covered_pois": p2_covered_pois,
            "by_depth": coverage_by_depth,
            "by_layer": coverage_by_layer,
        },
        "sensitivity": sensitivity,
        "sample_selection": "query_count descending, then query_id ascending",
        "samples": samples,
    }
    return output_files, metrics


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_report(metrics: Mapping[str, Any]) -> str:
    queries = metrics["queries"]
    category = metrics["category_mapping"]
    edges = metrics["edges"]
    coverage = metrics["poi_coverage"]
    lines = [
        "# QG-PRQK P2.5-CAT 全量结果",
        "",
        "本报告是 Train-only 既有统计的类别后处理产物，不是下游效果实验。",
        "",
        "## 监督深度",
        "",
        "| 深度 | Query 数 | 占比 |",
        "|---|---:|---:|",
    ]
    for label in DEPTH_LABELS:
        count = int(queries["depth_counts"][label])
        ratio = float(queries["depth_ratios"][label])
        lines.append(f"| {label} | {count:,} | {ratio:.6%} |")
    lines.extend(
        [
            "",
            "## 类别与边合同",
            "",
            f"- POI mapping：{int(category['rows']):,} 行；缺失类别 0；行序错位 0。",
            f"- 类别词表：{int(category['fine_category_count'])} 个 fine / "
            f"{int(category['coarse_category_count'])} 个 coarse；active 实际出现 "
            f"{int(category['observed_fine_category_count'])} 个 fine，fine→coarse 一致。",
            f"- 保留 Query–POI 边：{int(edges['retained_query_poi_edges']):,}；"
            f"展开到层后的边：{int(edges['layer_edge_rows']):,}。",
            "- support：`log1p(min(query_count,20))/log1p(20)`；"
            "reliability：`support * concentration^gamma * (1-normalized_entropy)`。",
            f"- 最大 Query–层概率和误差："
            f"{float(edges['max_query_layer_probability_sum_error']):.3e}；"
            f"最大边权和误差：{float(edges['max_query_layer_weight_sum_error']):.3e}。",
            "",
            "## POI 覆盖",
            "",
            "| 层 | 覆盖 POI | P2 覆盖内占比 | 全目录占比 | 边行数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for layer in ("S1", "S2", "S3"):
        item = coverage["by_layer"][layer]
        lines.append(
            f"| {layer} | {int(item['poi_count']):,} | "
            f"{float(item['ratio_of_p2_covered_pois']):.6%} | "
            f"{float(item['ratio_of_catalog']):.6%} | "
            f"{int(item['edge_rows']):,} |"
        )
    lines.extend(
        [
            "",
            "## 阈值敏感性",
            "",
            "| 设置 | D0 | D1 | D2 | D3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["sensitivity"]:
        counts = item["depth_counts"]
        lines.append(
            f"| {_escape_markdown(item['name'])} | "
            f"{int(counts['D0_CONTEXTUAL']):,} | "
            f"{int(counts['D1_COARSE']):,} | "
            f"{int(counts['D2_FINE']):,} | "
            f"{int(counts['D3_EXACT']):,} |"
        )
    lines.extend(
        [
            "",
            "## 可读样例",
            "",
            "样例按 `query_count` 降序、`query_id` 升序确定性选择，每层 20 条。",
        ]
    )
    for label in DEPTH_LABELS:
        lines.extend(
            [
                "",
                f"### {label}",
                "",
                "| query_id | 原 Query | count | Fine(C/H) | Coarse(C/H) | 保留 POI 边 |",
                "|---:|---|---:|---|---|---:|",
            ]
        )
        for sample in metrics["samples"][label]:
            lines.append(
                f"| {int(sample['query_id'])} | "
                f"{_escape_markdown(sample['representative_raw_query'])} | "
                f"{int(sample['query_count']):,} | "
                f"{_escape_markdown(sample['dominant_fine_category_id'])} "
                f"({float(sample['fine_concentration']):.3f}/"
                f"{float(sample['fine_normalized_entropy']):.3f}) | "
                f"{_escape_markdown(sample['dominant_coarse_category_id'])} "
                f"({float(sample['coarse_concentration']):.3f}/"
                f"{float(sample['coarse_normalized_entropy']):.3f}) | "
                f"{int(sample['retained_poi_edge_count'])} |"
            )
    lines.extend(
        [
            "",
            "## 数据隔离",
            "",
            "运行只读取冻结 P2 `query_stats/query_poi_pairs`、类别资产、BGE 行映射"
            "和必要 POI 类别 metadata；未打开原始 Train、Validation 或 Test。",
            "",
        ]
    )
    return "\n".join(lines)


def _group_manifest(directory: Path, files: Sequence[Mapping[str, Any]], staging: Path) -> dict[str, Any]:
    return {
        "dir": str(directory.relative_to(staging)),
        "rows": sum(int(item["rows"]) for item in files),
        "files": list(files),
    }


def _plain_file_manifest(path: Path, staging: Path) -> dict[str, Any]:
    return {
        "file": str(path.relative_to(staging)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_category_depth(
    config: CategoryBuildConfig,
    *,
    resume: bool = False,
) -> CategoryDepthBuildResult:
    """Build the complete P2.5-CAT artifact from frozen P2 and POI metadata."""

    output_dir = config.query_depth_output_dir
    if output_dir.exists():
        if resume:
            return validate_category_depth(output_dir, config)
        raise CategoryDepthError(f"P2.5 输出已存在且 overwrite=false：{output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        raise CategoryDepthError(f"存在未完成目录，请先审计：{staging}")

    started = time.perf_counter()
    bundle = preflight_category_depth(config)
    gate_evidence = require_gate_evidence(config)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    staging.mkdir()

    target_pois = _collect_p2_target_pois(bundle)
    category_dir = staging / config.category.output_mapping_name
    (
        mapping_files,
        target_categories,
        category_metrics,
        fine_paths,
    ) = _build_category_mapping(
        config,
        bundle,
        category_dir,
        target_pois,
    )
    del target_pois
    query_files, query_metrics = _process_query_shards(
        config,
        bundle,
        target_categories,
        fine_paths,
        staging,
    )
    metrics: dict[str, Any] = {
        "schema_version": "qg-prqk-p2-5-cat-metrics-v1",
        "is_sample": False,
        "category_mapping": category_metrics,
        **query_metrics,
    }
    metrics_path = staging / "p2_5_cat_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report_path = staging / "p2_5_cat_report.md"
    report_path.write_text(_render_report(metrics), encoding="utf-8")
    resolved_path = staging / "config_resolved.yaml"
    resolved_payload = config.resolved_payload() | {
        "source_config": {
            "path": str(config.source_path),
            "sha256": config.source_sha256,
            "resolved_signature": config.signature(),
        }
    }
    resolved_path.write_text(
        yaml.safe_dump(
            resolved_payload,
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    code = implementation_signature()
    outputs = {
        "category_mapping": _group_manifest(category_dir, mapping_files, staging),
        "query_category_stats": _group_manifest(
            staging / "query_category_stats.parquet",
            query_files["query_category_stats"],
            staging,
        ),
        "query_category_depth": _group_manifest(
            staging / "query_category_depth.parquet",
            query_files["query_category_depth"],
            staging,
        ),
        "query_poi_layer_edges": _group_manifest(
            staging / "query_poi_layer_edges.parquet",
            query_files["query_poi_layer_edges"],
            staging,
        ),
        "metrics": _plain_file_manifest(metrics_path, staging),
        "report": _plain_file_manifest(report_path, staging),
        "resolved_config": _plain_file_manifest(resolved_path, staging),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "method": config.method_name,
        "version": config.method_version,
        "city": config.city,
        "built_at": utc_now(),
        "config": {
            "path": str(config.source_path),
            "sha256": config.source_sha256,
            "resolved_signature": config.signature(),
        },
        "code": code,
        "inputs": {
            "p2_manifest": {
                "path": str(bundle.p2_manifest_path),
                "sha256": config.frozen.p2_manifest_sha256,
                "schema_version": bundle.p2_manifest["schema_version"],
                "unique_queries": int(bundle.p2_manifest["stats"]["unique_queries"]),
                "unique_query_poi_pairs": int(
                    bundle.p2_manifest["stats"]["unique_query_poi_pairs"]
                ),
            },
            "category_indices": {
                "path": str(config.paths.category_indices),
                "sha256": config.frozen.category_indices_sha256,
                "shape": [config.frozen.poi_rows],
                "dtype": "int32",
            },
            "category_vocab_data": {
                "path": str(bundle.vocab_data_path),
                "sha256": config.frozen.category_vocab_data_sha256,
            },
            "category_source_manifest": {
                "path": str(config.paths.category_source_manifest),
                "sha256": config.frozen.category_source_manifest_sha256,
            },
            "bge_manifest": {
                "path": str(config.paths.embedding_manifest),
                "sha256": config.frozen.poi_embedding_manifest_sha256,
            },
            "poi_ids": {
                "path": str(config.paths.poi_ids),
                "sha256": config.frozen.poi_ids_sha256,
            },
            "poi_catalog_sources": [
                {
                    "path": str(source.path),
                    "rows": source.expected_rows,
                    "size_bytes": source.expected_bytes,
                    "sha256": source.expected_sha256,
                }
                for source in bundle.catalog_sources
            ],
        },
        "query_depth": {
            "priority": ["D3_EXACT", "D2_FINE", "D1_COARSE", "D0_CONTEXTUAL"],
            "d3_definition": "P2 is_high_confidence=true unchanged",
            "d2_fine_category": vars(config.query_depth.d2_fine),
            "d1_coarse_category": vars(config.query_depth.d1_coarse),
            "support_formula": (
                "log1p(min(query_count,support_cap))/log1p(support_cap)"
            ),
            "support_cap": config.query_depth.support_cap,
            "reliability_formula": (
                "support_score*concentration^gamma*(1-normalized_entropy)"
            ),
            "reliability_gamma": config.query_depth.reliability_gamma,
            "layer_masks": {
                "D0_CONTEXTUAL": [0, 0, 0],
                "D1_COARSE": [1, 0, 0],
                "D2_FINE": [1, 1, 0],
                "D3_EXACT": [1, 1, 1],
            },
        },
        "gates": gate_evidence,
        "source_access": {
            "p2_artifacts_read": ["manifest.json", "query_stats", "query_poi_pairs"],
            "p2_query_shards_read": False,
            "p2_false_negative_mask_read": False,
            "raw_train_read": False,
            "validation_read": False,
            "test_read": False,
            "poi_catalog_metadata_read": ["poi_id", "category", "category_code"],
            "poi_embedding_values_read": False,
            "poi_embedding_header_read": True,
        },
        "stats": metrics,
        "outputs": outputs,
        "validation": {
            "full_p2_only": True,
            "p2_unchanged_by_sha256": True,
            "category_row_order_matches_bge": True,
            "category_full_coverage": True,
            "fine_to_coarse_unique": True,
            "d3_exact_core_unchanged": True,
            "query_layer_edges_normalized": True,
            "valid_and_test_read": False,
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (staging / "_SUCCESS").touch()
    os.replace(staging, output_dir)
    return CategoryDepthBuildResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        category_rows=category_metrics["rows"],
        unique_queries=int(metrics["queries"]["total"]),
        layer_edges=int(metrics["edges"]["layer_edge_rows"]),
        depth_counts={
            key: int(value)
            for key, value in metrics["queries"]["depth_counts"].items()
        },
        reused=False,
    )


def _validate_output_group(
    output_dir: Path,
    raw: Any,
    schema: pa.Schema,
    label: str,
) -> tuple[tuple[Path, ...], int]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("files"), list):
        raise CategoryDepthError(f"P2.5 {label} 输出清单非法")
    relative_dir = Path(str(raw.get("dir", "")))
    if relative_dir.is_absolute() or ".." in relative_dir.parts:
        raise CategoryDepthError(f"P2.5 {label} 输出目录非法")
    directory = output_dir / relative_dir
    paths: list[Path] = []
    rows = 0
    for expected_shard, entry in enumerate(raw["files"]):
        if not isinstance(entry, Mapping):
            raise CategoryDepthError(f"P2.5 {label} 文件清单非法")
        if int(entry.get("shard_id", -1)) != expected_shard:
            raise CategoryDepthError(f"P2.5 {label} shard 顺序非法")
        file_name = str(entry.get("file", ""))
        if not file_name or Path(file_name).name != file_name:
            raise CategoryDepthError(f"P2.5 {label} 文件名非法")
        path = directory / file_name
        if not path.is_file():
            raise CategoryDepthError(f"P2.5 {label} 文件不存在：{path}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise CategoryDepthError(f"P2.5 {label} 文件大小不一致：{path}")
        if sha256_file(path) != entry.get("sha256"):
            raise CategoryDepthError(f"P2.5 {label} SHA256 不一致：{path}")
        if not pq.read_schema(path).equals(schema, check_metadata=False):
            raise CategoryDepthError(f"P2.5 {label} schema 不一致：{path}")
        file_rows = pq.ParquetFile(path).metadata.num_rows
        if file_rows != int(entry.get("rows", -1)):
            raise CategoryDepthError(f"P2.5 {label} 行数不一致：{path}")
        paths.append(path)
        rows += file_rows
    if rows != int(raw.get("rows", -1)):
        raise CategoryDepthError(f"P2.5 {label} 总行数不一致")
    return tuple(paths), rows


def _validate_plain_output(
    output_dir: Path,
    raw: Any,
    label: str,
) -> Path:
    if not isinstance(raw, Mapping):
        raise CategoryDepthError(f"P2.5 {label} 清单非法")
    relative = Path(str(raw.get("file", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CategoryDepthError(f"P2.5 {label} 路径非法")
    path = output_dir / relative
    if (
        not path.is_file()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or sha256_file(path) != raw.get("sha256")
    ):
        raise CategoryDepthError(f"P2.5 {label} 文件指纹不一致")
    return path


def _validate_mapping_content(
    paths: Sequence[Path],
    config: CategoryBuildConfig,
) -> tuple[int, int, int]:
    expected_row = 0
    fine_codes: set[str] = set()
    coarse_codes: set[str] = set()
    for path in paths:
        table = pq.read_table(
            path,
            columns=[
                "poi_row_index",
                "category",
                "fine_category_id",
                "coarse_category_id",
            ],
        )
        row_indices = table["poi_row_index"].to_numpy(zero_copy_only=False)
        expected = np.arange(expected_row, expected_row + len(table), dtype=np.int64)
        if not np.array_equal(row_indices, expected):
            raise CategoryDepthError("category mapping poi_row_index 不连续")
        categories = table["category"].to_pylist()
        fine = table["fine_category_id"].to_pylist()
        coarse = table["coarse_category_id"].to_pylist()
        for category_path, fine_id, coarse_id in zip(
            categories, fine, coarse, strict=True
        ):
            if not isinstance(category_path, str) or not category_path.strip():
                raise CategoryDepthError("category mapping 存在空可读类别")
            if (
                not isinstance(fine_id, str)
                or ASCII_CATEGORY_CODE.fullmatch(fine_id) is None
                or coarse_id != fine_id[:2]
            ):
                raise CategoryDepthError("category mapping fine/coarse 关系非法")
            fine_codes.add(fine_id)
            coarse_codes.add(coarse_id)
        expected_row += len(table)
    if expected_row != config.frozen.poi_rows:
        raise CategoryDepthError("category mapping 全量行数不一致")
    return expected_row, len(fine_codes), len(coarse_codes)


def _validate_query_and_edge_content(
    stats_paths: Sequence[Path],
    depth_paths: Sequence[Path],
    edge_paths: Sequence[Path],
    tolerance: float,
) -> tuple[dict[str, int], int, float, float]:
    expected_query_id = 0
    depth_counts: Counter[str] = Counter()
    edge_rows = 0
    max_probability_error = 0.0
    max_weight_error = 0.0
    for stats_path, depth_path, edge_path in zip(
        stats_paths, depth_paths, edge_paths, strict=True
    ):
        stats = pq.read_table(
            stats_path, columns=["query_id", "is_p2_exact_core"]
        ).to_pylist()
        depth = pq.read_table(
            depth_path,
            columns=[
                "query_id",
                "supervision_depth",
                "supervision_label",
                "supervise_s1",
                "supervise_s2",
                "supervise_s3",
                "s1_reliability",
                "s2_reliability",
                "s3_reliability",
            ],
        ).to_pylist()
        if len(stats) != len(depth):
            raise CategoryDepthError("Query category stats/depth 行数不一致")
        depth_by_query: dict[int, int] = {}
        reliability_by_query_layer: dict[tuple[int, int], float] = {}
        for stats_row, depth_row in zip(stats, depth, strict=True):
            query_id = int(stats_row["query_id"])
            value = int(depth_row["supervision_depth"])
            label = str(depth_row["supervision_label"])
            if (
                query_id != expected_query_id
                or int(depth_row["query_id"]) != query_id
                or value not in range(4)
                or label != DEPTH_LABELS[value]
                or bool(stats_row["is_p2_exact_core"]) != (value == 3)
            ):
                raise CategoryDepthError("Query depth/D3/query_id 合同不一致")
            expected_query_id += 1
            expected_mask = tuple(value >= layer for layer in (1, 2, 3))
            actual_mask = (
                bool(depth_row["supervise_s1"]),
                bool(depth_row["supervise_s2"]),
                bool(depth_row["supervise_s3"]),
            )
            if actual_mask != expected_mask:
                raise CategoryDepthError(f"Query {query_id} layer mask 不一致")
            depth_by_query[query_id] = value
            for layer in (1, 2, 3):
                reliability = float(depth_row[f"s{layer}_reliability"])
                if not 0.0 <= reliability <= 1.0:
                    raise CategoryDepthError("Query reliability 越界")
                if expected_mask[layer - 1]:
                    reliability_by_query_layer[(query_id, layer)] = reliability
                elif reliability != 0.0:
                    raise CategoryDepthError("未启用层的 reliability 必须为 0")
            depth_counts[label] += 1

        probability_sums: Counter[tuple[int, int]] = Counter()
        weight_sums: Counter[tuple[int, int]] = Counter()
        observed_pairs: set[tuple[int, int, str]] = set()
        for row in pq.read_table(edge_path).to_pylist():
            query_id = int(row["query_id"])
            layer = int(row["layer"])
            value = int(row["supervision_depth"])
            key = (query_id, layer)
            pair_key = (query_id, layer, str(row["target_poi_id"]))
            if (
                query_id not in depth_by_query
                or value != depth_by_query[query_id]
                or layer not in (1, 2, 3)
                or layer > value
                or row["supervision_label"] != DEPTH_LABELS[value]
                or pair_key in observed_pairs
                or row["coarse_category_id"] != row["fine_category_id"][:2]
            ):
                raise CategoryDepthError("Query–POI layer edge 结构非法")
            probability = float(row["edge_probability"])
            reliability = float(row["query_reliability"])
            weight = float(row["edge_weight"])
            if (
                not 0.0 < probability <= 1.0
                or not 0.0 <= reliability <= 1.0
                or abs(reliability - reliability_by_query_layer[key]) > tolerance
                or abs(weight - probability * reliability) > tolerance
            ):
                raise CategoryDepthError("Query–POI layer edge 权重非法")
            observed_pairs.add(pair_key)
            probability_sums[key] += probability
            weight_sums[key] += weight
            edge_rows += 1
        expected_layer_keys = set(reliability_by_query_layer)
        if set(probability_sums) != expected_layer_keys:
            raise CategoryDepthError("启用 Query–层与实际边不一致")
        for key in expected_layer_keys:
            probability_error = abs(probability_sums[key] - 1.0)
            weight_error = abs(weight_sums[key] - reliability_by_query_layer[key])
            max_probability_error = max(max_probability_error, probability_error)
            max_weight_error = max(max_weight_error, weight_error)
            if probability_error > tolerance or weight_error > tolerance:
                raise CategoryDepthError("Query–层边权未归一")
    return (
        {label: depth_counts[label] for label in DEPTH_LABELS},
        edge_rows,
        max_probability_error,
        max_weight_error,
    )


def validate_category_depth(
    output_dir: Path,
    config: CategoryBuildConfig,
) -> CategoryDepthBuildResult:
    """Independently validate a completed full P2.5-CAT artifact."""

    output_dir = output_dir.resolve()
    if output_dir != config.query_depth_output_dir.resolve():
        raise CategoryDepthError("validate-only 输出路径必须是 canonical query_depth")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file() or not (output_dir / "_SUCCESS").is_file():
        raise CategoryDepthError("P2.5 输出缺少 manifest.json 或 _SUCCESS")
    manifest = _load_json(manifest_path, "P2.5 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "completed":
        raise CategoryDepthError("P2.5 manifest 版本或状态非法")
    config_info = manifest.get("config")
    outputs = manifest.get("outputs")
    stats = manifest.get("stats")
    access = manifest.get("source_access")
    if not all(isinstance(item, Mapping) for item in (config_info, outputs, stats, access)):
        raise CategoryDepthError("P2.5 manifest 缺少 config/outputs/stats/source_access")
    assert isinstance(config_info, Mapping)
    assert isinstance(outputs, Mapping)
    assert isinstance(stats, Mapping)
    assert isinstance(access, Mapping)
    if (
        config_info.get("sha256") != config.source_sha256
        or config_info.get("resolved_signature") != config.signature()
        or sha256_file(config.paths.p2_query_stats / "manifest.json")
        != config.frozen.p2_manifest_sha256
        or any(
            access.get(key) is not False
            for key in ("raw_train_read", "validation_read", "test_read")
        )
        or access.get("p2_query_shards_read") is not False
        or access.get("p2_false_negative_mask_read") is not False
    ):
        raise CategoryDepthError("P2.5 配置、冻结 P2 或数据隔离证据不一致")
    mapping_paths, mapping_rows = _validate_output_group(
        output_dir,
        outputs.get("category_mapping"),
        CATEGORY_MAPPING_SCHEMA,
        "category_mapping",
    )
    stats_paths, query_rows = _validate_output_group(
        output_dir,
        outputs.get("query_category_stats"),
        QUERY_CATEGORY_STATS_SCHEMA,
        "query_category_stats",
    )
    depth_paths, depth_rows = _validate_output_group(
        output_dir,
        outputs.get("query_category_depth"),
        QUERY_CATEGORY_DEPTH_SCHEMA,
        "query_category_depth",
    )
    edge_paths, edge_rows = _validate_output_group(
        output_dir,
        outputs.get("query_poi_layer_edges"),
        QUERY_POI_LAYER_EDGE_SCHEMA,
        "query_poi_layer_edges",
    )
    metrics_path = _validate_plain_output(output_dir, outputs.get("metrics"), "metrics")
    _validate_plain_output(output_dir, outputs.get("report"), "report")
    _validate_plain_output(
        output_dir, outputs.get("resolved_config"), "resolved_config"
    )
    metrics = _load_json(metrics_path, "P2.5 metrics")
    expected_queries = int(metrics["queries"]["total"])
    if (
        mapping_rows != config.frozen.poi_rows
        or query_rows != expected_queries
        or depth_rows != expected_queries
        or edge_rows != int(metrics["edges"]["layer_edge_rows"])
    ):
        raise CategoryDepthError("P2.5 manifest/metrics 输出行数不守恒")
    checked_rows, fine_count, coarse_count = _validate_mapping_content(
        mapping_paths, config
    )
    category_metrics = metrics.get("category_mapping")
    if not isinstance(category_metrics, Mapping):
        raise CategoryDepthError("P2.5 metrics 缺少 category_mapping")
    observed_fine_count = int(
        category_metrics.get(
            "observed_fine_category_count",
            category_metrics.get("fine_category_count", -1),
        )
    )
    missing_fine_codes = category_metrics.get("missing_fine_category_codes", [])
    if (
        checked_rows != mapping_rows
        or fine_count != observed_fine_count
        or fine_count > config.category.fine_expected_count
        or coarse_count != config.category.coarse_expected_count
        or not isinstance(missing_fine_codes, list)
        or len(missing_fine_codes)
        != config.category.fine_expected_count - fine_count
    ):
        raise CategoryDepthError("P2.5 category mapping 类别覆盖不一致")
    tolerance = float(metrics["edges"]["normalization_tolerance"])
    (
        depth_counts,
        checked_edge_rows,
        probability_error,
        weight_error,
    ) = _validate_query_and_edge_content(
        stats_paths, depth_paths, edge_paths, tolerance
    )
    expected_depth_counts = {
        key: int(value) for key, value in metrics["queries"]["depth_counts"].items()
    }
    if (
        depth_counts != expected_depth_counts
        or checked_edge_rows != edge_rows
        or probability_error > tolerance
        or weight_error > tolerance
        or expected_depth_counts["D3_EXACT"]
        != int(_load_json(config.paths.p2_query_stats / "manifest.json", "P2 manifest")["stats"]["retained_queries"])
    ):
        raise CategoryDepthError("P2.5 depth/D3/edge 独立复核失败")
    return CategoryDepthBuildResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        category_rows=mapping_rows,
        unique_queries=query_rows,
        layer_edges=edge_rows,
        depth_counts=depth_counts,
        reused=True,
    )
