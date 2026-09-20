"""Build deterministic Train-only query–POI statistics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from qg_prqk.artifacts import sha256_file, utc_now
from qg_prqk.config import QGPRQKConfig, QueryStatsConfig
from qg_prqk.data.contracts import (
    QGPRQKDataContractError,
    load_json_object,
    validate_sft_manifest,
    validate_train_record,
)
from qg_prqk.data.query_normalization import NORMALIZATION_VERSION, normalize_query

try:
    import orjson
except ImportError:  # pragma: no cover - supported environment includes orjson
    orjson = None


SCHEMA_VERSION = "qg-prqk-query-stats-v1"
HASH_PERSONALIZATION = b"qg-query-v1"
SHARD_SCHEMA = pa.schema(
    [
        pa.field("raw_query", pa.string(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
    ]
)
QUERY_STATS_SCHEMA = pa.schema(
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
        pa.field("top2_share", pa.float64(), nullable=False),
        pa.field("margin", pa.float64(), nullable=False),
        pa.field("entropy", pa.float64(), nullable=False),
        pa.field("normalized_entropy", pa.float64(), nullable=False),
        pa.field("is_high_confidence", pa.bool_(), nullable=False),
    ]
)
QUERY_POI_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
        pa.field("pair_count", pa.int64(), nullable=False),
        pa.field("pair_share", pa.float64(), nullable=False),
        pa.field("is_top1", pa.bool_(), nullable=False),
        pa.field("retained_for_sid", pa.bool_(), nullable=False),
        pa.field("sid_weight", pa.float64(), nullable=False),
    ]
)
FALSE_NEGATIVE_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
        pa.field("pair_count", pa.int64(), nullable=False),
        pa.field("pair_share", pa.float64(), nullable=False),
    ]
)


class QueryStatsError(RuntimeError):
    """Raised when QG Query statistics cannot be built or validated."""


@dataclass(frozen=True)
class QuerySummary:
    normalized_query: str
    representative_raw_query: str
    query_count: int
    distinct_poi_count: int
    top1_poi_id: str
    top1_count: int
    top1_share: float
    top2_share: float
    margin: float
    entropy: float
    normalized_entropy: float
    is_high_confidence: bool
    sid_weight: float


@dataclass(frozen=True)
class QueryStatsBuildResult:
    output_dir: Path
    manifest_path: Path
    source_rows: int
    unique_queries: int
    unique_query_poi_pairs: int
    retained_queries: int
    false_negative_pairs: int
    reused: bool


def stable_query_shard(normalized_query: str, num_shards: int) -> int:
    """Map one normalized Query to a stable QG-specific hash shard."""

    if not isinstance(normalized_query, str) or not normalized_query:
        raise QueryStatsError("normalized_query 必须是非空字符串")
    if not 1 <= num_shards <= 65_536:
        raise QueryStatsError("num_shards 必须位于 [1,65536]")
    digest = hashlib.blake2b(
        normalized_query.encode("utf-8"),
        digest_size=8,
        person=HASH_PERSONALIZATION,
    ).digest()
    return int.from_bytes(digest, "big") % num_shards


def summarize_query(
    normalized_query: str,
    pair_counts: Mapping[str, int],
    raw_query_counts: Mapping[str, int],
    settings: QueryStatsConfig,
) -> QuerySummary:
    """Compute filtering statistics for one normalized Query group."""

    if not pair_counts or any(count <= 0 for count in pair_counts.values()):
        raise QueryStatsError("pair_counts 必须包含正计数")
    if not raw_query_counts or any(count <= 0 for count in raw_query_counts.values()):
        raise QueryStatsError("raw_query_counts 必须包含正计数")
    query_count = sum(pair_counts.values())
    if sum(raw_query_counts.values()) != query_count:
        raise QueryStatsError("raw Query 与 Query–POI 计数不守恒")
    ranked = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
    top1_poi_id, top1_count = ranked[0]
    top1_share = top1_count / query_count
    top2_share = ranked[1][1] / query_count if len(ranked) > 1 else 0.0
    margin = top1_share - top2_share
    probabilities = [count / query_count for _, count in ranked]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized_entropy = entropy / math.log(len(ranked)) if len(ranked) > 1 else 0.0
    is_high_confidence = (
        query_count >= settings.min_query_count
        and top1_count >= settings.min_pair_count
        and top1_share >= settings.min_top1_share
        and margin >= settings.min_margin
        and normalized_entropy <= settings.max_normalized_entropy
    )
    sid_weight = 0.0
    if is_high_confidence:
        sid_weight = max(
            0.0,
            min(math.log1p(top1_count), math.log(21.0))
            * top1_share**2
            * (1.0 - normalized_entropy),
        )
    representative_raw_query = min(
        raw_query_counts,
        key=lambda raw_query: (-raw_query_counts[raw_query], raw_query),
    )
    return QuerySummary(
        normalized_query=normalized_query,
        representative_raw_query=representative_raw_query,
        query_count=query_count,
        distinct_poi_count=len(ranked),
        top1_poi_id=top1_poi_id,
        top1_count=top1_count,
        top1_share=top1_share,
        top2_share=top2_share,
        margin=margin,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        is_high_confidence=is_high_confidence,
        sid_weight=sid_weight,
    )


def false_negative_targets(
    pair_counts: Mapping[str, int],
    settings: QueryStatsConfig,
) -> tuple[str, ...]:
    """Return all observed positives that must be masked during negative mining."""

    total = sum(pair_counts.values())
    if total <= 0:
        raise QueryStatsError("pair_counts 总数必须大于 0")
    return tuple(
        poi_id
        for poi_id, count in sorted(pair_counts.items())
        if count >= settings.false_negative_min_count
        and count / total >= settings.false_negative_min_share
    )


def _part_name(shard_id: int, num_shards: int) -> str:
    width = max(5, len(str(num_shards)))
    return f"part-{shard_id:0{width}d}-of-{num_shards:0{width}d}.parquet"


def _table(schema: pa.Schema, columns: Sequence[Sequence[Any]]) -> pa.Table:
    arrays = [
        pa.array(values, type=field.type)
        for values, field in zip(columns, schema, strict=True)
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_table(path: Path, schema: pa.Schema, columns: Sequence[Sequence[Any]]) -> None:
    table = _table(schema, columns)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=max(1, len(table)),
    )


def _parse_record(raw_line: bytes, source: str) -> Mapping[str, Any]:
    try:
        record = orjson.loads(raw_line) if orjson is not None else json.loads(raw_line)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise QueryStatsError(f"{source} JSON 非法") from error
    if not isinstance(record, Mapping):
        raise QueryStatsError(f"{source} 必须是 JSON object")
    return record


class _ShardWriter:
    def __init__(self, directory: Path, settings: QueryStatsConfig) -> None:
        self.directory = directory
        self.settings = settings
        self.buffers: list[tuple[list[str], list[str], list[str]]] = [
            ([], [], []) for _ in range(settings.num_shards)
        ]
        self.writers: list[pq.ParquetWriter | None] = [None] * settings.num_shards
        self.rows = [0] * settings.num_shards
        directory.mkdir()

    def add(self, raw_query: str, normalized_query: str, poi_id: str) -> None:
        shard_id = stable_query_shard(normalized_query, self.settings.num_shards)
        columns = self.buffers[shard_id]
        columns[0].append(raw_query)
        columns[1].append(normalized_query)
        columns[2].append(poi_id)
        self.rows[shard_id] += 1
        if len(columns[0]) >= self.settings.buffer_rows_per_shard:
            self._flush(shard_id)

    def _flush(self, shard_id: int) -> None:
        columns = self.buffers[shard_id]
        if not columns[0]:
            return
        writer = self.writers[shard_id]
        if writer is None:
            writer = pq.ParquetWriter(
                self.directory / _part_name(shard_id, self.settings.num_shards),
                SHARD_SCHEMA,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            self.writers[shard_id] = writer
        writer.write_table(_table(SHARD_SCHEMA, columns))
        for values in columns:
            values.clear()

    def close(self) -> None:
        for shard_id in range(self.settings.num_shards):
            self._flush(shard_id)
            writer = self.writers[shard_id]
            if writer is None:
                writer = pq.ParquetWriter(
                    self.directory / _part_name(shard_id, self.settings.num_shards),
                    SHARD_SCHEMA,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.close()


def _scan_train(
    train_path: Path,
    shard_dir: Path,
    settings: QueryStatsConfig,
    config: QGPRQKConfig,
    *,
    limit: int | None,
) -> tuple[int, int, str, list[int]]:
    writer = _ShardWriter(shard_dir, settings)
    digest = hashlib.sha256()
    source_bytes = 0
    rows = 0
    progress = tqdm(
        desc="QG Train Query normalization",
        unit="row",
        disable=not config.runtime.show_progress,
        total=limit,
    )
    try:
        with train_path.open("rb", buffering=8 * 1024 * 1024) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if limit is not None and rows >= limit:
                    break
                if not raw_line.strip():
                    raise QueryStatsError(f"{train_path}:{line_number} 是空行")
                digest.update(raw_line)
                source_bytes += len(raw_line)
                source = f"{train_path}:{line_number}"
                record = _parse_record(raw_line, source)
                try:
                    raw_query, poi_id = validate_train_record(
                        record, config.data_contracts, source
                    )
                except QGPRQKDataContractError as error:
                    raise QueryStatsError(str(error)) from error
                normalized_query = normalize_query(raw_query)
                writer.add(raw_query, normalized_query, poi_id)
                rows += 1
                progress.update(1)
    finally:
        progress.close()
        writer.close()
    return rows, source_bytes, digest.hexdigest(), writer.rows


def _file_info(path: Path, shard_id: int) -> dict[str, Any]:
    return {
        "shard_id": shard_id,
        "file": path.name,
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _aggregate_shard(
    shard_path: Path,
    shard_id: int,
    first_query_id: int,
    settings: QueryStatsConfig,
    output_dirs: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    pairs_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    raws_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    parquet = pq.ParquetFile(shard_path)
    source_rows = 0
    for batch in parquet.iter_batches(batch_size=settings.read_batch_rows):
        raw_queries = batch.column(0).to_pylist()
        normalized_queries = batch.column(1).to_pylist()
        poi_ids = batch.column(2).to_pylist()
        for raw_query, normalized_query, poi_id in zip(
            raw_queries, normalized_queries, poi_ids, strict=True
        ):
            pairs_by_query[normalized_query][poi_id] += 1
            raws_by_query[normalized_query][raw_query] += 1
        source_rows += len(batch)

    queries = sorted(pairs_by_query)
    stats_columns: list[list[Any]] = [[] for _ in QUERY_STATS_SCHEMA]
    pair_columns: list[list[Any]] = [[] for _ in QUERY_POI_SCHEMA]
    false_negative_columns: list[list[Any]] = [[] for _ in FALSE_NEGATIVE_SCHEMA]
    metrics: dict[str, Any] = {
        "source_rows": source_rows,
        "queries": 0,
        "pairs": 0,
        "retained_queries": 0,
        "false_negative_pairs": 0,
        "covered_pois": set(),
        "retained_pois": set(),
        "query_count_histogram": Counter(),
        "samples": {"high_confidence": [], "ambiguous": [], "low_frequency": []},
    }
    for offset, query in enumerate(queries):
        query_id = first_query_id + offset
        query_pairs = pairs_by_query[query]
        query_raws = raws_by_query[query]
        summary = summarize_query(query, query_pairs, query_raws, settings)
        stats_values = (
            query_id,
            shard_id,
            query,
            summary.representative_raw_query,
            summary.query_count,
            summary.distinct_poi_count,
            summary.top1_poi_id,
            summary.top1_count,
            summary.top1_share,
            summary.top2_share,
            summary.margin,
            summary.entropy,
            summary.normalized_entropy,
            summary.is_high_confidence,
        )
        for column, value in zip(stats_columns, stats_values, strict=True):
            column.append(value)
        masked_targets = set(false_negative_targets(query_pairs, settings))
        for poi_id, count in sorted(query_pairs.items()):
            share = count / summary.query_count
            is_top1 = poi_id == summary.top1_poi_id
            retained = summary.is_high_confidence and is_top1
            pair_values = (
                query_id,
                query,
                poi_id,
                count,
                share,
                is_top1,
                retained,
                summary.sid_weight if retained else 0.0,
            )
            for column, value in zip(pair_columns, pair_values, strict=True):
                column.append(value)
            if poi_id in masked_targets:
                false_negative_values = (query_id, query, poi_id, count, share)
                for column, value in zip(
                    false_negative_columns, false_negative_values, strict=True
                ):
                    column.append(value)
            metrics["covered_pois"].add(poi_id)
        metrics["queries"] += 1
        metrics["pairs"] += len(query_pairs)
        metrics["retained_queries"] += int(summary.is_high_confidence)
        metrics["false_negative_pairs"] += len(masked_targets)
        if summary.is_high_confidence:
            metrics["retained_pois"].add(summary.top1_poi_id)
        bucket = "1" if summary.query_count == 1 else "2-4" if summary.query_count <= 4 else "5+"
        metrics["query_count_histogram"][bucket] += 1
        sample = {
            "query_id": query_id,
            "normalized_query": query,
            "representative_raw_query": summary.representative_raw_query,
            "query_count": summary.query_count,
            "distinct_poi_count": summary.distinct_poi_count,
            "top1_poi_id": summary.top1_poi_id,
            "top1_share": summary.top1_share,
            "margin": summary.margin,
            "normalized_entropy": summary.normalized_entropy,
        }
        category = (
            "high_confidence"
            if summary.is_high_confidence
            else "ambiguous"
            if summary.distinct_poi_count > 1
            else "low_frequency"
        )
        if len(metrics["samples"][category]) < 5:
            metrics["samples"][category].append(sample)

    name = _part_name(shard_id, settings.num_shards)
    _write_table(output_dirs["query_stats"] / name, QUERY_STATS_SCHEMA, stats_columns)
    _write_table(output_dirs["query_poi_pairs"] / name, QUERY_POI_SCHEMA, pair_columns)
    _write_table(
        output_dirs["false_negative_mask"] / name,
        FALSE_NEGATIVE_SCHEMA,
        false_negative_columns,
    )
    files = {
        key: _file_info(directory / name, shard_id)
        for key, directory in output_dirs.items()
    }
    return metrics, files


def _validate_file_group(
    output_dir: Path,
    output: Any,
    schema: pa.Schema,
    expected_shards: int,
) -> int:
    if not isinstance(output, Mapping) or not isinstance(output.get("files"), list):
        raise QueryStatsError("Query 统计 manifest 输出清单非法")
    files = output["files"]
    if len(files) != expected_shards:
        raise QueryStatsError("Query 统计 manifest 分片数不一致")
    rows = 0
    directory = output_dir / str(output.get("dir", ""))
    for shard_id, info in enumerate(files):
        path = directory / str(info.get("file", ""))
        if int(info.get("shard_id", -1)) != shard_id or not path.is_file():
            raise QueryStatsError(f"Query 统计分片缺失或顺序非法：{path}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != schema:
            raise QueryStatsError(f"Query 统计 Schema 不一致：{path}")
        if parquet.metadata.num_rows != int(info.get("rows", -1)):
            raise QueryStatsError(f"Query 统计行数不一致：{path}")
        if path.stat().st_size != int(info.get("size_bytes", -1)):
            raise QueryStatsError(f"Query 统计文件大小不一致：{path}")
        if sha256_file(path) != info.get("sha256"):
            raise QueryStatsError(f"Query 统计 SHA256 不一致：{path}")
        rows += parquet.metadata.num_rows
    return rows


def validate_query_stats(output_dir: Path) -> QueryStatsBuildResult:
    """Validate a completed P2 output and every recorded Parquet shard."""

    output_dir = output_dir.resolve()
    manifest = load_json_object(output_dir / "manifest.json", "QG Query 统计 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "completed":
        raise QueryStatsError("QG Query 统计 manifest 状态或版本非法")
    if not (output_dir / "_SUCCESS").is_file():
        raise QueryStatsError("QG Query 统计缺少 _SUCCESS")
    settings = manifest.get("settings")
    outputs = manifest.get("outputs")
    stats = manifest.get("stats")
    if not all(isinstance(value, Mapping) for value in (settings, outputs, stats)):
        raise QueryStatsError("QG Query 统计 manifest 缺少 settings/outputs/stats")
    num_shards = int(settings["num_shards"])
    schemas = {
        "query_shards": SHARD_SCHEMA,
        "query_stats": QUERY_STATS_SCHEMA,
        "query_poi_pairs": QUERY_POI_SCHEMA,
        "false_negative_mask": FALSE_NEGATIVE_SCHEMA,
    }
    rows = {
        key: _validate_file_group(output_dir, outputs.get(key), schema, num_shards)
        for key, schema in schemas.items()
    }
    expected = {
        "source_rows": rows["query_shards"],
        "unique_queries": rows["query_stats"],
        "unique_query_poi_pairs": rows["query_poi_pairs"],
        "false_negative_pairs": rows["false_negative_mask"],
    }
    for key, value in expected.items():
        if int(stats.get(key, -1)) != value:
            raise QueryStatsError(f"QG Query 统计 {key} 与 manifest 不一致")
    report = outputs.get("stats_report")
    if not isinstance(report, Mapping):
        raise QueryStatsError("QG Query 统计 manifest 缺少 stats_report")
    report_path = output_dir / str(report.get("file", ""))
    if not report_path.is_file():
        raise QueryStatsError("QG Query 统计缺少 stats_report 文件")
    if report_path.stat().st_size != int(report.get("size_bytes", -1)):
        raise QueryStatsError("QG Query stats_report 文件大小不一致")
    if sha256_file(report_path) != report.get("sha256"):
        raise QueryStatsError("QG Query stats_report SHA256 不一致")
    return QueryStatsBuildResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        source_rows=expected["source_rows"],
        unique_queries=expected["unique_queries"],
        unique_query_poi_pairs=expected["unique_query_poi_pairs"],
        retained_queries=int(stats["retained_queries"]),
        false_negative_pairs=expected["false_negative_pairs"],
        reused=True,
    )


def _manifest_output(directory: Path, files: list[dict[str, Any]], staging: Path) -> dict[str, Any]:
    return {
        "dir": str(directory.relative_to(staging)),
        "rows": sum(int(info["rows"]) for info in files),
        "files": files,
    }


def build_query_stats(
    config: QGPRQKConfig,
    *,
    output_dir: Path,
    limit: int | None = None,
) -> QueryStatsBuildResult:
    """Build QG normalized Query assets from the canonical Train split."""

    if limit is not None and limit <= 0:
        raise QueryStatsError("limit 必须大于 0")
    output_dir = output_dir.resolve()
    if output_dir.parent != config.paths.output_dir.resolve():
        raise QueryStatsError("P2 输出必须是配置 output_dir 下的 query_stats 子目录")
    if output_dir.name != "query_stats":
        raise QueryStatsError("P2 输出目录名必须为 query_stats")
    train_path = (config.paths.sft_data_dir / "train.jsonl").resolve()
    if train_path.name != "train.jsonl":
        raise QueryStatsError("Query 统计只允许读取 train.jsonl")
    if not train_path.is_file():
        raise QueryStatsError(f"Train 文件不存在：{train_path}")
    if sha256_file(config.paths.sft_manifest) != config.frozen_inputs.sft_manifest_sha256:
        raise QueryStatsError("SFT manifest SHA256 与冻结配置不一致")
    sft_manifest = load_json_object(config.paths.sft_manifest, "SFT manifest")
    try:
        split_rows = validate_sft_manifest(
            sft_manifest, config.data_contracts, config.frozen_inputs
        )
    except QGPRQKDataContractError as error:
        raise QueryStatsError(str(error)) from error
    expected_train_rows = split_rows["train.jsonl"]
    requested_limit = min(limit, expected_train_rows) if limit is not None else None
    signature = config.signature()
    provenance_path = config.paths.project_root / "qg_prqk/configs/code_provenance.yaml"
    if not provenance_path.is_file():
        raise QueryStatsError(f"代码 provenance 不存在：{provenance_path}")
    try:
        provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise QueryStatsError("代码 provenance 读取失败") from error
    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("source_repository_commit"), str
    ):
        raise QueryStatsError("代码 provenance 缺少 source_repository_commit")

    if output_dir.exists():
        if config.runtime.resume:
            result = validate_query_stats(output_dir)
            manifest = load_json_object(result.manifest_path, "QG Query 统计 manifest")
            source = manifest.get("source", {})
            if manifest.get("config_signature") != signature:
                raise QueryStatsError("已有 P2 产物与当前配置签名不一致")
            if source.get("declared_train_sha256") != config.frozen_inputs.train_sha256:
                raise QueryStatsError("已有 P2 产物与当前 Train 指纹不一致")
            if source.get("limit") != requested_limit:
                raise QueryStatsError("已有 P2 产物的 limit 与当前请求不一致")
            return result
        if not config.runtime.overwrite:
            raise QueryStatsError(f"输出已存在且 overwrite=false：{output_dir}")
        shutil.rmtree(output_dir)

    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        if not config.runtime.overwrite:
            raise QueryStatsError(f"存在未完成目录，请先审计：{staging}")
        shutil.rmtree(staging)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    directories = {
        name: staging / name
        for name in (
            "query_shards",
            "query_stats",
            "query_poi_pairs",
            "false_negative_mask",
        )
    }
    for name, directory in directories.items():
        if name != "query_shards":
            directory.mkdir()

    started = time.perf_counter()
    source_rows, source_bytes, scanned_sha256, shard_rows = _scan_train(
        train_path,
        directories["query_shards"],
        config.query_stats,
        config,
        limit=requested_limit,
    )
    if requested_limit is None:
        if source_rows != expected_train_rows:
            raise QueryStatsError(
                f"Train 行数 {source_rows:,} != manifest {expected_train_rows:,}"
            )
        if scanned_sha256 != config.frozen_inputs.train_sha256:
            raise QueryStatsError("Train 全量 SHA256 与冻结配置不一致")
    elif source_rows != requested_limit:
        raise QueryStatsError(f"Train 样例不足：期望 {requested_limit}，实际 {source_rows}")

    output_files: dict[str, list[dict[str, Any]]] = {
        key: [] for key in directories
    }
    totals: dict[str, Any] = {
        "queries": 0,
        "pairs": 0,
        "retained_queries": 0,
        "false_negative_pairs": 0,
        "covered_pois": set(),
        "retained_pois": set(),
        "query_count_histogram": Counter(),
        "samples": {"high_confidence": [], "ambiguous": [], "low_frequency": []},
    }
    for shard_id in tqdm(
        range(config.query_stats.num_shards),
        desc="QG Query–POI statistics",
        unit="shard",
        disable=not config.runtime.show_progress,
    ):
        shard_path = directories["query_shards"] / _part_name(
            shard_id, config.query_stats.num_shards
        )
        output_files["query_shards"].append(_file_info(shard_path, shard_id))
        metrics, files = _aggregate_shard(
            shard_path,
            shard_id,
            totals["queries"],
            config.query_stats,
            {key: directories[key] for key in directories if key != "query_shards"},
        )
        if metrics["source_rows"] != shard_rows[shard_id]:
            raise QueryStatsError(f"Query shard {shard_id} 行数不守恒")
        for key, info in files.items():
            output_files[key].append(info)
        for key in ("queries", "pairs", "retained_queries", "false_negative_pairs"):
            totals[key] += metrics[key]
        totals["covered_pois"].update(metrics["covered_pois"])
        totals["retained_pois"].update(metrics["retained_pois"])
        totals["query_count_histogram"].update(metrics["query_count_histogram"])
        for category, samples in metrics["samples"].items():
            remaining = 5 - len(totals["samples"][category])
            totals["samples"][category].extend(samples[: max(0, remaining)])

    if sum(shard_rows) != source_rows:
        raise QueryStatsError("Query shard 总行数与 Train 扫描行数不守恒")
    stats = {
        "source_rows": source_rows,
        "unique_queries": totals["queries"],
        "unique_query_poi_pairs": totals["pairs"],
        "retained_queries": totals["retained_queries"],
        "filtered_queries": totals["queries"] - totals["retained_queries"],
        "retention_rate": (
            totals["retained_queries"] / totals["queries"] if totals["queries"] else 0.0
        ),
        "false_negative_pairs": totals["false_negative_pairs"],
        "covered_pois": len(totals["covered_pois"]),
        "retained_pois": len(totals["retained_pois"]),
        "query_count_histogram": dict(totals["query_count_histogram"]),
    }
    report = {
        "schema_version": "qg-prqk-query-stats-report-v1",
        "is_sample": requested_limit is not None,
        "stats": stats,
        "samples": totals["samples"],
    }
    report_path = staging / "stats_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "method": config.project.method,
        "version": config.project.version,
        "built_at": utc_now(),
        "config_path": str(config.source_path),
        "config_sha256": config.source_sha256,
        "config_signature": signature,
        "code": {
            "source_repository_commit": provenance["source_repository_commit"],
            "worktree_state": "not_asserted_by_runtime",
            "provenance_path": str(provenance_path),
            "provenance_sha256": sha256_file(provenance_path),
        },
        "source": {
            "split": "train",
            "train_path": str(train_path),
            "sft_manifest_path": str(config.paths.sft_manifest),
            "sft_manifest_sha256": config.frozen_inputs.sft_manifest_sha256,
            "declared_train_rows": expected_train_rows,
            "declared_train_sha256": config.frozen_inputs.train_sha256,
            "limit": requested_limit,
            "scanned_rows": source_rows,
            "scanned_bytes": source_bytes,
            "scanned_sha256": scanned_sha256,
            "is_prefix_sample": requested_limit is not None,
            "valid_and_test_read": False,
        },
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "steps": [
                "Unicode NFKC",
                "English lowercase",
                "common punctuation canonicalization",
                "collapse whitespace",
                "strip boundary whitespace",
            ],
            "location_words_deleted": False,
        },
        "settings": {
            **config.resolved_payload()["query_stats"],
            "hash": "blake2b-64/qg-query-v1",
            "query_id_order": "shard ascending, then normalized Query lexicographic",
        },
        "weight_formula": (
            "max(0,min(log(1+n_qp),log(21))*top1_share^2*(1-H_norm))"
        ),
        "stats": stats,
        "outputs": {
            key: _manifest_output(directories[key], output_files[key], staging)
            for key in directories
        }
        | {
            "stats_report": {
                "file": report_path.name,
                "size_bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            }
        },
        "validation": {
            "train_only": True,
            "valid_and_test_read": False,
            "source_row_conservation": True,
            "normalized_query_colocated_by_stable_hash": True,
            "query_ids_contiguous": True,
            "top1_tie_break": "pair count descending, then poi_id lexicographic",
            "representative_raw_query_tie_break": (
                "raw count descending, then raw Query lexicographic"
            ),
            "false_negative_mask_keeps_all_threshold_positives": True,
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
    return QueryStatsBuildResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        source_rows=source_rows,
        unique_queries=totals["queries"],
        unique_query_poi_pairs=totals["pairs"],
        retained_queries=totals["retained_queries"],
        false_negative_pairs=totals["false_negative_pairs"],
        reused=False,
    )
