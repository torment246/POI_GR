"""Aggregate Train Query shards into bounded-memory Query and POI statistics."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .query_shards import QueryShardError, validate_query_shards


SCHEMA_VERSION = "train-query-stats-v1"
POI_HASH_PERSONALIZATION = b"poi-stats-v1"
QUERY_CATALOG_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("query_shard_id", pa.int32(), nullable=False),
        pa.field("query", pa.string(), nullable=False),
        pa.field("train_order_count", pa.int64(), nullable=False),
        pa.field("poi_df", pa.int32(), nullable=False),
    ]
)
QUERY_POI_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
        pa.field("train_order_count", pa.int64(), nullable=False),
    ]
)
POI_STATS_SCHEMA = pa.schema(
    [
        pa.field("target_poi_id", pa.string(), nullable=False),
        pa.field("train_order_count", pa.int64(), nullable=False),
        pa.field("unique_query_count", pa.int32(), nullable=False),
    ]
)


class QueryStatsError(RuntimeError):
    """Raised when Query statistics cannot be built or validated."""


@dataclass(frozen=True)
class QueryStatsBuildResult:
    output_dir: Path
    manifest_path: Path
    source_rows: int
    unique_queries: int
    unique_query_poi_pairs: int
    covered_pois: int
    reused: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise QueryStatsError(f"{name} 不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise QueryStatsError(f"{name} JSON 非法：{path}") from error
    if not isinstance(payload, dict):
        raise QueryStatsError(f"{name} 必须是 JSON object：{path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _part_name(partition_id: int, num_partitions: int) -> str:
    width = max(5, len(str(num_partitions)))
    return f"part-{partition_id:0{width}d}-of-{num_partitions:0{width}d}.parquet"


def stable_poi_partition(poi_id: str, num_partitions: int) -> int:
    """Map a POI ID to a deterministic bounded-memory aggregation partition."""

    if not isinstance(poi_id, str) or not poi_id.strip():
        raise QueryStatsError("poi_id 必须是非空字符串")
    if num_partitions <= 0 or num_partitions > 65_536:
        raise QueryStatsError("num_partitions 必须位于 [1, 65536]")
    digest = hashlib.blake2b(
        poi_id.encode("utf-8"),
        digest_size=8,
        person=POI_HASH_PERSONALIZATION,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % num_partitions


def _table_from_columns(schema: pa.Schema, columns: list[list[Any]]) -> pa.Table:
    arrays = [
        pa.array(values, type=field.type)
        for values, field in zip(columns, schema, strict=True)
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_table(path: Path, schema: pa.Schema, columns: list[list[Any]]) -> None:
    table = _table_from_columns(schema, columns)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=max(1, len(table)),
    )


def _file_info(path: Path, partition_id: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "partition_id": partition_id,
        "file": path.name,
        "rows": parquet.metadata.num_rows,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


class _PoiPairPartitioner:
    def __init__(
        self,
        output_dir: Path,
        *,
        num_partitions: int,
        buffer_rows_per_partition: int,
    ) -> None:
        if buffer_rows_per_partition <= 0:
            raise QueryStatsError("buffer_rows_per_partition 必须大于 0")
        self.output_dir = output_dir
        self.num_partitions = num_partitions
        self.buffer_rows_per_partition = buffer_rows_per_partition
        self.buffers: list[tuple[list[int], list[str], list[int]]] = [
            ([], [], []) for _ in range(num_partitions)
        ]
        self.writers: list[pq.ParquetWriter | None] = [None] * num_partitions
        self.rows = [0] * num_partitions
        output_dir.mkdir(parents=True)

    def add(self, query_id: int, poi_id: str, order_count: int) -> None:
        partition_id = stable_poi_partition(poi_id, self.num_partitions)
        query_ids, poi_ids, order_counts = self.buffers[partition_id]
        query_ids.append(query_id)
        poi_ids.append(poi_id)
        order_counts.append(order_count)
        self.rows[partition_id] += 1
        if len(query_ids) >= self.buffer_rows_per_partition:
            self._flush(partition_id)

    def _flush(self, partition_id: int) -> None:
        columns = self.buffers[partition_id]
        if not columns[0]:
            return
        writer = self.writers[partition_id]
        if writer is None:
            writer = pq.ParquetWriter(
                self.output_dir / _part_name(partition_id, self.num_partitions),
                QUERY_POI_SCHEMA,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            self.writers[partition_id] = writer
        writer.write_table(_table_from_columns(QUERY_POI_SCHEMA, list(columns)))
        for values in columns:
            values.clear()

    def close(self) -> None:
        for partition_id in range(self.num_partitions):
            self._flush(partition_id)
            writer = self.writers[partition_id]
            if writer is None:
                writer = pq.ParquetWriter(
                    self.output_dir / _part_name(partition_id, self.num_partitions),
                    QUERY_POI_SCHEMA,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.close()


def _aggregate_query_shard(
    path: Path,
    *,
    query_shard_id: int,
    first_query_id: int,
    read_batch_rows: int,
    query_catalog_path: Path,
    query_poi_path: Path,
    poi_partitioner: _PoiPairPartitioner,
) -> tuple[int, int, int, int]:
    pair_counts: dict[tuple[str, str], int] = {}
    source_rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=read_batch_rows,
        columns=["query", "target_poi_id"],
    ):
        queries = batch.column(0).to_pylist()
        poi_ids = batch.column(1).to_pylist()
        for query, poi_id in zip(queries, poi_ids, strict=True):
            key = (query, poi_id)
            pair_counts[key] = pair_counts.get(key, 0) + 1
        source_rows += len(batch)

    sorted_pairs = sorted(pair_counts.items(), key=lambda item: item[0])
    del pair_counts
    catalog_columns: list[list[Any]] = [[], [], [], [], []]
    query_poi_columns: list[list[Any]] = [[], [], []]
    query_id = first_query_id
    pair_index = 0
    order_count_sum = 0

    while pair_index < len(sorted_pairs):
        query = sorted_pairs[pair_index][0][0]
        group_start = pair_index
        query_order_count = 0
        while (
            pair_index < len(sorted_pairs) and sorted_pairs[pair_index][0][0] == query
        ):
            (_, poi_id), order_count = sorted_pairs[pair_index]
            query_poi_columns[0].append(query_id)
            query_poi_columns[1].append(poi_id)
            query_poi_columns[2].append(order_count)
            poi_partitioner.add(query_id, poi_id, order_count)
            query_order_count += order_count
            pair_index += 1
        poi_df = pair_index - group_start
        catalog_columns[0].append(query_id)
        catalog_columns[1].append(query_shard_id)
        catalog_columns[2].append(query)
        catalog_columns[3].append(query_order_count)
        catalog_columns[4].append(poi_df)
        order_count_sum += query_order_count
        query_id += 1

    _write_table(query_catalog_path, QUERY_CATALOG_SCHEMA, catalog_columns)
    _write_table(query_poi_path, QUERY_POI_SCHEMA, query_poi_columns)
    return (
        source_rows,
        len(catalog_columns[0]),
        len(query_poi_columns[0]),
        order_count_sum,
    )


def _aggregate_poi_partition(
    work_path: Path,
    output_path: Path,
    *,
    read_batch_rows: int,
) -> tuple[int, int, int]:
    poi_stats: dict[str, list[int]] = {}
    parquet = pq.ParquetFile(work_path)
    for batch in parquet.iter_batches(
        batch_size=read_batch_rows,
        columns=["target_poi_id", "train_order_count"],
    ):
        poi_ids = batch.column(0).to_pylist()
        counts = batch.column(1).to_pylist()
        for poi_id, order_count in zip(poi_ids, counts, strict=True):
            stats = poi_stats.get(poi_id)
            if stats is None:
                poi_stats[poi_id] = [order_count, 1]
            else:
                stats[0] += order_count
                stats[1] += 1

    sorted_stats = sorted(poi_stats.items())
    columns: list[list[Any]] = [[], [], []]
    order_count_sum = 0
    unique_query_count_sum = 0
    for poi_id, (order_count, unique_query_count) in sorted_stats:
        columns[0].append(poi_id)
        columns[1].append(order_count)
        columns[2].append(unique_query_count)
        order_count_sum += order_count
        unique_query_count_sum += unique_query_count
    _write_table(output_path, POI_STATS_SCHEMA, columns)
    return len(sorted_stats), order_count_sum, unique_query_count_sum


def _validate_file_group(
    directory: Path,
    files: Any,
    schema: pa.Schema,
) -> int:
    if not isinstance(files, list):
        raise QueryStatsError(f"manifest 文件清单非法：{directory}")
    total_rows = 0
    for expected_partition, file_info in enumerate(files):
        if not isinstance(file_info, Mapping):
            raise QueryStatsError(f"manifest 文件项非法：{directory}")
        if int(file_info.get("partition_id", -1)) != expected_partition:
            raise QueryStatsError(f"manifest 分区顺序非法：{directory}")
        path = directory / str(file_info.get("file", ""))
        if not path.is_file():
            raise QueryStatsError(f"缺少统计分片：{path}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != schema:
            raise QueryStatsError(f"统计分片 Schema 不一致：{path}")
        rows = parquet.metadata.num_rows
        if rows != int(file_info.get("rows", -1)):
            raise QueryStatsError(f"统计分片行数不一致：{path}")
        if path.stat().st_size != int(file_info.get("size_bytes", -1)):
            raise QueryStatsError(f"统计分片大小不一致：{path}")
        if _sha256_file(path) != file_info.get("sha256"):
            raise QueryStatsError(f"统计分片 SHA256 不一致：{path}")
        total_rows += rows
    return total_rows


def _manifest_files(outputs: Mapping[str, Any], name: str) -> Any:
    value = outputs.get(name)
    if not isinstance(value, Mapping):
        raise QueryStatsError(f"Query 统计 manifest 缺少 outputs.{name}")
    return value.get("files")


def validate_train_query_stats(output_dir: Path) -> QueryStatsBuildResult:
    """Validate all completed Query statistics files against their manifest."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = _load_json_object(manifest_path, "Query 统计 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise QueryStatsError("Query 统计 manifest schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise QueryStatsError("Query 统计 manifest 状态不是 completed")
    if not (output_dir / "_SUCCESS").is_file():
        raise QueryStatsError("Query 统计目录缺少 _SUCCESS")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise QueryStatsError("Query 统计 manifest 缺少 outputs")
    catalog_rows = _validate_file_group(
        output_dir / "query_catalog",
        _manifest_files(outputs, "query_catalog"),
        QUERY_CATALOG_SCHEMA,
    )
    pair_rows = _validate_file_group(
        output_dir / "query_poi",
        _manifest_files(outputs, "query_poi"),
        QUERY_POI_SCHEMA,
    )
    poi_rows = _validate_file_group(
        output_dir / "poi_stats",
        _manifest_files(outputs, "poi_stats"),
        POI_STATS_SCHEMA,
    )
    stats = manifest.get("stats")
    if not isinstance(stats, Mapping):
        raise QueryStatsError("Query 统计 manifest 缺少 stats")
    expected = {
        "unique_queries": catalog_rows,
        "unique_query_poi_pairs": pair_rows,
        "covered_pois": poi_rows,
    }
    for field, actual in expected.items():
        if int(stats.get(field, -1)) != actual:
            raise QueryStatsError(f"Query 统计 {field}={actual} 与 manifest 不一致")
    return QueryStatsBuildResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        source_rows=int(stats["source_rows"]),
        unique_queries=catalog_rows,
        unique_query_poi_pairs=pair_rows,
        covered_pois=poi_rows,
        reused=True,
    )


def build_train_query_stats(
    query_shards_dir: Path,
    output_dir: Path,
    *,
    poi_partitions: int = 256,
    poi_buffer_rows: int = 2_048,
    read_batch_rows: int = 65_536,
    show_progress: bool = True,
) -> QueryStatsBuildResult:
    """Build deterministic Query, Query–POI, and POI coverage statistics."""

    if poi_partitions <= 0 or poi_partitions > 65_536:
        raise QueryStatsError("poi_partitions 必须位于 [1, 65536]")
    if read_batch_rows <= 0:
        raise QueryStatsError("read_batch_rows 必须大于 0")
    query_shards_dir = query_shards_dir.resolve()
    output_dir = output_dir.resolve()
    try:
        validate_query_shards(query_shards_dir)
    except QueryShardError as error:
        raise QueryStatsError(str(error)) from error
    source_manifest_path = query_shards_dir / "manifest.json"
    source_manifest_sha256 = _sha256_file(source_manifest_path)
    source_manifest = _load_json_object(source_manifest_path, "Query 分片 manifest")
    sharding = source_manifest.get("sharding")
    if not isinstance(sharding, Mapping):
        raise QueryStatsError("Query 分片 manifest 缺少 sharding")
    source_files = sharding.get("files")
    if not isinstance(source_files, list) or not source_files:
        raise QueryStatsError("Query 分片 manifest 文件清单非法")
    source_rows_expected = int(sharding.get("total_rows", -1))

    if output_dir.exists():
        result = validate_train_query_stats(output_dir)
        existing = _load_json_object(result.manifest_path, "Query 统计 manifest")
        source = existing.get("source")
        config = existing.get("config")
        if not isinstance(source, Mapping) or not isinstance(config, Mapping):
            raise QueryStatsError("已有 Query 统计 manifest 缺少 source/config")
        if source.get("query_shards_manifest_sha256") != source_manifest_sha256:
            raise QueryStatsError("已有 Query 统计与当前 Query 分片指纹不一致")
        expected_config = {
            "poi_partitions": poi_partitions,
            "poi_buffer_rows": poi_buffer_rows,
            "read_batch_rows": read_batch_rows,
        }
        for key, value in expected_config.items():
            if int(config.get(key, -1)) != value:
                raise QueryStatsError(f"已有 Query 统计 config.{key} 与当前请求不一致")
        return result

    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    if staging_dir.exists():
        raise QueryStatsError(
            f"存在未完成的 Query 统计目录，请先审计后处理：{staging_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    query_catalog_dir = staging_dir / "query_catalog"
    query_poi_dir = staging_dir / "query_poi"
    poi_stats_dir = staging_dir / "poi_stats"
    poi_work_dir = staging_dir / ".poi_pair_partitions"
    query_catalog_dir.mkdir()
    query_poi_dir.mkdir()
    poi_stats_dir.mkdir()

    started = time.perf_counter()
    poi_partitioner = _PoiPairPartitioner(
        poi_work_dir,
        num_partitions=poi_partitions,
        buffer_rows_per_partition=poi_buffer_rows,
    )
    source_rows = 0
    unique_queries = 0
    unique_pairs = 0
    query_order_count_sum = 0
    query_catalog_files: list[dict[str, Any]] = []
    query_poi_files: list[dict[str, Any]] = []
    progress = tqdm(
        source_files,
        desc="Train Query statistics",
        unit="shard",
        disable=not show_progress,
    )
    for query_shard_id, source_file in enumerate(progress):
        if not isinstance(source_file, Mapping):
            raise QueryStatsError("Query 分片 manifest 文件项非法")
        source_path = query_shards_dir / str(source_file.get("file", ""))
        catalog_path = query_catalog_dir / _part_name(
            query_shard_id,
            len(source_files),
        )
        query_poi_path = query_poi_dir / _part_name(
            query_shard_id,
            len(source_files),
        )
        shard_rows, shard_queries, shard_pairs, shard_orders = _aggregate_query_shard(
            source_path,
            query_shard_id=query_shard_id,
            first_query_id=unique_queries,
            read_batch_rows=read_batch_rows,
            query_catalog_path=catalog_path,
            query_poi_path=query_poi_path,
            poi_partitioner=poi_partitioner,
        )
        if shard_rows != int(source_file.get("rows", -1)):
            raise QueryStatsError(
                f"Query 分片 {query_shard_id} 聚合行数与 manifest 不一致"
            )
        source_rows += shard_rows
        unique_queries += shard_queries
        unique_pairs += shard_pairs
        query_order_count_sum += shard_orders
        query_catalog_files.append(_file_info(catalog_path, query_shard_id))
        query_poi_files.append(_file_info(query_poi_path, query_shard_id))
    poi_partitioner.close()

    covered_pois = 0
    poi_order_count_sum = 0
    poi_unique_query_count_sum = 0
    poi_stats_files: list[dict[str, Any]] = []
    poi_progress = tqdm(
        range(poi_partitions),
        desc="POI coverage statistics",
        unit="partition",
        disable=not show_progress,
    )
    for partition_id in poi_progress:
        work_path = poi_work_dir / _part_name(partition_id, poi_partitions)
        output_path = poi_stats_dir / _part_name(partition_id, poi_partitions)
        partition_pois, partition_orders, partition_queries = _aggregate_poi_partition(
            work_path,
            output_path,
            read_batch_rows=read_batch_rows,
        )
        covered_pois += partition_pois
        poi_order_count_sum += partition_orders
        poi_unique_query_count_sum += partition_queries
        poi_stats_files.append(_file_info(output_path, partition_id))
    shutil.rmtree(poi_work_dir)

    if source_rows != source_rows_expected:
        raise QueryStatsError(
            f"聚合源行数 {source_rows:,} != Query 分片 {source_rows_expected:,}"
        )
    query_df_sum = sum(
        int(
            pq.read_table(
                query_catalog_dir / file_info["file"],
                columns=["poi_df"],
            )["poi_df"]
            .to_numpy()
            .sum()
        )
        for file_info in query_catalog_files
    )
    conservation = {
        "source_rows_equal_query_order_count": (source_rows == query_order_count_sum),
        "source_rows_equal_poi_order_count": source_rows == poi_order_count_sum,
        "query_poi_pairs_equal_query_df_sum": (unique_pairs == query_df_sum),
        "query_poi_pairs_equal_poi_unique_query_sum": (
            unique_pairs == poi_unique_query_count_sum
        ),
    }
    if not all(conservation.values()):
        raise QueryStatsError(f"Query 统计守恒校验失败：{conservation}")

    source = source_manifest.get("source")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "built_at": _utc_now(),
        "source": {
            "query_shards_dir": str(query_shards_dir),
            "query_shards_manifest": str(source_manifest_path),
            "query_shards_manifest_sha256": source_manifest_sha256,
            "train_file": source.get("train_file")
            if isinstance(source, Mapping)
            else None,
            "train_sha256": source.get("train_sha256")
            if isinstance(source, Mapping)
            else None,
        },
        "config": {
            "query_shards": len(source_files),
            "poi_partitions": poi_partitions,
            "poi_buffer_rows": poi_buffer_rows,
            "read_batch_rows": read_batch_rows,
            "compression": "zstd",
        },
        "contract": {
            "query_normalization": "none",
            "query_id_dtype": "int64",
            "query_id_range": [0, unique_queries],
            "query_id_end_exclusive": True,
            "query_id_order": (
                "query shard ascending, then exact raw Query Python lexicographic order"
            ),
            "e1_weight": "one per distinct Query-POI pair",
            "e2_fields": ["train_order_count", "poi_df"],
            "statistics_split": "train only",
        },
        "stats": {
            "source_rows": source_rows,
            "unique_queries": unique_queries,
            "unique_query_poi_pairs": unique_pairs,
            "covered_pois": covered_pois,
            "query_order_count_sum": query_order_count_sum,
            "query_df_sum": query_df_sum,
            "poi_order_count_sum": poi_order_count_sum,
            "poi_unique_query_count_sum": poi_unique_query_count_sum,
        },
        "outputs": {
            "query_catalog": {
                "dir": str(query_catalog_dir.relative_to(staging_dir)),
                "rows": unique_queries,
                "files": query_catalog_files,
            },
            "query_poi": {
                "dir": str(query_poi_dir.relative_to(staging_dir)),
                "rows": unique_pairs,
                "files": query_poi_files,
            },
            "poi_stats": {
                "dir": str(poi_stats_dir.relative_to(staging_dir)),
                "rows": covered_pois,
                "files": poi_stats_files,
            },
        },
        "validation": {
            **conservation,
            "same_raw_query_colocated_by_source_hash": True,
            "query_ids_contiguous": True,
            "query_poi_pairs_unique": True,
            "poi_stats_partitioned_by_stable_hash": True,
            "temporary_poi_pair_partitions_removed": True,
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    _write_json_atomic(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    os.replace(staging_dir, output_dir)
    return QueryStatsBuildResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        source_rows=source_rows,
        unique_queries=unique_queries,
        unique_query_poi_pairs=unique_pairs,
        covered_pois=covered_pois,
        reused=False,
    )
