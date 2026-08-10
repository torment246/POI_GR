"""Build sparse Train Query aggregates aligned to the frozen POI catalog."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from .query_stats import QUERY_POI_SCHEMA, stable_poi_partition, validate_train_query_stats


SCHEMA_VERSION = "query-poi-aggregates-v1"
OUTPUT_FILES = {
    "covered_poi_rows": ("covered_poi_rows.npy", "int64"),
    "e1_query_mean": ("e1_query_mean.npy", "float16"),
    "e2_query_weighted": ("e2_query_weighted.npy", "float16"),
    "unique_query_count": ("unique_query_count.npy", "int32"),
    "train_order_count": ("train_order_count.npy", "int64"),
    "e2_weight_sum": ("e2_weight_sum.npy", "float32"),
}


class QueryAugmentationError(RuntimeError):
    """Raised when Query-to-POI aggregation is invalid or incomplete."""


@dataclass(frozen=True)
class QueryAugmentationConfig:
    job_name: str
    query_stats_dir: Path
    expected_stats_manifest_sha256: str
    query_embeddings_dir: Path
    expected_query_embeddings_sha256: str
    poi_embedding_dir: Path
    expected_poi_ids_sha256: str
    expected_poi_rows: int
    expected_covered_pois: int
    output_dir: Path
    read_batch_rows: int
    resume: bool


@dataclass(frozen=True)
class QueryAugmentationResult:
    output_dir: Path
    manifest_path: Path
    covered_pois: int
    embedding_dim: int
    partitions: int
    reused: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise QueryAugmentationError(f"{name} 不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise QueryAugmentationError(f"{name} JSON 非法：{path}") from error
    if not isinstance(value, dict):
        raise QueryAugmentationError(f"{name} 必须是 JSON object：{path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueryAugmentationError(f"{name} 必须是 YAML mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QueryAugmentationError(f"{name} 必须是正整数")
    return value


def _path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QueryAugmentationError(f"{name} 必须是非空路径")
    result = Path(value)
    return result if result.is_absolute() else project_root / result


def _sha_setting(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise QueryAugmentationError(f"{name} 必须是 64 位 SHA256")
    return value


def load_query_augmentation_config(
    config_path: Path,
    project_root: Path,
) -> QueryAugmentationConfig:
    """Load the frozen E1/E2 sparse aggregation configuration."""

    with config_path.open("r", encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "配置根节点")
    input_raw = _mapping(raw.get("input"), "input")
    output_raw = _mapping(raw.get("output"), "output")
    runtime_raw = _mapping(raw.get("runtime"), "runtime")
    job_name = raw.get("job_name")
    if not isinstance(job_name, str) or not job_name.strip():
        raise QueryAugmentationError("job_name 必须是非空字符串")
    return QueryAugmentationConfig(
        job_name=job_name,
        query_stats_dir=_path(
            input_raw.get("query_stats_dir"), project_root, "input.query_stats_dir"
        ),
        expected_stats_manifest_sha256=_sha_setting(
            input_raw.get("expected_stats_manifest_sha256"),
            "input.expected_stats_manifest_sha256",
        ),
        query_embeddings_dir=_path(
            input_raw.get("query_embeddings_dir"),
            project_root,
            "input.query_embeddings_dir",
        ),
        expected_query_embeddings_sha256=_sha_setting(
            input_raw.get("expected_query_embeddings_sha256"),
            "input.expected_query_embeddings_sha256",
        ),
        poi_embedding_dir=_path(
            input_raw.get("poi_embedding_dir"),
            project_root,
            "input.poi_embedding_dir",
        ),
        expected_poi_ids_sha256=_sha_setting(
            input_raw.get("expected_poi_ids_sha256"),
            "input.expected_poi_ids_sha256",
        ),
        expected_poi_rows=_positive_int(
            input_raw.get("expected_poi_rows"), "input.expected_poi_rows"
        ),
        expected_covered_pois=_positive_int(
            input_raw.get("expected_covered_pois"),
            "input.expected_covered_pois",
        ),
        output_dir=_path(output_raw.get("dir"), project_root, "output.dir"),
        read_batch_rows=_positive_int(
            runtime_raw.get("read_batch_rows"), "runtime.read_batch_rows"
        ),
        resume=bool(output_raw.get("resume", True)),
    )


def override_query_augmentation_output(
    config: QueryAugmentationConfig,
    output_dir: Path | None,
) -> QueryAugmentationConfig:
    """Return a config with an optional independent output directory."""

    if output_dir is None:
        return config
    return QueryAugmentationConfig(
        **{**config.__dict__, "output_dir": output_dir}
    )


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _load_query_df(
    stats_dir: Path,
    files: list[Mapping[str, Any]],
    total_queries: int,
    read_batch_rows: int,
) -> np.ndarray:
    query_df = np.empty(total_queries, dtype=np.int32)
    expected_query_id = 0
    for file_info in files:
        path = stats_dir / "query_catalog" / str(file_info.get("file", ""))
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_rows,
            columns=["query_id", "poi_df"],
        ):
            query_ids = batch.column(0).to_numpy(zero_copy_only=False)
            values = batch.column(1).to_numpy(zero_copy_only=False)
            stop = expected_query_id + len(query_ids)
            expected = np.arange(expected_query_id, stop, dtype=np.int64)
            if not np.array_equal(query_ids, expected):
                raise QueryAugmentationError("Query 目录 query_id 不连续")
            query_df[expected_query_id:stop] = values
            expected_query_id = stop
    if expected_query_id != total_queries or np.any(query_df <= 0):
        raise QueryAugmentationError("Query DF 行数或取值非法")
    return query_df


def _table_from_pair_columns(columns: tuple[list[int], list[str], list[int]]) -> pa.Table:
    arrays = [
        pa.array(values, type=field.type)
        for values, field in zip(columns, QUERY_POI_SCHEMA, strict=True)
    ]
    return pa.Table.from_arrays(arrays, schema=QUERY_POI_SCHEMA)


class _PairPartitionWriter:
    def __init__(self, output_dir: Path, partitions: int, buffer_rows: int) -> None:
        self.output_dir = output_dir
        self.partitions = partitions
        self.buffer_rows = buffer_rows
        self.buffers: list[tuple[list[int], list[str], list[int]]] = [
            ([], [], []) for _ in range(partitions)
        ]
        self.writers: list[pq.ParquetWriter | None] = [None] * partitions
        self.rows = [0] * partitions

    def _path(self, partition_id: int) -> Path:
        return self.output_dir / (
            f"part-{partition_id:05d}-of-{self.partitions:05d}.parquet"
        )

    def add(self, query_id: int, poi_id: str, count: int) -> None:
        partition_id = stable_poi_partition(poi_id, self.partitions)
        columns = self.buffers[partition_id]
        columns[0].append(query_id)
        columns[1].append(poi_id)
        columns[2].append(count)
        self.rows[partition_id] += 1
        if len(columns[0]) >= self.buffer_rows:
            self.flush(partition_id)

    def flush(self, partition_id: int) -> None:
        columns = self.buffers[partition_id]
        if not columns[0]:
            return
        writer = self.writers[partition_id]
        if writer is None:
            writer = pq.ParquetWriter(
                self._path(partition_id),
                QUERY_POI_SCHEMA,
                compression="zstd",
                use_dictionary=True,
            )
            self.writers[partition_id] = writer
        writer.write_table(_table_from_pair_columns(columns))
        for values in columns:
            values.clear()

    def close(self) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for partition_id in range(self.partitions):
            self.flush(partition_id)
            writer = self.writers[partition_id]
            if writer is None:
                writer = pq.ParquetWriter(
                    self._path(partition_id),
                    QUERY_POI_SCHEMA,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.close()
            path = self._path(partition_id)
            files.append(
                {
                    "partition_id": partition_id,
                    "file": path.name,
                    "rows": self.rows[partition_id],
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        return files


def _partition_pairs_by_poi(
    stats_dir: Path,
    source_files: list[Mapping[str, Any]],
    output_dir: Path,
    *,
    partitions: int,
    expected_rows: int,
    stats_manifest_sha256: str,
    read_batch_rows: int,
    show_progress: bool,
) -> tuple[list[Mapping[str, Any]], str]:
    manifest_path = output_dir / "manifest.json"
    if output_dir.is_dir():
        manifest = _load_json(manifest_path, "POI 重分片 manifest")
        if manifest.get("status") != "completed" or not (
            output_dir / "_SUCCESS"
        ).is_file():
            raise QueryAugmentationError("已有 POI 重分片目录未完成")
        if manifest.get("query_stats_manifest_sha256") != stats_manifest_sha256:
            raise QueryAugmentationError("已有 POI 重分片与 Query 统计指纹不一致")
        if int(manifest.get("partitions", -1)) != partitions:
            raise QueryAugmentationError("已有 POI 重分片数量不一致")
        files = manifest.get("files")
        if not isinstance(files, list) or sum(
            int(item.get("rows", -1)) for item in files if isinstance(item, Mapping)
        ) != expected_rows:
            raise QueryAugmentationError("已有 POI 重分片行数不一致")
        return files, _sha256_file(manifest_path)

    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    if staging_dir.exists():
        raise QueryAugmentationError(f"存在未完成 POI 重分片目录：{staging_dir}")
    staging_dir.mkdir()
    writer = _PairPartitionWriter(
        staging_dir,
        partitions,
        buffer_rows=max(256, min(read_batch_rows, 2_048)),
    )
    source_rows = 0
    progress = tqdm(
        source_files,
        desc="Query-POI repartition",
        unit="query-shard",
        disable=not show_progress,
    )
    try:
        for file_info in progress:
            path = stats_dir / "query_poi" / str(file_info.get("file", ""))
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=read_batch_rows,
                columns=["query_id", "target_poi_id", "train_order_count"],
            ):
                query_ids = batch.column(0).to_pylist()
                poi_ids = batch.column(1).to_pylist()
                counts = batch.column(2).to_pylist()
                for query_id, poi_id, count in zip(
                    query_ids, poi_ids, counts, strict=True
                ):
                    writer.add(int(query_id), str(poi_id), int(count))
                source_rows += len(batch)
        files = writer.close()
    finally:
        progress.close()
    if source_rows != expected_rows or sum(item["rows"] for item in files) != expected_rows:
        raise QueryAugmentationError("POI 重分片前后行数不守恒")
    manifest = {
        "schema_version": "query-poi-by-poi-v1",
        "status": "completed",
        "built_at": _utc_now(),
        "query_stats_manifest_sha256": stats_manifest_sha256,
        "source_rows": source_rows,
        "partitions": partitions,
        "partition_rule": "stable_poi_partition from train-query-stats-v1",
        "files": files,
    }
    _write_json_atomic(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    os.replace(staging_dir, output_dir)
    return files, _sha256_file(output_dir / "manifest.json")


def _load_numeric_poi_index(
    poi_ids_path: Path,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_ids = np.empty(expected_rows, dtype=np.int64)
    row_count = 0
    with poi_ids_path.open("r", encoding="utf-8") as handle:
        for row_count, line in enumerate(handle, start=1):
            if row_count > expected_rows:
                raise QueryAugmentationError("POI ID 行数超过配置")
            try:
                value = json.loads(line)
                numeric_ids[row_count - 1] = int(value)
            except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as error:
                raise QueryAugmentationError(
                    f"POI ID 第 {row_count} 行不是 int64 可表示的数字标识"
                ) from error
    if row_count != expected_rows:
        raise QueryAugmentationError(
            f"POI ID 行数 {row_count} != 配置 {expected_rows}"
        )
    order = np.argsort(numeric_ids, kind="stable")
    sorted_ids = numeric_ids[order]
    if len(sorted_ids) > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise QueryAugmentationError("POI ID 不唯一")
    return sorted_ids, order.astype(np.int64, copy=False)


def _map_poi_ids(
    poi_ids: list[str],
    sorted_ids: np.ndarray,
    sorted_rows: np.ndarray,
) -> np.ndarray:
    try:
        numeric = np.fromiter((int(value) for value in poi_ids), dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise QueryAugmentationError("Query–POI 统计包含非数字 POI ID") from error
    positions = np.searchsorted(sorted_ids, numeric)
    if np.any(positions >= len(sorted_ids)):
        raise QueryAugmentationError("Query–POI 统计包含目录外 POI ID")
    if not np.array_equal(sorted_ids[positions], numeric):
        raise QueryAugmentationError("Query–POI 统计包含目录外 POI ID")
    return sorted_rows[positions]


def _normalize_rows(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise QueryAugmentationError(f"{name} 聚合向量存在零范数或非有限值")
    values /= norms[:, None]
    return values


def aggregate_query_poi_partition(
    query_ids: np.ndarray,
    poi_rows: np.ndarray,
    order_counts: np.ndarray,
    query_df: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    covered_poi_count: int,
) -> dict[str, np.ndarray]:
    """Aggregate one bounded POI-hash partition for E1 and E2."""

    if not (len(query_ids) == len(poi_rows) == len(order_counts)):
        raise QueryAugmentationError("Query–POI 分片列长度非法")
    if not len(query_ids):
        embedding_dim = int(query_embeddings.shape[1])
        return {
            "covered_poi_rows": np.empty(0, dtype=np.int64),
            "e1_query_mean": np.empty((0, embedding_dim), dtype=np.float16),
            "e2_query_weighted": np.empty((0, embedding_dim), dtype=np.float16),
            "unique_query_count": np.empty(0, dtype=np.int32),
            "train_order_count": np.empty(0, dtype=np.int64),
            "e2_weight_sum": np.empty(0, dtype=np.float32),
        }
    if np.any(query_ids < 0) or np.any(query_ids >= len(query_df)):
        raise QueryAugmentationError("Query–POI 分片包含越界 query_id")
    if np.any(order_counts <= 0):
        raise QueryAugmentationError("Query–POI count 必须大于 0")
    order = np.lexsort((query_ids, poi_rows))
    sorted_query_ids = query_ids[order]
    sorted_poi_rows = poi_rows[order]
    sorted_counts = order_counts[order]
    starts = np.r_[0, np.flatnonzero(sorted_poi_rows[1:] != sorted_poi_rows[:-1]) + 1]
    rows = sorted_poi_rows[starts]
    unique_query_count = np.diff(np.r_[starts, len(sorted_poi_rows)]).astype(np.int32)
    pair_vectors = np.asarray(query_embeddings[sorted_query_ids], dtype=np.float32)
    if not np.isfinite(pair_vectors).all():
        raise QueryAugmentationError("输入 Query embedding 存在 NaN/Inf")

    e1 = np.add.reduceat(pair_vectors, starts, axis=0)
    _normalize_rows(e1, "E1")
    idf = np.log((covered_poi_count + 1.0) / (query_df[sorted_query_ids] + 1.0))
    weights = np.log1p(sorted_counts.astype(np.float64)) * idf
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise QueryAugmentationError("E2 权重存在非正数或非有限值")
    weight_sums = np.add.reduceat(weights, starts)
    e2 = np.add.reduceat(pair_vectors * weights[:, None], starts, axis=0)
    _normalize_rows(e2, "E2")
    train_order_count = np.add.reduceat(sorted_counts, starts)
    return {
        "covered_poi_rows": rows.astype(np.int64, copy=False),
        "e1_query_mean": e1.astype(np.float16),
        "e2_query_weighted": e2.astype(np.float16),
        "unique_query_count": unique_query_count,
        "train_order_count": train_order_count.astype(np.int64, copy=False),
        "e2_weight_sum": weight_sums.astype(np.float32),
    }


def _output_shapes(covered_pois: int, embedding_dim: int) -> dict[str, tuple[int, ...]]:
    return {
        "covered_poi_rows": (covered_pois,),
        "e1_query_mean": (covered_pois, embedding_dim),
        "e2_query_weighted": (covered_pois, embedding_dim),
        "unique_query_count": (covered_pois,),
        "train_order_count": (covered_pois,),
        "e2_weight_sum": (covered_pois,),
    }


def _open_outputs(
    output_dir: Path,
    shapes: Mapping[str, tuple[int, ...]],
    *,
    resume: bool,
) -> dict[str, np.memmap]:
    values: dict[str, np.memmap] = {}
    for name, (filename, dtype) in OUTPUT_FILES.items():
        path = output_dir / filename
        if resume:
            array = np.load(path, mmap_mode="r+")
            if array.shape != shapes[name] or str(array.dtype) != dtype:
                raise QueryAugmentationError(f"断点数组 {filename} shape/dtype 不一致")
        else:
            array = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.dtype(dtype),
                shape=shapes[name],
            )
        values[name] = array
    return values


def _signature(
    config: QueryAugmentationConfig,
    *,
    selected_partitions: int,
    covered_pois: int,
    embedding_dim: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stats_manifest_sha256": config.expected_stats_manifest_sha256,
        "query_embeddings_sha256": config.expected_query_embeddings_sha256,
        "poi_ids_sha256": config.expected_poi_ids_sha256,
        "poi_rows": config.expected_poi_rows,
        "selected_partitions": selected_partitions,
        "covered_pois": covered_pois,
        "embedding_dim": embedding_dim,
        "e1_weight": "1 per distinct Query-POI pair",
        "e2_weight": "log1p(count_qi) * log((N+1)/(df_q+1))",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _array_validation(path: Path, sample_rows: int = 2_048) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r")
    all_finite = True
    for start in range(0, len(values), 8_192):
        if not np.isfinite(np.asarray(values[start : start + 8_192])).all():
            all_finite = False
            break
    if not all_finite:
        raise QueryAugmentationError(f"{path.name} 存在 NaN/Inf")
    result: dict[str, Any] = {"all_finite": True}
    if values.ndim == 2:
        indices = np.linspace(
            0, len(values) - 1, min(sample_rows, len(values)), dtype=np.int64
        )
        norms = np.linalg.norm(np.asarray(values[indices], dtype=np.float32), axis=1)
        result["l2_norm"] = {
            "sample_rows": int(len(indices)),
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        }
    return result


def validate_query_poi_aggregates(
    output_dir: Path,
    *,
    expected_stats_manifest_sha256: str | None = None,
) -> QueryAugmentationResult:
    """Fully validate hashes, shapes, row uniqueness, and vector finiteness."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "E1/E2 聚合 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise QueryAugmentationError("E1/E2 聚合 manifest schema 不兼容")
    if manifest.get("status") != "completed" or not (output_dir / "_SUCCESS").is_file():
        raise QueryAugmentationError("E1/E2 聚合产物未完成")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise QueryAugmentationError("E1/E2 聚合 manifest 缺少 inputs/outputs")
    if (
        expected_stats_manifest_sha256 is not None
        and inputs.get("query_stats_manifest_sha256")
        != expected_stats_manifest_sha256
    ):
        raise QueryAugmentationError("E1/E2 聚合统计输入不是配置冻结版本")
    covered_pois = int(manifest.get("aggregation", {}).get("covered_pois", -1))
    embedding_dim = int(manifest.get("aggregation", {}).get("embedding_dim", -1))
    shapes = _output_shapes(covered_pois, embedding_dim)
    for name, (filename, dtype) in OUTPUT_FILES.items():
        spec = outputs.get(name)
        if not isinstance(spec, Mapping):
            raise QueryAugmentationError(f"manifest 缺少输出 {name}")
        path = output_dir / filename
        values = np.load(path, mmap_mode="r")
        if values.shape != shapes[name] or str(values.dtype) != dtype:
            raise QueryAugmentationError(f"{filename} shape/dtype 与 manifest 不一致")
        if _sha256_file(path) != spec.get("sha256"):
            raise QueryAugmentationError(f"{filename} SHA256 不一致")
    rows = np.load(output_dir / OUTPUT_FILES["covered_poi_rows"][0], mmap_mode="r")
    if np.any(rows < 0) or len(np.unique(rows)) != covered_pois:
        raise QueryAugmentationError("covered_poi_rows 越界或不唯一")
    for name in ("e1_query_mean", "e2_query_weighted", "e2_weight_sum"):
        _array_validation(output_dir / OUTPUT_FILES[name][0])
    progress = _load_json(output_dir / "progress.json", "E1/E2 聚合进度")
    partitions = int(manifest.get("aggregation", {}).get("partitions", -1))
    query_shards = int(manifest.get("aggregation", {}).get("query_shards", -1))
    if (
        progress.get("status") != "completed"
        or int(progress.get("next_query_shard", -1)) != query_shards
        or int(progress.get("next_normalization_row", -1)) != covered_pois
    ):
        raise QueryAugmentationError("E1/E2 聚合进度与 manifest 不一致")
    return QueryAugmentationResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        covered_pois=covered_pois,
        embedding_dim=embedding_dim,
        partitions=partitions,
        reused=True,
    )


def run_query_poi_aggregation(
    config: QueryAugmentationConfig,
    *,
    project_root: Path,
    max_partitions: int | None = None,
    show_progress: bool = True,
) -> QueryAugmentationResult:
    """Build sparse E1/E2 aggregates in RAM and atomically publish final NPYs."""

    started = time.perf_counter()
    stats = validate_train_query_stats(config.query_stats_dir)
    stats_manifest_path = stats.manifest_path
    stats_manifest_sha256 = _sha256_file(stats_manifest_path)
    if stats_manifest_sha256 != config.expected_stats_manifest_sha256:
        raise QueryAugmentationError("Query 统计 manifest 不是配置冻结版本")
    stats_manifest = _load_json(stats_manifest_path, "Query 统计 manifest")
    stats_outputs = stats_manifest.get("outputs")
    if not isinstance(stats_outputs, Mapping):
        raise QueryAugmentationError("Query 统计 manifest 缺少 outputs")
    query_catalog_spec = stats_outputs.get("query_catalog")
    query_poi_spec = stats_outputs.get("query_poi")
    poi_stats_spec = stats_outputs.get("poi_stats")
    if not all(
        isinstance(value, Mapping)
        for value in (query_catalog_spec, query_poi_spec, poi_stats_spec)
    ):
        raise QueryAugmentationError("Query 统计 manifest 输出结构非法")
    query_catalog_files = list(query_catalog_spec.get("files", []))
    query_poi_files = list(query_poi_spec.get("files", []))
    poi_stats_files = list(poi_stats_spec.get("files", []))
    if not query_poi_files or not poi_stats_files:
        raise QueryAugmentationError("Query–POI 或 POI 统计分片为空")
    total_partitions = len(poi_stats_files)
    if max_partitions is not None and max_partitions <= 0:
        raise QueryAugmentationError("max_partitions 必须大于 0")
    selected_partitions = (
        total_partitions
        if max_partitions is None
        else min(max_partitions, total_partitions)
    )
    covered_pois = sum(
        int(file_info.get("rows", -1))
        for file_info in poi_stats_files[:selected_partitions]
    )
    if max_partitions is None and covered_pois != config.expected_covered_pois:
        raise QueryAugmentationError(
            f"覆盖 POI {covered_pois} != 配置 {config.expected_covered_pois}"
        )

    query_manifest_path = config.query_embeddings_dir / "manifest.json"
    query_manifest = _load_json(query_manifest_path, "Train Query embedding manifest")
    query_output = query_manifest.get("output")
    if query_manifest.get("status") != "completed" or not isinstance(
        query_output, Mapping
    ):
        raise QueryAugmentationError("Train Query embedding 产物未完成")
    query_embeddings_path = config.query_embeddings_dir / str(
        query_output.get("embeddings", "")
    )
    if _sha256_file(query_embeddings_path) != config.expected_query_embeddings_sha256:
        raise QueryAugmentationError("Train Query embedding SHA256 不是配置冻结版本")
    query_embeddings = np.load(query_embeddings_path, mmap_mode="r")
    total_queries = int(query_catalog_spec.get("rows", -1))
    if query_embeddings.shape[0] != total_queries or query_embeddings.ndim != 2:
        raise QueryAugmentationError("Train Query embedding shape 与统计不一致")
    embedding_dim = int(query_embeddings.shape[1])

    poi_manifest_path = config.poi_embedding_dir / "manifest.json"
    poi_ids_path = config.poi_embedding_dir / "poi_ids.jsonl"
    poi_manifest = _load_json(poi_manifest_path, "POI embedding manifest")
    if poi_manifest.get("output", {}).get("shape") != [
        config.expected_poi_rows,
        embedding_dim,
    ]:
        raise QueryAugmentationError("POI embedding shape 与 Query embedding 不一致")
    if _sha256_file(poi_ids_path) != config.expected_poi_ids_sha256:
        raise QueryAugmentationError("POI ID SHA256 不是配置冻结版本")

    output_dir = config.output_dir.resolve()
    if (output_dir / "_SUCCESS").is_file():
        return validate_query_poi_aggregates(
            output_dir,
            expected_stats_manifest_sha256=config.expected_stats_manifest_sha256,
        )
    if output_dir.exists():
        raise QueryAugmentationError(f"正式输出目录已存在但未完成：{output_dir}")
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    if staging_dir.exists():
        raise QueryAugmentationError(f"存在未完成 staging 目录：{staging_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    manifest_path = staging_dir / "manifest.json"
    progress_path = staging_dir / "progress.json"
    signature = _signature(
        config,
        selected_partitions=selected_partitions,
        covered_pois=covered_pois,
        embedding_dim=embedding_dim,
    )
    accumulator_bytes = (
        covered_pois * embedding_dim * np.dtype(np.float32).itemsize * 2
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_name": config.job_name,
        "status": "running",
        "started_at": _utc_now(),
        "signature": signature,
        "inputs": {
            "query_stats_dir": str(config.query_stats_dir.resolve()),
            "query_stats_manifest": str(stats_manifest_path.resolve()),
            "query_stats_manifest_sha256": stats_manifest_sha256,
            "query_embeddings": str(query_embeddings_path.resolve()),
            "query_embeddings_sha256": config.expected_query_embeddings_sha256,
            "poi_embedding_dir": str(config.poi_embedding_dir.resolve()),
            "poi_manifest_sha256": _sha256_file(poi_manifest_path),
            "poi_ids": str(poi_ids_path.resolve()),
            "poi_ids_sha256": config.expected_poi_ids_sha256,
        },
        "aggregation": {
            "partitions": selected_partitions,
            "full_partitions": total_partitions,
            "query_shards": len(query_poi_files),
            "covered_pois": covered_pois,
            "full_covered_pois": config.expected_covered_pois,
            "unique_queries": total_queries,
            "embedding_dim": embedding_dim,
            "e1_weight": "1 per distinct Query-POI pair",
            "e2_weight": "log1p(count_qi) * log((N+1)/(df_q+1))",
            "e2_N": config.expected_covered_pois,
            "query_aggregate_normalization": "L2 after weighted sum",
            "output_row_order": (
                "selected POI hash partitions ascending, then POI catalog row ascending"
            ),
            "accumulation_order": (
                "Query shard ascending for sequential Query embedding reads"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "read_batch_rows": config.read_batch_rows,
            "accumulator_storage": "RAM float32; final artifacts only on project disk",
            "accumulation_memory_estimated_gib": accumulator_bytes / 1024**3,
            "resumable": False,
        },
        "git": _git_state(project_root),
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        progress_path,
        {
            "signature": signature,
            "status": "running",
            "stage": "initializing",
            "next_query_shard": 0,
            "next_normalization_row": 0,
            "recoverable": False,
            "updated_at": _utc_now(),
        },
    )

    offsets = np.zeros(selected_partitions + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        [int(info.get("rows", -1)) for info in poi_stats_files[:selected_partitions]],
        dtype=np.int64,
    )
    shapes = _output_shapes(covered_pois, embedding_dim)
    covered_rows = np.empty(covered_pois, dtype=np.int64)
    unique_query_count = np.empty(covered_pois, dtype=np.int32)
    train_order_count = np.empty(covered_pois, dtype=np.int64)
    weight_accumulator = np.zeros(covered_pois, dtype=np.float32)
    try:
        query_df = _load_query_df(
            config.query_stats_dir,
            query_catalog_files,
            total_queries,
            config.read_batch_rows,
        )
        sorted_poi_ids, sorted_poi_rows = _load_numeric_poi_index(
            poi_ids_path, config.expected_poi_rows
        )
        for partition_id in range(selected_partitions):
            stats_info = poi_stats_files[partition_id]
            stats_path = config.query_stats_dir / "poi_stats" / str(
                stats_info.get("file", "")
            )
            table = pq.read_table(
                stats_path,
                columns=[
                    "target_poi_id",
                    "train_order_count",
                    "unique_query_count",
                ],
            )
            poi_rows = _map_poi_ids(
                table.column("target_poi_id").to_pylist(),
                sorted_poi_ids,
                sorted_poi_rows,
            )
            order = np.argsort(poi_rows, kind="stable")
            start = int(offsets[partition_id])
            stop = int(offsets[partition_id + 1])
            if len(order) != stop - start:
                raise QueryAugmentationError("POI 统计分片行数与 manifest 不一致")
            covered_rows[start:stop] = poi_rows[order]
            train_order_count[start:stop] = table.column(
                "train_order_count"
            ).to_numpy(zero_copy_only=False)[order]
            unique_query_count[start:stop] = table.column(
                "unique_query_count"
            ).to_numpy(zero_copy_only=False)[order]
        if np.any(covered_rows < 0) or np.any(
            covered_rows >= config.expected_poi_rows
        ):
            raise QueryAugmentationError("covered_poi_rows 存在越界值")
        if len(np.unique(covered_rows)) != covered_pois:
            raise QueryAugmentationError("covered_poi_rows 不唯一")

        position_by_poi_row = np.full(config.expected_poi_rows, -1, dtype=np.int32)
        position_by_poi_row[covered_rows] = np.arange(covered_pois, dtype=np.int32)
        e1_accumulator = np.zeros((covered_pois, embedding_dim), dtype=np.float32)
        e2_accumulator = np.zeros((covered_pois, embedding_dim), dtype=np.float32)
        progress = tqdm(
            total=len(query_poi_files),
            desc="E1/E2 RAM accumulation",
            unit="query-shard",
            disable=not show_progress,
        )
        for query_shard_id, pair_info in enumerate(query_poi_files):
            pair_path = config.query_stats_dir / "query_poi" / str(
                pair_info.get("file", "")
            )
            table = pq.read_table(
                pair_path,
                columns=["query_id", "target_poi_id", "train_order_count"],
            )
            query_ids = table.column("query_id").to_numpy(zero_copy_only=False)
            poi_rows = _map_poi_ids(
                table.column("target_poi_id").to_pylist(),
                sorted_poi_ids,
                sorted_poi_rows,
            )
            counts = table.column("train_order_count").to_numpy(zero_copy_only=False)
            positions = position_by_poi_row[poi_rows]
            selected = positions >= 0
            if np.any(selected):
                selected_query_ids = query_ids[selected]
                selected_positions = positions[selected]
                selected_counts = counts[selected]
                pair_vectors = np.asarray(
                    query_embeddings[selected_query_ids], dtype=np.float32
                )
                weights = np.log1p(selected_counts.astype(np.float64)) * np.log(
                    (config.expected_covered_pois + 1.0)
                    / (query_df[selected_query_ids] + 1.0)
                )
                if not np.isfinite(weights).all() or np.any(weights <= 0):
                    raise QueryAugmentationError("E2 权重存在非正数或非有限值")
                order = np.argsort(selected_positions, kind="stable")
                sorted_positions = selected_positions[order]
                starts = np.r_[
                    0,
                    np.flatnonzero(
                        sorted_positions[1:] != sorted_positions[:-1]
                    )
                    + 1,
                ]
                unique_positions = sorted_positions[starts]
                sorted_vectors = pair_vectors[order]
                e1_accumulator[unique_positions] += np.add.reduceat(
                    sorted_vectors, starts, axis=0
                )
                sorted_weights = weights[order]
                e2_accumulator[unique_positions] += np.add.reduceat(
                    sorted_vectors * sorted_weights[:, None], starts, axis=0
                )
                weight_accumulator[unique_positions] += np.add.reduceat(
                    sorted_weights, starts
                )
            completed = query_shard_id + 1
            if completed % 16 == 0 or completed == len(query_poi_files):
                _write_json_atomic(
                    progress_path,
                    {
                        "signature": signature,
                        "status": "running",
                        "stage": "accumulation",
                        "next_query_shard": completed,
                        "next_normalization_row": 0,
                        "recoverable": False,
                        "updated_at": _utc_now(),
                    },
                )
            progress.update(1)
        progress.close()

        if max_partitions is None:
            if int(train_order_count.sum()) != stats.source_rows:
                raise QueryAugmentationError("POI Train 订单数不守恒")
            if int(unique_query_count.sum()) != stats.unique_query_poi_pairs:
                raise QueryAugmentationError("POI 唯一 Query 数不守恒")
        if np.any(weight_accumulator <= 0):
            raise QueryAugmentationError("存在 E2 权重和非正的覆盖 POI")

        np.save(staging_dir / OUTPUT_FILES["covered_poi_rows"][0], covered_rows)
        np.save(
            staging_dir / OUTPUT_FILES["unique_query_count"][0], unique_query_count
        )
        np.save(staging_dir / OUTPUT_FILES["train_order_count"][0], train_order_count)
        np.save(staging_dir / OUTPUT_FILES["e2_weight_sum"][0], weight_accumulator)
        e1_output = np.lib.format.open_memmap(
            staging_dir / OUTPUT_FILES["e1_query_mean"][0],
            mode="w+",
            dtype=np.float16,
            shape=shapes["e1_query_mean"],
        )
        e2_output = np.lib.format.open_memmap(
            staging_dir / OUTPUT_FILES["e2_query_weighted"][0],
            mode="w+",
            dtype=np.float16,
            shape=shapes["e2_query_weighted"],
        )
        normalization_progress = tqdm(
            total=covered_pois,
            desc="E1/E2 normalization",
            unit="poi",
            disable=not show_progress,
        )
        for start in range(0, covered_pois, 2_048):
            stop = min(start + 2_048, covered_pois)
            e1_chunk = e1_accumulator[start:stop].copy()
            e2_chunk = e2_accumulator[start:stop].copy()
            _normalize_rows(e1_chunk, "E1")
            _normalize_rows(e2_chunk, "E2")
            e1_output[start:stop] = e1_chunk.astype(np.float16)
            e2_output[start:stop] = e2_chunk.astype(np.float16)
            normalization_progress.update(stop - start)
        normalization_progress.close()
        e1_output.flush()
        e2_output.flush()

        outputs: dict[str, Any] = {}
        validation: dict[str, Any] = {
            "covered_poi_rows_unique": True,
            "covered_poi_rows_in_range": True,
            "train_order_count_conserved": max_partitions is not None
            or int(train_order_count.sum()) == stats.source_rows,
            "unique_query_count_conserved": max_partitions is not None
            or int(unique_query_count.sum()) == stats.unique_query_poi_pairs,
        }
        for name, (filename, dtype) in OUTPUT_FILES.items():
            path = staging_dir / filename
            outputs[name] = {
                "file": filename,
                "shape": list(shapes[name]),
                "dtype": dtype,
                "sha256": _sha256_file(path),
            }
            if name in ("e1_query_mean", "e2_query_weighted", "e2_weight_sum"):
                validation[name] = _array_validation(path)
        _write_json_atomic(
            progress_path,
            {
                "signature": signature,
                "status": "completed",
                "stage": "completed",
                "next_query_shard": len(query_poi_files),
                "next_normalization_row": covered_pois,
                "recoverable": False,
                "updated_at": _utc_now(),
            },
        )
        manifest["outputs"] = outputs
        manifest["validation"] = validation
        manifest["runtime"].update(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
            }
        )
        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        (staging_dir / "_SUCCESS").touch()
        os.replace(staging_dir, output_dir)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = _utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json_atomic(manifest_path, manifest)
        raise
    return QueryAugmentationResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        covered_pois=covered_pois,
        embedding_dim=embedding_dim,
        partitions=selected_partitions,
        reused=False,
    )
