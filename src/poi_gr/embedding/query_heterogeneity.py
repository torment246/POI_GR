"""Build Train-only heterogeneity features for adaptive Query fusion."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from .query_augmentation import (
    _load_numeric_poi_index,
    _load_query_df,
    _map_poi_ids,
    validate_query_poi_aggregates,
)
from .query_stats import validate_train_query_stats


SCHEMA_VERSION = "query-heterogeneity-v1"
OUTPUT_FILES = {
    "e2_weight_square_sum": ("e2_weight_square_sum.npy", "float64"),
    "e2_effective_query_count": ("e2_effective_query_count.npy", "float32"),
    "content_query_cosine": ("content_query_cosine.npy", "float32"),
}


class QueryHeterogeneityError(RuntimeError):
    """Raised when E3 heterogeneity statistics are invalid."""


@dataclass(frozen=True)
class QueryHeterogeneityConfig:
    job_name: str
    query_stats_dir: Path
    expected_stats_manifest_sha256: str
    aggregates_dir: Path
    expected_aggregates_manifest_sha256: str
    poi_embedding_dir: Path
    expected_poi_embeddings_sha256: str
    expected_poi_ids_sha256: str
    expected_poi_rows: int
    expected_covered_pois: int
    output_dir: Path
    read_batch_rows: int
    cosine_chunk_rows: int


@dataclass(frozen=True)
class QueryHeterogeneityResult:
    output_dir: Path
    manifest_path: Path
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


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise QueryHeterogeneityError(f"{name} 不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise QueryHeterogeneityError(f"{name} 必须是 JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueryHeterogeneityError(f"{name} 必须是 mapping")
    return value


def _path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QueryHeterogeneityError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise QueryHeterogeneityError(f"{name} 必须是 64 位 SHA256")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QueryHeterogeneityError(f"{name} 必须是正整数")
    return value


def load_query_heterogeneity_config(
    config_path: Path,
    project_root: Path,
) -> QueryHeterogeneityConfig:
    """Load the frozen E3 feature-build configuration."""

    with config_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle), "配置根节点")
    input_config = _mapping(root.get("input"), "input")
    output_config = _mapping(root.get("output"), "output")
    runtime_config = _mapping(root.get("runtime"), "runtime")
    job_name = root.get("job_name")
    if not isinstance(job_name, str) or not job_name.strip():
        raise QueryHeterogeneityError("job_name 必须是非空字符串")
    return QueryHeterogeneityConfig(
        job_name=job_name,
        query_stats_dir=_path(
            input_config.get("query_stats_dir"),
            project_root,
            "input.query_stats_dir",
        ),
        expected_stats_manifest_sha256=_sha(
            input_config.get("expected_stats_manifest_sha256"),
            "input.expected_stats_manifest_sha256",
        ),
        aggregates_dir=_path(
            input_config.get("aggregates_dir"),
            project_root,
            "input.aggregates_dir",
        ),
        expected_aggregates_manifest_sha256=_sha(
            input_config.get("expected_aggregates_manifest_sha256"),
            "input.expected_aggregates_manifest_sha256",
        ),
        poi_embedding_dir=_path(
            input_config.get("poi_embedding_dir"),
            project_root,
            "input.poi_embedding_dir",
        ),
        expected_poi_embeddings_sha256=_sha(
            input_config.get("expected_poi_embeddings_sha256"),
            "input.expected_poi_embeddings_sha256",
        ),
        expected_poi_ids_sha256=_sha(
            input_config.get("expected_poi_ids_sha256"),
            "input.expected_poi_ids_sha256",
        ),
        expected_poi_rows=_positive_int(
            input_config.get("expected_poi_rows"),
            "input.expected_poi_rows",
        ),
        expected_covered_pois=_positive_int(
            input_config.get("expected_covered_pois"),
            "input.expected_covered_pois",
        ),
        output_dir=_path(
            output_config.get("dir"), project_root, "output.dir"
        ),
        read_batch_rows=_positive_int(
            runtime_config.get("read_batch_rows"),
            "runtime.read_batch_rows",
        ),
        cosine_chunk_rows=_positive_int(
            runtime_config.get("cosine_chunk_rows"),
            "runtime.cosine_chunk_rows",
        ),
    )


def compute_effective_query_count(
    weight_sum: np.ndarray,
    weight_square_sum: np.ndarray,
    unique_query_count: np.ndarray,
) -> np.ndarray:
    """Compute Kish effective count and enforce its mathematical bounds."""

    if not (
        weight_sum.shape
        == weight_square_sum.shape
        == unique_query_count.shape
    ):
        raise QueryHeterogeneityError("有效 Query 数输入 shape 不一致")
    if (
        not np.isfinite(weight_sum).all()
        or not np.isfinite(weight_square_sum).all()
        or np.any(weight_sum <= 0)
        or np.any(weight_square_sum <= 0)
        or np.any(unique_query_count <= 0)
    ):
        raise QueryHeterogeneityError("有效 Query 数输入包含非法值")
    effective = np.square(weight_sum, dtype=np.float64) / weight_square_sum
    tolerance = 1e-6 * np.maximum(unique_query_count, 1)
    if np.any(effective < 1.0 - tolerance) or np.any(
        effective > unique_query_count + tolerance
    ):
        raise QueryHeterogeneityError("有效 Query 数超出 [1, unique_query_count]")
    effective = np.clip(effective, 1.0, unique_query_count)
    return effective.astype(np.float32)


def content_query_cosine(
    content_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute aligned cosine similarity without assuming exact unit norms."""

    content = np.asarray(content_embeddings, dtype=np.float32)
    query = np.asarray(query_embeddings, dtype=np.float32)
    if content.shape != query.shape or content.ndim != 2:
        raise QueryHeterogeneityError("内容与 Query 聚合向量 shape 不一致")
    content_norm = np.linalg.norm(content, axis=1)
    query_norm = np.linalg.norm(query, axis=1)
    denominator = content_norm * query_norm
    if (
        not np.isfinite(denominator).all()
        or np.any(denominator <= 0)
        or not np.isfinite(content).all()
        or not np.isfinite(query).all()
    ):
        raise QueryHeterogeneityError("余弦输入存在零范数或非有限值")
    cosine = np.sum(content * query, axis=1) / denominator
    if np.any(cosine < -1.0001) or np.any(cosine > 1.0001):
        raise QueryHeterogeneityError("内容–Query 余弦超出 [-1, 1]")
    return np.clip(cosine, -1.0, 1.0).astype(np.float32)


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _distribution(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values.astype(np.float64), [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    return {
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(values, dtype=np.float64)),
    }


def validate_query_heterogeneity(
    output_dir: Path,
    *,
    expected_aggregates_manifest_sha256: str | None = None,
) -> QueryHeterogeneityResult:
    """Validate E3 feature hashes, shapes, finiteness, and bounds."""

    manifest = _load_json(output_dir / "manifest.json", "E3 特征 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise QueryHeterogeneityError("E3 特征 schema 不兼容")
    if manifest.get("status") != "completed" or not (
        output_dir / "_SUCCESS"
    ).is_file():
        raise QueryHeterogeneityError("E3 特征产物未完成")
    if (
        expected_aggregates_manifest_sha256 is not None
        and manifest.get("inputs", {}).get("aggregates_manifest_sha256")
        != expected_aggregates_manifest_sha256
    ):
        raise QueryHeterogeneityError("E3 特征不是冻结 E1/E2 聚合版本")
    covered_pois = int(manifest.get("features", {}).get("covered_pois", -1))
    outputs = manifest.get("outputs")
    if covered_pois <= 0 or not isinstance(outputs, Mapping):
        raise QueryHeterogeneityError("E3 特征 manifest 字段非法")
    arrays: dict[str, np.ndarray] = {}
    for name, (filename, dtype) in OUTPUT_FILES.items():
        spec = outputs.get(name)
        if not isinstance(spec, Mapping):
            raise QueryHeterogeneityError(f"manifest 缺少 {name}")
        path = output_dir / filename
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (covered_pois,) or str(values.dtype) != dtype:
            raise QueryHeterogeneityError(f"{filename} shape/dtype 非法")
        if not np.isfinite(values).all():
            raise QueryHeterogeneityError(f"{filename} 存在 NaN/Inf")
        if _sha256_file(path) != spec.get("sha256"):
            raise QueryHeterogeneityError(f"{filename} SHA256 不一致")
        arrays[name] = values
    effective = arrays["e2_effective_query_count"]
    cosine = arrays["content_query_cosine"]
    if np.any(effective < 1) or np.any(cosine < -1) or np.any(cosine > 1):
        raise QueryHeterogeneityError("E3 特征取值越界")
    return QueryHeterogeneityResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        covered_pois=covered_pois,
        reused=True,
    )


def build_query_heterogeneity(
    config: QueryHeterogeneityConfig,
    *,
    project_root: Path,
    show_progress: bool = True,
) -> QueryHeterogeneityResult:
    """Stream Train Query pairs and atomically publish small E3 features."""

    output_dir = config.output_dir.resolve()
    if (output_dir / "_SUCCESS").is_file():
        return validate_query_heterogeneity(
            output_dir,
            expected_aggregates_manifest_sha256=(
                config.expected_aggregates_manifest_sha256
            ),
        )
    if output_dir.exists():
        raise QueryHeterogeneityError(f"正式输出目录已存在但未完成：{output_dir}")
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    if staging_dir.exists():
        raise QueryHeterogeneityError(f"存在未完成 staging：{staging_dir}")

    stats = validate_train_query_stats(config.query_stats_dir)
    stats_manifest_sha256 = _sha256_file(stats.manifest_path)
    if stats_manifest_sha256 != config.expected_stats_manifest_sha256:
        raise QueryHeterogeneityError("Query 统计 manifest SHA256 不一致")
    aggregate_manifest_path = config.aggregates_dir / "manifest.json"
    if (
        _sha256_file(aggregate_manifest_path)
        != config.expected_aggregates_manifest_sha256
    ):
        raise QueryHeterogeneityError("E1/E2 聚合 manifest SHA256 不一致")
    validate_query_poi_aggregates(config.aggregates_dir)

    poi_embeddings_path = config.poi_embedding_dir / "embeddings.npy"
    poi_ids_path = config.poi_embedding_dir / "poi_ids.jsonl"
    if _sha256_file(poi_embeddings_path) != config.expected_poi_embeddings_sha256:
        raise QueryHeterogeneityError("POI embedding SHA256 不一致")
    if _sha256_file(poi_ids_path) != config.expected_poi_ids_sha256:
        raise QueryHeterogeneityError("POI ID SHA256 不一致")

    aggregate_manifest = _load_json(
        aggregate_manifest_path, "E1/E2 聚合 manifest"
    )
    if (
        int(aggregate_manifest.get("aggregation", {}).get("covered_pois", -1))
        != config.expected_covered_pois
    ):
        raise QueryHeterogeneityError("E1/E2 覆盖 POI 数不一致")
    covered_rows = np.load(
        config.aggregates_dir / "covered_poi_rows.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    unique_query_count = np.load(
        config.aggregates_dir / "unique_query_count.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    stored_weight_sum = np.load(
        config.aggregates_dir / "e2_weight_sum.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    e2_embeddings = np.load(
        config.aggregates_dir / "e2_query_weighted.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    poi_embeddings = np.load(
        poi_embeddings_path, mmap_mode="r", allow_pickle=False
    )
    if poi_embeddings.shape[0] != config.expected_poi_rows:
        raise QueryHeterogeneityError("POI embedding 行数不一致")

    stats_manifest = _load_json(stats.manifest_path, "Query 统计 manifest")
    stats_outputs = _mapping(stats_manifest.get("outputs"), "统计 outputs")
    query_catalog = _mapping(
        stats_outputs.get("query_catalog"), "统计 query_catalog"
    )
    query_poi = _mapping(stats_outputs.get("query_poi"), "统计 query_poi")
    query_catalog_files = list(query_catalog.get("files", []))
    query_poi_files = list(query_poi.get("files", []))
    if not query_catalog_files or not query_poi_files:
        raise QueryHeterogeneityError("Query 统计分片为空")
    total_queries = int(query_catalog.get("rows", -1))
    query_df = _load_query_df(
        config.query_stats_dir,
        query_catalog_files,
        total_queries,
        config.read_batch_rows,
    )
    sorted_poi_ids, sorted_poi_rows = _load_numeric_poi_index(
        poi_ids_path, config.expected_poi_rows
    )
    position_by_poi_row = np.full(
        config.expected_poi_rows, -1, dtype=np.int32
    )
    position_by_poi_row[covered_rows] = np.arange(
        config.expected_covered_pois, dtype=np.int32
    )
    recomputed_weight_sum = np.zeros(config.expected_covered_pois, dtype=np.float64)
    weight_square_sum = np.zeros(config.expected_covered_pois, dtype=np.float64)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        progress = tqdm(
            total=len(query_poi_files),
            desc="E3 heterogeneity",
            unit="query-shard",
            disable=not show_progress,
        )
        selected_pairs = 0
        for pair_info in query_poi_files:
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
            order_counts = table.column("train_order_count").to_numpy(
                zero_copy_only=False
            )
            positions = position_by_poi_row[poi_rows]
            selected = positions >= 0
            selected_positions = positions[selected]
            selected_query_ids = query_ids[selected]
            weights = np.log1p(order_counts[selected].astype(np.float64)) * np.log(
                (config.expected_covered_pois + 1.0)
                / (query_df[selected_query_ids] + 1.0)
            )
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise QueryHeterogeneityError("E2 权重存在非正数或非有限值")
            order = np.argsort(selected_positions, kind="stable")
            sorted_positions = selected_positions[order]
            starts = np.r_[
                0,
                np.flatnonzero(sorted_positions[1:] != sorted_positions[:-1]) + 1,
            ]
            unique_positions = sorted_positions[starts]
            sorted_weights = weights[order]
            recomputed_weight_sum[unique_positions] += np.add.reduceat(
                sorted_weights, starts
            )
            weight_square_sum[unique_positions] += np.add.reduceat(
                np.square(sorted_weights), starts
            )
            selected_pairs += len(weights)
            progress.update(1)
        progress.close()

        relative_error = np.max(
            np.abs(recomputed_weight_sum - stored_weight_sum.astype(np.float64))
            / np.maximum(recomputed_weight_sum, 1e-12)
        )
        if relative_error > 5e-6:
            raise QueryHeterogeneityError(
                f"重算 E2 权重和与冻结产物偏差过大：{relative_error}"
            )
        effective = compute_effective_query_count(
            recomputed_weight_sum,
            weight_square_sum,
            unique_query_count,
        )
        cosine_path = staging_dir / OUTPUT_FILES["content_query_cosine"][0]
        cosine = np.lib.format.open_memmap(
            cosine_path,
            mode="w+",
            dtype=np.float32,
            shape=(config.expected_covered_pois,),
        )
        e2_embeddings_in_memory = np.array(
            e2_embeddings,
            dtype=np.float16,
            copy=True,
        )
        progress = tqdm(
            total=config.expected_poi_rows,
            desc="Sequential content cosine",
            unit="catalog-row",
            disable=not show_progress,
        )
        for start in range(0, config.expected_poi_rows, config.cosine_chunk_rows):
            stop = min(start + config.cosine_chunk_rows, config.expected_poi_rows)
            aggregate_positions = position_by_poi_row[start:stop]
            selected = aggregate_positions >= 0
            if np.any(selected):
                selected_positions = aggregate_positions[selected]
                content_chunk = np.asarray(
                    poi_embeddings[start:stop], dtype=np.float32
                )
                cosine[selected_positions] = content_query_cosine(
                    content_chunk[selected],
                    e2_embeddings_in_memory[selected_positions],
                )
            progress.update(stop - start)
        progress.close()
        cosine.flush()
        del cosine
        del e2_embeddings_in_memory

        for name, values in (
            ("e2_weight_square_sum", weight_square_sum),
            ("e2_effective_query_count", effective),
        ):
            filename, dtype = OUTPUT_FILES[name]
            path = staging_dir / filename
            with path.open("wb") as handle:
                np.save(handle, values.astype(dtype, copy=False), allow_pickle=False)

        output_specs: dict[str, Any] = {}
        for name, (filename, dtype) in OUTPUT_FILES.items():
            path = staging_dir / filename
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            output_specs[name] = {
                "file": filename,
                "shape": list(values.shape),
                "dtype": dtype,
                "sha256": _sha256_file(path),
            }
        cosine_values = np.load(
            staging_dir / OUTPUT_FILES["content_query_cosine"][0],
            mmap_mode="r",
            allow_pickle=False,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "job_name": config.job_name,
            "status": "completed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "inputs": {
                "query_stats_dir": str(config.query_stats_dir.resolve()),
                "query_stats_manifest_sha256": stats_manifest_sha256,
                "aggregates_dir": str(config.aggregates_dir.resolve()),
                "aggregates_manifest_sha256": (
                    config.expected_aggregates_manifest_sha256
                ),
                "poi_embeddings": str(poi_embeddings_path.resolve()),
                "poi_embeddings_sha256": config.expected_poi_embeddings_sha256,
                "poi_ids": str(poi_ids_path.resolve()),
                "poi_ids_sha256": config.expected_poi_ids_sha256,
            },
            "features": {
                "covered_pois": config.expected_covered_pois,
                "selected_query_poi_pairs": selected_pairs,
                "weight_formula": (
                    "log1p(count_qi) * log((N+1)/(df_q+1))"
                ),
                "effective_query_count": "(sum w)^2 / sum(w^2)",
                "content_query_cosine": (
                    "cosine(frozen content embedding, E2 weighted query embedding)"
                ),
                "weight_sum_max_relative_error": float(relative_error),
                "effective_query_count_distribution": _distribution(effective),
                "content_query_cosine_distribution": _distribution(cosine_values),
            },
            "outputs": output_specs,
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "read_batch_rows": config.read_batch_rows,
                "cosine_chunk_rows": config.cosine_chunk_rows,
            },
            "git": _git_state(project_root),
        }
        _write_json_atomic(staging_dir / "manifest.json", manifest)
        (staging_dir / "_SUCCESS").touch()
        os.replace(staging_dir, output_dir)
        validated = validate_query_heterogeneity(
            output_dir,
            expected_aggregates_manifest_sha256=(
                config.expected_aggregates_manifest_sha256
            ),
        )
        return QueryHeterogeneityResult(
            output_dir=validated.output_dir,
            manifest_path=validated.manifest_path,
            covered_pois=validated.covered_pois,
            reused=False,
        )
    except Exception as error:
        if staging_dir.exists():
            _write_json_atomic(
                staging_dir / "failure.json",
                {
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise
