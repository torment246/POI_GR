"""Encode the frozen Train Query catalog into a resumable aligned NPY file."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from .core import (
    ModelConfig,
    OutputConfig,
    TextEncoder,
    _load_encoder,
    encode_texts_to_npy,
    load_job_config,
)
from .query_stats import validate_train_query_stats


SCHEMA_VERSION = "train-query-embeddings-v1"


class QueryEncodingError(RuntimeError):
    """Raised when Train Query encoding cannot be run or validated."""


@dataclass(frozen=True)
class TrainQueryEncodingConfig:
    job_name: str
    query_stats_dir: Path
    expected_stats_manifest_sha256: str
    expected_queries: int
    source_model_config: Path
    model: ModelConfig
    output: OutputConfig


@dataclass(frozen=True)
class TrainQueryEncodingResult:
    output_dir: Path
    manifest_path: Path
    embeddings_path: Path
    rows: int
    embedding_dim: int
    reused: bool


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise QueryEncodingError(f"{name} 不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise QueryEncodingError(f"{name} JSON 非法：{path}") from error
    if not isinstance(payload, dict):
        raise QueryEncodingError(f"{name} 必须是 JSON object：{path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueryEncodingError(f"{name} 必须是 YAML mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QueryEncodingError(f"{name} 必须是正整数")
    return value


def _resolve_path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QueryEncodingError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_train_query_encoding_config(
    config_path: Path,
    project_root: Path,
) -> TrainQueryEncodingConfig:
    """Load the frozen Train Query encoding configuration."""

    with config_path.open("r", encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle), "配置根节点")
    input_raw = _require_mapping(raw.get("input"), "input")
    model_raw = _require_mapping(raw.get("model"), "model")
    output_raw = _require_mapping(raw.get("output"), "output")
    job_name = raw.get("job_name")
    if not isinstance(job_name, str) or not job_name.strip():
        raise QueryEncodingError("job_name 必须是非空字符串")

    source_model_config = _resolve_path(
        model_raw.get("source_embedding_config"),
        project_root,
        "model.source_embedding_config",
    )
    source_job = load_job_config(source_model_config, project_root)
    model = replace(
        source_job.model,
        device=str(model_raw.get("device", source_job.model.device)),
        batch_size=_positive_int(model_raw.get("batch_size"), "model.batch_size"),
        encode_buffer_size=_positive_int(
            model_raw.get("encode_buffer_size"),
            "model.encode_buffer_size",
        ),
        max_seq_length=_positive_int(
            model_raw.get("max_seq_length"),
            "model.max_seq_length",
        ),
        prompt_name=None,
    )
    if model.encode_buffer_size < model.batch_size:
        raise QueryEncodingError("model.encode_buffer_size 不能小于 batch_size")
    if not model.normalize_embeddings:
        raise QueryEncodingError("Train Query 编码必须启用 L2 normalize")
    if model.padding_side != "right":
        raise QueryEncodingError("BGE-M3 Train Query 编码必须使用 right padding")
    if model.torch_dtype != "bfloat16":
        raise QueryEncodingError("Train Query 编码必须使用 bfloat16 模型")

    output = OutputConfig(
        dir=_resolve_path(output_raw.get("dir"), project_root, "output.dir"),
        embedding_dtype=str(output_raw.get("embedding_dtype", "float16")),
        checkpoint_interval_batches=_positive_int(
            output_raw.get("checkpoint_interval_batches"),
            "output.checkpoint_interval_batches",
        ),
        resume=bool(output_raw.get("resume", True)),
    )
    if output.embedding_dtype != "float16":
        raise QueryEncodingError("Train Query embedding 输出必须为 float16")

    expected_sha256 = input_raw.get("expected_manifest_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise QueryEncodingError("input.expected_manifest_sha256 非法")
    return TrainQueryEncodingConfig(
        job_name=job_name,
        query_stats_dir=_resolve_path(
            input_raw.get("query_stats_dir"),
            project_root,
            "input.query_stats_dir",
        ),
        expected_stats_manifest_sha256=expected_sha256,
        expected_queries=_positive_int(
            input_raw.get("expected_queries"),
            "input.expected_queries",
        ),
        source_model_config=source_model_config,
        model=model,
        output=output,
    )


def apply_query_encoding_overrides(
    config: TrainQueryEncodingConfig,
    *,
    output_dir: Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    encode_buffer_size: int | None = None,
) -> TrainQueryEncodingConfig:
    """Apply explicit CLI overrides without changing the frozen source config."""

    model = replace(
        config.model,
        device=device if device is not None else config.model.device,
        batch_size=batch_size if batch_size is not None else config.model.batch_size,
        encode_buffer_size=(
            encode_buffer_size
            if encode_buffer_size is not None
            else config.model.encode_buffer_size
        ),
    )
    if model.batch_size <= 0 or model.encode_buffer_size < model.batch_size:
        raise QueryEncodingError("覆盖后的 batch/buffer 配置非法")
    output = replace(
        config.output,
        dir=output_dir if output_dir is not None else config.output.dir,
    )
    return replace(config, model=model, output=output)


def iter_query_catalog(
    query_stats_dir: Path,
    *,
    start_row: int = 0,
    stop_row: int | None = None,
    read_batch_rows: int = 65_536,
) -> Iterator[str]:
    """Yield exact raw Queries in contiguous query_id order."""

    if start_row < 0:
        raise QueryEncodingError("start_row 不能为负数")
    if stop_row is not None and stop_row < start_row:
        raise QueryEncodingError("stop_row 不能小于 start_row")
    manifest = _load_json_object(
        query_stats_dir / "manifest.json", "Query 统计 manifest"
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise QueryEncodingError("Query 统计 manifest 缺少 outputs")
    catalog = outputs.get("query_catalog")
    if not isinstance(catalog, Mapping) or not isinstance(catalog.get("files"), list):
        raise QueryEncodingError("Query 统计 manifest 缺少 Query 目录文件")
    total_rows = int(catalog.get("rows", -1))
    effective_stop = total_rows if stop_row is None else stop_row
    if effective_stop > total_rows:
        raise QueryEncodingError("stop_row 超出 Query 目录总行数")

    global_row = 0
    yielded = 0
    for file_info in catalog["files"]:
        if not isinstance(file_info, Mapping):
            raise QueryEncodingError("Query 目录文件项非法")
        file_rows = int(file_info.get("rows", -1))
        file_stop = global_row + file_rows
        if file_stop <= start_row:
            global_row = file_stop
            continue
        if global_row >= effective_stop:
            break
        path = query_stats_dir / "query_catalog" / str(file_info.get("file", ""))
        parquet = pq.ParquetFile(path)
        local_row = 0
        for batch in parquet.iter_batches(
            batch_size=read_batch_rows,
            columns=["query_id", "query"],
        ):
            query_ids = batch.column(0).to_pylist()
            queries = batch.column(1).to_pylist()
            for offset, (query_id, query) in enumerate(
                zip(query_ids, queries, strict=True)
            ):
                row = global_row + local_row + offset
                if query_id != row:
                    raise QueryEncodingError(
                        f"Query 目录 query_id={query_id} 与行号 {row} 不一致"
                    )
                if row < start_row:
                    continue
                if row >= effective_stop:
                    break
                if not isinstance(query, str) or not query.strip():
                    raise QueryEncodingError(f"Query 目录第 {row} 行为空")
                yielded += 1
                yield query
            local_row += len(batch)
            if global_row + local_row >= effective_stop:
                break
        global_row = file_stop
    expected_yielded = effective_stop - start_row
    if yielded != expected_yielded:
        raise QueryEncodingError(f"只读取 {yielded} 个 Query，期望 {expected_yielded}")


def _signature(
    config: TrainQueryEncodingConfig,
    *,
    stats_manifest_sha256: str,
    total_rows: int,
    embedding_dim: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stats_manifest_sha256": stats_manifest_sha256,
        "total_rows": total_rows,
        "source_model_config": str(config.source_model_config),
        "model": {**asdict(config.model), "path": str(config.model.path)},
        "output_dtype": config.output.embedding_dtype,
        "embedding_dim": embedding_dim,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


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


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _embedding_metrics(path: Path, sample_rows: int = 2_048) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r")
    total_rows = values.shape[0]
    all_finite = True
    for start in range(0, total_rows, 8_192):
        stop = min(start + 8_192, total_rows)
        if not np.isfinite(np.asarray(values[start:stop])).all():
            all_finite = False
            break
    if not all_finite:
        raise QueryEncodingError("Train Query embedding 存在 NaN/Inf")
    if total_rows <= sample_rows:
        indices = np.arange(total_rows)
    else:
        indices = np.linspace(0, total_rows - 1, sample_rows, dtype=np.int64)
    sampled = np.asarray(values[indices], dtype=np.float32)
    norms = np.linalg.norm(sampled, axis=1)
    return {
        "all_finite": True,
        "norm_sample_rows": int(len(indices)),
        "l2_norm_min": float(norms.min()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_max": float(norms.max()),
    }


def validate_train_query_embeddings(
    output_dir: Path,
    *,
    expected_stats_manifest_sha256: str | None = None,
) -> TrainQueryEncodingResult:
    """Validate a completed Train Query embedding artifact."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = _load_json_object(manifest_path, "Train Query embedding manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise QueryEncodingError("Train Query embedding manifest schema 不兼容")
    if manifest.get("status") != "completed":
        raise QueryEncodingError("Train Query embedding manifest 状态不是 completed")
    if not (output_dir / "_SUCCESS").is_file():
        raise QueryEncodingError("Train Query embedding 缺少 _SUCCESS")
    source = manifest.get("source")
    output = manifest.get("output")
    if not isinstance(source, Mapping) or not isinstance(output, Mapping):
        raise QueryEncodingError("Train Query embedding manifest 缺少 source/output")
    stats_manifest_path = Path(str(source.get("query_stats_manifest", "")))
    stats_sha256 = _sha256_file(stats_manifest_path)
    if stats_sha256 != source.get("query_stats_manifest_sha256"):
        raise QueryEncodingError("Query 统计 manifest SHA256 与编码产物不一致")
    if (
        expected_stats_manifest_sha256 is not None
        and stats_sha256 != expected_stats_manifest_sha256
    ):
        raise QueryEncodingError("Query 统计 manifest 不是配置冻结的版本")

    embeddings_path = output_dir / str(output.get("embeddings", ""))
    if not embeddings_path.is_file():
        raise QueryEncodingError(f"Train Query embedding 不存在：{embeddings_path}")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    expected_shape = tuple(output.get("shape", ()))
    if embeddings.shape != expected_shape:
        raise QueryEncodingError("Train Query embedding shape 与 manifest 不一致")
    if str(embeddings.dtype) != output.get("dtype"):
        raise QueryEncodingError("Train Query embedding dtype 与 manifest 不一致")
    if _sha256_file(embeddings_path) != output.get("sha256"):
        raise QueryEncodingError("Train Query embedding SHA256 与 manifest 不一致")
    progress = _load_json_object(output_dir / "progress.json", "编码进度")
    if progress.get("status") != "completed":
        raise QueryEncodingError("Train Query embedding progress 不是 completed")
    if int(progress.get("next_row", -1)) != embeddings.shape[0]:
        raise QueryEncodingError("Train Query embedding progress 行数不一致")
    return TrainQueryEncodingResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        embeddings_path=embeddings_path,
        rows=embeddings.shape[0],
        embedding_dim=embeddings.shape[1],
        reused=True,
    )


def run_train_query_encoding(
    config: TrainQueryEncodingConfig,
    *,
    project_root: Path,
    max_rows: int | None = None,
    show_progress: bool = True,
    encoder_loader: Callable[[ModelConfig], tuple[TextEncoder, str, float]] = (
        _load_encoder
    ),
) -> TrainQueryEncodingResult:
    """Encode the frozen unique Query catalog with resumable row alignment."""

    stats_result = validate_train_query_stats(config.query_stats_dir)
    stats_manifest_path = stats_result.manifest_path
    stats_manifest_sha256 = _sha256_file(stats_manifest_path)
    if stats_manifest_sha256 != config.expected_stats_manifest_sha256:
        raise QueryEncodingError("Query 统计 manifest SHA256 不是配置冻结的版本")
    if stats_result.unique_queries != config.expected_queries:
        raise QueryEncodingError(
            f"唯一 Query 数 {stats_result.unique_queries} != 配置 {config.expected_queries}"
        )
    if max_rows is not None and max_rows <= 0:
        raise QueryEncodingError("max_rows 必须大于 0")
    total_rows = (
        config.expected_queries
        if max_rows is None
        else min(max_rows, config.expected_queries)
    )
    output_dir = config.output.dir.resolve()
    embeddings_path = output_dir / "embeddings.npy"
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"

    if (output_dir / "_SUCCESS").is_file():
        completed = _load_json_object(manifest_path, "Train Query embedding manifest")
        model = completed.get("model")
        if not isinstance(model, Mapping):
            raise QueryEncodingError("已完成 manifest 缺少 model")
        completed_signature = _signature(
            config,
            stats_manifest_sha256=stats_manifest_sha256,
            total_rows=total_rows,
            embedding_dim=int(model.get("embedding_dim", 0)),
        )
        if completed_signature != completed.get("signature"):
            raise QueryEncodingError("已完成产物与当前编码配置不一致")
        return validate_train_query_embeddings(
            output_dir,
            expected_stats_manifest_sha256=config.expected_stats_manifest_sha256,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    has_progress = progress_path.is_file()
    if not has_progress and (embeddings_path.exists() or manifest_path.exists()):
        raise QueryEncodingError("输出目录已有产物但缺少 progress.json")

    encoder, device, model_load_seconds = encoder_loader(config.model)
    if hasattr(encoder, "eval"):
        encoder.eval()  # type: ignore[attr-defined]
    embedding_dim = encoder.get_sentence_embedding_dimension()
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise QueryEncodingError("编码器未返回有效 embedding 维度")
    signature = _signature(
        config,
        stats_manifest_sha256=stats_manifest_sha256,
        total_rows=total_rows,
        embedding_dim=embedding_dim,
    )
    start_row = 0
    if has_progress:
        progress = _load_json_object(progress_path, "编码进度")
        if progress.get("signature") != signature:
            raise QueryEncodingError("已有断点与当前数据或编码配置不一致")
        if not config.output.resume:
            raise QueryEncodingError("存在编码断点但 output.resume=false")
        start_row = int(progress.get("next_row", 0))
        if start_row < 0 or start_row > total_rows:
            raise QueryEncodingError("编码断点 next_row 超出合法范围")
        if progress.get("status") == "completed" and start_row == total_rows:
            raise QueryEncodingError("编码已完成但缺少 _SUCCESS，请先审计产物")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_name": config.job_name,
        "status": "running",
        "started_at": _utc_now(),
        "source": {
            "query_stats_dir": str(config.query_stats_dir.resolve()),
            "query_stats_manifest": str(stats_manifest_path.resolve()),
            "query_stats_manifest_sha256": stats_manifest_sha256,
            "full_query_rows": config.expected_queries,
            "selected_query_rows": total_rows,
            "query_id_range": [0, total_rows],
            "query_normalization": "none",
        },
        "model": {
            **asdict(config.model),
            "path": str(config.model.path),
            "source_embedding_config": str(config.source_model_config),
            "actual_device": device,
            "embedding_dim": embedding_dim,
        },
        "output": {
            "dir": str(output_dir),
            "embeddings": embeddings_path.name,
            "shape": [total_rows, embedding_dim],
            "dtype": config.output.embedding_dtype,
            "row_mapping": "row_index equals query_id",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "sentence_transformers": _package_version("sentence-transformers"),
            "model_load_seconds": model_load_seconds,
        },
        "git": _git_state(project_root),
        "signature": signature,
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        progress_path,
        {
            "signature": signature,
            "status": "running",
            "next_row": start_row,
            "total_rows": total_rows,
            "updated_at": _utc_now(),
        },
    )

    cuda_device = None
    if device.startswith("cuda"):
        import torch

        cuda_device = torch.device(device)
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        manifest["runtime"]["gpu"] = torch.cuda.get_device_name(cuda_device)
        _write_json_atomic(manifest_path, manifest)

    encode_started = time.perf_counter()
    try:
        encode_metrics = encode_texts_to_npy(
            encoder,
            iter_query_catalog(
                config.query_stats_dir,
                start_row=start_row,
                stop_row=total_rows,
            ),
            embeddings_path=embeddings_path,
            progress_path=progress_path,
            total_rows=total_rows,
            embedding_dim=embedding_dim,
            start_row=start_row,
            model_config=config.model,
            output_config=config.output,
            signature=signature,
            show_progress=show_progress,
            progress_desc="Train Query embedding",
        )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            encode_metrics.update(
                {
                    "cuda_peak_memory_allocated_bytes": allocated,
                    "cuda_peak_memory_reserved_bytes": reserved,
                    "cuda_peak_memory_allocated_gib": allocated / 1024**3,
                    "cuda_peak_memory_reserved_gib": reserved / 1024**3,
                }
            )
        manifest["validation"] = _embedding_metrics(embeddings_path)
        manifest["output"]["sha256"] = _sha256_file(embeddings_path)
        manifest["runtime"].update(encode_metrics)
        manifest["runtime"]["total_seconds_this_run"] = (
            time.perf_counter() - encode_started
        )
        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        (output_dir / "_SUCCESS").touch()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = _utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json_atomic(manifest_path, manifest)
        raise
    finally:
        del encoder
        gc.collect()
        if cuda_device is not None:
            torch.cuda.empty_cache()

    return TrainQueryEncodingResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        embeddings_path=embeddings_path,
        rows=total_rows,
        embedding_dim=embedding_dim,
        reused=False,
    )
