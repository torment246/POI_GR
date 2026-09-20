"""Build a resumable normalized query-embedding cache."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.config import QGPRQKConfig, QueryEmbeddingConfig
from qg_prqk.data.query_statistics import validate_query_stats


SCHEMA_VERSION = "qg-prqk-query-embeddings-v1"


class QueryEmbeddingError(RuntimeError):
    """Raised when the normalized Query cache cannot be built or validated."""


class TextEncoder(Protocol):
    def get_sentence_embedding_dimension(self) -> int | None:
        ...

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> np.ndarray:
        ...


@dataclass(frozen=True)
class QueryEmbeddingResult:
    output_dir: Path
    manifest_path: Path
    embeddings_path: Path
    rows: int
    embedding_dim: int
    reused: bool


def load_text_encoder(
    config: QueryEmbeddingConfig,
) -> tuple[TextEncoder, str, float]:
    """Load the frozen local BGE model without a Query instruction."""

    import torch

    if not config.model_path.is_dir():
        raise QueryEmbeddingError(f"BGE 模型目录不存在：{config.model_path}")
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise QueryEmbeddingError("配置要求 CUDA，但当前进程无法访问 GPU")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(config.torch_dtype)
    if dtype is None:
        raise QueryEmbeddingError(f"不支持 torch_dtype={config.torch_dtype}")
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if config.attention:
        model_kwargs["attn_implementation"] = config.attention
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    encoder = SentenceTransformer(
        str(config.model_path),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": config.padding_side},
    )
    encoder.max_seq_length = config.max_seq_length
    encoder.eval()
    return encoder, device, time.perf_counter() - started


def _chunked(values: Iterator[str], size: int) -> Iterator[list[str]]:
    while chunk := list(itertools.islice(values, size)):
        yield chunk


def _query_stats_contract(query_stats_dir: Path) -> tuple[dict[str, Any], int, str]:
    result = validate_query_stats(query_stats_dir)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise QueryEmbeddingError("P2 manifest 缺少 outputs")
    query_stats = outputs.get("query_stats")
    if not isinstance(query_stats, Mapping) or not isinstance(
        query_stats.get("files"), list
    ):
        raise QueryEmbeddingError("P2 manifest 缺少 query_stats 文件清单")
    return manifest, result.unique_queries, sha256_file(result.manifest_path)


def iter_normalized_queries(
    query_stats_dir: Path,
    *,
    start_row: int = 0,
    stop_row: int | None = None,
    read_batch_rows: int = 65_536,
) -> Iterator[str]:
    """Yield normalized Queries in contiguous query_id order."""

    manifest, total_rows, _ = _query_stats_contract(query_stats_dir)
    if start_row < 0:
        raise QueryEmbeddingError("start_row 不能为负数")
    effective_stop = total_rows if stop_row is None else stop_row
    if effective_stop < start_row or effective_stop > total_rows:
        raise QueryEmbeddingError("Query cache 行范围非法")
    output = manifest["outputs"]["query_stats"]
    directory = query_stats_dir / str(output["dir"])
    global_row = 0
    yielded = 0
    for file_info in output["files"]:
        if not isinstance(file_info, Mapping):
            raise QueryEmbeddingError("P2 Query 统计文件项非法")
        file_rows = int(file_info.get("rows", -1))
        file_stop = global_row + file_rows
        if file_stop <= start_row:
            global_row = file_stop
            continue
        if global_row >= effective_stop:
            break
        path = directory / str(file_info.get("file", ""))
        local_row = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=read_batch_rows,
            columns=["query_id", "normalized_query"],
        ):
            query_ids = batch.column(0).to_pylist()
            queries = batch.column(1).to_pylist()
            for offset, (query_id, query) in enumerate(
                zip(query_ids, queries, strict=True)
            ):
                row = global_row + local_row + offset
                if int(query_id) != row:
                    raise QueryEmbeddingError(
                        f"P2 query_id={query_id} 与全局行号 {row} 不一致"
                    )
                if row < start_row:
                    continue
                if row >= effective_stop:
                    break
                if not isinstance(query, str) or not query:
                    raise QueryEmbeddingError(f"P2 第 {row} 个 normalized Query 为空")
                yielded += 1
                yield query
            local_row += len(batch)
            if global_row + local_row >= effective_stop:
                break
        global_row = file_stop
    if yielded != effective_stop - start_row:
        raise QueryEmbeddingError(
            f"Query cache 只读取 {yielded} 行，期望 {effective_stop - start_row}"
        )


def _signature(
    config: QGPRQKConfig,
    query_stats_manifest_sha256: str,
    selected_rows: int,
    embedding_dim: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "query_stats_manifest_sha256": query_stats_manifest_sha256,
        "selected_rows": selected_rows,
        "embedding_dim": embedding_dim,
        "model": {
            **asdict(config.query_embedding),
            "model_path": str(config.query_embedding.model_path),
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _embedding_validation(path: Path) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or not min(values.shape):
        raise QueryEmbeddingError("Query embedding 必须是非空二维数组")
    all_finite = True
    for start in range(0, len(values), 8_192):
        if not np.isfinite(np.asarray(values[start : start + 8_192])).all():
            all_finite = False
            break
    if not all_finite:
        raise QueryEmbeddingError("Query embedding 存在 NaN/Inf")
    indices = np.linspace(0, len(values) - 1, min(2_048, len(values)), dtype=np.int64)
    norms = np.linalg.norm(np.asarray(values[indices], dtype=np.float32), axis=1)
    if np.max(np.abs(norms - 1.0)) > 2e-3:
        raise QueryEmbeddingError("Query embedding 未保持 L2 normalize")
    return {
        "all_finite": True,
        "norm_sample_rows": len(indices),
        "l2_norm_min": float(norms.min()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_max": float(norms.max()),
    }


def validate_query_embeddings(output_dir: Path) -> QueryEmbeddingResult:
    """Validate one completed Query embedding cache."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QueryEmbeddingError(f"Query embedding manifest 非法：{manifest_path}") from error
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "completed":
        raise QueryEmbeddingError("Query embedding manifest 版本或状态非法")
    if not (output_dir / "_SUCCESS").is_file():
        raise QueryEmbeddingError("Query embedding 缺少 _SUCCESS")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise QueryEmbeddingError("Query embedding manifest 缺少 output")
    embeddings_path = output_dir / str(output.get("file", ""))
    values = np.load(embeddings_path, mmap_mode="r")
    if list(values.shape) != output.get("shape") or str(values.dtype) != output.get("dtype"):
        raise QueryEmbeddingError("Query embedding shape/dtype 与 manifest 不一致")
    if sha256_file(embeddings_path) != output.get("sha256"):
        raise QueryEmbeddingError("Query embedding SHA256 与 manifest 不一致")
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed" or int(progress.get("next_row", -1)) != len(values):
        raise QueryEmbeddingError("Query embedding progress 未完成或行数不一致")
    _embedding_validation(embeddings_path)
    return QueryEmbeddingResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        embeddings_path=embeddings_path,
        rows=len(values),
        embedding_dim=values.shape[1],
        reused=True,
    )


def build_query_embeddings(
    config: QGPRQKConfig,
    *,
    query_stats_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    encoder_loader: Callable[
        [QueryEmbeddingConfig], tuple[TextEncoder, str, float]
    ] = load_text_encoder,
) -> QueryEmbeddingResult:
    """Encode P2 normalized Queries with row index equal to query_id."""

    query_stats_dir = query_stats_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.parent != config.paths.output_dir.resolve() or output_dir.name != "query_embeddings":
        raise QueryEmbeddingError("Query embedding 必须写入配置 output_dir/query_embeddings")
    if limit is not None and limit <= 0:
        raise QueryEmbeddingError("limit 必须大于 0")
    p2_manifest, total_rows, p2_sha256 = _query_stats_contract(query_stats_dir)
    selected_rows = min(limit, total_rows) if limit is not None else total_rows
    p2_is_sample = bool(p2_manifest.get("source", {}).get("is_prefix_sample"))

    if (output_dir / "_SUCCESS").is_file():
        if not config.runtime.resume:
            raise QueryEmbeddingError(f"输出已完成且 resume=false：{output_dir}")
        result = validate_query_embeddings(output_dir)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source", {}).get("query_stats_manifest_sha256") != p2_sha256:
            raise QueryEmbeddingError("已有 Query cache 与当前 P2 manifest 不一致")
        if result.rows != selected_rows:
            raise QueryEmbeddingError("已有 Query cache 行数与当前 limit 不一致")
        return result
    if output_dir.exists() and config.runtime.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and not progress_path.exists() and any(output_dir.iterdir()):
        raise QueryEmbeddingError("Query embedding 输出非空但缺少 progress.json")

    encoder, actual_device, model_load_seconds = encoder_loader(config.query_embedding)
    dimension = encoder.get_sentence_embedding_dimension()
    if not isinstance(dimension, int) or dimension != config.data_contracts.embedding_dim:
        raise QueryEmbeddingError("Query encoder 维度与冻结 POI BGE 不一致")
    signature = _signature(config, p2_sha256, selected_rows, dimension)
    start_row = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not config.runtime.resume or progress.get("signature") != signature:
            raise QueryEmbeddingError("Query cache 断点与当前请求不兼容")
        start_row = int(progress.get("next_row", -1))
        if not 0 <= start_row <= selected_rows:
            raise QueryEmbeddingError("Query cache 断点行数非法")
        values = np.load(embeddings_path, mmap_mode="r+")
        if values.shape != (selected_rows, dimension):
            raise QueryEmbeddingError("Query cache 断点 shape 非法")
    else:
        values = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.dtype(config.query_embedding.output_dtype),
            shape=(selected_rows, dimension),
        )
        write_json_atomic(
            progress_path,
            {
                "signature": signature,
                "status": "running",
                "next_row": 0,
                "total_rows": selected_rows,
                "updated_at": utc_now(),
            },
        )

    started = time.perf_counter()
    next_row = start_row
    batches_since_checkpoint = 0
    progress_bar = tqdm(
        total=selected_rows,
        initial=start_row,
        desc="QG normalized Query embedding",
        unit="row",
        disable=not config.runtime.show_progress,
    )
    try:
        queries = iter_normalized_queries(
            query_stats_dir, start_row=start_row, stop_row=selected_rows
        )
        for query_buffer in _chunked(queries, config.query_embedding.encode_buffer_size):
            vectors = np.asarray(
                encoder.encode(
                    query_buffer,
                    batch_size=config.query_embedding.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    prompt_name=config.query_embedding.prompt_name,
                ),
                dtype=np.float32,
            )
            if vectors.shape != (len(query_buffer), dimension):
                raise QueryEmbeddingError("Query encoder 输出 shape 非法")
            stop = next_row + len(query_buffer)
            values[next_row:stop] = vectors.astype(values.dtype, copy=False)
            next_row = stop
            progress_bar.update(len(query_buffer))
            batches_since_checkpoint += (
                len(query_buffer) + config.query_embedding.batch_size - 1
            ) // config.query_embedding.batch_size
            if batches_since_checkpoint >= config.query_embedding.checkpoint_interval_batches:
                values.flush()
                write_json_atomic(
                    progress_path,
                    {
                        "signature": signature,
                        "status": "running",
                        "next_row": next_row,
                        "total_rows": selected_rows,
                        "updated_at": utc_now(),
                    },
                    overwrite=True,
                )
                batches_since_checkpoint = 0
        if next_row != selected_rows:
            raise QueryEmbeddingError(f"只编码 {next_row} 行，期望 {selected_rows}")
        values.flush()
    finally:
        progress_bar.close()
        del values
        del encoder
        gc.collect()
    validation = _embedding_validation(embeddings_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "built_at": utc_now(),
        "config_path": str(config.source_path),
        "config_sha256": config.source_sha256,
        "config_signature": config.signature(),
        "source": {
            "query_stats_dir": str(query_stats_dir),
            "query_stats_manifest_sha256": p2_sha256,
            "query_normalization": config.method.query_normalization,
            "full_query_rows": total_rows,
            "selected_query_rows": selected_rows,
            "query_id_range": [0, selected_rows],
            "p2_is_prefix_sample": p2_is_sample,
            "valid_and_test_read": False,
        },
        "model": {
            **asdict(config.query_embedding),
            "model_path": str(config.query_embedding.model_path),
            "actual_device": actual_device,
            "embedding_dim": dimension,
            "model_load_seconds": model_load_seconds,
        },
        "output": {
            "file": embeddings_path.name,
            "shape": [selected_rows, dimension],
            "dtype": config.query_embedding.output_dtype,
            "row_mapping": "row_index equals P2 normalized query_id",
            "sha256": sha256_file(embeddings_path),
        },
        "validation": validation,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "rows_encoded_this_run": selected_rows - start_row,
        },
        "signature": signature,
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        progress_path,
        {
            "signature": signature,
            "status": "completed",
            "next_row": selected_rows,
            "total_rows": selected_rows,
            "updated_at": utc_now(),
        },
        overwrite=True,
    )
    (output_dir / "_SUCCESS").touch()
    return QueryEmbeddingResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        embeddings_path=embeddings_path,
        rows=selected_rows,
        embedding_dim=dimension,
        reused=False,
    )
