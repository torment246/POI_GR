"""Atomic, resumable exact-query encoding with bounded sequential writes."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.adapters.data import D3Query
from qg_prqk.adapters.selection_config import AdapterSelectionConfig
from qg_prqk.adapters.gate import _event, normalize_encoded_queries
from qg_prqk.data.query_embeddings import TextEncoder, load_text_encoder


SCHEMA_VERSION = "qg-prqk-d3-query-chunk-cache-v1"
BUFFER_BYTES = 16 * 1024 * 1024


class QueryEmbeddingCacheError(RuntimeError):
    """Raised when a cache or interrupted prefix fails validation."""


def _norms(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(np.abs(norms - 1.0) > 2e-3):
        raise QueryEmbeddingCacheError("Query cache 存在非有限值、零向量或非单位向量")
    return norms


def _inside_outputs(path: Path, config: AdapterSelectionConfig) -> Path:
    path = path.resolve()
    if not path.is_relative_to(config.base.project_root / "qg_prqk/outputs"):
        raise QueryEmbeddingCacheError("Query cache 及恢复来源必须位于 qg_prqk/outputs")
    return path


def _write_npy(path: Path, shape: tuple[int, int], blocks: Any) -> None:
    """Publish a standard NPY only after sequential data and fsync complete."""

    if path.exists():
        raise QueryEmbeddingCacheError(f"拒绝覆盖已发布 NPY：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    written = 0
    with temporary.open("xb", buffering=BUFFER_BYTES) as handle:
        np.lib.format.write_array_header_1_0(
            handle, {"descr": "<f2", "fortran_order": False, "shape": shape}
        )
        for block in blocks:
            values = np.asarray(block, dtype="<f2", order="C")
            if values.ndim != 2 or values.shape[1] != shape[1]:
                raise QueryEmbeddingCacheError("顺序 NPY 写入维度不一致")
            handle.write(values.tobytes(order="C"))
            written += len(values)
        if written != shape[0]:
            raise QueryEmbeddingCacheError("顺序 NPY 写入行数不一致")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _legacy_prefix(
    source: Path, queries: Sequence[D3Query], config: AdapterSelectionConfig
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Trust only contiguous logged blocks bound to the frozen D3 selection."""

    source = _inside_outputs(source, config)
    manifest_path = source / "internal_holdout/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["source_hashes"]
    expected = {
        "p2_manifest_sha256": config.base.category_config.frozen.p2_manifest_sha256,
        "p2_5_manifest_sha256": config.base.p2_5_manifest_sha256,
        "gate_manifest_sha256": config.gate_manifest_sha256,
    }
    if any(sources.get(key) != value for key, value in expected.items()):
        raise QueryEmbeddingCacheError("中断来源的 P2/P2.5/Gate 不匹配")
    selection_path = source / "d3_full_selection.parquet"
    if sha256_file(selection_path) != sources["full_d3_selection_sha256"]:
        raise QueryEmbeddingCacheError("中断来源 D3 selection 哈希错误")
    table = pq.read_table(selection_path, columns=["query_id", "normalized_query"])
    if table.column(0).to_pylist() != [q.query_id for q in queries] or table.column(
        1
    ).to_pylist() != [q.normalized_query for q in queries]:
        raise QueryEmbeddingCacheError("中断来源 Query ID、文本或行序不匹配")
    log_path = source / "run_log.jsonl"
    completed = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("stage") != "query_encoding_progress":
            continue
        stop = min(
            completed + config.base.query_embedding.encode_buffer_size, len(queries)
        )
        if event.get("rows") != len(queries) or event.get("next_row") != stop:
            raise QueryEmbeddingCacheError("中断来源编码日志不连续")
        completed = stop
    if not completed:
        raise QueryEmbeddingCacheError("中断来源没有已确认的编码块")
    embeddings_path = source / "raw_query_embeddings_d3.npy"
    values = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape
        != (len(queries), config.base.category_config.frozen.poi_embedding_dim)
        or values.dtype != np.float16
    ):
        raise QueryEmbeddingCacheError("中断来源 embedding shape/dtype 错误")
    return (
        values,
        completed,
        {
            "directory": str(source),
            "confirmed_rows": completed,
            "unconfirmed_tail_reused": False,
            "source_embedding_sha256": sha256_file(embeddings_path),
            "source_log_sha256": sha256_file(log_path),
            "source_selection_sha256": sources["full_d3_selection_sha256"],
            "source_holdout_manifest_sha256": sha256_file(manifest_path),
        },
    )


def build_d3_query_cache(
    config: AdapterSelectionConfig,
    queries: Sequence[D3Query],
    *,
    source_dir: Path | None = None,
    encoder_loader: Callable[..., tuple[TextEncoder, str, float]] = load_text_encoder,
) -> tuple[Path, dict[str, Any]]:
    """Resume committed chunks, optionally recover a legacy prefix, and assemble."""

    directory = _inside_outputs(
        config.base.output_dir / "query_cache_exact_full", config
    )
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "run_log.jsonl"
    progress_path = directory / "progress.json"
    output_path = directory / "embeddings.npy"
    dimension = config.base.category_config.frozen.poi_embedding_dim
    chunk_rows = config.base.query_embedding.encode_buffer_size
    if np.dtype(config.base.query_embedding.output_dtype) != np.float16:
        raise QueryEmbeddingCacheError("D3 chunk cache 只接受冻结 float16 编码口径")
    query_hash = hashlib.sha256()
    for query in queries:
        query_hash.update(
            json.dumps(
                [query.query_id, query.normalized_query], ensure_ascii=False
            ).encode("utf-8")
            + b"\n"
        )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "query_id_text_sha256": query_hash.hexdigest(),
        "rows": len(queries),
        "embedding_dim": dimension,
        "dtype": "float16",
        "chunk_rows": chunk_rows,
    }
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("contract") != contract:
            raise QueryEmbeddingCacheError("Query cache 断点与当前配置/Query 顺序不匹配")
    else:
        state = {"contract": contract, "chunks": [], "recovery": None}
        write_json_atomic(progress_path, state)
    completed = 0
    for chunk in state["chunks"]:
        path = directory / chunk["file"]
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_stop = min(completed + chunk_rows, len(queries))
        if (
            chunk["start"] != completed
            or chunk["stop"] != expected_stop
            or values.shape != (expected_stop - completed, dimension)
            or values.dtype != np.float16
            or sha256_file(path) != chunk["sha256"]
        ):
            raise QueryEmbeddingCacheError("已提交 Query chunk 的行序、shape 或 hash 错误")
        _norms(values)
        completed = expected_stop
    if (directory / "_SUCCESS").exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            completed != len(queries)
            or manifest.get("contract") != contract
            or sha256_file(output_path) != manifest["output_sha256"]
        ):
            raise QueryEmbeddingCacheError("完整 Query cache 的行数、配置或 hash 错误")
        published = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if (
            published.shape != (len(queries), dimension)
            or published.dtype != np.float16
        ):
            raise QueryEmbeddingCacheError("完整 Query cache shape/dtype 错误")
        return output_path, manifest

    legacy = None
    legacy_rows = 0
    if source_dir is not None:
        legacy, legacy_rows, recovery = _legacy_prefix(source_dir, queries, config)
        if state["recovery"] not in (None, recovery):
            raise QueryEmbeddingCacheError("Query cache 的恢复来源发生变化")
        state["recovery"] = recovery
        write_json_atomic(progress_path, state, overwrite=True)
    started = time.perf_counter()
    encoder = None
    _event(
        log_path,
        "query_cache_started",
        next_row=completed,
        rows=len(queries),
        recoverable_rows=legacy_rows,
    )
    for start in range(completed, len(queries), chunk_rows):
        stop = min(start + chunk_rows, len(queries))
        if stop <= legacy_rows:
            vectors = np.array(legacy[start:stop], dtype=np.float16, copy=True)
            origin = "validated_interrupted_prefix"
        else:
            if encoder is None:
                _event(log_path, "query_encoder_loading")
                encoder, device, load_seconds = encoder_loader(
                    config.base.query_embedding
                )
                if encoder.get_sentence_embedding_dimension() != dimension:
                    raise QueryEmbeddingCacheError("BGE 编码器维度错误")
                _event(
                    log_path,
                    "query_encoder_loaded",
                    device=device,
                    seconds=load_seconds,
                )
            encoding_started = time.perf_counter()
            vectors = encoder.encode(
                [q.normalized_query for q in queries[start:stop]],
                batch_size=config.base.query_embedding.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name=config.base.query_embedding.prompt_name,
            )
            vectors = normalize_encoded_queries(vectors).astype(np.float16)
            origin = "bge_encoded"
            _event(
                log_path,
                "query_chunk_encoded",
                start=start,
                stop=stop,
                seconds=time.perf_counter() - encoding_started,
            )
        if vectors.shape != (stop - start, dimension):
            raise QueryEmbeddingCacheError("Query encoder 输出 shape 错误")
        norms = _norms(vectors)
        chunk_path = directory / f"chunk_{start:07d}_{stop:07d}.npy"
        # A published file without a journal entry was interrupted before commit.
        if chunk_path.exists():
            os.rename(
                chunk_path,
                chunk_path.with_name(
                    f"{chunk_path.name}.unconfirmed.{uuid.uuid4().hex}"
                ),
            )
        write_started = time.perf_counter()
        _write_npy(chunk_path, vectors.shape, [vectors])
        state["chunks"].append(
            {
                "file": chunk_path.name,
                "start": start,
                "stop": stop,
                "sha256": sha256_file(chunk_path),
                "origin": origin,
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
                "norm_sum": float(norms.astype(np.float64).sum()),
            }
        )
        write_json_atomic(progress_path, state, overwrite=True)
        _event(
            log_path,
            "query_chunk_committed",
            next_row=stop,
            rows=len(queries),
            origin=origin,
            write_seconds=time.perf_counter() - write_started,
        )
    del encoder, legacy
    gc.collect()
    if output_path.exists():
        os.rename(
            output_path,
            output_path.with_name(f"embeddings.unconfirmed.{uuid.uuid4().hex}.npy"),
        )
    _event(log_path, "query_cache_assembling", rows=len(queries))
    _write_npy(
        output_path,
        (len(queries), dimension),
        (
            np.load(directory / c["file"], mmap_mode="r", allow_pickle=False)
            for c in state["chunks"]
        ),
    )
    manifest = {
        **state,
        "status": "completed",
        "built_at": utc_now(),
        "rows": len(queries),
        "embedding_dim": dimension,
        "dtype": "float16",
        "output_sha256": sha256_file(output_path),
        "l2_norm_min": min(c["norm_min"] for c in state["chunks"]),
        "l2_norm_max": max(c["norm_max"] for c in state["chunks"]),
        "l2_norm_mean": sum(c["norm_sum"] for c in state["chunks"]) / len(queries),
        "elapsed_seconds": time.perf_counter() - started,
        "business_validation_read": False,
        "business_test_read": False,
    }
    write_json_atomic(directory / "manifest.json", manifest, overwrite=True)
    (directory / "_SUCCESS").touch()
    _event(
        log_path,
        "query_cache_completed",
        rows=len(queries),
        sha256=manifest["output_sha256"],
    )
    return output_path, manifest
