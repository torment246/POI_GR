"""Build resumable query graphs and depth-specific query views."""

from __future__ import annotations

import gc
import json
import os
import resource
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.pipeline_config import DownstreamConfig
from qg_prqk.data.query_graph_data import (
    D3RawReader,
    MEDIUM_LIMIT,
    VECTOR_CHECK_ROWS,
    QueryGraphDataError,
    QueryGraphInputs,
    graph_metrics,
    read_d3_raw,
    validate_limit,
)
from qg_prqk.adapters.cache import _norms, _write_npy
from qg_prqk.adapters.gate import normalize_encoded_queries
from qg_prqk.data.query_embeddings import load_text_encoder


SCHEMA_VERSION = "qg-prqk-p4-query-graph-v3"

D3Source = np.ndarray | Callable[[int, int], np.ndarray]


def validate_final_payload(
    payload: Mapping[str, Any], config: DownstreamConfig
) -> None:
    """Require FINAL role and its frozen full-training configuration."""
    final = json.loads(
        Path(config.upstream_artifacts["final_adapter_manifest"]["path"]).read_text()
    )
    if (
        payload.get("schema_version") != "qg-prqk-query-adapter-v1"
        or payload.get("role") != "P3A-FULL-FINAL"
        or payload.get("p3a_full_config_signature") != config.upstream.signature()
        or payload.get("p3a_config_signature") != config.upstream.base.signature()
        or payload.get("adapter_config") != asdict(config.upstream.base.adapter)
        or payload.get("embedding_dim")
        != config.upstream.base.category_config.frozen.poi_embedding_dim
        or payload.get("trained_epochs") != final["training"]["fixed_epochs"]
        or payload.get("initial_state_sha256")
        != final["source_hashes"]["initial_state_sha256"]
    ):
        raise QueryGraphDataError("checkpoint 不是冻结 P3A-FULL-FINAL（禁止 Gate/SELECT）")


def load_final_adapter(config: DownstreamConfig) -> Callable[[np.ndarray], np.ndarray]:
    """Load the unchanged FINAL architecture for inference, never optimizer updates."""
    import torch
    from qg_prqk.adapters.model import ResidualQueryAdapter

    payload = torch.load(
        config.final_adapter_path, map_location="cpu", weights_only=True
    )
    validate_final_payload(payload, config)
    device = config.upstream.base.query_embedding.device
    if device != "cuda" or not torch.cuda.is_available():
        raise QueryGraphDataError("真实 P4 要求 CUDA；不得静默回退 CPU")
    adapter = config.upstream.base.adapter
    model = ResidualQueryAdapter(
        payload["embedding_dim"],
        adapter.bottleneck,
        residual_scale=adapter.residual_scale,
        dropout=adapter.dropout,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    def adapt(values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            inputs = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
            return model(inputs).float().cpu().numpy().astype(np.float16)

    return adapt


def _event(stage: str, **values: Any) -> None:
    print(
        json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False),
        flush=True,
    )


def output_directory(
    config: DownstreamConfig, limit: int, gate: str = "sample"
) -> Path:
    validate_limit(limit, gate)
    path = config.output_dir / "query_graph" / f"{gate}_{limit:06d}"
    if not path.resolve().is_relative_to(config.output_dir.resolve()):
        raise QueryGraphDataError("P4 输出越出 512 namespace")
    return path


def _contract(
    config: DownstreamConfig,
    inputs: QueryGraphInputs,
    chunk_rows: int,
    gate: str,
    prefix: QueryGraphPrefix | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "input_signature": (
            inputs.streaming_signature() if gate == "full" else inputs.signature()
        ),
        "query_rows": len(inputs.nodes),
        "embedding_dim": inputs.embedding_dim,
        "dtype": "float16",
        "chunk_rows": chunk_rows,
        "gate": gate,
        "reused_prefix": prefix.source if prefix is not None else None,
        "selection": "first_active_queries_in_p2_5_query_id_order",
        "code": {
            path.relative_to(Path(__file__).parents[1]).as_posix(): sha256_file(path)
            for path in (
                Path(__file__).parent / "query_graph_data.py",
                Path(__file__).parent / "query_graph.py",
                Path(__file__).parents[1] / "commands/query_graph.py",
            )
        },
    }


def _save_table(path: Path, table: pa.Table) -> None:
    if path.exists():
        if not pq.read_table(path).equals(table):
            raise QueryGraphDataError(f"既有图结构与冻结输入不一致：{path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_vectors(path: Path, digest: str, shape: tuple[int, int]) -> np.ndarray:
    if sha256_file(path) != digest:
        raise QueryGraphDataError(f"已提交 embedding 哈希错误：{path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != np.float16:
        raise QueryGraphDataError("P4 embedding shape/dtype 错误")
    for start in range(0, len(values), VECTOR_CHECK_ROWS):
        _norms(values[start : start + VECTOR_CHECK_ROWS])
    return values


def _check_views(
    raw: np.ndarray, views: np.ndarray, depths: np.ndarray, d3_raw: np.ndarray
) -> None:
    if not np.array_equal(raw[depths < 3], views[depths < 3]):
        raise QueryGraphDataError("D1/D2 view 必须与 Raw BGE bit-exact")
    if not np.array_equal(raw[depths == 3], d3_raw[depths == 3]):
        raise QueryGraphDataError("D3 Raw 必须与冻结 cache bit-exact，不得重编码")


def _d3_slice(source: D3Source, start: int, stop: int) -> np.ndarray:
    return source(start, stop) if callable(source) else source[start:stop]


def _check_chunks(
    directory: Path, state: Mapping[str, Any], inputs: QueryGraphInputs, d3_raw: D3Source
) -> int:
    stop = 0
    depths = np.array(inputs.nodes["supervision_depth"])
    for chunk in state["chunks"]:
        expected_stop = min(stop + state["contract"]["chunk_rows"], len(inputs.nodes))
        if (
            chunk["start"] != stop
            or chunk["stop"] != expected_stop
            or expected_stop <= stop
        ):
            raise QueryGraphDataError("P4 cache 断点行序不连续")
        shape = (expected_stop - stop, inputs.embedding_dim)
        for key in ("raw", "view"):
            expected_file = f"{key}_{stop:06d}_{expected_stop:06d}.npy"
            if chunk[key]["file"] != expected_file:
                raise QueryGraphDataError("P4 cache 文件名与行序不符")
        raw = _read_vectors(
            directory / chunk["raw"]["file"], chunk["raw"]["sha256"], shape
        )
        views = _read_vectors(
            directory / chunk["view"]["file"], chunk["view"]["sha256"], shape
        )
        _check_views(
            raw,
            views,
            depths[stop:expected_stop],
            _d3_slice(d3_raw, stop, expected_stop),
        )
        stop = expected_stop
    return stop


def _preserve_uncommitted(path: Path) -> None:
    if path.exists():
        destination = path.with_name(f"{path.name}.unconfirmed.{uuid.uuid4().hex}")
        os.replace(path, destination)
        _event(
            "uncommitted_file_preserved", source=str(path), destination=str(destination)
        )


def _embedding_metrics(
    raw: np.ndarray, views: np.ndarray, depths: np.ndarray
) -> dict[str, Any]:
    result = {}
    for depth in (1, 2, 3):
        selected = depths == depth
        if not np.any(selected):
            result[f"D{depth}"] = {"rows": 0, "cosine_drift_mean": None}
            continue
        indices = np.flatnonzero(selected)
        drift = np.empty(len(indices), dtype=np.float32)
        bit_exact = True
        for start in range(0, len(indices), VECTOR_CHECK_ROWS):
            rows = indices[start : start + VECTOR_CHECK_ROWS]
            a = normalize_encoded_queries(raw[rows])
            b = normalize_encoded_queries(views[rows])
            drift[start : start + len(rows)] = np.clip(1 - np.sum(a * b, axis=1), 0, 2)
            bit_exact = bit_exact and np.array_equal(raw[rows], views[rows])
        result[f"D{depth}"] = {
            "rows": int(selected.sum()),
            "raw_view_bit_exact": bool(bit_exact),
            "cosine_drift_mean": float(drift.mean()),
            "cosine_drift_max": float(drift.max()),
        }
    return result


@dataclass(frozen=True)
class QueryGraphPrefix:
    raw: np.ndarray
    views: np.ndarray
    source: dict[str, Any]


def load_frozen_prefix(
    config: DownstreamConfig,
    inputs: QueryGraphInputs,
    d3_raw: D3Source,
    manifest_path: Path,
    expected_sha256: str,
    *,
    source_gate: str,
) -> QueryGraphPrefix:
    """Validate a pinned completed gate before reusing its exact prefix."""
    if source_gate not in ("sample", "medium"):
        raise QueryGraphDataError("复用前缀只能来自 sample 或 medium")
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to((config.output_dir / "query_graph").resolve()):
        raise QueryGraphDataError("复用前缀必须位于同一 512 Query 图目录")
    directory = manifest_path.parent
    if sha256_file(manifest_path) != expected_sha256:
        raise QueryGraphDataError("复用前缀 manifest SHA256 不匹配")
    manifest = json.loads(manifest_path.read_text())
    marker = json.loads((directory / "_SUCCESS").read_text())
    contract = manifest["contract"]
    rows = contract["query_rows"]
    if (
        marker.get("manifest_sha256") != expected_sha256
        or manifest.get("schema_version") not in (
            "qg-prqk-p4-query-graph-sample-v1",
            "qg-prqk-p4-query-graph-v2",
            SCHEMA_VERSION,
        )
        or manifest.get("phase") != f"P4-CAT-{source_gate.upper()}"
        or manifest.get("status") != "completed"
        or manifest.get("is_full") is not False
        or contract.get("config_signature") != config.signature()
        or manifest.get("query_view_policy") != config.query_view_policy
        or manifest.get("codebook_sizes") != list(config.codebook_sizes)
        or (
            source_gate == "sample"
            and not 1 <= rows <= min(1000, len(inputs.nodes))
        )
        or (
            source_gate == "medium"
            and rows != min(MEDIUM_LIMIT, len(inputs.nodes))
        )
        or contract.get("embedding_dim") != inputs.embedding_dim
        or contract.get("selection") != "first_active_queries_in_p2_5_query_id_order"
        or any(inputs.source_files.get(k) != v for k, v in manifest["sources"].items())
    ):
        raise QueryGraphDataError("复用前缀的来源/配置/角色/行数不匹配")
    expected_tables = {
        "query_nodes.parquet": inputs.nodes.slice(0, rows),
        "query_poi_edges.parquet": inputs.edges.filter(
            pc.less(inputs.edges["node_row"], rows)
        ),
        "false_negative_mask.parquet": inputs.false_negatives.filter(
            pc.less(inputs.false_negatives["node_row"], rows)
        ),
    }
    if set(manifest["artifacts"]) != set(expected_tables) | {
        "config_resolved.json",
        "raw_query_embeddings.npy",
        "query_embeddings.npy",
    }:
        raise QueryGraphDataError("复用前缀 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        if entry["file"] != name or sha256_file(directory / name) != entry["sha256"]:
            raise QueryGraphDataError("复用前缀 artifact 哈希错误")
    for name, table in expected_tables.items():
        if not pq.read_table(directory / name).equals(table):
            raise QueryGraphDataError("复用前缀的 Query/POI/文本/图不是当前精确前缀")
    if (
        json.loads((directory / "config_resolved.json").read_text())
        != config.resolved_payload()
    ):
        raise QueryGraphDataError("复用前缀 resolved config 不匹配")
    prefix_inputs = replace(
        inputs,
        nodes=expected_tables["query_nodes.parquet"],
        edges=expected_tables["query_poi_edges.parquet"],
        false_negatives=expected_tables["false_negative_mask.parquet"],
    )
    state = json.loads((directory / "progress.json").read_text())
    if state["contract"] != contract or state["chunks"] != manifest["chunks"]:
        raise QueryGraphDataError("复用前缀的断点与 manifest 不匹配")
    if _check_chunks(directory, state, prefix_inputs, d3_raw) != rows:
        raise QueryGraphDataError("复用前缀分块不完整")
    matrices = []
    for key, filename in (
        ("raw", "raw_query_embeddings.npy"),
        ("view", "query_embeddings.npy"),
    ):
        values = _read_vectors(
            directory / filename,
            manifest["artifacts"][filename]["sha256"],
            (rows, inputs.embedding_dim),
        )
        for chunk in state["chunks"]:
            block = np.load(
                directory / chunk[key]["file"], mmap_mode="r", allow_pickle=False
            )
            if not np.array_equal(values[chunk["start"] : chunk["stop"]], block):
                raise QueryGraphDataError("复用前缀合并向量与分块不匹配")
        matrices.append(values)
    return QueryGraphPrefix(
        *matrices,
        source={
            "manifest_path": str(manifest_path),
            "manifest_sha256": expected_sha256,
            "rows": rows,
        },
    )


def load_sample_prefix(
    config: DownstreamConfig,
    inputs: QueryGraphInputs,
    d3_raw: D3Source,
    manifest_path: Path,
    expected_sha256: str,
) -> QueryGraphPrefix:
    """Compatibility wrapper for the accepted sample gate."""
    return load_frozen_prefix(
        config,
        inputs,
        d3_raw,
        manifest_path,
        expected_sha256,
        source_gate="sample",
    )


def build_query_graph(
    config: DownstreamConfig,
    inputs: QueryGraphInputs,
    *,
    resume: bool = False,
    gate: str = "sample",
    reuse_sample: Path | None = None,
    reuse_sample_sha256: str | None = None,
    reuse_medium: Path | None = None,
    reuse_medium_sha256: str | None = None,
    encoder_loader: Callable = load_text_encoder,
    adapter_loader: Callable = load_final_adapter,
) -> dict[str, Any]:
    """Commit raw/view chunks atomically, then publish one bounded gate."""
    directory = output_directory(config, len(inputs.nodes), gate)
    if (reuse_sample is None) != (reuse_sample_sha256 is None):
        raise QueryGraphDataError("复用 sample 路径和 SHA256 必须同时提供")
    if (reuse_medium is None) != (reuse_medium_sha256 is None):
        raise QueryGraphDataError("复用 medium 路径和 SHA256 必须同时提供")
    if gate == "medium" and reuse_sample is None:
        raise QueryGraphDataError("medium 必须指定已验收 sample 及其 SHA256")
    if gate == "full" and reuse_medium is None:
        raise QueryGraphDataError("full 必须指定已验收 medium 及其 SHA256")
    if gate == "sample" and (reuse_sample is not None or reuse_medium is not None):
        raise QueryGraphDataError("sample 不得复用其他 gate")
    if gate == "medium" and reuse_medium is not None:
        raise QueryGraphDataError("medium 只能复用 sample")
    if gate == "full" and reuse_sample is not None:
        raise QueryGraphDataError("full 只能复用 medium")
    if directory.exists() and not resume:
        raise QueryGraphDataError(f"输出已存在；只有 --resume 可验证复用：{directory}")
    started = time.perf_counter()
    _event("p4_d3_cache_validation_started")
    d3_raw: D3Source = D3RawReader(inputs) if gate == "full" else read_d3_raw(inputs)
    _event(
        "p4_d3_cache_validated",
        rows=int(sum(inputs.nodes["supervision_depth"].to_numpy() == 3)),
    )
    prefix = None
    if reuse_sample:
        prefix = load_frozen_prefix(
            config,
            inputs,
            d3_raw,
            reuse_sample,
            reuse_sample_sha256,
            source_gate="sample",
        )
    elif reuse_medium:
        prefix = load_frozen_prefix(
            config,
            inputs,
            d3_raw,
            reuse_medium,
            reuse_medium_sha256,
            source_gate="medium",
        )
    prefix_rows = len(prefix.raw) if prefix else 0
    encoding = config.upstream.base.query_embedding
    chunk_rows = (
        encoding.encode_buffer_size
        if gate in ("medium", "full")
        else encoding.batch_size
    )
    contract = _contract(config, inputs, chunk_rows, gate, prefix)
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        if state.get("contract") != contract:
            raise QueryGraphDataError("P4 断点的配置、数据或源码签名变化")
    else:
        if any(directory.iterdir()):
            raise QueryGraphDataError("非空目录缺少 progress，不能推测恢复状态")
        state = {"contract": contract, "chunks": []}
        write_json_atomic(progress_path, state)
    if (directory / "_SUCCESS").exists():
        return validate_query_graph(
            config, inputs, gate=gate, d3_raw=d3_raw, prefix=prefix
        )
    completed = _check_chunks(directory, state, inputs, d3_raw)
    _save_table(directory / "query_nodes.parquet", inputs.nodes)
    _save_table(directory / "query_poi_edges.parquet", inputs.edges)
    _save_table(directory / "false_negative_mask.parquet", inputs.false_negatives)
    resolved_path = directory / "config_resolved.json"
    if resolved_path.exists():
        if json.loads(resolved_path.read_text()) != config.resolved_payload():
            raise QueryGraphDataError("P4 resolved config 与断点不一致")
    else:
        write_json_atomic(resolved_path, config.resolved_payload())
    texts = inputs.nodes["normalized_query"].to_pylist()
    depths = inputs.nodes["supervision_depth"].to_numpy(zero_copy_only=False)
    encoder, adapter = None, None
    newly_encoded = 0
    _event(
        "p4_gate_started",
        gate=gate,
        rows=len(inputs.nodes),
        resumed_rows=completed,
        reused_prefix_rows=prefix_rows,
    )
    for start in range(completed, len(inputs.nodes), chunk_rows):
        stop = min(start + chunk_rows, len(inputs.nodes))
        raw = _d3_slice(d3_raw, start, stop).copy()
        inherited = max(0, min(stop, prefix_rows) - start)
        if inherited:
            raw[:inherited] = prefix.raw[start : start + inherited]
        pending = np.arange(start, stop) >= prefix_rows
        raw_indices = np.flatnonzero((depths[start:stop] < 3) & pending)
        adapted_indices = np.flatnonzero((depths[start:stop] == 3) & pending)
        if len(raw_indices):
            if encoder is None:
                _event("p4_bge_loading")
                encoder, device, load_seconds = encoder_loader(
                    config.upstream.base.query_embedding
                )
                if encoder.get_sentence_embedding_dimension() != inputs.embedding_dim:
                    raise QueryGraphDataError("BGE Query 编码维数错误")
                _event("p4_bge_loaded", device=device, seconds=load_seconds)
            encoding = config.upstream.base.query_embedding
            values = encoder.encode(
                [texts[start + int(i)] for i in raw_indices],
                batch_size=encoding.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name=encoding.prompt_name,
            )
            if values.shape != (len(raw_indices), inputs.embedding_dim):
                raise QueryGraphDataError("BGE Query 返回 shape 错误")
            raw[raw_indices] = normalize_encoded_queries(values).astype(np.float16)
            newly_encoded += len(raw_indices)
        views = raw.copy()
        if inherited:
            views[:inherited] = prefix.views[start : start + inherited]
        if len(adapted_indices):
            if adapter is None:
                adapter = adapter_loader(config)
                _event("p4_final_adapter_loaded")
            for offset in range(0, len(adapted_indices), encoding.batch_size):
                indices = adapted_indices[offset : offset + encoding.batch_size]
                adapted = adapter(raw[indices])
                if adapted.shape != (len(indices), inputs.embedding_dim):
                    raise QueryGraphDataError("FINAL Adapter 返回 shape 错误")
                views[indices] = adapted
        _norms(raw)
        _norms(views)
        entry = {"start": start, "stop": stop}
        for key, vectors in (("raw", raw), ("view", views)):
            path = directory / f"{key}_{start:06d}_{stop:06d}.npy"
            _preserve_uncommitted(path)
            _write_npy(path, vectors.shape, [vectors])
            entry[key] = _artifact(path)
        state["chunks"].append(entry)
        write_json_atomic(progress_path, state, overwrite=True)
        _event("p4_chunk_committed", next_row=stop, rows=len(inputs.nodes))
    del encoder, adapter
    gc.collect()
    for key, filename in (
        ("raw", "raw_query_embeddings.npy"),
        ("view", "query_embeddings.npy"),
    ):
        path = directory / filename
        _preserve_uncommitted(path)
        _write_npy(
            path,
            (len(inputs.nodes), inputs.embedding_dim),
            (
                np.load(directory / c[key]["file"], mmap_mode="r", allow_pickle=False)
                for c in state["chunks"]
            ),
        )
    raw = np.load(
        directory / "raw_query_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    views = np.load(
        directory / "query_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    metrics = {
        **graph_metrics(inputs, bounded_memory=gate == "full"),
        "embedding_views": _embedding_metrics(raw, views, depths),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "built_at": utc_now(),
        "phase": f"P4-CAT-{gate.upper()}",
        "is_sample": gate == "sample",
        "is_full": gate == "full",
        "contract": contract,
        "chunks": state["chunks"],
        "metrics": metrics,
        "query_view_policy": config.query_view_policy,
        "codebook_sizes": list(config.codebook_sizes),
        "poi_nodes": {
            "rows": inputs.poi_rows,
            "row_mapping": "frozen active P2.5 category_mapping",
        },
        "sources": dict(inputs.source_files),
        "source_access": {
            "raw_train_read": False,
            "business_validation_read": False,
            "business_test_read": False,
            "poi_embedding_values_read": False,
        },
        "representation": {
            "global_direction_removal_applied": False,
            "query_residual_applied": False,
            "adapter_model_updated": False,
            "d3_raw_reencoded": False,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "resumed_rows": completed,
            "newly_encoded_d1_d2_rows": newly_encoded,
            "reused_prefix_rows": prefix_rows,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "artifacts": {
            name: _artifact(directory / name)
            for name in (
                "query_nodes.parquet",
                "query_poi_edges.parquet",
                "false_negative_mask.parquet",
                "config_resolved.json",
                "raw_query_embeddings.npy",
                "query_embeddings.npy",
            )
        },
        "next_status": f"HOLD_FOR_P4_{gate.upper()}_REVIEW",
    }
    _preserve_uncommitted(directory / "manifest.json")
    write_json_atomic(directory / "manifest.json", manifest)
    # The success marker is published last and pins the completed manifest.
    write_json_atomic(
        directory / "_SUCCESS",
        {"manifest_sha256": sha256_file(directory / "manifest.json")},
    )
    return validate_query_graph(config, inputs, gate=gate, d3_raw=d3_raw, prefix=prefix)


def validate_query_graph(
    config: DownstreamConfig,
    inputs: QueryGraphInputs,
    *,
    gate: str = "sample",
    d3_raw: D3Source | None = None,
    prefix: QueryGraphPrefix | None = None,
) -> dict[str, Any]:
    """Independently check tables, hashes, chunk assembly and Query view routing."""
    directory = output_directory(config, len(inputs.nodes), gate)
    marker = json.loads((directory / "_SUCCESS").read_text())
    manifest_path = directory / "manifest.json"
    if sha256_file(manifest_path) != marker["manifest_sha256"]:
        raise QueryGraphDataError("P4 manifest 与成功标记哈希不一致")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("is_sample") is not (gate == "sample")
        or manifest.get("is_full") is not (gate == "full")
        or manifest.get("query_view_policy") != config.query_view_policy
        or manifest.get("codebook_sizes") != list(config.codebook_sizes)
        or manifest.get("phase") != f"P4-CAT-{gate.upper()}"
        or manifest.get("next_status") != f"HOLD_FOR_P4_{gate.upper()}_REVIEW"
    ):
        raise QueryGraphDataError("P4 manifest schema/状态/view policy 错误")
    if d3_raw is None:
        d3_raw = D3RawReader(inputs) if gate == "full" else read_d3_raw(inputs)
    reuse = manifest["contract"].get("reused_prefix")
    if prefix is None and reuse:
        prefix = load_frozen_prefix(
            config,
            inputs,
            d3_raw,
            Path(reuse["manifest_path"]),
            reuse["manifest_sha256"],
            source_gate="sample" if gate == "medium" else "medium",
        )
    if gate in ("medium", "full") and prefix is None:
        raise QueryGraphDataError(f"{gate} 缺少前缀复用来源")
    encoding = config.upstream.base.query_embedding
    contract = _contract(
        config,
        inputs,
        encoding.encode_buffer_size
        if gate in ("medium", "full")
        else encoding.batch_size,
        gate,
        prefix,
    )
    state = json.loads((directory / "progress.json").read_text())
    if (
        manifest["contract"] != contract
        or state.get("contract") != contract
        or manifest["chunks"] != state.get("chunks")
        or manifest["sources"] != dict(inputs.source_files)
    ):
        raise QueryGraphDataError("P4 配置/数据/源码/断点签名不符")
    expected_names = {
        "query_nodes.parquet",
        "query_poi_edges.parquet",
        "false_negative_mask.parquet",
        "config_resolved.json",
        "raw_query_embeddings.npy",
        "query_embeddings.npy",
    }
    if set(manifest["artifacts"]) != expected_names:
        raise QueryGraphDataError("P4 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        if entry["file"] != name or sha256_file(directory / name) != entry["sha256"]:
            raise QueryGraphDataError(f"P4 artifact 哈希错误：{name}")
    for name, expected in (
        ("query_nodes.parquet", inputs.nodes),
        ("query_poi_edges.parquet", inputs.edges),
        ("false_negative_mask.parquet", inputs.false_negatives),
    ):
        if not pq.read_table(directory / name).equals(expected):
            raise QueryGraphDataError("P4 节点/边/FN 与冻结输入不一致")
    if (
        json.loads((directory / "config_resolved.json").read_text())
        != config.resolved_payload()
    ):
        raise QueryGraphDataError("P4 resolved config 与冻结配置不一致")
    if _check_chunks(directory, state, inputs, d3_raw) != len(inputs.nodes):
        raise QueryGraphDataError("P4 已提交 cache 行数不完整")
    for key, filename in (
        ("raw", "raw_query_embeddings.npy"),
        ("view", "query_embeddings.npy"),
    ):
        values = _read_vectors(
            directory / filename,
            manifest["artifacts"][filename]["sha256"],
            (len(inputs.nodes), inputs.embedding_dim),
        )
        for chunk in state["chunks"]:
            original = np.load(
                directory / chunk[key]["file"], mmap_mode="r", allow_pickle=False
            )
            if not np.array_equal(values[chunk["start"] : chunk["stop"]], original):
                raise QueryGraphDataError("完整 embedding 与原子提交块逐值不一致")
    raw = np.load(
        directory / "raw_query_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    views = np.load(
        directory / "query_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    if prefix and (
        not np.array_equal(raw[: len(prefix.raw)], prefix.raw)
        or not np.array_equal(views[: len(prefix.views)], prefix.views)
    ):
        raise QueryGraphDataError(f"{gate} 的前缀向量未逐值复用")
    metrics = {
        **graph_metrics(inputs, bounded_memory=gate == "full"),
        "embedding_views": _embedding_metrics(
            raw, views, np.array(inputs.nodes["supervision_depth"])
        ),
    }
    if metrics != manifest["metrics"]:
        raise QueryGraphDataError("P4 metrics 与独立复算不一致")
    return {
        "status": "validated",
        "phase": f"P4-CAT-{gate.upper()}",
        "output_dir": str(directory),
        "manifest_sha256": sha256_file(manifest_path),
        "metrics": metrics,
        "next_status": f"HOLD_FOR_P4_{gate.upper()}_REVIEW",
    }
