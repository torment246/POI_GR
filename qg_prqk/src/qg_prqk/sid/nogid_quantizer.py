"""Train and validate the S1/S2-parent no-GID S3 variant."""

from __future__ import annotations

import gc
import json
import os
import resource
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.base_quantizer import _distribution_metrics, projection_residual
from qg_prqk.sid.relational_quantizer import _graph_metrics, _to_device_normalized_float32
from qg_prqk.sid.local_data import LocalCodebookDataError
from qg_prqk.sid.geo import GEO_FEATURE_DIM
from qg_prqk.sid.nogid_config import NoGIDCodebookConfig
from qg_prqk.sid.nogid_data import HARD_EDGE_SCHEMA, POI_METADATA_SCHEMA, NoGIDCodebookInputs, load_nogid_codebook_inputs
from qg_prqk.sid.nogid_geo import s1_s2_parent_keys
from qg_prqk.sid.local_quantizer import _hard_metrics, _prefix_metrics, _proxy_subset_metrics, _s3_graph, fit_s3


SCHEMA_VERSION = "qg-prqk-p7-nogid-s1s2-parent-s3-v1"
DEFAULT_CHUNK_ROWS = 8192


def _torch():
    try:
        import torch
    except ImportError as error:
        raise LocalCodebookDataError("no-GID S3 需要 PyTorch") from error
    return torch


def _event(stage: str, **values: Any) -> None:
    print(json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False), flush=True)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise LocalCodebookDataError(f"no-GID 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_tensor_npy(path: Path, values, *, dtype: np.dtype, chunk_rows: int) -> None:
    if path.exists():
        raise LocalCodebookDataError(f"no-GID 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        target_dtype = np.dtype(dtype)
        with temporary.open("wb") as handle:
            np.lib.format.write_array_header_2_0(
                handle,
                {"descr": np.lib.format.dtype_to_descr(target_dtype), "fortran_order": False, "shape": tuple(values.shape)},
            )
            for start in range(0, len(values), chunk_rows):
                stop = min(start + chunk_rows, len(values))
                block = np.ascontiguousarray(values[start:stop].detach().float().cpu().numpy(), dtype=target_dtype)
                handle.write(block.tobytes(order="C"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        result.update(shape=list(values.shape), dtype=str(values.dtype))
    elif path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        result.update(rows=parquet.metadata.num_rows, schema=str(parquet.schema_arrow))
    return result


def output_directory(config: NoGIDCodebookConfig, gate: str) -> Path:
    if gate not in ("sample", "full") or gate not in config.gates:
        raise LocalCodebookDataError("no-GID S3 只开放 sample/full")
    settings = config.gates[gate]
    output = config.output_dir / f"{gate}_{int(settings['d3_query_rows']):06d}q_{int(settings['poi_rows']):06d}p"
    if not output.resolve().is_relative_to(config.output_dir.resolve()):
        raise LocalCodebookDataError("no-GID 输出越出冻结 namespace")
    return output


def next_status_for_gate(gate: str) -> str:
    statuses = {"sample": "READY_FOR_P7_NOGID_FULL", "full": "HOLD_FOR_NOGID_S3_REVIEW"}
    try:
        return statuses[gate]
    except KeyError as error:
        raise LocalCodebookDataError(f"未知 no-GID gate：{gate}") from error


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/base_quantizer.py": directory / "base_quantizer.py",
        "sid/relational_quantizer.py": directory / "relational_quantizer.py",
        "sid/local_data.py": directory / "local_data.py",
        "sid/local_quantizer.py": directory / "local_quantizer.py",
        "sid/nogid_config.py": directory / "nogid_config.py",
        "sid/nogid_geo.py": directory / "nogid_geo.py",
        "sid/nogid_data.py": directory / "nogid_data.py",
        "sid/nogid_quantizer.py": directory / "nogid_quantizer.py",
        "commands/local_codebook_nogid.py": (
            directory.parent / "commands/local_codebook_nogid.py"
        ),
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _expected_artifacts() -> Iterable[str]:
    yield from (
        "config_resolved.json",
        "continuous_geo_standardization.json",
        "selected_poi_rows.npy",
        "selected_d3_query_rows.npy",
        "poi_parent_s1_s2.npy",
        "poi_parent_keys.npy",
        "poi_continuous_geo_features.npy",
        "poi_singleton_parent_mask.npy",
        "poi_nogid_metadata.parquet",
        "query_poi_edges_s3.parquet",
        "false_negative_mask_s3.parquet",
        "hard_entity_edges.parquet",
        "poi_codebook_s3.npy",
        "query_codebook_s3.npy",
        "continuous_geo_codebook_s3.npy",
        "poi_assignments_s3.npy",
        "query_assignments_s3.npy",
        "poi_sid_s1_s2_s3.npy",
        "query_sid_s1_s2_s3.npy",
        "poi_residual_after_s3.npy",
        "query_residual_after_s3_d3.npy",
        "metrics.json",
    )


def build_nogid_codebook(
    config: NoGIDCodebookConfig,
    inputs: NoGIDCodebookInputs,
    *,
    gate: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Train and publish immutable S3 artifacts without reading or emitting GID."""
    started = time.perf_counter()
    torch = _torch()
    if device != "cuda" or not torch.cuda.is_available():
        raise LocalCodebookDataError("no-GID S3 要求 CUDA；不得静默回退 CPU")
    if chunk_rows <= 0:
        raise LocalCodebookDataError("no-GID chunk_rows 必须为正整数")
    output = output_directory(config, gate)
    if output.exists():
        raise LocalCodebookDataError(f"no-GID overwrite=false：{output}")
    output.mkdir(parents=True)
    write_json_atomic(output / "config_resolved.json", config.resolved_payload())
    write_json_atomic(output / "continuous_geo_standardization.json", dict(inputs.geo_standardization))
    _atomic_npy(output / "selected_poi_rows.npy", inputs.selected_poi_rows)
    _atomic_npy(output / "selected_d3_query_rows.npy", inputs.d3_query_rows)
    prefix = np.asarray(inputs.p6_poi_assignments_s1_s2, dtype=np.int32)
    _atomic_npy(output / "poi_parent_s1_s2.npy", prefix)
    _atomic_npy(output / "poi_parent_keys.npy", inputs.parent_groups)
    _atomic_npy(output / "poi_continuous_geo_features.npy", inputs.geo_features)
    _atomic_npy(output / "poi_singleton_parent_mask.npy", inputs.singleton_mask)
    pq.write_table(inputs.poi_metadata, output / "poi_nogid_metadata.parquet", compression="zstd")
    pq.write_table(inputs.s3_edges, output / "query_poi_edges_s3.parquet", compression="zstd")
    pq.write_table(inputs.false_negative_mask, output / "false_negative_mask_s3.parquet", compression="zstd")
    pq.write_table(inputs.hard_edges, output / "hard_entity_edges.parquet", compression="zstd")

    target = torch.device(device)
    torch.cuda.reset_peak_memory_stats(target)
    poi_values = _to_device_normalized_float32(inputs.poi_residual_after_s2, device=target, chunk_rows=chunk_rows)
    query_values = _to_device_normalized_float32(inputs.query_residual_after_s2, device=target, chunk_rows=chunk_rows)
    _event(
        "p7_nogid_s3_started",
        gate=gate,
        poi_rows=len(poi_values),
        d3_query_rows=len(query_values),
        hard_edge_rows=len(inputs.hard_edges),
    )
    poi_book, query_book, geo_book, poi_assignment, query_assignment, fit = fit_s3(
        inputs,  # type: ignore[arg-type]
        poi_values,
        query_values,
        torch.from_numpy(inputs.initial_poi_codebook_s3).to(target),
        settings=config.base.base.resolved_payload()["s3"],
        prqk=config.base.prqk,
        warmup=config.base.base.resolved_payload()["warmup"],
        local_refinement=config.local_refinement,
        chunk_rows=chunk_rows,
    )
    poi_s3 = poi_assignment.detach().cpu().numpy().astype(np.int32)
    d3_query_s3 = query_assignment.detach().cpu().numpy().astype(np.int32)
    full_query_s3 = np.full(len(inputs.p6_query_nodes), -1, dtype=np.int32)
    full_query_s3[inputs.d3_query_rows] = d3_query_s3
    poi_sid = np.column_stack((prefix, poi_s3)).astype(np.int32)
    query_sid = np.column_stack((np.asarray(inputs.p6_query_assignments_s1_s2), full_query_s3)).astype(np.int32)
    poi_after_s3, poi_residual_metrics = projection_residual(poi_values, poi_book, poi_assignment)
    query_after_s3, query_residual_metrics = projection_residual(query_values, query_book, query_assignment)
    _atomic_npy(output / "poi_codebook_s3.npy", poi_book.detach().cpu().numpy().astype(np.float32))
    _atomic_npy(output / "query_codebook_s3.npy", query_book.detach().cpu().numpy().astype(np.float32))
    _atomic_npy(output / "continuous_geo_codebook_s3.npy", geo_book.detach().cpu().numpy().astype(np.float32))
    _atomic_npy(output / "poi_assignments_s3.npy", poi_s3)
    _atomic_npy(output / "query_assignments_s3.npy", full_query_s3)
    _atomic_npy(output / "poi_sid_s1_s2_s3.npy", poi_sid)
    _atomic_npy(output / "query_sid_s1_s2_s3.npy", query_sid)
    _atomic_tensor_npy(output / "poi_residual_after_s3.npy", poi_after_s3, dtype=np.float16, chunk_rows=chunk_rows)
    _atomic_tensor_npy(output / "query_residual_after_s3_d3.npy", query_after_s3, dtype=np.float16, chunk_rows=chunk_rows)

    graph_result = _graph_metrics(_s3_graph(inputs), poi_s3, d3_query_s3)  # type: ignore[arg-type]
    hard_result = _hard_metrics(inputs, poi_s3)  # type: ignore[arg-type]
    _, parent_counts = np.unique(inputs.parent_groups, return_counts=True)
    metrics = {
        "fit": fit,
        "poi_assignment": _distribution_metrics(poi_s3, 512),
        "query_assignment": _distribution_metrics(d3_query_s3, 512),
        "conditional_s3_accuracy": graph_result,
        "hard_entity_collision": hard_result,
        "semantic_distortion": {"poi": fit["final_poi_content_mean"], "query": fit["final_query_content_mean"]},
        "continuous_geo_distortion": fit["final_geo_distortion_mean"],
        "poi_residual": poi_residual_metrics,
        "query_residual": query_residual_metrics,
        "parent_groups": {
            "definition": ["s1", "s2"],
            "count": int(len(parent_counts)),
            "singleton_groups": int(np.count_nonzero(parent_counts == 1)),
            "singleton_poi_rows": int(inputs.singleton_mask.sum()),
            "max_size": int(parent_counts.max()),
        },
        "bucket_metrics": {"sid3": _prefix_metrics(poi_sid)},
        "category_path_proxy_subsets": _proxy_subset_metrics(inputs, poi_s3),  # type: ignore[arg-type]
        "structural_gate": {
            "s1_s2_frozen": True,
            "d3_only": True,
            "parent_key": ["s1", "s2"],
            "gid_read": False,
            "gid_computed": False,
            "gid_artifact_emitted": False,
            "continuous_geo_used_only_in_s3": True,
            "category_classification_cost_used_in_s3": False,
            "false_negative_pairs_excluded": True,
            "all_poi_codes_active": int(np.unique(poi_s3).size) == 512,
            "sample_role": "CODE_DATA_GPU_ARTIFACT_SMOKE_ONLY" if gate == "sample" else None,
            "formal_metric_basis": gate == "full",
        },
    }
    write_json_atomic(output / "metrics.json", metrics)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "config_sha256": config.source_sha256,
        "gate": gate,
        "poi_rows": len(inputs.selected_poi_rows),
        "p6_query_rows": len(inputs.p6_query_nodes),
        "d3_query_rows": len(inputs.d3_query_rows),
        "s3_edge_rows": len(inputs.s3_edges),
        "embedding_dim": inputs.embedding_dim,
        "codebook_size": 512,
        "parent_key": ["s1", "s2"],
        "final_sid_positions": ["s1", "s2", "s3"],
        "gid_used": False,
        "algorithm": {
            "prqk": dict(config.base.prqk),
            "s3": dict(config.base.base.resolved_payload()["s3"]),
            "warmup": dict(config.base.base.resolved_payload()["warmup"]),
            "hard_graph": dict(config.hard_graph),
            "continuous_geo": dict(config.continuous_geo),
            "local_refinement": dict(config.local_refinement),
            "initialization": "P5_A0_S3_CODEBOOK_AND_ASSIGNMENT",
            "upstream_endpoint": "FROZEN_P6_S1_S2",
        },
        "source_hashes": dict(inputs.source_hashes),
        "gate_source": dict(config.gates[gate]["p6_manifest"]),
        "gate_sequence": config.authorization["gate_sequence"],
        "chunk_rows": chunk_rows,
        "code": _code_files(),
        "business_validation_query_read": False,
        "business_test_query_read": False,
    }
    artifact_names = list(_expected_artifacts())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": f"P7-CAT-NOGID-{gate.upper()}",
        "role": "S3_D3_QUERY_S1S2_PARENT_CONTINUOUS_GEO_NO_GID",
        "built_at": utc_now(),
        "contract": contract,
        "metrics": metrics,
        "source_access": {
            "p6_frozen_s1_s2_read": True,
            "p5_s3_initialization_read": True,
            "p4_train_s3_graph_read": True,
            "active_poi_metadata_read": True,
            "gid_read_or_computed": False,
            "business_validation_query_read": False,
            "business_test_query_read": False,
        },
        "artifacts": {name: _artifact(output / name) for name in artifact_names},
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(target) / (1024 * 1024),
        },
        "next_status": next_status_for_gate(gate),
    }
    write_json_atomic(output / "manifest.json", manifest)
    validate_nogid_codebook(config, gate=gate, require_success_marker=False)
    write_json_atomic(output / "_SUCCESS", {"manifest_sha256": sha256_file(output / "manifest.json")})
    _event(
        "p7_nogid_s3_completed",
        gate=gate,
        hard_iterations=fit["hard_iterations"],
        weighted_s3_accuracy=graph_result["weighted_agreement"],
        distinct_sid=metrics["bucket_metrics"]["sid3"]["distinct"],
        next_status=next_status_for_gate(gate),
    )
    del poi_values, query_values, poi_after_s3, query_after_s3, poi_book, query_book, geo_book
    gc.collect()
    torch.cuda.empty_cache()
    return validate_nogid_codebook(config, gate=gate)


def validate_nogid_codebook(
    config: NoGIDCodebookConfig,
    *,
    gate: str,
    require_success_marker: bool = True,
) -> dict[str, Any]:
    """Validate no-GID artifacts, frozen S1/S2 prefixes, and stop boundary."""
    output = output_directory(config, gate)
    manifest_path = output / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if require_success_marker:
        marker = json.loads((output / "_SUCCESS").read_text(encoding="utf-8"))
        if marker.get("manifest_sha256") != manifest_sha:
            raise LocalCodebookDataError("no-GID manifest 与 _SUCCESS 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != f"P7-CAT-NOGID-{gate.upper()}"
        or manifest.get("role") != "S3_D3_QUERY_S1S2_PARENT_CONTINUOUS_GEO_NO_GID"
        or manifest.get("contract", {}).get("config_signature") != config.signature()
        or manifest.get("contract", {}).get("gid_used") is not False
        or manifest.get("contract", {}).get("final_sid_positions") != ["s1", "s2", "s3"]
        or manifest.get("next_status") != next_status_for_gate(gate)
    ):
        raise LocalCodebookDataError("no-GID manifest 状态/合同/停止点不匹配")
    if manifest["contract"].get("code") != _code_files():
        raise LocalCodebookDataError("no-GID 运行源码已变化")
    if set(manifest.get("artifacts", {})) != set(_expected_artifacts()):
        raise LocalCodebookDataError("no-GID artifact 清单不完整")
    if any(
        "gid6" in name.lower() or "geohash" in name.lower()
        for name in manifest["artifacts"]
    ):
        raise LocalCodebookDataError("no-GID 输出中不得含 GID6/geohash artifact")
    for name, entry in manifest["artifacts"].items():
        if entry.get("file") != name or sha256_file(output / name) != entry.get("sha256"):
            raise LocalCodebookDataError(f"no-GID artifact 缺失或哈希错误：{name}")

    settings = config.gates[gate]
    poi_rows = int(settings["poi_rows"])
    query_rows = int(settings["p6_query_rows"])
    d3_rows = int(settings["d3_query_rows"])
    dimension = int(manifest["contract"]["embedding_dim"])
    expected_arrays = {
        "selected_poi_rows.npy": ((poi_rows,), np.int64),
        "selected_d3_query_rows.npy": ((d3_rows,), np.int64),
        "poi_parent_s1_s2.npy": ((poi_rows, 2), np.int32),
        "poi_parent_keys.npy": ((poi_rows,), np.int64),
        "poi_continuous_geo_features.npy": ((poi_rows, GEO_FEATURE_DIM), np.float32),
        "poi_singleton_parent_mask.npy": ((poi_rows,), np.bool_),
        "poi_codebook_s3.npy": ((512, dimension), np.float32),
        "query_codebook_s3.npy": ((512, dimension), np.float32),
        "continuous_geo_codebook_s3.npy": ((512, GEO_FEATURE_DIM), np.float32),
        "poi_assignments_s3.npy": ((poi_rows,), np.int32),
        "query_assignments_s3.npy": ((query_rows,), np.int32),
        "poi_sid_s1_s2_s3.npy": ((poi_rows, 3), np.int32),
        "query_sid_s1_s2_s3.npy": ((query_rows, 3), np.int32),
        "poi_residual_after_s3.npy": ((poi_rows, dimension), np.float16),
        "query_residual_after_s3_d3.npy": ((d3_rows, dimension), np.float16),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in expected_arrays.items():
        values = np.load(output / name, mmap_mode="r", allow_pickle=False)
        if values.shape != shape or values.dtype != np.dtype(dtype) or not np.isfinite(values).all():
            raise LocalCodebookDataError(f"no-GID {name} shape/dtype/finite 错误")
        arrays[name] = values
    assignments = arrays["poi_assignments_s3.npy"]
    query_assignments = arrays["query_assignments_s3.npy"]
    if np.any(assignments < 0) or np.any(assignments >= 512):
        raise LocalCodebookDataError("no-GID POI S3 assignment 超界")
    d3_mask = np.zeros(query_rows, dtype=bool)
    d3_mask[arrays["selected_d3_query_rows.npy"]] = True
    if np.any(query_assignments[d3_mask] < 0) or np.any(query_assignments[d3_mask] >= 512) or np.any(query_assignments[~d3_mask] != -1):
        raise LocalCodebookDataError("no-GID Query S3 mask/assignment 错误")
    p6_manifest = (config.project_root / str(settings["p6_manifest"]["path"])).resolve()
    p6_dir = p6_manifest.parent
    poi_prefix = np.load(p6_dir / "poi_assignments_s1_s2.npy", mmap_mode="r", allow_pickle=False)
    query_prefix = np.load(p6_dir / "query_assignments_s1_s2.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(arrays["poi_parent_s1_s2.npy"], poi_prefix):
        raise LocalCodebookDataError("no-GID parent 未逐值冻结 P6 S1/S2")
    if not np.array_equal(arrays["poi_parent_keys.npy"], s1_s2_parent_keys(poi_prefix[:, 0], poi_prefix[:, 1])):
        raise LocalCodebookDataError("no-GID parent key 不是 S1/S2")
    if not np.array_equal(arrays["poi_sid_s1_s2_s3.npy"][:, :2], poi_prefix) or not np.array_equal(
        arrays["poi_sid_s1_s2_s3.npy"][:, 2], assignments
    ):
        raise LocalCodebookDataError("no-GID POI SID 不是严格三位 S1/S2/S3")
    query_sid = arrays["query_sid_s1_s2_s3.npy"]
    if not np.array_equal(query_sid[:, :2], query_prefix) or not np.array_equal(query_sid[:, 2], query_assignments):
        raise LocalCodebookDataError("no-GID Query SID 与冻结 prefix/S3 不一致")

    metadata = pq.ParquetFile(output / "poi_nogid_metadata.parquet")
    hard = pq.ParquetFile(output / "hard_entity_edges.parquet")
    edges = pq.ParquetFile(output / "query_poi_edges_s3.parquet")
    if metadata.metadata.num_rows != poi_rows or not metadata.schema_arrow.equals(POI_METADATA_SCHEMA):
        raise LocalCodebookDataError("no-GID POI metadata schema/行数错误")
    if not hard.schema_arrow.equals(HARD_EDGE_SCHEMA) or edges.metadata.num_rows != d3_rows:
        raise LocalCodebookDataError("no-GID hard/S3 edge schema 或行数错误")
    hard_table = hard.read()
    if len(hard_table):
        if pc.any(pc.equal(hard_table["poi_row_index"], hard_table["neighbor_poi_row_index"])).as_py():
            raise LocalCodebookDataError("no-GID hard edge 含 self-loop")
        if pc.min(hard_table["composite_similarity"]).as_py() + 1.0e-6 < 0.50:
            raise LocalCodebookDataError("no-GID hard edge 低于等价阈值 0.50")
        counts = np.unique(hard_table["poi_row_index"].to_numpy(zero_copy_only=False), return_counts=True)[1]
        if int(counts.max()) > 20:
            raise LocalCodebookDataError("no-GID hard edge 超过 Top-20")
    singleton = arrays["poi_singleton_parent_mask.npy"]
    geo = arrays["poi_continuous_geo_features.npy"]
    if np.any(geo[singleton] != 0.0):
        raise LocalCodebookDataError("no-GID singleton continuous Geo 必须为 0")
    norms = np.linalg.norm(geo[~singleton], axis=1)
    if len(norms) and np.any((norms > 1.0e-6) & (np.abs(norms - 1.0) > 2.0e-5)):
        raise LocalCodebookDataError("no-GID continuous Geo 非零行必须 L2 normalize")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if manifest.get("metrics") != metrics:
        raise LocalCodebookDataError("no-GID manifest 与 metrics.json 不一致")
    return {
        "status": "p7_nogid_validated",
        "phase": manifest["phase"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "poi_rows": poi_rows,
        "d3_query_rows": d3_rows,
        "hard_edge_rows": hard.metadata.num_rows,
        "metrics": metrics,
        "next_status": manifest["next_status"],
    }


def load_and_build_nogid_codebook(
    config: NoGIDCodebookConfig,
    *,
    gate: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Load Train-only inputs and build one no-GID S3 gate."""
    return build_nogid_codebook(
        config,
        load_nogid_codebook_inputs(config, gate=gate),
        gate=gate,
        device=device,
        chunk_rows=chunk_rows,
    )
