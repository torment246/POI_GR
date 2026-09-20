"""Compare three-token SIDs using an S1/S2-only S3 parent."""

from __future__ import annotations

import json
import os
import resource
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.data.query_graph_data import EDGE_SCHEMA, NODE_SCHEMA
from qg_prqk.sid.nogid_data import POI_METADATA_SCHEMA
from qg_prqk.sid.evaluation_data import SidEvaluationDataError, SidEvaluationInputs, SidMethodArtifacts, _read_category_mapping, _require_npy
from qg_prqk.sid.evaluation import (
    BUCKET_SCHEMA,
    POI_SID_SCHEMA,
    _alignment_metrics,
    _artifact,
    _bucket_table,
    _codes_per_category,
    _distribution,
    _map_edges,
    _path_metrics,
    _summary,
    partition_metrics,
    prefix_probe,
)
from qg_prqk.sid.nogid_evaluation_config import NoGIDEvaluationConfig


SCHEMA_VERSION = "qg-prqk-p8-nogid-comparison-v1"


def _event(stage: str, **values: Any) -> None:
    print(json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False), flush=True)


def _manifest_path(config: NoGIDEvaluationConfig, name: str) -> Path:
    return (config.project_root / str(config.frozen_manifests[name]["path"])).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/evaluation_data.py": directory / "evaluation_data.py",
        "sid/evaluation.py": directory / "evaluation.py",
        "sid/nogid_evaluation_config.py": directory / "nogid_evaluation_config.py",
        "sid/nogid_evaluation.py": directory / "nogid_evaluation.py",
        "commands/evaluate_sid_nogid.py": (
            directory.parent / "commands/evaluate_sid_nogid.py"
        ),
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _atomic_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise SidEvaluationDataError(f"P8 no-GID 输出已存在：{target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.writing")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    if path.exists():
        raise SidEvaluationDataError(f"P8 no-GID 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=65_536)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise SidEvaluationDataError(f"P8 no-GID 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_nogid_evaluation_inputs(config: NoGIDEvaluationConfig) -> dict[str, Any]:
    """Validate the five frozen manifests and required array headers."""
    paths = {name: _manifest_path(config, name) for name in config.frozen_manifests}
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected_hashes = {name: str(item["sha256"]) for name, item in config.frozen_manifests.items()}
    if actual_hashes != expected_hashes:
        raise SidEvaluationDataError("P8 no-GID frozen manifest SHA256 不匹配")
    manifests = {name: _load_json(path) for name, path in paths.items()}
    expected_phases = {
        "p4": "P4-CAT-FULL",
        "p5_a0": "P5-CAT-FULL",
        "p6": "P6-CAT-FULL",
        "p7_gid": "P7-CAT-FULL",
        "p7_nogid": "P7-CAT-NOGID-FULL",
    }
    for name, phase in expected_phases.items():
        if manifests[name].get("status") != "completed" or manifests[name].get("phase") != phase:
            raise SidEvaluationDataError(f"P8 no-GID 上游 {name} 状态或 phase 不匹配")
    if manifests["p7_nogid"].get("contract", {}).get("gid_used") is not False:
        raise SidEvaluationDataError("P8 no-GID S3 manifest 未声明 gid_used=false")
    if manifests["p7_nogid"]["contract"].get("final_sid_positions") != ["s1", "s2", "s3"]:
        raise SidEvaluationDataError("P8 no-GID S3 不是三位 SID")

    poi_rows, query_rows, d3_rows, dimension = (
        config.rows["poi"],
        config.rows["query"],
        config.rows["d3_query"],
        config.rows["embedding_dim"],
    )
    p5_dir, p6_dir, p7_gid_dir, p7_nogid_dir = (
        paths["p5_a0"].parent,
        paths["p6"].parent,
        paths["p7_gid"].parent,
        paths["p7_nogid"].parent,
    )
    selected_p6 = _require_npy(p6_dir / "selected_poi_rows.npy", (poi_rows,), np.int64)
    selected_gid = _require_npy(p7_gid_dir / "selected_poi_rows.npy", (poi_rows,), np.int64)
    selected_nogid = _require_npy(p7_nogid_dir / "selected_poi_rows.npy", (poi_rows,), np.int64)
    if not np.array_equal(selected_p6, selected_gid) or not np.array_equal(selected_p6, selected_nogid):
        raise SidEvaluationDataError("P8 no-GID 三种方法 POI 行空间不一致")
    _require_npy(p5_dir / "poi_assignments_s1_s2_s3.npy", (poi_rows, 3), np.int32)
    _require_npy(p6_dir / "query_residual_s0.npy", (query_rows, dimension), np.float16)
    for directory in (p7_gid_dir, p7_nogid_dir):
        _require_npy(directory / "poi_sid_s1_s2_s3.npy", (poi_rows, 3), np.int32)
        _require_npy(directory / "query_sid_s1_s2_s3.npy", (query_rows, 3), np.int32)
        for view in ("poi", "query"):
            _require_npy(directory / f"{view}_codebook_s3.npy", (512, dimension), np.float32)
    for level in (1, 2, 3):
        _require_npy(p5_dir / f"poi_codebook_s{level}.npy", (512, dimension), np.float32)
    for level in (1, 2):
        for view in ("poi", "query"):
            _require_npy(p6_dir / f"{view}_codebook_s{level}.npy", (512, dimension), np.float32)
    nodes = pq.ParquetFile(p6_dir / "query_nodes.parquet")
    edges12 = pq.ParquetFile(p6_dir / "query_poi_edges_s1_s2.parquet")
    edges3 = pq.ParquetFile(p7_nogid_dir / "query_poi_edges_s3.parquet")
    metadata = pq.ParquetFile(p7_nogid_dir / "poi_nogid_metadata.parquet")
    if nodes.metadata.num_rows != query_rows or not nodes.schema_arrow.equals(NODE_SCHEMA):
        raise SidEvaluationDataError("P8 no-GID query_nodes schema/行数不匹配")
    if not edges12.schema_arrow.equals(EDGE_SCHEMA) or not edges3.schema_arrow.equals(EDGE_SCHEMA):
        raise SidEvaluationDataError("P8 no-GID Query edge schema 不匹配")
    if edges3.metadata.num_rows != d3_rows or not metadata.schema_arrow.equals(POI_METADATA_SCHEMA):
        raise SidEvaluationDataError("P8 no-GID S3 edge/metadata 合同不匹配")
    return {
        "status": "p8_nogid_input_headers_validated",
        "poi_rows": poi_rows,
        "query_rows": query_rows,
        "d3_query_rows": d3_rows,
        "embedding_dim": dimension,
        "source_hashes": actual_hashes,
        "final_sid_positions": ["s1", "s2", "s3"],
        "s3_candidate_parent": ["s1", "s2"],
        "gid_artifact_values_read": False,
        "business_validation_read": False,
        "business_test_read": False,
        "output_written": False,
    }


def _method(
    poi_books: tuple[np.ndarray, np.ndarray, np.ndarray],
    query_books: tuple[np.ndarray, np.ndarray, np.ndarray],
    poi_sid: np.ndarray,
    query_sid: np.ndarray | None,
    residual_metrics: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
) -> SidMethodArtifacts:
    return SidMethodArtifacts(poi_books, query_books, poi_sid, query_sid, residual_metrics)


def _load_inputs(config: NoGIDEvaluationConfig) -> tuple[SidEvaluationInputs, dict[str, SidMethodArtifacts], dict[str, Path], dict[str, Any]]:
    summary = inspect_nogid_evaluation_inputs(config)
    paths = {name: _manifest_path(config, name) for name in config.frozen_manifests}
    manifests = {name: _load_json(path) for name, path in paths.items()}
    p5_dir, p6_dir, p7_gid_dir, p7_nogid_dir = (
        paths["p5_a0"].parent,
        paths["p6"].parent,
        paths["p7_gid"].parent,
        paths["p7_nogid"].parent,
    )
    selected = np.load(p6_dir / "selected_poi_rows.npy", mmap_mode="r", allow_pickle=False)
    p5_levels = manifests["p5_a0"]["metrics"]["level_metrics"]
    p6_levels = manifests["p6"]["metrics"]["levels"]
    a0 = _method(
        tuple(np.load(p5_dir / f"poi_codebook_s{level}.npy", mmap_mode="r") for level in (1, 2, 3)),  # type: ignore[arg-type]
        tuple(np.load(p5_dir / f"poi_codebook_s{level}.npy", mmap_mode="r") for level in (1, 2, 3)),  # type: ignore[arg-type]
        np.load(p5_dir / "poi_assignments_s1_s2_s3.npy", mmap_mode="r"),
        None,
        tuple(level["residual"] for level in p5_levels),  # type: ignore[arg-type]
    )
    common_poi = tuple(np.load(p6_dir / f"poi_codebook_s{level}.npy", mmap_mode="r") for level in (1, 2))
    common_query = tuple(np.load(p6_dir / f"query_codebook_s{level}.npy", mmap_mode="r") for level in (1, 2))
    a4_gid = _method(
        (*common_poi, np.load(p7_gid_dir / "poi_codebook_s3.npy", mmap_mode="r")),
        (*common_query, np.load(p7_gid_dir / "query_codebook_s3.npy", mmap_mode="r")),
        np.load(p7_gid_dir / "poi_sid_s1_s2_s3.npy", mmap_mode="r"),
        np.load(p7_gid_dir / "query_sid_s1_s2_s3.npy", mmap_mode="r"),
        (p6_levels[0]["poi_residual"], p6_levels[1]["poi_residual"], manifests["p7_gid"]["metrics"]["poi_residual"]),
    )
    a4_nogid = _method(
        (*common_poi, np.load(p7_nogid_dir / "poi_codebook_s3.npy", mmap_mode="r")),
        (*common_query, np.load(p7_nogid_dir / "query_codebook_s3.npy", mmap_mode="r")),
        np.load(p7_nogid_dir / "poi_sid_s1_s2_s3.npy", mmap_mode="r"),
        np.load(p7_nogid_dir / "query_sid_s1_s2_s3.npy", mmap_mode="r"),
        (p6_levels[0]["poi_residual"], p6_levels[1]["poi_residual"], manifests["p7_nogid"]["metrics"]["poi_residual"]),
    )
    methods = {"A0_POI_ONLY": a0, "A4_GID_PARENT": a4_gid, "A4_NOGID_S1S2_PARENT": a4_nogid}
    category = _read_category_mapping(config.category_mapping)
    selected_category = category.take(pa.array(selected))
    metadata = pq.read_table(p7_nogid_dir / "poi_nogid_metadata.parquet", columns=["poi_row_index", "poi_id"])
    if not np.array_equal(metadata["poi_row_index"].to_numpy(zero_copy_only=False), selected):
        raise SidEvaluationDataError("P8 no-GID metadata 与 POI 行空间错位")
    inputs = SidEvaluationInputs(
        selected_poi_rows=selected,
        query_nodes=pq.read_table(p6_dir / "query_nodes.parquet"),
        edges_s1_s2=pq.read_table(p6_dir / "query_poi_edges_s1_s2.parquet"),
        edges_s3=pq.read_table(p7_nogid_dir / "query_poi_edges_s3.parquet"),
        query_residual_s0=np.load(p6_dir / "query_residual_s0.npy", mmap_mode="r"),
        poi_ids=metadata["poi_id"].combine_chunks(),
        coarse_category_indices=selected_category["coarse_category_index"].to_numpy(zero_copy_only=False).astype(np.int32),
        fine_category_indices=selected_category["fine_category_index"].to_numpy(zero_copy_only=False).astype(np.int32),
        gid6=np.empty((len(selected), 0), dtype=np.uint8),
        a0=a0,
        a4=a4_nogid,
        p2_5_metrics={},
        p4_metrics=manifests["p4"]["metrics"],
        p2_5_samples={},
        source_paths=paths,
        source_hashes=summary["source_hashes"],
        embedding_dim=config.rows["embedding_dim"],
    )
    return inputs, methods, paths, manifests


def _static_metrics(
    method: SidMethodArtifacts,
    coarse_categories: np.ndarray,
    fine_categories: np.ndarray,
) -> dict[str, Any]:
    sid = np.asarray(method.poi_sid, dtype=np.int32)
    paired = []
    for poi_book, query_book in zip(method.poi_codebooks, method.query_codebooks, strict=True):
        poi = np.asarray(poi_book, dtype=np.float64)
        query = np.asarray(query_book, dtype=np.float64)
        poi /= np.linalg.norm(poi, axis=1, keepdims=True).clip(min=1.0e-12)
        query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1.0e-12)
        paired.append(_summary(np.sum(poi * query, axis=1)))
    return {
        "paired_poi_query_codebook_cosine": paired,
        "codebook_levels": [_distribution(sid[:, level]) for level in range(3)],
        "paths": {
            "s1": _path_metrics(sid[:, :1]),
            "s1_s2": _path_metrics(sid[:, :2]),
            "s1_s2_s3": _path_metrics(sid),
        },
        "category": {
            "coarse_vs_s1": partition_metrics(sid[:, 0], coarse_categories),
            "fine_vs_s1_s2": partition_metrics(sid[:, :2], fine_categories),
            "s1_codes_per_coarse_category": _codes_per_category(sid[:, 0], coarse_categories),
            "s1_s2_paths_per_fine_category": _codes_per_category(sid[:, :2], fine_categories),
        },
        "residual_energy": [dict(item) for item in method.residual_metrics],
    }


def _weighted_probe(probe: Mapping[str, Any]) -> dict[str, float]:
    teacher = probe["teacher_forced"]
    return {
        name: float(payload["overall"]["weighted_accuracy"])
        for name, payload in teacher.items()
    } | {
        "autoregressive_s1_s2_s3": float(
            probe["free_running"]["cumulative_prefix"]["s1_to_s3"]["overall"]["weighted_accuracy"]
        ),
        "s3_prediction_coverage": float(probe["prediction_coverage"]["s3_applicable"]),
    }


def _comparison(static: Mapping[str, Any], probes: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"methods": {}}
    for name in ("A0_POI_ONLY", "A4_GID_PARENT", "A4_NOGID_S1S2_PARENT"):
        path = static[name]["paths"]["s1_s2_s3"]
        result["methods"][name] = {
            "distinct_sid": path["distinct"],
            "distinct_ratio": path["distinct_ratio_of_pois"],
            "collision_excess": path["collision_excess"],
            "bucket_max": int(path["bucket"]["max"]),
            "weighted_prefix_probe": _weighted_probe(probes[name]),
        }
    base = result["methods"]["A0_POI_ONLY"]
    old = result["methods"]["A4_GID_PARENT"]
    new = result["methods"]["A4_NOGID_S1S2_PARENT"]
    result["nogid_minus_a0"] = {
        "distinct_sid": int(new["distinct_sid"] - base["distinct_sid"]),
        "collision_excess": int(new["collision_excess"] - base["collision_excess"]),
        "bucket_max": int(new["bucket_max"] - base["bucket_max"]),
    }
    result["nogid_minus_gid_parent"] = {
        "distinct_sid": int(new["distinct_sid"] - old["distinct_sid"]),
        "collision_excess": int(new["collision_excess"] - old["collision_excess"]),
        "bucket_max": int(new["bucket_max"] - old["bucket_max"]),
        "weighted_prefix_probe": {
            key: float(new["weighted_prefix_probe"][key] - old["weighted_prefix_probe"][key])
            for key in new["weighted_prefix_probe"]
        },
    }
    return result


def _poi_sid_table(inputs: SidEvaluationInputs, method: SidMethodArtifacts) -> pa.Table:
    sid = np.asarray(method.poi_sid, dtype=np.int32)
    _, bucket_ids = np.unique(sid, axis=0, return_inverse=True)
    return pa.Table.from_arrays(
        [
            pa.array(inputs.selected_poi_rows, type=pa.int64()),
            inputs.poi_ids,
            pa.array(sid[:, 0], type=pa.int32()),
            pa.array(sid[:, 1], type=pa.int32()),
            pa.array(sid[:, 2], type=pa.int32()),
            pa.array(bucket_ids, type=pa.int64()),
        ],
        schema=POI_SID_SCHEMA,
    )


def _report(comparison: Mapping[str, Any]) -> str:
    methods = comparison["methods"]
    lines = [
        "# NoGID S1/S2-parent S3 全量比较",
        "",
        "三种方法统一使用三位 `[S1,S2,S3]`，Prefix Probe 的 S3 候选只按 `(S1,S2)` 约束，不读取 GID。",
        "",
        "| 方法 | distinct SID | collision excess | max bucket | S3 teacher weighted | cumulative weighted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("A0_POI_ONLY", "A4_GID_PARENT", "A4_NOGID_S1S2_PARENT"):
        row = methods[name]
        probe = row["weighted_prefix_probe"]
        lines.append(
            f"| {name} | {row['distinct_sid']:,} | {row['collision_excess']:,} | {row['bucket_max']} | "
            f"{probe['query_correct_s1_s2_to_s3']:.6f} | {probe['autoregressive_s1_s2_s3']:.6f} |"
        )
    lines.extend(
        [
            "",
            "该表是静态 content-only probe，不等同于 Qwen SFT。结果完成后仍停在 `HOLD_FOR_NOGID_S3_REVIEW`。",
            "",
        ]
    )
    return "\n".join(lines)


def _expected_artifacts() -> Iterable[str]:
    yield from (
        "config_resolved.json",
        "static_metrics.json",
        "prefix_probe_metrics.json",
        "comparison.json",
        "report.md",
        "poi_codebook_s1.npy",
        "poi_codebook_s2.npy",
        "poi_codebook_s3.npy",
        "query_codebook_s1.npy",
        "query_codebook_s2.npy",
        "query_codebook_s3.npy",
        "poi_sid_s1_s2_s3.npy",
        "query_sid_s1_s2_s3.npy",
        "poi_sid.parquet",
        "sid_bucket_index.parquet",
    )


def build_nogid_evaluation(
    config: NoGIDEvaluationConfig,
    *,
    device: str = "cuda",
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Run a full fair three-token comparison and publish the no-GID SID."""
    started = time.perf_counter()
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise SidEvaluationDataError("P8 no-GID Prefix Probe 要求 CUDA")
    if chunk_rows <= 0 or config.output_dir.exists():
        raise SidEvaluationDataError(f"P8 no-GID chunk_rows 非法或 overwrite=false：{config.output_dir}")
    inputs, methods, paths, manifests = _load_inputs(config)
    config.output_dir.mkdir(parents=True)
    write_json_atomic(config.output_dir / "config_resolved.json", config.resolved_payload())
    torch.cuda.reset_peak_memory_stats(torch.device(device))

    edges = {
        1: _map_edges(inputs, inputs.edges_s1_s2, 1),
        2: _map_edges(inputs, inputs.edges_s1_s2, 2),
        3: _map_edges(inputs, inputs.edges_s3, 3),
    }
    static: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    for name, method in methods.items():
        _event("p8_nogid_method_started", method=name)
        static[name] = _static_metrics(method, inputs.coarse_category_indices, inputs.fine_category_indices)
        if method.query_sid is not None:
            static[name]["query_poi_training_assignment_alignment"] = _alignment_metrics(
                np.asarray(method.query_sid), np.asarray(method.poi_sid), edges
            )
        probes[name], _ = prefix_probe(
            inputs,
            method,
            method_name=name,
            s3_parent_mode="s1_s2",
            device=torch.device(device),
            chunk_rows=chunk_rows,
        )
        torch.cuda.empty_cache()
    comparison = _comparison(static, probes)
    write_json_atomic(config.output_dir / "static_metrics.json", {
        "schema_version": SCHEMA_VERSION,
        "methods": static,
        "scope": {"three_token_sid_only": True, "gid_artifact_read": False, "external_baseline_run": False},
    })
    write_json_atomic(config.output_dir / "prefix_probe_metrics.json", {
        "schema_version": SCHEMA_VERSION,
        "s3_parent": ["s1", "s2"],
        "methods": probes,
    })
    write_json_atomic(config.output_dir / "comparison.json", comparison)
    _atomic_text(config.output_dir / "report.md", _report(comparison))

    p6_dir = paths["p6"].parent
    p7_nogid_dir = paths["p7_nogid"].parent
    for level in (1, 2):
        for view in ("poi", "query"):
            _atomic_copy(p6_dir / f"{view}_codebook_s{level}.npy", config.output_dir / f"{view}_codebook_s{level}.npy")
    for view in ("poi", "query"):
        _atomic_copy(p7_nogid_dir / f"{view}_codebook_s3.npy", config.output_dir / f"{view}_codebook_s3.npy")
    _atomic_copy(p7_nogid_dir / "poi_sid_s1_s2_s3.npy", config.output_dir / "poi_sid_s1_s2_s3.npy")
    _atomic_copy(p7_nogid_dir / "query_sid_s1_s2_s3.npy", config.output_dir / "query_sid_s1_s2_s3.npy")
    _atomic_parquet(config.output_dir / "poi_sid.parquet", _poi_sid_table(inputs, methods["A4_NOGID_S1S2_PARENT"]))
    _atomic_parquet(
        config.output_dir / "sid_bucket_index.parquet",
        _bucket_table(np.asarray(methods["A4_NOGID_S1S2_PARENT"].poi_sid)),
    )

    artifacts = {name: _artifact(config.output_dir / name) for name in _expected_artifacts()}
    outcome = {
        "outcome": "REVIEW_REQUIRED",
        "nogid_improves_over_a0_structure": comparison["nogid_minus_a0"]["distinct_sid"] > 0,
        "nogid_improves_over_gid_parent_structure": comparison["nogid_minus_gid_parent"]["distinct_sid"] > 0,
        "automatic_parameter_change": False,
        "downstream_started": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": "P8-CAT-NOGID-COMPARISON-FULL",
        "role": "A0_A4_GID_A4_NOGID_THREE_TOKEN_STATIC_COMPARISON",
        "built_at": utc_now(),
        "contract": {
            "config_signature": config.signature(),
            "config_sha256": config.source_sha256,
            "poi_rows": config.rows["poi"],
            "query_rows": config.rows["query"],
            "d3_query_rows": config.rows["d3_query"],
            "final_sid_positions": ["s1", "s2", "s3"],
            "s3_candidate_parent": ["s1", "s2"],
            "gid_artifact_read": False,
            "source_paths": {name: str(path) for name, path in paths.items()},
            "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
            "code": _code_files(),
        },
        "comparison": comparison,
        "evaluation_outcome": outcome,
        "artifacts": artifacts,
        "source_access": {
            "frozen_train_artifacts_read": True,
            "gid_artifact_values_read": False,
            "raw_business_order_read": False,
            "business_validation_read": False,
            "business_test_read": False,
            "external_baseline_run": False,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "chunk_rows": chunk_rows,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        },
        "upstream_metrics": {
            "p7_gid_weighted_s3_alignment": manifests["p7_gid"]["metrics"]["conditional_s3_accuracy"]["weighted_agreement"],
            "p7_nogid_weighted_s3_alignment": manifests["p7_nogid"]["metrics"]["conditional_s3_accuracy"]["weighted_agreement"],
        },
        "next_status": "HOLD_FOR_NOGID_S3_REVIEW",
    }
    write_json_atomic(config.output_dir / "manifest.json", manifest)
    validate_nogid_evaluation(config, require_success_marker=False)
    write_json_atomic(config.output_dir / "_SUCCESS", {"manifest_sha256": sha256_file(config.output_dir / "manifest.json")})
    _event("p8_nogid_completed", next_status="HOLD_FOR_NOGID_S3_REVIEW", **comparison["methods"]["A4_NOGID_S1S2_PARENT"])
    return validate_nogid_evaluation(config)


def validate_nogid_evaluation(
    config: NoGIDEvaluationConfig,
    *,
    require_success_marker: bool = True,
) -> dict[str, Any]:
    """Validate publication hashes, three-token schemas, sources, and stop boundary."""
    manifest_path = config.output_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if require_success_marker:
        marker = _load_json(config.output_dir / "_SUCCESS")
        if marker.get("manifest_sha256") != manifest_sha:
            raise SidEvaluationDataError("P8 no-GID manifest 与 _SUCCESS 不一致")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P8-CAT-NOGID-COMPARISON-FULL"
        or manifest.get("contract", {}).get("config_signature") != config.signature()
        or manifest.get("contract", {}).get("final_sid_positions") != ["s1", "s2", "s3"]
        or manifest.get("contract", {}).get("s3_candidate_parent") != ["s1", "s2"]
        or manifest.get("contract", {}).get("gid_artifact_read") is not False
        or manifest.get("next_status") != "HOLD_FOR_NOGID_S3_REVIEW"
    ):
        raise SidEvaluationDataError("P8 no-GID manifest 状态/合同/停止点不匹配")
    if manifest["contract"].get("code") != _code_files():
        raise SidEvaluationDataError("P8 no-GID 运行源码已变化")
    source_hashes = {name: sha256_file(_manifest_path(config, name)) for name in config.frozen_manifests}
    if manifest["contract"].get("source_hashes") != source_hashes:
        raise SidEvaluationDataError("P8 no-GID frozen source 已变化")
    if set(manifest.get("artifacts", {})) != set(_expected_artifacts()):
        raise SidEvaluationDataError("P8 no-GID artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        if entry.get("file") != name or sha256_file(config.output_dir / name) != entry.get("sha256"):
            raise SidEvaluationDataError(f"P8 no-GID artifact 缺失或哈希错误：{name}")
    poi_sid = np.load(config.output_dir / "poi_sid_s1_s2_s3.npy", mmap_mode="r")
    query_sid = np.load(config.output_dir / "query_sid_s1_s2_s3.npy", mmap_mode="r")
    if poi_sid.shape != (config.rows["poi"], 3) or query_sid.shape != (config.rows["query"], 3):
        raise SidEvaluationDataError("P8 no-GID SID shape 不是严格三位")
    if np.any(poi_sid < 0) or np.any(poi_sid >= 512):
        raise SidEvaluationDataError("P8 no-GID POI SID token 超界")
    poi_table = pq.ParquetFile(config.output_dir / "poi_sid.parquet")
    bucket_table = pq.ParquetFile(config.output_dir / "sid_bucket_index.parquet")
    if poi_table.metadata.num_rows != config.rows["poi"] or not poi_table.schema_arrow.equals(POI_SID_SCHEMA):
        raise SidEvaluationDataError("P8 no-GID POI SID table schema/行数错误")
    if not bucket_table.schema_arrow.equals(BUCKET_SCHEMA):
        raise SidEvaluationDataError("P8 no-GID bucket index schema 错误")
    if sum(bucket_table.read(columns=["bucket_size"])["bucket_size"].to_pylist()) != config.rows["poi"]:
        raise SidEvaluationDataError("P8 no-GID bucket 数量不守恒")
    comparison = _load_json(config.output_dir / "comparison.json")
    if manifest.get("comparison") != comparison or set(comparison["methods"]) != {
        "A0_POI_ONLY", "A4_GID_PARENT", "A4_NOGID_S1S2_PARENT"
    }:
        raise SidEvaluationDataError("P8 no-GID comparison 内容或方法集合不匹配")
    return {
        "status": "p8_nogid_validated",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "comparison": comparison,
        "evaluation_outcome": manifest["evaluation_outcome"],
        "next_status": manifest["next_status"],
    }
