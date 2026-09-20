"""Static-evaluation inputs assembled from completed QG artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file
from qg_prqk.data.query_supervision import CATEGORY_MAPPING_SCHEMA
from qg_prqk.data.query_graph_data import EDGE_SCHEMA, NODE_SCHEMA
from qg_prqk.sid.evaluation_config import SidEvaluationConfig


class SidEvaluationDataError(ValueError):
    """Raised when a frozen P8 source is missing, mutated, or misaligned."""


@dataclass(frozen=True)
class SidMethodArtifacts:
    """One method's POI/query codebooks and assignments in the same token space."""

    poi_codebooks: tuple[np.ndarray, np.ndarray, np.ndarray]
    query_codebooks: tuple[np.ndarray, np.ndarray, np.ndarray]
    poi_sid: np.ndarray
    query_sid: np.ndarray | None
    residual_metrics: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]


@dataclass(frozen=True)
class SidEvaluationInputs:
    """Aligned data needed for the A0/A4 static comparison and publication."""

    selected_poi_rows: np.ndarray
    query_nodes: pa.Table
    edges_s1_s2: pa.Table
    edges_s3: pa.Table
    query_residual_s0: np.ndarray
    poi_ids: pa.Array
    coarse_category_indices: np.ndarray
    fine_category_indices: np.ndarray
    gid6: np.ndarray
    a0: SidMethodArtifacts
    a4: SidMethodArtifacts
    p2_5_metrics: Mapping[str, Any]
    p4_metrics: Mapping[str, Any]
    p2_5_samples: Mapping[str, Any]
    source_paths: Mapping[str, Path]
    source_hashes: Mapping[str, str]
    embedding_dim: int


def _require_npy(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    if not path.is_file():
        raise SidEvaluationDataError(f"P8 冻结 NPY 不存在：{path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != np.dtype(dtype):
        raise SidEvaluationDataError(
            f"P8 {path.name} 合同不匹配：{values.shape}/{values.dtype}，"
            f"预期 {shape}/{np.dtype(dtype)}"
        )
    return values


def _require_parquet(path: Path, schema: pa.Schema, rows: int) -> None:
    if not path.is_file():
        raise SidEvaluationDataError(f"P8 冻结 Parquet 不存在：{path}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != rows or not parquet.schema_arrow.equals(schema):
        raise SidEvaluationDataError(f"P8 {path.name} schema/行数不匹配")


def _read_category_mapping(path: Path) -> pa.Table:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise SidEvaluationDataError("P8 category mapping 不存在")
    tables = [pq.read_table(file) for file in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if not table.schema.equals(CATEGORY_MAPPING_SCHEMA):
        raise SidEvaluationDataError("P8 category mapping schema 不匹配")
    rows = table["poi_row_index"].to_numpy(zero_copy_only=False)
    if not np.array_equal(rows, np.arange(len(table), dtype=np.int64)):
        raise SidEvaluationDataError("P8 category mapping 未按 active POI 行号排序")
    return table


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paths(config: SidEvaluationConfig, gate: str) -> dict[str, Path]:
    if gate not in ("sample", "full"):
        raise SidEvaluationDataError("P8 只开放 sample/full")
    p7_manifest = (
        config.project_root / str(config.gates[gate]["p7_manifest"]["path"])
    ).resolve()
    p7 = _load_json(p7_manifest)
    p6_manifest = (
        config.project_root / str(p7["contract"]["gate_source"]["path"])
    ).resolve()
    p5_manifest = Path(config.base.base.frozen_inputs["p5_full_manifest"]["path"])
    p4_manifest = Path(config.base.base.frozen_inputs["p4_full_manifest"]["path"])
    p2_5_manifest = Path(config.base.base.base.upstream_artifacts["p2_5_manifest"]["path"])
    return {
        "p2_5_manifest": p2_5_manifest,
        "p4_manifest": p4_manifest,
        "p5_manifest": p5_manifest,
        "p6_manifest": p6_manifest,
        "p7_manifest": p7_manifest,
    }


def inspect_sid_evaluation_inputs(config: SidEvaluationConfig, *, gate: str) -> dict[str, Any]:
    """Validate upstream completion, hashes, schemas, and shapes without metric work."""
    paths = _paths(config, gate)
    manifests = {name: _load_json(path) for name, path in paths.items()}
    p7 = manifests["p7_manifest"]
    p6 = manifests["p6_manifest"]
    p5 = manifests["p5_manifest"]
    p4 = manifests["p4_manifest"]
    p2_5 = manifests["p2_5_manifest"]
    settings = config.gates[gate]
    poi_rows = int(settings["poi_rows"])
    query_rows = int(settings["query_rows"])
    d3_rows = int(settings["d3_query_rows"])
    if (
        p7.get("status") != "completed"
        or p7.get("phase") != f"P7-CAT-{gate.upper()}"
        or p6.get("status") != "completed"
        or p6.get("phase") != f"P6-CAT-{gate.upper()}"
        or p5.get("status") != "completed"
        or p5.get("phase") != "P5-CAT-FULL"
        or p4.get("status") != "completed"
        or p4.get("phase") != "P4-CAT-FULL"
        or p2_5.get("status") != "completed"
    ):
        raise SidEvaluationDataError("P8 上游阶段未全部完成")
    expected_hashes = {
        "p7_manifest": str(config.gates[gate]["p7_manifest"]["sha256"]),
        "p6_manifest": str(p7["contract"]["gate_source"]["sha256"]),
        "p5_manifest": str(p7["contract"]["source_hashes"]["p5_manifest_sha256"]),
        "p4_manifest": str(p7["contract"]["source_hashes"]["p4_manifest_sha256"]),
        "p2_5_manifest": str(config.base.base.base.upstream_artifacts["p2_5_manifest"]["sha256"]),
    }
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise SidEvaluationDataError("P8 上游 manifest SHA256 与冻结合同不一致")
    p7_dir, p6_dir, p5_dir = (
        paths["p7_manifest"].parent,
        paths["p6_manifest"].parent,
        paths["p5_manifest"].parent,
    )
    dimension = int(p7["contract"]["embedding_dim"])
    if (
        int(p7["contract"]["poi_rows"]) != poi_rows
        or int(p7["contract"]["p6_query_rows"]) != query_rows
        or int(p7["contract"]["d3_query_rows"]) != d3_rows
        or dimension != 1024
    ):
        raise SidEvaluationDataError("P8 P7 规模与配置不匹配")
    _require_npy(p7_dir / "selected_poi_rows.npy", (poi_rows,), np.int64)
    _require_npy(p7_dir / "poi_gid6.npy", (poi_rows, 6), np.uint8)
    _require_npy(p7_dir / "poi_sid_s1_s2_s3.npy", (poi_rows, 3), np.int32)
    _require_npy(p7_dir / "query_sid_s1_s2_s3.npy", (query_rows, 3), np.int32)
    _require_npy(p6_dir / "query_residual_s0.npy", (query_rows, dimension), np.float16)
    _require_npy(p6_dir / "poi_assignments_s1_s2.npy", (poi_rows, 2), np.int32)
    _require_npy(p6_dir / "query_assignments_s1_s2.npy", (query_rows, 2), np.int32)
    _require_npy(p5_dir / "poi_assignments_s1_s2_s3.npy", (716_245, 3), np.int32)
    for level in (1, 2, 3):
        _require_npy(p5_dir / f"poi_codebook_s{level}.npy", (512, dimension), np.float32)
    for level in (1, 2):
        _require_npy(p6_dir / f"poi_codebook_s{level}.npy", (512, dimension), np.float32)
        _require_npy(p6_dir / f"query_codebook_s{level}.npy", (512, dimension), np.float32)
    for view in ("poi", "query"):
        _require_npy(p7_dir / f"{view}_codebook_s3.npy", (512, dimension), np.float32)
    _require_parquet(p6_dir / "query_nodes.parquet", NODE_SCHEMA, query_rows)
    _require_parquet(
        p6_dir / "query_poi_edges_s1_s2.parquet",
        EDGE_SCHEMA,
        int(p6["contract"]["selection"]["actual_s1_s2_edge_rows"]),
    )
    _require_parquet(
        p7_dir / "query_poi_edges_s3.parquet", EDGE_SCHEMA, d3_rows
    )
    return {
        "status": "p8_input_headers_validated",
        "phase": f"P8-CAT-{gate.upper()}",
        "config_signature": config.signature(),
        "poi_rows": poi_rows,
        "query_rows": query_rows,
        "d3_query_rows": d3_rows,
        "embedding_dim": dimension,
        "source_hashes": actual_hashes,
        "large_artifact_values_read": False,
        "business_validation_read": False,
        "business_test_read": False,
        "output_written": False,
        "next_status": f"READY_TO_RUN_P8_{gate.upper()}",
    }


def load_sid_evaluation_inputs(config: SidEvaluationConfig, *, gate: str) -> SidEvaluationInputs:
    """Load aligned sample/full arrays after the read-only header contract passes."""
    summary = inspect_sid_evaluation_inputs(config, gate=gate)
    paths = _paths(config, gate)
    p2_5, p4, p5, p6, p7 = (
        _load_json(paths["p2_5_manifest"]),
        _load_json(paths["p4_manifest"]),
        _load_json(paths["p5_manifest"]),
        _load_json(paths["p6_manifest"]),
        _load_json(paths["p7_manifest"]),
    )
    p5_dir, p6_dir, p7_dir = (
        paths["p5_manifest"].parent,
        paths["p6_manifest"].parent,
        paths["p7_manifest"].parent,
    )
    poi_rows, query_rows = summary["poi_rows"], summary["query_rows"]
    selected = np.load(p7_dir / "selected_poi_rows.npy", mmap_mode="r", allow_pickle=False)
    category = _read_category_mapping(config.base.base.base.query_depth_dir / "category_mapping.parquet")
    selected_category = category.take(pa.array(selected))
    metadata = pq.read_table(
        p7_dir / "poi_geo_metadata.parquet", columns=["poi_row_index", "poi_id"]
    )
    if not np.array_equal(
        metadata["poi_row_index"].to_numpy(zero_copy_only=False), selected
    ):
        raise SidEvaluationDataError("P8 P7 metadata 与 selected_poi_rows 未对齐")
    a0_sid_full = np.load(
        p5_dir / "poi_assignments_s1_s2_s3.npy", mmap_mode="r", allow_pickle=False
    )
    a0_sid = np.asarray(a0_sid_full[selected], dtype=np.int32)
    a4_sid = np.load(p7_dir / "poi_sid_s1_s2_s3.npy", mmap_mode="r", allow_pickle=False)
    p5_levels = tuple(p5["metrics"]["level_metrics"])
    p6_levels = tuple(p6["metrics"]["levels"])
    p7_metrics = p7["metrics"]
    a0 = SidMethodArtifacts(
        poi_codebooks=tuple(
            np.load(p5_dir / f"poi_codebook_s{level}.npy", mmap_mode="r", allow_pickle=False)
            for level in (1, 2, 3)
        ),  # type: ignore[arg-type]
        query_codebooks=tuple(
            np.load(p5_dir / f"poi_codebook_s{level}.npy", mmap_mode="r", allow_pickle=False)
            for level in (1, 2, 3)
        ),  # type: ignore[arg-type]
        poi_sid=a0_sid,
        query_sid=None,
        residual_metrics=tuple(level["residual"] for level in p5_levels),  # type: ignore[arg-type]
    )
    a4 = SidMethodArtifacts(
        poi_codebooks=(
            np.load(p6_dir / "poi_codebook_s1.npy", mmap_mode="r", allow_pickle=False),
            np.load(p6_dir / "poi_codebook_s2.npy", mmap_mode="r", allow_pickle=False),
            np.load(p7_dir / "poi_codebook_s3.npy", mmap_mode="r", allow_pickle=False),
        ),
        query_codebooks=(
            np.load(p6_dir / "query_codebook_s1.npy", mmap_mode="r", allow_pickle=False),
            np.load(p6_dir / "query_codebook_s2.npy", mmap_mode="r", allow_pickle=False),
            np.load(p7_dir / "query_codebook_s3.npy", mmap_mode="r", allow_pickle=False),
        ),
        poi_sid=a4_sid,
        query_sid=np.load(
            p7_dir / "query_sid_s1_s2_s3.npy", mmap_mode="r", allow_pickle=False
        ),
        residual_metrics=(
            p6_levels[0]["poi_residual"],
            p6_levels[1]["poi_residual"],
            p7_metrics["poi_residual"],
        ),
    )
    p2_5_metrics_path = paths["p2_5_manifest"].parent / str(
        p2_5["outputs"]["metrics"]["file"]
    )
    p2_5_metrics = _load_json(p2_5_metrics_path)
    query_nodes = pq.read_table(p6_dir / "query_nodes.parquet")
    if len(query_nodes) != query_rows or len(a4_sid) != poi_rows:
        raise SidEvaluationDataError("P8 Query/POI 行数在加载期间变化")
    return SidEvaluationInputs(
        selected_poi_rows=selected,
        query_nodes=query_nodes,
        edges_s1_s2=pq.read_table(p6_dir / "query_poi_edges_s1_s2.parquet"),
        edges_s3=pq.read_table(p7_dir / "query_poi_edges_s3.parquet"),
        query_residual_s0=np.load(
            p6_dir / "query_residual_s0.npy", mmap_mode="r", allow_pickle=False
        ),
        poi_ids=metadata["poi_id"].combine_chunks(),
        coarse_category_indices=selected_category["coarse_category_index"]
        .to_numpy(zero_copy_only=False)
        .astype(np.int32),
        fine_category_indices=selected_category["fine_category_index"]
        .to_numpy(zero_copy_only=False)
        .astype(np.int32),
        gid6=np.load(p7_dir / "poi_gid6.npy", mmap_mode="r", allow_pickle=False),
        a0=a0,
        a4=a4,
        p2_5_metrics=p2_5_metrics,
        p4_metrics=p4["metrics"],
        p2_5_samples=p2_5_metrics["samples"],
        source_paths=paths,
        source_hashes=summary["source_hashes"],
        embedding_dim=summary["embedding_dim"],
    )
