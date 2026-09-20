"""Frozen-input and graph-closure contracts for S1/S2 refinement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.data.query_supervision import CATEGORY_MAPPING_SCHEMA
from qg_prqk.data.query_graph_data import EDGE_SCHEMA, NODE_SCHEMA
from qg_prqk.sid.relational_config import RelationalCodebookConfig, validate_relational_frozen_inputs
from qg_prqk.sid.base_data import deterministic_selected_rows, selected_rows_sha256


class RelationalCodebookDataError(ValueError):
    """Raised when P6 inputs or a graph-closed gate selection are invalid."""


@dataclass(frozen=True)
class GraphClosedSelection:
    """A complete S1/S2 query subgraph plus every POI it references."""

    nodes: pa.Table
    edges: pa.Table
    selected_query_rows: np.ndarray
    selected_poi_rows: np.ndarray
    base_poi_rows: int
    added_graph_target_rows: int


@dataclass(frozen=True)
class RelationalCodebookInputs:
    """Selected real P6 tensors and graph metadata in stable global row order."""

    selection: GraphClosedSelection
    query_embeddings: np.ndarray
    poi_residual_s0: np.ndarray
    initial_poi_codebooks: tuple[np.ndarray, np.ndarray]
    initial_poi_assignments: tuple[np.ndarray, np.ndarray]
    fine_category_indices: np.ndarray
    coarse_category_indices: np.ndarray
    fine_to_coarse: np.ndarray
    global_mean: np.ndarray
    source_hashes: Mapping[str, str]
    total_poi_rows: int
    total_query_rows: int
    embedding_dim: int


def _select_or_reuse_full_array(
    source: np.ndarray, selected_rows: np.ndarray, *, dtype: np.dtype[Any]
) -> np.ndarray:
    target_dtype = np.dtype(dtype)
    is_identity = (
        len(selected_rows) == len(source)
        and (not len(selected_rows) or int(selected_rows[0]) == 0)
        and (not len(selected_rows) or int(selected_rows[-1]) == len(source) - 1)
        and np.array_equal(selected_rows, np.arange(len(source), dtype=selected_rows.dtype))
    )
    if is_identity and source.dtype == target_dtype:
        return source
    return np.asarray(source[selected_rows], dtype=target_dtype)


def _require_npy(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    if not path.is_file():
        raise RelationalCodebookDataError(f"P6 冻结 NPY 不存在：{path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != np.dtype(dtype):
        raise RelationalCodebookDataError(
            f"P6 冻结 NPY 合同不匹配：{path.name}，"
            f"实际 {values.shape}/{values.dtype}，预期 {shape}/{np.dtype(dtype)}"
        )
    return values


def _require_parquet(path: Path, schema: pa.Schema, rows: int) -> None:
    if not path.is_file():
        raise RelationalCodebookDataError(f"P6 冻结 Parquet 不存在：{path}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != rows or not parquet.schema_arrow.equals(schema):
        raise RelationalCodebookDataError(f"P6 冻结 Parquet schema/行数不匹配：{path.name}")


def _require_parquet_dataset(path: Path, schema: pa.Schema, rows: int) -> int:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files or any(not item.is_file() for item in files):
        raise RelationalCodebookDataError(f"P6 冻结 Parquet dataset 不存在：{path}")
    total = 0
    for file in files:
        parquet = pq.ParquetFile(file)
        if not parquet.schema_arrow.equals(schema):
            raise RelationalCodebookDataError(f"P6 类别 mapping schema 不匹配：{file}")
        total += parquet.metadata.num_rows
    if total != rows:
        raise RelationalCodebookDataError(f"P6 类别 mapping 行数必须为 {rows}，实际 {total}")
    return len(files)


def inspect_relational_input_headers(config: RelationalCodebookConfig) -> dict[str, Any]:
    """Check full P4/P5/category headers without loading large matrix values."""
    validate_relational_frozen_inputs(config)
    p4_manifest_path = Path(config.frozen_inputs["p4_full_manifest"]["path"])
    p5_manifest_path = Path(config.frozen_inputs["p5_full_manifest"]["path"])
    p4_manifest = json.loads(p4_manifest_path.read_text(encoding="utf-8"))
    p5_manifest = json.loads(p5_manifest_path.read_text(encoding="utf-8"))
    p4_dir = p4_manifest_path.parent
    p5_dir = p5_manifest_path.parent
    query_rows = int(p4_manifest["contract"]["query_rows"])
    query_edge_rows = int(p4_manifest["metrics"]["edge_rows"])
    poi_rows = int(p5_manifest["contract"]["poi_rows"])
    embedding_dim = int(p5_manifest["contract"]["embedding_dim"])
    codebook_size = int(config.codebook_sizes[0])
    if (
        query_rows != 342_879
        or poi_rows != 716_245
        or embedding_dim != 1024
        or p4_manifest.get("poi_nodes", {}).get("rows") != poi_rows
        or p4_manifest["contract"].get("embedding_dim") != embedding_dim
    ):
        raise RelationalCodebookDataError("P4/P5 full 行数或向量维度合同不一致")

    _require_parquet(p4_dir / "query_nodes.parquet", NODE_SCHEMA, query_rows)
    _require_parquet(p4_dir / "query_poi_edges.parquet", EDGE_SCHEMA, query_edge_rows)
    category_files = _require_parquet_dataset(
        config.base.query_depth_dir / "category_mapping.parquet",
        CATEGORY_MAPPING_SCHEMA,
        poi_rows,
    )
    _require_npy(
        p4_dir / "query_embeddings.npy",
        shape=(query_rows, embedding_dim),
        dtype=np.float16,
    )
    selected = _require_npy(
        p5_dir / "selected_poi_rows.npy", shape=(poi_rows,), dtype=np.int64
    )
    if int(selected[0]) != 0 or int(selected[-1]) != poi_rows - 1:
        raise RelationalCodebookDataError("P5 full selected_poi_rows 不是 active POI identity 行序")
    _require_npy(
        p5_dir / "global_mean.npy", shape=(embedding_dim,), dtype=np.float32
    )
    for level in (1, 2):
        _require_npy(
            p5_dir / f"poi_codebook_s{level}.npy",
            shape=(codebook_size, embedding_dim),
            dtype=np.float32,
        )
        assignments = _require_npy(
            p5_dir / f"poi_assignments_s{level}.npy",
            shape=(poi_rows,),
            dtype=np.int32,
        )
        if int(assignments.min()) < 0 or int(assignments.max()) >= codebook_size:
            raise RelationalCodebookDataError(f"P5 S{level} assignment 超出 [0,{codebook_size - 1}]")
    for name in ("poi_residual_s0.npy", "poi_residual_after_s1.npy"):
        _require_npy(
            p5_dir / name,
            shape=(poi_rows, embedding_dim),
            dtype=np.float16,
        )
    return {
        "status": "p6_input_headers_validated",
        "config_signature": config.signature(),
        "hard_max_iter": int(config.prqk["max_iter"]),
        "codebook_sizes": list(config.codebook_sizes),
        "poi_rows": poi_rows,
        "query_rows": query_rows,
        "s1_s2_edge_rows": int(
            p4_manifest["metrics"]["layer_edge_rows"]["S1"]
            + p4_manifest["metrics"]["layer_edge_rows"]["S2"]
        ),
        "embedding_dim": embedding_dim,
        "category_mapping_files": category_files,
        "large_artifact_values_read": False,
        "large_artifact_sha256_rechecked": False,
        "business_validation_read": False,
        "business_test_read": False,
        "output_written": False,
        "sampling_protocol_confirmed": False,
    }


def _sorted_unique_rows(
    values: np.ndarray, *, upper_bound: int, label: str
) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
        raise RelationalCodebookDataError(f"{label} 必须是一维非空整数数组")
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if rows[0] < 0 or rows[-1] >= upper_bound:
        raise RelationalCodebookDataError(f"{label} 超出冻结行号范围")
    if not np.array_equal(rows, np.unique(rows)):
        raise RelationalCodebookDataError(f"{label} 必须严格递增且无重复")
    return rows


def build_graph_closed_selection(
    nodes: pa.Table,
    edges: pa.Table,
    *,
    base_poi_rows: np.ndarray,
    selected_query_rows: np.ndarray,
    total_poi_rows: int,
) -> GraphClosedSelection:
    """Keep complete S1/S2 edges and add all referenced POIs to the gate set."""
    if not nodes.schema.equals(NODE_SCHEMA) or not edges.schema.equals(EDGE_SCHEMA):
        raise RelationalCodebookDataError("P6 节点/边 schema 必须与冻结 P4 完全一致")
    query_rows = _sorted_unique_rows(
        selected_query_rows, upper_bound=len(nodes), label="selected_query_rows"
    )
    poi_rows = _sorted_unique_rows(
        base_poi_rows, upper_bound=total_poi_rows, label="base_poi_rows"
    )
    query_values = pa.array(query_rows, type=pa.int64())
    selected_nodes = nodes.filter(pc.is_in(nodes["node_row"], value_set=query_values))
    if len(selected_nodes) != len(query_rows):
        raise RelationalCodebookDataError("selected_query_rows 未与 P4 node_row 一一对应")
    edge_mask = pc.and_(
        pc.is_in(edges["node_row"], value_set=query_values),
        pc.less_equal(edges["layer"], pa.scalar(2, type=pa.int8())),
    )
    selected_edges = edges.filter(edge_mask)
    if not len(selected_edges):
        raise RelationalCodebookDataError("P6 graph-closed selection 不得为空")

    node_rows = selected_nodes["node_row"].to_numpy(zero_copy_only=False)
    depths = selected_nodes["supervision_depth"].to_numpy(zero_copy_only=False)
    reliability = {
        1: selected_nodes["s1_reliability"].to_numpy(zero_copy_only=False),
        2: selected_nodes["s2_reliability"].to_numpy(zero_copy_only=False),
    }
    expected = {
        (int(row), layer): float(reliability[layer][index])
        for index, (row, depth) in enumerate(zip(node_rows, depths, strict=True))
        for layer in (1, 2)
        if int(depth) >= layer
    }
    probability_sums = {key: 0.0 for key in expected}
    weight_sums = {key: 0.0 for key in expected}
    for row in selected_edges.select(
        ["node_row", "layer", "edge_probability", "edge_weight"]
    ).to_pylist():
        key = (int(row["node_row"]), int(row["layer"]))
        if key not in expected:
            raise RelationalCodebookDataError("P6 边层级超出 Query supervision depth")
        probability_sums[key] += float(row["edge_probability"])
        weight_sums[key] += float(row["edge_weight"])
    if set(key for key, value in probability_sums.items() if value > 0) != set(expected):
        raise RelationalCodebookDataError("P6 选中 Query 缺少应有的 S1/S2 完整边")
    for key, expected_weight in expected.items():
        if abs(probability_sums[key] - 1.0) > 1.0e-10:
            raise RelationalCodebookDataError("P6 不允许截断或重新归一 Query 层边概率")
        if abs(weight_sums[key] - expected_weight) > 1.0e-10:
            raise RelationalCodebookDataError("P6 边权总质量必须保持冻结 Query reliability")

    targets = np.unique(
        selected_edges["poi_row_index"].to_numpy(zero_copy_only=False)
    ).astype(np.int64)
    if targets[0] < 0 or targets[-1] >= total_poi_rows:
        raise RelationalCodebookDataError("P6 graph target POI 行号越界")
    closed_poi_rows = np.union1d(poi_rows, targets).astype(np.int64)
    return GraphClosedSelection(
        nodes=selected_nodes,
        edges=selected_edges,
        selected_query_rows=query_rows,
        selected_poi_rows=closed_poi_rows,
        base_poi_rows=len(poi_rows),
        added_graph_target_rows=len(closed_poi_rows) - len(poi_rows),
    )


def _read_category_mapping(path: Path) -> pa.Table:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    tables = [pq.read_table(file) for file in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if not table.schema.equals(CATEGORY_MAPPING_SCHEMA):
        raise RelationalCodebookDataError("P6 类别 mapping schema 不匹配")
    rows = table["poi_row_index"].to_numpy(zero_copy_only=False)
    if not np.array_equal(rows, np.arange(len(table), dtype=np.int64)):
        raise RelationalCodebookDataError("P6 类别 mapping 未按 active POI 行号严格递增")
    return table


def _read_category_vocab(path: Path) -> pa.Table:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise RelationalCodebookDataError("P6 category vocab 不存在")
    tables = [pq.read_table(file) for file in files]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def load_relational_codebook_inputs(config: RelationalCodebookConfig, *, gate: str) -> RelationalCodebookInputs:
    """Load one reviewed graph-closed P6 scale without business Valid/Test."""
    available_gates = set(config.p6_gates) - {"selection", "seed"}
    if gate not in available_gates or gate not in ("sample", "medium", "full"):
        raise RelationalCodebookDataError(
            f"P6 配置 {config.protocol_id} 未开放 {gate}；"
            f"可用规模为 {sorted(available_gates)}"
        )
    summary = inspect_relational_input_headers(config)
    gate_config = config.p6_gates[gate]
    p4_manifest_path = Path(config.frozen_inputs["p4_full_manifest"]["path"])
    p5_manifest_path = Path(config.frozen_inputs["p5_full_manifest"]["path"])
    p4_manifest = json.loads(p4_manifest_path.read_text(encoding="utf-8"))
    p5_manifest = json.loads(p5_manifest_path.read_text(encoding="utf-8"))
    p4_dir = p4_manifest_path.parent
    p5_dir = p5_manifest_path.parent

    nodes = pq.read_table(p4_dir / "query_nodes.parquet")
    edges = pq.read_table(p4_dir / "query_poi_edges.parquet")
    query_rows = int(gate_config["query_rows"])
    selected_query_rows = np.arange(query_rows, dtype=np.int64)
    base_poi_rows = deterministic_selected_rows(
        summary["poi_rows"], int(gate_config["base_poi_rows"]), int(config.p6_gates["seed"])
    )
    selection = build_graph_closed_selection(
        nodes,
        edges,
        base_poi_rows=base_poi_rows,
        selected_query_rows=selected_query_rows,
        total_poi_rows=summary["poi_rows"],
    )
    if (
        len(selection.selected_poi_rows)
        != int(gate_config["expected_closed_poi_rows"])
        or len(selection.edges) != int(gate_config["expected_s1_s2_edge_rows"])
    ):
        raise RelationalCodebookDataError("P6 图闭包规模与用户确认配置不一致")

    query_source = np.load(
        p4_dir / "query_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    query_embeddings = _select_or_reuse_full_array(
        query_source, selection.selected_query_rows, dtype=np.float16
    )
    poi_residual_source = np.load(
        p5_dir / "poi_residual_s0.npy", mmap_mode="r", allow_pickle=False
    )
    poi_residual_s0 = _select_or_reuse_full_array(
        poi_residual_source, selection.selected_poi_rows, dtype=np.float16
    )
    initial_codebooks = tuple(
        np.load(p5_dir / f"poi_codebook_s{level}.npy", allow_pickle=False)
        for level in (1, 2)
    )
    initial_assignments = tuple(
        _select_or_reuse_full_array(
            np.load(
                p5_dir / f"poi_assignments_s{level}.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            selection.selected_poi_rows,
            dtype=np.int32,
        )
        for level in (1, 2)
    )
    category = _read_category_mapping(config.base.query_depth_dir / "category_mapping.parquet")
    fine_all = category["fine_category_index"].to_numpy(zero_copy_only=False)
    coarse_all = category["coarse_category_index"].to_numpy(zero_copy_only=False)
    fine = np.asarray(fine_all[selection.selected_poi_rows], dtype=np.int32)
    coarse = np.asarray(coarse_all[selection.selected_poi_rows], dtype=np.int16)
    resolved = config.base.resolved_payload()
    fine_count = int(resolved["category"]["fine_category_id"]["expected_count"])
    fine_to_coarse = np.full(fine_count, -1, dtype=np.int16)
    coarse_id_to_index = {
        str(coarse_id): int(coarse_index)
        for coarse_id, coarse_index in zip(
            category["coarse_category_id"].to_pylist(),
            coarse_all,
            strict=True,
        )
    }
    vocab_path = (
        config.base.upstream.base.project_root
        / resolved["category"]["existing_mapping"]["category_vocab"]
    ).resolve()
    vocab = _read_category_vocab(vocab_path)
    if set(vocab.column_names) != {"category_code", "category_index"}:
        raise RelationalCodebookDataError("P6 category vocab 字段不匹配")
    for row in vocab.to_pylist():
        fine_index = int(row["category_index"])
        coarse_id = str(row["category_code"])[:2]
        if not 0 <= fine_index < fine_count or coarse_id not in coarse_id_to_index:
            raise RelationalCodebookDataError("P6 category vocab index/coarse 前缀非法")
        fine_to_coarse[fine_index] = coarse_id_to_index[coarse_id]
    for fine_index, coarse_index in zip(fine_all, coarse_all, strict=True):
        current = int(fine_to_coarse[int(fine_index)])
        if current != int(coarse_index):
            raise RelationalCodebookDataError("P6 fine category 不能映射到多个 coarse category")
    if np.any(fine_to_coarse < 0):
        raise RelationalCodebookDataError("P6 fine category mapping 不完整")

    global_mean = np.load(p5_dir / "global_mean.npy", allow_pickle=False)
    source_hashes = {
        "p4_full_manifest_sha256": config.frozen_inputs["p4_full_manifest"]["sha256"],
        "p5_full_manifest_sha256": config.frozen_inputs["p5_full_manifest"]["sha256"],
        "query_embeddings_sha256": p4_manifest["artifacts"]["query_embeddings.npy"]["sha256"],
        "poi_residual_s0_sha256": p5_manifest["artifacts"]["poi_residual_s0.npy"]["sha256"],
        "category_mapping_manifest_sha256": config.base.upstream_artifacts["p2_5_manifest"]["sha256"],
        "base_poi_rows_sha256": selected_rows_sha256(base_poi_rows),
        "closed_poi_rows_sha256": selected_rows_sha256(selection.selected_poi_rows),
        "selected_query_rows_sha256": selected_rows_sha256(selected_query_rows),
    }
    if "p6_sample_manifest" in config.frozen_inputs:
        source_hashes["p6_sample_manifest_sha256"] = config.frozen_inputs[
            "p6_sample_manifest"
        ]["sha256"]
    return RelationalCodebookInputs(
        selection=selection,
        query_embeddings=query_embeddings,
        poi_residual_s0=poi_residual_s0,
        initial_poi_codebooks=initial_codebooks,  # type: ignore[arg-type]
        initial_poi_assignments=initial_assignments,  # type: ignore[arg-type]
        fine_category_indices=fine,
        coarse_category_indices=coarse,
        fine_to_coarse=fine_to_coarse,
        global_mean=global_mean,
        source_hashes=source_hashes,
        total_poi_rows=summary["poi_rows"],
        total_query_rows=summary["query_rows"],
        embedding_dim=summary["embedding_dim"],
    )
