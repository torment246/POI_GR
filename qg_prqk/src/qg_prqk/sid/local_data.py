"""Frozen S3 inputs, local geography, and hard-entity graph."""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file
from qg_prqk.data.query_supervision import CATEGORY_MAPPING_SCHEMA
from qg_prqk.data.query_graph_data import EDGE_SCHEMA, MASK_SCHEMA, NODE_SCHEMA
from qg_prqk.sid.local_config import LocalCodebookConfig
from qg_prqk.sid.geo import encode_geohash_tokens, local_geo_features, parent_group_ids


POI_METADATA_SCHEMA = pa.schema(
    [
        pa.field("poi_row_index", pa.int64(), nullable=False),
        pa.field("poi_id", pa.string(), nullable=False),
        pa.field("displayname", pa.string(), nullable=False),
        pa.field("alias", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("category_code", pa.string(), nullable=False),
        pa.field("address", pa.string(), nullable=False),
        pa.field("lat", pa.float64(), nullable=False),
        pa.field("lng", pa.float64(), nullable=False),
    ]
)
HARD_EDGE_SCHEMA = pa.schema(
    [
        pa.field("poi_row_index", pa.int64(), nullable=False),
        pa.field("neighbor_poi_row_index", pa.int64(), nullable=False),
        pa.field("parent_group", pa.int64(), nullable=False),
        pa.field("fine_category_index", pa.int32(), nullable=False),
        pa.field("bge_similarity", pa.float32(), nullable=False),
        pa.field("name_alias_similarity", pa.float32(), nullable=False),
        pa.field("address_similarity", pa.float32(), nullable=False),
        pa.field("same_gid6", pa.bool_(), nullable=False),
        pa.field("composite_similarity", pa.float32(), nullable=False),
    ]
)
P7_POI_METADATA_SCHEMA = pa.schema(
    [
        *POI_METADATA_SCHEMA,
        pa.field("gid6", pa.binary(6), nullable=False),
        pa.field("parent_group", pa.int64(), nullable=False),
        pa.field("is_singleton_parent", pa.bool_(), nullable=False),
        pa.field("fine_category_index", pa.int32(), nullable=False),
    ]
)


class LocalCodebookDataError(ValueError):
    """Raised when P7 data violate the frozen P4/P5/P6 contracts."""


@dataclass(frozen=True)
class HardEntityGraph:
    """Directed Top-k hard-neighbor edges in selected-POI local row space."""

    poi_rows: np.ndarray
    neighbor_rows: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class LocalCodebookInputs:
    """All tensors and metadata needed to fit S3 without Valid/Test queries."""

    selected_poi_rows: np.ndarray
    p6_query_nodes: pa.Table
    d3_query_rows: np.ndarray
    s3_edges: pa.Table
    false_negative_mask: pa.Table
    poi_metadata: pa.Table
    poi_embeddings: np.ndarray
    poi_residual_after_s2: np.ndarray
    query_residual_after_s2: np.ndarray
    p6_poi_assignments_s1_s2: np.ndarray
    p6_query_assignments_s1_s2: np.ndarray
    initial_poi_codebook_s3: np.ndarray
    initial_poi_assignments_s3: np.ndarray
    fine_category_indices: np.ndarray
    gid6_codes: np.ndarray
    parent_groups: np.ndarray
    geo_features: np.ndarray
    geo_standardization: Mapping[str, object]
    singleton_mask: np.ndarray
    hard_graph: HardEntityGraph
    hard_edges: pa.Table
    protected_pairs: frozenset[tuple[int, int]]
    source_hashes: Mapping[str, str]
    total_poi_rows: int
    embedding_dim: int


def _gate(config: LocalCodebookConfig, gate: str) -> Mapping[str, Any]:
    if gate not in ("sample", "full") or gate not in config.gates:
        raise LocalCodebookDataError("P7 只开放 sample/full")
    return config.gates[gate]


def _manifest_path(config: LocalCodebookConfig, gate: str) -> Path:
    return (
        config.project_root / str(_gate(config, gate)["p6_manifest"]["path"])
    ).resolve()


def _require_npy(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    if not path.is_file():
        raise LocalCodebookDataError(f"P7 冻结 NPY 不存在：{path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != shape or values.dtype != np.dtype(dtype):
        raise LocalCodebookDataError(
            f"P7 {path.name} 合同不匹配：{values.shape}/{values.dtype}，"
            f"预期 {shape}/{np.dtype(dtype)}"
        )
    return values


def inspect_local_codebook_inputs(config: LocalCodebookConfig, *, gate: str) -> dict[str, Any]:
    """Validate P7 headers and stage boundaries without reading catalog values."""
    settings = _gate(config, gate)
    p6_manifest_path = _manifest_path(config, gate)
    p6_manifest = json.loads(p6_manifest_path.read_text(encoding="utf-8"))
    p6_dir = p6_manifest_path.parent
    poi_rows = int(settings["poi_rows"])
    query_rows = int(settings["p6_query_rows"])
    d3_rows = int(settings["d3_query_rows"])
    dimension = int(p6_manifest["contract"]["embedding_dim"])
    if (
        p6_manifest.get("status") != "completed"
        or p6_manifest.get("phase") != f"P6-CAT-{gate.upper()}"
        or p6_manifest["contract"]["selection"]["actual_closed_poi_rows"] != poi_rows
        or p6_manifest["contract"]["query_rows"] != query_rows
        or dimension != 1024
    ):
        raise LocalCodebookDataError("P7 上游 P6 manifest 状态或规模不匹配")
    selected = _require_npy(p6_dir / "selected_poi_rows.npy", (poi_rows,), np.int64)
    if not np.array_equal(selected, np.unique(selected)):
        raise LocalCodebookDataError("P7 P6 selected_poi_rows 必须递增且无重复")
    _require_npy(p6_dir / "poi_residual_after_s2.npy", (poi_rows, dimension), np.float16)
    _require_npy(p6_dir / "query_residual_after_s2.npy", (query_rows, dimension), np.float16)
    _require_npy(p6_dir / "poi_assignments_s1_s2.npy", (poi_rows, 2), np.int32)
    _require_npy(p6_dir / "query_assignments_s1_s2.npy", (query_rows, 2), np.int32)
    nodes_file = pq.ParquetFile(p6_dir / "query_nodes.parquet")
    if nodes_file.metadata.num_rows != query_rows or not nodes_file.schema_arrow.equals(NODE_SCHEMA):
        raise LocalCodebookDataError("P7 P6 query_nodes schema/行数不匹配")
    nodes = nodes_file.read(columns=["supervision_depth"])
    if pc.sum(pc.equal(nodes["supervision_depth"], 3)).as_py() != d3_rows:
        raise LocalCodebookDataError("P7 D3 Query 行数与配置不匹配")

    p5_manifest_path = Path(config.base.frozen_inputs["p5_full_manifest"]["path"])
    p5_manifest = json.loads(p5_manifest_path.read_text(encoding="utf-8"))
    p5_dir = p5_manifest_path.parent
    total_poi_rows = int(p5_manifest["contract"]["poi_rows"])
    _require_npy(p5_dir / "poi_codebook_s3.npy", (512, dimension), np.float32)
    _require_npy(p5_dir / "poi_assignments_s3.npy", (total_poi_rows,), np.int32)

    p4_manifest_path = Path(config.base.frozen_inputs["p4_full_manifest"]["path"])
    p4_manifest = json.loads(p4_manifest_path.read_text(encoding="utf-8"))
    p4_dir = p4_manifest_path.parent
    edge_file = pq.ParquetFile(p4_dir / "query_poi_edges.parquet")
    mask_file = pq.ParquetFile(p4_dir / "false_negative_mask.parquet")
    if not edge_file.schema_arrow.equals(EDGE_SCHEMA) or not mask_file.schema_arrow.equals(MASK_SCHEMA):
        raise LocalCodebookDataError("P7 P4 edge/false-negative schema 不匹配")
    if p4_manifest["metrics"]["layer_edge_rows"]["S3"] != 291_590:
        raise LocalCodebookDataError("P7 P4 full S3 edge 行数不匹配")
    return {
        "status": "p7_input_headers_validated",
        "phase": f"P7-CAT-{gate.upper()}",
        "config_signature": config.signature(),
        "poi_rows": poi_rows,
        "p6_query_rows": query_rows,
        "d3_query_rows": d3_rows,
        "s3_edge_rows": int(settings["s3_edge_rows"]),
        "embedding_dim": dimension,
        "p6_manifest_sha256": sha256_file(p6_manifest_path),
        "p5_manifest_sha256": sha256_file(p5_manifest_path),
        "p4_manifest_sha256": sha256_file(p4_manifest_path),
        "catalog_values_read": False,
        "business_validation_query_read": False,
        "business_test_query_read": False,
        "output_written": False,
    }


def _read_category_mapping(path: Path) -> pa.Table:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    tables = [pq.read_table(file) for file in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if not table.schema.equals(CATEGORY_MAPPING_SCHEMA):
        raise LocalCodebookDataError("P7 category mapping schema 不匹配")
    rows = table["poi_row_index"].to_numpy(zero_copy_only=False)
    if not np.array_equal(rows, np.arange(len(table), dtype=np.int64)):
        raise LocalCodebookDataError("P7 category mapping 未按 active POI 行号排序")
    return table


def _catalog_metadata(config: LocalCodebookConfig, selected_rows: np.ndarray) -> pa.Table:
    catalog_dir = (config.project_root / str(config.poi_catalog["path"])).resolve()
    source_files = sorted(catalog_dir.glob("part-*.json"))
    if not source_files:
        raise LocalCodebookDataError("P7 active POI catalog 没有 part-*.json")
    selected = np.asarray(selected_rows, dtype=np.int64)
    if selected.ndim != 1 or not len(selected) or not np.array_equal(selected, np.unique(selected)):
        raise LocalCodebookDataError("P7 selected POI rows 必须递增且无重复")
    records: list[dict[str, Any]] = []
    selected_position = 0
    global_row = 0
    ids_path = catalog_dir / "poi_ids.jsonl"
    with ids_path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as id_handle:
        for source in source_files:
            with source.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as handle:
                for raw_line in handle:
                    expected_line = id_handle.readline()
                    if not expected_line:
                        raise LocalCodebookDataError("P7 catalog 行数多于 poi_ids")
                    expected_id = json.loads(expected_line)
                    row = json.loads(raw_line)
                    if row.get("poi_id") != expected_id:
                        raise LocalCodebookDataError(f"P7 catalog/poi_ids 行序错位：row={global_row}")
                    if selected_position < len(selected) and global_row == int(selected[selected_position]):
                        missing = [name for name in config.poi_catalog["allowed_fields"] if name not in row]
                        if missing:
                            raise LocalCodebookDataError(f"P7 POI 缺少字段：{missing}")
                        record = {
                            "poi_row_index": global_row,
                            "poi_id": str(row["poi_id"]),
                            "displayname": str(row["displayname"] or ""),
                            "alias": str(row["alias"] or ""),
                            "category": str(row["category"] or ""),
                            "category_code": str(row["category_code"] or ""),
                            "address": str(row["address"] or ""),
                            "lat": float(row["lat"]),
                            "lng": float(row["lng"]),
                        }
                        if (
                            len(record["category_code"]) != 6
                            or not record["category_code"].isdigit()
                            or not math.isfinite(record["lat"])
                            or not math.isfinite(record["lng"])
                        ):
                            raise LocalCodebookDataError(f"P7 POI 类别/经纬度非法：row={global_row}")
                        records.append(record)
                        selected_position += 1
                    global_row += 1
        if id_handle.readline():
            raise LocalCodebookDataError("P7 poi_ids 行数多于 catalog")
    if global_row != 716_245 or selected_position != len(selected):
        raise LocalCodebookDataError("P7 active catalog 总行数或 selected POI 覆盖不匹配")
    return pa.Table.from_pylist(records, schema=POI_METADATA_SCHEMA)


def _surface_ngrams(value: str) -> set[str]:
    compact = "".join(unicodedata.normalize("NFKC", value).lower().split())
    if not compact:
        return set()
    if len(compact) < 2:
        return {compact}
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _protected_pairs(
    nodes: pa.Table,
    s3_edges: pa.Table,
    mask: pa.Table,
    selected_poi_rows: np.ndarray,
) -> frozenset[tuple[int, int]]:
    targets = {
        int(row["node_row"]): int(row["poi_row_index"])
        for row in s3_edges.select(["node_row", "poi_row_index"]).to_pylist()
    }
    d3_node_rows = set(nodes.filter(pc.equal(nodes["supervision_depth"], 3))["node_row"].to_pylist())
    selected_set = set(np.asarray(selected_poi_rows, dtype=np.int64).tolist())
    result: set[tuple[int, int]] = set()
    for row in mask.select(["node_row", "poi_row_index"]).to_pylist():
        node = int(row["node_row"])
        other = int(row["poi_row_index"])
        if node not in d3_node_rows or node not in targets:
            continue
        target = targets[node]
        if target == other or target not in selected_set or other not in selected_set:
            continue
        result.add((min(target, other), max(target, other)))
    return frozenset(result)


def build_hard_entity_graph(
    selected_poi_rows: np.ndarray,
    embeddings: np.ndarray,
    metadata: pa.Table,
    parent_groups: np.ndarray,
    fine_categories: np.ndarray,
    protected_pairs: frozenset[tuple[int, int]],
    *,
    weights: Mapping[str, float],
    threshold: float,
    max_neighbors: int,
) -> tuple[HardEntityGraph, pa.Table]:
    """Build deterministic directed Top-20 edges inside exact parent/category groups."""
    global_rows = np.asarray(selected_poi_rows, dtype=np.int64)
    values = np.asarray(embeddings)
    parents = np.asarray(parent_groups, dtype=np.int64)
    categories = np.asarray(fine_categories, dtype=np.int32)
    if (
        values.ndim != 2
        or len(values) != len(global_rows)
        or parents.shape != global_rows.shape
        or categories.shape != global_rows.shape
        or len(metadata) != len(global_rows)
        or threshold != 0.60
        or max_neighbors != 20
    ):
        raise LocalCodebookDataError("P7 hard graph 输入或冻结阈值不匹配")
    names = metadata["displayname"].to_pylist()
    aliases = metadata["alias"].to_pylist()
    addresses = metadata["address"].to_pylist()
    compound = parents.astype(np.int64) * (int(categories.max()) + 1) + categories.astype(np.int64)
    order = np.argsort(compound, kind="stable")
    sorted_keys = compound[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])
    output: list[dict[str, Any]] = []
    for boundary in range(len(boundaries) - 1):
        members = order[boundaries[boundary] : boundaries[boundary + 1]]
        if len(members) <= 1:
            continue
        block = np.asarray(values[members], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        block = np.divide(block, norms, out=np.zeros_like(block), where=norms > 1.0e-12)
        bge = np.clip(block @ block.T, 0.0, 1.0)
        name_sets = [(_surface_ngrams(names[row]), _surface_ngrams(aliases[row])) for row in members]
        address_sets = [_surface_ngrams(addresses[row]) for row in members]
        for local_source, source_row in enumerate(members):
            ranked: list[tuple[float, int, float, float, float]] = []
            source_global = int(global_rows[source_row])
            for local_target, target_row in enumerate(members):
                if local_source == local_target:
                    continue
                target_global = int(global_rows[target_row])
                if (min(source_global, target_global), max(source_global, target_global)) in protected_pairs:
                    continue
                name_similarity = max(
                    _jaccard(left, right)
                    for left in name_sets[local_source]
                    for right in name_sets[local_target]
                )
                address_similarity = _jaccard(address_sets[local_source], address_sets[local_target])
                bge_similarity = float(bge[local_source, local_target])
                score = (
                    float(weights["bge"]) * bge_similarity
                    + float(weights["name_alias"]) * name_similarity
                    + float(weights["category"])
                    + float(weights["address"]) * address_similarity
                    + float(weights["geo"])
                )
                if score + 1.0e-12 < threshold:
                    continue
                ranked.append((score, int(target_row), bge_similarity, name_similarity, address_similarity))
            ranked.sort(key=lambda item: (-item[0], int(global_rows[item[1]])))
            for score, target_row, bge_similarity, name_similarity, address_similarity in ranked[:max_neighbors]:
                output.append(
                    {
                        "poi_row_index": source_global,
                        "neighbor_poi_row_index": int(global_rows[target_row]),
                        "parent_group": int(parents[source_row]),
                        "fine_category_index": int(categories[source_row]),
                        "bge_similarity": bge_similarity,
                        "name_alias_similarity": name_similarity,
                        "address_similarity": address_similarity,
                        "same_gid6": True,
                        "composite_similarity": score,
                    }
                )
    table = pa.Table.from_pylist(output, schema=HARD_EDGE_SCHEMA)
    if output:
        sources = np.searchsorted(global_rows, table["poi_row_index"].to_numpy(zero_copy_only=False))
        targets = np.searchsorted(global_rows, table["neighbor_poi_row_index"].to_numpy(zero_copy_only=False))
        edge_weights = table["composite_similarity"].to_numpy(zero_copy_only=False).astype(np.float32)
    else:
        sources = np.empty(0, dtype=np.int64)
        targets = np.empty(0, dtype=np.int64)
        edge_weights = np.empty(0, dtype=np.float32)
    return HardEntityGraph(sources.astype(np.int64), targets.astype(np.int64), edge_weights), table


def _selected_array(source: np.ndarray, selected: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    identity = len(source) == len(selected) and np.array_equal(selected, np.arange(len(source), dtype=np.int64))
    if identity and source.dtype == np.dtype(dtype):
        return source
    return np.asarray(source[selected], dtype=dtype)


def load_local_codebook_inputs(config: LocalCodebookConfig, *, gate: str) -> LocalCodebookInputs:
    """Load one P7 gate from P6 and build only Train-derived S3 supervision."""
    summary = inspect_local_codebook_inputs(config, gate=gate)
    settings = _gate(config, gate)
    p6_manifest_path = _manifest_path(config, gate)
    p6_dir = p6_manifest_path.parent
    selected = np.load(p6_dir / "selected_poi_rows.npy", allow_pickle=False)
    nodes = pq.read_table(p6_dir / "query_nodes.parquet")
    depths = nodes["supervision_depth"].to_numpy(zero_copy_only=False)
    d3_mask = depths == 3
    d3_query_rows = nodes["node_row"].to_numpy(zero_copy_only=False)[d3_mask].astype(np.int64)

    p4_manifest_path = Path(config.base.frozen_inputs["p4_full_manifest"]["path"])
    p4_manifest = json.loads(p4_manifest_path.read_text(encoding="utf-8"))
    p4_dir = p4_manifest_path.parent
    all_edges = pq.read_table(p4_dir / "query_poi_edges.parquet")
    d3_values = pa.array(d3_query_rows, type=pa.int64())
    s3_edges = all_edges.filter(
        pc.and_(pc.equal(all_edges["layer"], 3), pc.is_in(all_edges["node_row"], value_set=d3_values))
    )
    false_mask_all = pq.read_table(p4_dir / "false_negative_mask.parquet")
    false_mask = false_mask_all.filter(pc.is_in(false_mask_all["node_row"], value_set=d3_values))
    if len(s3_edges) != int(settings["s3_edge_rows"]):
        raise LocalCodebookDataError("P7 S3 edge 行数不匹配")
    edge_nodes = s3_edges["node_row"].to_numpy(zero_copy_only=False)
    if not np.array_equal(edge_nodes, d3_query_rows):
        raise LocalCodebookDataError("P7 D3 必须每条 Query 恰好一条有序 S3 exact edge")
    target_rows = s3_edges["poi_row_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    target_positions = np.searchsorted(selected, target_rows)
    if np.any(target_positions >= len(selected)) or not np.array_equal(selected[target_positions], target_rows):
        raise LocalCodebookDataError("P7 D3 exact target 未进入冻结 P6 POI 闭包")

    metadata = _catalog_metadata(config, selected)
    category = _read_category_mapping(config.base.base.query_depth_dir / "category_mapping.parquet")
    fine_all = category["fine_category_index"].to_numpy(zero_copy_only=False)
    fine = np.asarray(fine_all[selected], dtype=np.int32)
    if metadata["category_code"].to_pylist() != category.take(pa.array(selected))["fine_category_id"].to_pylist():
        raise LocalCodebookDataError("P7 catalog category_code 与冻结类别 mapping 错位")
    longitudes = metadata["lng"].to_numpy(zero_copy_only=False)
    latitudes = metadata["lat"].to_numpy(zero_copy_only=False)
    gid6 = encode_geohash_tokens(longitudes, latitudes, length=6)
    p6_poi_prefix = np.load(p6_dir / "poi_assignments_s1_s2.npy", mmap_mode="r", allow_pickle=False)
    p6_query_prefix = np.load(p6_dir / "query_assignments_s1_s2.npy", mmap_mode="r", allow_pickle=False)
    parents = parent_group_ids(gid6, p6_poi_prefix[:, 0], p6_poi_prefix[:, 1])
    geo_features, geo_stats, singleton = local_geo_features(longitudes, latitudes, gid6, parents)
    protected = _protected_pairs(nodes, s3_edges, false_mask, selected)

    poi_embeddings_source = np.load(config.base.base.upstream.base.poi_embeddings, mmap_mode="r", allow_pickle=False)
    poi_embeddings = _selected_array(poi_embeddings_source, selected, np.float16)
    hard_graph, hard_table = build_hard_entity_graph(
        selected,
        poi_embeddings,
        metadata,
        parents,
        fine,
        protected,
        weights=config.hard_graph["score_weights"],
        threshold=float(config.hard_graph["score_threshold"]),
        max_neighbors=int(config.hard_graph["max_neighbors"]),
    )
    metadata = pa.Table.from_arrays(
        [
            *metadata.columns,
            pa.array([bytes(row.tolist()) for row in gid6], type=pa.binary(6)),
            pa.array(parents, type=pa.int64()),
            pa.array(singleton, type=pa.bool_()),
            pa.array(fine, type=pa.int32()),
        ],
        schema=P7_POI_METADATA_SCHEMA,
    )
    metadata = metadata.replace_schema_metadata({b"hard_edge_rows": str(len(hard_table)).encode("ascii")})

    p5_manifest_path = Path(config.base.frozen_inputs["p5_full_manifest"]["path"])
    p5_manifest = json.loads(p5_manifest_path.read_text(encoding="utf-8"))
    p5_dir = p5_manifest_path.parent
    initial_all = np.load(p5_dir / "poi_assignments_s3.npy", mmap_mode="r", allow_pickle=False)
    query_residual_all = np.load(p6_dir / "query_residual_after_s2.npy", mmap_mode="r", allow_pickle=False)
    source_hashes = {
        "p6_manifest_sha256": sha256_file(p6_manifest_path),
        "p5_manifest_sha256": sha256_file(p5_manifest_path),
        "p4_manifest_sha256": sha256_file(p4_manifest_path),
        "p4_s3_edges_sha256": p4_manifest["artifacts"]["query_poi_edges.parquet"]["sha256"],
        "p4_false_negative_mask_sha256": p4_manifest["artifacts"]["false_negative_mask.parquet"]["sha256"],
        "active_catalog_manifest_sha256": str(config.poi_catalog["manifest_sha256"]),
        "active_poi_ids_sha256": str(config.poi_catalog["poi_ids_sha256"]),
        "poi_embeddings_sha256": p5_manifest["contract"]["source_hashes"]["embedding_sha256"],
    }
    return LocalCodebookInputs(
        selected_poi_rows=selected,
        p6_query_nodes=nodes,
        d3_query_rows=d3_query_rows,
        s3_edges=s3_edges,
        false_negative_mask=false_mask,
        poi_metadata=metadata,
        poi_embeddings=poi_embeddings,
        poi_residual_after_s2=np.load(p6_dir / "poi_residual_after_s2.npy", mmap_mode="r", allow_pickle=False),
        query_residual_after_s2=np.asarray(query_residual_all[d3_mask], dtype=np.float16),
        p6_poi_assignments_s1_s2=p6_poi_prefix,
        p6_query_assignments_s1_s2=p6_query_prefix,
        initial_poi_codebook_s3=np.load(p5_dir / "poi_codebook_s3.npy", allow_pickle=False),
        initial_poi_assignments_s3=_selected_array(initial_all, selected, np.int32),
        fine_category_indices=fine,
        gid6_codes=gid6,
        parent_groups=parents,
        geo_features=geo_features,
        geo_standardization=geo_stats,
        singleton_mask=singleton,
        hard_graph=hard_graph,
        hard_edges=hard_table,
        protected_pairs=protected,
        source_hashes=source_hashes,
        total_poi_rows=summary["poi_rows"],
        embedding_dim=summary["embedding_dim"],
    )
