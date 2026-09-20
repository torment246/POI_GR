"""Frozen S3 inputs and S1/S2-parent hard graph without GID."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file
from qg_prqk.data.query_graph_data import EDGE_SCHEMA
from qg_prqk.sid.local_data import (
    HardEntityGraph,
    LocalCodebookDataError,
    _catalog_metadata,
    _jaccard,
    _protected_pairs,
    _read_category_mapping,
    _selected_array,
    _surface_ngrams,
    inspect_local_codebook_inputs,
)
from qg_prqk.sid.nogid_config import NoGIDCodebookConfig
from qg_prqk.sid.nogid_geo import s1_s2_parent_geo_features, s1_s2_parent_keys


HARD_EDGE_SCHEMA = pa.schema(
    [
        pa.field("poi_row_index", pa.int64(), nullable=False),
        pa.field("neighbor_poi_row_index", pa.int64(), nullable=False),
        pa.field("parent_s1", pa.int32(), nullable=False),
        pa.field("parent_s2", pa.int32(), nullable=False),
        pa.field("fine_category_index", pa.int32(), nullable=False),
        pa.field("bge_similarity", pa.float32(), nullable=False),
        pa.field("name_alias_similarity", pa.float32(), nullable=False),
        pa.field("address_similarity", pa.float32(), nullable=False),
        pa.field("composite_similarity", pa.float32(), nullable=False),
    ]
)
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
        pa.field("parent_s1", pa.int32(), nullable=False),
        pa.field("parent_s2", pa.int32(), nullable=False),
        pa.field("parent_key", pa.int64(), nullable=False),
        pa.field("is_singleton_parent", pa.bool_(), nullable=False),
        pa.field("fine_category_index", pa.int32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class NoGIDCodebookInputs:
    """All tensors needed to fit S3 from the frozen S1/S2 endpoint."""

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


def inspect_nogid_codebook_inputs(
    config: NoGIDCodebookConfig,
    *,
    gate: str,
) -> dict[str, Any]:
    """Validate frozen headers without reading catalog values or computing GID."""
    summary = inspect_local_codebook_inputs(config, gate=gate)  # type: ignore[arg-type]
    return {
        **summary,
        "status": "p7_nogid_input_headers_validated",
        "phase": f"P7-CAT-NOGID-{gate.upper()}",
        "parent_key": ["s1", "s2"],
        "gid_read": False,
        "gid_computed": False,
    }


def build_nogid_hard_entity_graph(
    selected_poi_rows: np.ndarray,
    embeddings: np.ndarray,
    metadata: pa.Table,
    parent_s1_s2: np.ndarray,
    fine_categories: np.ndarray,
    protected_pairs: frozenset[tuple[int, int]],
    *,
    weights: Mapping[str, float],
    threshold: float,
    max_neighbors: int,
) -> tuple[HardEntityGraph, pa.Table]:
    """Build deterministic semantic hard edges inside `(S1,S2,fine)` groups."""
    global_rows = np.asarray(selected_poi_rows, dtype=np.int64)
    values = np.asarray(embeddings)
    prefix = np.asarray(parent_s1_s2, dtype=np.int32)
    categories = np.asarray(fine_categories, dtype=np.int32)
    if (
        values.ndim != 2
        or len(values) != len(global_rows)
        or prefix.shape != (len(global_rows), 2)
        or categories.shape != global_rows.shape
        or len(metadata) != len(global_rows)
        or threshold != 0.50
        or max_neighbors != 20
        or set(weights) != {"bge", "name_alias", "category", "address"}
    ):
        raise LocalCodebookDataError("no-GID hard graph 输入或冻结协议不匹配")
    parents = s1_s2_parent_keys(prefix[:, 0], prefix[:, 1])
    names = metadata["displayname"].to_pylist()
    aliases = metadata["alias"].to_pylist()
    addresses = metadata["address"].to_pylist()
    compound = parents * (int(categories.max()) + 1) + categories.astype(np.int64)
    order = np.argsort(compound, kind="stable")
    sorted_keys = compound[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])

    source_rows: list[int] = []
    neighbor_rows: list[int] = []
    parent_s1: list[int] = []
    parent_s2: list[int] = []
    fine_values: list[int] = []
    bge_values: list[float] = []
    name_values: list[float] = []
    address_values: list[float] = []
    score_values: list[float] = []
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
                )
                if score + 1.0e-12 >= threshold:
                    ranked.append((score, int(target_row), bge_similarity, name_similarity, address_similarity))
            ranked.sort(key=lambda item: (-item[0], int(global_rows[item[1]])))
            for score, target_row, bge_similarity, name_similarity, address_similarity in ranked[:max_neighbors]:
                source_rows.append(source_global)
                neighbor_rows.append(int(global_rows[target_row]))
                parent_s1.append(int(prefix[source_row, 0]))
                parent_s2.append(int(prefix[source_row, 1]))
                fine_values.append(int(categories[source_row]))
                bge_values.append(bge_similarity)
                name_values.append(name_similarity)
                address_values.append(address_similarity)
                score_values.append(score)
    table = pa.Table.from_arrays(
        [
            pa.array(source_rows, type=pa.int64()),
            pa.array(neighbor_rows, type=pa.int64()),
            pa.array(parent_s1, type=pa.int32()),
            pa.array(parent_s2, type=pa.int32()),
            pa.array(fine_values, type=pa.int32()),
            pa.array(bge_values, type=pa.float32()),
            pa.array(name_values, type=pa.float32()),
            pa.array(address_values, type=pa.float32()),
            pa.array(score_values, type=pa.float32()),
        ],
        schema=HARD_EDGE_SCHEMA,
    )
    if source_rows:
        sources = np.searchsorted(global_rows, np.asarray(source_rows, dtype=np.int64))
        targets = np.searchsorted(global_rows, np.asarray(neighbor_rows, dtype=np.int64))
        edge_weights = np.asarray(score_values, dtype=np.float32)
    else:
        sources = np.empty(0, dtype=np.int64)
        targets = np.empty(0, dtype=np.int64)
        edge_weights = np.empty(0, dtype=np.float32)
    return HardEntityGraph(sources, targets, edge_weights), table


def load_nogid_codebook_inputs(
    config: NoGIDCodebookConfig,
    *,
    gate: str,
) -> NoGIDCodebookInputs:
    """Load frozen Train-only inputs and derive S1/S2-parent structures."""
    summary = inspect_nogid_codebook_inputs(config, gate=gate)
    settings = config.gates[gate]
    p6_manifest_path = (config.project_root / str(settings["p6_manifest"]["path"])).resolve()
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
    s3_edges = all_edges.filter(pc.and_(pc.equal(all_edges["layer"], 3), pc.is_in(all_edges["node_row"], value_set=d3_values)))
    false_mask_all = pq.read_table(p4_dir / "false_negative_mask.parquet")
    false_mask = false_mask_all.filter(pc.is_in(false_mask_all["node_row"], value_set=d3_values))
    if len(s3_edges) != int(settings["s3_edge_rows"]) or not s3_edges.schema.equals(EDGE_SCHEMA):
        raise LocalCodebookDataError("no-GID S3 edge 行数或 schema 不匹配")
    edge_nodes = s3_edges["node_row"].to_numpy(zero_copy_only=False)
    if not np.array_equal(edge_nodes, d3_query_rows):
        raise LocalCodebookDataError("no-GID D3 Query 必须每条恰好一条有序 S3 edge")

    metadata = _catalog_metadata(config, selected)  # type: ignore[arg-type]
    category = _read_category_mapping(config.base.base.query_depth_dir / "category_mapping.parquet")
    fine_all = category["fine_category_index"].to_numpy(zero_copy_only=False)
    fine = np.asarray(fine_all[selected], dtype=np.int32)
    if metadata["category_code"].to_pylist() != category.take(pa.array(selected))["fine_category_id"].to_pylist():
        raise LocalCodebookDataError("no-GID catalog category_code 与冻结 mapping 错位")

    poi_prefix = np.load(p6_dir / "poi_assignments_s1_s2.npy", mmap_mode="r", allow_pickle=False)
    query_prefix = np.load(p6_dir / "query_assignments_s1_s2.npy", mmap_mode="r", allow_pickle=False)
    parents = s1_s2_parent_keys(poi_prefix[:, 0], poi_prefix[:, 1])
    geo_features, geo_stats, singleton = s1_s2_parent_geo_features(
        metadata["lng"].to_numpy(zero_copy_only=False),
        metadata["lat"].to_numpy(zero_copy_only=False),
        parents,
    )
    protected = _protected_pairs(nodes, s3_edges, false_mask, selected)
    poi_embedding_source = np.load(config.base.base.upstream.base.poi_embeddings, mmap_mode="r", allow_pickle=False)
    poi_embeddings = _selected_array(poi_embedding_source, selected, np.float16)
    hard_graph, hard_edges = build_nogid_hard_entity_graph(
        selected,
        poi_embeddings,
        metadata,
        poi_prefix,
        fine,
        protected,
        weights=config.hard_graph["score_weights"],
        threshold=float(config.hard_graph["score_threshold"]),
        max_neighbors=int(config.hard_graph["max_neighbors"]),
    )
    metadata = pa.Table.from_arrays(
        [
            *metadata.columns,
            pa.array(poi_prefix[:, 0], type=pa.int32()),
            pa.array(poi_prefix[:, 1], type=pa.int32()),
            pa.array(parents, type=pa.int64()),
            pa.array(singleton, type=pa.bool_()),
            pa.array(fine, type=pa.int32()),
        ],
        schema=POI_METADATA_SCHEMA,
    )
    metadata = metadata.replace_schema_metadata({b"hard_edge_rows": str(len(hard_edges)).encode("ascii")})

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
    return NoGIDCodebookInputs(
        selected_poi_rows=selected,
        p6_query_nodes=nodes,
        d3_query_rows=d3_query_rows,
        s3_edges=s3_edges,
        false_negative_mask=false_mask,
        poi_metadata=metadata,
        poi_embeddings=poi_embeddings,
        poi_residual_after_s2=np.load(p6_dir / "poi_residual_after_s2.npy", mmap_mode="r", allow_pickle=False),
        query_residual_after_s2=np.asarray(query_residual_all[d3_mask], dtype=np.float16),
        p6_poi_assignments_s1_s2=poi_prefix,
        p6_query_assignments_s1_s2=query_prefix,
        initial_poi_codebook_s3=np.load(p5_dir / "poi_codebook_s3.npy", allow_pickle=False),
        initial_poi_assignments_s3=_selected_array(initial_all, selected, np.int32),
        fine_category_indices=fine,
        parent_groups=parents,
        geo_features=geo_features,
        geo_standardization=geo_stats,
        singleton_mask=singleton,
        hard_graph=hard_graph,
        hard_edges=hard_edges,
        protected_pairs=protected,
        source_hashes=source_hashes,
        total_poi_rows=int(summary["poi_rows"]),
        embedding_dim=int(summary["embedding_dim"]),
    )
