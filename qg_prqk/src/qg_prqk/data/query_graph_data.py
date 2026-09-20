"""Contracts and bounded readers for the Train-derived query graph."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file
from qg_prqk.data.query_supervision import QUERY_POI_LAYER_EDGE_SCHEMA
from qg_prqk.sid.pipeline_config import DownstreamConfig


SAMPLE_LIMIT = 1000
MEDIUM_LIMIT = 50_000
FULL_LIMIT = 342_879
VECTOR_CHECK_ROWS = 8192


def validate_limit(limit: int, gate: str) -> None:
    """Require a bounded gate or the one frozen full-data row count."""
    if type(limit) is not int:
        raise QueryGraphDataError("P4 limit 必须是整数")
    if gate == "sample" and 1 <= limit <= SAMPLE_LIMIT:
        return
    if gate == "medium" and 1 <= limit <= MEDIUM_LIMIT:
        return
    if gate == "full" and limit == FULL_LIMIT:
        return
    raise QueryGraphDataError(
        "P4 仅开放 sample 1..1000、medium 1..50000 或固定 full 342879"
    )


NODE_SCHEMA = pa.schema(
    [
        pa.field("node_row", pa.int64(), nullable=False),
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("query_shard_id", pa.int32(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("query_count", pa.int64(), nullable=False),
        pa.field("supervision_depth", pa.int8(), nullable=False),
        pa.field("layer_mask", pa.list_(pa.bool_(), 3), nullable=False),
        pa.field("support_score", pa.float64(), nullable=False),
        *[
            pa.field(f"s{i}_reliability", pa.float64(), nullable=False)
            for i in (1, 2, 3)
        ],
        pa.field("dominant_fine_category_id", pa.string(), nullable=False),
        pa.field("dominant_coarse_category_id", pa.string(), nullable=False),
        pa.field("query_view", pa.string(), nullable=False),
        pa.field("d3_cache_row", pa.int64(), nullable=False),
        pa.field("exact_target_poi_id", pa.string(), nullable=False),
    ]
)
EDGE_SCHEMA = pa.schema(
    [
        pa.field("node_row", pa.int64(), nullable=False),
        pa.field("poi_row_index", pa.int64(), nullable=False),
        *QUERY_POI_LAYER_EDGE_SCHEMA,
    ]
)
MASK_SCHEMA = pa.schema(
    [
        pa.field("node_row", pa.int64(), nullable=False),
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("poi_row_index", pa.int64(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
    ]
)


class QueryGraphDataError(ValueError):
    """Raised when graph data differ from the frozen P2.5/P3A contract."""


@dataclass(frozen=True)
class QueryGraphInputs:
    nodes: pa.Table
    edges: pa.Table
    false_negatives: pa.Table
    source_files: Mapping[str, str]
    cache_dir: Path
    cache_manifest: Mapping[str, Any]
    poi_rows: int
    embedding_dim: int

    def signature(self) -> str:
        payload = {
            "nodes": self.nodes.to_pylist(),
            "edges": self.edges.to_pylist(),
            "false_negatives": self.false_negatives.to_pylist(),
            "source_files": dict(self.source_files),
            "poi_rows": self.poi_rows,
            "embedding_dim": self.embedding_dim,
        }
        digest = hashlib.sha256()
        encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False)
        # Preserve the v1 byte contract without materializing a second giant JSON.
        for piece in encoder.iterencode(payload):
            digest.update(piece.encode("utf-8"))
        return digest.hexdigest()

    def streaming_signature(self) -> str:
        """Hash Arrow batches without converting the full graph to Python rows."""
        digest = hashlib.sha256()
        for name, table in (
            ("nodes", self.nodes),
            ("edges", self.edges),
            ("false_negatives", self.false_negatives),
        ):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(table.schema).encode("utf-8") + b"\0")
            digest.update(len(table).to_bytes(8, "big"))
            for batch in table.to_batches(max_chunksize=8192):
                serialized = batch.serialize()
                digest.update(len(serialized).to_bytes(8, "big"))
                digest.update(serialized)
        metadata = {
            "source_files": dict(self.source_files),
            "poi_rows": self.poi_rows,
            "embedding_dim": self.embedding_dim,
        }
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()


def assemble_graph(
    depths: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    false_negatives: Sequence[Mapping[str, Any]],
    pois: Mapping[str, Mapping[str, Any]],
    d3_lookup: Mapping[int, tuple[int, str, str]],
    *,
    poi_rows: int,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Map stable query/POI IDs to rows without changing supervision mass."""
    nodes = []
    by_id = {}
    previous_id = -1
    for record in depths:
        qid, depth = int(record["query_id"]), int(record["supervision_depth"])
        if qid <= previous_id or depth not in (1, 2, 3):
            raise QueryGraphDataError("Query ID 必须严格递增，节点只允许 D1/D2/D3")
        previous_id = qid
        if not record["normalized_query"] or int(record["query_count"]) <= 0:
            raise QueryGraphDataError("Query 文本/计数非法")
        if not 0 < float(record["support_score"]) <= 1:
            raise QueryGraphDataError("capped support 超出 (0,1]")
        for layer in (1, 2, 3):
            reliability = float(record[f"s{layer}_reliability"])
            if bool(record[f"supervise_s{layer}"]) != (layer <= depth):
                raise QueryGraphDataError("Query layer mask 与监督深度冲突")
            if not np.isfinite(reliability) or not 0 <= reliability <= 1:
                raise QueryGraphDataError("Query reliability 非法")
            if layer > depth and reliability != 0:
                raise QueryGraphDataError("未启用层的 reliability 必须为 0")
        cache_row, exact_target = -1, ""
        if depth == 3:
            if qid not in d3_lookup:
                raise QueryGraphDataError("D3 Query 缺少冻结 cache 行")
            cache_row, text, exact_target = d3_lookup[qid]
            if cache_row < 0 or text != record["normalized_query"]:
                raise QueryGraphDataError("D3 cache 的 Query ID/文本不匹配")
        node = {
            field.name: record[field.name]
            for field in NODE_SCHEMA
            if field.name in record
        }
        node.update(
            node_row=len(nodes),
            query_view="final_adapter" if depth == 3 else "raw_bge",
            d3_cache_row=cache_row,
            exact_target_poi_id=exact_target,
            layer_mask=[i <= depth for i in (1, 2, 3)],
        )
        nodes.append(node)
        by_id[qid] = node

    def poi_row(poi_id: str) -> int:
        if poi_id not in pois:
            raise QueryGraphDataError(f"图/FN target 不在 active POI mapping：{poi_id}")
        row = int(pois[poi_id]["poi_row_index"])
        if not 0 <= row < poi_rows:
            raise QueryGraphDataError("POI 行号越界")
        return row

    mapped_edges = []
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    seen_edges = set()
    for edge in edges:
        qid, layer = int(edge["query_id"]), int(edge["layer"])
        if qid not in by_id:
            raise QueryGraphDataError("边引用未知 Query 节点")
        node = by_id[qid]
        depth = node["supervision_depth"]
        pid = str(edge["target_poi_id"])
        row = poi_row(pid)
        key = (qid, layer, pid)
        if key in seen_edges or not 1 <= layer <= depth:
            raise QueryGraphDataError("重复边或越层边（D1/D2 不得进入 S3）")
        seen_edges.add(key)
        if (
            edge["normalized_query"] != node["normalized_query"]
            or edge["supervision_depth"] != depth
            or edge["query_shard_id"] != node["query_shard_id"]
        ):
            raise QueryGraphDataError("边与 Query 节点字段错位")
        if any(
            edge[k] != pois[pid][k] for k in ("fine_category_id", "coarse_category_id")
        ):
            raise QueryGraphDataError("边类别与 POI mapping 不一致")
        if depth < 3:
            category = "coarse" if depth == 1 else "fine"
            if (
                edge[f"{category}_category_id"]
                != node[f"dominant_{category}_category_id"]
            ):
                raise QueryGraphDataError("类别 Query 的边不属于保留类别")
        elif pid != node["exact_target_poi_id"]:
            raise QueryGraphDataError("D3 次要合理目标不能作为强对齐边")
        reliability = node[f"s{layer}_reliability"]
        probability, weight = (
            float(edge["edge_probability"]),
            float(edge["edge_weight"]),
        )
        if (
            not 0 < probability <= 1
            or not np.isfinite(weight)
            or abs(edge["query_reliability"] - reliability) > 1e-9
            or abs(weight - probability * reliability) > 1e-9
            or int(edge["pair_count"]) <= 0
        ):
            raise QueryGraphDataError("边概率/最终边权与 reliability 冲突")
        groups.setdefault((qid, layer), []).append(edge)
        mapped_edges.append(dict(edge, node_row=node["node_row"], poi_row_index=row))
    for record in depths:
        qid = record["query_id"]
        count = 0
        for layer in range(1, record["supervision_depth"] + 1):
            group = groups.get((qid, layer), [])
            total = sum(e["pair_count"] for e in group)
            if not group or abs(sum(e["edge_probability"] for e in group) - 1) > 1e-9:
                raise QueryGraphDataError("Query 启用层缺边或条件概率不守恒")
            if any(
                abs(e["edge_probability"] - e["pair_count"] / total) > 1e-9
                for e in group
            ):
                raise QueryGraphDataError("条件概率不是保留 pair count 的归一结果")
            if len(group) != record["retained_poi_edge_count"]:
                raise QueryGraphDataError("保留 POI 边数量不守恒")
            count += len(group)
        if count != record["retained_layer_edge_count"]:
            raise QueryGraphDataError("层级边数量不守恒")
    masks, seen_masks = [], set()
    for record in false_negatives:
        qid, pid = int(record["query_id"]), str(record["target_poi_id"])
        if qid not in by_id or (qid, pid) in seen_masks:
            raise QueryGraphDataError("FN 引用未知 Query 或存在重复 target")
        seen_masks.add((qid, pid))
        masks.append(
            {
                "node_row": by_id[qid]["node_row"],
                "query_id": qid,
                "target_poi_id": pid,
                "poi_row_index": poi_row(pid),
            }
        )
    mapped_edges.sort(key=lambda e: (e["node_row"], e["layer"], e["poi_row_index"]))
    masks.sort(key=lambda e: (e["node_row"], e["poi_row_index"]))
    return (
        pa.Table.from_pylist(nodes, schema=NODE_SCHEMA),
        pa.Table.from_pylist(mapped_edges, schema=EDGE_SCHEMA),
        pa.Table.from_pylist(masks, schema=MASK_SCHEMA),
    )


def graph_metrics(inputs: QueryGraphInputs, *, bounded_memory: bool = False) -> dict[str, Any]:
    """Compute graph counts and conservation errors.

    ``bounded_memory`` uses Arrow aggregation for the full gate and avoids Python
    dictionaries for every graph edge.
    """
    if bounded_memory:
        depths = inputs.nodes["supervision_depth"]
        layers = inputs.edges["layer"]
        grouped = inputs.edges.group_by(["node_row", "layer"]).aggregate(
            [("edge_probability", "sum"), ("edge_weight", "sum")]
        )
        node_rows = grouped["node_row"].to_numpy(zero_copy_only=False)
        group_layers = grouped["layer"].to_numpy(zero_copy_only=False)
        probability_sums = grouped["edge_probability_sum"].to_numpy(
            zero_copy_only=False
        )
        weight_sums = grouped["edge_weight_sum"].to_numpy(zero_copy_only=False)
        reliability = np.column_stack(
            [
                inputs.nodes[f"s{layer}_reliability"].to_numpy(
                    zero_copy_only=False
                )
                for layer in (1, 2, 3)
            ]
        )
        return {
            "query_rows": len(inputs.nodes),
            "depth_counts": {
                f"D{depth}": int(pc.sum(pc.equal(depths, depth)).as_py())
                for depth in (1, 2, 3)
            },
            "layer_edge_rows": {
                f"S{layer}": int(pc.sum(pc.equal(layers, layer)).as_py())
                for layer in (1, 2, 3)
            },
            "edge_rows": len(inputs.edges),
            "false_negative_rows": len(inputs.false_negatives),
            "covered_poi_rows": int(
                pc.count_distinct(inputs.edges["poi_row_index"]).as_py()
            ),
            "max_probability_sum_error": float(
                np.max(np.abs(probability_sums - 1), initial=0)
            ),
            "max_weight_sum_error": float(
                np.max(
                    np.abs(
                        weight_sums
                        - reliability[node_rows, group_layers.astype(np.int64) - 1]
                    ),
                    initial=0,
                )
            ),
        }
    nodes, edges = inputs.nodes.to_pylist(), inputs.edges.to_pylist()
    probability_sums: Counter[tuple[int, int]] = Counter()
    weight_sums: Counter[tuple[int, int]] = Counter()
    for edge in edges:
        key = (edge["node_row"], edge["layer"])
        probability_sums[key] += edge["edge_probability"]
        weight_sums[key] += edge["edge_weight"]
    return {
        "query_rows": len(nodes),
        "depth_counts": {
            f"D{d}": sum(n["supervision_depth"] == d for n in nodes) for d in (1, 2, 3)
        },
        "layer_edge_rows": {
            f"S{d}": sum(e["layer"] == d for e in edges) for d in (1, 2, 3)
        },
        "edge_rows": len(edges),
        "false_negative_rows": len(inputs.false_negatives),
        "covered_poi_rows": len(set(e["poi_row_index"] for e in edges)),
        "max_probability_sum_error": max(
            (abs(v - 1) for v in probability_sums.values()), default=0
        ),
        "max_weight_sum_error": max(
            (
                abs(value - nodes[node_row][f"s{layer}_reliability"])
                for (node_row, layer), value in weight_sums.items()
            ),
            default=0,
        ),
    }


def _child(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise QueryGraphDataError("manifest 文件路径越界")
    return path


def _checked(path: Path, expected: str, sources: dict[str, str]) -> Path:
    if sha256_file(path) != expected:
        raise QueryGraphDataError(f"冻结来源 SHA256 不匹配：{path}")
    sources[str(path.resolve())] = expected
    return path


def _group_rows(
    root: Path,
    manifest: Mapping[str, Any],
    name: str,
    sources: dict[str, str],
    *,
    shard_ids: set[int] | None = None,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    group = manifest["outputs"][name]
    directory = _child(root, group["dir"])
    for entry in sorted(group["files"], key=lambda e: e["shard_id"]):
        if shard_ids is not None and entry["shard_id"] not in shard_ids:
            continue
        path = _checked(_child(directory, entry["file"]), entry["sha256"], sources)
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != entry["rows"]:
            raise QueryGraphDataError("冻结 Parquet 行数不符")
        for batch in parquet.iter_batches(batch_size=8192, columns=columns):
            yield from batch.to_pylist()


def prepare_query_graph_inputs(
    config: DownstreamConfig, *, limit: int = SAMPLE_LIMIT, gate: str = "sample"
) -> QueryGraphInputs:
    """Read a bounded active-Query prefix, never raw orders or business splits."""
    validate_limit(limit, gate)
    sources: dict[str, str] = {}
    manifests = {}
    for name, entry in config.upstream_artifacts.items():
        path = _checked(Path(entry["path"]), entry["sha256"], sources)
        if name != "final_adapter_checkpoint":
            manifests[name] = json.loads(path.read_text(encoding="utf-8"))
            if manifests[name].get("status") != "completed":
                raise QueryGraphDataError(f"上游未完成：{name}")
    base = config.upstream.base
    category = base.category_config
    p25 = manifests["p2_5_manifest"]
    full = manifests["p3a_full_manifest"]
    final = manifests["final_adapter_manifest"]
    cache = manifests["d3_query_cache_manifest"]
    p2_root = category.paths.p2_query_stats
    p2_path = _checked(
        p2_root / "manifest.json", category.frozen.p2_manifest_sha256, sources
    )
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    if (
        p2.get("status") != "completed"
        or p2["source"]["split"] != "train"
        or p2["source"]["valid_and_test_read"] is not False
        or p25["source_access"]["raw_train_read"] is not False
        or p25["source_access"]["validation_read"] is not False
        or p25["source_access"]["test_read"] is not False
        or p25["inputs"]["p2_manifest"]["sha256"] != category.frozen.p2_manifest_sha256
    ):
        raise QueryGraphDataError("P2/P2.5 来源不是冻结 Train-only 产物")
    if (
        final.get("schema_version") != "qg-prqk-p3a-full-final-v1"
        or final["config_signature"] != config.upstream.signature()
        or final["query_view_policy"] != config.query_view_policy
        or final["training"]["training_rows"] != config.upstream.d3_rows
        or final["artifacts"]["query_adapter_exact_final.pt"]["sha256"]
        != config.upstream_artifacts["final_adapter_checkpoint"]["sha256"]
    ):
        raise QueryGraphDataError("FINAL manifest 角色/配置/全 D3 来源不符")
    expected_contract = {
        "schema_version": "qg-prqk-d3-query-chunk-cache-v1",
        "config_signature": config.upstream.signature(),
        "rows": config.upstream.d3_rows,
        "embedding_dim": category.frozen.poi_embedding_dim,
        "dtype": "float16",
        "chunk_rows": base.query_embedding.encode_buffer_size,
    }
    if any(cache["contract"].get(k) != v for k, v in expected_contract.items()):
        raise QueryGraphDataError("D3 cache 编码配置/行数与冻结 P3A-FULL 不一致")

    depths = []
    next_query_id = 0
    for row in _group_rows(
        config.query_depth_dir, p25, "query_category_depth", sources
    ):
        if row["query_id"] != next_query_id:
            raise QueryGraphDataError("P2.5 全局 Query ID 不连续")
        next_query_id += 1
        if row["supervision_depth"] > 0:
            if gate == "full" and len(depths) == limit:
                raise QueryGraphDataError("full active Query 行数超过冻结值 342879")
            depths.append(row)
            if gate != "full" and len(depths) == limit:
                break
    if len(depths) != limit:
        raise QueryGraphDataError(f"可用 Query 数与 {gate} limit 不一致")
    selected_ids = {n["query_id"] for n in depths}
    shard_ids = {n["query_shard_id"] for n in depths}
    edges = [
        e
        for e in _group_rows(
            config.query_depth_dir,
            p25,
            "query_poi_layer_edges",
            sources,
            shard_ids=shard_ids,
        )
        if e["query_id"] in selected_ids
    ]
    masks = [
        e
        for e in _group_rows(
            p2_root,
            p2,
            "false_negative_mask",
            sources,
            shard_ids=shard_ids,
            columns=["query_id", "target_poi_id"],
        )
        if e["query_id"] in selected_ids
    ]
    needed_pois = {e["target_poi_id"] for e in edges + masks}
    pois = {}
    expected_row = 0
    for row in _group_rows(
        config.query_depth_dir,
        p25,
        "category_mapping",
        sources,
        columns=["poi_id", "poi_row_index", "fine_category_id", "coarse_category_id"],
    ):
        if row["poi_row_index"] != expected_row:
            raise QueryGraphDataError("active POI mapping 行序不连续")
        expected_row += 1
        if row["poi_id"] in needed_pois:
            if row["poi_id"] in pois:
                raise QueryGraphDataError("active POI mapping 存在重复 ID")
            pois[row["poi_id"]] = row
    if expected_row != category.frozen.poi_rows:
        raise QueryGraphDataError("active POI mapping 总行数错误")

    selection_entry = full["artifacts"]["d3_full_selection.parquet"]
    selection_path = _checked(
        config.upstream.output_dir / "d3_full_selection.parquet",
        selection_entry["sha256"],
        sources,
    )
    if selection_entry["sha256"] != final["source_hashes"]["full_d3_selection_sha256"]:
        raise QueryGraphDataError("FINAL 与 D3 selection 来源不一致")
    d3_ids = {n["query_id"] for n in depths if n["supervision_depth"] == 3}
    lookup = {}
    query_hash = hashlib.sha256()
    selection_rows = 0
    for batch in pq.ParquetFile(selection_path).iter_batches(
        batch_size=8192,
        columns=["gate_row", "query_id", "normalized_query", "target_poi_id"],
    ):
        for row in batch.to_pylist():
            if row["gate_row"] != selection_rows:
                raise QueryGraphDataError("D3 selection cache 行序不连续")
            query_hash.update(
                json.dumps(
                    [row["query_id"], row["normalized_query"]], ensure_ascii=False
                ).encode("utf-8")
                + b"\n"
            )
            if row["query_id"] in d3_ids:
                if row["query_id"] in lookup:
                    raise QueryGraphDataError("D3 selection Query ID 重复")
                lookup[row["query_id"]] = (
                    selection_rows,
                    row["normalized_query"],
                    row["target_poi_id"],
                )
            selection_rows += 1
    if (
        selection_rows != config.upstream.d3_rows
        or query_hash.hexdigest() != cache["contract"]["query_id_text_sha256"]
    ):
        raise QueryGraphDataError("D3 cache 与完整 selection 的 Query ID/文本/行序哈希不匹配")
    nodes, mapped_edges, mapped_masks = assemble_graph(
        depths, edges, masks, pois, lookup, poi_rows=category.frozen.poi_rows
    )
    cache_dir = Path(
        config.upstream_artifacts["d3_query_cache_manifest"]["path"]
    ).parent
    cache_rows = nodes["d3_cache_row"].to_numpy(zero_copy_only=False)
    needed_cache_rows = cache_rows[cache_rows >= 0]
    stop = 0
    for chunk in cache["chunks"]:
        if chunk["start"] != stop or chunk["stop"] != min(
            stop + cache["contract"]["chunk_rows"], selection_rows
        ):
            raise QueryGraphDataError("D3 cache 分块不连续")
        stop = chunk["stop"]
        if np.any(
            (needed_cache_rows >= chunk["start"]) & (needed_cache_rows < stop)
        ):
            sources[str(_child(cache_dir, chunk["file"]))] = chunk["sha256"]
    if stop != selection_rows:
        raise QueryGraphDataError("D3 cache 分块不完整")
    return QueryGraphInputs(
        nodes,
        mapped_edges,
        mapped_masks,
        sources,
        cache_dir,
        cache,
        category.frozen.poi_rows,
        category.frozen.poi_embedding_dim,
    )


class D3RawReader:
    """Read frozen D3 vectors by node interval through verified mmap chunks."""

    def __init__(self, inputs: QueryGraphInputs) -> None:
        self.inputs = inputs
        depths = inputs.nodes["supervision_depth"].to_numpy(zero_copy_only=False)
        self.node_rows = np.flatnonzero(depths == 3)
        all_cache_rows = inputs.nodes["d3_cache_row"].to_numpy(zero_copy_only=False)
        self.cache_rows = all_cache_rows[self.node_rows]
        if np.any(self.cache_rows < 0):
            raise QueryGraphDataError("D3 Query 缺少冻结 cache 行")
        self._arrays: dict[str, np.ndarray] = {}

    def _array(self, chunk: Mapping[str, Any]) -> np.ndarray:
        filename = str(chunk["file"])
        if filename not in self._arrays:
            path = _child(self.inputs.cache_dir, filename)
            _checked(path, str(chunk["sha256"]), {})
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            expected = (
                int(chunk["stop"]) - int(chunk["start"]),
                self.inputs.embedding_dim,
            )
            if values.dtype != np.float16 or values.shape != expected:
                raise QueryGraphDataError("D3 chunk shape/dtype 错误")
            self._arrays[filename] = values
        return self._arrays[filename]

    def read(self, start: int, stop: int) -> np.ndarray:
        """Return one node-aligned interval; D1/D2 positions remain zero."""
        if not 0 <= start <= stop <= len(self.inputs.nodes):
            raise QueryGraphDataError("D3 读取区间越界")
        result = np.zeros((stop - start, self.inputs.embedding_dim), dtype=np.float16)
        left = int(np.searchsorted(self.node_rows, start, side="left"))
        right = int(np.searchsorted(self.node_rows, stop, side="left"))
        selected_nodes = self.node_rows[left:right]
        selected_cache = self.cache_rows[left:right]
        filled = np.zeros(len(selected_nodes), dtype=bool)
        for chunk in self.inputs.cache_manifest["chunks"]:
            chunk_start, chunk_stop = int(chunk["start"]), int(chunk["stop"])
            mask = (selected_cache >= chunk_start) & (selected_cache < chunk_stop)
            if not np.any(mask):
                continue
            values = self._array(chunk)
            result[selected_nodes[mask] - start] = values[
                selected_cache[mask] - chunk_start
            ]
            filled[mask] = True
        if not np.all(filled):
            raise QueryGraphDataError("D3 复用行数不守恒")
        if len(selected_nodes):
            selected_values = result[selected_nodes - start].astype(np.float32)
            if not np.isfinite(selected_values).all() or np.any(
                np.abs(np.linalg.norm(selected_values, axis=1) - 1) > 2e-3
            ):
                raise QueryGraphDataError("D3 cache 有非有限值或非单位向量")
        return result

    __call__ = read


def read_d3_raw(inputs: QueryGraphInputs) -> np.ndarray:
    """Read and hash only referenced immutable D3 chunks; never re-encode D3."""
    result = np.empty((len(inputs.nodes), inputs.embedding_dim), dtype=np.float16)
    reader = D3RawReader(inputs)
    for start in range(0, len(inputs.nodes), VECTOR_CHECK_ROWS):
        stop = min(start + VECTOR_CHECK_ROWS, len(inputs.nodes))
        result[start:stop] = reader(start, stop)
    return result
