"""Train-only exact-query selection and POI metadata preparation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.hard_negative_mining import PoiMetadata


SELECTION_SCHEMA_VERSION = "qg-prqk-p3a-d3-selection-v1"


class QueryAdapterDataError(RuntimeError):
    """Raised when real P2/P2.5/POI inputs violate the P3A contract."""


@dataclass(frozen=True)
class D3Query:
    query_id: int
    selection_hash: int
    normalized_query: str
    representative_raw_query: str
    query_count: int
    distinct_poi_count: int
    target_poi_id: str
    top1_count: int
    top1_share: float
    normalized_entropy: float
    fine_category_id: str
    coarse_category_id: str
    query_weight: float


SELECTION_SCHEMA = pa.schema(
    [
        pa.field("gate_row", pa.int64(), nullable=False),
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("selection_hash_hex", pa.string(), nullable=False),
        pa.field("normalized_query", pa.string(), nullable=False),
        pa.field("representative_raw_query", pa.string(), nullable=False),
        pa.field("query_count", pa.int64(), nullable=False),
        pa.field("distinct_poi_count", pa.int32(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
        pa.field("target_poi_row", pa.int64(), nullable=False),
        pa.field("top1_count", pa.int64(), nullable=False),
        pa.field("top1_share", pa.float64(), nullable=False),
        pa.field("normalized_entropy", pa.float64(), nullable=False),
        pa.field("fine_category_id", pa.string(), nullable=False),
        pa.field("coarse_category_id", pa.string(), nullable=False),
        pa.field("query_weight", pa.float64(), nullable=False),
        pa.field("false_negative_count", pa.int32(), nullable=False),
    ]
)


def stable_selection_hash(query_id: int, *, seed: int) -> int:
    """Return the deterministic sampling score used by both sample and Gate."""

    if query_id < 0:
        raise QueryAdapterDataError("query_id 不能为负数")
    digest = hashlib.blake2b(
        f"{seed}:{query_id}".encode("ascii"),
        digest_size=8,
        person=b"qg-p3a-v1",
    ).digest()
    return int.from_bytes(digest, "big")


def _sid_weight(
    top1_count: int, top1_share: float, normalized_entropy: float
) -> float:
    weight = (
        min(math.log1p(top1_count), math.log(21.0))
        * top1_share**2
        * (1.0 - normalized_entropy)
    )
    return max(0.0, weight)


def select_d3_queries(
    query_category_stats_dir: Path,
    *,
    limit: int,
    seed: int,
) -> tuple[list[D3Query], int]:
    """Select the globally smallest stable hashes from all D3 Exact queries."""

    if limit <= 1:
        raise QueryAdapterDataError("D3 selection limit 必须大于 1")
    paths = sorted(query_category_stats_dir.glob("*.parquet"))
    if not paths:
        raise QueryAdapterDataError(f"D3 stats 不存在：{query_category_stats_dir}")
    columns = [
        "query_id",
        "normalized_query",
        "representative_raw_query",
        "query_count",
        "distinct_poi_count",
        "top1_poi_id",
        "top1_count",
        "top1_share",
        "exact_normalized_entropy",
        "is_p2_exact_core",
        "dominant_fine_category_id",
        "dominant_coarse_category_id",
    ]
    heap: list[tuple[int, int, D3Query]] = []
    d3_rows = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=65_536, columns=columns
        ):
            values = batch.to_pydict()
            for index, is_exact in enumerate(values["is_p2_exact_core"]):
                if not is_exact:
                    continue
                d3_rows += 1
                query_id = int(values["query_id"][index])
                score = stable_selection_hash(query_id, seed=seed)
                record = D3Query(
                    query_id=query_id,
                    selection_hash=score,
                    normalized_query=str(values["normalized_query"][index]),
                    representative_raw_query=str(
                        values["representative_raw_query"][index]
                    ),
                    query_count=int(values["query_count"][index]),
                    distinct_poi_count=int(values["distinct_poi_count"][index]),
                    target_poi_id=str(values["top1_poi_id"][index]),
                    top1_count=int(values["top1_count"][index]),
                    top1_share=float(values["top1_share"][index]),
                    normalized_entropy=float(
                        values["exact_normalized_entropy"][index]
                    ),
                    fine_category_id=str(
                        values["dominant_fine_category_id"][index]
                    ),
                    coarse_category_id=str(
                        values["dominant_coarse_category_id"][index]
                    ),
                    query_weight=_sid_weight(
                        int(values["top1_count"][index]),
                        float(values["top1_share"][index]),
                        float(values["exact_normalized_entropy"][index]),
                    ),
                )
                item = (-score, -query_id, record)
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    selected = sorted(
        (item[2] for item in heap),
        key=lambda value: (value.selection_hash, value.query_id),
    )
    if len(selected) != min(limit, d3_rows):
        raise QueryAdapterDataError("D3 selection 行数不守恒")
    if any(not row.normalized_query or row.query_weight <= 0 for row in selected):
        raise QueryAdapterDataError("D3 selection 含空 Query 或非正权重")
    return selected, d3_rows


def load_false_negative_ids(
    false_negative_dir: Path,
    selected_query_ids: Iterable[int],
) -> dict[int, tuple[str, ...]]:
    """Load only selected Query false-negative targets from the frozen P2 mask."""

    selected = set(int(value) for value in selected_query_ids)
    result: dict[int, list[str]] = {query_id: [] for query_id in selected}
    paths = sorted(false_negative_dir.glob("*.parquet"))
    if not paths:
        raise QueryAdapterDataError(f"false-negative mask 不存在：{false_negative_dir}")
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=65_536, columns=["query_id", "target_poi_id"]
        ):
            query_ids = batch.column(0).to_pylist()
            poi_ids = batch.column(1).to_pylist()
            for query_id, poi_id in zip(query_ids, poi_ids, strict=True):
                integer_id = int(query_id)
                if integer_id in selected:
                    result[integer_id].append(str(poi_id))
    return {
        query_id: tuple(sorted(set(values)))
        for query_id, values in result.items()
    }


def resolve_poi_rows(
    poi_ids_path: Path,
    required_poi_ids: Iterable[str],
    *,
    expected_rows: int,
) -> dict[str, int]:
    """Resolve selected POI IDs without retaining the full 2.3M ID mapping."""

    required = set(str(value) for value in required_poi_ids)
    result: dict[str, int] = {}
    rows = 0
    with poi_ids_path.open("r", encoding="utf-8") as handle:
        for rows, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise QueryAdapterDataError(f"POI ID 第 {rows} 行不是合法 JSON") from error
            poi_id = str(raw)
            if poi_id in required:
                if poi_id in result:
                    raise QueryAdapterDataError(f"POI ID 重复：{poi_id}")
                result[poi_id] = rows - 1
    if rows != expected_rows:
        raise QueryAdapterDataError(f"POI ID 行数 {rows} != {expected_rows}")
    missing = sorted(required - set(result))
    if missing:
        raise QueryAdapterDataError(f"P3A target/FN POI 不在 BGE 行序：{missing[:5]}")
    return result


def load_selected_metadata(
    catalog_dir: Path,
    required_rows: Iterable[int],
    expected_ids_by_row: Mapping[int, str],
    *,
    expected_rows: int,
) -> dict[int, PoiMetadata]:
    """Scan the canonical catalog once and retain only ANN/target metadata rows."""

    required = set(int(value) for value in required_rows)
    result: dict[int, PoiMetadata] = {}
    paths = sorted(catalog_dir.glob("part-*.json"))
    if not paths:
        raise QueryAdapterDataError(f"POI catalog JSON 分片不存在：{catalog_dir}")
    row_index = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if row_index in required:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise QueryAdapterDataError(
                            f"POI catalog 非法：{path}:{line_number}"
                        ) from error
                    poi_id = str(raw.get("poi_id", ""))
                    expected = expected_ids_by_row.get(row_index)
                    if expected is not None and poi_id != expected:
                        raise QueryAdapterDataError(
                            f"POI catalog/BGE 行序错位 row={row_index}"
                        )
                    result[row_index] = PoiMetadata(
                        row=row_index,
                        poi_id=poi_id,
                        displayname=str(raw.get("displayname") or ""),
                        alias=str(raw.get("alias") or ""),
                        category=str(raw.get("category") or ""),
                        category_code=str(raw.get("category_code") or ""),
                        address=str(raw.get("address") or ""),
                        lat=float(raw.get("lat")),
                        lng=float(raw.get("lng")),
                    )
                row_index += 1
    if row_index != expected_rows:
        raise QueryAdapterDataError(f"POI catalog 行数 {row_index} != {expected_rows}")
    missing = sorted(required - set(result))
    if missing:
        raise QueryAdapterDataError(f"POI metadata 缺失行：{missing[:5]}")
    return result


def write_selection(
    path: Path,
    queries: Sequence[D3Query],
    target_rows: Sequence[int],
    false_negative_rows: Sequence[Sequence[int]],
) -> None:
    """Persist the exact Gate population and row mapping as one small Parquet."""

    if not (len(queries) == len(target_rows) == len(false_negative_rows)):
        raise QueryAdapterDataError("selection 输出首维不一致")
    table = pa.Table.from_pylist(
        [
            {
                "gate_row": index,
                "query_id": query.query_id,
                "selection_hash_hex": f"{query.selection_hash:016x}",
                "normalized_query": query.normalized_query,
                "representative_raw_query": query.representative_raw_query,
                "query_count": query.query_count,
                "distinct_poi_count": query.distinct_poi_count,
                "target_poi_id": query.target_poi_id,
                "target_poi_row": int(target_rows[index]),
                "top1_count": query.top1_count,
                "top1_share": query.top1_share,
                "normalized_entropy": query.normalized_entropy,
                "fine_category_id": query.fine_category_id,
                "coarse_category_id": query.coarse_category_id,
                "query_weight": query.query_weight,
                "false_negative_count": len(false_negative_rows[index]),
            }
            for index, query in enumerate(queries)
        ],
        schema=SELECTION_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
