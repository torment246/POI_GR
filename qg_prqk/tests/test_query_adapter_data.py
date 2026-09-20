from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.adapters.data import resolve_poi_rows, select_d3_queries


def _row(query_id: int, exact: bool = True) -> dict:
    return {
        "query_id": query_id,
        "normalized_query": f"query-{query_id}",
        "representative_raw_query": f"Query-{query_id}",
        "query_count": 2 + query_id % 5,
        "distinct_poi_count": 1,
        "top1_poi_id": str(1000 + query_id),
        "top1_count": 2,
        "top1_share": 1.0,
        "exact_normalized_entropy": 0.0,
        "is_p2_exact_core": exact,
        "dominant_fine_category_id": "100000",
        "dominant_coarse_category_id": "10",
    }


class QueryAdapterDataTest(unittest.TestCase):
    def test_selection_is_deterministic_and_sample_is_gate_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pq.write_table(pa.Table.from_pylist([_row(i) for i in range(30)]), root / "part-0.parquet")
            sample, total = select_d3_queries(root, limit=5, seed=42)
            gate, second_total = select_d3_queries(root, limit=12, seed=42)
        self.assertEqual((total, second_total), (30, 30))
        self.assertEqual(sample, gate[:5])
        self.assertEqual(
            [(item.selection_hash, item.query_id) for item in sample],
            sorted((item.selection_hash, item.query_id) for item in sample),
        )
        self.assertTrue(all(item.query_weight > 0 for item in sample))

    def test_resolve_only_required_poi_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "poi_ids.jsonl"
            path.write_text('"a"\n"b"\n"c"\n', encoding="utf-8")
            result = resolve_poi_rows(path, {"a", "c"}, expected_rows=3)
        self.assertEqual(result, {"a": 0, "c": 2})


if __name__ == "__main__":
    unittest.main()
