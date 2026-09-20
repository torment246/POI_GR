from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

from qg_prqk.data.query_graph_data import EDGE_SCHEMA, NODE_SCHEMA
from qg_prqk.sid.relational_data import (
    RelationalCodebookDataError,
    _select_or_reuse_full_array,
    build_graph_closed_selection,
)


def _node(row: int, depth: int, reliability: float) -> dict:
    return {
        "node_row": row,
        "query_id": row + 10,
        "query_shard_id": 0,
        "normalized_query": f"query-{row}",
        "query_count": 3,
        "supervision_depth": depth,
        "layer_mask": [True, depth >= 2, depth >= 3],
        "support_score": reliability,
        "s1_reliability": reliability,
        "s2_reliability": reliability if depth >= 2 else 0.0,
        "s3_reliability": reliability if depth >= 3 else 0.0,
        "dominant_fine_category_id": "121000",
        "dominant_coarse_category_id": "12",
        "query_view": "final_adapter" if depth == 3 else "raw_bge",
        "d3_cache_row": row if depth == 3 else -1,
        "exact_target_poi_id": str(row + 100) if depth == 3 else "",
    }


def _edge(node: int, poi: int, layer: int, probability: float, reliability: float) -> dict:
    return {
        "node_row": node,
        "poi_row_index": poi,
        "query_id": node + 10,
        "query_shard_id": 0,
        "normalized_query": f"query-{node}",
        "target_poi_id": str(poi + 100),
        "pair_count": 1,
        "pair_share": probability,
        "supervision_depth": 2 if node == 0 else 1,
        "supervision_label": "D2_FINE" if node == 0 else "D1_COARSE",
        "layer": layer,
        "hierarchy_source": "dominant_fine_category" if node == 0 else "dominant_coarse_category",
        "fine_category_id": "121000",
        "coarse_category_id": "12",
        "edge_probability": probability,
        "query_reliability": reliability,
        "edge_weight": probability * reliability,
    }


class RelationalCodebookSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = pa.Table.from_pylist(
            [_node(0, 2, 0.8), _node(1, 1, 0.6)], schema=NODE_SCHEMA
        )
        self.edges = pa.Table.from_pylist(
            [
                _edge(0, 2, 1, 0.25, 0.8),
                _edge(0, 3, 1, 0.75, 0.8),
                _edge(0, 2, 2, 0.25, 0.8),
                _edge(0, 3, 2, 0.75, 0.8),
                _edge(1, 4, 1, 1.0, 0.6),
            ],
            schema=EDGE_SCHEMA,
        )

    def test_adds_all_graph_targets_without_changing_edge_mass(self) -> None:
        selected = build_graph_closed_selection(
            self.nodes,
            self.edges,
            base_poi_rows=np.array([0, 1], dtype=np.int64),
            selected_query_rows=np.array([0, 1], dtype=np.int64),
            total_poi_rows=8,
        )
        np.testing.assert_array_equal(selected.selected_poi_rows, [0, 1, 2, 3, 4])
        self.assertEqual(selected.added_graph_target_rows, 3)
        self.assertEqual(len(selected.edges), 5)

    def test_rejects_partial_query_edges(self) -> None:
        partial = self.edges.slice(0, 1)
        with self.assertRaisesRegex(RelationalCodebookDataError, "截断|缺少"):
            build_graph_closed_selection(
                self.nodes.slice(0, 1),
                partial,
                base_poi_rows=np.array([0], dtype=np.int64),
                selected_query_rows=np.array([0], dtype=np.int64),
                total_poi_rows=8,
            )

    def test_rejects_unsorted_or_duplicate_rows(self) -> None:
        for query_rows in (
            np.array([1, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        ):
            with self.subTest(query_rows=query_rows):
                with self.assertRaises(RelationalCodebookDataError):
                    build_graph_closed_selection(
                        self.nodes,
                        self.edges,
                        base_poi_rows=np.array([0, 1], dtype=np.int64),
                        selected_query_rows=query_rows,
                        total_poi_rows=8,
                    )

    def test_reuses_identity_selection_but_copies_a_subset(self) -> None:
        source = np.arange(12, dtype=np.float16).reshape(4, 3)
        full = _select_or_reuse_full_array(
            source, np.arange(4, dtype=np.int64), dtype=np.float16
        )
        subset = _select_or_reuse_full_array(
            source, np.array([1, 3], dtype=np.int64), dtype=np.float16
        )
        self.assertTrue(np.shares_memory(source, full))
        self.assertFalse(np.shares_memory(source, subset))
        np.testing.assert_array_equal(subset, source[[1, 3]])


if __name__ == "__main__":
    unittest.main()
