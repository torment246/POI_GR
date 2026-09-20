from __future__ import annotations

import copy
import unittest

from qg_prqk.data.query_graph_data import QueryGraphDataError, assemble_graph


def synthetic_graph() -> tuple:
    depths, edges = [], []
    pois = {
        "p0": {
            "poi_row_index": 0,
            "fine_category_id": "100001",
            "coarse_category_id": "10",
        },
        "p1": {
            "poi_row_index": 1,
            "fine_category_id": "100001",
            "coarse_category_id": "10",
        },
    }
    for qid, depth in ((3, 1), (8, 2), (20, 3)):
        count = 1 if depth == 3 else 2
        depths.append(
            {
                "query_id": qid,
                "query_shard_id": 0,
                "normalized_query": f"query{qid}",
                "query_count": 10,
                "supervision_depth": depth,
                "support_score": 0.7,
                "dominant_fine_category_id": "100001",
                "dominant_coarse_category_id": "10",
                **{f"supervise_s{i}": i <= depth for i in (1, 2, 3)},
                **{
                    f"s{i}_reliability": 0.4 + i * 0.1 if i <= depth else 0
                    for i in (1, 2, 3)
                },
                "retained_poi_edge_count": count,
                "retained_layer_edge_count": count * depth,
            }
        )
        for layer in range(1, depth + 1):
            for i in range(count):
                probability = 1 / count
                edges.append(
                    {
                        "query_id": qid,
                        "query_shard_id": 0,
                        "normalized_query": f"query{qid}",
                        "target_poi_id": f"p{i}",
                        "pair_count": 2,
                        "pair_share": 0.2,
                        "supervision_depth": depth,
                        "supervision_label": f"D{depth}",
                        "layer": layer,
                        "hierarchy_source": "synthetic",
                        "fine_category_id": "100001",
                        "coarse_category_id": "10",
                        "edge_probability": probability,
                        "query_reliability": 0.4 + layer * 0.1,
                        "edge_weight": probability * (0.4 + layer * 0.1),
                    }
                )
    mask = [
        {"query_id": 20, "target_poi_id": "p0"},
        {"query_id": 20, "target_poi_id": "p1"},
    ]
    return depths, edges, mask, pois, {20: (1, "query20", "p0")}


class QueryGraphDataTest(unittest.TestCase):
    def test_depth_views_and_reliability_mass_are_preserved(self) -> None:
        nodes, edges, mask = assemble_graph(*synthetic_graph(), poi_rows=2)
        self.assertEqual(nodes["query_id"].to_pylist(), [3, 8, 20])
        self.assertEqual(
            nodes["query_view"].to_pylist(), ["raw_bge", "raw_bge", "final_adapter"]
        )
        self.assertEqual(nodes["d3_cache_row"].to_pylist(), [-1, -1, 1])
        self.assertEqual(len(edges), 9)
        self.assertEqual(len(mask), 2)
        self.assertAlmostEqual(
            sum(e["edge_weight"] for e in edges.to_pylist() if e["query_id"] == 3), 0.5
        )

    def test_rejects_depth_leak_and_duplicate_edges(self) -> None:
        for change in ("layer", "duplicate"):
            args = list(synthetic_graph())
            if change == "layer":
                args[1][0]["layer"] = 3
            else:
                args[1].append(copy.deepcopy(args[1][0]))
            with self.assertRaises(QueryGraphDataError):
                assemble_graph(*args, poi_rows=2)

    def test_rejects_lost_reliability_or_missing_edges(self) -> None:
        for change in ("weight", "missing"):
            args = list(synthetic_graph())
            if change == "weight":
                args[1][0]["edge_weight"] = args[1][0]["edge_probability"]
            else:
                args[1].pop()
            with self.assertRaises(QueryGraphDataError):
                assemble_graph(*args, poi_rows=2)

    def test_rejects_d3_text_mismatch_and_secondary_strong_target(self) -> None:
        args = list(synthetic_graph())
        args[4] = {20: (1, "wrong", "p0")}
        with self.assertRaisesRegex(QueryGraphDataError, "文本"):
            assemble_graph(*args, poi_rows=2)
        args = list(synthetic_graph())
        args[1][-1]["target_poi_id"] = "p1"
        with self.assertRaisesRegex(QueryGraphDataError, "强对齐"):
            assemble_graph(*args, poi_rows=2)

    def test_rejects_d0_missing_poi_and_nonmonotonic_ids(self) -> None:
        for change in ("D0", "poi", "ids"):
            args = list(synthetic_graph())
            if change == "D0":
                args[0][0]["supervision_depth"] = 0
            elif change == "poi":
                del args[3]["p1"]
            else:
                args[0].reverse()
            with self.assertRaises(QueryGraphDataError):
                assemble_graph(*args, poi_rows=2)


if __name__ == "__main__":
    unittest.main()
