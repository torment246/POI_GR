from __future__ import annotations

import unittest

from poi_gr.methods.ghr_sid.distinct_vector_relation_audit import (
    collapse_exact_vector_classes,
    summarize_pair_distances,
)


class DistinctVectorRelationAuditTest(unittest.TestCase):
    def test_collapse_keeps_mixed_group_vector_classes(self) -> None:
        collapsed = collapse_exact_vector_classes(
            [10, 10, 10, 20, 20, 30, 30],
            ["a", "a", "b", "c", "c", "d", "e"],
        )
        self.assertEqual(collapsed.representative_rows.tolist(), [0, 2, 5, 6])
        self.assertEqual(collapsed.original_group_ids.tolist(), [10, 10, 30, 30])
        self.assertEqual(collapsed.class_sizes.tolist(), [2, 1, 1, 1])
        self.assertEqual(collapsed.grouping.group_ids.tolist(), [0, 0, 1, 1])
        self.assertEqual(collapsed.grouping.group_sizes.tolist(), [2, 2])

    def test_distance_summary_uses_unordered_pairs(self) -> None:
        summary = summarize_pair_distances(
            [[0, 1], [2, 3]],
            [116.0, 116.0, 116.0, 116.0],
            [40.0, 40.0, 40.0, 40.001],
        )
        self.assertEqual(summary["pair_count"], 2)
        self.assertEqual(summary["same_coordinate_pair_count"], 1)
        self.assertAlmostEqual(summary["same_coordinate_pair_ratio"], 0.5)
        self.assertGreater(summary["pair_distance_meters"]["max"], 100)


if __name__ == "__main__":
    unittest.main()
