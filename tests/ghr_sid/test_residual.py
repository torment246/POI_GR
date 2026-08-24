"""Tests for GHR-SID static residual attribution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.residual import (  # noqa: E402
    collision_partition_stats,
    find_feature_irreducible_groups,
    normalize_static_text,
)
from poi_gr.methods.qgr_sid.relations import RELATION_TYPES  # noqa: E402


class ResidualGroupingTest(unittest.TestCase):
    def test_finds_only_full_static_signature_collisions(self) -> None:
        base = np.asarray([1, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        gid8 = np.asarray(
            [
                [1] * 8,
                [1] * 8,
                [1] * 8,
                [1, 1, 1, 1, 1, 1, 1, 2],
                [2] * 8,
                [2] * 8,
                [2] * 8,
            ],
            dtype=np.uint8,
        )
        relations = np.full((7, len(RELATION_TYPES)), -1, dtype=np.int16)
        relations[2, 0] = 9
        grouping = find_feature_irreducible_groups(
            base_sid_keys=base,
            gid8_codes=gid8,
            strict_relations=relations,
        )
        self.assertEqual(grouping.collision_rows.tolist(), [0, 1, 4, 5, 6])
        self.assertEqual(sorted(grouping.group_sizes.tolist()), [2, 3])
        self.assertEqual(len(np.unique(grouping.group_ids)), 2)
        for group_id, size in enumerate(grouping.group_sizes):
            self.assertEqual(
                len(grouping.member_order[
                    grouping.group_offsets[group_id] : grouping.group_offsets[group_id + 1]
                ]),
                size,
            )

    def test_partition_stats_count_excess_and_collision_rows(self) -> None:
        stats = collision_partition_stats(
            [0, 0, 0, 1, 1], ["a", "a", "b", "x", "y"]
        )
        self.assertEqual(stats["collision_excess"], 1)
        self.assertEqual(stats["collision_poi_count"], 2)
        self.assertEqual(stats["collision_group_count"], 1)
        self.assertEqual(stats["distinct_count"], 4)

    def test_normalization_removes_format_only_differences(self) -> None:
        self.assertEqual(
            normalize_static_text("Ａ座 - Foo １号"),
            normalize_static_text("a座foo1号"),
        )
        self.assertEqual(normalize_static_text(None), "")

    def test_rejects_wrong_gid_shape(self) -> None:
        with self.assertRaises(ValueError):
            find_feature_irreducible_groups(
                base_sid_keys=np.asarray([1, 1]),
                gid8_codes=np.zeros((2, 7), dtype=np.uint8),
                strict_relations=np.zeros((2, len(RELATION_TYPES)), dtype=np.int16),
            )


if __name__ == "__main__":
    unittest.main()
