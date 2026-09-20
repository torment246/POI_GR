from __future__ import annotations

import math
import unittest

from qg_prqk.config import QueryStatsConfig
from qg_prqk.data.query_statistics import (
    false_negative_targets,
    stable_query_shard,
    summarize_query,
)


def settings() -> QueryStatsConfig:
    return QueryStatsConfig(
        num_shards=8,
        buffer_rows_per_shard=4,
        read_batch_rows=16,
        min_query_count=2,
        min_pair_count=2,
        min_top1_share=0.75,
        min_margin=0.25,
        max_normalized_entropy=0.55,
        false_negative_min_share=0.10,
        false_negative_min_count=2,
    )


class QueryStatisticsTest(unittest.TestCase):
    def test_high_confidence_statistics_and_weight(self) -> None:
        summary = summarize_query(
            "北京南站", {"poi-a": 9, "poi-b": 1}, {"北京南站": 10}, settings()
        )
        self.assertEqual(summary.top1_poi_id, "poi-a")
        self.assertAlmostEqual(summary.top1_share, 0.9)
        self.assertAlmostEqual(summary.top2_share, 0.1)
        self.assertAlmostEqual(summary.margin, 0.8)
        expected_entropy = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
        self.assertAlmostEqual(summary.entropy, expected_entropy)
        self.assertAlmostEqual(summary.normalized_entropy, expected_entropy / math.log(2))
        self.assertTrue(summary.is_high_confidence)
        self.assertGreater(summary.sid_weight, 0.0)

    def test_single_observation_is_not_retained(self) -> None:
        summary = summarize_query("首都机场", {"poi-a": 1}, {"首都机场": 1}, settings())
        self.assertFalse(summary.is_high_confidence)
        self.assertEqual(summary.sid_weight, 0.0)
        self.assertEqual(summary.normalized_entropy, 0.0)

    def test_ambiguous_query_is_not_retained(self) -> None:
        summary = summarize_query(
            "万达广场", {"poi-a": 2, "poi-b": 2}, {"万达广场": 4}, settings()
        )
        self.assertFalse(summary.is_high_confidence)
        self.assertEqual(summary.margin, 0.0)
        self.assertAlmostEqual(summary.normalized_entropy, 1.0)

    def test_false_negative_mask_keeps_every_reasonable_positive(self) -> None:
        targets = false_negative_targets(
            {"poi-a": 7, "poi-b": 2, "poi-c": 1}, settings()
        )
        self.assertEqual(targets, ("poi-a", "poi-b"))

    def test_ties_and_representative_raw_query_are_deterministic(self) -> None:
        first = summarize_query(
            "abc", {"poi-b": 2, "poi-a": 2}, {"ABC": 2, "abc": 2}, settings()
        )
        second = summarize_query(
            "abc", {"poi-a": 2, "poi-b": 2}, {"abc": 2, "ABC": 2}, settings()
        )
        self.assertEqual(first, second)
        self.assertEqual(first.top1_poi_id, "poi-a")
        self.assertEqual(first.representative_raw_query, "ABC")

    def test_stable_qg_hash(self) -> None:
        self.assertEqual(stable_query_shard("北京南站", 256), stable_query_shard("北京南站", 256))
        self.assertGreaterEqual(stable_query_shard("北京南站", 256), 0)
        self.assertLess(stable_query_shard("北京南站", 256), 256)


if __name__ == "__main__":
    unittest.main()
