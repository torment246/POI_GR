"""Synthetic tests for bucket expansion and lightweight resolver ranking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.bucket_rerank.evaluation import (  # noqa: E402
    ExpandedCandidate,
    PoiMetadata,
    canonical_key,
    expand_trace_buckets,
    lexical_relevance,
    normalize_text,
    rank_candidates,
)
from poi_gr.methods.tiger.eval import TigerIdIndex  # noqa: E402


class BucketRerankTest(unittest.TestCase):
    def setUp(self) -> None:
        codes = ((1, 2, 3, 0), (1, 2, 3, 1), (7, 8, 9, 0))
        keys = np.asarray([TigerIdIndex.pack(value) for value in codes], dtype=np.int64)
        order = np.argsort(keys)
        self.index = TigerIdIndex(
            sorted_keys=keys[order],
            sorted_rows=order.astype(np.int64),
            poi_ids=pa.array(["poi-a", "poi-b", "poi-c"]),
        )

    def test_text_normalization_and_alias_match(self) -> None:
        metadata = PoiMetadata(
            row=0,
            poi_id="poi-a",
            displayname="中国移动(上地店)",
            address="北京市海淀区上地",
            alias="移动营业厅｜中国移动上地",
            category="生活服务:通信营业厅",
            lng=116.3,
            lat=39.9,
        )
        self.assertEqual(normalize_text("中国移动（上地）"), "中国移动上地")
        self.assertEqual(lexical_relevance("中国移动上地", metadata), 1.0)

    def test_canonical_key_ignores_alias_order(self) -> None:
        left = PoiMetadata(0, "a", "酒店", "地址", "甲｜乙", "住宿", 1.0, 2.0)
        right = PoiMetadata(1, "b", "酒店", "地址", "乙｜甲", "住宿", 1.0, 2.0)
        self.assertEqual(canonical_key(left), canonical_key(right))

    def test_expand_trace_buckets_preserves_bucket_order(self) -> None:
        trace = {
            "unique_expandable_buckets": [
                {
                    "codes": [7, 8, 9],
                    "first_beam_rank": 1,
                    "first_sequence_score": -0.1,
                    "bucket_size": 1,
                },
                {
                    "codes": [1, 2, 3],
                    "first_beam_rank": 3,
                    "first_sequence_score": -0.5,
                    "bucket_size": 2,
                },
            ]
        }
        expanded = expand_trace_buckets(trace, self.index)
        self.assertEqual([item.row for item in expanded], [2, 0, 1])
        self.assertEqual([item.bucket_order for item in expanded], [1, 2, 2])

    def test_bucket_and_global_ranking_have_distinct_contracts(self) -> None:
        candidates = (
            ExpandedCandidate(0, 1, 1, -0.1),
            ExpandedCandidate(1, 1, 1, -0.1),
            ExpandedCandidate(2, 2, 2, -0.2),
        )
        lexical = np.asarray([0.1, 0.9, 1.0])
        semantic = np.asarray([0.2, 0.8, 0.95])
        popularity = np.asarray([3.0, 1.0, 2.0])
        bucket_order = rank_candidates(
            candidates,
            lexical_scores=lexical,
            semantic_scores=semantic,
            popularity=popularity,
            variant="bucket_then_lexical",
        )
        global_order = rank_candidates(
            candidates,
            lexical_scores=lexical,
            semantic_scores=semantic,
            popularity=popularity,
            variant="global_lexical",
        )
        self.assertEqual(bucket_order.tolist(), [1, 0, 2])
        self.assertEqual(global_order.tolist(), [2, 1, 0])

    def test_legacy_bge_variant_matches_semantic_variant(self) -> None:
        candidates = (
            ExpandedCandidate(0, 1, 1, -0.1),
            ExpandedCandidate(1, 1, 1, -0.1),
        )
        lexical = np.asarray([0.1, 0.2])
        semantic = np.asarray([0.3, 0.9])
        popularity = np.asarray([1.0, 2.0])
        legacy = rank_candidates(
            candidates,
            lexical_scores=lexical,
            semantic_scores=semantic,
            popularity=popularity,
            variant="global_bge",
        )
        current = rank_candidates(
            candidates,
            lexical_scores=lexical,
            semantic_scores=semantic,
            popularity=popularity,
            variant="global_semantic",
        )
        np.testing.assert_array_equal(legacy, current)


if __name__ == "__main__":
    unittest.main()
