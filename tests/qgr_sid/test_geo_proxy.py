"""Tests for QGR-SID M2-B static GEO and request-GID selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.geo_proxy import (  # noqa: E402
    derive_geo_candidates,
    select_geo_encoding,
)


class GeoProxyTest(unittest.TestCase):
    def test_candidates_use_bucket_common_prefix_and_numeric_encoding(self) -> None:
        keys = np.asarray([10, 10, 10, 20, 20], dtype=np.int64)
        gids = np.asarray(
            [
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 7, 8, 9],
                [1, 2, 3, 7, 8, 10],
                [9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.uint8,
        )
        prefixes, candidates, order, starts = derive_geo_candidates(
            bucket_keys=keys, poi_gid_codes=gids
        )
        self.assertEqual(prefixes.tolist(), [3, 3, 3, 6, 6])
        self.assertEqual(candidates[:3, 0].tolist(), [4, 7, 7])
        self.assertEqual(
            candidates[:3, 1].tolist(),
            [32 + 4 * 32 + 5, 32 + 7 * 32 + 8, 32 + 7 * 32 + 8],
        )
        self.assertTrue(np.all(candidates[3:] == -1))
        self.assertEqual(order.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(starts.tolist(), [0, 3])

    def test_gid_guidance_can_choose_longer_more_predictable_geo_value(self) -> None:
        candidates = np.asarray(
            [[1, 100], [1, 101], [2, 102], [2, 103]], dtype=np.int16
        )
        early_orders = np.asarray([100, 100, 100, 100], dtype=np.int32)
        matches = np.asarray(
            [[0, 90], [0, 90], [0, 90], [0, 90]], dtype=np.int32
        )
        order = np.arange(4, dtype=np.int64)
        starts = np.asarray([0], dtype=np.int64)
        static = select_geo_encoding(
            geo_candidates=candidates,
            early_orders=early_orders,
            early_geo_matches=matches,
            bucket_order=order,
            bucket_starts=starts,
            query_guided=False,
            p99_order_count=100,
            prior_orders=20,
        )
        guided = select_geo_encoding(
            geo_candidates=candidates,
            early_orders=early_orders,
            early_geo_matches=matches,
            bucket_order=order,
            bucket_starts=starts,
            query_guided=True,
            p99_order_count=100,
            prior_orders=20,
        )
        self.assertEqual(static.lengths.tolist(), [2, 2, 2, 2])
        self.assertEqual(guided.lengths.tolist(), [2, 2, 2, 2])
        self.assertTrue(guided.resolved.all())

    def test_static_prefers_more_distinguishing_two_character_candidate(self) -> None:
        candidates = np.asarray(
            [[1, 100], [1, 101], [2, 102], [2, 103]], dtype=np.int16
        )
        result = select_geo_encoding(
            geo_candidates=candidates,
            early_orders=np.ones(4, dtype=np.int32),
            early_geo_matches=np.zeros((4, 2), dtype=np.int32),
            bucket_order=np.arange(4, dtype=np.int64),
            bucket_starts=np.asarray([0], dtype=np.int64),
            query_guided=False,
            p99_order_count=1,
            prior_orders=20,
        )
        self.assertEqual(result.values.tolist(), [100, 101, 102, 103])
        self.assertTrue(result.resolved.all())


if __name__ == "__main__":
    unittest.main()
