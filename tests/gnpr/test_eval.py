"""Synthetic tests for unconstrained GNPR identifier evaluation."""

from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

from poi_gr.methods.gnpr.eval import (
    GnprCandidate,
    GnprIdIndex,
    GnprTokenIds,
    empty_metrics,
    finalize_metrics,
    parse_generated_candidate,
    parse_target_codes,
    update_metrics,
)


class GnprEvalTest(unittest.TestCase):
    def test_dynamic_dedup_capacity_keeps_keys_distinct(self) -> None:
        for capacity in (2, 100, 223, 300):
            values = [[1, 2, 3, -1], [1, 2, 3, 0], [1, 2, 3, capacity - 1], [1, 2, 4, -1]]
            keys = np.asarray([GnprIdIndex.pack(value, capacity) for value in values], dtype=np.int64)
            self.assertEqual(len(set(keys)), 4)
            order = np.argsort(keys)
            index = GnprIdIndex(sorted_keys=keys[order], sorted_rows=order,
                                poi_ids=None, dedup_capacity=capacity)
            for row, value in enumerate(values):
                self.assertEqual(index.lookup(value), row)
            self.assertEqual(index.lookup([1, 2, 3, capacity]), -1)
            self.assertEqual(parse_target_codes(
                f"<TARGET_POI><a_1><b_2><c_3><d_{capacity - 1}></TARGET_POI>",
                dedup_capacity=capacity), (1, 2, 3, capacity - 1))

    def setUp(self) -> None:
        codes = ((1, 2, 3, -1), (1, 2, 3, 0), (7, 8, 9, 2))
        keys = np.asarray([GnprIdIndex.pack(value) for value in codes], dtype=np.int64)
        order = np.argsort(keys)
        self.index = GnprIdIndex(
            sorted_keys=keys[order],
            sorted_rows=order.astype(np.int64),
            poi_ids=pa.array(["poi-a", "poi-b", "poi-c"]),
        )
        self.tokens = GnprTokenIds(
            target_open=10,
            target_close=11,
            a=tuple(range(100, 612)),
            b=tuple(range(700, 1212)),
            c=tuple(range(1300, 1812)),
            dedup=tuple(range(1900, 2123)),
            eos=2,
        )

    def test_singleton_and_collision_lookups_are_distinct(self) -> None:
        self.assertEqual(self.index.lookup((1, 2, 3, -1)), 0)
        self.assertEqual(self.index.lookup((1, 2, 3, 0)), 1)
        self.assertEqual(self.index.lookup((1, 2, 3, 1)), -1)

    def test_target_serialization_accepts_conditional_dedup(self) -> None:
        self.assertEqual(
            parse_target_codes("<TARGET_POI><a_1><b_2><c_3></TARGET_POI>"),
            (1, 2, 3, -1),
        )
        self.assertEqual(
            parse_target_codes("<TARGET_POI><a_1><b_2><c_3><d_0></TARGET_POI>"),
            (1, 2, 3, 0),
        )
        with self.assertRaisesRegex(ValueError, "严格 GNPR"):
            parse_target_codes("<a_1><b_2><c_3>")

    def test_generated_singleton_and_collision_are_parsed(self) -> None:
        singleton = parse_generated_candidate(
            (10, 101, 702, 1303, 11, 2, 0),
            -0.1,
            tokens=self.tokens,
            index=self.index,
        )
        collision = parse_generated_candidate(
            (10, 101, 702, 1303, 1900, 11, 2),
            -0.2,
            tokens=self.tokens,
            index=self.index,
        )
        self.assertEqual(singleton.poi_row, 0)
        self.assertEqual(singleton.codes, (1, 2, 3, -1))
        self.assertEqual(collision.poi_row, 1)
        self.assertEqual(collision.codes, (1, 2, 3, 0))

    def test_invalid_beam_keeps_its_rank_slot(self) -> None:
        metrics = empty_metrics()
        candidates = (
            GnprCandidate(None, None, -0.1, "identifier_not_in_corpus"),
            GnprCandidate((1, 2, 3, -1), 0, -0.2, None),
        )
        rank = update_metrics(metrics, target_row=0, candidates=candidates)
        result = finalize_metrics(metrics)
        self.assertEqual(rank, 2)
        self.assertEqual(result["hr@1"], 0.0)
        self.assertEqual(result["hr@3"], 1.0)
        self.assertEqual(result["valid_id_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
