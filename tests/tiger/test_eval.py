"""Synthetic tests for paper-aligned TIGER invalid-ID evaluation."""

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

from poi_gr.methods.tiger.eval import (  # noqa: E402
    TigerCandidate,
    TigerIdIndex,
    empty_metrics,
    finalize_metrics,
    parse_target_codes,
    update_metrics,
)


class TigerEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        codes = ((1, 2, 3, 0), (1, 2, 3, 1), (7, 8, 9, 0))
        keys = np.asarray([TigerIdIndex.pack(value) for value in codes], dtype=np.int64)
        order = np.argsort(keys)
        self.index = TigerIdIndex(
            sorted_keys=keys[order],
            sorted_rows=order.astype(np.int64),
            poi_ids=pa.array(["poi-a", "poi-b", "poi-c"]),
        )

    def test_four_code_lookup_is_unique(self) -> None:
        self.assertEqual(self.index.lookup((1, 2, 3, 0)), 0)
        self.assertEqual(self.index.lookup((1, 2, 3, 1)), 1)
        self.assertEqual(self.index.lookup((1, 2, 4, 0)), -1)

    def test_target_serialization_is_strict(self) -> None:
        self.assertEqual(
            parse_target_codes("<TARGET_POI><S1_1><S2_2><S3_3><C_4></TARGET_POI>"),
            (1, 2, 3, 4),
        )
        with self.assertRaisesRegex(ValueError, "严格 TIGER"):
            parse_target_codes("<S1_1><S2_2><S3_3><C_4>")

    def test_invalid_beam_keeps_its_rank_slot(self) -> None:
        metrics = empty_metrics()
        candidates = (
            TigerCandidate(None, None, -0.1, "identifier_not_in_corpus"),
            TigerCandidate((1, 2, 3, 0), 0, -0.2, None),
        )
        rank = update_metrics(metrics, target_row=0, candidates=candidates)
        result = finalize_metrics(metrics)
        self.assertEqual(rank, 2)
        self.assertEqual(result["hr@1"], 0.0)
        self.assertEqual(result["hr@3"], 1.0)
        self.assertEqual(result["invalid_id_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
