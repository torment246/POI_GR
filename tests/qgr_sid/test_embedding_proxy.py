"""Tests for QGR-SID Query-center and lexical-first residual ranking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.embedding_proxy import (  # noqa: E402
    rank_embedding_candidates,
)


class EmbeddingProxyRankingTest(unittest.TestCase):
    def test_embedding_similarity_precedes_popularity(self) -> None:
        rank = rank_embedding_candidates(
            candidates=np.asarray([0, 1, 2]),
            target=2,
            similarities=np.asarray([0.1, 0.2, 0.9]),
            early_orders=np.asarray([100, 50, 1]),
            poi_ids=np.asarray([10, 11, 12]),
        )
        self.assertEqual(rank, 1)

    def test_lexical_evidence_precedes_embedding_and_embedding_breaks_ties(self) -> None:
        candidates = np.asarray([0, 1, 2])
        early = np.asarray([100, 50, 1])
        poi_ids = np.asarray([10, 11, 12])
        lexical = (
            np.asarray([1, 1, 0]),
            np.asarray([0, 0, 0]),
            np.asarray([True, True, False]),
        )
        rank = rank_embedding_candidates(
            candidates=candidates,
            target=1,
            similarities=np.asarray([0.2, 0.8, 1.0]),
            early_orders=early,
            poi_ids=poi_ids,
            lexical_keys=lexical,
        )
        self.assertEqual(rank, 1)
        losing_rank = rank_embedding_candidates(
            candidates=candidates,
            target=2,
            similarities=np.asarray([0.2, 0.8, 1.0]),
            early_orders=early,
            poi_ids=poi_ids,
            lexical_keys=lexical,
        )
        self.assertEqual(losing_rank, 3)


if __name__ == "__main__":
    unittest.main()
