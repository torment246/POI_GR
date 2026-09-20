from __future__ import annotations

import unittest

import numpy as np

from qg_prqk.adapters.gate import (
    _ranks_and_margins,
    normalize_encoded_queries,
    retrieval_metrics,
)


class QueryAdapterGateTest(unittest.TestCase):
    def test_rank_one_target_still_uses_following_hardest_negative(self) -> None:
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        poi = np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
        ranks, positive, hardest = _ranks_and_margins(
            query,
            poi,
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([[1.0, 0.8]], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            [(0,)],
        )
        self.assertEqual(ranks.tolist(), [1])
        self.assertTrue(np.allclose(positive, [1.0]))
        self.assertTrue(np.allclose(hardest, [0.8]))

    def test_bf16_like_vectors_are_explicitly_renormalized(self) -> None:
        values = np.asarray([[1.003, 0.0], [0.0, 0.997]], dtype=np.float32)
        normalized = normalize_encoded_queries(values)
        self.assertTrue(
            np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-7)
        )

    def test_retrieval_metrics_uses_requested_subset(self) -> None:
        ranks = np.asarray([1, 11, 51], dtype=np.int32)
        positive = np.asarray([0.9, 0.5, 0.2], dtype=np.float32)
        hardest = np.asarray([0.8, 0.7, 0.6], dtype=np.float32)
        metrics = retrieval_metrics(
            ranks,
            positive,
            hardest,
            np.asarray([True, True, False]),
        )
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_10"], 0.5)
        self.assertAlmostEqual(metrics["mean_hard_margin"], -0.05, places=6)


if __name__ == "__main__":
    unittest.main()
