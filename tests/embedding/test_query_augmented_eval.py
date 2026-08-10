"""Tests for sparse Query-augmented retrieval helpers."""

from __future__ import annotations

import unittest

import numpy as np

from poi_gr.embedding.query_augmented_eval import (
    adaptive_alpha_from_effective_count,
    category_residual_query_rows,
    fuse_embedding_chunk,
    paired_rank_comparison,
    rank_metrics,
    target_ranks_from_topk,
)


class QueryAugmentedEvalTest(unittest.TestCase):
    def test_category_residual_removes_shared_projection(self) -> None:
        query = np.asarray(
            [[1.0, 0.0], [2**-0.5, 2**-0.5]], dtype=np.float32
        )
        mean = query.mean(axis=0, keepdims=True)
        residual = category_residual_query_rows(
            query,
            np.asarray([0, 0], dtype=np.int32),
            mean,
            beta=0.5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(residual, axis=1), 1.0, atol=1e-6
        )
        self.assertTrue(np.all(residual @ mean[0] < query @ mean[0]))

    def test_complete_category_centering_is_supported(self) -> None:
        residual = category_residual_query_rows(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([[0.5, 0.5]], dtype=np.float32),
            beta=1.0,
        )
        np.testing.assert_allclose(
            np.linalg.norm(residual, axis=1), 1.0, atol=1e-6
        )
        np.testing.assert_allclose(residual[0], -residual[1], atol=1e-6)

    def test_sparse_fusion_applies_category_residual_online(self) -> None:
        base = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float16)
        aggregates = np.asarray([[0.8, 0.6], [0.6, 0.8]], dtype=np.float16)
        positions = np.asarray([0, 1], dtype=np.int32)
        plain = fuse_embedding_chunk(
            base, aggregates, positions, start=0, stop=2, alpha=0.3
        )
        residual = fuse_embedding_chunk(
            base,
            aggregates,
            positions,
            start=0,
            stop=2,
            alpha=0.3,
            category_by_aggregate=np.asarray([0, 0], dtype=np.int32),
            category_query_means=np.asarray([[0.7, 0.7]], dtype=np.float32),
            residual_beta=0.5,
        )
        self.assertFalse(np.array_equal(plain, residual))
        np.testing.assert_allclose(
            np.linalg.norm(residual, axis=1), 1.0, atol=1e-6
        )

    def test_adaptive_alpha_decreases_with_heterogeneity(self) -> None:
        alpha = adaptive_alpha_from_effective_count(
            np.asarray([1.0, 5.0, 20.0], dtype=np.float32),
            alpha_min=0.2,
            alpha_max=0.45,
            tau=5.0,
            gamma=1.0,
        )
        self.assertTrue(np.all(alpha[:-1] > alpha[1:]))
        self.assertTrue(np.all(alpha >= 0.2))
        self.assertTrue(np.all(alpha <= 0.45))

    def test_sparse_fusion_normalizes_only_covered_rows(self) -> None:
        base = np.asarray(
            [[1.0, 0.0], [0.0, 0.9], [2**-0.5, 2**-0.5]],
            dtype=np.float16,
        )
        aggregates = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float16)
        positions = np.asarray([0, -1, 1], dtype=np.int32)
        fused = fuse_embedding_chunk(
            base,
            aggregates,
            positions,
            start=0,
            stop=3,
            alpha=0.5,
            fusion_chunk_rows=2,
        )
        np.testing.assert_allclose(fused[0], [2**-0.5, 2**-0.5], atol=1e-5)
        np.testing.assert_allclose(fused[1], base[1].astype(np.float32), atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(fused[2])), 1.0, places=5)

    def test_sparse_fusion_accepts_per_aggregate_alpha(self) -> None:
        base = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
        aggregates = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float16)
        fused = fuse_embedding_chunk(
            base,
            aggregates,
            np.asarray([0, 1], dtype=np.int32),
            start=0,
            stop=2,
            alpha=np.asarray([0.2, 0.4], dtype=np.float32),
        )
        self.assertGreater(float(fused[1, 0]), float(fused[0, 1]))

    def test_constant_alpha_array_matches_scalar_fusion(self) -> None:
        base = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]],
            dtype=np.float16,
        )
        aggregates = np.asarray(
            [[0.0, 1.0], [1.0, 0.0]], dtype=np.float16
        )
        positions = np.asarray([0, -1, 1], dtype=np.int32)
        scalar = fuse_embedding_chunk(
            base,
            aggregates,
            positions,
            start=0,
            stop=3,
            alpha=0.3,
        )
        adaptive = fuse_embedding_chunk(
            base,
            aggregates,
            positions,
            start=0,
            stop=3,
            alpha=np.full(len(aggregates), 0.3, dtype=np.float32),
        )
        np.testing.assert_array_equal(adaptive, scalar)

    def test_metrics_and_paired_comparison(self) -> None:
        topk = np.asarray([[3, 1, 2], [4, 5, 6], [9, 8, 7]], dtype=np.int64)
        targets = np.asarray([1, 6, 0], dtype=np.int64)
        ranks = target_ranks_from_topk(topk, targets)
        np.testing.assert_array_equal(ranks, [2, 3, -1])
        metrics = rank_metrics(ranks)
        self.assertAlmostEqual(metrics["hit_at_3"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr_at_10"], (1 / 2 + 1 / 3) / 3)
        comparison = paired_rank_comparison(
            np.asarray([3, -1, 1], dtype=np.int16),
            ranks,
            top_k=3,
        )
        self.assertEqual(comparison, {"wins": 2, "ties": 0, "losses": 1})


if __name__ == "__main__":
    unittest.main()
