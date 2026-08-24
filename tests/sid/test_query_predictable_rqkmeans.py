"""Contracts for Query-Predictable dual-view residual K-Means."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sid.query_predictable_rqkmeans import (  # noqa: E402
    assign_dual_view_numpy,
    build_category_residual_query_rows,
    recover_query_view_from_fused,
    route_covered_topk_numpy,
    shrink_query_centers_numpy,
    subtract_assigned_centers,
    update_dual_view_centers,
)


class QueryPredictableRQKMeansTest(unittest.TestCase):
    def test_recovers_query_view_from_normalized_fusion(self) -> None:
        content = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        alpha = 0.3
        fused = (1.0 - alpha) * content + alpha * query
        fused /= np.linalg.norm(fused, axis=1, keepdims=True)
        recovered = recover_query_view_from_fused(
            content, fused, alpha=alpha
        )
        np.testing.assert_allclose(recovered, query, atol=1e-6)

    def test_category_residual_query_rows_are_normalized(self) -> None:
        rows = np.asarray([[1.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        categories = np.asarray([0, 1], dtype=np.int32)
        means = np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
        residual = build_category_residual_query_rows(
            rows, categories, means, beta=0.5
        )
        np.testing.assert_allclose(np.linalg.norm(residual, axis=1), 1.0)

    def test_uncovered_rows_exactly_degenerate_to_content_assignment(self) -> None:
        content = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        centers = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        # If zeros were treated as an observed Query view, the second row would
        # incorrectly move to token 0 under this large Query weight.
        query = np.zeros_like(content)
        query_centers = np.asarray([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
        covered = np.asarray([False, False])
        assignment = assign_dual_view_numpy(
            content,
            centers,
            query=query,
            query_centers=query_centers,
            query_covered=covered,
            query_weight=100.0,
        )
        np.testing.assert_array_equal(assignment.labels, [0, 1])
        np.testing.assert_array_equal(assignment.query_squared_distance, [0.0, 0.0])

    def test_query_signal_can_refine_a_covered_assignment(self) -> None:
        content = np.asarray([[0.2, 0.0]], dtype=np.float32)
        content_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        query_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        covered = np.asarray([True])
        content_only = assign_dual_view_numpy(content, content_centers)
        dual_view = assign_dual_view_numpy(
            content,
            content_centers,
            query=query,
            query_centers=query_centers,
            query_covered=covered,
            query_weight=1.0,
        )
        np.testing.assert_array_equal(content_only.labels, [0])
        np.testing.assert_array_equal(dual_view.labels, [1])

    def test_query_centers_only_use_covered_rows(self) -> None:
        content = np.asarray(
            [[0.0, 0.0], [0.2, 0.0], [10.0, 0.0]], dtype=np.float32
        )
        query = np.asarray(
            [[2.0, 0.0], [1000.0, 0.0], [8.0, 0.0]], dtype=np.float32
        )
        covered = np.asarray([True, False, True])
        labels = np.asarray([0, 0, 1], dtype=np.int32)
        old = np.zeros((2, 2), dtype=np.float32)
        content_centers, query_centers, content_counts, query_counts = (
            update_dual_view_centers(
                content, query, covered, labels, old, old
            )
        )
        np.testing.assert_allclose(content_centers[:, 0], [0.1, 10.0])
        np.testing.assert_allclose(query_centers[:, 0], [2.0, 8.0])
        np.testing.assert_array_equal(content_counts, [2, 1])
        np.testing.assert_array_equal(query_counts, [1, 1])

    def test_two_views_share_codes_but_update_separate_residuals(self) -> None:
        labels = np.asarray([0, 1], dtype=np.int32)
        content = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        query = np.asarray([[5.0, 0.0], [0.0, 7.0]], dtype=np.float32)
        content_centers = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query_centers = np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        next_content = subtract_assigned_centers(content, content_centers, labels)
        next_query = subtract_assigned_centers(query, query_centers, labels)
        np.testing.assert_array_equal(next_content, [[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(next_query, [[3.0, 0.0], [0.0, 3.0]])

    def test_query_center_shrinkage_uses_support_reliability(self) -> None:
        query = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 3.0]], dtype=np.float32
        )
        labels = np.asarray([0, 0, 1], dtype=np.int32)
        covered = np.asarray([True, True, True])
        centers, counts, reliability = shrink_query_centers_numpy(
            query,
            labels,
            covered,
            3,
            support_tau=2.0,
        )
        np.testing.assert_array_equal(counts, [2, 1, 0])
        np.testing.assert_allclose(reliability, [0.5, 1.0 / 3.0, 0.0])
        global_mean = np.asarray([2.0 / 3.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(centers[2], global_mean)
        self.assertLess(np.linalg.norm(centers[1] - global_mean), 2.0)

    def test_content_preserving_route_changes_only_covered_rows(self) -> None:
        content = np.asarray(
            [[0.1, 0.0], [0.1, 0.0], [9.9, 0.0]], dtype=np.float32
        )
        content_centers = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=np.float32
        )
        query = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32
        )
        query_centers = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32
        )
        covered = np.asarray([True, False, False])
        reference = np.asarray([0, 0, 2], dtype=np.int32)
        result = route_covered_topk_numpy(
            content,
            content_centers,
            query,
            query_centers,
            covered,
            reference,
            query_weight=2.0,
            content_top_t=2,
        )
        np.testing.assert_array_equal(result.labels, [1, 0, 2])
        np.testing.assert_array_equal(result.labels[~covered], reference[~covered])

    def test_content_preserving_route_always_keeps_reference_candidate(self) -> None:
        content = np.asarray([[0.0, 0.0]], dtype=np.float32)
        content_centers = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]], dtype=np.float32
        )
        query = np.asarray([[0.0, 0.0]], dtype=np.float32)
        query_centers = np.zeros_like(content_centers)
        result = route_covered_topk_numpy(
            content,
            content_centers,
            query,
            query_centers,
            np.asarray([True]),
            np.asarray([2], dtype=np.int32),
            query_weight=0.0,
            content_top_t=1,
        )
        np.testing.assert_array_equal(result.labels, [0])


if __name__ == "__main__":
    unittest.main()
