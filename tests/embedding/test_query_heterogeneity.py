"""Tests for E3 Train-only heterogeneity features."""

from __future__ import annotations

import unittest

import numpy as np

from poi_gr.embedding.query_heterogeneity import (
    QueryHeterogeneityError,
    compute_effective_query_count,
    content_query_cosine,
)


class QueryHeterogeneityTest(unittest.TestCase):
    def test_effective_query_count_respects_weight_concentration(self) -> None:
        effective = compute_effective_query_count(
            np.asarray([2.0, 4.0], dtype=np.float64),
            np.asarray([2.0, 10.0], dtype=np.float64),
            np.asarray([2, 3], dtype=np.int32),
        )
        np.testing.assert_allclose(effective, [2.0, 1.6], atol=1e-6)

    def test_effective_query_count_rejects_impossible_bounds(self) -> None:
        with self.assertRaises(QueryHeterogeneityError):
            compute_effective_query_count(
                np.asarray([3.0]),
                np.asarray([1.0]),
                np.asarray([2]),
            )

    def test_content_query_cosine(self) -> None:
        cosine = content_query_cosine(
            np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float16),
            np.asarray([[0.0, 1.0], [2.0, 2.0]], dtype=np.float16),
        )
        np.testing.assert_allclose(cosine, [0.0, 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
