"""Tests for E4 category-common Query residual helpers."""

from __future__ import annotations

import unittest

import numpy as np

from poi_gr.embedding.query_category_residual import residual_filename


class QueryCategoryResidualTest(unittest.TestCase):
    def test_residual_filename_is_stable(self) -> None:
        self.assertEqual(
            residual_filename(0.25),
            "e4_category_residual_beta_0p25.npy",
        )

    def test_category_raw_mean_residual_removes_shared_direction(self) -> None:
        query = np.asarray(
            [[1.0, 0.0], [2**-0.5, 2**-0.5]], dtype=np.float32
        )
        mean = query.mean(axis=0)
        residual = query - 0.5 * mean
        residual /= np.linalg.norm(residual, axis=1, keepdims=True)
        original_shared_projection = query @ mean
        residual_shared_projection = residual @ mean
        self.assertTrue(
            np.all(residual_shared_projection < original_shared_projection)
        )


if __name__ == "__main__":
    unittest.main()
