from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qg_prqk.sid.visualization import (  # noqa: E402
    code_usage_metrics,
    sample_exact_prefix_pairs,
    transition_matrix,
)


class SIDVisualizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sid = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=np.int32,
        )

    def test_code_usage_includes_empty_slots(self) -> None:
        metrics = code_usage_metrics(np.asarray([0, 0, 1, 2]), codebook_size=4)

        self.assertEqual(metrics["active_codes"], 3)
        self.assertAlmostEqual(metrics["active_ratio"], 0.75)
        self.assertAlmostEqual(metrics["effective_codes"], 8 / 3)
        self.assertAlmostEqual(metrics["gini"], 0.375)
        self.assertEqual(metrics["count_min"], 0)
        self.assertEqual(metrics["count_max"], 2)

    def test_transition_matrix_conserves_rows(self) -> None:
        matrix = transition_matrix(self.sid, 0, 1, codebook_size=4)

        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(int(matrix.sum()), len(self.sid))
        self.assertEqual(int(matrix[0, 0]), 3)
        self.assertEqual(int(matrix[1, 1]), 1)

    def _assert_exact_prefix(self, prefix_length: int) -> None:
        first = sample_exact_prefix_pairs(
            self.sid, prefix_length, pair_count=40, seed=42, codebook_size=4
        )
        second = sample_exact_prefix_pairs(
            self.sid, prefix_length, pair_count=40, seed=42, codebook_size=4
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first[:, 0] != first[:, 1]))
        left = self.sid[first[:, 0]]
        right = self.sid[first[:, 1]]
        shared = np.sum(np.cumprod(left == right, axis=1), axis=1)
        self.assertTrue(np.all(shared == prefix_length))

    def test_exact_prefix_zero_pair_sampling(self) -> None:
        self._assert_exact_prefix(0)

    def test_exact_prefix_one_pair_sampling(self) -> None:
        self._assert_exact_prefix(1)

    def test_exact_prefix_two_pair_sampling(self) -> None:
        self._assert_exact_prefix(2)

    def test_exact_prefix_three_pair_sampling(self) -> None:
        self._assert_exact_prefix(3)


if __name__ == "__main__":
    unittest.main()
