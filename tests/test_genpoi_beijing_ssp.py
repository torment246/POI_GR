"""Synthetic tests for Beijing-adapted GenPOI SSP."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.genpoi_beijing_ssp import (  # noqa: E402
    BeijingSspError,
    calibrate_safe_prefix_thresholds,
    evaluate_safe_prefixes,
    ordinal_targets,
    select_safe_prefix_depth,
)


class GenPoiBeijingSspTest(unittest.TestCase):
    def test_ordinal_targets_are_nested(self) -> None:
        self.assertEqual(ordinal_targets(2), (0, 0, 0, 0))
        self.assertEqual(ordinal_targets(4), (1, 1, 0, 0))
        self.assertEqual(ordinal_targets(6), (1, 1, 1, 1))

    def test_selection_requires_all_shallower_decisions(self) -> None:
        thresholds = {3: 0.5, 4: 0.6, 5: 0.7, 6: 0.8}
        self.assertEqual(
            select_safe_prefix_depth((0.9, 0.8, 0.6, 0.95), thresholds),
            4,
        )
        self.assertEqual(
            select_safe_prefix_depth((0.4, 0.9, 0.9, 0.9), thresholds),
            0,
        )

    def test_calibration_respects_conservative_false_positive_budget(self) -> None:
        labels = np.asarray([2, 3, 3, 4, 4, 5, 5, 6], dtype=np.int64)
        probabilities = np.asarray(
            [
                [0.4, 0.2, 0.1, 0.0],
                [0.9, 0.4, 0.1, 0.0],
                [0.8, 0.3, 0.2, 0.0],
                [0.9, 0.8, 0.3, 0.1],
                [0.9, 0.7, 0.2, 0.1],
                [0.9, 0.9, 0.8, 0.2],
                [0.9, 0.9, 0.7, 0.3],
                [0.9, 0.9, 0.9, 0.9],
            ],
            dtype=np.float64,
        )
        thresholds = calibrate_safe_prefix_thresholds(
            probabilities,
            labels,
            max_unsafe_rate=0.0,
        )
        selected = [
            select_safe_prefix_depth(row, thresholds) for row in probabilities
        ]
        metrics = evaluate_safe_prefixes(labels.tolist(), selected)
        self.assertEqual(metrics.unsafe_pruning_rate, 0.0)
        self.assertGreater(metrics.useful_prefix_rate, 0.0)

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(BeijingSspError):
            select_safe_prefix_depth(
                (0.9, 1.1, 0.8, 0.7),
                {3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5},
            )


if __name__ == "__main__":
    unittest.main()
