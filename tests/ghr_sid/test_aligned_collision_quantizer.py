"""Synthetic tests for the globally aligned TIGER collision suffix."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.aligned_collision_quantizer import (  # noqa: E402
    assign_unique_pairs,
    compose_collision_features,
    compose_five_layer_identifiers,
    multiscale_geo_features,
)


class AlignedCollisionQuantizerTest(unittest.TestCase):
    def test_multiscale_geo_is_deterministic_and_normalized(self) -> None:
        coordinates = np.asarray(
            [[116.3, 39.9], [116.31, 39.91]], dtype=np.float64
        )
        first, origin = multiscale_geo_features(
            coordinates, [1.0, 4.0], origin_lng_lat=[116.3, 39.9]
        )
        second, second_origin = multiscale_geo_features(
            coordinates, [1.0, 4.0], origin_lng_lat=origin
        )
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
        self.assertEqual(origin, second_origin)
        self.assertEqual(first.shape, (2, 8))

    def test_features_center_semantics_within_each_bucket(self) -> None:
        latent = np.asarray(
            [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
            dtype=np.float32,
        )
        geo = np.ones((4, 2), dtype=np.float32) / np.sqrt(2.0)
        features = compose_collision_features(
            latent, geo, np.asarray([10, 10, 20, 20]), geo_weight=0.5
        )
        np.testing.assert_allclose(features[0, :2], [-1.0, 0.0])
        np.testing.assert_allclose(features[1, :2], [1.0, 0.0])
        np.testing.assert_allclose(features[:, 2:], geo * 0.5)

    def test_unique_assignment_resolves_nearest_pair_collision(self) -> None:
        features = np.asarray([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)
        buckets = np.asarray([0, 0, 1, 1], dtype=np.int64)
        prototypes = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
        candidates = np.asarray(
            [[0, 1], [0, 1], [2, 3], [2, 3]], dtype=np.int32
        )
        distances = np.square(features - prototypes[candidates, 0])
        pairs, metrics = assign_unique_pairs(
            features, buckets, candidates, distances, prototypes
        )
        self.assertEqual(len(np.unique(pairs[:2])), 2)
        self.assertEqual(len(np.unique(pairs[2:])), 2)
        self.assertEqual(metrics["constrained_bucket_count"], 2)
        self.assertEqual(metrics["forced_reassignment_row_count"], 2)

    def test_five_layer_ids_are_unique_inside_base_bucket(self) -> None:
        base = np.asarray([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=np.int32)
        identifiers, suffix = compose_five_layer_identifiers(
            base,
            np.asarray([0, 1]),
            np.asarray([0, 33]),
            32,
        )
        self.assertEqual(suffix.tolist(), [[0, 0], [1, 1], [0, 0]])
        self.assertEqual(len(np.unique(identifiers, axis=0)), 3)


if __name__ == "__main__":
    unittest.main()
