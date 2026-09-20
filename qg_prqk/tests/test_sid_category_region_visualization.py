"""Synthetic tests for matched-category sampling and exact SID prefix counts."""

from __future__ import annotations

import unittest

import numpy as np

from qg_prqk.sid.category_region_visualization import (
    balanced_category_sample,
    catalog_prefix_metrics,
    label_distribution,
    prefix_examples,
    sid_codeword_features,
)


class CategoryRegionVisualizationTests(unittest.TestCase):
    def test_balanced_sample_is_fixed_and_without_replacement(self):
        labels = np.repeat(["住宅", "美食", "购物"], 20)
        rows, anchors = balanced_category_sample(
            labels, ["住宅", "美食", "购物"], 5, 42
        )
        repeated, repeated_anchors = balanced_category_sample(
            labels, ["住宅", "美食", "购物"], 5, 42
        )
        np.testing.assert_array_equal(rows, repeated)
        np.testing.assert_array_equal(anchors, repeated_anchors)
        self.assertEqual(len(set(rows)), 15)
        np.testing.assert_array_equal(
            labels[rows], np.repeat(["住宅", "美食", "购物"], 5)
        )
        np.testing.assert_array_equal(anchors, rows[[0, 5, 10]])

    def test_sample_contract_errors(self):
        for categories, count in [(["x"], 3), (["x", "x"], 1), (["x"], 0)]:
            with self.assertRaises(ValueError):
                balanced_category_sample(["x", "x"], categories, count, 42)

    def test_codeword_features_are_equal_weight_not_integer_distances(self):
        sid = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
        book = np.array([[3, 0], [0, 7]], dtype=np.float32)
        features = sid_codeword_features(sid, [book, book * 100, book * 0.01])
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(features.reshape(2, 3, 2), axis=2), 1 / np.sqrt(3), atol=1e-6
        )
        np.testing.assert_allclose(
            features[0], [1 / np.sqrt(3), 0, 0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0]
        )
        # A permutation of code ids and corresponding rows changes no geometry.
        permuted = sid_codeword_features(
            1 - sid, [book[::-1], (book * 100)[::-1], (book * 0.01)[::-1]]
        )
        np.testing.assert_array_equal(features, permuted)

    def test_invalid_codewords_fail(self):
        for sid, book in [
            (np.array([[0, 0, 2]]), np.eye(2)),
            (np.array([[0, 0, 0]]), np.zeros((2, 2))),
            (np.array([[0, 0, 0]]), np.full((2, 2), np.nan)),
            (np.array([[-1, 0, 0]]), np.eye(2)),
        ]:
            with self.assertRaises(ValueError):
                sid_codeword_features(sid, [book] * 3)

    def test_distribution_preserves_counts_and_deterministic_ties(self):
        result = label_distribution(["b", "a", "b", "a", "c", "d"], 2)
        self.assertEqual([entry["label"] for entry in result["top"]], ["a", "b"])
        self.assertEqual(result["other_count"], 2)
        self.assertEqual(sum(result["counts"].values()), result["total"])
        self.assertAlmostEqual(
            sum(entry["share"] for entry in result["top"]) + result["other_share"], 1
        )

    def test_prefixes_use_full_buckets_not_only_anchor_category(self):
        sid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        coarse = np.array(["a", "b", "c", "a"])
        fine = np.array(["a:x", "b:y", "c:z", "a:x"])
        region = np.array(["wx4g0", "wx4g0", "wx4g1", "wx4g1"])
        entries = prefix_examples(sid, np.array([0]), coarse, fine, region, 3)
        self.assertEqual([entry["poi_count"] for entry in entries], [3, 2, 1])
        self.assertEqual(
            entries[0]["coarse_category"]["counts"], {"a": 1, "b": 1, "c": 1}
        )
        self.assertEqual(entries[1]["fine_category"]["counts"], {"a:x": 1, "b:y": 1})
        for entry in entries:
            for key in ("coarse_category", "fine_category", "region_geohash5"):
                self.assertEqual(entry[key]["total"], entry["poi_count"])

    def test_catalog_purity_is_population_weighted_and_excludes_singletons(self):
        sid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        labels = np.array(["a", "a", "b", "b"])
        metrics = catalog_prefix_metrics(sid, labels, labels, labels, codebook_size=2)
        self.assertEqual(metrics[0]["unique_prefixes"], 2)
        self.assertEqual(metrics[0]["coarse_category"]["poi_weighted_top1_share"], 0.75)
        self.assertAlmostEqual(
            metrics[0]["coarse_category"]["non_singleton_poi_weighted_top1_share"],
            2 / 3,
        )
        self.assertEqual(metrics[2]["singleton_poi_share"], 1)
        self.assertIsNone(
            metrics[2]["coarse_category"]["non_singleton_poi_weighted_top1_share"]
        )

    def test_a4_shared_first_two_levels_give_identical_prefix_statistics(self):
        first = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        second = first.copy()
        second[1, 2] = 1
        labels = np.array(["a", "b", "a"])
        one = prefix_examples(first, np.array([0]), labels, labels, labels, 3)
        two = prefix_examples(second, np.array([0]), labels, labels, labels, 3)
        self.assertEqual(one[:2], two[:2])
        self.assertNotEqual(one[2], two[2])


if __name__ == "__main__":
    unittest.main()
