"""Synthetic tests for fixed-parent S3 evidence and deterministic case selection."""

import unittest

import numpy as np

from qg_prqk.sid.local_visualization import choose_parents, local_coordinates, parent_collisions, same_code_pairs


class LocalVisualizationTest(unittest.TestCase):
    def test_cross_parent_shared_code_is_not_a_collision(self):
        pairs, pois = parent_collisions(np.array([0, 0, 0, 1]), np.array([7, 7, 8, 7]))
        np.testing.assert_array_equal(pairs, [1, 0])
        np.testing.assert_array_equal(pois, [2, 0])

    def test_relabeling_without_partition_change_preserves_counts(self):
        groups = np.zeros(4, dtype=int)
        first = parent_collisions(groups, np.array([1, 1, 2, 2]))
        second = parent_collisions(groups, np.array([8, 8, 9, 9]))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_selection_is_deterministic_and_has_no_final_label_input(self):
        groups = np.repeat(np.arange(4), 4)
        initial = np.tile([1, 1, 2, 3], 4)
        a, audit = choose_parents(groups, initial, np.arange(4))
        b, _ = choose_parents(groups, initial, np.arange(4))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(audit['selection_uses_final_s3'])

    def test_selection_rejects_empty_eligibility(self):
        with self.assertRaises(ValueError):
            choose_parents(np.array([0, 0]), np.array([1, 1]), np.array([0]))

    def test_geo_keeps_coincident_points_and_meter_distance(self):
        xy = local_coordinates(np.array([116., 116., 116.]), np.array([40., 40., 40.001]))
        np.testing.assert_array_equal(xy[0], xy[1])
        self.assertAlmostEqual(float(np.linalg.norm(xy[2] - xy[0])), 111.19508, places=3)

    def test_pair_list_excludes_self_and_reverse(self):
        np.testing.assert_array_equal(same_code_pairs(np.array([4, 4, 4])), [[0, 1], [0, 2], [1, 2]])

    def test_illegal_code_rejected(self):
        with self.assertRaises(ValueError):
            parent_collisions(np.array([0]), np.array([512]))


if __name__ == '__main__':
    unittest.main()
