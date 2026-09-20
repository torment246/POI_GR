from __future__ import annotations

import unittest

import numpy as np

from qg_prqk.sid.evaluation import (
    _candidate_index,
    _candidate_mask,
    _local_gid_metrics,
    _path_metrics,
    _s3_parent_keys,
    partition_metrics,
)


class SidEvaluationTest(unittest.TestCase):
    def test_gid_and_s1_s2_parent_keys_are_distinct_contracts(self) -> None:
        gid = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.uint8)
        s1 = np.array([4, 4])
        s2 = np.array([7, 7])
        gid_keys = _s3_parent_keys(gid, s1, s2)
        semantic_keys = s1.astype(np.int64) * 512 + s2
        self.assertNotEqual(int(gid_keys[0]), int(gid_keys[1]))
        self.assertEqual(int(semantic_keys[0]), int(semantic_keys[1]))

    def test_partition_metrics_perfect_and_mixed(self) -> None:
        perfect = partition_metrics(np.array([0, 0, 1, 1]), np.array([3, 3, 4, 4]))
        self.assertAlmostEqual(perfect["purity"], 1.0)
        self.assertAlmostEqual(perfect["nmi_geometric"], 1.0)
        mixed = partition_metrics(np.array([0, 0, 1, 1]), np.array([3, 4, 3, 4]))
        self.assertAlmostEqual(mixed["purity"], 0.5)
        self.assertAlmostEqual(mixed["nmi_geometric"], 0.0)

    def test_candidate_index_marks_missing_parent(self) -> None:
        index = _candidate_index(
            np.array([4, 4, 4, 7]), np.array([1, 1, 3, 2])
        )
        mask, valid = _candidate_mask(index, np.array([4, 7, 8]))
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [1, 3])
        self.assertEqual(np.flatnonzero(mask[1]).tolist(), [2])
        self.assertFalse(mask[2].any())

    def test_path_and_same_gid_pair_metrics(self) -> None:
        sid = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 4], [8, 9, 10]])
        path = _path_metrics(sid)
        self.assertEqual(path["distinct"], 3)
        self.assertEqual(path["collision_excess"], 1)
        gid = np.array([[1] * 6, [1] * 6, [1] * 6, [2] * 6], dtype=np.uint8)
        local = _local_gid_metrics(gid, sid)
        self.assertEqual(local["same_gid6_unordered_pairs"], 3)
        self.assertEqual(local["same_gid6_same_sid_pairs"], 1)
        self.assertAlmostEqual(local["local_entity_separation_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
