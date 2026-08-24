from __future__ import annotations

import unittest

import numpy as np

from poi_gr.methods.ghr_sid.residual_vector_audit import (
    build_partition_cascade,
    minimum_gid_labels_for_duplicate_vectors,
)


class ResidualVectorAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.group_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        self.embedding_keys = [b"a", b"a", b"b", b"c", b"c"]
        self.gid_keys = {
            length: [
                b"x1" if row == 0 else b"x2" if row == 1 else b"x3"
                for row in range(5)
            ]
            for length in range(7, 13)
        }
        for length in range(7, 13):
            self.gid_keys[length][3] = b"z"
            self.gid_keys[length][4] = b"z"
        self.coordinate_keys = [(1, 1), (2, 2), (3, 3), (4, 4), (4, 4)]

    def test_vector_then_gid_keeps_same_coordinate_duplicates(self) -> None:
        cascades = build_partition_cascade(
            group_ids=self.group_ids,
            embedding_keys=self.embedding_keys,
            gid_keys=self.gid_keys,
            coordinate_keys=self.coordinate_keys,
        )
        self.assertEqual(
            cascades["exact_embedding_then_geography"]["exact_embedding"][
                "collision_excess"
            ],
            2,
        )
        self.assertEqual(
            cascades["exact_embedding_then_geography"][
                "exact_embedding_gid7"
            ]["collision_excess"],
            1,
        )
        self.assertEqual(
            cascades["exact_embedding_then_geography"][
                "exact_embedding_coordinate"
            ]["collision_excess"],
            1,
        )

    def test_minimum_gid_labels_are_per_poi(self) -> None:
        labels, distribution = minimum_gid_labels_for_duplicate_vectors(
            group_ids=self.group_ids,
            embedding_keys=self.embedding_keys,
            gid_keys=self.gid_keys,
            coordinate_keys=self.coordinate_keys,
        )
        self.assertEqual(labels[:3], ["gid7", "gid7", "not_duplicate_vector"])
        self.assertEqual(labels[3:], ["same_coordinate_unresolved"] * 2)
        self.assertEqual(
            distribution,
            {
                "gid7": 2,
                "not_duplicate_vector": 1,
                "same_coordinate_unresolved": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
