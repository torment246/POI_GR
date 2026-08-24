"""Tests for the bounded QGR-SID relation-path compiler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.compiler import (  # noqa: E402
    branch_entropy_bits,
    compile_relation_bucket,
)


class RelationCompilerTest(unittest.TestCase):
    def test_branch_entropy_uses_request_weights(self) -> None:
        entropy = branch_entropy_bits(
            np.asarray([1, 1, 2]), np.asarray([1.0, 1.0, 2.0])
        )
        self.assertAlmostEqual(entropy, 1.0)

    def test_static_compiler_builds_variable_paths_and_marks_unique_leaves(self) -> None:
        relations = np.asarray(
            [
                [1, 10, -1],
                [1, 11, -1],
                [2, -1, 20],
                [2, -1, 21],
            ],
            dtype=np.int16,
        )
        orders = np.asarray([10, 8, 6, 4], dtype=np.int32)
        matches = np.zeros_like(relations, dtype=np.int32)
        result = compile_relation_bucket(
            relations,
            orders,
            matches,
            [0.5, 0.5, 0.5],
            query_guided=False,
            max_pairs=3,
        )
        self.assertTrue(result.relation_only_resolved.all())
        lengths = (result.path_types >= 0).sum(axis=1)
        self.assertTrue(np.all(lengths <= 2))
        self.assertGreaterEqual(result.decision_count, 2)

    def test_query_guidance_can_change_root_relation(self) -> None:
        relations = np.asarray(
            [
                [1, 10],
                [1, 11],
                [2, 10],
                [2, 11],
            ],
            dtype=np.int16,
        )
        orders = np.asarray([100, 100, 100, 100], dtype=np.int32)
        matches = np.asarray(
            [
                [0, 90],
                [0, 90],
                [0, 90],
                [0, 90],
            ],
            dtype=np.int32,
        )
        static = compile_relation_bucket(
            relations,
            orders,
            matches,
            [0.0, 0.9],
            query_guided=False,
        )
        guided = compile_relation_bucket(
            relations,
            orders,
            matches,
            [0.0, 0.9],
            query_guided=True,
        )
        self.assertTrue(np.all(static.path_types[:, 0] == 0))
        self.assertTrue(np.all(guided.path_types[:, 0] == 1))

    def test_relation_only_resolution_rejects_duplicate_paths(self) -> None:
        relations = np.asarray([[1], [1], [2]], dtype=np.int16)
        orders = np.ones(3, dtype=np.int32)
        result = compile_relation_bucket(
            relations,
            orders,
            np.zeros_like(relations),
            [0.5],
            query_guided=False,
        )
        self.assertEqual(result.relation_only_resolved.tolist(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
