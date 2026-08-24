"""Tests for query-free geo-first relational collision compilation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.structure import (  # noqa: E402
    STAGE_COARSE_GEO,
    STAGE_FINE_GEO,
    STAGE_RELATION,
    STAGE_UNRESOLVED,
    compile_geo_relation_structure,
)
from poi_gr.methods.qgr_sid.relations import RELATION_TYPES  # noqa: E402


def _relations(rows: int) -> np.ndarray:
    return np.full((rows, len(RELATION_TYPES)), -1, dtype=np.int16)


class StructureCompilerTest(unittest.TestCase):
    def test_stage_order_is_geo_then_relation_then_fine_geo(self) -> None:
        keys = np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        gids = np.asarray(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 2, 1, 1],
                [2, 2, 2, 2, 2, 2, 1, 1],
                [2, 2, 2, 2, 2, 2, 1, 1],
                [3, 3, 3, 3, 3, 3, 1, 1],
                [3, 3, 3, 3, 3, 3, 1, 2],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
            dtype=np.uint8,
        )
        relations = _relations(len(keys))
        building = RELATION_TYPES.index("R_BUILDING")
        relations[2, building] = 1
        relations[3, building] = 2
        compilation = compile_geo_relation_structure(
            base_sid_keys=keys,
            gid_codes=gids,
            strict_relations=relations,
            coarse_precision=6,
            fine_precision=8,
            max_relation_pairs=3,
            full_poi_count=10,
        )
        self.assertEqual(
            compilation.resolution_stage.tolist(),
            [
                STAGE_COARSE_GEO,
                STAGE_COARSE_GEO,
                STAGE_RELATION,
                STAGE_RELATION,
                STAGE_FINE_GEO,
                STAGE_FINE_GEO,
                STAGE_UNRESOLVED,
                STAGE_UNRESOLVED,
            ],
        )
        self.assertTrue(np.all(compilation.relation_path_types[:2] == -1))
        self.assertTrue(np.all(compilation.fine_geo_lengths[:4] == 0))
        self.assertEqual(
            compilation.metrics["resolution"]["unresolved_group_count"], 1
        )

    def test_unique_missing_relation_branch_can_close_early(self) -> None:
        keys = np.asarray([7, 7, 7], dtype=np.int64)
        gids = np.asarray([[1] * 8, [1] * 8, [1] * 8], dtype=np.uint8)
        relations = _relations(3)
        phase = RELATION_TYPES.index("R_PHASE")
        relations[0, phase] = 1
        relations[1, phase] = 2
        compilation = compile_geo_relation_structure(
            base_sid_keys=keys,
            gid_codes=gids,
            strict_relations=relations,
            coarse_precision=6,
            fine_precision=8,
        )
        self.assertEqual(
            compilation.resolution_stage.tolist(),
            [STAGE_RELATION, STAGE_RELATION, STAGE_RELATION],
        )
        self.assertEqual(compilation.relation_path_types[2].tolist(), [-1, -1, -1])

    def test_geo_paths_are_minimal_after_bucket_common_prefix(self) -> None:
        keys = np.asarray([9, 9, 9], dtype=np.int64)
        gids = np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 9, 6, 7, 8],
                [1, 2, 3, 4, 9, 10, 7, 8],
            ],
            dtype=np.uint8,
        )
        compilation = compile_geo_relation_structure(
            base_sid_keys=keys,
            gid_codes=gids,
            strict_relations=_relations(3),
            coarse_precision=6,
            fine_precision=8,
        )
        self.assertEqual(compilation.root_prefix_lengths.tolist(), [4, 4, 4])
        self.assertEqual(compilation.coarse_geo_lengths.tolist(), [1, 2, 2])
        self.assertEqual(
            compilation.resolution_stage.tolist(),
            [STAGE_COARSE_GEO, STAGE_COARSE_GEO, STAGE_COARSE_GEO],
        )

    def test_rejects_invalid_precision(self) -> None:
        with self.assertRaises(ValueError):
            compile_geo_relation_structure(
                base_sid_keys=np.asarray([1, 1]),
                gid_codes=np.zeros((2, 8), dtype=np.uint8),
                strict_relations=_relations(2),
                coarse_precision=8,
                fine_precision=8,
            )


if __name__ == "__main__":
    unittest.main()
