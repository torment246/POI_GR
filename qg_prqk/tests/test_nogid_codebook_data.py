from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qg_prqk.sid.nogid_data import HARD_EDGE_SCHEMA, build_nogid_hard_entity_graph  # noqa: E402
from qg_prqk.sid.nogid_geo import s1_s2_parent_geo_features, s1_s2_parent_keys  # noqa: E402


class NoGIDCodebookDataTest(unittest.TestCase):
    def test_parent_key_uses_only_s1_s2(self) -> None:
        keys = s1_s2_parent_keys(
            np.array([0, 1, 1, 511]),
            np.array([2, 3, 3, 511]),
        )
        np.testing.assert_array_equal(keys, np.array([2, 515, 515, 262143]))

    def test_continuous_geo_is_parent_relative_without_gid(self) -> None:
        keys = s1_s2_parent_keys(np.array([1, 1, 2]), np.array([4, 4, 5]))
        features, stats, singleton = s1_s2_parent_geo_features(
            np.array([116.30, 116.31, 116.40]),
            np.array([39.90, 39.91, 39.95]),
            keys,
        )
        self.assertEqual(features.shape, (3, 5))
        self.assertFalse(stats["gid_or_geohash_used"])
        np.testing.assert_array_equal(singleton, np.array([False, False, True]))
        np.testing.assert_array_equal(features[2], np.zeros(5, dtype=np.float32))
        nonzero = np.linalg.norm(features[:2], axis=1)
        np.testing.assert_allclose(nonzero, np.ones(2), atol=2.0e-5)

    def test_hard_graph_is_scoped_by_s1_s2_and_has_no_gid_column(self) -> None:
        metadata = pa.table(
            {
                "displayname": ["测试门店", "测试门店", "其他"],
                "alias": ["", "", ""],
                "address": ["同一地址", "同一地址", "远处"],
            }
        )
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        graph, table = build_nogid_hard_entity_graph(
            np.array([10, 11, 12]),
            embeddings,
            metadata,
            np.array([[1, 2], [1, 2], [1, 3]], dtype=np.int32),
            np.array([7, 7, 7], dtype=np.int32),
            frozenset(),
            weights={"bge": 0.35, "name_alias": 0.30, "category": 0.15, "address": 0.10},
            threshold=0.50,
            max_neighbors=20,
        )
        self.assertTrue(table.schema.equals(HARD_EDGE_SCHEMA))
        self.assertNotIn("gid", " ".join(table.column_names).lower())
        self.assertEqual(len(table), 2)
        np.testing.assert_array_equal(graph.poi_rows, np.array([0, 1]))
        np.testing.assert_array_equal(graph.neighbor_rows, np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
