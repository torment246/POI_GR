from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

from qg_prqk.sid.local_data import (
    POI_METADATA_SCHEMA,
    P7_POI_METADATA_SCHEMA,
    build_hard_entity_graph,
)


class LocalCodebookDataTest(unittest.TestCase):
    def test_p7_metadata_schema_keeps_derived_fields_non_nullable(self) -> None:
        for name in (
            "gid6",
            "parent_group",
            "is_singleton_parent",
            "fine_category_index",
        ):
            self.assertFalse(P7_POI_METADATA_SCHEMA.field(name).nullable)

    def test_hard_graph_obeys_parent_category_threshold_and_fn_mask(self) -> None:
        metadata = pa.Table.from_pylist(
            [
                {
                    "poi_row_index": index,
                    "poi_id": str(index),
                    "displayname": "北京站东广场",
                    "alias": "北京站",
                    "category": "交通设施:火车站",
                    "category_code": "150100" if index < 3 else "150200",
                    "address": "北京市东城区",
                    "lat": 39.9,
                    "lng": 116.4,
                }
                for index in range(4)
            ],
            schema=POI_METADATA_SCHEMA,
        )
        embeddings = np.array(
            [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [1.0, 0.0]], dtype=np.float16
        )
        graph, table = build_hard_entity_graph(
            np.arange(4, dtype=np.int64),
            embeddings,
            metadata,
            np.array([7, 7, 8, 7]),
            np.array([2, 2, 2, 3], dtype=np.int32),
            frozenset({(0, 1)}),
            weights={"bge": 0.35, "name_alias": 0.30, "category": 0.15, "address": 0.10, "geo": 0.10},
            threshold=0.60,
            max_neighbors=20,
        )
        self.assertEqual(len(table), 0)
        self.assertEqual(len(graph.weights), 0)

        graph, table = build_hard_entity_graph(
            np.arange(4, dtype=np.int64),
            embeddings,
            metadata,
            np.array([7, 7, 7, 7]),
            np.array([2, 2, 2, 3], dtype=np.int32),
            frozenset({(0, 1)}),
            weights={"bge": 0.35, "name_alias": 0.30, "category": 0.15, "address": 0.10, "geo": 0.10},
            threshold=0.60,
            max_neighbors=20,
        )
        pairs = set(zip(table["poi_row_index"].to_pylist(), table["neighbor_poi_row_index"].to_pylist()))
        self.assertEqual(pairs, {(0, 2), (2, 0), (1, 2), (2, 1)})
        self.assertNotIn((0, 1), pairs)
        self.assertNotIn((1, 0), pairs)
        self.assertTrue(np.all(graph.weights >= 0.60))


if __name__ == "__main__":
    unittest.main()
