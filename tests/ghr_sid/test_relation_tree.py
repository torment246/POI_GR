"""Tests for the shared strong-relation tree compiler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.fine_relations import RawFineRelations  # noqa: E402
from poi_gr.methods.ghr_sid.g6_relation_tree import (  # noqa: E402
    build_display_address_strict_relations,
)
from poi_gr.methods.ghr_sid.minimum_description import (  # noqa: E402
    CollisionGrouping,
)
from poi_gr.methods.ghr_sid.relation_tree import (  # noqa: E402
    TREE_RELATION_INDEX,
    TREE_RELATION_TYPES,
    build_relation_trie,
    build_tree_relation_values,
    compile_relation_trees,
)
from poi_gr.methods.ghr_sid.unified_relations import (  # noqa: E402
    UNIFIED_RELATION_INDEX,
    UNIFIED_RELATION_TYPES,
    UnifiedRelationMatrix,
)
from poi_gr.methods.qgr_sid.relations import RELATION_TYPES  # noqa: E402


def _grouping(rows: int) -> CollisionGrouping:
    return CollisionGrouping(
        collision_rows=np.arange(rows, dtype=np.int64),
        group_ids=np.zeros(rows, dtype=np.int32),
        group_sizes=np.asarray([rows], dtype=np.int32),
        member_order=np.arange(rows, dtype=np.int64),
        group_offsets=np.asarray([0, rows], dtype=np.int64),
    )


def _raw(child: str = "") -> RawFineRelations:
    return RawFineRelations(
        numeric_values=(-1,) * 14,
        category_major=-1,
        category_mid=-1,
        category_leaf="",
        layer=-1,
        road_entity="",
        parent_entity="",
        child_entity=child,
    )


class RelationTreeTest(unittest.TestCase):
    def test_constant_zone_is_retained_as_building_ancestor(self) -> None:
        values = np.full(
            (2, len(TREE_RELATION_TYPES)), -1, dtype=np.int64
        )
        values[:, TREE_RELATION_INDEX["R_ZONE_NUM"]] = 4
        values[:, TREE_RELATION_INDEX["R_BUILDING"]] = [24, 40]
        compilation = compile_relation_trees(
            grouping=_grouping(2),
            relation_values=values,
            poi_ids=np.asarray([20, 10], dtype=np.int64),
        )
        self.assertEqual(compilation.path_lengths.tolist(), [2, 2])
        self.assertEqual(
            compilation.path_types[0, :2].tolist(),
            [
                TREE_RELATION_INDEX["R_ZONE_NUM"],
                TREE_RELATION_INDEX["R_BUILDING"],
            ],
        )
        self.assertTrue(compilation.semantic_resolved.all())
        self.assertTrue(np.all(compilation.dedup_codes == -1))

    def test_dedup_is_only_added_to_residual_semantic_leaf(self) -> None:
        values = np.full(
            (4, len(TREE_RELATION_TYPES)), -1, dtype=np.int64
        )
        values[:, TREE_RELATION_INDEX["R_ZONE_NUM"]] = [4, 4, 1, 1]
        values[:, TREE_RELATION_INDEX["R_BUILDING"]] = [40, 24, 38, 38]
        values[:, TREE_RELATION_INDEX["R_UNIT"]] = [-1, -1, 2, 2]
        compilation = compile_relation_trees(
            grouping=_grouping(4),
            relation_values=values,
            poi_ids=np.asarray([40, 30, 20, 10], dtype=np.int64),
        )
        self.assertEqual(compilation.dedup_codes.tolist(), [-1, -1, 1, 0])
        self.assertEqual(compilation.metrics["semantic_collision_excess"], 1)
        self.assertEqual(compilation.metrics["final_collision_excess"], 0)
        trie = build_relation_trie(
            range(4), compilation, vocabularies={}
        )
        self.assertTrue(trie)

    def test_explicit_child_building_is_canonicalized(self) -> None:
        unified_values = np.full(
            (1, len(UNIFIED_RELATION_TYPES)), -1, dtype=np.int64
        )
        unified_values[0, UNIFIED_RELATION_INDEX["R_CHILD_NO"]] = 20
        unified = UnifiedRelationMatrix(
            values=unified_values, vocabularies={}, metrics={}
        )
        projected = build_tree_relation_values(
            unified, raw_relations=[_raw("20号楼")]
        )
        self.assertEqual(
            projected[0, TREE_RELATION_INDEX["R_BUILDING"]], 40
        )
        self.assertNotIn("R_CHILD_NO", TREE_RELATION_TYPES)

    def test_weak_category_and_layer_relations_are_not_in_schema(self) -> None:
        self.assertNotIn("R_CATEGORY_LEAF", TREE_RELATION_TYPES)
        self.assertNotIn("R_LAYER", TREE_RELATION_TYPES)
        self.assertNotIn("R_PARENT_P01", TREE_RELATION_TYPES)
        self.assertNotIn("R_COMPOUND_HEAD", TREE_RELATION_TYPES)
        self.assertNotIn("R_INNER", TREE_RELATION_TYPES)

    def test_alias_only_relation_is_excluded_from_tree_source(self) -> None:
        values = build_display_address_strict_relations(
            [
                {
                    "displayname": "冠城大通百旺府5区-4号楼",
                    "address": "北京市海淀区冠城大通百旺府5区",
                    "alias": "北京海淀区永丰嘉园五区4号楼3单元303室",
                }
            ]
        )
        relation_index = {
            relation_type: index
            for index, relation_type in enumerate(RELATION_TYPES)
        }
        self.assertEqual(values[0, relation_index["R_BUILDING"]], 4)
        self.assertEqual(values[0, relation_index["R_UNIT"]], -1)
        self.assertEqual(values[0, relation_index["R_ROOM"]], -1)


if __name__ == "__main__":
    unittest.main()
