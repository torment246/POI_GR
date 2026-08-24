"""Tests for direct G6-to-entity minimum descriptions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.fine_relations import (  # noqa: E402
    build_fine_relation_matrix,
    extract_raw_fine_relations,
)
from poi_gr.methods.ghr_sid.minimum_description import (  # noqa: E402
    CollisionGrouping,
    compile_minimum_descriptions,
)
from poi_gr.methods.ghr_sid.unified_relations import (  # noqa: E402
    SEMANTIC_TIERS,
    UNIFIED_RELATION_INDEX,
    UNIFIED_RELATION_TYPES,
    build_unified_relation_matrix,
    decode_unified_relation_value,
)
from poi_gr.methods.qgr_sid.relations import (  # noqa: E402
    RELATION_TYPES,
    extract_numeric_relations,
)


def _grouping(rows: int) -> CollisionGrouping:
    return CollisionGrouping(
        collision_rows=np.arange(rows, dtype=np.int64),
        group_ids=np.zeros(rows, dtype=np.int32),
        group_sizes=np.asarray([rows], dtype=np.int32),
        member_order=np.arange(rows, dtype=np.int64),
        group_offsets=np.asarray([0, rows], dtype=np.int64),
    )


def _strict_matrix(records: list[dict[str, object]]) -> np.ndarray:
    relation_index = {
        relation_type: index
        for index, relation_type in enumerate(RELATION_TYPES)
    }
    matrix = np.full(
        (len(records), len(RELATION_TYPES)), -1, dtype=np.int16
    )
    for row, record in enumerate(records):
        for relation in extract_numeric_relations(record).relations:
            matrix[row, relation_index[relation.relation_type]] = (
                relation.numeric_value
            )
    return matrix


class UnifiedRelationTest(unittest.TestCase):
    def test_c12b_is_one_building_relation_after_g6(self) -> None:
        names = (
            "绿茵花园别墅西区秋月区",
            "绿茵花园别墅西区四季区",
            "绿茵花园别墅西区-夏荫区12A号楼",
            "绿茵花园别墅西区-春风区C12B号楼",
            "绿茵花园别墅西区",
            "绿茵花园别墅西区春风C11号",
            "绿茵花园别墅西区-综合楼",
            "绿茵花园别墅西区春风C区",
        )
        categories = (
            "281011",
            "281011",
            "282000",
            "282000",
            "281011",
            "282000",
            "288000",
            "281011",
        )
        records = [
            {
                "displayname": name,
                "address": "北京市顺义区",
                "alias": "",
                "category_code": category,
                "category": "",
                "layer": 11,
            }
            for name, category in zip(names, categories, strict=True)
        ]
        fine = build_fine_relation_matrix(
            [extract_raw_fine_relations(record) for record in records],
            entity_min_support=2,
            entity_vocab_size=128,
        )
        unified = build_unified_relation_matrix(
            strict_relations=_strict_matrix(records), fine_relations=fine
        )
        compilation = compile_minimum_descriptions(
            grouping=_grouping(len(records)),
            relation_values=unified.values,
            semantic_tiers=SEMANTIC_TIERS,
            max_pairs=3,
            max_tokens=6,
        )
        target = 3
        active = np.flatnonzero(compilation.path_types[target] >= 0)
        self.assertEqual(active.tolist(), [0])
        relation_index = int(compilation.path_types[target, 0])
        self.assertEqual(UNIFIED_RELATION_TYPES[relation_index], "R_BUILDING")
        self.assertEqual(
            decode_unified_relation_value(
                "R_BUILDING",
                int(compilation.path_values[target, 0]),
                unified.vocabularies,
            ),
            "C12B",
        )

    def test_uses_two_relations_only_when_one_cannot_distinguish(self) -> None:
        values = np.full(
            (3, len(UNIFIED_RELATION_TYPES)), -1, dtype=np.int64
        )
        building = UNIFIED_RELATION_INDEX["R_BUILDING"]
        unit = UNIFIED_RELATION_INDEX["R_UNIT"]
        values[:, building] = [2, 2, 4]
        values[:, unit] = [2, 4, 2]
        compilation = compile_minimum_descriptions(
            grouping=_grouping(3),
            relation_values=values,
            semantic_tiers=SEMANTIC_TIERS,
            max_pairs=3,
            max_tokens=6,
        )
        self.assertEqual(
            np.sum(compilation.path_types >= 0, axis=1).tolist(), [2, 1, 1]
        )
        self.assertTrue(compilation.resolved.all())

    def test_identical_entities_keep_the_same_empty_description(self) -> None:
        values = np.full(
            (2, len(UNIFIED_RELATION_TYPES)), -1, dtype=np.int64
        )
        values[:, UNIFIED_RELATION_INDEX["R_BUILDING"]] = 4
        compilation = compile_minimum_descriptions(
            grouping=_grouping(2),
            relation_values=values,
            semantic_tiers=SEMANTIC_TIERS,
        )
        self.assertFalse(compilation.resolved.any())
        self.assertEqual(compilation.metrics["collision_excess"], 1)


if __name__ == "__main__":
    unittest.main()
