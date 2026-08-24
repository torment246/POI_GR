"""Tests for fine-grained static entity relations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.entity_structure import (  # noqa: E402
    PostR3Grouping,
    compile_fine_relation_paths,
    reconstruct_post_r3_groups,
)
from poi_gr.methods.ghr_sid.fine_relations import (  # noqa: E402
    CHARACTER_VOCABULARY_NAME,
    FINE_RELATION_TYPES,
    build_fine_relation_matrix,
    decode_fine_relation_value,
    extract_raw_fine_relations,
    split_parent_child,
)


class FineRelationExtractionTest(unittest.TestCase):
    def test_splits_code_parent_and_hospital_child(self) -> None:
        self.assertEqual(
            split_parent_child("E-058(悦荟·万科购物中心)"),
            ("悦荟·万科购物中心", "E-058"),
        )
        self.assertEqual(
            split_parent_child("中国中医科学院眼科医院-针灸科"),
            ("中国中医科学院眼科医院", "针灸科"),
        )
        self.assertEqual(
            split_parent_child("德胜门外大街200号203号"),
            ("德胜门外大街200号", "203号"),
        )
        self.assertEqual(
            split_parent_child("3-208(银泰百货)"),
            ("银泰百货", "3-208"),
        )

    def test_extracts_typed_internal_numbers(self) -> None:
        parking = extract_raw_fine_relations(
            {
                "displayname": "华冠购物中心停车场-178车位",
                "address": "北京市房山区秋树街38号",
                "category_code": "191100",
                "layer": "11",
            }
        )
        parking_index = FINE_RELATION_TYPES.index("R_FINE_PARKING_SLOT")
        child_index = FINE_RELATION_TYPES.index("R_FINE_CHILD_NO")
        self.assertEqual(parking.numeric_values[parking_index], 178)
        self.assertEqual(parking.numeric_values[child_index], 178)

        code = extract_raw_fine_relations(
            {
                "displayname": "E-058(悦荟·万科购物中心)",
                "address": "北京市昌平区南环路10号",
                "category_code": "132000",
                "layer": 11,
            }
        )
        alpha_index = FINE_RELATION_TYPES.index("R_FINE_ALPHA_PREFIX")
        number_index = FINE_RELATION_TYPES.index("R_FINE_ALPHA_ENTITY_NO")
        code_index = FINE_RELATION_TYPES.index("R_FINE_ENTITY_CODE")
        self.assertEqual(code.numeric_values[alpha_index], -1)
        self.assertEqual(code.numeric_values[number_index], -1)
        self.assertEqual(
            decode_fine_relation_value(
                "R_FINE_ENTITY_CODE",
                code.numeric_values[code_index],
                {},
            ),
            "E058",
        )

    def test_extracts_expanded_compound_and_building_numbers(self) -> None:
        compound = extract_raw_fine_relations(
            {
                "displayname": "北辰红橡墅5区-12-2栋",
                "address": "北京市顺义区",
                "category_code": "282000",
                "layer": 11,
            }
        )
        head_index = FINE_RELATION_TYPES.index("R_FINE_COMPOUND_HEAD_NO")
        tail_index = FINE_RELATION_TYPES.index("R_FINE_COMPOUND_TAIL_NO")
        building_index = FINE_RELATION_TYPES.index("R_FINE_BUILDING_NO")
        self.assertEqual(compound.numeric_values[head_index], 12)
        self.assertEqual(compound.numeric_values[tail_index], 2)
        self.assertEqual(compound.numeric_values[building_index], 2)

        building = extract_raw_fine_relations(
            {
                "displayname": "嘉浩别墅4036栋-楼门",
                "address": "北京市顺义区",
                "category_code": "282000",
                "layer": 11,
            }
        )
        self.assertEqual(building.numeric_values[building_index], 4036)

        address = extract_raw_fine_relations(
            {
                "displayname": "安定门内大街135-1号",
                "address": "北京市东城区",
                "category_code": "282000",
                "layer": 11,
            }
        )
        address_index = FINE_RELATION_TYPES.index("R_FINE_ADDRESS_MAIN_NO")
        self.assertEqual(address.numeric_values[address_index], 135)
        self.assertEqual(address.numeric_values[head_index], 135)

        numeric_code = extract_raw_fine_relations(
            {
                "displayname": "3-208(银泰百货)",
                "address": "北京市大兴区",
                "category_code": "132000",
                "layer": 11,
            }
        )
        child_index = FINE_RELATION_TYPES.index("R_FINE_CHILD_NO")
        self.assertEqual(numeric_code.numeric_values[child_index], 208)
        self.assertEqual(numeric_code.numeric_values[head_index], 3)

    def test_preserves_complete_alphanumeric_entity_code(self) -> None:
        relation = extract_raw_fine_relations(
            {
                "displayname": "绿茵花园别墅西区-春风区C12B号楼",
                "address": "北京市顺义区",
                "category_code": "282000",
                "layer": 11,
            }
        )
        code_index = FINE_RELATION_TYPES.index("R_FINE_BUILDING_CODE")
        child_index = FINE_RELATION_TYPES.index("R_FINE_CHILD_NO")
        prefix_index = FINE_RELATION_TYPES.index("R_FINE_ALPHA_PREFIX")
        number_index = FINE_RELATION_TYPES.index("R_FINE_ALPHA_ENTITY_NO")
        encoded = relation.numeric_values[code_index]
        self.assertGreater(encoded, 0)
        self.assertEqual(
            decode_fine_relation_value(
                "R_FINE_BUILDING_CODE", encoded, {}
            ),
            "C12B",
        )
        self.assertEqual(relation.numeric_values[child_index], -1)
        self.assertEqual(relation.numeric_values[prefix_index], -1)
        self.assertEqual(relation.numeric_values[number_index], -1)

        matrix = build_fine_relation_matrix(
            [relation, relation], entity_min_support=2, entity_vocab_size=16
        )
        self.assertEqual(matrix.values.dtype, np.int64)
        self.assertEqual(int(matrix.values[0, code_index]), encoded)

        for displayname, relation_type, expected in (
            (
                "绿茵花园别墅西区-夏荫区12A号楼",
                "R_FINE_BUILDING_CODE",
                "12A",
            ),
            (
                "绿茵花园别墅西区春风C11号",
                "R_FINE_ENTITY_CODE",
                "C11",
            ),
        ):
            with self.subTest(displayname=displayname):
                candidate = extract_raw_fine_relations(
                    {
                        "displayname": displayname,
                        "address": "北京市顺义区",
                        "category_code": "282000",
                        "layer": 11,
                    }
                )
                relation_index = FINE_RELATION_TYPES.index(relation_type)
                self.assertEqual(
                    decode_fine_relation_value(
                        relation_type,
                        candidate.numeric_values[relation_index],
                        {},
                    ),
                    expected,
                )

    def test_entity_vocab_excludes_singletons(self) -> None:
        raw = [
            extract_raw_fine_relations(
                {
                    "displayname": name,
                    "address": "北京市海淀区复兴路",
                    "category_code": "241100",
                    "layer": 11,
                }
            )
            for name in ("医院-针灸科", "医院-针灸科", "医院-唯一科")
        ]
        matrix = build_fine_relation_matrix(
            raw, entity_min_support=2, entity_vocab_size=16
        )
        child_column = FINE_RELATION_TYPES.index("R_FINE_CHILD_ENTITY")
        self.assertGreater(matrix.values[0, child_column], 0)
        self.assertEqual(matrix.values[2, child_column], -1)
        self.assertNotIn(
            "唯一科", matrix.vocabularies["R_FINE_CHILD_ENTITY"]
        )

    def test_positional_entity_characters_are_shared_numeric_relations(self) -> None:
        raw = [
            extract_raw_fine_relations(
                {
                    "displayname": name,
                    "address": "北京市海淀区复兴路",
                    "category_code": "241100",
                    "layer": 11,
                }
            )
            for name in ("医院-针灸科", "医院-激光室")
        ]
        matrix = build_fine_relation_matrix(
            raw, entity_min_support=2, entity_vocab_size=16
        )
        first_child = FINE_RELATION_TYPES.index(
            "R_FINE_CHILD_PREFIX_CHAR_01"
        )
        self.assertNotEqual(
            matrix.values[0, first_child], matrix.values[1, first_child]
        )
        vocabulary = matrix.vocabularies[CHARACTER_VOCABULARY_NAME]
        self.assertEqual(
            vocabulary[int(matrix.values[0, first_child]) - 1], "针"
        )


class FineRelationStructureTest(unittest.TestCase):
    def test_reconstructs_only_rows_entering_fine_stage(self) -> None:
        base = np.asarray([1, 1, 1, 1, 2, 2], dtype=np.int64)
        root = np.asarray([4, 4, 4, 4, 5, 5], dtype=np.uint8)
        geo = np.full((6, 2), -1, dtype=np.int16)
        geo_len = np.zeros(6, dtype=np.uint8)
        relation_types = np.full((6, 3), -1, dtype=np.int16)
        relation_values = np.full((6, 3), -1, dtype=np.int16)
        stage = np.asarray([3, 3, 1, 2, 0, 0], dtype=np.uint8)
        grouping = reconstruct_post_r3_groups(
            base_sid_keys=base,
            root_prefix_lengths=root,
            coarse_geo_codes=geo,
            coarse_geo_lengths=geo_len,
            coarse_relation_types=relation_types,
            coarse_relation_values=relation_values,
            resolution_stage=stage,
        )
        self.assertEqual(grouping.collision_rows.tolist(), [0, 1, 4, 5])
        self.assertEqual(sorted(grouping.group_sizes.tolist()), [2, 2])

    def test_minimal_paths_stop_after_one_distinguishing_relation(self) -> None:
        grouping = PostR3Grouping(
            collision_rows=np.arange(4, dtype=np.int64),
            group_ids=np.zeros(4, dtype=np.int32),
            group_sizes=np.asarray([4], dtype=np.int32),
            member_order=np.arange(4, dtype=np.int64),
            group_offsets=np.asarray([0, 4], dtype=np.int64),
        )
        relations = np.asarray(
            [[1, -1], [2, -1], [-1, 3], [-1, 4]], dtype=np.int16
        )
        compilation = compile_fine_relation_paths(
            grouping=grouping,
            relation_values=relations,
            relation_count=2,
        )
        self.assertTrue(compilation.resolved.all())
        self.assertEqual(
            np.sum(compilation.path_types >= 0, axis=1).tolist(), [1, 1, 1, 1]
        )

    def test_identical_relations_remain_unresolved(self) -> None:
        grouping = PostR3Grouping(
            collision_rows=np.arange(2, dtype=np.int64),
            group_ids=np.zeros(2, dtype=np.int32),
            group_sizes=np.asarray([2], dtype=np.int32),
            member_order=np.arange(2, dtype=np.int64),
            group_offsets=np.asarray([0, 2], dtype=np.int64),
        )
        compilation = compile_fine_relation_paths(
            grouping=grouping,
            relation_values=np.asarray([[7], [7]], dtype=np.int16),
            relation_count=1,
        )
        self.assertFalse(compilation.resolved.any())
        self.assertEqual(compilation.metrics["collision_excess"], 1)

    def test_relation_presence_can_distinguish_from_missing_eos(self) -> None:
        grouping = PostR3Grouping(
            collision_rows=np.arange(2, dtype=np.int64),
            group_ids=np.zeros(2, dtype=np.int32),
            group_sizes=np.asarray([2], dtype=np.int32),
            member_order=np.arange(2, dtype=np.int64),
            group_offsets=np.asarray([0, 2], dtype=np.int64),
        )
        compilation = compile_fine_relation_paths(
            grouping=grouping,
            relation_values=np.asarray([[7], [-1]], dtype=np.int16),
            relation_count=1,
            allow_presence_split=True,
        )
        self.assertTrue(compilation.resolved.all())
        self.assertEqual(
            np.sum(compilation.path_types >= 0, axis=1).tolist(), [1, 0]
        )

    def test_wide_relation_values_are_not_truncated(self) -> None:
        grouping = PostR3Grouping(
            collision_rows=np.arange(2, dtype=np.int64),
            group_ids=np.zeros(2, dtype=np.int32),
            group_sizes=np.asarray([2], dtype=np.int32),
            member_order=np.arange(2, dtype=np.int64),
            group_offsets=np.asarray([0, 2], dtype=np.int64),
        )
        relation_values = np.asarray(
            [[2_000_000_001], [2_000_000_002]], dtype=np.int64
        )
        compilation = compile_fine_relation_paths(
            grouping=grouping,
            relation_values=relation_values,
            relation_count=1,
        )
        self.assertEqual(compilation.path_values.dtype, np.int64)
        self.assertEqual(
            compilation.path_values[:, 0].tolist(),
            relation_values[:, 0].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
