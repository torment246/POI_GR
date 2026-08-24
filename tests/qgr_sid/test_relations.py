"""Tests for deterministic QGR-SID numeric relation extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.relations import (  # noqa: E402
    NumericRelationError,
    extract_numeric_relations,
    extract_query_relation_features,
    parse_contextual_integer,
)


def relation_values(payload: dict[str, str]) -> dict[str, int]:
    result = extract_numeric_relations(payload)
    return {item.relation_type: item.numeric_value for item in result.relations}


class ContextualIntegerTest(unittest.TestCase):
    def test_arabic_chinese_positional_and_digit_sequence_forms(self) -> None:
        expected = {
            "0": 0,
            "一": 1,
            "十": 10,
            "十一": 11,
            "二十五": 25,
            "一零": 10,
            "一一": 11,
            "两百零三": 203,
            "一万零二": 10002,
        }
        for raw, value in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_contextual_integer(raw), value)

    def test_invalid_untriggered_text_is_rejected(self) -> None:
        with self.assertRaises(NumericRelationError):
            parse_contextual_integer("三里屯")


class NumericRelationExtractionTest(unittest.TestCase):
    def test_building_zone_and_unit_aliases_normalize_to_same_values(self) -> None:
        result = extract_numeric_relations(
            {
                "displayname": "草桥欣园4区5号楼-3单元",
                "address": "草桥欣园四区5号楼",
                "alias": "草桥欣园四区五号楼三单元｜草桥欣园4区5号楼3单元",
            }
        )
        self.assertEqual(
            {item.relation_type: item.numeric_value for item in result.relations},
            {"R_ZONE_NUM": 4, "R_BUILDING": 5, "R_UNIT": 3},
        )
        self.assertEqual(result.conflicts, ())

    def test_phase_direction_and_building_are_distinct_relations(self) -> None:
        values = relation_values(
            {"displayname": "格兰山水2期南区-18号楼"}
        )
        self.assertEqual(
            values,
            {"R_PHASE": 2, "R_ZONE_DIR": 4, "R_BUILDING": 18},
        )

    def test_address_number_qualifier_subnumber_and_building_do_not_conflict(self) -> None:
        values = relation_values(
            {"displayname": "百子湾东里甲219-1号院5号楼"}
        )
        self.assertEqual(values["R_ADDRESS_NO"], 219)
        self.assertEqual(values["R_QUALIFIER"], 1)
        self.assertEqual(values["R_SUBNO"], 1)
        self.assertEqual(values["R_BUILDING"], 5)

    def test_floor_basement_room_shop_and_leading_entity_number(self) -> None:
        values = relation_values(
            {
                "displayname": "258(凯德MALL)B3层12号铺",
                "address": "F2层301室",
            }
        )
        self.assertEqual(values["R_ENTITY_NO"], 258)
        self.assertEqual(values["R_BASEMENT"], 3)
        self.assertEqual(values["R_SHOP"], 12)
        self.assertEqual(values["R_FLOOR"], 2)
        self.assertEqual(values["R_ROOM"], 301)

    def test_direction_alpha_and_numeric_entrances_are_numeric(self) -> None:
        self.assertEqual(
            relation_values({"displayname": "北2门"}),
            {"R_ENTRANCE_DIR": 0, "R_ENTRANCE_NO": 2},
        )
        self.assertEqual(
            relation_values({"displayname": "A2口"}),
            {"R_ENTRANCE_ALPHA": 1, "R_ENTRANCE_NO": 2},
        )
        self.assertEqual(
            relation_values({"displayname": "3号门"}),
            {"R_ENTRANCE_NO": 3},
        )

    def test_long_codes_are_not_misread_as_basement_floor_or_entrance(self) -> None:
        values = relation_values(
            {
                "displayname": "晋B2008刀削面三六零三门诊部",
                "address": "B2101室F5013铺",
            }
        )
        self.assertNotIn("R_BASEMENT", values)
        self.assertNotIn("R_FLOOR", values)
        self.assertNotIn("R_ENTRANCE_NO", values)

    def test_short_latin_floor_forms_remain_supported(self) -> None:
        self.assertEqual(
            relation_values({"displayname": "地下B3层与F2层"})[
                "R_BASEMENT"
            ],
            3,
        )
        self.assertEqual(
            relation_values({"displayname": "2F层"})["R_FLOOR"],
            2,
        )

    def test_lower_priority_alias_conflict_does_not_override_displayname(self) -> None:
        result = extract_numeric_relations(
            {
                "displayname": "格兰山水18号楼",
                "alias": "格兰山水十八号楼｜格兰山水19号楼",
            }
        )
        self.assertEqual(relation_values({"displayname": "格兰山水18号楼"})["R_BUILDING"], 18)
        self.assertEqual(
            {item.relation_type: item.numeric_value for item in result.relations}[
                "R_BUILDING"
            ],
            18,
        )
        conflict = next(
            item for item in result.conflicts if item.relation_type == "R_BUILDING"
        )
        self.assertEqual(conflict.candidate_values, (18, 19))
        self.assertEqual(conflict.selected_value, 18)

    def test_same_priority_ambiguity_suppresses_relation(self) -> None:
        result = extract_numeric_relations(
            {"displayname": "18号楼与19号楼服务台"}
        )
        self.assertNotIn(
            "R_BUILDING", {item.relation_type for item in result.relations}
        )
        self.assertIsNone(result.conflicts[0].selected_value)

    def test_out_of_vocab_number_is_preserved_for_audit(self) -> None:
        values = relation_values({"displayname": "2056号楼"})
        self.assertEqual(values["R_BUILDING"], 2056)

    def test_query_features_keep_typed_and_untyped_numeric_signals(self) -> None:
        features = extract_query_relation_features("草桥欣园4区5号楼 219")
        self.assertIn(("R_ZONE_NUM", 4), features.typed_relations)
        self.assertIn(("R_BUILDING", 5), features.typed_relations)
        self.assertEqual(features.numeric_values, (4, 5, 219))

    def test_query_untyped_numbers_do_not_convert_chinese_name_characters(self) -> None:
        features = extract_query_relation_features("三里屯")
        self.assertEqual(features.typed_relations, ())
        self.assertEqual(features.numeric_values, ())


if __name__ == "__main__":
    unittest.main()
