"""Synthetic tests ensuring collision examples cannot silently become singletons."""

from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

from qg_prqk.sid.collision_visualization import (
    collision_bucket_sizes,
    collision_members,
    collision_members_html,
    select_collision_anchors,
)


class CollisionVisualizationTest(unittest.TestCase):
    def test_sizes_require_all_three_tokens(self):
        sid = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]])
        per_row, sizes = collision_bucket_sizes(sid)
        np.testing.assert_array_equal(per_row, [2, 2, 1, 1])
        np.testing.assert_array_equal(sizes, [2, 1, 1])

    def test_shared_anchors_are_deterministic_and_meet_all_three_bounds(self):
        first = np.array([[0, 0, 0]] * 3 + [[0, 0, 1]] * 3 + [[1, 0, 0]] * 3)
        second = first.copy()
        second[0, 2] = 4
        categories = np.array(["a"] * 6 + ["b"] * 3)
        sids = [first, second, first]
        anchors, audit = select_collision_anchors(
            sids, categories, ["a", "b"], seed=42, min_size=3, max_size=3
        )
        repeat, _ = select_collision_anchors(
            sids, categories, ["a", "b"], seed=42, min_size=3, max_size=3
        )
        np.testing.assert_array_equal(anchors, repeat)
        self.assertEqual(audit["common_eligible_poi_per_category"], {"a": 3, "b": 3})
        for sid in sids:
            sizes, _ = collision_bucket_sizes(sid)
            np.testing.assert_array_equal(sizes[anchors], [3, 3])

    def test_no_eligible_class_fails_instead_of_selecting_singletons(self):
        sid = np.array([[0, 0, 0], [0, 0, 1]])
        with self.assertRaisesRegex(ValueError, "不退回单例"):
            select_collision_anchors(
                [sid] * 3, np.array(["a", "b"]), ["a"], seed=42, min_size=2, max_size=20
            )

    def test_invalid_minimum_is_rejected(self):
        sid = np.array([[0, 0, 0]] * 3)
        with self.assertRaises(ValueError):
            select_collision_anchors(
                [sid] * 3, np.array(["a"] * 3), ["a"], seed=42, min_size=1, max_size=20
            )

    def test_invalid_sid_range_and_type_are_rejected(self):
        for sid in [
            np.array([[512, 0, 0]]),
            np.array([[0.0, 0, 0]]),
            np.array([[-1, 0, 0]]),
        ]:
            with self.assertRaises(ValueError):
                collision_bucket_sizes(sid)

    def test_complete_members_do_not_filter_categories_or_regions(self):
        table = pa.table({"poi_id": ["p1", "p2", "p3"], "category": ["a", "b", "c"]})
        sid = np.array([[0, 0, 0]] * 2 + [[0, 0, 1]])
        members = collision_members(
            table, sid, 0, np.array(["wx4g0", "wx4g1", "wx4g2"])
        )
        self.assertEqual([x["poi_id"] for x in members], ["p1", "p2"])
        self.assertEqual([x["geohash5"] for x in members], ["wx4g0", "wx4g1"])
        with self.assertRaises(ValueError):
            collision_members(table, sid, 2, np.array(["a", "b", "c"]))

    def test_duplicate_ids_do_not_count_as_a_collision(self):
        table = pa.table({"poi_id": ["p1", "p1"]})
        with self.assertRaises(ValueError):
            collision_members(table, np.array([[0, 0, 0]] * 2), 0, np.array(["a", "a"]))

    def test_html_escapes_business_text(self):
        member = {
            "poi_id": "p1",
            "displayname": "<script>alert(1)</script>",
            "category": "x&y",
            "geohash5": "wx4g0",
            "lng": 116,
            "lat": 40,
        }
        entry = {
            "level": 3,
            "anchor_index": 0,
            "prefix": [1, 2, 3],
            "poi_count": 2,
            "region_geohash5": {"counts": {"wx4g0": 2}},
            "members": [member, dict(member, poi_id="p2")],
        }
        rendered = collision_members_html(
            {
                "categories": ["a"],
                "anchor_poi_ids": ["p1"],
                "methods": {"A0": {"prefix_examples": [entry]}},
            }
        )
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("p2", rendered)


if __name__ == "__main__":
    unittest.main()
