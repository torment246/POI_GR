"""Synthetic brute-force checks for prefix migrations and singleton-safe outcomes."""

from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

from qg_prqk.sid.migration_analysis import (
    compare_peer_labels,
    compare_prefix_memberships,
    describe_example,
    prefix_keys,
    select_migration_rows,
    summarize_migration,
)


class SIDMigrationTest(unittest.TestCase):
    def test_exact_memberships_match_brute_force(self):
        rng = np.random.default_rng(13)
        before = rng.integers(0, 3, (70, 3))
        after = rng.integers(0, 3, (70, 3))
        labels = rng.integers(0, 4, 70)
        for depth in (1, 2, 3):
            result = compare_prefix_memberships(before, after, depth, labels, labels)
            for row in range(70):
                old = set(
                    np.flatnonzero(
                        np.all(before[:, :depth] == before[row, :depth], axis=1)
                    )
                )
                new = set(
                    np.flatnonzero(
                        np.all(after[:, :depth] == after[row, :depth], axis=1)
                    )
                )
                self.assertEqual(result["common_size"][row], len(old & new))
                self.assertEqual(result["membership_changed"][row], old != new)
                old.remove(row)
                new.remove(row)
                if old and new:
                    p = sum(labels[x] == labels[row] for x in old) / len(old)
                    q = sum(labels[x] == labels[row] for x in new) / len(new)
                    self.assertAlmostEqual(result["category"]["before_rate"][row], p)
                    self.assertAlmostEqual(result["category"]["after_rate"][row], q)
                    self.assertEqual(result["category"]["outcome"][row], np.sign(q - p))
                else:
                    self.assertEqual(result["category"]["outcome"][row], -2)

    def test_label_permutation_changes_codes_but_not_peer_membership(self):
        before = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]])
        after = before.copy()
        after[:, 0] = 1 - before[:, 0]
        result = compare_prefix_memberships(
            before, after, 3, np.array([0, 0, 1, 1]), np.zeros(4)
        )
        self.assertTrue(result["prefix_changed"].all())
        self.assertFalse(result["membership_changed"].any())
        self.assertTrue((result["peer_jaccard"] == 1).all())

    def test_same_code_can_have_different_members(self):
        before = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        after = before.copy()
        after[1, 0] = 1
        result = compare_prefix_memberships(before, after, 1, np.zeros(3), np.zeros(3))
        self.assertFalse(result["prefix_changed"][0])
        self.assertTrue(result["membership_changed"][0])

    def test_singleton_is_not_perfect_semantic_improvement(self):
        result = compare_peer_labels(
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
            np.array([1, 2, 1]),
            np.array([1, 1, 2]),
        )
        np.testing.assert_array_equal(result["outcome"], [-2, -2, -2])
        self.assertTrue(np.isnan(result["after_rate"][1]))

    def test_rational_ties_and_minorities_use_anchor_label(self):
        result = compare_peer_labels(
            np.array([2, 1]), np.array([3, 2]), np.array([3, 10]), np.array([5, 10])
        )
        np.testing.assert_array_equal(result["outcome"], [0, 1])
        self.assertEqual(result["before_rate"][1], 0)
        self.assertAlmostEqual(result["after_rate"][1], 1 / 9)

    def test_outcome_and_transition_counts_conserve_population(self):
        before = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]])
        after = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]])
        summary = summarize_migration(
            compare_prefix_memberships(before, after, 3, np.zeros(4), np.zeros(4)), 3
        )
        self.assertEqual(sum(summary["singleton_transitions"].values()), 4)
        self.assertEqual(summary["singleton_transitions"]["collision_to_singleton"], 1)
        self.assertEqual(summary["singleton_transitions"]["singleton_to_collision"], 1)
        self.assertEqual(sum(summary["category"]["counts"].values()), 4)

    def test_samples_deterministic_and_do_not_fallback_to_other_strata(self):
        comparison = {
            "prefix_changed": np.ones(9, dtype=bool),
            "membership_changed": np.ones(9, dtype=bool),
            "before_size": np.ones(9) * 3,
            "after_size": np.ones(9) * 4,
            "category": {"outcome": np.array([1] * 3 + [0] * 3 + [-1] * 3)},
        }
        first = select_migration_rows(comparison, 2, 42)
        self.assertEqual(first, select_migration_rows(comparison, 2, 42))
        self.assertEqual([x["eligible_poi"] for x in first], [3, 3, 3])
        comparison["after_size"][:] = 21
        self.assertTrue(
            all(
                x["anchor_row"] is None
                for x in select_migration_rows(comparison, 2, 42)
            )
        )

    def test_full_member_sets_and_bounded_previews(self):
        before = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        after = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0]])
        table = pa.table({"poi_row_index": [0, 1, 2], "poi_id": ["a", "b", "c"]})
        result = compare_prefix_memberships(
            before, after, 1, np.array([0, 1, 0]), np.array([0, 1, 0])
        )
        example, arrays = describe_example(
            table, before, after, 1, 0, result, np.array(["x", "y", "x"]), "test", 1
        )
        self.assertEqual(
            [example["groups"][x]["count"] for x in ("retained", "removed", "added")],
            [1, 1, 1],
        )
        np.testing.assert_array_equal(arrays["test_removed_poi_ids"], ["b"])
        self.assertEqual(example["groups"]["added"]["preview"][0]["old_prefix"], [1])

    def test_invalid_tokens_or_unaligned_rows_fail(self):
        with self.assertRaises(ValueError):
            prefix_keys(np.array([[512, 0, 0]]), 1)
        with self.assertRaises(ValueError):
            prefix_keys(np.array([[0.0, 0, 0]]), 1)
        with self.assertRaises(ValueError):
            compare_prefix_memberships(
                np.zeros((3, 3), dtype=int),
                np.zeros((2, 3), dtype=int),
                1,
                np.zeros(3),
                np.zeros(3),
            )

    def test_html_escapes_names_and_explains_preview_limits(self):
        from qg_prqk.sid.migration_plots import VARIANTS, migration_examples_html

        before = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        after = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0]])
        table = pa.table(
            {
                "poi_row_index": [0, 1, 2],
                "poi_id": ["a", "b", "c"],
                "displayname": ["<script>alert(1)</script>", "移出", "移入"],
                "category": ["甲", "乙", "甲"],
            }
        )
        labels = np.array([0, 1, 0])
        comparison = compare_prefix_memberships(before, after, 1, labels, labels)
        example, _ = describe_example(
            table, before, after, 1, 0, comparison, np.array(["x", "y", "x"]), "test"
        )
        rendered = migration_examples_html({name: [example] for name in VARIANTS})
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("预览每组最多 5 个 POI", rendered)
        self.assertIn("memberships.npz", rendered)


if __name__ == "__main__":
    unittest.main()
