from __future__ import annotations

import unittest

import numpy as np

from qg_prqk.sid.relational_data import RelationalCodebookDataError
from qg_prqk.sid.relational_evaluation import (
    FULL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _evaluation_phase,
    _evaluation_schema,
    partition_category_metrics,
)
from qg_prqk.sid.relational_quantizer import (
    build_category_cost_lookup,
    build_category_costs,
    objective_increase_streak,
    warmup_weights,
)


class RelationalCodebookWarmupTest(unittest.TestCase):
    def test_uses_half_query_graph_then_linear_ramp(self) -> None:
        self.assertEqual(
            warmup_weights(
                1, query_target=0.1, graph_target=0.2, category_target=0.08
            ),
            (0.05, 0.1, 0.0),
        )
        self.assertEqual(
            warmup_weights(
                6, query_target=0.1, graph_target=0.2, category_target=0.08
            ),
            (0.1, 0.2, 0.08),
        )
        self.assertEqual(
            warmup_weights(
                20, query_target=0.1, graph_target=0.2, category_target=0.08
            ),
            (0.1, 0.2, 0.08),
        )

    def test_rejects_invalid_iteration(self) -> None:
        with self.assertRaises(RelationalCodebookDataError):
            warmup_weights(
                0, query_target=0.1, graph_target=0.1, category_target=0.08
            )

    def test_objective_increase_gate_starts_after_warmup(self) -> None:
        streak = 0
        for iteration in (3, 4, 5, 6):
            streak = objective_increase_streak(
                streak,
                -0.1,
                iteration=iteration,
                warmup_end_iteration=6,
                tolerance=1.0e-4,
            )
            self.assertEqual(streak, 0)
        self.assertEqual(
            objective_increase_streak(
                streak,
                -0.1,
                iteration=7,
                warmup_end_iteration=6,
                tolerance=1.0e-4,
            ),
            1,
        )

    def test_evaluation_contract_preserves_sample_and_versions_full(self) -> None:
        self.assertEqual(_evaluation_schema("sample"), SCHEMA_VERSION)
        self.assertEqual(_evaluation_schema("full"), FULL_SCHEMA_VERSION)
        self.assertEqual(_evaluation_phase("sample"), "P6-CAT-SAMPLE-EVALUATION")
        self.assertEqual(_evaluation_phase("full"), "P6-CAT-FULL-EVALUATION")


class RelationalCodebookCategoryCostTest(unittest.TestCase):
    def test_compact_lookup_materializes_the_same_s1_rows(self) -> None:
        assignments = np.array([0, 0, 1, 1], dtype=np.int32)
        categories = np.array([0, 0, 1, 1], dtype=np.int32)
        lookup = build_category_cost_lookup(
            assignments, categories, 2, smoothing=1.0
        )
        dense = build_category_costs(assignments, categories, 2, smoothing=1.0)
        np.testing.assert_array_equal(lookup.costs_for_rows(0, 4), dense)
        np.testing.assert_array_equal(
            lookup.selected_costs(assignments), dense[np.arange(4), assignments]
        )

    def test_s1_cost_prefers_the_code_with_matching_category(self) -> None:
        assignments = np.array([0, 0, 1, 1], dtype=np.int32)
        categories = np.array([0, 0, 1, 1], dtype=np.int32)
        costs = build_category_costs(
            assignments, categories, 2, smoothing=1.0
        )
        self.assertLess(costs[0, 0], costs[0, 1])
        self.assertLess(costs[3, 1], costs[3, 0])

    def test_s2_cost_is_conditioned_on_s1_parent(self) -> None:
        parents = np.array([0, 0, 1, 1], dtype=np.int32)
        assignments = np.array([0, 1, 0, 1], dtype=np.int32)
        categories = np.array([0, 1, 1, 0], dtype=np.int32)
        costs = build_category_costs(
            assignments,
            categories,
            2,
            smoothing=1.0,
            parent_assignments=parents,
            fine_to_coarse=np.array([0, 0], dtype=np.int16),
        )
        self.assertLess(costs[0, 0], costs[0, 1])
        self.assertLess(costs[2, 0], costs[2, 1])

    def test_compact_lookup_materializes_the_same_s2_rows(self) -> None:
        parents = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        assignments = np.array([0, 1, 0, 1, 1], dtype=np.int32)
        categories = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        lookup = build_category_cost_lookup(
            assignments,
            categories,
            2,
            smoothing=1.0,
            parent_assignments=parents,
            fine_to_coarse=np.array([0, 0], dtype=np.int16),
        )
        dense = build_category_costs(
            assignments,
            categories,
            2,
            smoothing=1.0,
            parent_assignments=parents,
            fine_to_coarse=np.array([0, 0], dtype=np.int16),
        )
        np.testing.assert_array_equal(lookup.costs_for_rows(0, 5), dense)

    def test_rejects_missing_s2_hierarchy(self) -> None:
        with self.assertRaises(RelationalCodebookDataError):
            build_category_costs(
                np.array([0]),
                np.array([0]),
                2,
                smoothing=1.0,
                parent_assignments=np.array([0]),
            )

    def test_path_category_metric_differs_from_token_only(self) -> None:
        s1 = np.array([0, 0, 1, 1], dtype=np.int32)
        s2 = np.array([0, 1, 0, 1], dtype=np.int32)
        categories = np.array([0, 1, 1, 0], dtype=np.int32)
        token = partition_category_metrics(s2, categories)
        path = partition_category_metrics(np.column_stack((s1, s2)), categories)
        self.assertLess(token["purity"], path["purity"])
        self.assertEqual(path["purity"], 1.0)


if __name__ == "__main__":
    unittest.main()
