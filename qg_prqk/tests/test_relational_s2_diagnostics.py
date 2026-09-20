from __future__ import annotations

import copy
import unittest

from qg_prqk.sid.relational_attribution import NEAR_ZERO_QUERY_PRIOR
from qg_prqk.sid.relational_s2_diagnostics import (
    BRANCH_ORDER,
    additive_s2_decomposition,
    build_s2_diagnostic_branches,
    s2_factor_contrasts,
)


class RelationalS2DiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.s2 = {
            "query_distortion_weight": 0.1,
            "graph_alignment_weight": 0.1,
            "category_weight": 0.08,
            "category_smoothing": 16,
            "query_centroid_shrinkage": 32,
        }
        self.prqk = {
            "seed": 42,
            "topk_refinement": {
                "enabled": True,
                "topk": 5,
                "beta": 15,
                "max_iter": 5,
            },
        }

    def test_branches_change_only_declared_s2_factors(self) -> None:
        before = copy.deepcopy((self.s2, self.prqk))
        branches = build_s2_diagnostic_branches(self.s2, self.prqk)
        self.assertEqual(tuple(branches), BRANCH_ORDER)
        self.assertEqual((self.s2, self.prqk), before)
        self.assertFalse(branches["topk_off"]["prqk"]["topk_refinement"]["enabled"])
        self.assertEqual(branches["graph_off"]["s2"]["graph_alignment_weight"], 0.0)
        self.assertEqual(
            branches["near_zero_query_prior"]["s2"]["query_centroid_shrinkage"],
            NEAR_ZERO_QUERY_PRIOR,
        )
        self.assertEqual(branches["category_off"]["s2"]["category_weight"], 0.0)
        self.assertEqual(branches["canonical"]["s2"], self.s2)

    def test_additive_decomposition_closes_exactly(self) -> None:
        result = additive_s2_decomposition(
            p5_path=0.59,
            p5_s2_on_p6_s1=0.61,
            p6_s2_content_optimal=0.60,
            p6_s2_coupled=0.62,
        )
        effects = result["effects"]
        self.assertAlmostEqual(effects["upstream_s1_residual_shift_b_minus_a"], 0.02)
        self.assertAlmostEqual(effects["s2_query_centroid_adaptation_c_minus_b"], -0.01)
        self.assertAlmostEqual(effects["s2_coupled_assignment_tax_d_minus_c"], 0.02)
        self.assertAlmostEqual(effects["total_d_minus_a"], effects["component_sum"])

    def test_factor_contrast_sign_is_canonical_minus_ablation(self) -> None:
        branches = {}
        for index, name in enumerate(BRANCH_ORDER):
            branches[name] = {
                "query_cosine_distortion": float(index + 1),
                "weighted_graph_agreement": float(index + 2),
                "fine_path_purity": float(index + 3),
                "distinct_prefix": 100 + index,
            }
        result = s2_factor_contrasts(branches)
        self.assertEqual(
            result["topk5_at_canonical"]["query_distortion_effect"], -1.0
        )
        self.assertEqual(
            result["graph_x_query_prior_interaction"][
                "query_distortion_interaction"
            ],
            -1.0,
        )


if __name__ == "__main__":
    unittest.main()
