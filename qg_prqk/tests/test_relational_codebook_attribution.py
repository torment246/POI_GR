from __future__ import annotations

import copy
import unittest

import numpy as np

from qg_prqk.sid.relational_attribution import (
    BRANCH_ORDER,
    NEAR_ZERO_QUERY_PRIOR,
    build_diagnostic_branches,
    factor_contrasts,
    query_prior_metrics,
)


class RelationalCodebookAttributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.s1 = {
            "query_distortion_weight": 0.05,
            "graph_alignment_weight": 0.05,
            "category_weight": 0.08,
            "category_smoothing": 32,
            "query_centroid_shrinkage": 32,
        }
        self.s2 = {**self.s1, "query_distortion_weight": 0.1, "graph_alignment_weight": 0.1}
        self.prqk = {
            "seed": 42,
            "topk_refinement": {"enabled": True, "topk": 5, "beta": 15, "max_iter": 5},
        }

    def test_fixed_branches_change_only_declared_factors(self) -> None:
        before = copy.deepcopy((self.s1, self.s2, self.prqk))
        branches = build_diagnostic_branches(self.s1, self.s2, self.prqk)
        self.assertEqual(tuple(branches), BRANCH_ORDER)
        self.assertEqual((self.s1, self.s2, self.prqk), before)
        self.assertFalse(branches["topk_off"]["prqk"]["topk_refinement"]["enabled"])
        self.assertEqual(branches["graph_off"]["s1"]["graph_alignment_weight"], 0.0)
        self.assertEqual(
            branches["near_zero_query_prior"]["s2"]["query_centroid_shrinkage"],
            NEAR_ZERO_QUERY_PRIOR,
        )
        self.assertEqual(branches["category_off"]["s1"]["category_weight"], 0.0)
        self.assertEqual(branches["canonical"]["s1"], self.s1)

    def test_prior_mass_exposes_tau_dominance_for_sparse_query_codes(self) -> None:
        assignments = np.array([0, 0, 1, 2], dtype=np.int32)
        strong = query_prior_metrics(assignments, codebook_size=4, tau=32.0)
        weak = query_prior_metrics(
            assignments, codebook_size=4, tau=NEAR_ZERO_QUERY_PRIOR
        )
        self.assertGreater(strong["poi_prior_mass_query_weighted_mean"], 0.94)
        self.assertLess(weak["poi_prior_mass_query_weighted_mean"], 1.0e-5)

    def test_factor_contrasts_have_canonical_minus_ablation_sign(self) -> None:
        query_distortions = {
            "canonical": 4.0,
            "topk_off": 5.0,
            "graph_off": 2.0,
            "near_zero_query_prior": 3.0,
            "graph_off_near_zero_query_prior": 1.0,
            "category_off": 6.0,
        }
        branches = {}
        for index, name in enumerate(BRANCH_ORDER):
            branches[name] = {
                "levels": [
                    {
                        "query_cosine_distortion": query_distortions[name]
                        + level,
                        "weighted_graph_agreement": float(10 + index + level),
                        "category_purity": float(20 + index + level),
                    }
                    for level in (1, 2)
                ]
            }
        result = factor_contrasts(branches)
        self.assertEqual(
            result["topk5_at_canonical"]["levels"][0]["query_distortion_effect"],
            -1.0,
        )
        self.assertEqual(
            result["graph_x_query_prior_interaction"]["levels"][0][
                "query_distortion_interaction"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
