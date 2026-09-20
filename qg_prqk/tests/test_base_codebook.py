from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from qg_prqk.sid.base_data import (
    BaseCodebookAlgorithmConfig,
    TopKRefinementConfig,
    compute_global_mean,
    deterministic_selected_rows,
    rows_for_gate,
)
from qg_prqk.sid.base_diagnostics import (
    _run_branch,
    compare_diagnostic_branches,
    diagnostic_settings,
)
from qg_prqk.sid.base_quantizer import (
    _atomic_tensor_npy,
    fit_prqk_level,
    projection_residual,
    sid_metrics,
)
from qg_prqk.sid.base_topk_diagnostics import (
    compare_hard60_topk5,
    hard60_topk5_settings,
    refine_topk_from_hard_endpoint,
)


def settings() -> BaseCodebookAlgorithmConfig:
    return BaseCodebookAlgorithmConfig(
        metric="cosine",
        init="kmeans++",
        seed=42,
        remove_global_direction=True,
        residual="projection",
        min_iter=2,
        max_iter=8,
        objective_rel_tol=1.0e-4,
        assignment_change_tol=1.0e-3,
        patience=2,
        topk_refinement=TopKRefinementConfig(
            enabled=True, topk=2, beta=15.0, max_iter=2
        ),
    )


class BaseCodebookDataTest(unittest.TestCase):
    def test_gate_rows_and_seeded_sets_are_nested(self) -> None:
        self.assertEqual(rows_for_gate("sample", 716_245), 10_000)
        self.assertEqual(rows_for_gate("medium100k", 716_245), 100_000)
        self.assertEqual(rows_for_gate("medium500k", 716_245), 500_000)
        self.assertEqual(rows_for_gate("full", 716_245), 716_245)
        sample = deterministic_selected_rows(1000, 100, 42)
        medium = deterministic_selected_rows(1000, 500, 42)
        self.assertEqual(len(sample), 100)
        self.assertTrue(np.all(sample[:-1] < sample[1:]))
        self.assertTrue(set(sample).issubset(set(medium)))
        self.assertTrue(
            np.array_equal(sample, deterministic_selected_rows(1000, 100, 42))
        )

    def test_global_mean_normalizes_each_input_first(self) -> None:
        values = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        mean, metrics = compute_global_mean(values, chunk_rows=1)
        np.testing.assert_array_equal(mean, np.array([0.5, 0.5], dtype=np.float32))
        self.assertAlmostEqual(metrics["global_mean_norm"], np.sqrt(0.5), places=6)


class BaseCodebookQuantizerTest(unittest.TestCase):
    def test_tensor_artifact_is_written_as_atomic_chunked_npy(self) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "residual.npy"
            _atomic_tensor_npy(path, values, dtype=np.float16, chunk_rows=2)
            np.testing.assert_array_equal(
                np.load(path, allow_pickle=False), values.numpy().astype(np.float16)
            )
            self.assertEqual(list(path.parent.glob(".*.writing")), [])

    def test_projection_residual_is_orthogonal(self) -> None:
        values = torch.tensor(
            [[0.8, 0.6, 0.0], [0.2, 0.0, np.sqrt(0.96)]], dtype=torch.float32
        )
        centroids = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        assignments = torch.tensor([0, 1], dtype=torch.int64)
        residual, metrics = projection_residual(values, centroids, assignments)
        selected = centroids[assignments]
        self.assertLess(float(torch.abs((residual * selected).sum(1)).max()), 1e-6)
        self.assertEqual(metrics["zero_residual_rows"], 0)

    def test_spherical_fit_is_seed_deterministic_and_uses_soft_refinement(self) -> None:
        generator = torch.Generator().manual_seed(7)
        anchors = torch.eye(4, dtype=torch.float32)
        values = torch.cat(
            [
                anchor + 0.03 * torch.randn(32, 4, generator=generator)
                for anchor in anchors
            ]
        )
        values = torch.nn.functional.normalize(values, dim=1)
        first = fit_prqk_level(values, 4, settings(), level=1, chunk_rows=31)
        second = fit_prqk_level(values, 4, settings(), level=1, chunk_rows=31)
        np.testing.assert_array_equal(first[1].numpy(), second[1].numpy())
        np.testing.assert_allclose(first[0].numpy(), second[0].numpy(), atol=0, rtol=0)
        self.assertEqual(first[3]["soft_refinement_iterations"], 2)
        self.assertEqual(int(torch.unique(first[1]).numel()), 4)
        self.assertLess(first[3]["final_objective"], 0.02)

    def test_topk_diagnostic_refines_the_exact_hard_endpoint(self) -> None:
        generator = torch.Generator().manual_seed(17)
        values = torch.nn.functional.normalize(
            torch.randn(96, 8, generator=generator), dim=1
        )
        topk_settings = settings()
        hard_settings = replace(
            topk_settings,
            topk_refinement=replace(topk_settings.topk_refinement, enabled=False),
        )
        hard = fit_prqk_level(values, 4, hard_settings, level=1, chunk_rows=31)
        refined = refine_topk_from_hard_endpoint(
            values,
            hard[0],
            hard[1],
            hard[3],
            topk_settings,
            chunk_rows=31,
        )
        self.assertEqual(
            hard[3]["trace"],
            refined[3]["trace"][: hard[3]["hard_iterations"]],
        )
        self.assertEqual(refined[3]["soft_refinement_iterations"], 2)
        self.assertEqual(refined[1].shape, hard[1].shape)

    def test_sid_metrics_keep_collision_and_utilization_separate(self) -> None:
        assignments = np.array(
            [[0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int32
        )
        metrics = sid_metrics(assignments, 2)
        self.assertEqual(metrics["distinct_sid"], 3)
        self.assertEqual(metrics["collision_poi_rows"], 2)
        self.assertEqual(metrics["collision_excess"], 1)
        self.assertEqual(metrics["levels"][0]["active_codes"], 2)

    def test_diagnostic_settings_do_not_mutate_canonical_settings(self) -> None:
        canonical = settings()
        hard30 = diagnostic_settings(canonical, max_iter=30)
        hard60 = diagnostic_settings(canonical, max_iter=60)
        self.assertTrue(canonical.topk_refinement.enabled)
        self.assertFalse(hard30.topk_refinement.enabled)
        self.assertFalse(hard60.topk_refinement.enabled)
        self.assertEqual(hard30.max_iter, 30)
        self.assertEqual(hard60.max_iter, 60)
        with self.assertRaisesRegex(ValueError, "30 或 60"):
            diagnostic_settings(canonical, max_iter=31)

    def test_hard60_topk5_settings_preserve_canonical_refinement(self) -> None:
        canonical = replace(
            settings(),
            topk_refinement=TopKRefinementConfig(
                enabled=True,
                topk=5,
                beta=15.0,
                max_iter=5,
            ),
        )
        diagnostic = hard60_topk5_settings(canonical)
        self.assertEqual(canonical.max_iter, 8)
        self.assertEqual(diagnostic.max_iter, 60)
        self.assertEqual(
            diagnostic.topk_refinement,
            canonical.topk_refinement,
        )

    def test_diagnostic_comparison_reports_topk_and_iteration_evidence(self) -> None:
        def level(final: float, converged: bool, iterations: int) -> dict:
            return {
                "fit": {
                    "final_objective": final,
                    "hard_converged": converged,
                    "hard_iterations": iterations,
                    "trace": [
                        {
                            "stage": "hard_spherical_kmeans",
                            "objective": final - 0.01,
                        }
                    ],
                }
            }

        sid = sid_metrics(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32), 2)
        reference = {
            "level_metrics": [level(0.5, False, 30) for _ in range(3)],
            "sid": sid,
        }
        hard30 = {
            "level_metrics": [level(0.4, False, 30) for _ in range(3)],
            "sid": sid,
        }
        hard60 = {
            "level_metrics": [level(0.39, True, 40) for _ in range(3)],
            "sid": sid,
        }
        assignments = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        comparison = compare_diagnostic_branches(
            reference,
            assignments,
            hard30,
            assignments,
            hard60,
            assignments,
        )
        self.assertTrue(
            comparison["evidence"]["topk5_worse_than_hard30_off_all_levels"]
        )
        self.assertTrue(
            comparison["evidence"]["topk5_worse_than_its_own_hard30_trace_all_levels"]
        )
        self.assertEqual(
            comparison["evidence"]["max_iter_assessment"],
            "MAX_ITER_30_INSUFFICIENT_60_REACHES_FROZEN_STOP_RULE",
        )

    def test_topk_comparison_separates_direct_and_pipeline_evidence(self) -> None:
        hard_trace = [
            {
                "stage": "hard_spherical_kmeans",
                "iteration": 1,
                "objective": 0.40,
            }
        ]

        def level(
            final: float,
            *,
            ess: float = 2.0,
            gini: float = 0.2,
            soft_iterations: int = 0,
        ) -> dict:
            return {
                "fit": {
                    "final_objective": final,
                    "hard_converged": True,
                    "hard_iterations": 10,
                    "soft_refinement_iterations": soft_iterations,
                    "trace": hard_trace,
                },
                "assignment": {
                    "kish_ess": ess,
                    "gini": gini,
                },
            }

        reference_assignments = np.array(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]],
            dtype=np.int32,
        )
        all_off_assignments = np.array(
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
            dtype=np.int32,
        )
        local_assignments = all_off_assignments.copy()
        topk_assignments = reference_assignments.copy()
        reference = {
            "level_metrics": [level(0.50) for _ in range(3)],
            "sid": sid_metrics(reference_assignments, 3),
        }
        all_off = {
            "level_metrics": [level(0.40) for _ in range(3)],
            "sid": sid_metrics(all_off_assignments, 3),
        }
        local = {"level_metrics": [level(0.40, ess=2.0, gini=0.3) for _ in range(3)]}
        topk = {
            "level_metrics": [
                level(
                    0.41,
                    ess=2.5,
                    gini=0.2,
                    soft_iterations=5,
                )
                for _ in range(3)
            ],
            "sid": sid_metrics(topk_assignments, 3),
        }
        comparison = compare_hard60_topk5(
            reference,
            reference_assignments,
            all_off,
            all_off_assignments,
            local,
            local_assignments,
            topk,
            topk_assignments,
        )
        evidence = comparison["evidence"]
        self.assertTrue(evidence["hard_trace_matches_all_paired_runs"])
        self.assertTrue(evidence["topk5_worse_than_paired_pre_topk_all_levels"])
        self.assertTrue(evidence["topk5_improves_kish_ess_all_levels"])
        self.assertTrue(evidence["topk5_reduces_gini_all_levels"])
        self.assertTrue(evidence["topk5_increases_distinct_sid_vs_hard60_off"])
        self.assertFalse(evidence["canonical_change_applied"])

    def test_diagnostic_branch_writes_replayable_endpoint(self) -> None:
        generator = torch.Generator().manual_seed(11)
        values = torch.nn.functional.normalize(
            torch.randn(96, 8, generator=generator), dim=1
        )
        branch_settings = replace(
            settings(),
            max_iter=4,
            topk_refinement=replace(settings().topk_refinement, enabled=False),
        )
        config = SimpleNamespace(codebook_sizes=(4, 4, 4))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "hard"
            metrics = _run_branch(
                values, config, branch_settings, output, chunk_rows=31
            )
            assignments = np.load(
                output / "poi_assignments_s1_s2_s3.npy", allow_pickle=False
            )
            codebooks = np.load(
                output / "poi_codebooks_s1_s2_s3.npy", allow_pickle=False
            )
            self.assertEqual(assignments.shape, (96, 3))
            self.assertEqual(assignments.dtype, np.int32)
            self.assertEqual(codebooks.shape, (3, 4, 8))
            self.assertEqual(codebooks.dtype, np.float32)
            self.assertEqual(metrics["settings"]["max_iter"], 4)
            self.assertEqual(metrics["sid"]["levels"][0]["active_codes"], 4)


if __name__ == "__main__":
    unittest.main()
