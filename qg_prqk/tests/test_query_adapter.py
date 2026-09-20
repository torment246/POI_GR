from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from qg_prqk.adapters.model import (
    ResidualQueryAdapter,
    adapter_gate_passes,
    build_in_batch_valid_mask,
    build_negative_valid_mask,
    stable_train_dev_split,
    weighted_info_nce,
)


class QueryAdapterTest(unittest.TestCase):
    def test_adapter_shape_norm_and_identity_initialization(self) -> None:
        torch.manual_seed(3)
        raw = torch.randn(5, 8)
        adapter = ResidualQueryAdapter(8, 4, dropout=0.0)
        adapted = adapter(raw)
        self.assertEqual(adapted.shape, raw.shape)
        self.assertTrue(torch.allclose(adapted.norm(dim=1), torch.ones(5), atol=1e-6))
        self.assertTrue(torch.allclose(adapted, F.normalize(raw, dim=1), atol=1e-6))

    def test_false_negative_mask_removes_all_reasonable_positives(self) -> None:
        candidates = torch.tensor([[5, 6, 7, -1], [8, 9, 10, 11]])
        targets = torch.tensor([5, 8])
        mask = build_negative_valid_mask(candidates, targets, [{7}, {9, 11}])
        self.assertEqual(
            mask.tolist(),
            [[False, True, False, False], [False, False, True, False]],
        )

    def test_better_positive_alignment_has_lower_weighted_info_nce(self) -> None:
        positives = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
        negatives = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
        valid = torch.ones((2, 1), dtype=torch.bool)
        weights = torch.tensor([1.0, 2.0])
        good = weighted_info_nce(
            positives,
            positives,
            negatives,
            valid,
            weights,
            temperature=0.1,
        )
        bad = weighted_info_nce(
            torch.flip(positives, dims=[0]),
            positives,
            negatives,
            valid,
            weights,
            temperature=0.1,
        )
        self.assertLess(float(good), float(bad))

    def test_in_batch_false_negative_mask_changes_only_allowed_denominator(self) -> None:
        positives = F.normalize(torch.eye(3), dim=1)
        fixed_negatives = positives[torch.tensor([[1], [2], [0]])]
        fixed_valid = torch.ones((3, 1), dtype=torch.bool)
        weights = torch.ones(3)
        targets = torch.tensor([10, 11, 12])
        in_batch_mask = build_in_batch_valid_mask(
            targets,
            [{11}, set(), set()],
        )
        self.assertEqual(
            in_batch_mask.tolist(),
            [[False, False, True], [True, False, True], [True, True, False]],
        )
        loss = weighted_info_nce(
            positives,
            positives,
            fixed_negatives,
            fixed_valid,
            weights,
            temperature=0.1,
            in_batch_positive_embeddings=positives,
            in_batch_valid_mask=in_batch_mask,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_train_dev_split_is_deterministic_and_disjoint(self) -> None:
        first = stable_train_dev_split(list(range(100)), dev_fraction=0.05, seed=42)
        second = stable_train_dev_split(list(range(100)), dev_fraction=0.05, seed=42)
        self.assertEqual(first, second)
        train, dev = first
        self.assertEqual(len(dev), 5)
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(sorted(train + dev), list(range(100)))

    def test_gate_requires_both_overall_and_difficult_improvement(self) -> None:
        raw = {"recall_at_10": 0.5, "mean_hard_margin": -0.1}
        improved = {"recall_at_10": 0.6, "mean_hard_margin": 0.0}
        regressed = {"recall_at_10": 0.4, "mean_hard_margin": 0.0}
        self.assertTrue(adapter_gate_passes(raw, improved, raw, improved))
        self.assertFalse(adapter_gate_passes(raw, improved, raw, regressed))


if __name__ == "__main__":
    unittest.main()
