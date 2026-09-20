"""Synthetic tests for dynamic SID materialization and joint updates."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint.data import (  # noqa: E402
    DynamicSidTemplate,
    JointTigerBatch,
    SidTokenLayout,
    TigerJointDataError,
)
from poi_gr.methods.tiger_joint.training import (  # noqa: E402
    HuggingFaceCausalLMJointAdapter,
    JointTigerTrainingModule,
    compute_joint_tiger_loss,
    multi_positive_info_nce,
    teacher_forced_accuracy,
)
from poi_gr.sid.rqvae import RQVAE  # noqa: E402


class _TinyCausalBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        if use_cache or not return_dict:
            raise AssertionError(
                "joint adapter must disable cache and request dict output"
            )
        token_hidden = self.embedding(input_ids)
        mask = attention_mask.to(token_hidden.dtype).unsqueeze(-1)
        cumulative_hidden = (token_hidden * mask).cumsum(dim=1)
        cumulative_count = mask.cumsum(dim=1).clamp_min(1.0)
        return SimpleNamespace(last_hidden_state=cumulative_hidden / cumulative_count)


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = _TinyCausalBackbone(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


def _build_synthetic_batch() -> JointTigerBatch:
    input_ids = torch.tensor(
        [
            [4, 2, 2, 2, 5, 8, 14, 2, 2, 2, 15, 0],
            [4, 2, 2, 2, 5, 8, 14, 2, 2, 2, 15, 0],
            [4, 6, 6, 6, 5, 8, 14, 2, 2, 2, 15, 0],
        ],
        dtype=torch.long,
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, 6] = 14
    labels[:, 10] = 15
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 11] = 0
    template = DynamicSidTemplate(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        query_state_positions=torch.tensor([5, 5, 5]),
        history_sample_indices=torch.tensor([0, 1]),
        history_poi_rows=torch.tensor([101, 102]),
        history_sid_positions=torch.tensor([[1, 2, 3], [1, 2, 3]]),
        target_poi_rows=torch.tensor([100, 101, 100]),
        target_sid_positions=torch.tensor([[7, 8, 9], [7, 8, 9], [7, 8, 9]]),
    )

    unique_targets = torch.randn(2, 6)
    target_embeddings = torch.stack(
        (unique_targets[0], unique_targets[1], unique_targets[0])
    )
    catalog_embeddings = torch.cat(
        (unique_targets, torch.randn(2, 6)),
        dim=0,
    )
    return JointTigerBatch(
        template=template,
        history_embeddings=torch.randn(2, 6),
        target_embeddings=target_embeddings,
        catalog_poi_rows=torch.tensor([100, 101, 102, 103]),
        catalog_embeddings=catalog_embeddings,
    )


class TigerJointTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_one_step_updates_generator_projection_and_fresh_rqvae(self) -> None:
        batch = _build_synthetic_batch()
        token_layout = SidTokenLayout(
            (
                tuple(range(20, 24)),
                tuple(range(24, 28)),
                tuple(range(28, 32)),
            )
        )
        rqvae = RQVAE(
            input_dim=6,
            hidden_dim=5,
            latent_dim=4,
            codebook_sizes=(4, 4, 4),
        )
        generator = HuggingFaceCausalLMJointAdapter(
            _TinyCausalLM(vocab_size=40, hidden_size=8)
        )
        query_projection = nn.Linear(8, 4)
        optimizer = torch.optim.Adam(
            [
                *generator.parameters(),
                *query_projection.parameters(),
                *rqvae.parameters(),
            ],
            lr=1e-2,
        )

        with torch.no_grad():
            expected_history_codes = rqvae.encode_codes(batch.history_embeddings)
            expected_target_codes = rqvae.encode_codes(batch.target_embeddings)
        generator_before = generator.causal_lm.model.embedding.weight.detach().clone()
        projection_before = query_projection.weight.detach().clone()
        encoder_before = rqvae.encoder[0].weight.detach().clone()
        codebook_before = rqvae.quantizer.codebooks[0].weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        training_module = JointTigerTrainingModule(
            generator=generator,
            rqvae=rqvae,
            query_projection=query_projection,
            token_layout=token_layout,
            alignment_weight=0.25,
            rq_weight=0.5,
            temperature=0.2,
        )
        output = training_module(batch)

        self.assertTrue(torch.isfinite(output.total_loss))
        self.assertTrue(torch.isfinite(output.generation_loss))
        self.assertTrue(torch.isfinite(output.alignment_loss))
        self.assertTrue(torch.isfinite(output.rq_loss))
        self.assertGreaterEqual(float(output.teacher_forced_token_accuracy), 0.0)
        self.assertLessEqual(float(output.teacher_forced_token_accuracy), 1.0)
        self.assertGreaterEqual(float(output.teacher_forced_exact_match), 0.0)
        self.assertLessEqual(float(output.teacher_forced_exact_match), 1.0)
        self.assertGreaterEqual(float(output.teacher_forced_sid_token_accuracy), 0.0)
        self.assertLessEqual(float(output.teacher_forced_sid_token_accuracy), 1.0)
        self.assertGreaterEqual(float(output.teacher_forced_static_token_accuracy), 0.0)
        self.assertLessEqual(float(output.teacher_forced_static_token_accuracy), 1.0)
        torch.testing.assert_close(output.history_codes, expected_history_codes)
        torch.testing.assert_close(output.target_codes, expected_target_codes)
        torch.testing.assert_close(
            output.total_loss,
            output.generation_loss
            + 0.25 * output.alignment_loss
            + 0.5 * output.rq_loss,
        )

        target_token_ids = token_layout.tokens_for_codes(output.target_codes)
        target_indices = torch.arange(3).unsqueeze(1)
        torch.testing.assert_close(
            output.materialized.input_ids[
                target_indices, batch.template.target_sid_positions
            ],
            target_token_ids,
        )
        torch.testing.assert_close(
            output.materialized.labels[
                target_indices, batch.template.target_sid_positions
            ],
            target_token_ids,
        )
        history_token_ids = token_layout.tokens_for_codes(output.history_codes)
        torch.testing.assert_close(
            output.materialized.input_ids[
                batch.template.history_sample_indices.unsqueeze(1),
                batch.template.history_sid_positions,
            ],
            history_token_ids,
        )
        self.assertTrue(
            torch.equal(
                output.materialized.labels[:, 6],
                torch.full((3,), 14, dtype=torch.long),
            )
        )

        output.total_loss.backward()
        for parameter in (
            generator.causal_lm.model.embedding.weight,
            query_projection.weight,
            rqvae.encoder[0].weight,
            rqvae.quantizer.codebooks[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        optimizer.step()

        self.assertFalse(
            torch.equal(
                generator_before,
                generator.causal_lm.model.embedding.weight.detach(),
            )
        )
        self.assertFalse(
            torch.equal(projection_before, query_projection.weight.detach())
        )
        self.assertFalse(torch.equal(encoder_before, rqvae.encoder[0].weight))
        self.assertFalse(
            torch.equal(
                codebook_before,
                rqvae.quantizer.codebooks[0].weight.detach(),
            )
        )

    def test_teacher_forced_accuracy_reports_token_and_exact_match(self) -> None:
        labels = torch.tensor(
            [
                [-100, 2, 3, -100],
                [-100, 2, 4, -100],
            ],
            dtype=torch.long,
        )
        logits = torch.zeros((2, 4, 6), dtype=torch.float32)
        logits[0, 0, 2] = 5.0
        logits[0, 1, 3] = 5.0
        logits[1, 0, 2] = 5.0
        logits[1, 1, 1] = 5.0

        token_accuracy, exact_match = teacher_forced_accuracy(logits, labels)

        self.assertAlmostEqual(float(token_accuracy), 0.75)
        self.assertAlmostEqual(float(exact_match), 0.5)

    def test_recomputes_sid_tokens_after_codebook_index_change(self) -> None:
        batch = _build_synthetic_batch()
        token_layout = SidTokenLayout(
            (
                tuple(range(20, 24)),
                tuple(range(24, 28)),
                tuple(range(28, 32)),
            )
        )
        rqvae = RQVAE(
            input_dim=6,
            hidden_dim=5,
            latent_dim=4,
            codebook_sizes=(4, 4, 4),
        )
        generator = HuggingFaceCausalLMJointAdapter(
            _TinyCausalLM(vocab_size=40, hidden_size=8)
        )
        query_projection = nn.Linear(8, 4)

        first = compute_joint_tiger_loss(
            generator=generator,
            rqvae=rqvae,
            query_projection=query_projection,
            batch=batch,
            token_layout=token_layout,
            alignment_weight=0.25,
            rq_weight=0.5,
            temperature=0.2,
        )
        with torch.no_grad():
            for codebook in rqvae.quantizer.codebooks:
                codebook.weight.copy_(codebook.weight.roll(shifts=1, dims=0))
            expected_codes = rqvae.encode_codes(batch.target_embeddings)
        second = compute_joint_tiger_loss(
            generator=generator,
            rqvae=rqvae,
            query_projection=query_projection,
            batch=batch,
            token_layout=token_layout,
            alignment_weight=0.25,
            rq_weight=0.5,
            temperature=0.2,
        )

        self.assertFalse(torch.equal(first.target_codes, second.target_codes))
        torch.testing.assert_close(second.target_codes, expected_codes)
        expected_tokens = token_layout.tokens_for_codes(expected_codes)
        sample_indices = torch.arange(expected_codes.shape[0]).unsqueeze(1)
        torch.testing.assert_close(
            second.materialized.labels[
                sample_indices, batch.template.target_sid_positions
            ],
            expected_tokens,
        )

    def test_multi_positive_loss_does_not_treat_duplicate_poi_as_negative(
        self,
    ) -> None:
        queries = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        items = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        loss = multi_positive_info_nce(
            queries,
            items,
            torch.tensor([7, 7, 8]),
            temperature=0.1,
        )

        self.assertLess(float(loss.detach()), 1e-3)
        loss.backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(items.grad).all())

    def test_template_rejects_query_and_sid_slot_overlap(self) -> None:
        template = _build_synthetic_batch().template
        invalid_positions = template.target_sid_positions.clone()
        invalid_positions[0, 0] = template.query_state_positions[0]

        with self.assertRaisesRegex(TigerJointDataError, "query state"):
            replace(template, target_sid_positions=invalid_positions)


if __name__ == "__main__":
    unittest.main()
