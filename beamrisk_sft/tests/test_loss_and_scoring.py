from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from beamrisk_sft.loss import beam_risk_loss_from_scores, score_and_compute_beam_risk_loss
from beamrisk_sft.scoring import collate_completion_paths


class TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-0.3, 0.3, vocab_size))

    def forward(self, input_ids, logits_to_keep, **kwargs):
        batch = input_ids.shape[0]
        positions = logits_to_keep.numel()
        logits = self.bias.view(1, 1, -1).expand(batch, positions, -1)
        return SimpleNamespace(logits=logits)


def test_pairwise_gradient_increases_positive_and_decreases_negative() -> None:
    positive = torch.tensor([-3.0, -1.0], requires_grad=True)
    negative = torch.tensor([-2.0, -2.0], requires_grad=True)
    output = beam_risk_loss_from_scores(positive, negative, temperature=1.0)
    output.loss.backward()
    assert torch.all(positive.grad < 0)
    assert torch.all(negative.grad > 0)


def test_left_padding_aligns_variable_completion_ends() -> None:
    values = collate_completion_paths(
        prompts=((1, 2, 3), (4,)),
        completions=((5, 6), (7,)),
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    input_ids, attention, positions, targets, mask, logit_positions = values
    assert input_ids.tolist() == [[1, 2, 3, 5, 6], [0, 0, 0, 4, 7]]
    assert attention.tolist() == [[1, 1, 1, 1, 1], [0, 0, 0, 1, 1]]
    assert positions.tolist() == [[0, 1, 2, 3, 4], [0, 0, 0, 0, 1]]
    assert targets.tolist() == [[5, 6], [0, 7]]
    assert mask.tolist() == [[True, True], [False, True]]
    assert logit_positions.tolist() == [2, 3]


def test_scored_risk_loss_backpropagates_through_model() -> None:
    model = TinyCausalModel()
    result = score_and_compute_beam_risk_loss(
        model,
        prompts=((1, 2), (3,)),
        positive_paths=((4, 5), (6,)),
        negative_paths=((7, 8), (9,)),
        pad_token_id=0,
        temperature=1.0,
        margin=0.0,
        device=torch.device("cpu"),
    )
    result.loss.backward()
    assert model.bias.grad is not None
    assert torch.isfinite(model.bias.grad).all()


def test_scored_risk_loss_uses_real_qwen3_logits_contract() -> None:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        use_sliding_window=False,
    )
    model = Qwen3ForCausalLM(config)
    result = score_and_compute_beam_risk_loss(
        model,
        prompts=((1, 2, 3), (4, 5)),
        positive_paths=((6, 7), (8,)),
        negative_paths=((9, 10), (11,)),
        pad_token_id=0,
        temperature=1.0,
        margin=0.0,
        device=torch.device("cpu"),
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
