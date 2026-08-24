"""Lightweight tests for the MMBERT recall encoder adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.mmbert_recall import MMBertRecallEncoder  # noqa: E402


class FakeTokenizer:
    def __call__(self, sentences: list[str], **_: object) -> dict[str, torch.Tensor]:
        max_length = max(len(text) for text in sentences)
        input_ids = []
        attention_mask = []
        for text in sentences:
            values = list(range(1, len(text) + 1))
            padding = max_length - len(values)
            input_ids.append(values + [0] * padding)
            attention_mask.append([1] * len(values) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class FakeBaseEncoder(torch.nn.Module):
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> object:
        del attention_mask
        hidden = torch.stack(
            (input_ids.float(), input_ids.float() * 2), dim=-1
        )
        return SimpleNamespace(last_hidden_state=hidden)


class MMBertRecallEncoderTest(unittest.TestCase):
    def test_mean_pool_projection_and_normalization(self) -> None:
        projection = torch.nn.Linear(2, 128, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[0, 0] = 1
            projection.weight[1, 1] = 1
        encoder = MMBertRecallEncoder(
            tokenizer=FakeTokenizer(),
            encoder=FakeBaseEncoder(),
            projection=projection,
            device="cpu",
            max_seq_length=8,
        )

        vectors = encoder.encode(["a", "abc"], batch_size=2)

        self.assertEqual(vectors.shape, (2, 128))
        np.testing.assert_allclose(
            np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6
        )
        np.testing.assert_allclose(vectors[0, :2], vectors[1, :2], atol=1e-6)
        np.testing.assert_array_equal(vectors[:, 2:], np.zeros((2, 126)))


if __name__ == "__main__":
    unittest.main()
