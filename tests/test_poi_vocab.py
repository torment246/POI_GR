"""Tests for deterministic POI/PID tokenizer extension."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from scripts.prepare_poi_vocab import (
    EXPECTED_TOKEN_COUNT,
    PoiVocabError,
    load_requested_tokens,
    verify_atomic_tokens,
)


class PoiVocabTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_tokens(self, tokens: list[str], token_count: int | None = None) -> Path:
        path = self.root / "special_tokens.json"
        path.write_text(
            json.dumps(
                {
                    "additional_special_tokens": tokens,
                    "token_count": len(tokens) if token_count is None else token_count,
                }
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def tokenizer() -> PreTrainedTokenizerFast:
        backend = Tokenizer(WordLevel({"[UNK]": 0, "base": 1}, unk_token="[UNK]"))
        backend.pre_tokenizer = WhitespaceSplit()
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

    def test_all_requested_tokens_are_ordinary_atomic_and_stable(self) -> None:
        tokens = [f"<T_{index}>" for index in range(EXPECTED_TOKEN_COUNT)]
        path = self.write_tokens(tokens)
        self.assertEqual(load_requested_tokens(path), tokens)
        tokenizer = self.tokenizer()
        self.assertEqual(tokenizer.add_tokens(tokens, special_tokens=False), len(tokens))
        mapping = {
            token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens
        }
        verify_atomic_tokens(tokenizer, tokens, mapping)
        saved = self.root / "saved"
        tokenizer.save_pretrained(saved)
        reloaded = PreTrainedTokenizerFast.from_pretrained(saved)
        reloaded_mapping = {
            token: int(reloaded.convert_tokens_to_ids(token)) for token in tokens
        }
        self.assertEqual(mapping, reloaded_mapping)
        verify_atomic_tokens(reloaded, tokens, reloaded_mapping)

    def test_rejects_wrong_count_duplicate_and_d_minus_one(self) -> None:
        with self.assertRaises(PoiVocabError):
            load_requested_tokens(self.write_tokens(["<A>"], token_count=1))
        tokens = [f"<T_{index}>" for index in range(EXPECTED_TOKEN_COUNT - 1)]
        tokens.append(tokens[-1])
        with self.assertRaises(PoiVocabError):
            load_requested_tokens(self.write_tokens(tokens))
        tokens[-1] = "<D_-1>"
        with self.assertRaises(PoiVocabError):
            load_requested_tokens(self.write_tokens(tokens))

    def test_accepts_explicit_tiger_token_count(self) -> None:
        tokens = [f"<TIGER_{index}>" for index in range(12)]
        path = self.write_tokens(tokens)
        self.assertEqual(
            load_requested_tokens(path, expected_token_count=12),
            tokens,
        )
        with self.assertRaises(PoiVocabError):
            load_requested_tokens(path, expected_token_count=11)


if __name__ == "__main__":
    unittest.main()
