"""Unit tests for fixed five-layer aligned-collision SFT serialization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.aligned_collision_sft_data import (  # noqa: E402
    AlignedCollisionSftDataError,
    aligned_identifier_content,
    aligned_identifier_tokens,
)
from poi_gr.methods.ghr_sid.sft_data import build_ghr_special_tokens  # noqa: E402


class AlignedCollisionSftDataTest(unittest.TestCase):
    def test_identifier_serialization_is_fixed_and_layered(self) -> None:
        self.assertEqual(
            aligned_identifier_content((1, 2, 3, 4, 5)),
            "<S1_1><S2_2><S3_3><R1_4><R2_5>",
        )
        with self.assertRaises(AlignedCollisionSftDataError):
            aligned_identifier_content((1, 2, 3, 32, 0))

    def test_token_inventory_has_expected_capacity_and_no_collision_token(self) -> None:
        tokens = aligned_identifier_tokens()
        self.assertEqual(len(tokens), 3 * 1024 + 2 * 32)
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertIn("<R1_31>", tokens)
        self.assertIn("<R2_31>", tokens)
        self.assertNotIn("<C_0>", tokens)

    def test_complete_sft_inventory_is_deduplicated(self) -> None:
        payload = build_ghr_special_tokens(aligned_identifier_tokens())
        self.assertEqual(payload["token_count"], 5184)
        self.assertEqual(
            len(payload["additional_special_tokens"]),
            len(set(payload["additional_special_tokens"])),
        )


if __name__ == "__main__":
    unittest.main()
