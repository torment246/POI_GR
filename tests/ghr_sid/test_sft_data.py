"""Unit tests for deterministic GHR SFT identifier remapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.sft_data import (  # noqa: E402
    HISTORY_ID_PATTERN,
    TARGET_ID_PATTERN,
    _pack_tiger_codes,
    _row_from_match,
    build_ghr_special_tokens,
)


class GhrSftDataTest(unittest.TestCase):
    def test_tiger_key_pack_and_parse(self) -> None:
        content = "<POI_TIGER_ID><S1_10><S2_20><S3_30><C_4></POI_TIGER_ID>"
        match = HISTORY_ID_PATTERN.fullmatch(content)
        self.assertIsNotNone(match)
        key = _pack_tiger_codes((10, 20, 30, 4))
        self.assertEqual(_row_from_match(match, {key: 7}), 7)

    def test_target_pattern_is_anchored(self) -> None:
        content = "<TARGET_POI><S1_1><S2_2><S3_3><C_4></TARGET_POI>"
        self.assertIsNotNone(TARGET_ID_PATTERN.fullmatch(content))
        self.assertIsNone(TARGET_ID_PATTERN.fullmatch(content + "junk"))

    def test_special_tokens_share_gid_namespace(self) -> None:
        identifier_tokens = (
            "<S1_0>",
            "<S2_0>",
            "<S3_0>",
            "<G_0>",
            "<R_0>",
            "<V_0>",
            "<D>",
        )
        payload = build_ghr_special_tokens(identifier_tokens)
        tokens = payload["additional_special_tokens"]
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertEqual(tokens.count("<G_0>"), 1)
        self.assertIn("<POI_GHR_ID>", tokens)
        self.assertNotIn("<POI_TIGER_ID>", tokens)


if __name__ == "__main__":
    unittest.main()
