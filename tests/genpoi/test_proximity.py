"""Synthetic tests for the GenPOI proximity estimator contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.genpoi.proximity import (  # noqa: E402
    GenPoiProximityError,
    effective_prefix_length,
    extract_proximity_example,
    prefilled_gid_tokens,
)


class GenPoiProximityTest(unittest.TestCase):
    def record(self) -> dict[str, object]:
        return {
            "sample_id": "sample-a",
            "split": "valid",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "<HISTORY>\n"
                        "<USER_GID><G_x><G_x><G_x><G_x><G_x><G_x></USER_GID>\n"
                        "<QUERY>历史查询</QUERY>\n"
                        "</HISTORY>\n"
                        "<CURRENT>\n"
                        "<USER_GID><G_w><G_x><G_4><G_d><G_z><G_s></USER_GID>\n"
                        "<QUERY>北京西站</QUERY>\n"
                        "</CURRENT>"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "<G_w><G_x><G_4><G_d><G_y><G_y>"
                        "<S1_1><S2_2><S3_3>"
                    ),
                },
            ],
        }

    def test_label_uses_current_query_and_target_gid(self) -> None:
        example = extract_proximity_example(self.record(), expected_split="valid")
        self.assertEqual(example.query, "北京西站")
        self.assertEqual(example.user_gid, ("w", "x", "4", "d", "z", "s"))
        self.assertEqual(example.target_gid, ("w", "x", "4", "d", "y", "y"))
        self.assertEqual(example.label, 4)

    def test_gamma_two_prefills_only_relaxed_prefix(self) -> None:
        self.assertEqual(effective_prefix_length(0), 0)
        self.assertEqual(effective_prefix_length(2), 0)
        self.assertEqual(effective_prefix_length(6), 4)
        self.assertEqual(
            prefilled_gid_tokens(("w", "x", "4", "d", "z", "s"), 5),
            ("<G_w>", "<G_x>", "<G_4>"),
        )

    def test_invalid_gid_is_rejected(self) -> None:
        record = self.record()
        record["messages"][1]["content"] = "<G_w><G_x><S1_1><S2_2><S3_3>"  # type: ignore[index]
        with self.assertRaisesRegex(GenPoiProximityError, "6 个 Token"):
            extract_proximity_example(record, expected_split="valid")


if __name__ == "__main__":
    unittest.main()
