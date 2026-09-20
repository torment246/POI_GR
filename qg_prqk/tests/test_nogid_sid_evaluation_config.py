from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qg_prqk.sid.nogid_evaluation_config import load_nogid_evaluation_config  # noqa: E402


class NoGIDSidEvaluationConfigTest(unittest.TestCase):
    def test_loads_three_token_s1_s2_parent_protocol(self) -> None:
        config = load_nogid_evaluation_config(
            ROOT / "configs/qg_prqk_p8_nogid_s1s2_parent_comparison_full_v1.yaml"
        )
        self.assertEqual(config.comparison["final_sid_positions"], ["s1", "s2", "s3"])
        self.assertEqual(config.comparison["s3_candidate_parent"], ["s1", "s2"])
        self.assertFalse(config.publication["gid_artifact"])
        self.assertFalse(config.comparison["external_baseline"])

    def test_frozen_nogid_manifest_declares_no_gid(self) -> None:
        config = load_nogid_evaluation_config(
            ROOT / "configs/qg_prqk_p8_nogid_s1s2_parent_comparison_full_v1.yaml"
        )
        summary = config.resolved_payload()
        self.assertEqual(summary["publication"]["sid_source"], "A4_NOGID_S1S2_PARENT")
        self.assertEqual(summary["authorization"]["stop_after"], "HOLD_FOR_NOGID_S3_REVIEW")


if __name__ == "__main__":
    unittest.main()
