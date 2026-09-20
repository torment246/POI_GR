from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qg_prqk.sid.nogid_config import load_nogid_codebook_config  # noqa: E402


class NoGIDCodebookConfigTest(unittest.TestCase):
    def test_loads_s1_s2_parent_and_three_token_sid(self) -> None:
        config = load_nogid_codebook_config(
            ROOT / "configs/qg_prqk_p7_nogid_s1s2_parent_hard60_topk5_direct_full_v1.yaml"
        )
        self.assertEqual(config.hard_graph["parent_key"], ["s1", "s2"])
        self.assertFalse(config.continuous_geo["gid_or_geohash_used"])
        self.assertEqual(config.final_sid["positions"], ["s1", "s2", "s3"])
        self.assertFalse(config.final_sid["gid_in_identifier"])
        self.assertEqual(config.hard_graph["score_threshold"], 0.50)

    def test_protocol_has_no_gid_or_geohash_input(self) -> None:
        config = load_nogid_codebook_config(
            ROOT / "configs/qg_prqk_p7_nogid_s1s2_parent_hard60_topk5_direct_full_v1.yaml"
        )
        resolved = config.resolved_payload()
        self.assertFalse(resolved["continuous_geo"]["gid_or_geohash_used"])
        self.assertNotIn("gid6", resolved["hard_graph"]["parent_key"])


if __name__ == "__main__":
    unittest.main()
