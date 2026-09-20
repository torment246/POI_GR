from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from qg_prqk.sid.local_config import LocalCodebookConfigError, load_local_codebook_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml"


class LocalCodebookConfigTest(unittest.TestCase):
    def test_loads_reviewed_sample_then_full_contract(self) -> None:
        config = load_local_codebook_config(CONFIG)
        self.assertEqual(config.gates["sample"]["d3_query_rows"], 832)
        self.assertEqual(config.gates["full"]["d3_query_rows"], 291_590)
        self.assertEqual(config.local_refinement, {"candidate_codes": 32, "sweeps": 3})
        self.assertTrue(str(config.output_dir).startswith(str(ROOT / "outputs")))

    def test_rejects_unreviewed_threshold(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        payload["hard_graph"]["score_threshold"] = 0.55
        temporary = ROOT / "configs/.test_p7_invalid.yaml"
        temporary.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        try:
            with self.assertRaises(LocalCodebookConfigError):
                load_local_codebook_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
