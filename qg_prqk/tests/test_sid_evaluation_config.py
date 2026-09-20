from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from qg_prqk.sid.evaluation_config import SidEvaluationConfigError, load_sid_evaluation_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml"


class SidEvaluationConfigTest(unittest.TestCase):
    def test_loads_only_a0_a4_and_stops_for_review(self) -> None:
        config = load_sid_evaluation_config(CONFIG)
        self.assertEqual(config.comparison["methods"], [
            "A0_POI_ONLY_PRQK_INITIALIZATION",
            "A4_QUERY_CATEGORY_GEO_FULL",
        ])
        self.assertEqual(config.gates["full"]["poi_rows"], 716_245)
        self.assertEqual(config.authorization["stop_after"], "HOLD_FOR_REVIEW")
        self.assertFalse(config.publication["force_unique_sid"])

    def test_rejects_external_baseline(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        payload["comparison"]["external_baseline"] = True
        temporary = ROOT / "configs/.test_p8_invalid.yaml"
        temporary.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        try:
            with self.assertRaises(SidEvaluationConfigError):
                load_sid_evaluation_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
