"""Tests for the early-only QGR-SID reliability gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.relations import RELATION_TYPES  # noqa: E402
from poi_gr.methods.qgr_sid.reliable_proxy import (  # noqa: E402
    select_reliable_relation_types,
)


class ReliableRelationGateTest(unittest.TestCase):
    def test_gate_requires_both_support_and_predictability(self) -> None:
        per_relation = {
            name: {
                "early_support_order_count": 0,
                "early_value_aware_match_ratio": 0.0,
            }
            for name in RELATION_TYPES
        }
        per_relation["R_BUILDING"] = {
            "early_support_order_count": 10_000,
            "early_value_aware_match_ratio": 0.15,
        }
        per_relation["R_UNIT"] = {
            "early_support_order_count": 9_999,
            "early_value_aware_match_ratio": 0.9,
        }
        per_relation["R_FLOOR"] = {
            "early_support_order_count": 100_000,
            "early_value_aware_match_ratio": 0.149,
        }
        selected = select_reliable_relation_types(
            early_metrics={"per_relation": per_relation},
            min_support_orders=10_000,
            min_value_match_ratio=0.15,
        )
        self.assertEqual(selected, ("R_BUILDING",))


if __name__ == "__main__":
    unittest.main()
