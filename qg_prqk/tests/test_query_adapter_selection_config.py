from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from qg_prqk.adapters.selection_config import (
    AdapterSelectionConfigError,
    load_adapter_selection_config,
)


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = QG_ROOT / "configs/qg_prqk_p3a_full_1024x3_v1.yaml"


class QueryAdapterSelectionConfigTest(unittest.TestCase):
    def test_canonical_config_freezes_approved_protocol(self) -> None:
        config = load_adapter_selection_config(CONFIG_PATH)
        self.assertEqual(config.d3_rows, 291_590)
        self.assertEqual(config.gate_exclusion_rows, 50_000)
        self.assertEqual(config.internal_holdout_rows, 20_000)
        self.assertEqual(config.max_epochs, 3)
        self.assertEqual(config.base.source_path, config.gate.source_path)
        self.assertEqual(
            dict(config.query_view_policy),
            {"D1": "raw_bge", "D2": "raw_bge", "D3": "final_adapter"},
        )

    def test_rejects_changed_max_epoch(self) -> None:
        raw = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
        raw["base_p3a_config"] = str(
            QG_ROOT / "configs/qg_prqk_p3a_1024x3_v1.yaml"
        )
        raw["p3a_full"]["max_epochs"] = 4
        with tempfile.TemporaryDirectory(dir=QG_ROOT / "configs") as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                yaml.safe_dump(raw, allow_unicode=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AdapterSelectionConfigError, "max_epochs"):
                load_adapter_selection_config(path)

    def test_accepts_separate_immutable_gate_config(self) -> None:
        raw = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
        gate_path = QG_ROOT / "configs/qg_prqk_p3a_1024x3_v1.yaml"
        raw["base_p3a_config"] = str(gate_path)
        raw["gate_p3a_config"] = str(gate_path)
        raw["gate_p3a_config_sha256"] = raw["base_p3a_config_sha256"]
        with tempfile.TemporaryDirectory(dir=QG_ROOT / "configs") as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                yaml.safe_dump(raw, allow_unicode=True),
                encoding="utf-8",
            )
            config = load_adapter_selection_config(path)
        self.assertEqual(config.base.source_path, config.gate.source_path)


if __name__ == "__main__":
    unittest.main()
