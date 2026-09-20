from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from qg_prqk.sid.base_config import BaseCodebookConfigError, load_base_codebook_config
from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.base_data import (
    BaseCodebookDataError,
    gate_next_status,
    load_base_codebook_algorithm,
    output_directory,
)
from qg_prqk.sid.base_quantizer import _check_previous_gate


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG = QG_ROOT / "configs/qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml"


class BaseCodebookConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_selected_protocol_overlays_only_reviewed_p5_settings(self) -> None:
        config = load_base_codebook_config(CONFIG)
        settings = load_base_codebook_algorithm(config)
        base_prqk = config.base.resolved_payload()["prqk"]
        resolved_prqk = config.resolved_payload()["prqk"]
        self.assertEqual(settings.max_iter, 60)
        self.assertTrue(settings.topk_refinement.enabled)
        self.assertEqual(settings.topk_refinement.topk, 5)
        self.assertEqual(settings.topk_refinement.beta, 15.0)
        self.assertEqual(settings.topk_refinement.max_iter, 5)
        self.assertEqual(base_prqk["max_iter"], 30)
        for name, value in base_prqk.items():
            if name != "max_iter":
                self.assertEqual(resolved_prqk[name], value, name)
        self.assertNotEqual(config.signature(), config.base.signature())

    def test_output_namespace_and_review_state_are_isolated(self) -> None:
        config = load_base_codebook_config(CONFIG)
        output = output_directory(config, "sample", 10_000)
        self.assertEqual(output.parent.name, "poi_prqk_a0_hard60_topk5_v1")
        self.assertEqual(
            gate_next_status(config, "sample"),
            "HOLD_FOR_P5_HARD60_TOPK5_SAMPLE_REVIEW",
        )
        self.assertNotEqual(
            output,
            config.output_dir / "poi_prqk_a0/sample_010000",
        )

    def test_rejects_unreviewed_parameter_or_evidence_path(self) -> None:
        changed_iter = copy.deepcopy(self.raw)
        changed_iter["prqk"]["max_iter"] = 30
        changed_path = copy.deepcopy(self.raw)
        changed_path["selection_evidence"]["medium100k_manifest"]["path"] = (
            "qg_prqk/outputs/other/manifest.json"
        )
        for payload in (changed_iter, changed_path):
            with self.subTest(payload=payload):
                with self.assertRaises(BaseCodebookConfigError):
                    load_base_codebook_config(CONFIG, payload=payload)

    def test_direct_full_requires_and_records_reviewed_sample(self) -> None:
        config = load_base_codebook_config(CONFIG)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest = {
                "schema_version": "qg-prqk-p5-poi-prqk-a0-v1",
                "status": "completed",
                "phase": "P5-CAT-SAMPLE",
                "contract": {"config_signature": config.signature()},
                "next_status": gate_next_status(config, "sample"),
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            manifest_sha256 = sha256_file(manifest_path)
            (root / "_SUCCESS").write_text(
                json.dumps({"manifest_sha256": manifest_sha256}), encoding="utf-8"
            )
            previous = _check_previous_gate(
                config,
                "full",
                manifest_path,
                manifest_sha256,
                direct_full_from_sample_user_authorized=True,
            )
            self.assertEqual(previous["gate"], "sample")
            self.assertEqual(
                previous["gate_sequence_override"]["skipped_gates"],
                ["medium100k", "medium500k"],
            )
            with self.assertRaisesRegex(BaseCodebookDataError, "只允许用于 full"):
                _check_previous_gate(
                    config,
                    "medium100k",
                    manifest_path,
                    manifest_sha256,
                    direct_full_from_sample_user_authorized=True,
                )


if __name__ == "__main__":
    unittest.main()
