from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from qg_prqk.sid.relational_config import (
    DIRECT_FULL_AUTHORIZATION_ID,
    DIRECT_FULL_SCHEMA_VERSION,
    RelationalConfigError,
    load_relational_codebook_config,
    validate_relational_frozen_inputs,
)
from qg_prqk.sid.relational_quantizer import next_status_for_gate, output_directory


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG = QG_ROOT / "configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml"
DIRECT_FULL_CONFIG = (
    QG_ROOT
    / "configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml"
)


class RelationalCodebookConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_overlay_changes_only_hard_iteration_limit(self) -> None:
        config = load_relational_codebook_config(CONFIG)
        base = config.base.resolved_payload()
        resolved = config.resolved_payload()
        self.assertEqual(base["prqk"]["max_iter"], 30)
        self.assertEqual(resolved["prqk"]["max_iter"], 60)
        for key, value in base["prqk"].items():
            if key != "max_iter":
                self.assertEqual(resolved["prqk"][key], value, key)
        for section in ("s1", "s2", "s3", "warmup", "category", "query_depth"):
            self.assertEqual(resolved[section], base[section], section)
        self.assertEqual(config.codebook_sizes, (512, 512, 512))
        self.assertEqual(config.p6_gates["sample"]["base_poi_rows"], 10_000)
        self.assertEqual(config.p6_gates["sample"]["query_rows"], 1_000)
        self.assertEqual(config.p6_gates["sample"]["expected_closed_poi_rows"], 11_459)
        self.assertEqual(config.p6_gates["medium"]["expected_closed_poi_rows"], 148_860)
        self.assertNotEqual(config.signature(), config.base.signature())

    def test_outputs_do_not_overlap_p4_or_p5(self) -> None:
        config = load_relational_codebook_config(CONFIG)
        self.assertEqual(
            config.p6_output_dir.name,
            "poi_query_category_prqk_s1_s2_hard60_topk5_v1",
        )
        self.assertEqual(
            config.p7_output_dir.name,
            "poi_query_category_geo_prqk_s3_hard60_topk5_v1",
        )
        self.assertNotEqual(config.p6_output_dir, config.p7_output_dir)
        for entry in config.frozen_inputs.values():
            self.assertNotIn(str(config.p6_output_dir), entry["path"])

    def test_real_frozen_manifests_are_valid(self) -> None:
        validate_relational_frozen_inputs(load_relational_codebook_config(CONFIG))

    def test_rejects_unconfirmed_algorithm_changes(self) -> None:
        changed_weight = copy.deepcopy(self.raw)
        changed_weight["prqk"]["topk_refinement"]["beta"] = 10
        changed_limit = copy.deepcopy(self.raw)
        changed_limit["prqk"]["max_iter"] = 90
        changed_authorization = copy.deepcopy(self.raw)
        changed_authorization["authorization"]["id"] = "UNKNOWN"
        changed_gate = copy.deepcopy(self.raw)
        changed_gate["p6_gates"]["sample"]["query_rows"] = 999
        for payload in (changed_weight, changed_limit, changed_authorization, changed_gate):
            with self.subTest(payload=payload):
                with self.assertRaises(RelationalConfigError):
                    load_relational_codebook_config(CONFIG, payload=payload)


class RelationalCodebookDirectFullConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = yaml.safe_load(DIRECT_FULL_CONFIG.read_text(encoding="utf-8"))

    def test_direct_full_is_a_separate_reviewed_contract(self) -> None:
        config = load_relational_codebook_config(DIRECT_FULL_CONFIG)
        self.assertEqual(config.schema_version, DIRECT_FULL_SCHEMA_VERSION)
        self.assertEqual(config.authorization["id"], DIRECT_FULL_AUTHORIZATION_ID)
        self.assertEqual(set(config.p6_gates), {"selection", "seed", "sample", "full"})
        self.assertEqual(config.p6_gates["full"]["base_poi_rows"], 716_245)
        self.assertEqual(config.p6_gates["full"]["query_rows"], 342_879)
        self.assertEqual(
            config.p6_gates["full"]["expected_s1_s2_edge_rows"], 912_980
        )
        self.assertIn("p6_sample_manifest", config.frozen_inputs)
        self.assertEqual(
            output_directory(config, "full").name,
            "full_342879q_716245p",
        )
        self.assertEqual(next_status_for_gate("full"), "HOLD_FOR_P6_FULL_REVIEW")

    def test_direct_full_frozen_manifests_are_valid(self) -> None:
        validate_relational_frozen_inputs(
            load_relational_codebook_config(DIRECT_FULL_CONFIG)
        )

    def test_rejects_unconfirmed_full_scale_or_algorithm_change(self) -> None:
        changed_rows = copy.deepcopy(self.raw)
        changed_rows["p6_gates"]["full"]["query_rows"] -= 1
        changed_weight = copy.deepcopy(self.raw)
        changed_weight["prqk"]["topk_refinement"]["beta"] = 10
        changed_authorization = copy.deepcopy(self.raw)
        changed_authorization["authorization"]["id"] = "UNKNOWN"
        for payload in (changed_rows, changed_weight, changed_authorization):
            with self.subTest(payload=payload):
                with self.assertRaises(RelationalConfigError):
                    load_relational_codebook_config(DIRECT_FULL_CONFIG, payload=payload)


if __name__ == "__main__":
    unittest.main()
