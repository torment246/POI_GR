from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from qg_prqk.data.query_supervision_config import CategoryConfigError, load_category_config
from qg_prqk.sid.pipeline_config import (
    CURRENT_SID_CODEBOOK_SIZES,
    DownstreamConfigError,
    load_downstream_config,
)
from qg_prqk.adapters.selection_config import load_adapter_selection_config


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG = QG_ROOT / "configs/qg_prqk_v2_1_category_active_512x3.yaml"


class SidPipelineConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_capacity_changes_without_changing_vectors_or_algorithm(self) -> None:
        config = load_downstream_config(CONFIG)
        resolved = config.resolved_payload()
        old = yaml.safe_load(
            config.upstream.base.category_config.source_path.read_text(encoding="utf-8")
        )
        self.assertEqual(config.codebook_sizes, (512, 512, 512))
        self.assertEqual(CURRENT_SID_CODEBOOK_SIZES, (512, 512, 512))
        self.assertEqual(config.codebook_sizes, CURRENT_SID_CODEBOOK_SIZES)
        self.assertEqual(resolved["identifier"]["codebooks"], [512, 512, 512])
        self.assertEqual(resolved["frozen_inputs"]["poi_embedding_dim"], 1024)
        self.assertEqual(resolved["frozen_inputs"]["poi_rows"], 716245)
        for section in (
            "frozen_inputs", "category", "query_depth", "prqk", "s1", "s2",
            "s3", "warmup", "first_run", "runtime",
        ):
            self.assertEqual(resolved[section], old[section], section)
        self.assertEqual(config.query_view_policy, {
            "D1": "raw_bge", "D2": "raw_bge", "D3": "final_adapter",
        })
        self.assertEqual(config.signature(), load_downstream_config(CONFIG).signature())

    def test_upstream_and_downstream_directories_are_separate(self) -> None:
        config = load_downstream_config(CONFIG)
        self.assertIn("qg_prqk_512x3", str(config.output_dir))
        self.assertIn("qg_prqk_1024x3", str(config.query_depth_dir))
        self.assertEqual(config.query_depth_dir, config.upstream.base.query_depth_dir)
        self.assertTrue(config.final_adapter_path.name.endswith("exact_final.pt"))
        self.assertEqual(
            config.upstream.signature(),
            load_adapter_selection_config(config.upstream.source_path).signature(),
        )

    def test_rejects_old_capacity(self) -> None:
        self.raw["downstream"]["codebook_sizes"] = [1024, 1024, 1024]
        with self.assertRaisesRegex(DownstreamConfigError, "512"):
            load_downstream_config(CONFIG, payload=self.raw)

    def test_rejects_changed_query_views_or_stop_phase(self) -> None:
        for key, value in (
            ("query_view_policy", {"D1": "final_adapter", "D2": "raw_bge", "D3": "final_adapter"}),
            ("stop_after_phase", "SFT"),
        ):
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw)
                raw["downstream"][key] = value
                with self.assertRaises(DownstreamConfigError):
                    load_downstream_config(CONFIG, payload=raw)

    def test_rejects_in_place_upstream_outputs_and_path_escape(self) -> None:
        for path in (
            "qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active",
            "outputs/unauthorized",
            "qg_prqk/outputs",
        ):
            with self.subTest(path=path):
                raw = copy.deepcopy(self.raw)
                raw["downstream"]["output_dir"] = path
                with self.assertRaises(DownstreamConfigError):
                    load_downstream_config(CONFIG, payload=raw)

    def test_rejects_gate_or_select_as_final_checkpoint(self) -> None:
        self.raw["upstream_artifacts"]["final_adapter_checkpoint"]["path"] = (
            "qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/"
            "query_adapter_exact_full/select/query_adapter_exact_select_best.pt"
        )
        with self.assertRaisesRegex(DownstreamConfigError, "final_adapter_checkpoint"):
            load_downstream_config(CONFIG, payload=self.raw)

    def test_rejects_changed_frozen_config_hash(self) -> None:
        self.raw["base_category_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(DownstreamConfigError, "哈希"):
            load_downstream_config(CONFIG, payload=self.raw)

    def test_rejects_unknown_algorithm_overrides(self) -> None:
        self.raw["prqk"] = {"residual": "additive"}
        with self.assertRaisesRegex(DownstreamConfigError, "字段"):
            load_downstream_config(CONFIG, payload=self.raw)

    def test_old_p2_5_loader_cannot_run_downstream_config(self) -> None:
        with self.assertRaises(CategoryConfigError):
            load_category_config(CONFIG)


if __name__ == "__main__":
    unittest.main()
