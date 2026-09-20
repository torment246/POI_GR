from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from qg_prqk.config import (
    LEGACY_V1_CODEBOOK_SIZES,
    QGPRQKConfigError,
    load_config,
    load_legacy_config,
)


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = QG_ROOT / "configs/qg_prqk_1024x3.yaml"


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def write_config(self, raw: dict) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        config_dir = root / "qg_prqk/configs"
        config_dir.mkdir(parents=True)
        raw = copy.deepcopy(raw)
        raw["paths"]["project_root"] = "../.."
        path = config_dir / "config.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        return temporary, path

    def test_legacy_config_loads(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(config.project.codebook_sizes, (1024, 1024, 1024))
        self.assertEqual(config.method.hard_graph.threshold, 0.60)
        self.assertEqual(config.query_stats.min_top1_share, 0.75)
        self.assertEqual(config.query_embedding.max_seq_length, 128)
        self.assertIsNone(config.query_embedding.prompt_name)
        self.assertEqual(config.query_embedding.output_dtype, "float16")
        self.assertEqual(config.query_adapter.bottleneck, 64)
        self.assertEqual(config.query_adapter.temperature, 0.05)
        self.assertEqual(config.query_adapter.ann_top_k, 100)
        self.assertEqual(
            (
                config.query_adapter.semantic_ann_negatives,
                config.query_adapter.lexical_metadata_negatives,
                config.query_adapter.local_geo_negatives,
            ),
            (6, 4, 6),
        )
        self.assertEqual(
            config.paths.output_root,
            config.paths.project_root / "qg_prqk/outputs",
        )
        self.assertFalse(config.runtime.overwrite)
        self.assertTrue(str(config.paths.output_dir).startswith(str(config.paths.output_root)))

    def test_old_loader_name_is_a_legacy_compatibility_alias(self) -> None:
        self.assertIs(load_config, load_legacy_config)
        self.assertEqual(LEGACY_V1_CODEBOOK_SIZES, (1024, 1024, 1024))

    def test_current_sid_yaml_points_to_the_current_loader(self) -> None:
        current = QG_ROOT / "configs/qg_prqk_v2_1_category_active_512x3.yaml"
        with self.assertRaisesRegex(QGPRQKConfigError, "load_downstream_config"):
            load_legacy_config(current)

    def test_rejects_non_frozen_codebook(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["project"]["codebook_sizes"] = [512, 1024, 2048]
        temporary, path = self.write_config(raw)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(QGPRQKConfigError, "必须固定"):
            load_config(path)

    def test_rejects_nonexistent_poi_field(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["data_contracts"]["poi_method_fields"].append("brand")
        temporary, path = self.write_config(raw)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(QGPRQKConfigError, "不得包含"):
            load_config(path)

    def test_rejects_area_as_method_field(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["data_contracts"]["poi_method_fields"].append("area")
        temporary, path = self.write_config(raw)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(QGPRQKConfigError, "area/layer/click_score"):
            load_config(path)

    def test_rejects_output_outside_qg_root(self) -> None:
        with self.assertRaisesRegex(QGPRQKConfigError, "独立子目录"):
            load_config(CONFIG_PATH, output_dir=Path("outputs/not_qg"))

    def test_resume_and_overwrite_conflict(self) -> None:
        with self.assertRaisesRegex(QGPRQKConfigError, "不能同时"):
            load_config(CONFIG_PATH, resume=True, overwrite=True)

    def test_runtime_controls_do_not_change_semantic_signature(self) -> None:
        base = load_config(CONFIG_PATH)
        resumed = load_config(CONFIG_PATH, resume=True, sample_limit=99)
        overwritten = load_config(CONFIG_PATH, overwrite=True)
        self.assertEqual(base.signature(), resumed.signature())
        self.assertEqual(base.signature(), overwritten.signature())

    def test_rejects_changed_query_stats_threshold(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["query_stats"]["min_top1_share"] = 0.70
        temporary, path = self.write_config(raw)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(QGPRQKConfigError, "冻结口径"):
            load_config(path)

    def test_rejects_query_embedding_instruction(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["query_embedding"]["prompt_name"] = "query"
        temporary, path = self.write_config(raw)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(QGPRQKConfigError, "无 instruction"):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
