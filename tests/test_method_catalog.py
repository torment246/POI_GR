"""Tests for method-level baseline and innovation organization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.catalog import (  # noqa: E402
    MethodCatalogError,
    load_method_catalog,
    load_method_config,
)


class MethodCatalogTest(unittest.TestCase):
    def test_repository_catalog_has_one_current_and_three_baselines(self) -> None:
        catalog = load_method_catalog(
            PROJECT_ROOT / "configs/methods",
            PROJECT_ROOT,
        )
        self.assertEqual(
            set(catalog),
            {"current_main_v1", "genpoi", "tiger", "gnpr_sid"},
        )
        self.assertEqual(catalog["current_main_v1"].group, "current")
        self.assertEqual(
            {spec.group for key, spec in catalog.items() if key != "current_main_v1"},
            {"baseline"},
        )

    def test_repository_method_status_matches_completed_stages(self) -> None:
        catalog = load_method_catalog(
            PROJECT_ROOT / "configs/methods",
            PROJECT_ROOT,
        )
        current = catalog["current_main_v1"]
        self.assertEqual(current.status, "implemented")
        self.assertTrue(all(stage.status == "ready" for stage in current.stages))

        self.assertEqual(catalog["tiger"].task["max_history_events"], 10)
        self.assertEqual(
            catalog["tiger"].task["history_fields"],
            ["query", "user_lng", "user_lat", "poi_id", "event_time"],
        )
        self.assertEqual(catalog["gnpr_sid"].task["max_history_events"], 10)
        self.assertEqual(
            catalog["gnpr_sid"].task["target_fields"],
            ["query", "user_gid", "poi_id"],
        )
        self.assertEqual(
            catalog["gnpr_sid"].identifier["representation_features"],
            [
                "bge_m3_text",
                "category_code",
                "plus_code6",
            ],
        )
        self.assertIsNone(catalog["gnpr_sid"].identifier["catalog_filter"])
        self.assertEqual(
            catalog["gnpr_sid"].identifier["beijing_input_dimensions"]["total"],
            2192,
        )
        self.assertEqual(
            catalog["gnpr_sid"].identifier["token_order"],
            ["s1", "s2", "s3", "dedup_if_collision"],
        )
        self.assertEqual(
            catalog["genpoi"].task["history_fields"],
            ["query", "user_lng", "user_lat", "poi_id", "event_time"],
        )
        tiger = catalog["tiger"]
        self.assertEqual(tiger.status, "implemented")
        self.assertTrue(all(stage.status == "ready" for stage in tiger.stages))

        for method_id in ("genpoi", "gnpr_sid"):
            method = catalog[method_id]
            self.assertEqual(method.status, "partial")
            statuses = {stage.status for stage in method.stages}
            self.assertNotEqual(statuses, {"ready"})

    def test_missing_repository_command_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_dir = Path(temporary) / "current"
            config_dir.mkdir()
            config_path = config_dir / "broken.yaml"
            payload = {
                "schema_version": "poi-gr-method-v1",
                "method": {
                    "id": "broken",
                    "group": "current",
                    "display_name": "broken",
                    "status": "implemented",
                    "summary": "broken",
                    "paper": None,
                },
                "task": {
                    "type": "query_to_poi",
                    "target_fields": ["poi_id"],
                    "history_fields": [],
                    "max_history_events": None,
                },
                "identifier": {
                    "type": "test",
                    "token_order": ["sid"],
                },
                "pipeline": {
                    stage: {
                        "status": "ready",
                        "notes": "test",
                        "commands": [
                            {
                                "entrypoint": "scripts/not_found.py",
                                "config": None,
                                "purpose": "test",
                            }
                        ],
                    }
                    for stage in ("data", "identifier", "train", "evaluate")
                },
            }
            config_path.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MethodCatalogError, "不存在"):
                load_method_config(config_path, PROJECT_ROOT)

    def test_missing_stage_cannot_declare_commands(self) -> None:
        config_path = PROJECT_ROOT / "configs/methods/baselines/gnpr_sid.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["pipeline"]["train"]["status"] = "missing"
        payload["pipeline"]["train"]["commands"] = [
            {
                "entrypoint": "scripts/build_sft_main_data.py",
                "config": None,
                "purpose": "invalid",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            config_dir = temporary_root / "baselines"
            config_dir.mkdir()
            copied = config_dir / "gnpr_sid.yaml"
            copied.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MethodCatalogError, "不应声明"):
                load_method_config(copied, PROJECT_ROOT)

    def test_innovation_is_discovered_without_a_registry_code_change(self) -> None:
        source = PROJECT_ROOT / "configs/methods/current/main_v1.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["method"].update(
            {
                "id": "history_ablation",
                "group": "innovation",
                "display_name": "历史消融",
                "status": "implemented",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            config_root = Path(temporary)
            innovation_dir = config_root / "innovations"
            innovation_dir.mkdir()
            config_path = innovation_dir / "history_ablation.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            catalog = load_method_catalog(config_root, PROJECT_ROOT)
            self.assertEqual(set(catalog), {"history_ablation"})
            self.assertEqual(catalog["history_ablation"].group, "innovation")


if __name__ == "__main__":
    unittest.main()
