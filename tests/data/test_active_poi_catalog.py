"""Tests for building a POI catalog from current targets and histories."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.data import (
    ActiveCatalogConfig,
    ActiveCatalogError,
    build_active_poi_catalog,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class ActivePoiCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.orders_dir = self.root / "orders"
        self.poi_dir = self.root / "pois"
        self.output_dir = self.root / "active"
        self.temp_dir = self.root / "build_tmp"
        self.orders_dir.mkdir()
        self.poi_dir.mkdir()
        (self.orders_dir / "_SUCCESS").touch()
        (self.poi_dir / "_SUCCESS").touch()

        write_jsonl(
            self.orders_dir / "part-00000.json",
            [
                {
                    "source_dt": "20260701",
                    "poi_id": "p1",
                    "history_length": 2,
                    "history_sequence": [{"poi_id": "p2"}, {"poi_id": "p3"}],
                },
                {
                    "source_dt": "20260702",
                    "poi_id": "p2",
                    "history_length": 1,
                    "history_sequence": [{"poi_id": "p3"}],
                },
            ],
        )
        write_jsonl(
            self.orders_dir / "part-00001.json",
            [
                {
                    "source_dt": "20260703",
                    "poi_id": "p4",
                    "history_length": 0,
                    "history_sequence": [],
                }
            ],
        )
        write_jsonl(
            self.poi_dir / "part-00000.json",
            [
                {"poi_id": "p1", "text": "one"},
                {"poi_id": "unused", "text": "unused"},
                {"poi_id": "p3", "text": "three"},
            ],
        )
        write_jsonl(
            self.poi_dir / "part-00001.json",
            [
                {"poi_id": "p2", "text": "two"},
                {"poi_id": "p4", "text": "four"},
            ],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, **overrides: object) -> ActiveCatalogConfig:
        values: dict[str, object] = {
            "orders_dir": self.orders_dir,
            "poi_dir": self.poi_dir,
            "output_dir": self.output_dir,
            "temp_dir": self.temp_dir,
            "expected_order_rows": 3,
            "expected_history_occurrences": 3,
            "expected_current_unique": 3,
            "expected_history_unique": 2,
            "expected_union_unique": 4,
            "expected_poi_rows": 5,
            "progress_interval_rows": 100,
        }
        values.update(overrides)
        return ActiveCatalogConfig(**values)  # type: ignore[arg-type]

    def test_builds_exact_union_in_source_order(self) -> None:
        manifest = build_active_poi_catalog(self.config(), progress=lambda _: None)

        self.assertEqual(manifest["output"]["rows"], 4)
        self.assertTrue((self.output_dir / "_SUCCESS").is_file())
        stats = json.loads((self.output_dir / "stats.json").read_text())
        self.assertEqual(stats["intersection_unique"], 1)
        self.assertEqual(stats["history_only_unique"], 1)
        self.assertEqual(stats["current_only_unique"], 2)
        ids = [
            json.loads(line)
            for line in (self.output_dir / "poi_ids.jsonl").read_text().splitlines()
        ]
        self.assertEqual(ids, ["p1", "p3", "p2", "p4"])
        output_rows = []
        for path in sorted(self.output_dir.glob("part-*.json")):
            output_rows.extend(json.loads(line) for line in path.read_text().splitlines())
        self.assertEqual([row["poi_id"] for row in output_rows], ids)
        self.assertEqual(output_rows[0], {"poi_id": "p1", "text": "one"})

    def test_missing_active_poi_is_rejected_without_publishing(self) -> None:
        write_jsonl(
            self.orders_dir / "part-00001.json",
            [
                {
                    "source_dt": "20260703",
                    "poi_id": "missing",
                    "history_length": 0,
                    "history_sequence": [],
                }
            ],
        )
        with self.assertRaisesRegex(ActiveCatalogError, "缺少"):
            build_active_poi_catalog(
                self.config(expected_union_unique=4), progress=lambda _: None
            )
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.temp_dir.exists())

    def test_history_length_mismatch_is_rejected(self) -> None:
        write_jsonl(
            self.orders_dir / "part-00001.json",
            [
                {
                    "source_dt": "20260703",
                    "poi_id": "p4",
                    "history_length": 1,
                    "history_sequence": [],
                }
            ],
        )
        with self.assertRaisesRegex(ActiveCatalogError, "不一致"):
            build_active_poi_catalog(self.config(), progress=lambda _: None)
        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
