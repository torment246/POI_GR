"""Synthetic tests for bounded Beijing GNPR-SID inputs."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from poi_gr.methods.gnpr.sid_input import (  # noqa: E402
    active_feature_indices,
    build_feature_dimensions,
    hash_top_visitors,
    prepare_gnpr_sid_input,
    stable_user_hash,
)
from scripts.gnpr.prepare_sid_inputs import prepare_dataset  # noqa: E402


class GnprSidInputTest(unittest.TestCase):
    def test_hash_is_stable_bounded_and_deduplicated(self) -> None:
        first = stable_user_hash("user-1")
        self.assertEqual(first, stable_user_hash("user-1"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 8192)
        self.assertEqual(hash_top_visitors(["user-1", "user-1"]), (first,))

    def test_cold_poi_is_filtered_and_offsets_total_9384(self) -> None:
        dimensions = build_feature_dimensions(
            category_count=402,
            region_count=766,
        )
        self.assertEqual(dimensions.total_dim, 9384)
        cold = prepare_gnpr_sid_input(
            {
                "poi_id": "cold",
                "category_index": 1,
                "region_index": 2,
                "top_visit_hours": [],
                "top_visitor_ids": [],
                "interaction_count": 0,
            },
            dimensions=dimensions,
        )
        self.assertIsNone(cold)
        behavior = prepare_gnpr_sid_input(
            {
                "poi_id": "behavior",
                "category_index": 1,
                "region_index": 2,
                "top_visit_hours": [21, 8, 8],
                "top_visitor_ids": ["user-2", "user-1"],
                "interaction_count": 3,
            },
            dimensions=dimensions,
        )
        assert behavior is not None
        active = active_feature_indices(behavior, dimensions=dimensions)
        self.assertEqual(len(active), 2 + 2 + 2)
        self.assertTrue(all(0 <= value < 9384 for value in active))

    def test_streaming_parquet_conversion(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "raw"
            output_dir = root / "bounded"
            (input_dir / "poi_features.parquet").mkdir(parents=True)
            (input_dir / "category_vocab.parquet").mkdir()
            (input_dir / "region_vocab.parquet").mkdir()
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "poi_id": "poi-a",
                            "category_index": 0,
                            "region_index": 1,
                            "top_visit_hours": [8, 9],
                            "top_visitor_ids": ["user-1", "user-2"],
                            "interaction_count": 4,
                        },
                        {
                            "poi_id": "poi-cold",
                            "category_index": 1,
                            "region_index": 0,
                            "top_visit_hours": [],
                            "top_visitor_ids": [],
                            "interaction_count": 0,
                        },
                    ]
                ),
                input_dir / "poi_features.parquet" / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"category_code": "a", "category_index": 0},
                        {"category_code": "b", "category_index": 1},
                    ]
                ),
                input_dir / "category_vocab.parquet" / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"plus_code_region": "r1", "region_index": 0},
                        {"plus_code_region": "r2", "region_index": 1},
                        {"plus_code_region": "r3", "region_index": 2},
                    ]
                ),
                input_dir / "region_vocab.parquet" / "part-00000.parquet",
            )

            manifest = prepare_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                user_hash_buckets=8192,
                batch_size=1,
                expected_source_pois=2,
                expected_output_pois=1,
            )
            self.assertEqual(manifest["dimensions"]["total"], 8221)
            self.assertEqual(manifest["stats"]["filtered_cold_poi_count"], 1)
            table = pq.read_table(output_dir / "poi_sid_inputs.parquet")
            self.assertEqual(table.num_rows, 1)
            row = table.to_pylist()[0]
            self.assertEqual(row["poi_id"], "poi-a")
            self.assertEqual(len(row["user_hash_indices"]), 2)
            saved_manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["catalog_filter"], "interaction_count > 0")
            self.assertTrue((output_dir / "_SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()
