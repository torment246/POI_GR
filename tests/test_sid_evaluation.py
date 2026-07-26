"""Synthetic tests for the unified SID evaluation module and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sid_evaluation import SidEvaluationError, run_sid_evaluation


BASE_CODES = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 3],
        [1, 3, 4],
        [2, 4, 5],
    ],
    dtype=np.int64,
)


class SidEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_fixture(
        self,
        *,
        codes: np.ndarray = BASE_CODES,
        poi_ids: list[str] | None = None,
        codebook_sizes: list[int] | None = None,
    ) -> tuple[Path, Path]:
        codes = np.asarray(codes)
        if poi_ids is None:
            poi_ids = [f"p{index}" for index in range(1, len(codes) + 1)]
        if codebook_sizes is None:
            codebook_sizes = [4, 5, 6]

        codes_path = self.root / "sid_codes.npy"
        ids_path = self.root / "poi_ids.jsonl"
        manifest_path = self.root / "sid_manifest.json"
        poi_path = self.root / "pois.jsonl"
        np.save(codes_path, codes, allow_pickle=False)
        ids_path.write_text(
            "".join(json.dumps(poi_id) + "\n" for poi_id in poi_ids),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "sid-input-v1",
                    "sid_codes": {
                        "path": codes_path.name,
                        "shape": list(codes.shape),
                        "dtype": str(codes.dtype),
                    },
                    "poi_ids": {"path": ids_path.name},
                    "codebook_sizes": codebook_sizes,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        category_codes = ["A", "A", None, "B", "C", "B", "D", "D", "E"]
        click_scores = [1.0, 3.0, 2.0, 4.0, 4.0, 0.5, 5.0, 6.0, 7.0]
        poi_rows = []
        for index, poi_id in enumerate(poi_ids[: len(codes)]):
            poi_rows.append(
                {
                    "poi_id": poi_id,
                    "displayname": f"name-{poi_id}",
                    "address": f"address-{poi_id}",
                    "alias": f"alias-{poi_id}",
                    "category": f"category-{index}",
                    "category_code": (
                        category_codes[index] if index < len(category_codes) else "Z"
                    ),
                    "lng": 116.0 + index / 100,
                    "lat": 39.0 + index / 100,
                    "layer": index,
                    "click_score": (
                        click_scores[index] if index < len(click_scores) else float(index)
                    ),
                }
            )
        poi_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in poi_rows),
            encoding="utf-8",
        )
        return manifest_path, poi_path

    def test_metrics_prefix_purity_and_case_truncation(self) -> None:
        manifest_path, poi_path = self.write_fixture()
        output_dir = self.root / "output"
        metrics, cases = run_sid_evaluation(
            manifest_path,
            poi_path,
            output_dir,
            max_cases=2,
            max_pois_per_case=2,
        )

        basic = metrics["basic"]
        self.assertAlmostEqual(basic["distinct_sid_ratio"], 6 / 9)
        self.assertAlmostEqual(basic["collision_excess_ratio"], 3 / 9)
        self.assertAlmostEqual(basic["colliding_poi_ratio"], 5 / 9)
        self.assertEqual(
            len(
                {
                    basic["distinct_sid_ratio"],
                    basic["collision_excess_ratio"],
                    basic["colliding_poi_ratio"],
                }
            ),
            3,
        )
        self.assertEqual(basic["colliding_bucket_count"], 2)
        self.assertEqual(basic["singleton_sid_count"], 4)
        self.assertEqual(basic["bucket_size_max"], 3)

        layers = metrics["layers"]
        self.assertEqual([layer["used_token_count"] for layer in layers], [3, 5, 6])
        self.assertEqual(
            [layer["codebook_utilization_ratio"] for layer in layers],
            [3 / 4, 1.0, 1.0],
        )
        self.assertEqual(layers[0]["token_counts"], [6, 2, 1, 0])
        self.assertGreater(layers[0]["normalized_entropy"], 0.0)
        self.assertLessEqual(layers[0]["normalized_entropy"], 1.0)
        self.assertEqual(
            [prefix["prefix_bucket_count"] for prefix in metrics["prefixes"]],
            [3, 5, 6],
        )

        purity = metrics["prefixes"][0]["category_purity"]
        self.assertEqual(purity["labeled_poi_count"], 8)
        self.assertEqual(purity["missing_category_count"], 1)
        self.assertAlmostEqual(purity["category_coverage"], 8 / 9)
        self.assertAlmostEqual(purity["micro_purity"], 5 / 8)
        self.assertAlmostEqual(purity["macro_purity"], 0.8)
        self.assertEqual(purity["macro_valid_bucket_count"], 3)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["semantic_sid"], [0, 0, 0])
        self.assertEqual(cases[0]["bucket_size"], 3)
        self.assertEqual(cases[0]["output_poi_count"], 2)
        self.assertTrue(cases[0]["truncated"])
        self.assertEqual([poi["poi_id"] for poi in cases[0]["pois"]], ["p2", "p3"])
        self.assertEqual([poi["poi_id"] for poi in cases[1]["pois"]], ["p4", "p5"])
        required_fields = {
            "poi_id",
            "displayname",
            "address",
            "alias",
            "category",
            "category_code",
            "lng",
            "lat",
            "layer",
            "click_score",
        }
        self.assertEqual(set(cases[0]["pois"][0]), required_fields)
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {"metrics.json", "collision_cases.jsonl"},
        )

    def test_arbitrary_sid_layer_count(self) -> None:
        codes = np.asarray(
            [[0, 0, 0, 0], [0, 1, 2, 3], [1, 1, 3, 4]],
            dtype=np.int32,
        )
        manifest_path, poi_path = self.write_fixture(
            codes=codes,
            codebook_sizes=[2, 2, 4, 5],
        )
        metrics, _ = run_sid_evaluation(
            manifest_path,
            poi_path,
            self.root / "four-level-output",
        )
        self.assertEqual(len(metrics["layers"]), 4)
        self.assertEqual(len(metrics["prefixes"]), 4)
        self.assertEqual(metrics["inputs"]["sid_shape"], [3, 4])

    def test_duplicate_poi_id_is_rejected(self) -> None:
        poi_ids = ["p1", "p1", *[f"p{index}" for index in range(3, 10)]]
        manifest_path, poi_path = self.write_fixture(poi_ids=poi_ids)
        with self.assertRaisesRegex(SidEvaluationError, "POI ID 重复"):
            run_sid_evaluation(manifest_path, poi_path, self.root / "output")

    def test_poi_id_row_count_must_match_sid_rows(self) -> None:
        poi_ids = [f"p{index}" for index in range(1, 9)]
        manifest_path, poi_path = self.write_fixture(poi_ids=poi_ids)
        with self.assertRaisesRegex(SidEvaluationError, "行数.*不一致"):
            run_sid_evaluation(manifest_path, poi_path, self.root / "output")

    def test_negative_token_is_rejected(self) -> None:
        codes = BASE_CODES.copy()
        codes[0, 1] = -1
        manifest_path, poi_path = self.write_fixture(codes=codes)
        with self.assertRaisesRegex(SidEvaluationError, "负 Token"):
            run_sid_evaluation(manifest_path, poi_path, self.root / "output")

    def test_token_outside_declared_codebook_is_rejected(self) -> None:
        codes = BASE_CODES.copy()
        codes[0, 0] = 4
        manifest_path, poi_path = self.write_fixture(codes=codes)
        with self.assertRaisesRegex(SidEvaluationError, "超出码本范围"):
            run_sid_evaluation(manifest_path, poi_path, self.root / "output")

    def test_cli_smoke_writes_only_two_outputs(self) -> None:
        manifest_path, poi_path = self.write_fixture()
        output_dir = self.root / "cli-output"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_sid.py"),
                "--manifest",
                str(manifest_path),
                "--poi-data",
                str(poi_path),
                "--output-dir",
                str(output_dir),
                "--max-cases",
                "2",
                "--max-pois-per-case",
                "2",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["prefix_bucket_counts"], [3, 5, 6])
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {"metrics.json", "collision_cases.jsonl"},
        )


if __name__ == "__main__":
    unittest.main()
