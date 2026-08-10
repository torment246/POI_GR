"""Synthetic tests for deterministic Dedup PID construction."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.pid.dedup import (
    DedupPidError,
    assign_dedup_codes,
    build_dedup_pid,
    compose_final_pid,
    sha256_file,
)
from poi_gr.sid.evaluation import compute_basic_metrics


class DedupPidTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def sample_rows() -> list[tuple[str, list[int]]]:
        return [
            ("p-z", [1, 2, 3, 4, 5, 6, 10, 20, 30]),
            ("p-a", [1, 2, 3, 4, 5, 6, 10, 20, 30]),
            ("p-single", [1, 2, 3, 4, 5, 6, 11, 21, 31]),
            ("q-b", [2, 3, 4, 5, 6, 7, 12, 22, 32]),
            ("q-a", [2, 3, 4, 5, 6, 7, 12, 22, 32]),
        ]

    def write_fixture(
        self,
        *,
        rows: list[tuple[str, list[int]]] | None = None,
        duplicate_id: bool = False,
        omit_last_id: bool = False,
        bad_shape: bool = False,
    ) -> Path:
        rows = list(rows or self.sample_rows())
        poi_ids = [row[0] for row in rows]
        if duplicate_id:
            poi_ids[1] = poi_ids[0]
        codes = np.asarray([row[1] for row in rows], dtype=np.int32)
        if bad_shape:
            codes = codes[:, :8]

        base_dir = self.root / f"fixture-{len(list(self.root.iterdir()))}"
        base_dir.mkdir()
        codes_path = base_dir / "pid_codes.npy"
        ids_path = base_dir / "poi_ids.jsonl"
        manifest_path = base_dir / "pid_manifest.json"
        metrics_path = base_dir / "metrics.json"
        np.save(codes_path, codes, allow_pickle=False)
        written_ids = poi_ids[:-1] if omit_last_id else poi_ids
        ids_path.write_text(
            "".join(json.dumps(value) + "\n" for value in written_ids),
            encoding="utf-8",
        )
        ids_hash = hashlib.sha256(ids_path.read_bytes()).hexdigest()
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "geohash-pid-v1",
                    "status": "completed",
                    "poi_count": len(codes),
                    "poi_ids": {
                        "path": ids_path.name,
                        "rows": len(written_ids),
                        "sha256": ids_hash,
                        "unique": not duplicate_id,
                    },
                    "pid_codes": {
                        "path": codes_path.name,
                        "shape": list(codes.shape),
                        "dtype": str(codes.dtype),
                        "order": "gid_sid",
                        "token_order": [
                            "G1",
                            "G2",
                            "G3",
                            "G4",
                            "G5",
                            "G6",
                            "S1",
                            "S2",
                            "S3",
                        ],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        if codes.shape[1] == 9:
            basic, _, _, _ = compute_basic_metrics(codes)
            pid_metrics = dict(basic)
            pid_metrics["distinct_pid_count"] = pid_metrics.pop(
                "distinct_sid_count"
            )
            pid_metrics["distinct_pid_ratio"] = pid_metrics.pop(
                "distinct_sid_ratio"
            )
            pid_metrics["singleton_pid_count"] = pid_metrics.pop(
                "singleton_sid_count"
            )
        else:
            pid_metrics = {}
        metrics_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "gid6_sid_pid": pid_metrics,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def test_singleton_has_no_dedup_token(self) -> None:
        rows = self.sample_rows()
        codes = np.asarray([row[1] for row in rows], dtype=np.int32)
        assignment = assign_dedup_codes(
            codes, [row[0] for row in rows], dedup_capacity=512
        )
        final_codes = compose_final_pid(codes, assignment.dedup_codes)
        self.assertEqual(assignment.dedup_codes[2], -1)
        self.assertEqual(final_codes[2, 9], -1)

    def test_collision_codes_follow_poi_id_and_repeat_across_buckets(self) -> None:
        rows = self.sample_rows()
        codes = np.asarray([row[1] for row in rows], dtype=np.int32)
        assignment = assign_dedup_codes(
            codes, [row[0] for row in rows], dedup_capacity=512
        )
        self.assertEqual(assignment.dedup_codes.tolist(), [1, 0, -1, 1, 0])
        self.assertEqual(assignment.max_dedup_code, 1)

    def test_final_pid_is_unique_and_recovers_base_pid(self) -> None:
        rows = self.sample_rows()
        codes = np.asarray([row[1] for row in rows], dtype=np.int32)
        assignment = assign_dedup_codes(
            codes, [row[0] for row in rows], dedup_capacity=512
        )
        final_codes = compose_final_pid(codes, assignment.dedup_codes)
        self.assertEqual(len(np.unique(final_codes, axis=0)), len(rows))
        np.testing.assert_array_equal(final_codes[:, :9], codes)

    def test_input_order_change_preserves_poi_to_dedup_mapping(self) -> None:
        rows = self.sample_rows()
        reordered = [rows[index] for index in [4, 2, 0, 3, 1]]

        def mapping(values: list[tuple[str, list[int]]]) -> dict[str, int]:
            codes = np.asarray([row[1] for row in values], dtype=np.int32)
            assignment = assign_dedup_codes(
                codes, [row[0] for row in values], dedup_capacity=512
            )
            return {
                row[0]: int(code)
                for row, code in zip(values, assignment.dedup_codes)
            }

        self.assertEqual(mapping(rows), mapping(reordered))

    def test_build_writes_unique_reversible_mapping_and_manifest(self) -> None:
        manifest_path = self.write_fixture()
        output_dir = self.root / "output"
        result = build_dedup_pid(
            manifest_path,
            output_dir,
        )
        table = pq.read_table(output_dir / "poi_pid_mapping.parquet")
        mapping = table.to_pydict()
        self.assertEqual(table.num_rows, 5)
        self.assertEqual(len(set(mapping["poi_id"])), 5)
        self.assertEqual(len(set(mapping["final_pid_key"])), 5)
        self.assertEqual(mapping["dedup_code"], [1, 0, None, 1, 0])
        self.assertEqual(mapping["final_pid_length"], [10, 10, 9, 10, 10])
        self.assertEqual(mapping["requires_dedup"], [True, True, False, True, True])
        self.assertEqual(result.metrics["final_pid_unique_ratio"], 1.0)
        manifest = json.loads(
            (output_dir / "final_pid_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["dedup_assignment"]["singleton_sentinel"], -1)
        self.assertFalse(
            manifest["dedup_assignment"]["singleton_sentinel_is_token"]
        )
        self.assertEqual(
            manifest["final_pid_definition"]["singleton_length"], 9
        )
        self.assertEqual(
            set(path.name for path in output_dir.iterdir()),
            {
                "dedup_codes.npy",
                "final_pid_codes.npy",
                "poi_pid_mapping.parquet",
                "final_pid_manifest.json",
                "metrics.json",
            },
        )

    def test_repeated_build_keeps_mapping_hash(self) -> None:
        manifest_path = self.write_fixture()
        output_dir = self.root / "repeat-output"
        first = build_dedup_pid(
            manifest_path,
            output_dir,
            expected_baseline=None,
        )
        first_hash = sha256_file(output_dir / "poi_pid_mapping.parquet")
        second = build_dedup_pid(
            manifest_path,
            output_dir,
            expected_baseline=None,
        )
        second_hash = sha256_file(output_dir / "poi_pid_mapping.parquet")
        self.assertEqual(first.mapping_sha256, first_hash)
        self.assertEqual(second.mapping_sha256, second_hash)
        self.assertEqual(first_hash, second_hash)

    def test_invalid_inputs_are_rejected(self) -> None:
        cases = (
            ({"duplicate_id": True}, "POI ID 重复"),
            ({"omit_last_id": True}, "行数"),
            ({"bad_shape": True}, "shape"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                manifest_path = self.write_fixture(**kwargs)
                with self.assertRaisesRegex(DedupPidError, message):
                    build_dedup_pid(
                        manifest_path,
                        self.root / f"invalid-{len(list(self.root.iterdir()))}",
                        expected_baseline=None,
                    )


if __name__ == "__main__":
    unittest.main()
