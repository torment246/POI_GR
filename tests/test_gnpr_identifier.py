"""Synthetic tests for paper-compatible GNPR dedup token construction."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr_identifier import (
    GnprIdentifierError,
    build_gnpr_identifiers,
    compose_gnpr_ids,
    sha256_file,
)
from poi_gr.methods.tiger_identifier import assign_collision_codes


class GnprIdentifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def sample_rows() -> list[tuple[str, list[int]]]:
        return [
            ("p-z", [10, 20, 30]),
            ("p-a", [10, 20, 30]),
            ("p-single", [11, 21, 31]),
            ("q-b", [12, 22, 32]),
            ("q-a", [12, 22, 32]),
        ]

    def write_fixture(
        self,
        *,
        rows: list[tuple[str, list[int]]] | None = None,
        bad_shape: bool = False,
    ) -> Path:
        rows = list(rows or self.sample_rows())
        poi_ids = [row[0] for row in rows]
        codes = np.asarray([row[1] for row in rows], dtype=np.int32)
        if bad_shape:
            codes = codes[:, :2]

        fixture_dir = self.root / f"fixture-{len(list(self.root.iterdir()))}"
        fixture_dir.mkdir()
        codes_path = fixture_dir / "sid_codes.npy"
        ids_path = fixture_dir / "poi_ids.jsonl"
        manifest_path = fixture_dir / "sid_manifest.json"
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
                    "poi_ids": {
                        "path": ids_path.name,
                        "sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
                    },
                    "codebook_sizes": [64] * codes.shape[1],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def test_only_collisions_receive_dedup_code(self) -> None:
        rows = self.sample_rows()
        codes = np.asarray([row[1] for row in rows], dtype=np.int32)
        assignment = assign_collision_codes(codes, [row[0] for row in rows])
        identifiers = compose_gnpr_ids(
            codes,
            assignment.collision_codes,
            assignment.bucket_sizes_by_row,
        )

        self.assertEqual(identifiers[:, 3].tolist(), [1, 0, -1, 1, 0])
        self.assertEqual(len(np.unique(identifiers, axis=0)), len(rows))

    def test_build_is_unique_reversible_and_repeatable(self) -> None:
        manifest_path = self.write_fixture()
        output_dir = self.root / "output"
        first = build_gnpr_identifiers(manifest_path, output_dir)
        first_hash = sha256_file(output_dir / "poi_gnpr_id_mapping.parquet")
        second = build_gnpr_identifiers(manifest_path, output_dir)
        second_hash = sha256_file(output_dir / "poi_gnpr_id_mapping.parquet")

        table = pq.read_table(output_dir / "poi_gnpr_id_mapping.parquet")
        mapping = table.to_pydict()
        identifiers = np.load(output_dir / "gnpr_ids.npy")
        manifest = json.loads(
            (output_dir / "gnpr_id_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(mapping["dedup_code"], [1, 0, None, 1, 0])
        self.assertEqual(
            mapping["gnpr_id_key"],
            ["10-20-30|d1", "10-20-30|d0", "11-21-31", "12-22-32|d1", "12-22-32|d0"],
        )
        np.testing.assert_array_equal(
            identifiers[:, :3], np.load(manifest_path.parent / "sid_codes.npy")
        )
        self.assertEqual(
            manifest["gnpr_ids"]["serialized_length"],
            {"collision": 4, "singleton": 3},
        )
        self.assertEqual(first.mapping_sha256, first_hash)
        self.assertEqual(second.mapping_sha256, second_hash)
        self.assertEqual(first_hash, second_hash)

    def test_non_three_level_sid_is_rejected(self) -> None:
        manifest_path = self.write_fixture(bad_shape=True)
        with self.assertRaisesRegex(GnprIdentifierError, "三层"):
            build_gnpr_identifiers(manifest_path, self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
