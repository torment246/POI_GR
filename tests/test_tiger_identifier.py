"""Synthetic tests for deterministic TIGER collision token construction."""

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

from poi_gr.methods.tiger_identifier import (
    TigerIdentifierError,
    assign_collision_codes,
    build_tiger_identifiers,
    compose_tiger_ids,
    sha256_file,
)


class TigerIdentifierTest(unittest.TestCase):
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
        duplicate_id: bool = False,
        bad_shape: bool = False,
    ) -> Path:
        rows = list(rows or self.sample_rows())
        poi_ids = [row[0] for row in rows]
        if duplicate_id:
            poi_ids[1] = poi_ids[0]
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
                        "sha256": hashlib.sha256(
                            ids_path.read_bytes()
                        ).hexdigest(),
                    },
                    "codebook_sizes": [64] * codes.shape[1],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def test_singletons_and_collisions_always_have_fourth_token(self) -> None:
        rows = self.sample_rows()
        codes = np.asarray([row[1] for row in rows], dtype=np.int32)
        assignment = assign_collision_codes(
            codes,
            [row[0] for row in rows],
        )
        tiger_ids = compose_tiger_ids(codes, assignment.collision_codes)

        self.assertEqual(
            assignment.collision_codes.tolist(),
            [1, 0, 0, 1, 0],
        )
        self.assertEqual(tiger_ids.shape, (5, 4))
        self.assertEqual(tiger_ids[2, 3], 0)
        self.assertEqual(len(np.unique(tiger_ids, axis=0)), 5)
        self.assertEqual(assignment.max_bucket_size, 2)
        self.assertEqual(assignment.max_collision_code, 1)

    def test_input_order_does_not_change_poi_to_collision_mapping(self) -> None:
        rows = self.sample_rows()
        reordered = [rows[index] for index in [4, 2, 0, 3, 1]]

        def mapping(values: list[tuple[str, list[int]]]) -> dict[str, int]:
            codes = np.asarray([row[1] for row in values], dtype=np.int32)
            assignment = assign_collision_codes(
                codes,
                [row[0] for row in values],
            )
            return {
                row[0]: int(code)
                for row, code in zip(
                    values,
                    assignment.collision_codes,
                    strict=True,
                )
            }

        self.assertEqual(mapping(rows), mapping(reordered))

    def test_build_writes_unique_reversible_mapping_and_is_repeatable(self) -> None:
        manifest_path = self.write_fixture()
        output_dir = self.root / "output"
        first = build_tiger_identifiers(manifest_path, output_dir)
        first_hash = sha256_file(
            output_dir / "poi_tiger_id_mapping.parquet"
        )
        second = build_tiger_identifiers(manifest_path, output_dir)
        second_hash = sha256_file(
            output_dir / "poi_tiger_id_mapping.parquet"
        )

        table = pq.read_table(output_dir / "poi_tiger_id_mapping.parquet")
        mapping = table.to_pydict()
        tiger_ids = np.load(output_dir / "tiger_ids.npy")
        manifest = json.loads(
            (output_dir / "tiger_id_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(table.num_rows, 5)
        self.assertEqual(len(set(mapping["poi_id"])), 5)
        self.assertEqual(len(set(mapping["tiger_id_key"])), 5)
        self.assertEqual(mapping["collision_code"], [1, 0, 0, 1, 0])
        self.assertEqual(
            mapping["has_semantic_collision"],
            [True, True, False, True, True],
        )
        np.testing.assert_array_equal(tiger_ids[:, :3], np.load(
            manifest_path.parent / "sid_codes.npy"
        ))
        self.assertEqual(manifest["tiger_ids"]["fixed_length"], 4)
        self.assertEqual(
            manifest["collision_assignment"]["singleton_code"],
            0,
        )
        self.assertEqual(first.mapping_sha256, first_hash)
        self.assertEqual(second.mapping_sha256, second_hash)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            set((
                "collision_codes.npy",
                "tiger_ids.npy",
                "poi_tiger_id_mapping.parquet",
                "tiger_id_manifest.json",
                "metrics.json",
            )),
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        cases = (
            ({"duplicate_id": True}, "POI ID 重复"),
            ({"bad_shape": True}, "三层"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                manifest_path = self.write_fixture(**kwargs)
                with self.assertRaisesRegex(TigerIdentifierError, message):
                    build_tiger_identifiers(
                        manifest_path,
                        self.root / f"invalid-{len(list(self.root.iterdir()))}",
                    )


if __name__ == "__main__":
    unittest.main()
