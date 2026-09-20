from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.sid.identifiers import (
    FinalIdentifierError,
    assign_optional_dedup,
    build_final_identifiers,
    identifier_content,
)


def write_sid_source(root: Path) -> tuple[Path, Path]:
    source = root / "sid"
    source.mkdir()
    mapping = source / "poi_sid.parquet"
    pq.write_table(
        pa.table(
            {
                "poi_row_index": pa.array([0, 1, 2, 3], type=pa.int64()),
                "poi_id": ["p2", "p1", "p3", "p4"],
                "s1": pa.array([1, 1, 1, 2], type=pa.int32()),
                "s2": pa.array([2, 2, 2, 3], type=pa.int32()),
                "s3": pa.array([3, 3, 3, 4], type=pa.int32()),
            }
        ),
        mapping,
    )
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
    )
    (source / "_SUCCESS").touch()
    return mapping, manifest


class FinalIdentifiersTest(unittest.TestCase):
    def test_optional_dedup_is_lexicographic_and_only_for_collisions(self) -> None:
        base = np.asarray([[1, 2], [1, 2], [2, 3]], dtype=np.int32)
        result = assign_optional_dedup(base, ["p2", "p1", "p3"])
        self.assertEqual(result.codes.tolist(), [1, 0, -1])
        self.assertEqual(result.collision_bucket_count, 1)
        self.assertEqual(result.max_bucket_size, 2)

    def test_identifier_content_keeps_dedup_last(self) -> None:
        gid = identifier_content(
            [28, 29, 4, 12, 31, 24, 1, 2, 3],
            7,
            variant="a4_gid_parent",
        )
        self.assertTrue(gid.startswith("<G_w><G_x><G_4><G_d><G_z><G_s>"))
        self.assertTrue(gid.endswith("<S1_1><S2_2><S3_3><D_7>"))
        nogid = identifier_content([1, 2, 3], -1, variant="a4_nogid")
        self.assertEqual(nogid, "<S1_1><S2_2><S3_3>")

    def test_builds_distinct_gid_and_nogid_collision_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping, manifest = write_sid_source(root)
            gid_path = root / "gid.npy"
            np.save(
                gid_path,
                np.asarray(
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2, 2],
                        [3, 3, 3, 3, 3, 3],
                    ],
                    dtype=np.uint8,
                ),
                allow_pickle=False,
            )
            gid_manifest = root / "gid_manifest.json"
            gid_manifest.write_text(
                json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
            )
            gid_result = build_final_identifiers(
                variant="a4_gid_parent",
                sid_mapping_path=mapping,
                sid_manifest_path=manifest,
                gid_codes_path=gid_path,
                gid_manifest_path=gid_manifest,
                output_dir=root / "gid_output",
                chunk_rows=2,
            )
            nogid_result = build_final_identifiers(
                variant="a4_nogid",
                sid_mapping_path=mapping,
                sid_manifest_path=manifest,
                output_dir=root / "nogid_output",
                chunk_rows=2,
            )
            self.assertEqual(gid_result.metrics["base_distinct_count"], 3)
            self.assertEqual(gid_result.metrics["max_base_bucket_size"], 2)
            self.assertEqual(nogid_result.metrics["base_distinct_count"], 2)
            self.assertEqual(nogid_result.metrics["max_base_bucket_size"], 3)
            self.assertEqual(gid_result.metrics["final_unique_ratio"], 1.0)
            self.assertEqual(nogid_result.metrics["final_unique_ratio"], 1.0)
            with self.assertRaisesRegex(FinalIdentifierError, "禁止读取"):
                build_final_identifiers(
                    variant="a4_nogid",
                    sid_mapping_path=mapping,
                    sid_manifest_path=manifest,
                    gid_codes_path=gid_path,
                    output_dir=root / "invalid",
                )


if __name__ == "__main__":
    unittest.main()
