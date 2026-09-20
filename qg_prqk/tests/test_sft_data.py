from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.identifiers import build_final_identifiers
from qg_prqk.sft.data import build_paired_sft_data


def write_sid_source(root: Path) -> tuple[Path, Path]:
    source = root / "sid"
    source.mkdir()
    mapping = source / "poi_sid.parquet"
    pq.write_table(
        pa.table(
            {
                "poi_row_index": pa.array([0, 1, 2], type=pa.int64()),
                "poi_id": ["p2", "p1", "p3"],
                "s1": pa.array([1, 1, 2], type=pa.int32()),
                "s2": pa.array([2, 2, 3], type=pa.int32()),
                "s3": pa.array([3, 3, 4], type=pa.int32()),
            }
        ),
        mapping,
    )
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"status": "completed"}) + "\n")
    (source / "_SUCCESS").touch()
    return mapping, manifest


def write_tiger_identifier(root: Path, poi_ids: list[str]) -> Path:
    directory = root / "tiger_ids"
    directory.mkdir()
    mapping = directory / "poi_tiger_id_mapping.parquet"
    pq.write_table(
        pa.table(
            {
                "poi_id": poi_ids,
                "s1": pa.array([0, 1, 2], type=pa.int32()),
                "s2": pa.array([0, 1, 2], type=pa.int32()),
                "s3": pa.array([0, 1, 2], type=pa.int32()),
                "collision_code": pa.array([0, 0, 0], type=pa.int32()),
            }
        ),
        mapping,
    )
    (directory / "tiger_id_manifest.json").write_text(
        json.dumps({"status": "completed"}) + "\n"
    )
    return directory


def tiger_content(value: int) -> str:
    return f"<S1_{value}><S2_{value}><S3_{value}><C_0>"


def write_source_sft(root: Path, tiger_dir: Path) -> Path:
    directory = root / "source_sft"
    directory.mkdir()
    records = {
        "train": {
            "sample_id": "a",
            "order_id": "o1",
            "searchid": "s1",
            "target_poi_id": "p1",
            "target_tiger_id_key": "1-1-1|c0",
            "history_length": 1,
            "split": "train",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "<USER_GID><G_w><G_x><G_4><G_d><G_z><G_s></USER_GID>\n"
                        f"<POI_TIGER_ID>{tiger_content(0)}</POI_TIGER_ID>\n"
                        "<QUERY>测试</QUERY>"
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"<TARGET_POI>{tiger_content(1)}</TARGET_POI>",
                },
            ],
        },
        "valid": {
            "sample_id": "b",
            "order_id": "o2",
            "searchid": "s2",
            "target_poi_id": "p3",
            "target_tiger_id_key": "2-2-2|c0",
            "history_length": 0,
            "split": "valid",
            "messages": [
                {"role": "user", "content": "<QUERY>验证</QUERY>"},
                {
                    "role": "assistant",
                    "content": f"<TARGET_POI>{tiger_content(2)}</TARGET_POI>",
                },
            ],
        },
        "test": {
            "sample_id": "c",
            "order_id": "o3",
            "searchid": "s3",
            "target_poi_id": "p2",
            "target_tiger_id_key": "0-0-0|c0",
            "history_length": 0,
            "split": "test",
            "messages": [
                {"role": "user", "content": "<QUERY>测试集</QUERY>"},
                {
                    "role": "assistant",
                    "content": f"<TARGET_POI>{tiger_content(0)}</TARGET_POI>",
                },
            ],
        },
    }
    outputs = {}
    for split, record in records.items():
        path = directory / f"{split}.jsonl"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[path.name] = {"rows": 1, "sha256": sha256_file(path)}
    (directory / "stats.json").write_text(
        json.dumps(
            {
                "retained_sample_count": 3,
                "train_count": 1,
                "valid_count": 1,
                "test_count": 1,
            }
        )
        + "\n"
    )
    manifest_path = tiger_dir / "tiger_id_manifest.json"
    mapping_path = tiger_dir / "poi_tiger_id_mapping.parquet"
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "tiger-map-search-sft-data-v1",
                "status": "completed",
                "outputs": outputs,
                "tiger_identifier": {
                    "manifest_sha256": sha256_file(manifest_path),
                    "mapping_sha256": sha256_file(mapping_path),
                    "token_capacities": [512, 512, 512, 1],
                },
                "time_split": {"train": "x", "valid": "y", "test": "z"},
            }
        )
        + "\n"
    )
    return directory


class PairedSftDataTest(unittest.TestCase):
    def test_paired_transform_keeps_context_and_removes_gid_from_nogid_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sid_mapping, sid_manifest = write_sid_source(root)
            gid_codes = root / "gid.npy"
            np.save(
                gid_codes,
                np.asarray(
                    [
                        [28, 29, 4, 12, 31, 24],
                        [28, 29, 4, 12, 31, 24],
                        [1, 2, 3, 4, 5, 6],
                    ],
                    dtype=np.uint8,
                ),
                allow_pickle=False,
            )
            gid_manifest = root / "gid_manifest.json"
            gid_manifest.write_text(json.dumps({"status": "completed"}) + "\n")
            gid_final = root / "gid_final"
            nogid_final = root / "nogid_final"
            build_final_identifiers(
                variant="a4_gid_parent",
                sid_mapping_path=sid_mapping,
                sid_manifest_path=sid_manifest,
                gid_codes_path=gid_codes,
                gid_manifest_path=gid_manifest,
                output_dir=gid_final,
            )
            build_final_identifiers(
                variant="a4_nogid",
                sid_mapping_path=sid_mapping,
                sid_manifest_path=sid_manifest,
                output_dir=nogid_final,
            )
            tiger_dir = write_tiger_identifier(root, ["p2", "p1", "p3"])
            source = write_source_sft(root, tiger_dir)
            result = build_paired_sft_data(
                source_sft_dir=source,
                tiger_identifier_dir=tiger_dir,
                final_identifier_dirs={
                    "a4_gid_parent": gid_final,
                    "a4_nogid": nogid_final,
                },
                output_dirs={
                    "a4_gid_parent": root / "gid_data",
                    "a4_nogid": root / "nogid_data",
                },
                progress_every=1,
            )
            self.assertEqual(result.stats["source_rows"], 3)
            gid_record = json.loads((root / "gid_data/train.jsonl").read_text())
            nogid_record = json.loads((root / "nogid_data/train.jsonl").read_text())
            gid_user = gid_record["messages"][0]["content"]
            nogid_user = nogid_record["messages"][0]["content"]
            self.assertIn("<USER_GID><G_w>", gid_user)
            self.assertIn("<USER_GID><G_w>", nogid_user)
            self.assertNotIn("POI_TIGER_ID", gid_user)
            self.assertNotIn("POI_TIGER_ID", nogid_user)
            self.assertIn("<POI_QGPRQK_ID>", gid_user)
            self.assertIn("<D_", nogid_record["messages"][1]["content"])
            self.assertIn("<G_", gid_record["messages"][1]["content"])
            self.assertNotIn("<G_", nogid_record["messages"][1]["content"])
            self.assertEqual(gid_record["sample_id"], nogid_record["sample_id"])


if __name__ == "__main__":
    unittest.main()
