"""Synthetic tests for paired TIGER PID-order SFT data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger.pid_order_data import (  # noqa: E402
    TigerPidOrderDataError,
    build_tiger_pid_order_sft_data,
    pid_content,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402


class TigerPidOrderDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tiger_dir = self._write_tiger_mapping()
        self.pid_mapping, self.pid_manifest = self._write_pid_mapping()
        self.source_dir = self._write_source_sft()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_tiger_mapping(self) -> Path:
        path = self.root / "tiger_ids"
        path.mkdir()
        mapping = path / "poi_tiger_id_mapping.parquet"
        pq.write_table(
            pa.table(
                {
                    "poi_id": ["poi-a", "poi-b", "poi-target"],
                    "s1": pa.array([10, 30, 30], type=pa.int32()),
                    "s2": pa.array([11, 31, 31], type=pa.int32()),
                    "s3": pa.array([12, 32, 32], type=pa.int32()),
                    "collision_code": pa.array([0, 0, 1], type=pa.int32()),
                }
            ),
            mapping,
        )
        (path / "tiger_id_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "tiger-item-identifier-v1",
                    "status": "completed",
                    "source": {"sid_manifest_sha256": "same-tiger-sid"},
                    "mapping": {
                        "rows": 3,
                        "sha256": sha256_file(mapping),
                    },
                    "tiger_ids": {
                        "fixed_length": 4,
                        "token_order": ["S1", "S2", "S3", "C"],
                        "token_capacities": [1024, 1024, 1024, 2],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _write_pid_mapping(self) -> tuple[Path, Path]:
        path = self.root / "pid"
        path.mkdir()
        base_manifest = path / "base_pid_manifest.json"
        base_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "geohash-pid-v1",
                    "sid_source": {
                        "manifest_sha256": "same-tiger-sid",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        mapping = path / "poi_pid_mapping.parquet"
        gid_a = [1, 2, 3, 4, 5, 6]
        gid_collision = [7, 8, 9, 10, 11, 12]
        rows = [
            ("poi-a", gid_a, [10, 11, 12], None, False, 9),
            ("poi-b", gid_collision, [30, 31, 32], 0, True, 10),
            ("poi-target", gid_collision, [30, 31, 32], 1, True, 10),
        ]
        columns: dict[str, object] = {
            "poi_id": [row[0] for row in rows],
            **{
                f"g{index + 1}": pa.array(
                    [row[1][index] for row in rows], type=pa.int32()
                )
                for index in range(6)
            },
            **{
                f"s{index + 1}": pa.array(
                    [row[2][index] for row in rows], type=pa.int32()
                )
                for index in range(3)
            },
            "requires_dedup": [row[4] for row in rows],
            "dedup_code": pa.array([row[3] for row in rows], type=pa.int16()),
            "final_pid_length": pa.array(
                [row[5] for row in rows], type=pa.int8()
            ),
            "final_pid_key": [
                "1-2-3-4-5-6|10-11-12",
                "7-8-9-10-11-12|30-31-32|d0",
                "7-8-9-10-11-12|30-31-32|d1",
            ],
        }
        pq.write_table(pa.table(columns), mapping)
        manifest = path / "final_pid_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "dedup-pid-v1",
                    "status": "completed",
                    "poi_count": 3,
                    "base_pid_source": {"manifest": str(base_manifest)},
                    "dedup_assignment": {"dedup_token_capacity": 512},
                    "outputs": {
                        "poi_pid_mapping.parquet": {
                            "sha256": sha256_file(mapping),
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return mapping, manifest

    def _source_sample(self, split: str) -> dict[str, object]:
        prompt = (
            "<USER_ID><U_0001></USER_ID>\n"
            "<HISTORY>\n"
            "<EVENT>\n<USER_GID><G_w><G_x><G_4><G_g><G_0><G_0></USER_GID>\n"
            "<QUERY>历史 A</QUERY>\n"
            "<POI_TIGER_ID><S1_10><S2_11><S3_12><C_0></POI_TIGER_ID>\n"
            "</EVENT>\n"
            "<EVENT>\n<USER_GID><G_w><G_x><G_4><G_g><G_0><G_1></USER_GID>\n"
            "<QUERY>历史 B</QUERY>\n"
            "<POI_TIGER_ID><S1_30><S2_31><S3_32><C_0></POI_TIGER_ID>\n"
            "</EVENT>\n</HISTORY>\n"
            "<CURRENT>\n<USER_GID><G_w><G_x><G_4><G_g><G_0><G_2></USER_GID>\n"
            "<QUERY>当前 Query</QUERY>\n</CURRENT>"
        )
        return {
            "sample_id": f"sample-{split}",
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": (
                        "<TARGET_POI><S1_30><S2_31>"
                        "<S3_32><C_1></TARGET_POI>"
                    ),
                },
            ],
            "user_token": "<U_0001>",
            "order_id": f"order-{split}",
            "searchid": f"search-{split}",
            "target_poi_id": "poi-target",
            "target_tiger_id_key": "30-31-32|c1",
            "history_length": 2,
            "split": split,
        }

    def _write_source_sft(self) -> Path:
        path = self.root / "source"
        path.mkdir()
        outputs: dict[str, dict[str, object]] = {}
        for split in ("train", "valid", "test"):
            target = path / f"{split}.jsonl"
            target.write_text(
                json.dumps(self._source_sample(split), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            outputs[target.name] = {"rows": 1, "sha256": sha256_file(target)}
        tiger_manifest = self.tiger_dir / "tiger_id_manifest.json"
        tiger_mapping = self.tiger_dir / "poi_tiger_id_mapping.parquet"
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "tiger-map-search-sft-data-v1",
                    "status": "completed",
                    "tiger_identifier": {
                        "mapping_sha256": sha256_file(tiger_mapping),
                        "manifest_sha256": sha256_file(tiger_manifest),
                    },
                    "outputs": outputs,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _read_one(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _build(self, suffix: str):
        outputs = {
            "gid_sid": self.root / f"gid-sid-{suffix}",
            "sid_gid": self.root / f"sid-gid-{suffix}",
        }
        result = build_tiger_pid_order_sft_data(
            self.source_dir,
            self.tiger_dir,
            self.pid_mapping,
            self.pid_manifest,
            outputs,
        )
        return outputs, result

    def test_paired_outputs_only_swap_gid_and_sid(self) -> None:
        outputs, result = self._build("paired")
        gid_first = self._read_one(outputs["gid_sid"] / "train.jsonl")
        sid_first = self._read_one(outputs["sid_gid"] / "train.jsonl")
        for field in (
            "sample_id",
            "order_id",
            "searchid",
            "target_poi_id",
            "history_length",
            "requires_dedup",
        ):
            self.assertEqual(gid_first[field], sid_first[field])
        self.assertNotIn("target_tiger_id_key", gid_first)
        self.assertIn("<POI_PID>", gid_first["messages"][0]["content"])
        self.assertNotIn("<POI_TIGER_ID>", gid_first["messages"][0]["content"])
        self.assertEqual(
            gid_first["messages"][1]["content"],
            (
                "<TARGET_POI><G_7><G_8><G_9><G_b><G_c><G_d>"
                "<S1_30><S2_31><S3_32><D_1></TARGET_POI>"
            ),
        )
        self.assertEqual(
            sid_first["messages"][1]["content"],
            (
                "<TARGET_POI><S1_30><S2_31><S3_32>"
                "<G_7><G_8><G_9><G_b><G_c><G_d><D_1></TARGET_POI>"
            ),
        )
        self.assertTrue(
            result.manifests["gid_sid"]["fairness_contract"]
            ["only_poi_pid_gid_sid_order_differs"]
        )

    def test_dedup_is_optional_but_always_last(self) -> None:
        singleton = [1, 2, 3, 4, 5, 6, 10, 11, 12]
        colliding = [7, 8, 9, 10, 11, 12, 30, 31, 32]
        self.assertNotIn("<D_", pid_content(singleton, -1, order="gid_sid"))
        for order in ("gid_sid", "sid_gid"):
            self.assertTrue(
                pid_content(colliding, 1, order=order).endswith("<D_1>")
            )

    def test_two_full_builds_are_deterministic(self) -> None:
        _, first = self._build("first")
        _, second = self._build("second")
        self.assertEqual(first.output_hashes, second.output_hashes)

    def test_wrong_tiger_sid_source_is_rejected(self) -> None:
        manifest = json.loads(self.pid_manifest.read_text(encoding="utf-8"))
        base_path = Path(manifest["base_pid_source"]["manifest"])
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base["sid_source"]["manifest_sha256"] = "different"
        base_path.write_text(json.dumps(base) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(
            TigerPidOrderDataError, "不是由当前 TIGER 三层 SID 构建"
        ):
            self._build("mismatch")


if __name__ == "__main__":
    unittest.main()
