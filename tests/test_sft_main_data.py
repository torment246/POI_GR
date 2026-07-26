"""Synthetic tests for the order main-task SFT dataset."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.dedup_pid import sha256_file
from poi_gr.geohash_pid import encode_geohash
from poi_gr.sft_main_data import (
    SftDataValidationError,
    TimeSplit,
    build_sft_main_data,
    stable_sample_id,
)


class SftMainDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.split = TimeSplit(
            train_start=date(2026, 7, 1),
            train_end=date(2026, 7, 12),
            valid_date=date(2026, 7, 13),
            test_date=date(2026, 7, 14),
        )
        self.mapping_path, self.manifest_path = self.write_mapping()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_mapping(self) -> tuple[Path, Path]:
        mapping_path = self.root / "poi_pid_mapping.parquet"
        manifest_path = self.root / "final_pid_manifest.json"
        table = pa.table(
            {
                "poi_id": pa.array(["p-single", "p-dedup"]),
                "g1": pa.array([28, 28], type=pa.int32()),
                "g2": pa.array([29, 29], type=pa.int32()),
                "g3": pa.array([4, 4], type=pa.int32()),
                "g4": pa.array([15, 15], type=pa.int32()),
                "g5": pa.array([6, 7], type=pa.int32()),
                "g6": pa.array([6, 8], type=pa.int32()),
                "s1": pa.array([10, 20], type=pa.int32()),
                "s2": pa.array([11, 21], type=pa.int32()),
                "s3": pa.array([12, 22], type=pa.int32()),
                "requires_dedup": pa.array([False, True]),
                "dedup_code": pa.array([None, 3], type=pa.int16()),
                "final_pid_length": pa.array([9, 10], type=pa.int8()),
                "final_pid_key": pa.array(
                    [
                        "28-29-4-15-6-6|10-11-12",
                        "28-29-4-15-7-8|20-21-22|d3",
                    ]
                ),
            }
        )
        pq.write_table(table, mapping_path)
        mapping_hash = sha256_file(mapping_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "dedup-pid-v1",
                    "status": "completed",
                    "poi_count": 2,
                    "outputs": {
                        "poi_pid_mapping.parquet": {
                            "sha256": mapping_hash,
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return mapping_path, manifest_path

    @staticmethod
    def order(
        *,
        order_id: str,
        query: str | None,
        poi_id: str,
        create_time: str,
        longitude: float = 116.397,
        latitude: float = 39.908,
    ) -> dict[str, object]:
        return {
            "order_id": order_id,
            "searchid": f"search-{order_id}",
            "query": query,
            "disp_query": "未使用",
            "disp_lng": longitude,
            "disp_lat": latitude,
            "create_time": create_time,
            "source_dt": create_time[:10].replace("-", ""),
            "poi_id": poi_id,
            "dest_name": "未使用",
            "dest_lng": 116.4,
            "dest_lat": 39.9,
            "poi_displayname": "未使用",
            "rank": 1,
            "rank_sub": 0,
        }

    def write_orders(
        self, records: list[dict[str, object]], name: str = "orders"
    ) -> Path:
        orders_dir = self.root / name
        orders_dir.mkdir()
        (orders_dir / "_SUCCESS").touch()
        (orders_dir / "part-00000.json").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return orders_dir

    def valid_records(self) -> list[dict[str, object]]:
        duplicate = self.order(
            order_id="duplicate-order",
            query="重复订单",
            poi_id="p-single",
            create_time="2026-07-14 10:00:00",
        )
        return [
            self.order(
                order_id="train",
                query="  原始 Query  ",
                poi_id="p-single",
                create_time="2026-07-01 10:00:00",
            ),
            self.order(
                order_id="blank-null",
                query=None,
                poi_id="p-single",
                create_time="2026-07-12 10:00:00",
            ),
            self.order(
                order_id="blank-space",
                query=" \t ",
                poi_id="p-single",
                create_time="2026-07-12 11:00:00",
            ),
            self.order(
                order_id="valid",
                query="Dedup 目标",
                poi_id="p-dedup",
                create_time="2026-07-13 10:00:00",
            ),
            duplicate,
            dict(duplicate),
        ]

    def build_valid(self, output_name: str = "output"):
        orders_dir = self.write_orders(
            self.valid_records(), name=f"orders-{output_name}"
        )
        output_dir = self.root / output_name
        result = build_sft_main_data(
            orders_dir,
            self.mapping_path,
            self.manifest_path,
            output_dir,
            self.split,
        )
        return orders_dir, output_dir, result

    def read_jsonl(self, path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_filter_preserve_duplicates_split_and_messages(self) -> None:
        _, output_dir, result = self.build_valid()
        train = self.read_jsonl(output_dir / "train.jsonl")
        valid = self.read_jsonl(output_dir / "valid.jsonl")
        test = self.read_jsonl(output_dir / "test.jsonl")
        self.assertEqual((len(train), len(valid), len(test)), (1, 1, 2))
        self.assertEqual(result.stats["blank_query_count"], 2)
        self.assertEqual(result.stats["retained_sample_count"], 4)
        self.assertEqual(train[0]["messages"][0]["role"], "user")
        self.assertEqual(train[0]["messages"][1]["role"], "assistant")
        self.assertIn("<QUERY>  原始 Query  </QUERY>", train[0]["messages"][0]["content"])
        self.assertEqual(test[0]["order_id"], test[1]["order_id"])
        self.assertNotEqual(test[0]["sample_id"], test[1]["sample_id"])

    def test_user_geohash_and_pid_lengths(self) -> None:
        _, output_dir, _ = self.build_valid("tokens-output")
        train = self.read_jsonl(output_dir / "train.jsonl")[0]
        valid = self.read_jsonl(output_dir / "valid.jsonl")[0]
        geohash = encode_geohash(116.397, 39.908, 6)
        expected_user_gid = "".join(
            f"<G_{character}>" for character in geohash
        )
        self.assertIn(expected_user_gid, train["messages"][0]["content"])
        singleton = train["messages"][1]["content"]
        dedup = valid["messages"][1]["content"]
        self.assertEqual(singleton.count("<"), 9)
        self.assertEqual(dedup.count("<"), 10)
        self.assertNotIn("<D_-1>", singleton + dedup)
        self.assertTrue(dedup.endswith("<D_3>"))

    def test_sample_id_is_stable_and_source_location_based(self) -> None:
        orders_dir, output_dir, _ = self.build_valid("sample-id-output")
        relative = (
            next(path for path in orders_dir.iterdir() if path.name.startswith("part-"))
            .relative_to(orders_dir)
            .as_posix()
        )
        first = self.read_jsonl(output_dir / "train.jsonl")[0]
        self.assertEqual(first["sample_id"], stable_sample_id(relative, 1))
        self.assertEqual(len(first["sample_id"]), 64)

    def test_special_tokens_and_exact_output_set(self) -> None:
        _, output_dir, _ = self.build_valid("special-output")
        tokens = json.loads(
            (output_dir / "special_tokens.json").read_text(encoding="utf-8")
        )
        self.assertEqual(tokens["token_count"], 3620)
        self.assertNotIn("<D_-1>", tokens["additional_special_tokens"])
        self.assertIn("<S3_1023>", tokens["additional_special_tokens"])
        self.assertIn("<D_511>", tokens["additional_special_tokens"])
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {
                "train.jsonl",
                "valid.jsonl",
                "test.jsonl",
                "special_tokens.json",
                "manifest.json",
                "stats.json",
            },
        )

    def test_invalid_coordinates_and_unmatched_pid_are_reported(self) -> None:
        records = [
            self.order(
                order_id="bad-coordinate",
                query="非空",
                poi_id="p-single",
                create_time="2026-07-01 10:00:00",
                latitude=91.0,
            ),
            self.order(
                order_id="unmatched",
                query="非空",
                poi_id="missing-poi",
                create_time="2026-07-02 10:00:00",
            ),
        ]
        orders_dir = self.write_orders(records, "invalid-orders")
        output_dir = self.root / "invalid-output"
        with self.assertRaises(SftDataValidationError) as context:
            build_sft_main_data(
                orders_dir,
                self.mapping_path,
                self.manifest_path,
                output_dir,
                self.split,
            )
        self.assertEqual(
            context.exception.anomaly_counts["invalid_disp_lat"], 1
        )
        self.assertEqual(
            context.exception.anomaly_counts["unmatched_poi_id"], 1
        )
        self.assertFalse(output_dir.exists())

    def test_repeated_run_keeps_all_output_hashes(self) -> None:
        orders_dir = self.write_orders(self.valid_records(), "repeat-orders")
        output_dir = self.root / "repeat-output"
        build_sft_main_data(
            orders_dir,
            self.mapping_path,
            self.manifest_path,
            output_dir,
            self.split,
        )
        first = {
            path.name: sha256_file(path) for path in output_dir.iterdir()
        }
        build_sft_main_data(
            orders_dir,
            self.mapping_path,
            self.manifest_path,
            output_dir,
            self.split,
        )
        second = {
            path.name: sha256_file(path) for path in output_dir.iterdir()
        }
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
