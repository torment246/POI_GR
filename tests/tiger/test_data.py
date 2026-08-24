"""Synthetic tests for map-search-adapted TIGER SFT data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.pid.dedup import sha256_file
from poi_gr.methods.tiger.data import (
    HistoryWindow,
    TigerDataError,
    build_tiger_sft_data,
    stable_user_bucket,
    user_token,
)
from poi_gr.sft.data import TimeSplit


class TigerDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.split = TimeSplit(
            train_start=date(2026, 7, 1),
            train_end=date(2026, 7, 12),
            valid_date=date(2026, 7, 13),
            test_date=date(2026, 7, 14),
        )
        self.history_window = HistoryWindow(
            start=date(2026, 4, 1),
            end=date(2026, 6, 30),
        )
        self.tiger_id_dir = self.write_tiger_mapping()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_tiger_mapping(
        self,
        *,
        directory_name: str = "tiger-ids",
        base_codebook_sizes: tuple[int, int, int] = (1024, 1024, 1024),
    ) -> Path:
        tiger_id_dir = self.root / directory_name
        tiger_id_dir.mkdir()
        mapping_path = tiger_id_dir / "poi_tiger_id_mapping.parquet"
        table = pa.table(
            {
                "poi_id": pa.array(["poi-a", "poi-b", "poi-target"]),
                "s1": pa.array([10, 20, 30], type=pa.int32()),
                "s2": pa.array([11, 21, 31], type=pa.int32()),
                "s3": pa.array([12, 22, 32], type=pa.int32()),
                "base_sid_key": pa.array(["10-11-12", "20-21-22", "30-31-32"]),
                "base_sid_bucket_size": pa.array([1, 2, 2], type=pa.int32()),
                "has_semantic_collision": pa.array([False, True, True]),
                "collision_code": pa.array([0, 0, 1], type=pa.int32()),
                "tiger_id_key": pa.array(
                    ["10-11-12|c0", "20-21-22|c0", "30-31-32|c1"]
                ),
            }
        )
        pq.write_table(table, mapping_path)
        (tiger_id_dir / "tiger_id_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "tiger-item-identifier-v1",
                    "status": "completed",
                    "mapping": {
                        "path": "poi_tiger_id_mapping.parquet",
                        "rows": 3,
                        "sha256": sha256_file(mapping_path),
                        "poi_id_unique": True,
                        "tiger_id_key_unique": True,
                    },
                    "tiger_ids": {
                        "fixed_length": 4,
                        "token_order": ["S1", "S2", "S3", "C"],
                        "token_capacities": [*base_codebook_sizes, 2],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return tiger_id_dir

    @staticmethod
    def history_event(
        *,
        event_time: str,
        poi_id: str,
        query: str,
        longitude: float = 116.40,
        latitude: float = 39.90,
    ) -> dict[str, object]:
        suffix = (
            event_time.replace("-", "")
            .replace(":", "")
            .replace(" ", "")
            .replace("+", "")
        )
        return {
            "event_time": event_time,
            "order_id": f"history-order-{suffix}",
            "searchid": f"history-search-{suffix}",
            "query": query,
            "disp_lng": longitude,
            "disp_lat": latitude,
            "poi_id": poi_id,
            "dest_lng": 0.0,
            "dest_lat": 0.0,
            "dest_name": "不得进入 Prompt",
        }

    @classmethod
    def order(
        cls,
        *,
        order_id: str,
        create_time: str,
        history: list[dict[str, object]],
        passenger_id: str = "passenger-shared",
        query: str = "当前 Query",
    ) -> dict[str, object]:
        return {
            "order_id": order_id,
            "searchid": f"search-{order_id}",
            "passenger_id": passenger_id,
            "query": query,
            "disp_lng": 116.397,
            "disp_lat": 39.908,
            "create_time": create_time,
            "source_dt": create_time[:10].replace("-", ""),
            "poi_id": "poi-target",
            "history_length": len(history),
            "history_sequence": history,
            "dest_lng": 0.0,
            "dest_lat": 0.0,
            "dest_name": "不得进入 Prompt",
        }

    def write_orders(self, records: list[dict[str, object]]) -> Path:
        orders_dir = self.root / "orders"
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
        history = [
            self.history_event(
                event_time="2026-04-05T08:00:00.000+08:00",
                poi_id="poi-a",
                query="历史 A",
            ),
            self.history_event(
                event_time="2026-06-20T09:00:00.000+08:00",
                poi_id="poi-b",
                query="历史 B",
            ),
        ]
        return [
            self.order(
                order_id="train",
                create_time="2026-07-01 10:00:00",
                history=history,
            ),
            self.order(
                order_id="valid",
                create_time="2026-07-13 10:00:00",
                history=[],
            ),
            self.order(
                order_id="test",
                create_time="2026-07-14 10:00:00",
                history=history[:1],
            ),
        ]

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def build_valid(
        self,
        output_name: str = "output",
        *,
        max_retained_per_split: int | None = None,
    ):
        orders_dir = self.root / "orders"
        if not orders_dir.exists():
            orders_dir = self.write_orders(self.valid_records())
        output_dir = self.root / output_name
        result = build_tiger_sft_data(
            orders_dir,
            self.tiger_id_dir,
            output_dir,
            self.split,
            self.history_window,
            max_retained_per_split=max_retained_per_split,
        )
        return output_dir, result

    def test_split_history_coverage_and_exact_output_set(self) -> None:
        output_dir, result = self.build_valid()
        self.assertEqual(
            (
                result.stats["train_count"],
                result.stats["valid_count"],
                result.stats["test_count"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(result.stats["empty_history_count"], 1)
        self.assertEqual(result.stats["history_event_occurrence_count"], 3)
        self.assertAlmostEqual(result.stats["average_history_length"], 1.0)
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

    def test_prompt_contains_confirmed_fields_without_destination_leakage(self) -> None:
        output_dir, _ = self.build_valid("prompt-output")
        sample = self.read_jsonl(output_dir / "train.jsonl")[0]
        user_content = sample["messages"][0]["content"]
        self.assertTrue(user_content.startswith("<USER_ID><U_"))
        self.assertEqual(user_content.count("<EVENT>"), 2)
        self.assertEqual(user_content.count("<POI_TIGER_ID>"), 2)
        self.assertLess(
            user_content.index("<QUERY>历史 A</QUERY>"),
            user_content.index("<QUERY>历史 B</QUERY>"),
        )
        self.assertLess(
            user_content.index("</HISTORY>"),
            user_content.index("<CURRENT>"),
        )
        self.assertIn("<QUERY>当前 Query</QUERY>", user_content)
        self.assertNotIn("passenger-shared", user_content)
        self.assertNotIn("不得进入 Prompt", user_content)
        self.assertNotIn("2026-04-05", user_content)
        self.assertEqual(
            sample["messages"][1]["content"],
            (
                "<TARGET_POI><S1_30><S2_31>"
                "<S3_32><C_1></TARGET_POI>"
            ),
        )

    def test_user_hash_and_special_token_inventory_are_stable(self) -> None:
        self.assertEqual(
            stable_user_bucket("passenger-shared"),
            stable_user_bucket("passenger-shared"),
        )
        self.assertRegex(user_token("passenger-shared"), r"^<U_\d{4}>$")
        output_dir, _ = self.build_valid("token-output")
        payload = json.loads(
            (output_dir / "special_tokens.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["user_bucket_count"], 2000)
        self.assertEqual(payload["item_token_capacities"], [1024, 1024, 1024, 2])
        self.assertEqual(payload["token_count"], 5122)
        self.assertIn("<S3_1023>", payload["additional_special_tokens"])
        self.assertIn("<C_1>", payload["additional_special_tokens"])
        self.assertNotIn("<C_2>", payload["additional_special_tokens"])

    def test_asymmetric_base_codebooks_are_supported_when_explicit(self) -> None:
        asymmetric_id_dir = self.write_tiger_mapping(
            directory_name="asymmetric-tiger-ids",
            base_codebook_sizes=(64, 128, 256),
        )
        orders_dir = self.write_orders(self.valid_records())
        output_dir = self.root / "asymmetric-output"
        result = build_tiger_sft_data(
            orders_dir,
            asymmetric_id_dir,
            output_dir,
            self.split,
            self.history_window,
            expected_base_codebook_sizes=(64, 128, 256),
        )
        self.assertEqual(
            result.manifest["tiger_identifier"]["token_capacities"],
            [64, 128, 256, 2],
        )
        special_tokens = json.loads(
            (output_dir / "special_tokens.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            special_tokens["item_token_capacities"],
            [64, 128, 256, 2],
        )
        self.assertIn("<S3_255>", special_tokens["additional_special_tokens"])
        self.assertNotIn("<S1_64>", special_tokens["additional_special_tokens"])

    def test_base_codebook_mismatch_is_rejected(self) -> None:
        output_dir = self.root / "mismatched-codebook-output"
        with self.assertRaisesRegex(TigerDataError, "码本容量与预期不一致"):
            build_tiger_sft_data(
                self.write_orders(self.valid_records()),
                self.tiger_id_dir,
                output_dir,
                self.split,
                self.history_window,
                expected_base_codebook_sizes=(512, 1024, 2048),
            )
        self.assertFalse(output_dir.exists())

    def test_unsorted_history_is_rejected_without_output(self) -> None:
        history = [
            self.history_event(
                event_time="2026-06-20T09:00:00.000+08:00",
                poi_id="poi-a",
                query="后发生",
            ),
            self.history_event(
                event_time="2026-04-05T08:00:00.000+08:00",
                poi_id="poi-b",
                query="先发生",
            ),
        ]
        orders_dir = self.write_orders(
            [
                self.order(
                    order_id="invalid",
                    create_time="2026-07-01 10:00:00",
                    history=history,
                )
            ]
        )
        output_dir = self.root / "invalid-output"
        with self.assertRaisesRegex(TigerDataError, "没有按时间正序排列"):
            build_tiger_sft_data(
                orders_dir,
                self.tiger_id_dir,
                output_dir,
                self.split,
                self.history_window,
            )
        self.assertFalse(output_dir.exists())

    def test_missing_history_poi_mapping_is_rejected(self) -> None:
        history = [
            self.history_event(
                event_time="2026-05-01T09:00:00.000+08:00",
                poi_id="missing-poi",
                query="无映射",
            )
        ]
        orders_dir = self.write_orders(
            [
                self.order(
                    order_id="invalid-mapping",
                    create_time="2026-07-01 10:00:00",
                    history=history,
                )
            ]
        )
        with self.assertRaisesRegex(TigerDataError, "未匹配 TIGER identifier"):
            build_tiger_sft_data(
                orders_dir,
                self.tiger_id_dir,
                self.root / "missing-output",
                self.split,
                self.history_window,
            )

    def test_smoke_limit_and_repeated_build_are_deterministic(self) -> None:
        _, first = self.build_valid(
            "first-output",
            max_retained_per_split=1,
        )
        _, second = self.build_valid(
            "second-output",
            max_retained_per_split=1,
        )
        self.assertEqual(first.output_hashes, second.output_hashes)
        self.assertEqual(
            first.manifest["input"]["scan_mode"],
            "smoke_prefix_until_each_split_limit",
        )
        self.assertEqual(first.stats["retained_sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
