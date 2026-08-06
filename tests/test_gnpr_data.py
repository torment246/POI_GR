"""Synthetic tests for map-search-adapted GNPR SFT data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.dedup_pid import sha256_file
from poi_gr.methods.gnpr_data import (
    GnprDataError,
    HistoryWindow,
    build_gnpr_sft_data,
)
from poi_gr.sft_main_data import TimeSplit


class GnprDataTest(unittest.TestCase):
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
        self.gnpr_id_dir = self.write_gnpr_mapping()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_gnpr_mapping(self) -> Path:
        output_dir = self.root / "gnpr-ids"
        output_dir.mkdir()
        poi_ids_path = output_dir / "poi_ids.jsonl"
        ids_path = output_dir / "gnpr_ids.npy"
        mapping_path = output_dir / "poi_gnpr_id_mapping.parquet"
        poi_ids_path.write_text(
            '"100"\n"200"\n"300"\n',
            encoding="utf-8",
        )
        np.save(
            ids_path,
            np.asarray(
                [
                    [10, 11, 12, -1],
                    [20, 21, 22, 0],
                    [20, 21, 22, 1],
                ],
                dtype=np.int32,
            ),
            allow_pickle=False,
        )
        mapping_path.write_bytes(b"synthetic-mapping")
        (output_dir / "gnpr_id_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "gnpr-poi-identifier-v1",
                    "status": "completed",
                    "poi_ids": {
                        "path": poi_ids_path.name,
                        "rows": 3,
                        "sha256": sha256_file(poi_ids_path),
                        "unique": True,
                    },
                    "gnpr_ids": {
                        "path": ids_path.name,
                        "shape": [3, 4],
                        "dtype": "int32",
                        "sha256": sha256_file(ids_path),
                        "codebook_capacities": [512, 512, 512],
                        "dedup_token_capacity": 2,
                        "serialized_length": {
                            "singleton": 3,
                            "collision": 4,
                        },
                    },
                    "mapping": {
                        "path": mapping_path.name,
                        "rows": 3,
                        "sha256": sha256_file(mapping_path),
                        "poi_id_unique": True,
                        "gnpr_id_key_unique": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return output_dir

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
            "dest_name": "不得进入 Prompt",
        }

    @classmethod
    def order(
        cls,
        *,
        order_id: str,
        create_time: str,
        target_poi_id: str,
        history: list[dict[str, object]],
        query: str = "当前 Query",
    ) -> dict[str, object]:
        return {
            "order_id": order_id,
            "searchid": f"search-{order_id}",
            "passenger_id": "不得进入 Prompt 的用户",
            "query": query,
            "disp_lng": 116.397,
            "disp_lat": 39.908,
            "create_time": create_time,
            "source_dt": create_time[:10].replace("-", ""),
            "poi_id": target_poi_id,
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
                poi_id="100",
                query="历史 A",
            ),
            self.history_event(
                event_time="2026-06-20T09:00:00.000+08:00",
                poi_id="200",
                query="历史 B",
            ),
        ]
        return [
            self.order(
                order_id="train",
                create_time="2026-07-01 10:00:00",
                target_poi_id="300",
                history=history,
            ),
            self.order(
                order_id="valid",
                create_time="2026-07-13 10:00:00",
                target_poi_id="100",
                history=[],
            ),
            self.order(
                order_id="test",
                create_time="2026-07-14 10:00:00",
                target_poi_id="200",
                history=history[:1],
            ),
        ]

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def build_valid(self, output_name: str = "output"):
        orders_dir = self.root / "orders"
        if not orders_dir.exists():
            orders_dir = self.write_orders(self.valid_records())
        output_dir = self.root / output_name
        result = build_gnpr_sft_data(
            orders_dir,
            self.gnpr_id_dir,
            output_dir,
            self.split,
            self.history_window,
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

    def test_prompt_omits_time_user_and_destination_fields(self) -> None:
        output_dir, _ = self.build_valid("prompt-output")
        sample = self.read_jsonl(output_dir / "train.jsonl")[0]
        user_content = sample["messages"][0]["content"]
        self.assertTrue(user_content.startswith("<HISTORY>"))
        self.assertEqual(user_content.count("<EVENT>"), 2)
        self.assertEqual(user_content.count("<POI_GNPR_ID>"), 2)
        self.assertIn("<a_10><b_11><c_12>", user_content)
        self.assertIn("<a_20><b_21><c_22><d_0>", user_content)
        self.assertNotIn("<USER_ID>", user_content)
        self.assertNotIn("passenger", user_content)
        self.assertNotIn("2026-", user_content)
        self.assertNotIn("不得进入 Prompt", user_content)
        self.assertEqual(
            sample["messages"][1]["content"],
            "<TARGET_POI><a_20><b_21><c_22><d_1></TARGET_POI>",
        )
        self.assertNotIn("user_token", sample)

    def test_singleton_target_and_special_token_inventory(self) -> None:
        output_dir, _ = self.build_valid("token-output")
        valid = self.read_jsonl(output_dir / "valid.jsonl")[0]
        self.assertEqual(
            valid["messages"][1]["content"],
            "<TARGET_POI><a_10><b_11><c_12></TARGET_POI>",
        )
        tokens = json.loads(
            (output_dir / "special_tokens.json").read_text(encoding="utf-8")
        )
        self.assertEqual(tokens["user_tokens"], [])
        self.assertTrue(tokens["conditional_dedup_token"])
        self.assertEqual(tokens["item_token_capacities"], [512, 512, 512, 2])
        self.assertEqual(tokens["token_count"], 1584)
        self.assertIn("<d_1>", tokens["additional_special_tokens"])
        self.assertNotIn("<d_2>", tokens["additional_special_tokens"])

    def test_unsorted_history_is_rejected_without_output(self) -> None:
        history = [
            self.history_event(
                event_time="2026-06-20T09:00:00.000+08:00",
                poi_id="100",
                query="后发生",
            ),
            self.history_event(
                event_time="2026-04-05T08:00:00.000+08:00",
                poi_id="200",
                query="先发生",
            ),
        ]
        orders_dir = self.write_orders(
            [
                self.order(
                    order_id="invalid",
                    create_time="2026-07-01 10:00:00",
                    target_poi_id="300",
                    history=history,
                )
            ]
        )
        output_dir = self.root / "invalid-output"
        with self.assertRaisesRegex(GnprDataError, "没有按时间正序排列"):
            build_gnpr_sft_data(
                orders_dir,
                self.gnpr_id_dir,
                output_dir,
                self.split,
                self.history_window,
            )
        self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
