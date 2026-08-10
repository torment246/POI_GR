"""Synthetic tests for causal GenPOI SFT data construction."""

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
from poi_gr.methods.genpoi.data import (
    GenpoiDataError,
    HistoryWindow,
    build_genpoi_sft_data,
)
from poi_gr.sft.data import TimeSplit


class GenpoiDataTest(unittest.TestCase):
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
        self.mapping_path, self.manifest_path = self.write_mapping()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_mapping(self) -> tuple[Path, Path]:
        mapping_path = self.root / "poi_pid_mapping.parquet"
        manifest_path = self.root / "final_pid_manifest.json"
        table = pa.table(
            {
                "poi_id": pa.array(["poi-a", "poi-b", "poi-target"]),
                "g1": pa.array([28, 28, 28], type=pa.int32()),
                "g2": pa.array([29, 29, 29], type=pa.int32()),
                "g3": pa.array([4, 4, 4], type=pa.int32()),
                "g4": pa.array([15, 15, 15], type=pa.int32()),
                "g5": pa.array([6, 7, 8], type=pa.int32()),
                "g6": pa.array([6, 7, 8], type=pa.int32()),
                "s1": pa.array([10, 20, 30], type=pa.int32()),
                "s2": pa.array([11, 21, 31], type=pa.int32()),
                "s3": pa.array([12, 22, 32], type=pa.int32()),
                "requires_dedup": pa.array([False, True, False]),
                "dedup_code": pa.array([None, 3, None], type=pa.int16()),
                "final_pid_length": pa.array([9, 10, 9], type=pa.int8()),
                "final_pid_key": pa.array(
                    [
                        "28-29-4-15-6-6|10-11-12",
                        "28-29-4-15-7-7|20-21-22|d3",
                        "28-29-4-15-8-8|30-31-32",
                    ]
                ),
            }
        )
        pq.write_table(table, mapping_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "dedup-pid-v1",
                    "status": "completed",
                    "poi_count": 3,
                    "outputs": {
                        "poi_pid_mapping.parquet": {
                            "sha256": sha256_file(mapping_path),
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return mapping_path, manifest_path

    @staticmethod
    def history_event(
        *,
        event_time: str,
        poi_id: str,
        query: str,
        longitude: float = 116.40,
        latitude: float = 39.90,
    ) -> dict[str, object]:
        suffix = event_time.replace("-", "").replace(":", "").replace(" ", "")
        return {
            "event_time": event_time,
            "order_id": f"history-order-{suffix}",
            "searchid": f"history-search-{suffix}",
            "query": query,
            "disp_lng": longitude,
            "disp_lat": latitude,
            "poi_id": poi_id,
        }

    @classmethod
    def order(
        cls,
        *,
        order_id: str,
        create_time: str,
        history: list[dict[str, object]],
        query: str = "当前 Query",
    ) -> dict[str, object]:
        return {
            "order_id": order_id,
            "searchid": f"search-{order_id}",
            "passenger_id": f"user-{order_id}",
            "query": query,
            "disp_lng": 116.397,
            "disp_lat": 39.908,
            "create_time": create_time,
            "source_dt": create_time[:10].replace("-", ""),
            "poi_id": "poi-target",
            "history_length": len(history),
            "history_sequence": history,
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

    def build_valid(self, output_name: str = "output"):
        orders_dir = self.root / "orders"
        if not orders_dir.exists():
            orders_dir = self.write_orders(self.valid_records())
        output_dir = self.root / output_name
        result = build_genpoi_sft_data(
            orders_dir,
            self.mapping_path,
            self.manifest_path,
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
        self.assertEqual(result.stats["nonempty_history_count"], 2)
        self.assertEqual(result.stats["history_event_occurrence_count"], 3)
        self.assertAlmostEqual(result.stats["average_history_length"], 1.0)
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            set(
                (
                    "train.jsonl",
                    "valid.jsonl",
                    "test.jsonl",
                    "special_tokens.json",
                    "manifest.json",
                    "stats.json",
                )
            ),
        )

    def test_prompt_matches_genpoi_field_order(self) -> None:
        output_dir, _ = self.build_valid("prompt-output")
        sample = self.read_jsonl(output_dir / "train.jsonl")[0]
        user_content = sample["messages"][0]["content"]
        self.assertTrue(user_content.startswith("<HISTORY>\n<USER_GID>"))
        self.assertEqual(user_content.count("<POI_PID>"), 2)
        self.assertLess(
            user_content.index("<QUERY>历史 A</QUERY>"),
            user_content.index("<QUERY>历史 B</QUERY>"),
        )
        self.assertLess(
            user_content.index("</HISTORY>"),
            user_content.index("<CURRENT>"),
        )
        self.assertTrue(user_content.endswith("</CURRENT>"))
        self.assertIn("<QUERY>当前 Query</QUERY>", user_content)
        self.assertNotIn("2026-04-05", user_content)
        self.assertEqual(sample["history_length"], 2)
        self.assertEqual(sample["messages"][1]["role"], "assistant")
        self.assertNotIn("<POI_PID>", sample["messages"][1]["content"])

    def test_special_tokens_follow_beijing_genpoi_1024_codebook(self) -> None:
        output_dir, _ = self.build_valid("token-output")
        payload = json.loads(
            (output_dir / "special_tokens.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["sid_codebook_size"], 1024)
        self.assertEqual(payload["token_count"], 3626)
        self.assertIn("<HISTORY>", payload["additional_special_tokens"])
        self.assertIn("<S3_1023>", payload["additional_special_tokens"])
        self.assertNotIn("<S3_1024>", payload["additional_special_tokens"])

    def test_future_or_unsorted_history_is_rejected_without_output(self) -> None:
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
        with self.assertRaisesRegex(
            GenpoiDataError, "没有按时间正序排列"
        ):
            build_genpoi_sft_data(
                orders_dir,
                self.mapping_path,
                self.manifest_path,
                output_dir,
                self.split,
                self.history_window,
            )
        self.assertFalse(output_dir.exists())

    def test_history_length_mismatch_is_rejected(self) -> None:
        record = self.order(
            order_id="mismatch",
            create_time="2026-07-01 10:00:00",
            history=[],
        )
        record["history_length"] = 1
        orders_dir = self.write_orders([record])
        with self.assertRaisesRegex(
            GenpoiDataError, "实际长度不一致"
        ):
            build_genpoi_sft_data(
                orders_dir,
                self.mapping_path,
                self.manifest_path,
                self.root / "mismatch-output",
                self.split,
                self.history_window,
            )

    def test_repeated_build_is_deterministic(self) -> None:
        _, first = self.build_valid("first-output")
        _, second = self.build_valid("second-output")
        self.assertEqual(first.output_hashes, second.output_hashes)


if __name__ == "__main__":
    unittest.main()
