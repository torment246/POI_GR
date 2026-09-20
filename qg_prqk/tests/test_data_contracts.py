from __future__ import annotations

import unittest
from pathlib import Path

from qg_prqk.config import load_config
from qg_prqk.data.contracts import (
    QGPRQKDataContractError,
    validate_aligned_row_counts,
    validate_final_pid,
    validate_poi_record,
    validate_train_record,
)


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(QG_ROOT / "configs/qg_prqk_1024x3.yaml")


def poi_record() -> dict:
    return {
        "address": "示例路1号",
        "alias": "示例别名",
        "area": 1,
        "category": "示例分类",
        "category_code": "010101",
        "city": "北京",
        "click_score": 0.5,
        "displayname": "示例POI",
        "lat": 39.9,
        "layer": 1,
        "lng": 116.4,
        "poi_id": "poi-1",
        "source_dt": "20260715",
        "text": "示例POI 示例路1号 示例别名",
    }


def train_record() -> dict:
    return {
        "sample_id": "sample-1",
        "order_id": "order-1",
        "searchid": "search-1",
        "split": "train",
        "target_poi_id": "poi-1",
        "requires_dedup": False,
        "target_pid_key": "0,1,2",
        "messages": [
            {"role": "user", "content": "<QUERY>示例</QUERY>\n<USER_GID>abc</USER_GID>"},
            {"role": "assistant", "content": "<G0>"},
        ],
    }


class DataContractTest(unittest.TestCase):
    def test_valid_poi(self) -> None:
        self.assertEqual(
            validate_poi_record(poi_record(), CONFIG.data_contracts, "poi"),
            "poi-1",
        )

    def test_missing_poi_field(self) -> None:
        record = poi_record()
        del record["address"]
        with self.assertRaisesRegex(QGPRQKDataContractError, "缺少 POI 字段"):
            validate_poi_record(record, CONFIG.data_contracts, "poi")

    def test_valid_train_row(self) -> None:
        self.assertEqual(
            validate_train_record(train_record(), CONFIG.data_contracts, "train"),
            ("示例", "poi-1"),
        )

    def test_rejects_non_train_row(self) -> None:
        record = train_record()
        record["split"] = "valid"
        with self.assertRaisesRegex(QGPRQKDataContractError, "不是 train"):
            validate_train_record(record, CONFIG.data_contracts, "train")

    def test_rejects_misaligned_rows(self) -> None:
        with self.assertRaisesRegex(QGPRQKDataContractError, "行数不一致"):
            validate_aligned_row_counts(
                poi_rows=3, embedding_rows=2, poi_id_rows=3
            )

    def test_final_pid_lengths_and_ranges(self) -> None:
        singleton = validate_final_pid(
            [0, 1, 2, 3, 4, 5, 10, 20, 30],
            False,
            CONFIG.identifier,
        )
        collision = validate_final_pid(
            [0, 1, 2, 3, 4, 5, 10, 20, 30, 7],
            True,
            CONFIG.identifier,
        )
        self.assertEqual(len(singleton), 9)
        self.assertEqual(len(collision), 10)
        with self.assertRaisesRegex(QGPRQKDataContractError, "长度必须为 10"):
            validate_final_pid(singleton, True, CONFIG.identifier)


if __name__ == "__main__":
    unittest.main()
