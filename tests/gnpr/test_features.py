"""Synthetic contract tests for GNPR-SID POI features."""

from __future__ import annotations

import sys
import unittest
import importlib.util
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr.features import (  # noqa: E402
    GnprFeatureError,
    build_gnpr_feature_dataset,
    encode_plus_code_region,
    extract_gnpr_interactions,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "gnpr" / "build_poi_features.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_gnpr_poi_features", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


class GnprFeatureTest(unittest.TestCase):
    @staticmethod
    def history_event(
        event_time: str,
        poi_id: str,
        suffix: str,
    ) -> dict[str, str]:
        return {
            "event_time": event_time,
            "order_id": f"history-order-{suffix}",
            "searchid": f"history-search-{suffix}",
            "poi_id": poi_id,
        }

    @classmethod
    def order(
        cls,
        *,
        order_id: str,
        passenger_id: str,
        source_dt: str,
        create_time: str,
        poi_id: str,
        history: list[dict[str, str]],
    ) -> dict[str, object]:
        return {
            "order_id": order_id,
            "searchid": f"search-{order_id}",
            "passenger_id": passenger_id,
            "source_dt": source_dt,
            "birth_time": create_time,
            "create_time": create_time,
            "poi_id": poi_id,
            "history_length": len(history),
            "history_sequence": history,
        }

    def test_fixed_history_dedup_train_targets_and_four_features(self) -> None:
        user_one_history = [
            self.history_event(
                "2026-04-01T08:10:00+08:00",
                "poi-a",
                "a-8",
            ),
            self.history_event(
                "2026-05-01T09:20:00+08:00",
                "poi-b",
                "b-9",
            ),
        ]
        user_two_history = [
            self.history_event(
                "2026-06-01T08:30:00+08:00",
                "poi-a",
                "a-8-u2",
            )
        ]
        orders = [
            self.order(
                order_id="train-u1",
                passenger_id="user-1",
                source_dt="20260701",
                create_time="2026-07-01 10:00:00",
                poi_id="poi-a",
                history=user_one_history,
            ),
            self.order(
                order_id="valid-u1",
                passenger_id="user-1",
                source_dt="20260713",
                create_time="2026-07-13 22:00:00",
                poi_id="poi-c",
                history=user_one_history,
            ),
            self.order(
                order_id="test-u2",
                passenger_id="user-2",
                source_dt="20260714",
                create_time="2026-07-14 23:00:00",
                poi_id="poi-c",
                history=user_two_history,
            ),
        ]

        interactions, extraction_stats = extract_gnpr_interactions(
            orders,
            train_start=date(2026, 7, 1),
            train_end=date(2026, 7, 12),
        )
        self.assertEqual(extraction_stats["history_user_count"], 2)
        self.assertEqual(extraction_stats["history_event_count"], 3)
        self.assertEqual(extraction_stats["target_train_event_count"], 1)
        self.assertEqual(len(interactions), 4)

        poi_records = [
            {
                "poi_id": "poi-a",
                "category_code": "food",
                "lng": 116.4074,
                "lat": 39.9042,
            },
            {
                "poi_id": "poi-b",
                "category_code": "office",
                "lng": 116.4500,
                "lat": 39.9200,
            },
            {
                "poi_id": "poi-c",
                "category_code": "office",
                "lng": 116.5000,
                "lat": 39.9500,
            },
        ]
        dataset = build_gnpr_feature_dataset(poi_records, interactions)
        row_by_poi = {row.poi_id: row for row in dataset.rows}

        self.assertEqual(row_by_poi["poi-a"].top_visit_hours, (8, 10))
        self.assertEqual(row_by_poi["poi-a"].top_visitor_ids, ("user-1", "user-2"))
        self.assertEqual(row_by_poi["poi-a"].interaction_count, 3)
        self.assertEqual(row_by_poi["poi-b"].top_visit_hours, (9,))
        self.assertEqual(row_by_poi["poi-c"].top_visit_hours, ())
        self.assertEqual(row_by_poi["poi-c"].interaction_count, 0)
        self.assertEqual(dataset.stats["behavior_poi_count"], 2)
        self.assertEqual(dataset.category_vocab, {"food": 0, "office": 1})
        self.assertEqual(dataset.user_vocab, {"user-1": 0, "user-2": 1})
        self.assertTrue(row_by_poi["poi-a"].plus_code_region.endswith("+"))

    def test_rejects_inconsistent_fixed_history_for_same_user(self) -> None:
        first = [
            self.history_event("2026-04-01 08:00:00", "poi-a", "first")
        ]
        second = [
            self.history_event("2026-04-02 08:00:00", "poi-a", "second")
        ]
        orders = [
            self.order(
                order_id="one",
                passenger_id="user-1",
                source_dt="20260701",
                create_time="2026-07-01 10:00:00",
                poi_id="poi-a",
                history=first,
            ),
            self.order(
                order_id="two",
                passenger_id="user-1",
                source_dt="20260702",
                create_time="2026-07-02 10:00:00",
                poi_id="poi-a",
                history=second,
            ),
        ]
        with self.assertRaisesRegex(GnprFeatureError, "history_sequence 不一致"):
            extract_gnpr_interactions(
                orders,
                train_start=date(2026, 7, 1),
                train_end=date(2026, 7, 12),
            )

    def test_plus_code_uses_latitude_longitude_order(self) -> None:
        self.assertEqual(
            encode_plus_code_region(116.4074, 39.9042, code_length=6),
            "8PFRWC00+",
        )
        with self.assertRaisesRegex(GnprFeatureError, "Plus Code"):
            encode_plus_code_region(116.4074, 39.9042, code_length=7)
        self.assertEqual(
            SCRIPT_MODULE._encode_plus_code6(116.4074, 39.9042),
            "8PFRWC00+",
        )


if __name__ == "__main__":
    unittest.main()
