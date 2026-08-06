"""Lightweight tests for embedding retrieval metrics."""

from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.evaluate_embedding_retrieval import (
    EvaluationError,
    _pooling_description,
    compute_metrics,
    validate_reference_eval_order,
)


class EmbeddingRetrievalMetricsTest(unittest.TestCase):
    def test_pooling_description_supports_qwen_and_bge_modes(self) -> None:
        class Pooling:
            pooling_mode_cls_token = False
            pooling_mode_lasttoken = True

        self.assertEqual(_pooling_description([object(), Pooling()]), "last_token")

        Pooling.pooling_mode_cls_token = True
        Pooling.pooling_mode_lasttoken = False
        self.assertEqual(_pooling_description([Pooling()]), "cls")

    def test_metrics_and_query_length_buckets(self) -> None:
        records = [
            {"query": "北", "poi_id": "p0"},
            {"query": "北京站", "poi_id": "p1"},
            {"query": "首都国际机场", "poi_id": "p2"},
            {"query": "北京市朝阳区望京街道目的地", "poi_id": "p3"},
        ]
        target_indices = np.asarray([10, 20, 30, 40], dtype=np.int64)
        topk_indices = np.asarray(
            [
                [10, 11, 12],
                [21, 22, 20],
                [31, 32, 33],
                [41, 40, 42],
            ],
            dtype=np.int64,
        )

        target_ranks, metrics = compute_metrics(
            records,
            topk_indices,
            target_indices,
        )

        np.testing.assert_array_equal(target_ranks, [1, 3, -1, 2])
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["hit_at_1"], 0.25)
        self.assertEqual(metrics["hit_at_3"], 0.75)
        self.assertEqual(metrics["hit_at_20"], 0.75)
        self.assertAlmostEqual(metrics["mrr_at_10"], (1 + 1 / 3 + 0 + 1 / 2) / 4)
        self.assertEqual(metrics["query_length_buckets"]["1-2"]["samples"], 1)
        self.assertEqual(metrics["query_length_buckets"]["3-5"]["samples"], 1)
        self.assertEqual(metrics["query_length_buckets"]["6-10"]["samples"], 1)
        self.assertEqual(metrics["query_length_buckets"][">10"]["samples"], 1)

    def test_reference_order_must_match_e1_mapping(self) -> None:
        records = [
            {"order_id": "o1", "poi_id": "p1"},
            {"order_id": "o2", "poi_id": "p2"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mapping_path = root / "rows.jsonl"
            mapping_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"row_index": 0, "order_id": "o1", "target_poi_id": "p1"},
                        {"row_index": 1, "order_id": "o2", "target_poi_id": "p2"},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            mapping_sha256 = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "model": "qwen3_0.6b",
                        "instruction": "none",
                        "gate_zero": {"eval_sha256": "eval-sha"},
                        "outputs": {
                            "query_mapping": str(mapping_path),
                            "query_mapping_sha256": mapping_sha256,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result, _ = validate_reference_eval_order(
                records,
                "eval-sha",
                manifest_path,
            )
            self.assertEqual(result["order_id_rows_match"], 2)

            records.reverse()
            with self.assertRaisesRegex(EvaluationError, "order_id 顺序"):
                validate_reference_eval_order(records, "eval-sha", manifest_path)


if __name__ == "__main__":
    unittest.main()
