"""Synthetic tests for paper-aligned TIGER invalid-ID evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger.eval import (  # noqa: E402
    TigerCandidate,
    TigerIdIndex,
    TigerLegalPathConstraint,
    TigerTokenIds,
    empty_bucket_metrics,
    empty_metrics,
    finalize_bucket_metrics,
    finalize_metrics,
    parse_generated_candidate,
    parse_target_codes,
    update_bucket_metrics,
    update_metrics,
)
from scripts.tiger.evaluate_retrieval import (  # noqa: E402
    JsonlRecordSequence,
    build_full_split_view,
    validate_candidate_trace_row,
)


class TigerEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        codes = ((1, 2, 3, 0), (1, 2, 3, 1), (7, 8, 9, 0))
        keys = np.asarray([TigerIdIndex.pack(value) for value in codes], dtype=np.int64)
        order = np.argsort(keys)
        self.index = TigerIdIndex(
            sorted_keys=keys[order],
            sorted_rows=order.astype(np.int64),
            poi_ids=pa.array(["poi-a", "poi-b", "poi-c"]),
        )

    def test_four_code_lookup_is_unique(self) -> None:
        self.assertEqual(self.index.lookup((1, 2, 3, 0)), 0)
        self.assertEqual(self.index.lookup((1, 2, 3, 1)), 1)
        self.assertEqual(self.index.lookup((1, 2, 4, 0)), -1)

    def test_asymmetric_capacities_use_mixed_radix_lookup(self) -> None:
        capacities = (2, 3, 4, 5)
        codes = ((0, 2, 3, 4), (1, 0, 0, 0))
        keys = np.asarray(
            [TigerIdIndex.pack(value, capacities) for value in codes],
            dtype=np.int64,
        )
        order = np.argsort(keys)
        index = TigerIdIndex(
            sorted_keys=keys[order],
            sorted_rows=order.astype(np.int64),
            poi_ids=pa.array(["poi-a", "poi-b"]),
            token_capacities=capacities,
        )
        self.assertEqual(index.lookup(codes[0]), 0)
        self.assertEqual(index.lookup(codes[1]), 1)
        self.assertEqual(index.lookup((1, 0, 0, 5)), -1)

    def test_target_serialization_is_strict(self) -> None:
        self.assertEqual(
            parse_target_codes("<TARGET_POI><S1_1><S2_2><S3_3><C_4></TARGET_POI>"),
            (1, 2, 3, 4),
        )
        with self.assertRaisesRegex(ValueError, "严格 TIGER"):
            parse_target_codes("<S1_1><S2_2><S3_3><C_4>")

    def test_invalid_beam_keeps_its_rank_slot(self) -> None:
        metrics = empty_metrics()
        candidates = (
            TigerCandidate(None, None, -0.1, "identifier_not_in_corpus"),
            TigerCandidate((1, 2, 3, 0), 0, -0.2, None),
        )
        rank = update_metrics(metrics, target_row=0, candidates=candidates)
        result = finalize_metrics(metrics)
        self.assertEqual(rank, 2)
        self.assertEqual(result["hr@1"], 0.0)
        self.assertEqual(result["hr@3"], 1.0)
        self.assertEqual(result["invalid_id_rate"], 0.5)

    def test_bucket_metrics_collapse_duplicate_full_ids(self) -> None:
        metrics = empty_bucket_metrics()
        candidates = (
            TigerCandidate((1, 2, 3, 99), None, -0.1, "identifier_not_in_corpus"),
            TigerCandidate((1, 2, 3, 0), 0, -0.2, None),
            TigerCandidate((7, 8, 9, 0), 2, -0.3, None),
            TigerCandidate(None, None, -0.4, "invalid_structure"),
        )
        ranking = update_bucket_metrics(
            metrics,
            target_codes=(1, 2, 3, 1),
            candidates=candidates,
            index=self.index,
        )
        result = finalize_bucket_metrics(metrics)

        self.assertEqual(self.index.bucket_size((1, 2, 3)), 2)
        self.assertEqual(self.index.bucket_size((1, 2, 4)), 0)
        self.assertEqual(ranking.raw_slot_target_rank, 1)
        self.assertEqual(ranking.unique_target_rank, 1)
        self.assertEqual(ranking.unique_buckets, ((1, 2, 3), (7, 8, 9)))
        self.assertEqual(ranking.unique_bucket_first_beam_ranks, (1, 3))
        self.assertEqual(ranking.unique_bucket_sizes, (2, 1))
        self.assertEqual(ranking.duplicate_bucket_candidates, 1)
        self.assertEqual(ranking.non_expandable_candidates, 1)
        self.assertEqual(result["raw_slot_bucket_hr@1"], 1.0)
        self.assertEqual(result["unique_bucket_hr@1"], 1.0)
        self.assertEqual(result["unique_bucket_count_mean"], 2.0)
        self.assertEqual(result["target_bucket_size_histogram"], {"2": 1})

    def test_bucket_prefix_survives_invalid_suffix(self) -> None:
        tokens = TigerTokenIds(
            target_open=100,
            target_close=101,
            s1=tuple(range(200, 1224)),
            s2=tuple(range(1300, 2324)),
            s3=tuple(range(2400, 3424)),
            collision=tuple(range(3500, 3806)),
            eos=2,
        )
        candidate = parse_generated_candidate(
            (100, 201, 1302, 2403, 3501, 101),
            -0.5,
            tokens=tokens,
            index=self.index,
        )
        self.assertEqual(candidate.error, "missing_eos")
        self.assertIsNone(candidate.codes)
        self.assertEqual(candidate.base_bucket, (1, 2, 3))

        metrics = empty_bucket_metrics()
        ranking = update_bucket_metrics(
            metrics,
            target_codes=(1, 2, 3, 1),
            candidates=(candidate,),
            index=self.index,
        )
        self.assertEqual(ranking.raw_slot_target_rank, 1)
        self.assertEqual(ranking.unique_target_rank, 1)

    def test_trace_validation_rejects_inconsistent_bucket_size(self) -> None:
        row = {
            "target_poi_id": "poi-b",
            "target_base_bucket": [1, 2, 3],
            "exact_target_rank": None,
            "raw_slot_bucket_target_rank": 1,
            "unique_bucket_target_rank": 1,
            "unique_expandable_buckets": [
                {
                    "codes": [1, 2, 3],
                    "first_beam_rank": 1,
                    "first_sequence_score": -0.5,
                    "bucket_size": 2,
                }
            ],
            "candidates": [
                {
                    "beam_rank": 1,
                    "sequence_score": -0.5,
                    "base_bucket": [1, 2, 3],
                    "base_bucket_size": 0,
                    "poi_id": None,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "唯一可展开桶"):
            validate_candidate_trace_row(row, candidate_count_per_row=1)
        row["candidates"][0]["base_bucket_size"] = 2
        validate_candidate_trace_row(row, candidate_count_per_row=1)

    def test_legal_path_constraint_only_allows_corpus_prefixes(self) -> None:
        tokens = TigerTokenIds(
            target_open=100,
            target_close=101,
            s1=tuple(range(200, 1224)),
            s2=tuple(range(1300, 2324)),
            s3=tuple(range(2400, 3424)),
            collision=tuple(range(3500, 3806)),
            eos=2,
        )
        constraint = TigerLegalPathConstraint(
            index=self.index,
            tokens=tokens,
            prompt_width=3,
        )
        self.assertEqual(constraint.allowed_next(()), [100])
        self.assertEqual(constraint.allowed_next((100,)), [201, 207])
        self.assertEqual(constraint.allowed_next((100, 201)), [1302])
        self.assertEqual(constraint.allowed_next((100, 201, 1302)), [2403])
        self.assertEqual(
            constraint.allowed_next((100, 201, 1302, 2403)),
            [3500, 3501],
        )
        self.assertEqual(
            constraint.allowed_next((100, 201, 1302, 2403, 3501)),
            [101],
        )
        self.assertEqual(
            constraint.allowed_next((100, 201, 1302, 2403, 3501, 101)),
            [2],
        )
        with self.assertRaisesRegex(ValueError, "不在冻结语料库"):
            constraint.allowed_next((100, 201, 1302, 2404))

    def test_full_split_view_streams_jsonl_without_copying(self) -> None:
        rows = [
            {"sample_id": "a", "split": "test"},
            {"sample_id": "b", "split": "test"},
            {"sample_id": "c", "split": "test"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "test.jsonl"
            data_file.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            records = JsonlRecordSequence(data_file, expected_rows=3)
            view = build_full_split_view(
                data_file,
                split="test",
                source_rows=3,
                source_sha256="frozen-hash",
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(records[-1]["sample_id"], "c")
            self.assertEqual(
                [record["sample_id"] for record in records[1:3]], ["b", "c"]
            )
            self.assertEqual(view.data_path, data_file.resolve())
            self.assertEqual(view.row_count, 3)
            self.assertFalse(view.manifest["copy_materialized"])
            self.assertEqual(view.manifest["date"], "2026-07-14")

    def test_full_split_view_rejects_manifest_row_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "test.jsonl"
            data_file.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "实际行数 1 != manifest 2"):
                JsonlRecordSequence(data_file, expected_rows=2)


if __name__ == "__main__":
    unittest.main()
