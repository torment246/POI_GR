"""Synthetic tests for constrained generative retrieval metrics and resume."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.generative_eval import (  # noqa: E402
    CandidateBatch,
    GenerativeEvalError,
    RankedCandidate,
    RetrievalMetricsAccumulator,
    RunProgressStore,
    authorize_test_file,
    build_fixed_validation_subset,
    build_reference_aligned_validation_subset,
    classify_error_types,
    hit_and_ndcg,
    load_ssp_predictions,
    rank_and_deduplicate_candidates,
    target_pid_is_confined_to_causal_history,
    validate_split_manifest,
)
from poi_gr.pid_trie import (  # noqa: E402
    CompactPidTrie,
    PidTokenIds,
    build_compact_trie_arrays,
)


class GenerativeEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokens = PidTokenIds(
            gid=np.arange(100, 132, dtype=np.int32),
            sid=(
                np.arange(1000, 2024, dtype=np.int32),
                np.arange(3000, 4024, dtype=np.int32),
                np.arange(5000, 6024, dtype=np.int32),
            ),
            dedup=np.arange(7000, 7512, dtype=np.int32),
            eos=99,
        )
        self.codes = np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 10, 20, 30, -1],
                [1, 2, 3, 4, 5, 7, 11, 21, 31, 0],
                [1, 2, 3, 4, 5, 7, 11, 21, 31, 1],
            ],
            dtype=np.int32,
        )
        self.trie = CompactPidTrie(**build_compact_trie_arrays(self.codes, self.tokens))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def pid(self, row: int) -> tuple[int, ...]:
        codes = self.codes[row]
        values = [int(self.tokens.gid[codes[index]]) for index in range(6)]
        values.extend(
            int(self.tokens.sid[index][codes[index + 6]]) for index in range(3)
        )
        if codes[9] >= 0:
            values.append(int(self.tokens.dedup[codes[9]]))
        return tuple(values)

    def path(self, row: int) -> list[int]:
        return [*self.pid(row), self.tokens.eos]

    def candidate(self, row: int, score: float = 0.0) -> RankedCandidate:
        return RankedCandidate(self.pid(row), row, score)

    def test_toy_hr_and_ndcg(self) -> None:
        result = hit_and_ndcg(
            self.pid(1),
            [self.candidate(0), self.candidate(1), self.candidate(2)],
            available_candidates=10,
        )
        self.assertEqual(result["hr@1"], 0)
        self.assertEqual(result["hr@3"], 1)
        self.assertAlmostEqual(result["ndcg@3"], 1 / math.log2(3))
        self.assertEqual(result["target_rank"], 2)

    def test_unavailable_k_is_null(self) -> None:
        result = hit_and_ndcg(
            self.pid(0),
            [self.candidate(0)],
            available_candidates=1,
        )
        self.assertEqual(result["hr@1"], 1)
        self.assertIsNone(result["hr@3"])
        self.assertIsNone(result["ndcg@10"])

    def test_singleton_and_dedup_group_statistics(self) -> None:
        accumulator = RetrievalMetricsAccumulator(available_candidates=10)
        singleton_batch = CandidateBatch(
            candidates=(self.candidate(0),),
            raw_count=1,
            valid_structure_count=1,
            valid_pid_count=1,
            duplicate_count=0,
            generated_lengths=(10,),
        )
        dedup_batch = CandidateBatch(
            candidates=(self.candidate(2), self.candidate(1)),
            raw_count=2,
            valid_structure_count=2,
            valid_pid_count=2,
            duplicate_count=0,
            generated_lengths=(11, 11),
        )
        accumulator.update(self.pid(0), singleton_batch)
        accumulator.update(self.pid(1), dedup_batch)
        result = accumulator.finalize()
        self.assertEqual(result["groups"]["singleton"]["sample_count"], 1)
        self.assertEqual(result["groups"]["singleton"]["hr@1"], 1.0)
        self.assertEqual(result["groups"]["dedup"]["sample_count"], 1)
        self.assertEqual(result["groups"]["dedup"]["hr@1"], 0.0)
        self.assertEqual(
            result["dedup_conditional_accuracy"]["condition_sample_count"], 1
        )
        self.assertEqual(result["dedup_conditional_accuracy"]["accuracy"], 0.0)

    def test_candidate_dedup_and_tie_break_are_stable(self) -> None:
        batch = rank_and_deduplicate_candidates(
            [self.path(2), self.path(1), self.path(1)],
            [-1.0, -1.0, -2.0],
            trie=self.trie,
            token_ids=self.tokens,
            top_k=10,
        )
        self.assertEqual(
            [candidate.pid_tokens for candidate in batch.candidates],
            [self.pid(1), self.pid(2)],
        )
        self.assertEqual(batch.duplicate_count, 1)
        self.assertEqual(batch.valid_pid_count, 3)

    def test_repeated_target_pid_is_allowed_only_in_causal_history(self) -> None:
        target = "<G_w><G_x><G_4><G_f><G_f><G_w><S1_1><S2_2><S3_3>"
        history = (
            "<HISTORY>\n"
            f"<POI_PID>{target}</POI_PID>\n"
            "</HISTORY>\n"
            "<CURRENT>\n<QUERY>repeat visit</QUERY>\n</CURRENT>"
        )
        self.assertTrue(target_pid_is_confined_to_causal_history(history, target))

        current_leak = history.replace(
            "<QUERY>repeat visit</QUERY>",
            f"<QUERY>{target}</QUERY>",
        )
        self.assertFalse(
            target_pid_is_confined_to_causal_history(current_leak, target)
        )

        history_query_leak = history.replace(
            "<HISTORY>\n",
            f"<HISTORY>\n<QUERY>{target}</QUERY>\n",
        )
        self.assertFalse(
            target_pid_is_confined_to_causal_history(history_query_leak, target)
        )

    def test_error_classification_and_dedup_condition(self) -> None:
        errors = classify_error_types(
            self.pid(1),
            [self.candidate(2), self.candidate(1)],
        )
        self.assertIn("base_pid_correct_dedup_error", errors)
        self.assertIn("target_in_top10_not_top1", errors)
        self.assertNotIn("top10_miss", errors)

    def test_resume_does_not_double_count(self) -> None:
        run_dir = self.root / "run"
        store = RunProgressStore(
            run_dir,
            config={"checkpoint": "toy", "beam": 10},
            available_candidates=10,
        )
        metrics = RetrievalMetricsAccumulator(available_candidates=10)
        metrics.update(
            self.pid(0),
            CandidateBatch(
                candidates=(self.candidate(0),),
                raw_count=1,
                valid_structure_count=1,
                valid_pid_count=1,
                duplicate_count=0,
                generated_lengths=(10,),
            ),
        )
        store.commit_chunk(
            start_line=0,
            end_line=1,
            end_byte=123,
            metrics=metrics,
            error_cases={},
            inference_seconds=1.0,
            generated_token_count=10,
            actual_batch_size=16,
            peak_memory_bytes=100,
        )
        resumed = RunProgressStore(
            run_dir,
            config={"checkpoint": "toy", "beam": 10},
            available_candidates=10,
        )
        self.assertEqual(resumed.next_line, 1)
        self.assertEqual(resumed.metrics.sample_count, 1)
        with self.assertRaisesRegex(GenerativeEvalError, "不连续"):
            resumed.commit_chunk(
                start_line=0,
                end_line=1,
                end_byte=123,
                metrics=metrics,
                error_cases={},
                inference_seconds=1.0,
                generated_token_count=10,
                actual_batch_size=16,
                peak_memory_bytes=100,
            )
        with self.assertRaisesRegex(GenerativeEvalError, "配置已变化"):
            RunProgressStore(
                run_dir,
                config={"checkpoint": "other", "beam": 10},
                available_candidates=10,
            )

    def test_validation_and_test_date_roles(self) -> None:
        valid = self.root / "valid.jsonl"
        test = self.root / "test.jsonl"
        valid.write_text("{}\n", encoding="utf-8")
        test.write_text("{}\n", encoding="utf-8")
        manifest = {
            "outputs": {
                "valid.jsonl": {
                    "rows": 597_421,
                    "sha256": "valid-hash",
                },
                "test.jsonl": {
                    "rows": 606_682,
                    "sha256": "test-hash",
                },
            },
            "time_split": {
                "valid": "2026-07-13",
                "test": "2026-07-14",
            },
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        _, valid_count, _ = validate_split_manifest(
            valid,
            split="valid",
            verify_hash=False,
        )
        _, test_count, _ = validate_split_manifest(
            test,
            split="test",
            verify_hash=False,
        )
        self.assertEqual(valid_count, 597_421)
        self.assertEqual(test_count, 606_682)
        manifest["time_split"]["valid"] = "2026-07-14"
        (self.root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(GenerativeEvalError, "日期"):
            validate_split_manifest(valid, split="valid", verify_hash=False)

    def test_test_file_is_not_authorized_before_freeze(self) -> None:
        test = self.root / "test.jsonl"
        test.write_text("{}\n", encoding="utf-8")
        selected = self.root / "selected_config.json"
        with self.assertRaisesRegex(GenerativeEvalError, "禁止读取"):
            authorize_test_file(selected, test)
        selected.write_text(
            json.dumps({"status": "draft"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(GenerativeEvalError, "禁止读取"):
            authorize_test_file(selected, test)
        selected.write_text(
            json.dumps({"status": "frozen"}),
            encoding="utf-8",
        )
        self.assertEqual(
            authorize_test_file(selected, test)["status"],
            "frozen",
        )

    def test_beijing_ssp_predictions_use_direct_safe_depths(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        predictions_dir = self.root / "beijing_ssp"
        predictions_dir.mkdir()
        predictions_path = predictions_dir / "predictions.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample_id": ["a", "b"],
                    "user_gid": ["wx4g0b", "wx4g0c"],
                    "predicted_lambda": pa.array([0, 4], type=pa.int8()),
                    "prefill_length": pa.array([0, 4], type=pa.int8()),
                }
            ),
            predictions_path,
        )
        output_hash = __import__("hashlib").sha256(
            predictions_path.read_bytes()
        ).hexdigest()
        (predictions_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "genpoi-beijing-ssp-predictions-v1",
                    "status": "completed",
                    "selection_strategy": "direct_ordinal_safe_prefix",
                    "useful_prefix_depths": [3, 4, 5, 6],
                    "rows": 2,
                    "limit": None,
                    "source_validation_jsonl_sha256": "fixed-valid-hash",
                    "target_fields_used": [],
                    "output_file": str(predictions_path),
                    "output_sha256": output_hash,
                }
            ),
            encoding="utf-8",
        )
        predictions, manifest = load_ssp_predictions(
            predictions_dir,
            evaluation_data_sha256="fixed-valid-hash",
            expected_rows=2,
        )
        self.assertEqual(predictions["a"].prefill_length, 0)
        self.assertEqual(predictions["b"].prefill_length, 4)
        self.assertEqual(
            manifest["schema_version"],
            "genpoi-beijing-ssp-predictions-v1",
        )

        table = pq.read_table(predictions_path)
        table = table.set_column(
            3,
            "prefill_length",
            pa.array([0, 3], type=pa.int8()),
        )
        pq.write_table(table, predictions_path)
        manifest_path = predictions_dir / "manifest.json"
        invalid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invalid_manifest["output_sha256"] = __import__("hashlib").sha256(
            predictions_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
        with self.assertRaisesRegex(GenerativeEvalError, "前缀长度非法"):
            load_ssp_predictions(
                predictions_dir,
                evaluation_data_sha256="fixed-valid-hash",
                expected_rows=2,
            )

    def test_fixed_validation_subset_is_deterministic_and_source_ordered(self) -> None:
        valid = self.root / "valid.jsonl"
        records = [
            {
                "sample_id": f"{index:03d}",
                "split": "valid",
                "messages": [],
            }
            for index in reversed(range(20))
        ]
        valid.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        source_hash = __import__("hashlib").sha256(valid.read_bytes()).hexdigest()
        first = build_fixed_validation_subset(
            valid,
            self.root / "subset",
            source_rows=20,
            source_sha256=source_hash,
            subset_size=5,
        )
        selected = [
            json.loads(line)["sample_id"]
            for line in first.data_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(selected, ["004", "003", "002", "001", "000"])
        second = build_fixed_validation_subset(
            valid,
            self.root / "subset",
            source_rows=20,
            source_sha256=source_hash,
            subset_size=5,
        )
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(
            first.manifest["selected_row_indices_sha256"],
            second.manifest["selected_row_indices_sha256"],
        )

    def test_reference_subset_aligns_business_keys_and_target_poi(self) -> None:
        valid = self.root / "valid.jsonl"
        reference = self.root / "reference.jsonl"
        source_records = [
            {
                "sample_id": "method-c",
                "order_id": "order-c",
                "searchid": "search-c",
                "target_poi_id": "poi-c",
            },
            {
                "sample_id": "method-a",
                "order_id": "order-a",
                "searchid": "search-a",
                "target_poi_id": "poi-a",
            },
            {
                "sample_id": "method-b",
                "order_id": "order-b",
                "searchid": "search-b",
                "target_poi_id": "poi-b",
            },
        ]
        reference_records = [
            {
                "sample_id": "baseline-b",
                "order_id": "order-b",
                "searchid": "search-b",
                "target_poi_id": "poi-b",
            },
            {
                "sample_id": "baseline-a",
                "order_id": "order-a",
                "searchid": "search-a",
                "target_poi_id": "poi-a",
            },
        ]
        valid.write_text(
            "".join(json.dumps(record) + "\n" for record in source_records),
            encoding="utf-8",
        )
        reference.write_text(
            "".join(json.dumps(record) + "\n" for record in reference_records),
            encoding="utf-8",
        )
        source_hash = __import__("hashlib").sha256(valid.read_bytes()).hexdigest()
        subset = build_reference_aligned_validation_subset(
            valid,
            reference,
            self.root / "aligned",
            source_rows=3,
            source_sha256=source_hash,
        )
        aligned = [
            json.loads(line)
            for line in subset.data_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [record["order_id"] for record in aligned],
            ["order-b", "order-a"],
        )
        self.assertEqual(
            subset.manifest["selection_method"],
            "exact_order_id_searchid_match",
        )
        rebuilt = build_reference_aligned_validation_subset(
            valid,
            reference,
            self.root / "aligned",
            source_rows=3,
            source_sha256=source_hash,
        )
        self.assertEqual(subset.sha256, rebuilt.sha256)

    def test_reference_subset_rejects_target_poi_mismatch(self) -> None:
        valid = self.root / "valid.jsonl"
        reference = self.root / "reference.jsonl"
        valid.write_text(
            json.dumps(
                {
                    "order_id": "order-a",
                    "searchid": "search-a",
                    "target_poi_id": "poi-new",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        reference.write_text(
            json.dumps(
                {
                    "order_id": "order-a",
                    "searchid": "search-a",
                    "target_poi_id": "poi-old",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        source_hash = __import__("hashlib").sha256(valid.read_bytes()).hexdigest()
        with self.assertRaisesRegex(GenerativeEvalError, "目标 POI 不一致"):
            build_reference_aligned_validation_subset(
                valid,
                reference,
                self.root / "mismatch",
                source_rows=1,
                source_sha256=source_hash,
            )


if __name__ == "__main__":
    unittest.main()
