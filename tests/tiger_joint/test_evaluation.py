"""Synthetic tests for TIGER-Joint Bucket-HR semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint.data import SidTokenLayout  # noqa: E402
from poi_gr.methods.tiger_joint.evaluation import (  # noqa: E402
    JointBucketIndex,
    JointLegalPathConstraint,
    JointSidCandidateParser,
    compute_joint_sid_metrics,
    empty_bucket_metrics,
    finalize_bucket_metrics,
    materialize_joint_prompt,
    rank_candidate_buckets,
    update_bucket_metrics,
)
from poi_gr.methods.tiger_joint.preparation import DynamicTigerExample  # noqa: E402


class TigerJointEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = SidTokenLayout(((10, 11), (20, 21), (30, 31)))
        self.codes = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.int32,
        )
        self.index = JointBucketIndex.from_codes(self.codes, (2, 2, 2))
        self.parser = JointSidCandidateParser(
            token_layout=self.layout,
            target_open_token_id=5,
            target_close_token_id=6,
        )

    def test_bucket_index_and_static_metrics(self) -> None:
        self.assertEqual(self.index.bucket_size((0, 0, 0)), 2)
        self.assertEqual(self.index.bucket_size((1, 1, 1)), 1)
        self.assertEqual(self.index.bucket_size((0, 1, 0)), 0)
        metrics = compute_joint_sid_metrics(self.codes, (2, 2, 2))
        self.assertEqual(metrics["basic"]["distinct_sid_count"], 2)
        self.assertEqual(metrics["basic"]["bucket_size_max"], 2)
        self.assertEqual(
            [layer["used_token_count"] for layer in metrics["layers"]],
            [2, 2, 2],
        )
        self.assertEqual(
            [prefix["distinct_sid_count"] for prefix in metrics["prefixes"]],
            [2, 2, 2],
        )

    def test_invalid_close_keeps_valid_bucket_prefix(self) -> None:
        candidates = [
            self.parser.parse((5, 10, 20, 30, 99), 5.0),
            self.parser.parse((5, 10, 20, 30, 6), 4.0),
            self.parser.parse((5, 10, 21, 30, 6), 3.0),
            self.parser.parse((99, 10, 20, 30, 6), 2.0),
            self.parser.parse((5, 11, 21, 31, 6), 1.0),
        ]
        ranking = rank_candidate_buckets(
            target_codes=(0, 0, 0),
            candidates=candidates,
            index=self.index,
        )
        self.assertEqual(ranking.raw_slot_target_rank, 1)
        self.assertEqual(ranking.unique_target_rank, 1)
        self.assertEqual(ranking.unique_buckets, ((0, 0, 0), (1, 1, 1)))
        self.assertEqual(ranking.duplicate_bucket_candidates, 1)
        self.assertEqual(ranking.nonexpandable_candidates, 1)
        self.assertEqual(ranking.prefix_invalid_candidates, 1)
        self.assertEqual(ranking.complete_format_valid_candidates, 3)

    def test_unique_rank_compacts_invalid_slots(self) -> None:
        candidates = [
            self.parser.parse((99, 10, 20, 30, 6), 3.0),
            self.parser.parse((5, 10, 21, 30, 6), 2.0),
            self.parser.parse((5, 10, 20, 30, 6), 1.0),
        ]
        metrics = empty_bucket_metrics()
        ranking = update_bucket_metrics(
            metrics,
            target_codes=(0, 0, 0),
            candidates=candidates,
            index=self.index,
        )
        result = finalize_bucket_metrics(metrics)
        self.assertEqual(ranking.raw_slot_target_rank, 3)
        self.assertEqual(ranking.unique_target_rank, 1)
        self.assertEqual(result["raw_slot_bucket_hr@1"], 0.0)
        self.assertEqual(result["raw_slot_bucket_hr@3"], 1.0)
        self.assertEqual(result["unique_bucket_hr@1"], 1.0)
        self.assertEqual(result["target_bucket_size_histogram"], {"2-5": 1})

    def test_legal_path_constraint_only_allows_catalog_buckets(self) -> None:
        constraint = JointLegalPathConstraint(
            index=self.index,
            token_layout=self.layout,
            target_open_token_id=5,
            target_close_token_id=6,
            eos_token_id=2,
            prompt_width=3,
        )
        self.assertEqual(constraint.allowed_next(()), [5])
        self.assertEqual(constraint.allowed_next((5,)), [10, 11])
        self.assertEqual(constraint.allowed_next((5, 10)), [20])
        self.assertEqual(constraint.allowed_next((5, 10, 20)), [30])
        self.assertEqual(constraint.allowed_next((5, 10, 20, 30)), [6])
        self.assertEqual(constraint.allowed_next((5, 10, 20, 30, 6)), [2])
        self.assertEqual(constraint(0, [100, 101, 102, 5, 11]), [21])
        with self.assertRaisesRegex(ValueError, "不在冻结目录"):
            constraint.allowed_next((5, 10, 21))

    def test_constrained_metric_records_decoding_protocol(self) -> None:
        metrics = empty_bucket_metrics()
        update_bucket_metrics(
            metrics,
            target_codes=(0, 0, 0),
            candidates=(self.parser.parse((5, 10, 20, 30, 6), 1.0),),
            index=self.index,
        )
        result = finalize_bucket_metrics(metrics, legal_path_constraint=True)
        self.assertEqual(
            result["decoding"],
            "epoch3_catalog_legal_path_constrained_beam_search",
        )
        self.assertEqual(result["expandable_candidate_rate"], 1.0)

    def test_materialize_prompt_uses_frozen_history_codes_only(self) -> None:
        example = DynamicTigerExample(
            sample_id="sample",
            split="valid",
            input_ids=(100, 10, 20, 30, 5, 10, 20, 30, 6),
            labels=(-100, -100, -100, -100, 5, 10, 20, 30, 6),
            query_state_position=3,
            history_poi_rows=(1,),
            history_sid_positions=((1, 2, 3),),
            target_poi_row=0,
            target_sid_positions=(5, 6, 7),
        )
        prompt = materialize_joint_prompt(
            example,
            sid_codes=np.asarray([[0, 0, 0], [1, 0, 1]], dtype=np.int32),
            token_layout=self.layout,
        )
        self.assertEqual(prompt, (100, 11, 20, 31))


if __name__ == "__main__":
    unittest.main()
