"""Synthetic tests for SID teacher-forcing diagnostics."""

from __future__ import annotations

import math
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from poi_gr.pid.trie import PidTokenIds

from poi_gr.sft.teacher_forcing import (
    TeacherForcingError,
    empty_teacher_forcing_metrics,
    finalize_teacher_forcing_metrics,
    score_target_logits,
    update_teacher_forcing_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sft" / "diagnose_sid_teacher_forcing.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "diagnose_sid_teacher_forcing",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


class TeacherForcingMetricsTest(unittest.TestCase):
    def test_target_logit_scoring_reports_top1_top10_and_nll(self) -> None:
        logits = torch.zeros((2, 12), dtype=torch.float32)
        logits[0, 3] = 5.0
        logits[1, 11] = -5.0
        top1, top10, nll = score_target_logits(logits, torch.tensor([3, 11]))
        self.assertEqual(top1.tolist(), [True, False])
        self.assertEqual(top10.tolist(), [True, False])
        self.assertLess(float(nll[0]), float(nll[1]))

    def test_position_prefix_group_and_sequence_metrics(self) -> None:
        metrics = empty_teacher_forcing_metrics()
        update_teacher_forcing_metrics(
            metrics,
            position_names=(
                "target_open",
                "sid_1",
                "sid_2",
                "sid_3",
                "target_close",
                "eos",
            ),
            top1_correct=(True, True, False, True, True, True),
            top10_correct=(True, True, True, True, True, True),
            nll_values=(0.1, 0.2, 0.3, 0.4, 0.1, 0.1),
            semantic_indices=(1, 2, 3),
            identifier_indices=(1, 2, 3),
            group="singleton",
        )
        result = finalize_teacher_forcing_metrics(metrics)
        self.assertEqual(result["positions"]["sid_2"]["top1_accuracy"], 0.0)
        self.assertEqual(
            result["aliases"]["after_sid3_decision"]["top1_accuracy"], 1.0
        )
        self.assertEqual(
            result["semantic_prefixes"]["1"]["top1_all_accuracy"], 1.0
        )
        self.assertEqual(
            result["semantic_prefixes"]["2"]["top1_all_accuracy"], 0.0
        )
        self.assertEqual(result["full_identifier"]["top1_all_accuracy"], 0.0)
        self.assertEqual(result["full_identifier"]["top10_all_accuracy"], 1.0)
        self.assertTrue(math.isfinite(result["target_perplexity"]))
        self.assertEqual(result["groups"]["singleton"]["sample_count"], 1)

    def test_context_prefix_exact_metrics(self) -> None:
        metrics = empty_teacher_forcing_metrics()
        update_teacher_forcing_metrics(
            metrics,
            position_names=(
                "gid_1",
                "gid_2",
                "sid_1",
                "sid_2",
                "sid_3",
            ),
            top1_correct=(True, False, True, True, True),
            top10_correct=(True, True, True, True, True),
            nll_values=(0.1, 0.2, 0.1, 0.1, 0.1),
            semantic_indices=(2, 3, 4),
            identifier_indices=(0, 1, 2, 3, 4),
            group="singleton",
            context_indices=(0, 1),
        )
        result = finalize_teacher_forcing_metrics(metrics)
        self.assertEqual(
            result["context_prefixes"]["1"]["top1_all_accuracy"],
            1.0,
        )
        self.assertEqual(
            result["context_prefixes"]["2"]["top1_all_accuracy"],
            0.0,
        )
        self.assertEqual(
            result["context_prefixes"]["2"]["top10_all_accuracy"],
            1.0,
        )

    def test_genpoi_singleton_and_dedup_positions(self) -> None:
        tokens = PidTokenIds(
            gid=np.arange(100, 132, dtype=np.int32),
            sid=(
                np.arange(1000, 2024, dtype=np.int32),
                np.arange(3000, 4024, dtype=np.int32),
                np.arange(5000, 6024, dtype=np.int32),
            ),
            dedup=np.arange(7000, 7512, dtype=np.int32),
            eos=99,
        )
        base_target = [100, 101, 102, 103, 104, 105, 1007, 3008, 5009]
        record = {
            "sample_id": "sample",
            "split": "valid",
            "requires_dedup": False,
            "messages": [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "target"},
            ],
        }
        with patch.object(
            SCRIPT_MODULE,
            "encode_prompt_like_training",
            return_value=([1, 2], base_target),
        ):
            singleton = SCRIPT_MODULE.encode_genpoi_example(
                record,
                tokenizer=object(),
                template=object(),
                tokens=tokens,
                cutoff_len=512,
            )
        self.assertEqual(singleton.semantic_indices, (6, 7, 8))
        self.assertEqual(singleton.context_indices, (0, 1, 2, 3, 4, 5))
        self.assertEqual(singleton.identifier_indices, tuple(range(9)))
        self.assertEqual(singleton.position_names[-1], "eos")

        record["requires_dedup"] = True
        with patch.object(
            SCRIPT_MODULE,
            "encode_prompt_like_training",
            return_value=([1, 2], [*base_target, 7003]),
        ):
            dedup = SCRIPT_MODULE.encode_genpoi_example(
                record,
                tokenizer=object(),
                template=object(),
                tokens=tokens,
                cutoff_len=512,
            )
        self.assertEqual(dedup.identifier_indices, tuple(range(10)))
        self.assertEqual(dedup.position_names[-2:], ("disambiguation", "eos"))
        self.assertEqual(dedup.group, "dedup")

    def test_mismatched_observation_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(TeacherForcingError, "长度不一致"):
            update_teacher_forcing_metrics(
                empty_teacher_forcing_metrics(),
                position_names=("sid_1",),
                top1_correct=(True,),
                top10_correct=(),
                nll_values=(0.1,),
                semantic_indices=(0, 0, 0),
                identifier_indices=(0,),
                group="broken",
            )


if __name__ == "__main__":
    unittest.main()
