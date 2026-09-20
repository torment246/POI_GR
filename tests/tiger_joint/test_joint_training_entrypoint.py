"""Lightweight tests for the streamed TIGER-Joint training entry point."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiger_joint.train_joint import (  # noqa: E402
    MetricAccumulator,
    TigerJointTrainingError,
    accumulation_group_sizes,
    buffered_shuffle,
    build_joint_schedulers,
    build_line_shards,
    config_signature,
    global_behavior_batch_size,
    load_config,
    probe_metrics,
    validate_rank_rng_states,
)


class TigerJointTrainingEntrypointTest(unittest.TestCase):
    def test_qwen_and_rq_use_matching_warmup_cosine_schedules(self) -> None:
        qwen_parameter = torch.nn.Parameter(torch.ones(()))
        rq_parameter = torch.nn.Parameter(torch.ones(()))
        qwen_optimizer = torch.optim.AdamW([qwen_parameter], lr=1e-3)
        rq_optimizer = torch.optim.Adam([rq_parameter], lr=3e-4)
        qwen_scheduler, rq_scheduler, warmup_steps = build_joint_schedulers(
            qwen_optimizer,
            rq_optimizer,
            planned_optimizer_steps=10,
            warmup_ratio=0.2,
        )

        self.assertEqual(warmup_steps, 2)
        qwen_learning_rates = []
        rq_learning_rates = []
        for _ in range(10):
            qwen_optimizer.step()
            rq_optimizer.step()
            qwen_scheduler.step()
            rq_scheduler.step()
            qwen_learning_rates.append(qwen_scheduler.get_last_lr()[0])
            rq_learning_rates.append(rq_scheduler.get_last_lr()[0])

        for qwen_lr, rq_lr in zip(qwen_learning_rates, rq_learning_rates):
            self.assertAlmostEqual(qwen_lr / 1e-3, rq_lr / 3e-4)
        self.assertLess(qwen_learning_rates[0], qwen_learning_rates[1])
        self.assertEqual(qwen_learning_rates[1], 1e-3)
        self.assertLess(qwen_learning_rates[-1], qwen_learning_rates[-2])
        self.assertAlmostEqual(qwen_learning_rates[-1], 0.0)

    def test_both_scheduler_states_resume_on_the_same_next_step(self) -> None:
        qwen_parameter = torch.nn.Parameter(torch.ones(()))
        rq_parameter = torch.nn.Parameter(torch.ones(()))
        qwen_optimizer = torch.optim.AdamW([qwen_parameter], lr=1e-3)
        rq_optimizer = torch.optim.Adam([rq_parameter], lr=3e-4)
        qwen_scheduler, rq_scheduler, _ = build_joint_schedulers(
            qwen_optimizer,
            rq_optimizer,
            planned_optimizer_steps=10,
            warmup_ratio=0.2,
        )
        for _ in range(4):
            qwen_optimizer.step()
            rq_optimizer.step()
            qwen_scheduler.step()
            rq_scheduler.step()

        restored_qwen_optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.ones(()))], lr=1e-3
        )
        restored_rq_optimizer = torch.optim.Adam(
            [torch.nn.Parameter(torch.ones(()))], lr=3e-4
        )
        restored_qwen_scheduler, restored_rq_scheduler, _ = build_joint_schedulers(
            restored_qwen_optimizer,
            restored_rq_optimizer,
            planned_optimizer_steps=10,
            warmup_ratio=0.2,
        )
        restored_qwen_optimizer.load_state_dict(qwen_optimizer.state_dict())
        restored_rq_optimizer.load_state_dict(rq_optimizer.state_dict())
        restored_qwen_scheduler.load_state_dict(qwen_scheduler.state_dict())
        restored_rq_scheduler.load_state_dict(rq_scheduler.state_dict())

        qwen_optimizer.step()
        rq_optimizer.step()
        qwen_scheduler.step()
        rq_scheduler.step()
        restored_qwen_optimizer.step()
        restored_rq_optimizer.step()
        restored_qwen_scheduler.step()
        restored_rq_scheduler.step()

        self.assertEqual(restored_qwen_scheduler.last_epoch, qwen_scheduler.last_epoch)
        self.assertEqual(restored_rq_scheduler.last_epoch, rq_scheduler.last_epoch)
        self.assertAlmostEqual(
            restored_qwen_scheduler.get_last_lr()[0],
            qwen_scheduler.get_last_lr()[0],
        )
        self.assertAlmostEqual(
            restored_rq_scheduler.get_last_lr()[0], rq_scheduler.get_last_lr()[0]
        )

    def test_buffered_shuffle_is_deterministic_and_preserves_rows(self) -> None:
        first = list(
            buffered_shuffle(
                iter(range(20)),
                buffer_rows=5,
                rng=np.random.default_rng(42),
            )
        )
        second = list(
            buffered_shuffle(
                iter(range(20)),
                buffer_rows=5,
                rng=np.random.default_rng(42),
            )
        )

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(20)))
        self.assertNotEqual(first, list(range(20)))

    def test_probe_metrics_reports_utilization_and_churn(self) -> None:
        previous = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [1, 0, 2], [0, 2, 2]],
            dtype=torch.long,
        )
        current = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 2]],
            dtype=torch.long,
        )

        result = probe_metrics(
            current,
            previous,
            codebook_sizes=(2, 4, 4),
        )

        self.assertEqual(result["distinct_full_sids"], 4)
        self.assertEqual(result["used_codes_by_level"], [2, 3, 3])
        self.assertEqual(result["utilization_by_level"], [1.0, 0.75, 0.75])
        self.assertAlmostEqual(result["full_sid_churn_rate"], 0.75)
        self.assertEqual(result["churn_rate_by_level"], [0.25, 0.25, 0.25])

    def test_metric_accumulator_uses_explicit_target_rows(self) -> None:
        output = SimpleNamespace(
            total_loss=torch.tensor(3.0),
            generation_loss=torch.tensor(2.0),
            alignment_loss=torch.tensor(1.0),
            rq_loss=torch.tensor(0.5),
            teacher_forced_token_accuracy=torch.tensor(0.4),
            teacher_forced_sid_token_accuracy=torch.tensor(0.2),
            teacher_forced_static_token_accuracy=torch.tensor(0.6),
            teacher_forced_exact_match=torch.tensor(0.1),
            target_codes=torch.tensor(
                [[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=torch.long
            ),
        )
        accumulator = MetricAccumulator()

        accumulator.add(
            output,
            torch.tensor([10, 10, 11]),
            catalog_samples=5,
        )

        summary = accumulator.summary(alignment_weight=0.1, rq_weight=1.0)
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["catalog_samples"], 5)
        self.assertEqual(summary["mean_distinct_target_rows_per_microbatch"], 2)
        self.assertEqual(summary["mean_distinct_target_sids_per_microbatch"], 2)
        self.assertAlmostEqual(summary["generation_loss"], 2.0)
        self.assertAlmostEqual(summary["total_loss"], 2.6)
        with self.assertRaisesRegex(TigerJointTrainingError, "shape"):
            accumulator.add(
                output,
                torch.tensor([10, 11]),
                catalog_samples=5,
            )

    def test_config_overrides_batch_layout_and_signs_it(self) -> None:
        args = SimpleNamespace(
            behavior_batch_size=4,
            gradient_accumulation_steps=2,
            catalog_batch_size=32,
        )
        config = load_config(
            PROJECT_ROOT / "configs/tiger_joint/v1_server.yaml",
            args,
        )

        self.assertEqual(config.behavior_batch_size, 4)
        self.assertEqual(config.gradient_accumulation_steps, 2)
        self.assertEqual(config.catalog_batch_size, 32)
        first = config_signature(
            config,
            max_optimizer_steps=10,
            world_size=4,
            save_checkpoint=False,
            save_optimizer_state=False,
        )
        second = config_signature(
            config,
            max_optimizer_steps=10,
            world_size=4,
            save_checkpoint=True,
            save_optimizer_state=False,
        )
        self.assertNotEqual(first, second)

    def test_line_shards_cover_jsonl_without_overlap(self) -> None:
        rows = [f'{{"row": {index}}}\n'.encode("utf-8") for index in range(7)]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "train.jsonl"
            path.write_bytes(b"".join(rows))

            shards = build_line_shards(path, expected_rows=7, world_size=4)

            self.assertEqual([shard.rows for shard in shards], [1, 2, 2, 2])
            self.assertEqual(shards[0].start_offset, 0)
            self.assertEqual(shards[-1].end_offset, path.stat().st_size)
            for left, right in zip(shards, shards[1:]):
                self.assertEqual(left.end_row, right.start_row)
                self.assertEqual(left.end_offset, right.start_offset)

    def test_global_batch_size_includes_each_rank_tail(self) -> None:
        shards = [SimpleNamespace(rows=value) for value in (5, 6, 5, 6)]

        self.assertEqual(
            global_behavior_batch_size(
                shards, microbatch_index=0, per_rank_batch_size=4
            ),
            16,
        )
        self.assertEqual(
            global_behavior_batch_size(
                shards, microbatch_index=1, per_rank_batch_size=4
            ),
            6,
        )
        behavior_rows, catalog_rows = accumulation_group_sizes(
            shards,
            first_microbatch_index=0,
            microbatches_per_epoch=2,
            per_rank_behavior_batch_size=4,
            per_rank_catalog_batch_size=3,
            accumulation_steps=2,
        )
        self.assertEqual(behavior_rows, 22)
        self.assertEqual(catalog_rows, 24)

    def test_rank_rng_states_are_bound_to_world_size(self) -> None:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.tensor([1, 2, 3], dtype=torch.uint8),
        }

        validated = validate_rank_rng_states([state, state], world_size=2)

        self.assertEqual(len(validated), 2)
        with self.assertRaisesRegex(TigerJointTrainingError, "数量"):
            validate_rank_rng_states([state], world_size=2)
        with self.assertRaisesRegex(TigerJointTrainingError, "内容"):
            validate_rank_rng_states([{"python": random.getstate()}], world_size=1)


if __name__ == "__main__":
    unittest.main()
