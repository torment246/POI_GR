from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from qg_prqk.sft import evaluation as ev
from qg_prqk.sft import full_test_evaluation as full
from qg_prqk.sft.evaluation_data import SftEvaluationError, resolve, signature
from qg_prqk.sft.full_test_data import DEFAULT_CONFIG, VARIANTS, load_test_config


def index() -> ev.FinalIdIndex:
    return ev.FinalIdIndex({(10, 11, 12): 0}, {"p": (10, 11, 12)}, 1, 2, 3, 3, 1)


def plan(root: Path, variant: str = "a4_nogid") -> dict:
    return dict(variant=variant, checkpoint=dict(path=str(root)),
                config=dict(output_root=str(root), decoding=dict(batch_size=32, chunk_size=2),
                            variants={variant: dict(max_new_tokens=7)}),
                data=dict(rows=3, business_keys_sha256="synthetic",
                          sources={variant: dict(file=str(root / "unused.jsonl"), sha256="mocked")}))


class FullTestEvaluationTest(unittest.TestCase):
    def test_config_freezes_date_rows_and_validation_protocol(self) -> None:
        config = load_test_config(resolve(DEFAULT_CONFIG))
        self.assertEqual(config["source_rows"], 606682)
        self.assertEqual(config["decoding"]["num_beams"], 10)
        self.assertEqual(config["variants"]["a4_gid_parent"]["expected_step"], 8949)
        self.assertNotIn("fixed_reference", config)
        original = yaml.safe_load(resolve(DEFAULT_CONFIG).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for patch in (dict(split="valid"), dict(date="2026-07-13"), dict(source_rows=10000),
                          dict(validation_protocol_sha256="wrong"),
                          dict(output_root="qg_prqk/outputs/eval/sft_epoch3_fixed10k_generalization_v1")):
                path = Path(tmp) / "config.yaml"
                path.write_text(yaml.safe_dump({**original, **patch}))
                with self.assertRaises(SftEvaluationError):
                    load_test_config(path)

    def test_test_encoder_matches_validation_but_rejects_wrong_split_and_truncation(self) -> None:
        row = dict(split="test", identifier_variant="a4_nogid", target_poi_id="p", requires_dedup=False,
                   messages=[dict(role="user", content="x"), dict(role="assistant", content="y")])
        kwargs = dict(variant="a4_nogid", tokenizer=None, template=None, index=index())
        with mock.patch.object(ev, "encode_prompt_like_training", return_value=([7, 8], [1, 10, 11, 12, 2])):
            self.assertEqual(full.encode_test_record(row, **kwargs), ev.encode_record({**row, "split": "valid"}, **kwargs))
            for patch in (dict(split="valid"), dict(target_poi_id="missing"), dict(requires_dedup=True)):
                with self.assertRaises(SftEvaluationError):
                    full.encode_test_record({**row, **patch}, **kwargs)
        with mock.patch.object(ev, "encode_prompt_like_training", side_effect=SftEvaluationError("截断 Source")):
            with self.assertRaisesRegex(SftEvaluationError, "截断"):
                full.encode_test_record(row, **kwargs)

    def test_generation_matches_existing_validation_exactly(self) -> None:
        calls = []
        def generate(**kwargs):
            calls.append({k: v for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")})
            self.assertEqual(kwargs["input_ids"].tolist(), [[7, 8]])
            paths = [[1, 99, 99, 99, 2, 3]] + [[1, 10, 11, 12, 2, 3]] * 9
            return SimpleNamespace(sequences=torch.tensor([[7, 8, *p] for p in paths]),
                                   sequences_scores=-torch.arange(1, 11).float())
        kwargs = dict(variant="a4_nogid", model=SimpleNamespace(device="cpu", generate=generate),
                      tokenizer=SimpleNamespace(pad_token_id=0), template=None, index=index(), batch_size=32, max_new_tokens=7)
        with mock.patch.object(full, "encode_test_record", return_value=([7, 8], 0)), \
             mock.patch.object(ev, "encode_record", return_value=([7, 8], 0)):
            actual = full.evaluate_test_chunk([{}], **kwargs)
            expected = ev.evaluate_chunk([{}], **kwargs)
        self.assertEqual(actual, expected)
        self.assertEqual(calls[0], calls[1])
        self.assertNotIn("prefix_allowed_tokens_fn", calls[0])
        self.assertEqual(ev.finalize_metrics(actual)["hr@1"], 0)

    def test_oom_partial_chunk_resume_and_immutable_completed_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = plan(root)
            calls = []
            def chunk(rows, **kwargs):
                calls.append((len(rows), kwargs["batch_size"]))
                if len(calls) == 1:
                    raise RuntimeError("CUDA out of memory")
                if len(calls) == 3:
                    raise RuntimeError("synthetic interruption")
                metrics = ev.empty_metrics()
                for _ in rows:
                    ev.update_metrics(metrics, target_row=0, candidates=[ev.Candidate((10, 11, 12), 0, -1, None)] * 10)
                return metrics
            with mock.patch.object(full, "verify_plan"), \
                 mock.patch.object(full, "JsonlRecordSequence", return_value=[{}, {}, {}]), \
                 mock.patch.object(ev, "load_lf_tokenizer_and_template", return_value=([1, 2, 3], None)), \
                 mock.patch.object(ev, "build_final_id_index"), \
                 mock.patch.object(ev, "load_generation_model", return_value=SimpleNamespace(device="cpu")) as loader, \
                 mock.patch.object(full, "evaluate_test_chunk", side_effect=chunk), \
                 mock.patch.object(torch.cuda, "reset_peak_memory_stats"), \
                 mock.patch.object(torch.cuda, "max_memory_allocated", return_value=100):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    full.evaluate_test(frozen)
                progress = json.loads((root / "a4_nogid/results/progress.json").read_text())
                self.assertEqual(progress["next_line"], 2)
                result = full.evaluate_test(frozen)
                self.assertEqual(calls, [(2, 32), (2, 16), (1, 16), (1, 16)])
                self.assertEqual(result["metrics"]["sample_count"], 3)
                self.assertEqual(full.evaluate_test(frozen), result)
                self.assertEqual(loader.call_count, 2)
                changed = copy.deepcopy(frozen)
                changed["config"]["decoding"]["chunk_size"] = 1
                with self.assertRaisesRegex(SftEvaluationError, "来源不同"):
                    full.evaluate_test(changed)
            self.assertNotEqual(full.result_directory(frozen, 2), full.result_directory(frozen, None))

    def test_log_forwarding_reaches_file_and_main_console(self) -> None:
        log, console = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(console):
            full.forward_log(io.StringIO("500/606682\n1000/606682\n"), log)
        self.assertEqual(log.getvalue(), console.getvalue())
        self.assertIn("1000/606682", console.getvalue())

    def test_two_workers_use_two_gpus_and_failure_terminates_only_peer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plans = {v: (root / v / "plan.json", plan(root, v)) for v in VARIANTS}
            calls = []
            failed = SimpleNamespace(pid=101, stdout=io.StringIO("fake failure\n"), poll=lambda: 2)
            peer = SimpleNamespace(pid=102, stdout=io.StringIO("fake running\n"), poll=lambda: None,
                                   terminate=mock.Mock(), wait=mock.Mock(), returncode=-15)
            def start(command, **kwargs):
                calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                return failed if len(calls) == 1 else peer
            with mock.patch.object(full, "resolve", side_effect=lambda p: root / Path(p).name), \
                 mock.patch.object(full.subprocess, "Popen", side_effect=start), \
                 mock.patch.dict(full.os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}):
                with self.assertRaisesRegex(SftEvaluationError, "a4_gid_parent Test 失败"):
                    full.schedule_tests(plans, smoke_limit=8)
            self.assertEqual([gpu for _, gpu in calls], ["2", "5"])
            self.assertTrue(all("evaluate-sft-test" in cmd for cmd, _ in calls))
            peer.terminate.assert_called_once()
            peer.wait.assert_called_once()

    def test_dry_run_never_uses_gpu_or_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(full, "prepare_plans", return_value={}), \
             mock.patch.object(full, "validate_hardware") as hardware, \
             mock.patch.object(full, "schedule_tests") as workers:
            self.assertEqual(full.run_test_suite(dict(output_root=tmp), dry_run=True, smoke_limit=None), 0)
            hardware.assert_not_called()
            workers.assert_not_called()

    def test_summary_requires_both_full_results_and_separates_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plans = {v: (root / v / "plan.json", plan(root, v)) for v in VARIANTS}
            for variant, (_, frozen) in plans.items():
                metrics = ev.empty_metrics()
                for _ in range(3):
                    ev.update_metrics(metrics, target_row=0, candidates=[ev.Candidate((10, 11, 12), 0, -1, None)] * 10)
                config = full.run_config(frozen, None)
                full.write_immutable(full.result_directory(frozen, None) / "result.json",
                                     dict(status="completed", config=config, signature=signature(config), metrics=ev.finalize_metrics(metrics)))
            path = full.summarize_tests(plans, smoke_limit=None)
            summary = json.loads(path.read_text())
            self.assertEqual(summary["delta_gid_minus_nogid"]["hr@10"], 0)
            self.assertEqual([r["sample_count"] for r in summary["rows"]], [3, 3])
            with self.assertRaises(SftEvaluationError):
                full.summarize_tests(plans, smoke_limit=1)


if __name__ == "__main__":
    unittest.main()
