from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from qg_prqk.artifacts import sha256_file
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft import evaluation_suite as suite
from qg_prqk.sft.evaluation_data import SftEvaluationError, SUBSETS, signature


def index(width: int = 3) -> ev.FinalIdIndex:
    plain = tuple(range(10, 10 + width))
    other = tuple(range(20, 20 + width))
    return ev.FinalIdIndex({plain: 0, other + (50,): 1}, {"poi0": plain, "poi1": other + (50,)}, 1, 2, 3, width, 2)


class CandidateTest(unittest.TestCase):
    def test_two_lengths_dedup_and_eos(self) -> None:
        for width in (3, 9):
            catalog = index(width)
            for poi, tokens in catalog.tokens_by_poi.items():
                candidate = ev.parse_candidate([1, *tokens, 2, 3, 0, 0], -1.0, catalog)
                self.assertIsNone(candidate.error)
                self.assertEqual(candidate.poi_row, int(poi[-1]))
            self.assertEqual(ev.parse_candidate([1, *catalog.tokens_by_poi["poi0"], 2], -1.0, catalog).error, "missing_eos")
            self.assertEqual(ev.parse_candidate([1, *catalog.tokens_by_poi["poi1"][:-1], 2, 3], -1.0, catalog).error, "corpus_miss")
            self.assertEqual(ev.parse_candidate([1, *catalog.tokens_by_poi["poi0"], 50, 2, 3], -1.0, catalog).error, "corpus_miss")

    def test_nogid_cannot_accept_gid_target(self) -> None:
        self.assertIsNotNone(ev.parse_candidate([1, *range(10, 19), 2, 3], -1.0, index()).error)

    def test_invalid_and_duplicate_beams_keep_rank(self) -> None:
        metrics = ev.empty_metrics()
        ev.update_metrics(metrics, target_row=1, candidates=[
            ev.Candidate(None, None, -1, "corpus_miss"), ev.Candidate((1,), 0, -2, None),
            ev.Candidate((1,), 0, -3, None), ev.Candidate((2,), 1, -4, None),
        ])
        result = ev.finalize_metrics(metrics)
        self.assertEqual(result["hr@3"], 0)
        self.assertEqual(result["hr@5"], 1)
        self.assertEqual(result["mrr@10"], 0.25)
        self.assertAlmostEqual(result["ndcg@10"], 1 / math.log2(5))
        self.assertEqual(result["valid_id_rate"], 0.75)

    def test_left_padding_preserves_all_source_tokens(self) -> None:
        ids, mask = ev.pad_prompts([[4, 5], [6]], 0, "cpu")
        self.assertEqual(ids.tolist(), [[4, 5], [0, 6]])
        self.assertEqual(mask.tolist(), [[1, 1], [0, 1]])


class EvaluationLoopTest(unittest.TestCase):
    def test_two_gpu_scheduler_assigns_both_variants_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launched = []
            occupied = set()
            peak = 0
            def start(command, **kwargs):
                nonlocal peak
                gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                self.assertNotIn(gpu, occupied)
                occupied.add(gpu)
                peak = max(peak, len(occupied))
                launched.append((command, gpu))
                calls = 0
                def poll():
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return None
                    occupied.remove(gpu)
                    return 0
                return SimpleNamespace(pid=100 + len(launched), poll=poll)
            plans = {v: (Path(tmp) / v / "plan.json", dict(variant=v)) for v in suite.VARIANTS}
            with mock.patch.object(suite, "resolve", side_effect=lambda p: Path(tmp) / Path(p).name), \
                 mock.patch.object(suite.os, "fsencode", return_value=b"/synthetic-short-tmp"), \
                 mock.patch.object(suite.subprocess, "Popen", side_effect=start), \
                 mock.patch.object(suite.time, "sleep"), \
                 mock.patch.dict(suite.os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}):
                suite.schedule_workers(plans, smoke_limit=8)
            self.assertEqual(peak, 2)
            self.assertEqual(occupied, set())
            self.assertEqual([gpu for _, gpu in launched], ["2", "5"] * 5)
            cells = [(Path(cmd[cmd.index("--plan") + 1]).parent.name,
                      cmd[cmd.index("--worker-subset") + 1]) for cmd, _ in launched]
            self.assertEqual(cells, [(v, subset) for subset in SUBSETS for v in suite.VARIANTS])
            for cmd, _ in launched:
                self.assertEqual(cmd[-2:], ["--smoke-limit", "8"])

    def test_scheduler_rejects_four_visible_devices(self) -> None:
        with mock.patch.dict(suite.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}), \
             mock.patch.object(suite.subprocess, "Popen") as start:
            with self.assertRaisesRegex(SftEvaluationError, "两个"):
                suite.schedule_workers({}, smoke_limit=None)
            start.assert_not_called()

    def test_worker_failure_stops_only_its_peer_and_no_pending_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            failed = SimpleNamespace(pid=101, poll=lambda: 1)
            peer = SimpleNamespace(pid=102, poll=lambda: None, terminate=mock.Mock(),
                                   wait=mock.Mock(), returncode=-15)
            plans = {v: (Path(tmp) / v / "plan.json", dict(variant=v)) for v in suite.VARIANTS}
            with mock.patch.object(suite, "resolve", side_effect=lambda p: Path(tmp) / Path(p).name), \
                 mock.patch.object(suite.os, "fsencode", return_value=b"/synthetic-short-tmp"), \
                 mock.patch.object(suite.subprocess, "Popen", side_effect=[failed, peer]) as start, \
                 mock.patch.dict(suite.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
                with self.assertRaisesRegex(SftEvaluationError, "a4_gid_parent/fixed10k"):
                    suite.schedule_workers(plans, smoke_limit=None)
            self.assertEqual(start.call_count, 2)
            peer.terminate.assert_called_once()
            peer.wait.assert_called_once()

    def test_dry_run_keeps_old_plan_and_checks_both_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in suite.VARIANTS:
                (root / variant).mkdir()
                (root / variant / "plan.json").write_text('{"old_four_gpu_plan": true}\n')
            with mock.patch.object(suite, "prepare_plan", side_effect=lambda c, v: dict(variant=v)) as prepare, \
                 mock.patch.object(suite, "schedule_workers") as schedule, \
                 mock.patch.object(suite, "validate_hardware") as hardware:
                self.assertEqual(suite.run_suite(dict(output_root=tmp), "both", dry_run=True, smoke_limit=None), 0)
            self.assertEqual(prepare.call_count, 2)
            schedule.assert_not_called()
            hardware.assert_not_called()
            for variant in suite.VARIANTS:
                self.assertEqual((root / variant / "plan.json").read_text(), '{"old_four_gpu_plan": true}\n')
                self.assertEqual(json.loads((root / variant / suite.PLAN_FILENAME).read_text()), dict(variant=variant))

    def test_two_gpu_hardware_gate(self) -> None:
        with mock.patch.object(torch.cuda, "device_count", return_value=2), \
             mock.patch.object(torch.cuda, "get_device_name", return_value="RTX PRO 6000D"), \
             mock.patch.object(torch.cuda, "mem_get_info", return_value=(40 * 1024**3, 96 * 1024**3)):
            suite.validate_hardware()
        with mock.patch.object(torch.cuda, "device_count", return_value=4):
            with self.assertRaisesRegex(SftEvaluationError, "2 张"):
                suite.validate_hardware()

    def test_generation_contract_keeps_invalid_rank(self) -> None:
        catalog = index()
        def generate(**kwargs):
            self.assertEqual(kwargs["num_beams"], 10)
            self.assertEqual(kwargs["num_return_sequences"], 10)
            self.assertEqual(kwargs["max_new_tokens"], 7)
            self.assertFalse(kwargs["do_sample"])
            self.assertNotIn("prefix_allowed_tokens_fn", kwargs)
            paths = [[1, 99, 99, 99, 2, 3]] + [[1, 10, 11, 12, 2, 3]] * 9
            sequences = torch.tensor([[8, 9, *path] for path in paths])
            return SimpleNamespace(sequences=sequences, sequences_scores=-torch.arange(1, 11).float())
        model = SimpleNamespace(device="cpu", generate=generate)
        with mock.patch.object(ev, "encode_record", return_value=([8, 9], 0)):
            metrics = ev.evaluate_chunk([{}], variant="a4_nogid", model=model, tokenizer=SimpleNamespace(pad_token_id=0),
                                        template=None, index=catalog, batch_size=32, max_new_tokens=7)
        result = ev.finalize_metrics(metrics)
        self.assertEqual(result["hr@1"], 0)
        self.assertEqual(result["hr@3"], 1)
        self.assertEqual(result["mrr@10"], 0.5)

    def test_chunk_retry_completion_resume_and_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            data.write_text('{"split":"valid"}\n' * 3)
            plan = dict(variant="a4_nogid", source_sha256={}, checkpoint=dict(path=tmp, files={}),
                        config=dict(output_root=tmp, subset_size=3, variants=dict(a4_nogid=dict(max_new_tokens=7)),
                                    decoding=dict(batch_size=32, chunk_size=2)),
                        data=dict(outputs=dict(fixed10k=dict(file=str(data), sha256=sha256_file(data)))))
            calls = []
            def chunk(records, **kwargs):
                calls.append((len(records), kwargs["batch_size"]))
                if len(calls) == 1:
                    raise RuntimeError("CUDA out of memory")
                metrics = ev.empty_metrics()
                for _ in records:
                    ev.update_metrics(metrics, target_row=0, candidates=[ev.Candidate((1,), 0, -1, None)] * 10)
                return metrics
            with mock.patch.object(ev, "verify_worker_inputs"), \
                 mock.patch.object(ev, "load_lf_tokenizer_and_template", return_value=([1, 2, 3], None)), \
                 mock.patch.object(ev, "build_final_id_index"), \
                 mock.patch.object(ev, "load_generation_model", return_value=SimpleNamespace(device="cpu")) as loader, \
                 mock.patch.object(ev, "evaluate_chunk", side_effect=chunk), \
                 mock.patch.object(torch.cuda, "reset_peak_memory_stats"), \
                 mock.patch.object(torch.cuda, "max_memory_allocated", return_value=100):
                result = ev.evaluate_subset(plan, "fixed10k")
                self.assertEqual(calls, [(2, 32), (2, 16), (1, 16)])
                self.assertEqual(result["metrics"]["sample_count"], 3)
                self.assertEqual(ev.evaluate_subset(plan, "fixed10k"), result)
                self.assertEqual(loader.call_count, 1)
                plan["config"]["decoding"]["chunk_size"] = 1
                with self.assertRaisesRegex(SftEvaluationError, "已有结果"):
                    ev.evaluate_subset(plan, "fixed10k")

    def test_macro_uses_only_four_generalization_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(variant="a4_nogid", config=dict(output_root=tmp, subset_size=10000))
            for i, name in enumerate(SUBSETS):
                path = Path(tmp) / "a4_nogid/results" / name / "result.json"
                path.parent.mkdir(parents=True)
                value = 1.0 if i == 0 else 0.25
                metrics = {k: value for k in ("hr@1", "hr@3", "hr@5", "hr@10", "ndcg@10", "mrr@10", "valid_id_rate")}
                path.write_text(json.dumps(dict(status="completed", metrics=dict(sample_count=10000, **metrics),
                    config=dict(plan_signature=signature(plan), subset=name, num_beams=10,
                                decoding="unconstrained_beam_search", smoke_limit=None))))
            output = suite.summarize(plan)
            self.assertEqual(json.loads(output.read_text())["generalization_macro_average"]["hr@10"], 0.25)


if __name__ == "__main__":
    unittest.main()
