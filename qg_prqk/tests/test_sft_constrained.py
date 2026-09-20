from __future__ import annotations

import itertools
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from qg_prqk.sft import evaluation as ev
from qg_prqk.sft import constrained_evaluation as suite
from qg_prqk.sft.constrained_decoding import (
    ConstrainedGenerationModel,
    FinalIdPrefixIndex,
    PrefixConstraint,
)
from qg_prqk.sft.evaluation_data import SftEvaluationError, SUBSETS, signature


def catalog(width: int = 3) -> ev.FinalIdIndex:
    paths = [
        tuple(range(10, 10 + width)),
        tuple(range(20, 20 + width)) + (50,),
        tuple(range(20, 20 + width)) + (51,),
    ]
    return ev.FinalIdIndex(
        {p: i for i, p in enumerate(paths)},
        {str(i): p for i, p in enumerate(paths)},
        1,
        2,
        3,
        width,
        len(paths),
    )


class PrefixTest(unittest.TestCase):
    def test_every_prefix_matches_brute_force_both_variants(self) -> None:
        for width in (3, 9):
            trie = FinalIdPrefixIndex(catalog(width))
            prefixes = {p[:i] for p in trie.paths for i in range(len(p))}
            for prefix in prefixes:
                expected = sorted(
                    {p[len(prefix)] for p in trie.paths if p[: len(prefix)] == prefix}
                )
                self.assertEqual(list(trie.allowed_next(prefix)), expected)
            for path in trie.paths:
                self.assertEqual(trie.allowed_next(path), (3,))
                self.assertIsNone(ev.parse_candidate(path, -1.0, catalog(width)).error)
            self.assertLessEqual(trie.allowed_next.cache_info().currsize, 65536)

    def test_singleton_closes_and_collision_requires_last_dedup(self) -> None:
        trie = FinalIdPrefixIndex(catalog())
        self.assertEqual(trie.allowed_next((1, 10, 11, 12)), (2,))
        self.assertEqual(trie.allowed_next((1, 20, 21, 22)), (50, 51))
        self.assertEqual(trie.allowed_next((1, 20, 21, 22, 50)), (2,))
        for invalid in (
            (1, 10, 11, 12, 50),
            (1, 20, 21, 22, 2),
            (1, 20, 11),
            (1, 3),
            (9,),
            (1, 10, 11, 12, 2, 3, 4),
        ):
            with self.assertRaises(SftEvaluationError):
                trie.allowed_next(invalid)

    def test_left_padding_and_batch_id_do_not_prune_catalog(self) -> None:
        trie = FinalIdPrefixIndex(catalog())
        callback = PrefixConstraint(trie, 4)
        self.assertEqual(callback(0, [0, 0, 100, 101, 1]), [10, 20])
        self.assertEqual(callback(1, [7, 8, 9, 10, 1]), [10, 20])

    def test_wrapper_changes_only_prefix_callback(self) -> None:
        model = SimpleNamespace(
            device="cpu", generate=mock.Mock(return_value="generated")
        )
        proxy = ConstrainedGenerationModel(model, FinalIdPrefixIndex(catalog()))
        kwargs = dict(
            input_ids=SimpleNamespace(shape=(2, 6)),
            max_new_tokens=7,
            num_beams=10,
            renormalize_logits=True,
        )
        self.assertEqual(proxy.generate(**kwargs), "generated")
        actual = model.generate.call_args.kwargs
        self.assertEqual(
            {k: v for k, v in actual.items() if k != "prefix_allowed_tokens_fn"}, kwargs
        )
        self.assertEqual(actual["prefix_allowed_tokens_fn"].prompt_width, 6)
        with self.assertRaises(SftEvaluationError):
            proxy.generate(**{**kwargs, "max_new_tokens": 6})

    def test_real_transformers_beam_search_on_tiny_cpu_model(self) -> None:
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(42)
        torch.set_num_threads(2)
        for width in (3, 9):
            paths = [
                tuple([10] * (width - 2) + [20 + a, 30 + b]) + ((50,) if a else ())
                for a, b in itertools.product(range(3), range(4))
            ]
            index = ev.FinalIdIndex(
                {p: i for i, p in enumerate(paths)},
                {str(i): p for i, p in enumerate(paths)},
                1,
                2,
                3,
                width,
                len(paths),
            )
            model = Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    eos_token_id=3,
                    pad_token_id=0,
                )
            ).eval()
            proxy = ConstrainedGenerationModel(model, FinalIdPrefixIndex(index))
            with torch.inference_mode():
                result = proxy.generate(
                    input_ids=torch.tensor([[0, 7, 8], [9, 7, 8]]),
                    attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
                    max_new_tokens=width + 4,
                    num_beams=10,
                    num_return_sequences=10,
                    do_sample=False,
                    length_penalty=1.0,
                    early_stopping=True,
                    renormalize_logits=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=3,
                    pad_token_id=0,
                )
            self.assertEqual(len(result.sequences), 20)
            for sequence, score in zip(
                result.sequences[:, 3:].tolist(), result.sequences_scores.tolist()
            ):
                self.assertIsNone(ev.parse_candidate(sequence, score, index).error)


class SuiteTest(unittest.TestCase):
    def test_scheduler_uses_two_distinct_devices_and_forwards_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = {
                v: {
                    "config": {"output_root": str(root / "results")},
                    "run_control": str(root / "logs"),
                }
                for v in suite.VARIANTS
            }
            workers = [
                SimpleNamespace(
                    pid=i,
                    stdout=io.StringIO("synthetic progress\n"),
                    wait=lambda: 0,
                    poll=lambda: 0,
                )
                for i in (101, 102)
            ]
            with (
                mock.patch.object(
                    suite, "resolve", side_effect=lambda p: root / Path(p).name
                ),
                mock.patch.object(
                    suite.os, "fsencode", return_value=b"/synthetic-short-tmp"
                ),
                mock.patch.dict(suite.os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}),
                mock.patch.object(
                    suite.subprocess, "Popen", side_effect=workers
                ) as spawn,
            ):
                suite.schedule(plans, 2, fixed_only=True)
            self.assertEqual(
                [
                    call.kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                    for call in spawn.call_args_list
                ],
                ["2", "5"],
            )
            for call in spawn.call_args_list:
                self.assertIn("--fixed-only", call.args[0])
                self.assertEqual(call.args[0][-3:-1], ["--smoke-limit", "2"])

    def test_scheduler_rejects_long_temp_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = {
                v: {"config": {"output_root": str(root)}, "run_control": str(root)}
                for v in suite.VARIANTS
            }
            with (
                mock.patch.object(suite, "resolve", return_value=root / ("x" * 65)),
                mock.patch.dict(suite.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}),
                mock.patch.object(suite.subprocess, "Popen") as spawn,
            ):
                with self.assertRaisesRegex(SftEvaluationError, "64"):
                    suite.schedule(plans, 2)
                spawn.assert_not_called()

    def test_rejects_legacy_output(self) -> None:
        for path in (
            "qg_prqk/outputs/eval/sft_epoch3_fixed10k_generalization_v1",
            "/tmp/constrained",
        ):
            with self.assertRaises(SftEvaluationError):
                suite.isolated_output(path)

    def test_existing_json_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            suite.immutable_json(path, {"a": 1})
            suite.immutable_json(path, {"a": 1})
            with self.assertRaises(SftEvaluationError):
                suite.immutable_json(path, {"a": 2})

    def test_failed_smoke_prevents_full_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("output_root: unused\n")
            with (
                mock.patch.object(
                    suite, "isolated_output", return_value=Path(directory)
                ),
                mock.patch.object(suite, "prepare_plans", return_value={}),
                mock.patch.object(suite, "validate_hardware"),
                mock.patch.object(
                    suite, "schedule", side_effect=SftEvaluationError("smoke failed")
                ) as schedule,
            ):
                self.assertEqual(suite.main(["--config", str(config)]), 2)
                schedule.assert_called_once_with({}, 2, fixed_only=True)

    def test_summary_excludes_fixed_set_and_compares_matched_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = {}
            for variant in suite.VARIANTS:
                refs = {}
                for subset in SUBSETS:
                    old = root / variant / f"{subset}_old.json"
                    old.parent.mkdir(parents=True, exist_ok=True)
                    old.write_text(
                        json.dumps({"metrics": {m: 0.2 for m in suite.METRICS}})
                    )
                    refs[subset] = dict(path=str(old), sha256=suite.sha256_file(old))
                plan = dict(
                    variant=variant,
                    config={"output_root": str(root)},
                    baseline_results=refs,
                )
                plans[variant] = plan
                for subset in SUBSETS:
                    settings = dict(
                        plan_signature=signature(plan),
                        decoding=suite.MODE,
                        subset=subset,
                        variant=variant,
                        smoke_limit=None,
                    )
                    metrics = {
                        m: 0.9 if subset == "fixed10k" else 0.4 for m in suite.METRICS
                    }
                    metrics.update(sample_count=10000, valid_id_rate=1.0)
                    path = root / variant / "results" / subset / "result.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(
                            dict(
                                status="completed",
                                config=settings,
                                signature=signature(settings),
                                metrics=metrics,
                            )
                        )
                    )
            output = json.loads(suite.summarize(plans, None).read_text())
            self.assertEqual(len(output["rows"]), 10)
            self.assertEqual(
                output["generalization_macro_average"][suite.VARIANTS[0]]["hr@10"], 0.4
            )
            self.assertAlmostEqual(
                output["rows"][0]["delta_constrained_minus_unconstrained"]["hr@10"], 0.7
            )


if __name__ == "__main__":
    unittest.main()
