from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sft import a0_evaluation as a0
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft.evaluation_data import SftEvaluationError, SUBSETS, signature
from qg_prqk.sid.identifiers import identifier_content, identifier_key


class A0EvaluationTest(unittest.TestCase):
    def test_gid_first_catalog_uses_a0_provenance_and_last_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.array([[0] * 6 + [1, 2, 3], [1] * 6 + [4, 5, 6]], dtype=np.int32)
            dedup = np.array([-1, 0], dtype=np.int32)
            for name, values in [('base_identifier_codes.npy', base), ('dedup_codes.npy', dedup),
                                 ('requires_dedup.npy', dedup >= 0)]:
                np.save(root / name, values)
            contents = [identifier_content(base[i], int(dedup[i]), variant='a4_gid_parent') for i in range(2)]
            tokens = {'<TARGET_POI>': 1, '</TARGET_POI>': 2}
            for content in contents:
                for token in a0.re.findall(r'<[^>]+>', content):
                    if token not in tokens:
                        tokens[token] = len(tokens) + 10
            write_json_atomic(root / 'qg_prqk_token_mapping.json', {'tokens': tokens})
            pq.write_table(pa.Table.from_pylist([
                dict(poi_row_index=i, poi_id=f'p{i}',
                     final_id_key=identifier_key(base[i], int(dedup[i]), variant='a4_gid_parent')) for i in range(2)
            ]), root / 'poi_final_id_mapping.parquet')
            manifest = dict(variant='a0_gid', status='completed', artifacts={
                p.name: {'sha256': sha256_file(p)} for p in root.iterdir()
                if p.suffix in ('.npy', '.parquet')})
            write_json_atomic(root / 'manifest.json', manifest)
            config = dict(tokenizer=str(root), variants={'a0_gid': dict(identifier_dir=str(root),
                          identifier_manifest_sha256=sha256_file(root / 'manifest.json'))})
            tokenizer = SimpleNamespace(eos_token_id=3, encode=lambda token, **_: [tokens[token]])
            index = a0.build_index(config, tokenizer, expected_pois=2)
            self.assertEqual(index.base_width, 9)
            self.assertEqual(len(index.tokens_by_poi['p0']), 9)
            self.assertEqual(len(index.tokens_by_poi['p1']), 10)
            self.assertEqual(index.tokens_by_poi['p1'][-1], tokens['<D_0>'])
            self.assertEqual(index.tokens_by_poi['p1'][6], tokens['<S1_4>'])
            self.assertEqual(a0.FinalIdPrefixIndex(index).max_length, 13)

    def test_both_modes_reuse_same_chunk_scorer_and_resume_completed(self):
        for mode in a0.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                plan = dict(config={'output_root': tmp, 'decoding': {'batch_size': 32, 'chunk_size': 500}},
                            data={'outputs': {'fixed10k': {'file': '/synthetic', 'sha256': 'synthetic'}}})
                metrics = ev.empty_metrics()
                for _ in range(2):
                    ev.update_metrics(metrics, target_row=0, candidates=[ev.Candidate((4,), 0, -1., None)] * 10)
                with mock.patch.object(a0, 'require_hash'), \
                     mock.patch.dict(sys.modules, {'torch': mock.Mock()}), \
                     mock.patch.object(a0, 'read_records', return_value=[{}] * 10000), \
                     mock.patch.object(ev, 'evaluate_chunk', return_value=metrics) as chunk:
                    kwargs = dict(model=None, tokenizer=None, template=None, index=None, smoke_limit=2)
                    result = a0.evaluate_subset(plan, mode, 'fixed10k', **kwargs)
                    self.assertEqual(result['metrics']['sample_count'], 2)
                    self.assertEqual(result['metrics']['hr@1'], 1.)
                    self.assertEqual(chunk.call_args.kwargs['variant'], 'a0_gid')
                    self.assertEqual(chunk.call_args.kwargs['max_new_tokens'], 13)
                    self.assertEqual(result, a0.evaluate_subset(plan, mode, 'fixed10k', **kwargs))
                    self.assertEqual(chunk.call_count, 1)
                    with self.assertRaises(SftEvaluationError):
                        a0.evaluate_subset({**plan, 'changed': True}, mode, 'fixed10k', **kwargs)

    def test_constraint_rejects_invalid_candidates_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(config={'output_root': tmp, 'decoding': {'batch_size': 32, 'chunk_size': 500}},
                        data={'outputs': {'fixed10k': {'file': '/synthetic', 'sha256': 'synthetic'}}})
            metrics = ev.empty_metrics()
            ev.update_metrics(metrics, target_row=0, candidates=[ev.Candidate(None, None, -1., 'corpus_miss')] * 10)
            with mock.patch.object(a0, 'require_hash'), mock.patch.dict(sys.modules, {'torch': mock.Mock()}), \
                 mock.patch.object(a0, 'read_records', return_value=[{}] * 10000), \
                 mock.patch.object(ev, 'evaluate_chunk', return_value=metrics):
                with self.assertRaises(SftEvaluationError):
                    a0.evaluate_subset(plan, 'constrained', 'fixed10k', model=None, tokenizer=None,
                                       template=None, index=None, smoke_limit=1)
            self.assertFalse(list(Path(tmp).rglob('progress.json')))

    def test_two_gpu_schedule_isolates_modes_and_forwards_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def start(command, **kwargs):
                calls.append((command, kwargs))
                return SimpleNamespace(pid=100 + len(calls), stdout=io.StringIO('500/10000\n'),
                                       wait=lambda **_: 0, poll=lambda: 0)
            plan = {'config': {'run_control': tmp, 'output_root': tmp}}
            with mock.patch.object(a0.subprocess, 'Popen', side_effect=start), \
                 mock.patch.dict(a0.os.environ, {'CUDA_VISIBLE_DEVICES': '2,5'}):
                a0.schedule(plan, 2)
            self.assertEqual([kw['env']['CUDA_VISIBLE_DEVICES'] for _, kw in calls], ['2', '5'])
            self.assertEqual([cmd[cmd.index('--worker-mode')+1] for cmd, _ in calls], list(a0.MODES))
            self.assertTrue(all(cmd[-2:] == ['--smoke-limit', '2'] for cmd, _ in calls))
            self.assertTrue(all(len(kw['env']['TMPDIR'].encode()) <= 64 for _, kw in calls))

    def test_single_a100_schedule_runs_modes_sequentially_on_same_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, active = [], 0
            def start(command, **kwargs):
                nonlocal active
                self.assertEqual(active, 0)
                active = 1
                calls.append((command, kwargs))
                def wait(**_):
                    nonlocal active
                    active = 0
                    return 0
                return SimpleNamespace(pid=200 + len(calls), stdout=io.StringIO('500/10000\n'),
                                       wait=wait, poll=lambda: 0)
            plan = {'config': {'run_control': tmp, 'output_root': tmp}}
            with mock.patch.object(a0.subprocess, 'Popen', side_effect=start), \
                 mock.patch.dict(a0.os.environ, {'CUDA_VISIBLE_DEVICES': '7'}):
                a0.schedule(plan, 2, single_a100=True)
            self.assertEqual([kw['env']['CUDA_VISIBLE_DEVICES'] for _, kw in calls], ['7', '7'])
            self.assertEqual([cmd[cmd.index('--worker-mode') + 1] for cmd, _ in calls], list(a0.MODES))
            self.assertEqual(active, 0)

    def test_single_a100_hardware_gate(self):
        cuda = SimpleNamespace(device_count=lambda: 1,
                               get_device_name=lambda _: 'NVIDIA A100-SXM4-40GB',
                               mem_get_info=lambda _: (35 * 1024**3, 40 * 1024**3))
        with mock.patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            a0.validate_hardware(single_a100=True)
        cuda.get_device_name = lambda _: 'NVIDIA RTX PRO 6000'
        with mock.patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(SftEvaluationError, 'A100'):
                a0.validate_hardware(single_a100=True)

    def test_worker_loads_once_and_visits_five_frozen_subsets(self):
        plan = {'checkpoint': {'path': '/synthetic'}, 'config': {}}
        with mock.patch.object(a0, 'verify_plan'), mock.patch.object(a0, 'build_index'), \
             mock.patch.object(ev, 'load_lf_tokenizer_and_template', return_value=([1], None)), \
             mock.patch.object(ev, 'load_generation_model') as load, mock.patch.object(a0, 'evaluate_subset') as evaluate:
            a0.worker(plan, 'unconstrained', 2)
        self.assertEqual(load.call_count, 1)
        self.assertEqual([call.args[2] for call in evaluate.call_args_list], list(SUBSETS))

    def test_summary_excludes_fixed_from_macro_and_keeps_modes_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {'config': {'output_root': tmp}}
            for mode in a0.MODES:
                for i, subset in enumerate(SUBSETS):
                    settings = dict(plan_signature=signature(plan), decoding=mode, subset=subset, smoke_limit=None)
                    result = dict(status='completed', config=settings,
                                  metrics=dict(sample_count=10000, **{k: (1. if i == 0 else .2) for k in a0.METRICS}))
                    result['metrics']['valid_id_rate'] = 1.
                    write_json_atomic(Path(tmp) / mode / 'results' / subset / 'result.json', result)
            result = a0.summarize(plan, None)
            self.assertEqual(len(result['rows']), 10)
            self.assertEqual(result['generalization_macro_average']['constrained']['hr@1'], .2)
            self.assertTrue((Path(tmp) / 'results/summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
