"""Synthetic contracts for A0 full Test remapping, resume and dual-GPU isolation."""
from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from qg_prqk.artifacts import sha256_file
from qg_prqk.sft import a0_full_test as full, evaluation as ev
from qg_prqk.sft.full_test_data import file_stat

OLD = '<G_0>' * 6 + '<S1_1><S2_2><S3_3>'
NEW = '<G_0>' * 6 + '<S1_4><S2_5><S3_6><D_0>'


def record(number: int = 0) -> dict:
    return dict(split='test', identifier_variant='a4_gid_parent', order_id=str(number), searchid='s',
                target_poi_id='p', history_length=1, requires_dedup=False,
                messages=[dict(role='user', content=f'<HISTORY><POI_QGPRQK_ID>{OLD}</POI_QGPRQK_ID></HISTORY><CURRENT>same query</CURRENT>'),
                          dict(role='assistant', content=f'<TARGET_POI>{OLD}</TARGET_POI>')])


def remap(row: dict) -> dict:
    return full.remap_test_record(row, old_rows={OLD: 0}, row_by_poi={'p': 0}, content=lambda _: NEW,
                                 key=lambda _: 'new-key', requires_dedup=[True])


def plan(root: Path) -> dict:
    return dict(checkpoint=dict(path='/synthetic'), data=dict(business_keys_sha256='synthetic'),
                config=dict(output_root=str(root), run_control=str(root / 'logs'), source_rows=3,
                            date='2026-07-14', source=dict(sha256='synthetic'),
                            decoding=dict(batch_size=32, chunk_size=2)))


def metrics(count: int, *, valid: bool = True) -> dict:
    result = ev.empty_metrics()
    for _ in range(count):
        candidate = ev.Candidate((4,), 0, -1., None) if valid else ev.Candidate(None, None, -1., 'corpus_miss')
        ev.update_metrics(result, target_row=0, candidates=[candidate] * 10)
    return result


class A0FullTestTest(unittest.TestCase):
    def test_remap_changes_all_identifiers_and_preserves_request(self):
        original = record()
        snapshot = copy.deepcopy(original)
        actual = remap(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(actual['split'], 'test')
        self.assertEqual(actual['identifier_variant'], 'a0_gid')
        self.assertTrue(actual['requires_dedup'])
        for key in ['order_id', 'searchid', 'target_poi_id', 'history_length']:
            self.assertEqual(actual[key], original[key])
        self.assertIn(NEW, actual['messages'][0]['content'])
        self.assertEqual(actual['messages'][1]['content'], f'<TARGET_POI>{NEW}</TARGET_POI>')
        self.assertEqual(full.current_context(actual), full.current_context(original))

    def test_remap_rejects_wrong_split_target_or_history(self):
        for patch in [dict(split='valid'), dict(identifier_variant='a0_gid'), dict(target_poi_id='missing'), dict(history_length=0)]:
            with self.assertRaises(ValueError):
                remap({**record(), **patch})
        bad = record()
        bad['messages'][0]['content'] = bad['messages'][0]['content'].replace('<S1_1>', '<S1_2>')
        with self.assertRaises(ValueError):
            remap(bad)
        bad = record()
        bad['messages'][1]['content'] = '<TARGET_POI>wrong</TARGET_POI>'
        with self.assertRaises(ValueError):
            remap(bad)

    def test_stream_scan_hashes_all_rows_and_rejects_duplicates_or_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            def source(rows, expected=3):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                return dict(file=str(path), rows=expected, sha256=sha256_file(path), **file_stat(path))
            spec = source([record(i) for i in range(3)])
            inspector = mock.Mock()
            receipt = full.scan_source(spec, remap, inspect=inspector)
            self.assertEqual((receipt['rows'], receipt['history_events']), (3, 3))
            self.assertEqual(inspector.call_count, 3)
            view = full.A0TestRecords(spec, remap)
            self.assertEqual(view[1:3], [remap(record(1)), remap(record(2))])
            self.assertEqual(view[-1], remap(record(2)))
            for rows in [[record(0)]*3, [record(0)], [record(i) for i in range(4)]]:
                with self.assertRaises(ValueError):
                    full.scan_source(source(rows), remap)
            spec = source([record(i) for i in range(3)])
            spec['sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                full.scan_source(spec, remap)

    def test_chunk_oom_and_interruption_resume_without_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = plan(root)
            calls = []
            def chunk(rows, **kwargs):
                calls.append((len(rows), kwargs['batch_size'], kwargs['variant']))
                if len(calls) == 1:
                    raise RuntimeError('CUDA out of memory')
                if len(calls) == 3:
                    raise RuntimeError('synthetic interruption')
                return metrics(len(rows))
            with mock.patch.dict(sys.modules, {'torch': mock.Mock()}), mock.patch.object(full, 'evaluate_test_chunk', side_effect=chunk):
                args = dict(records=[{}]*3, model=None, tokenizer=None, template=None, index=None, smoke_limit=None)
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    full.evaluate_test(frozen, 'constrained', **args)
                progress = full.load_json(root / 'constrained/results/progress.json')
                self.assertEqual(progress['next_line'], 2)
                result = full.evaluate_test(frozen, 'constrained', **args)
                self.assertEqual(result['metrics']['sample_count'], 3)
                self.assertEqual(full.evaluate_test(frozen, 'constrained', **args), result)
                self.assertEqual(calls, [(2,32,'a0_gid'),(2,16,'a0_gid'),(1,16,'a0_gid'),(1,16,'a0_gid')])
                with self.assertRaises(ValueError):
                    full.evaluate_test({**frozen, 'changed': True}, 'constrained', **args)

    def test_invalid_beams_keep_ranks_only_unconstrained_and_smoke_is_isolated(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {'torch': mock.Mock()}):
            frozen = plan(Path(tmp)); invalid = metrics(1, valid=False)
            args = dict(records=[{}]*3, model=None, tokenizer=None, template=None, index=None, smoke_limit=1)
            with mock.patch.object(full, 'evaluate_test_chunk', return_value=invalid):
                with self.assertRaises(ValueError):full.evaluate_test(frozen, 'constrained', **args)
                result = full.evaluate_test(frozen, 'unconstrained', **args)
            self.assertEqual(result['metrics']['valid_id_rate'], 0)
            self.assertTrue((Path(tmp)/'unconstrained/smoke1/result.json').exists())
            self.assertFalse((Path(tmp)/'unconstrained/results/result.json').exists())
            self.assertFalse((Path(tmp)/'constrained/smoke1/progress.json').exists())

    def test_summary_requires_two_matching_modes_and_full_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = plan(root)
            for mode in full.MODES:
                settings = full.run_settings(frozen, mode, None)
                full.immutable_json(root/mode/'results/result.json',dict(status='completed',config=settings,
                    signature=full.signature(settings),metrics=ev.finalize_metrics(metrics(3))))
            result = full.summarize(frozen, None)
            self.assertEqual([r['sample_count'] for r in result['rows']], [3,3])
            self.assertEqual(result['delta_constrained_minus_unconstrained']['hr@10'], 0)
            with self.assertRaises(ValueError):full.summarize(frozen, 2)
            changed = copy.deepcopy(frozen);changed['config']['source_rows']=4
            with self.assertRaises(ValueError):full.summarize(changed,None)

    def test_dual_gpu_workers_use_test_entry_and_separate_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def start(command, **kwargs):
                calls.append((command,kwargs['env']))
                return SimpleNamespace(pid=100+len(calls),stdout=io.StringIO('Test progress\n'),wait=lambda **_:0,poll=lambda:0)
            with mock.patch.object(full.subprocess,'Popen',side_effect=start),mock.patch.dict(full.os.environ,{'CUDA_VISIBLE_DEVICES':'2,5'}):
                full.schedule(plan(Path(tmp)),2)
            self.assertEqual([env['CUDA_VISIBLE_DEVICES'] for _,env in calls],['2','5'])
            self.assertEqual([cmd[cmd.index('--worker-mode')+1] for cmd,_ in calls],list(full.MODES))
            self.assertTrue(all(cmd[1].endswith('evaluate_a0_gid_test.py') for cmd,_ in calls))

    def test_failed_worker_terminates_only_its_peer(self):
        with tempfile.TemporaryDirectory() as tmp:
            failed=SimpleNamespace(pid=101,stdout=io.StringIO('failed\n'),wait=lambda **_:2,poll=lambda:2)
            peer=SimpleNamespace(pid=102,stdout=io.StringIO('peer\n'),wait=mock.Mock(return_value=-15),poll=lambda:None,terminate=mock.Mock())
            with mock.patch.object(full.subprocess,'Popen',side_effect=[failed,peer]),mock.patch.dict(full.os.environ,{'CUDA_VISIBLE_DEVICES':'0,1'}):
                with self.assertRaises(ValueError):full.schedule(plan(Path(tmp)),2)
            peer.terminate.assert_called_once()

    def test_dry_run_and_prepare_only_do_not_start_gpu(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(full,'load_protocol',return_value=plan(Path(tmp))['config']), \
             mock.patch.object(full,'prepare') as prepare, mock.patch.object(full.a0,'validate_hardware') as hardware, \
             mock.patch.object(full,'schedule') as schedule:
            with mock.patch.object(sys,'argv',['test','--dry-run']):self.assertEqual(full.main(),0)
            prepare.assert_not_called()
            with mock.patch.object(sys,'argv',['test','--prepare-only']):self.assertEqual(full.main(),0)
            prepare.assert_called_once();hardware.assert_not_called();schedule.assert_not_called()


if __name__ == '__main__':
    unittest.main()
