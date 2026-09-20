"""Synthetic four-model alignment and completed-result gates."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.sft.evaluate_active_test import METHODS, MODES, scan_aligned_sources, signature, summarize


class ActiveFullTest(unittest.TestCase):
    def test_gnpr_forwards_test_split_and_preserves_valid_default(self):
        from scripts.gnpr import evaluate_retrieval as ev

        for options, expected in [({}, "valid"), ({"split": "test"}, "test")]:
            with patch.object(ev, "encode_gnpr_record", side_effect=ValueError("encoding sentinel")) as encode:
                with self.assertRaisesRegex(ValueError, "encoding sentinel"):
                    ev.evaluate_chunk([{}], model=None, tokenizer=None, template=None, tokens=None,
                                      index=SimpleNamespace(dedup_capacity=4), checkpoint_name="unused",
                                      batch_size=1, cutoff_len=1024, num_beams=10, existing_error_count=0,
                                      **options)
                self.assertEqual(encode.call_args.kwargs["split"], expected)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(rows=3, date="2026-07-14", output_root=str(self.root / "output"), models={})
        for name in METHODS:
            directory = self.root / name
            directory.mkdir()
            records = [dict(split="test", order_id=str(i), searchid=str(i), target_poi_id=str(i), history_length=0,
                            messages=[dict(role="user", content="<CURRENT>query</CURRENT>"),
                                      dict(role="assistant", content=name)]) for i in range(3)]
            path = directory / "test.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            self.config["models"][name] = dict(data_dir=str(directory), test_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_alignment_accepts_different_identifiers_and_last_partial_chunk(self):
        data = scan_aligned_sources(self.config)
        self.assertEqual(data['rows'], 3)
        self.assertEqual(set(data['sources']), set(METHODS))

    def test_alignment_rejects_wrong_target_current_split_or_row_count(self):
        path = self.root / METHODS[-1] / 'test.jsonl'
        original = path.read_text()
        for field, value in [('target_poi_id', 'wrong'), ('split', 'valid')]:
            rows = list(map(json.loads, original.splitlines()))
            rows[1][field] = value
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(ValueError):
                scan_aligned_sources(self.config)
        path.write_text(original.replace('query', 'another'))
        with self.assertRaises(ValueError):
            scan_aligned_sources(self.config)
        path.write_text(original.splitlines()[0] + '\n')
        with self.assertRaises(ValueError):
            scan_aligned_sources(self.config)

    def test_summary_rejects_smoke_partial_and_illegal_constrained_results(self):
        plan = dict(data={'business_keys_sha256': 'keys'})
        for name in METHODS:
            metrics = {f'{kind}@{k}': .5 for kind in ('hr', 'ndcg') for k in (1, 3, 5, 10)}
            metrics.update(sample_count=3, valid_id_rate=1.0, generation_validity={'valid_pid_ratio': 1.0})
            payload = dict(status='completed', method=name, plan_signature=signature(plan), decoding=MODES[name],
                           split='test', date=self.config['date'], smoke_limit=None,
                           result=dict(status='completed', metrics=metrics))
            path = Path(self.config['output_root']) / name / 'full/suite_result.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload))
        self.assertEqual(len(summarize(self.config, plan, None)['results']), 4)
        path = Path(self.config['output_root']) / 'qg_hrq_gid/full/suite_result.json'
        original = json.loads(path.read_text())
        for change in ('smoke', 'partial', 'invalid'):
            payload = copy.deepcopy(original)
            if change == 'smoke': payload['smoke_limit'] = 3
            elif change == 'partial': payload['result']['metrics']['sample_count'] = 2
            else: payload['result']['metrics']['valid_id_rate'] = .99
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                summarize(self.config, plan, None)


if __name__ == '__main__':
    unittest.main()
