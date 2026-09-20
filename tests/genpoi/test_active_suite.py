"""Synthetic contracts for the active five-set evaluation launcher."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.genpoi.evaluate_active_suite import METRICS, SUBSETS, subset_command, summarize


class ActiveSuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = dict(output=self.root, checkpoint=Path('/checkpoint-8868'),
                           tokenizer=Path('/tokenizer'), valid=Path('/valid.jsonl'), trie=Path('/trie'),
                           references={name: {'path': Path(f'/{name}.jsonl'), 'sha256': name} for name in SUBSETS})
        self.payloads = {}
        for name in SUBSETS:
            config = dict(checkpoint='/checkpoint-8868', epoch=3.0, cutoff_len=1024,
                          constraint_mode='tcg_ssp', ssp_gamma=2, num_beams=10,
                          num_return_sequences=10, top_k=10, smoke_limit=None,
                          trie_dir='/trie', tokenizer='/tokenizer',
                          dataset_context={'reference_sha256': name},
                          checkpoint_model_sha256='model', tokenizer_json_sha256='tokenizer',
                          trie_manifest_sha256='trie', ssp_head_sha256='head')
            metrics = {key: 1.0 if name == 'fixed10k' else 0.25 for key in METRICS}
            metrics.update(sample_count=10000, generation_validity={'valid_pid_ratio': 1.0})
            self.payloads[name] = dict(status='completed', results=[dict(
                status='completed', covered_rows=10000, config=config, metrics=metrics)])
        self.write()

    def write(self):
        for name, payload in self.payloads.items():
            path = self.root / name / 'valid_checkpoint_results.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(payload))

    def test_macro_excludes_fixed_traffic(self):
        result = summarize(self.inputs)
        self.assertEqual(len(result['results']), 5)
        self.assertEqual(result['generalization_macro']['hr@10'], 0.25)

    def test_incomplete_wrong_reference_and_protocol_are_rejected(self):
        original = copy.deepcopy(self.payloads)
        for field, value in [('covered_rows', 9999), ('status', 'running')]:
            self.payloads = copy.deepcopy(original)
            self.payloads['fixed10k']['results'][0][field] = value
            self.write()
            with self.assertRaises(ValueError):
                summarize(self.inputs)
        for field, value in [('cutoff_len', 512), ('ssp_head_sha256', 'other'),
                             ('dataset_context', {'reference_sha256': 'wrong'}), ('smoke_limit', 100)]:
            self.payloads = copy.deepcopy(original)
            self.payloads['fixed10k']['results'][0]['config'][field] = value
            self.write()
            with self.assertRaises(ValueError):
                summarize(self.inputs)

    def test_each_subset_has_isolated_output_and_frozen_reference(self):
        for name in SUBSETS:
            command = subset_command(self.inputs, name)
            self.assertEqual(command[command.index('--output-dir') + 1], str(self.root / name))
            self.assertEqual(command[command.index('--reference-validation-subset') + 1], f'/{name}.jsonl')
            self.assertEqual(command[command.index('--expected-step') + 1], '8868')


if __name__ == '__main__':
    unittest.main()
