"""Resume and final partial-chunk contracts for constrained full Test."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qg_prqk.sft import constrained_full_test as suite
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft.evaluation_data import SftEvaluationError, signature


class ConstrainedFullTest(unittest.TestCase):
    def test_rejects_wrong_signature_count_and_invalid_candidates(self):
        settings = {'sample_count': 5}
        progress = dict(signature=signature(settings), next_line=0, metrics=ev.empty_metrics(), actual_batch_size=32)
        suite.validate_progress(progress, settings)
        for key, value in [('signature', 'wrong'), ('next_line', 1), ('actual_batch_size', 0)]:
            with self.assertRaises(SftEvaluationError):
                suite.validate_progress({**progress, key: value}, settings)

    def test_last_partial_chunk_and_completed_reuse(self):
        import torch
        config = dict(date='2026-07-14', decoding={'chunk_size': 2},
                      variants={'a4_nogid': {'checkpoint': 'unused', 'max_new_tokens': 7}})
        source = dict(file='unused', rows=5, sha256='source')
        sizes = []

        def chunk(records, **kwargs):
            self.assertIsInstance(kwargs['model'], suite.ConstrainedGenerationModel)
            sizes.append(len(records))
            metrics = ev.empty_metrics()
            for _ in records:
                ev.update_metrics(metrics, target_row=0,
                                  candidates=[ev.Candidate(codes=(1, 2, 3), poi_row=0, score=0.0, error=None)] * 10)
            return metrics

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(suite, 'JsonlRecordSequence', return_value=[{}] * 5), \
                patch.object(ev, 'load_lf_tokenizer_and_template', return_value=([1], None)), \
                patch.object(ev, 'build_final_id_index', return_value=None), \
                patch.object(suite, 'FinalIdPrefixIndex', return_value=SimpleNamespace(max_length=7)), \
                patch.object(ev, 'load_generation_model', return_value=SimpleNamespace(device='cpu')) as loader, \
                patch.object(suite, 'evaluate_test_chunk', side_effect=chunk), \
                patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
            args = dict(variant='a4_nogid', source=source, output=Path(tmp), plan_signature='plan')
            result = suite.evaluate_constrained_test(config, **args)
            self.assertEqual(sizes, [2, 2, 1])
            self.assertEqual(result['metrics']['sample_count'], 5)
            self.assertEqual(result['metrics']['valid_id_rate'], 1.0)
            self.assertEqual(suite.evaluate_constrained_test(config, **args), result)
            self.assertEqual(loader.call_count, 1)


if __name__ == '__main__':
    unittest.main()
