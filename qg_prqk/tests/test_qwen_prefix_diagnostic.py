"""Check shifted teacher positions, legal masks and original beam ranks."""
import unittest
from types import SimpleNamespace

from qg_prqk.sft import prefix_diagnostic as diagnostic
from qg_prqk.sft.constrained_decoding import FinalIdPrefixIndex


class PrefixDiagnosticTest(unittest.TestCase):
    def test_selection_depends_on_business_key_not_identifier(self):
        rows = [dict(order_id=str(i), searchid=str(i), target='a') for i in range(10000)]
        changed = [{**r, 'target': 'b'} for r in rows]
        self.assertEqual(diagnostic.select_indices(rows, 100), diagnostic.select_indices(changed, 100))
        self.assertEqual(len(set(diagnostic.select_indices(rows, 100))), 100)
        with self.assertRaises(ValueError):
            diagnostic.select_indices(rows[:-1], 100)

    def test_prefix_keeps_gid_errors_and_original_beam_position(self):
        target = list(range(12))
        wrong = target.copy()
        wrong[1] = 999
        result = diagnostic.beam_prefixes(target, [wrong, target])
        self.assertFalse(result['gid6_s1']['1'])
        self.assertTrue(result['final_id']['10'])
        self.assertEqual(diagnostic.target_stages(target)[7:10], ['s1', 's2', 's3'])
        self.assertEqual(diagnostic.target_stages(list(range(13)))[10], 'dedup')

    def test_masking_changes_rank_and_probability_not_target(self):
        import torch
        observed = diagnostic.token_observation(torch.tensor([4., 3., 2., 1.]), 2, [2, 3])
        self.assertEqual(observed['rank'], 3)
        self.assertEqual(observed['legal_rank'], 1)
        self.assertGreater(observed['nll'], observed['legal_nll'])
        self.assertLess(observed['legal_mass'], 1)
        with self.assertRaises(ValueError):
            diagnostic.token_observation(torch.ones(4), 2, [0, 1])

    def test_real_qwen_teacher_matches_individual_unpadded_next_token(self):
        import torch
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                                            num_hidden_layers=1, num_attention_heads=4,
                                            num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                                            pad_token_id=0, eos_token_id=0)).eval()
        paths = [tuple(range(3, 12)), tuple(range(3, 11)) + (12, 13)]
        index = SimpleNamespace(row_by_tokens={p: i for i, p in enumerate(paths)}, poi_count=2,
                                target_open=1, target_close=2, eos=0, base_width=9)
        trie = FinalIdPrefixIndex(index)
        prompts = [[20, 21], [20, 21, 22, 23, 24]]
        targets = [[1, *p, 2, 0] for p in paths]
        batch = diagnostic.teacher_forced(model, SimpleNamespace(pad_token_id=0), trie, prompts, targets)
        for row, (prompt, target) in enumerate(zip(prompts, targets)):
            for j, stage in enumerate(diagnostic.target_stages(target)):
                with torch.inference_mode():
                    logits = model(input_ids=torch.tensor([prompt + target[:j]]), use_cache=False).logits[0, -1]
                expected = diagnostic.token_observation(logits, target[j], trie.allowed_next(tuple(target[:j])))
                self.assertEqual(batch[row][stage]['rank'], expected['rank'])
                self.assertAlmostEqual(batch[row][stage]['nll'], expected['nll'], places=5)

    def test_recorder_preserves_return_and_frozen_sort(self):
        import torch
        sequences = [[9, 9, 4], [9, 9, 3]] + [[9, 9, i] for i in range(5, 13)]
        response = SimpleNamespace(sequences=torch.tensor(sequences), sequences_scores=torch.ones(10))
        model = SimpleNamespace(device='cpu', generate=lambda **kwargs: response)
        recorder = diagnostic.RecordingModel(model)
        actual = recorder.generate(input_ids=torch.ones((1, 2), dtype=torch.long))
        self.assertIs(actual, response)
        self.assertEqual(recorder.beams[0]['tokens'][0], [3])


if __name__ == '__main__':
    unittest.main()
