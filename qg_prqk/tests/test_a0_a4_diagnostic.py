"""Synthetic contracts for the paired-input and S1 rank audit."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("diagnostic", Path(__file__).parents[1] / "scripts/diagnose_a0_a4.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DiagnosticTest(unittest.TestCase):
    def test_tied_scores_keep_lowest_token_first(self):
        scores = np.array([[.5, .5, .1], [.1, .9, .5]], dtype=np.float32)
        np.testing.assert_array_equal(diagnostic.stable_ranks(scores, np.array([1, 2])), [2, 2])

    def test_pairing_ignores_only_history_identifier(self):
        def record(sid):
            content = '<POI_QGPRQK_ID>' + '<G_0>' * 6 + f'<S1_{sid}><S2_2><S3_3></POI_QGPRQK_ID>'
            return dict(split='valid', order_id='order', searchid='search', target_poi_id='poi', history_length=1,
                        messages=[dict(role='user', content=content+'<CURRENT>query</CURRENT>'),
                                  dict(role='assistant', content='target')])
        a, b = record(1), record(2)
        self.assertEqual(diagnostic.request_without_ids(a), diagnostic.request_without_ids(b))
        b['messages'][0]['content'] += 'changed request'
        self.assertNotEqual(diagnostic.request_without_ids(a), diagnostic.request_without_ids(b))
        b['history_length'] = 2
        with self.assertRaises(AssertionError):
            diagnostic.request_without_ids(b)


if __name__ == '__main__':
    unittest.main()
