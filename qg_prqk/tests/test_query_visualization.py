"""Synthetic checks for query-balanced prefix statistics."""
import unittest
import numpy as np
from qg_prqk.sid.query_visualization import (
    distribution, prefix_codes, prefix_features, select_case, select_projection_cases,
)


class QueryVisualizationTest(unittest.TestCase):
    def test_projection_selection_is_positive_disjoint_and_repeatable(self):
        candidates = [dict(query=str(i), poi_rows=list(range(i * 5, i * 5 + 5)),
                           query_count=10, before={'top_prefix_share': .4},
                           after={'top_prefix_share': .8 if i < 7 else .3}) for i in range(10)]
        first = select_projection_cases(candidates)
        self.assertEqual(first, select_projection_cases(candidates))
        self.assertEqual(len(first), 5)
        self.assertEqual(len({r for q in first for r in q['poi_rows']}), 25)
        self.assertTrue(all(q['after']['top_prefix_share'] > .4 for q in first))
        with self.assertRaises(ValueError):
            select_projection_cases(candidates[:4])

    def test_prefix_vectors_use_unit_codewords_not_numeric_ids(self):
        sid = np.array([[0, 1], [0, 1], [1, 0]])
        books = [np.array([[2., 0.], [0., 3.]])] * 2
        vectors = prefix_features(sid, books)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.)
        np.testing.assert_array_equal(vectors[0], vectors[1])
        self.assertFalse(np.array_equal(vectors[0], vectors[2]))
        with self.assertRaises(ValueError):
            prefix_features(sid, [np.zeros((2, 2))] * 2)

    def test_prefix_keeps_parent(self):
        sid = np.array([[1, 2, 3], [2, 2, 3]])
        self.assertNotEqual(*prefix_codes(sid, 2))
        np.testing.assert_array_equal(prefix_codes(sid, 1), [1, 2])

    def test_order_mass_not_target_count(self):
        d = distribution(np.array([0, 0, 1]), np.array([3, 1, 4]), np.array([10, 30]))
        self.assertEqual(d['top_prefix_share'], .5)
        self.assertEqual(d['entropy_bits'], 1.)
        self.assertEqual(d['prefix_count'], 2)
        self.assertEqual(d['expected_bucket_size'], 20.)

    def test_merging_is_not_free(self):
        d = distribution(np.array([0, 0]), np.array([1, 1]), np.array([100]))
        self.assertEqual(d['top_prefix_share'], 1.)
        self.assertEqual(d['entropy_bits'], 0.)
        self.assertEqual(d['expected_bucket_size'], 100.)

    def test_single_target_trivial(self):
        d = distribution(np.array([0]), np.array([9]), np.array([8]))
        self.assertEqual(d['top_prefix_share'], 1.)
        self.assertEqual(d['prefix_count'], 1)

    def test_case_selection_independent_of_final(self):
        qs = [dict(query=str(i), poi_rows=[1, 2, 3], query_count=5,
                   before={'prefix_count': 2}, after={'top_prefix_share': i}) for i in range(10)]
        first = select_case(qs, np.random.default_rng(42))['query']
        for q in qs:
            q['after']['top_prefix_share'] = -99
        self.assertEqual(first, select_case(qs, np.random.default_rng(42))['query'])

    def test_invalid_counts_rejected(self):
        with self.assertRaises(ValueError):
            distribution(np.array([0]), np.array([0]), np.array([10]))

    def test_contrast_selection_explicitly_stratifies_outcome(self):
        qs = [dict(query=str(i), poi_rows=[1, 2, 3], query_count=5,
                   before={'prefix_count': 2, 'top_prefix_share': .5},
                   after={'top_prefix_share': share}) for i, share in enumerate((.4, .5, .8))]
        self.assertEqual(select_case(qs, np.random.default_rng(42), 'improved')['query'], '2')
        self.assertEqual(select_case(qs, np.random.default_rng(42), 'worsened')['query'], '0')
        with self.assertRaises(ValueError):
            select_case(qs[1:], np.random.default_rng(42), 'worsened')


if __name__ == '__main__':
    unittest.main()
