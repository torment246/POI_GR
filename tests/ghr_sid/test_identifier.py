import unittest

import numpy as np

from poi_gr.methods.ghr_sid.identifier import (
    assign_leaf_dedup_codes,
    base1024_digits,
    build_identifier_vocabulary,
    serialize_identifier_codes,
)


class GhrIdentifierTest(unittest.TestCase):
    def test_base1024_digits_are_high_order_first(self) -> None:
        self.assertEqual(base1024_digits(0), (0,))
        self.assertEqual(base1024_digits(1024), (1, 0))
        self.assertEqual(base1024_digits(1024**2 + 2), (1, 0, 2))

    def test_identifier_serialization_uses_numeric_namespaces(self) -> None:
        vocabulary = build_identifier_vocabulary(21)
        codes = serialize_identifier_codes(
            base_sid=(1, 2, 3),
            geo_codes=(24, 17, 2),
            relation_types=(5, 6),
            relation_values=(4, 40),
            dedup_code=1,
            vocabulary=vocabulary,
        )
        tokens = [vocabulary.tokens[code] for code in codes]
        self.assertEqual(
            tokens,
            [
                "<S1_1>",
                "<S2_2>",
                "<S3_3>",
                "<G_s>",
                "<G_j>",
                "<G_2>",
                "<R_5>",
                "<V_4>",
                "<R_6>",
                "<V_40>",
                "<D>",
                "<V_1>",
            ],
        )

    def test_dedup_is_local_and_numeric_poi_ordered(self) -> None:
        group_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        types = np.asarray(
            [[5, -1], [5, -1], [6, -1], [-1, -1], [-1, -1]],
            dtype=np.int16,
        )
        values = np.asarray(
            [[4, -1], [4, -1], [8, -1], [-1, -1], [-1, -1]],
            dtype=np.int64,
        )
        lengths = np.asarray([1, 1, 1, 0, 0], dtype=np.uint8)
        poi_ids = np.asarray([20, 10, 30, 50, 40], dtype=np.int64)
        actual = assign_leaf_dedup_codes(
            group_ids=group_ids,
            path_types=types,
            path_values=values,
            path_lengths=lengths,
            poi_ids=poi_ids,
        )
        np.testing.assert_array_equal(actual, [1, 0, -1, 1, 0])


if __name__ == "__main__":
    unittest.main()
