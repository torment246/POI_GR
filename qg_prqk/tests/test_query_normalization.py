from __future__ import annotations

import unittest

from qg_prqk.data.query_normalization import QueryNormalizationError, normalize_query


class QueryNormalizationTest(unittest.TestCase):
    def test_nfkc_lowercase_whitespace_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_query("  ＡＢＣ　北京路（东门），  3Ｆ！ "),
            "abc 北京路(东门), 3f!",
        )

    def test_location_words_are_preserved(self) -> None:
        query = "朝阳区 建国路 万达广场 3楼 东入口 T3航站楼"
        normalized = normalize_query(query)
        for location_word in ("朝阳区", "建国路", "万达广场", "3楼", "东入口", "航站楼"):
            self.assertIn(location_word, normalized)

    def test_normalization_is_idempotent(self) -> None:
        once = normalize_query("北京——南站……  A口")
        self.assertEqual(normalize_query(once), once)

    def test_rejects_empty_normalized_query(self) -> None:
        with self.assertRaises(QueryNormalizationError):
            normalize_query(" \t\n ")


if __name__ == "__main__":
    unittest.main()
