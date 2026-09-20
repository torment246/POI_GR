from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qg_prqk.sft.vocabulary import SftVocabError, load_shared_tokens


class SharedSftVocabularyTest(unittest.TestCase):
    def test_requires_identical_ordered_token_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "gid.json"
            second = root / "nogid.json"
            payload = {
                "additional_special_tokens": ["<S1_0>", "<D_0>"],
                "token_count": 2,
            }
            first.write_text(json.dumps(payload), encoding="utf-8")
            second.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_shared_tokens(first, second), ["<S1_0>", "<D_0>"]
            )
            payload["additional_special_tokens"].reverse()
            second.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SftVocabError):
                load_shared_tokens(first, second)


if __name__ == "__main__":
    unittest.main()
