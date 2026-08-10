"""Synthetic tests for compact Final PID Trie semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.pid.trie import (  # noqa: E402
    CompactPidTrie,
    PidTokenIds,
    PidTrieError,
    TriePrefilledPrefixConstraint,
    TriePrefixConstraint,
    build_compact_trie_arrays,
)


class CompactPidTrieTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = PidTokenIds(
            gid=np.arange(100, 132, dtype=np.int32),
            sid=(
                np.arange(1000, 2024, dtype=np.int32),
                np.arange(3000, 4024, dtype=np.int32),
                np.arange(5000, 6024, dtype=np.int32),
            ),
            dedup=np.arange(7000, 7512, dtype=np.int32),
            eos=99,
        )
        self.codes = np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 10, 20, 30, -1],
                [1, 2, 3, 4, 5, 7, 11, 21, 31, 0],
                [1, 2, 3, 4, 5, 7, 11, 21, 31, 1],
                [2, 3, 4, 5, 6, 7, 12, 22, 32, -1],
            ],
            dtype=np.int32,
        )
        arrays = build_compact_trie_arrays(self.codes, self.tokens)
        self.trie = CompactPidTrie(**arrays)

    def path(self, row: int) -> list[int]:
        codes = self.codes[row]
        result = [int(self.tokens.gid[codes[index]]) for index in range(6)]
        result.extend(
            int(self.tokens.sid[level][codes[level + 6]]) for level in range(3)
        )
        if codes[9] >= 0:
            result.append(int(self.tokens.dedup[codes[9]]))
        result.append(self.tokens.eos)
        return result

    def test_singleton_allows_only_eos_after_s3(self) -> None:
        prefix = self.path(0)[:-1]
        self.assertEqual(self.trie.children(self.trie.traverse(prefix)).tolist(), [99])
        self.assertEqual(self.trie.lookup(self.path(0)), 0)

    def test_dedup_allows_only_codes_present_in_base_bucket(self) -> None:
        base_prefix = self.path(1)[:9]
        allowed = self.trie.children(self.trie.traverse(base_prefix)).tolist()
        self.assertEqual(allowed, [7000, 7001])
        self.assertEqual(self.trie.advance(self.trie.traverse(base_prefix), 7002), -1)

    def test_dedup_token_allows_only_eos(self) -> None:
        prefix = self.path(2)[:-1]
        self.assertEqual(self.trie.children(self.trie.traverse(prefix)).tolist(), [99])
        self.assertEqual(self.trie.lookup(self.path(2)), 2)

    def test_invalid_branch_is_blocked_and_minus_one_is_not_a_token(self) -> None:
        singleton_prefix = self.path(0)[:-1]
        self.assertEqual(
            self.trie.advance(self.trie.traverse(singleton_prefix), -1), -1
        )
        invalid = self.path(0)
        invalid[0] = int(self.tokens.gid[31])
        self.assertEqual(self.trie.traverse(invalid), -1)
        self.assertNotIn(-1, self.trie.child_token_ids.tolist())

    def test_each_leaf_maps_to_one_unique_poi_row(self) -> None:
        self.assertEqual(self.trie.leaf_count, len(self.codes))
        rows = [self.trie.lookup(self.path(index)) for index in range(len(self.codes))]
        self.assertEqual(rows, list(range(len(self.codes))))
        self.assertEqual(len(set(rows)), len(rows))

    def test_duplicate_final_pid_is_rejected(self) -> None:
        duplicated = np.concatenate((self.codes, self.codes[[0]]), axis=0)
        with self.assertRaisesRegex(PidTrieError, "叶子"):
            build_compact_trie_arrays(duplicated, self.tokens)

    def test_input_order_does_not_change_paths(self) -> None:
        reordered = self.codes[[3, 1, 0, 2]]
        other = CompactPidTrie(**build_compact_trie_arrays(reordered, self.tokens))
        self.assertEqual(
            self.trie.child_token_ids.tolist(),
            other.child_token_ids.tolist(),
        )
        self.assertEqual(
            self.trie.child_node_ids.tolist(),
            other.child_node_ids.tolist(),
        )

    def test_same_input_rebuild_is_byte_identical(self) -> None:
        first = build_compact_trie_arrays(self.codes, self.tokens)
        second = build_compact_trie_arrays(self.codes, self.tokens)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_transformers_terminal_callback_keeps_only_eos(self) -> None:
        prompt = [10, 20, 30]
        path = self.path(0)
        constraint = TriePrefixConstraint(
            self.trie,
            prompt_width=len(prompt),
            eos_token_id=self.tokens.eos,
        )
        self.assertEqual(constraint(0, np.asarray(prompt + path)), [99])

    def test_prefilled_constraint_continues_from_each_gid_prefix(self) -> None:
        prompt = [10, 20, 30]
        first_prefix = self.path(0)[:3]
        second_prefix = self.path(3)[:3]
        constraint = TriePrefilledPrefixConstraint(
            self.trie,
            prompt_width=len(prompt) + len(first_prefix),
            eos_token_id=self.tokens.eos,
            prefilled_prefixes=(first_prefix, second_prefix),
        )
        self.assertEqual(
            constraint(0, np.asarray(prompt + first_prefix)),
            [self.path(0)[3]],
        )
        self.assertEqual(
            constraint(1, np.asarray(prompt + second_prefix)),
            [self.path(3)[3]],
        )


if __name__ == "__main__":
    unittest.main()
