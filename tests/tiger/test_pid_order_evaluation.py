"""Synthetic checks for paired variable-length PID evaluation semantics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.tiger.evaluate_pid_order_retrieval import (  # noqa: E402
    PidOrderLegalPathConstraint,
    PidOrderTokens,
    canonicalize_pid,
    ordered_pid_is_valid,
    parse_candidate,
)
from poi_gr.pid.trie import (  # noqa: E402
    CompactPidTrie,
    PidTokenIds,
    build_compact_trie_arrays,
)
from poi_gr.methods.tiger.eval import atomic_json  # noqa: E402


class PidOrderEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pid_tokens = PidTokenIds(
            gid=np.arange(100, 132, dtype=np.int32),
            sid=(
                np.arange(1000, 2024, dtype=np.int32),
                np.arange(3000, 4024, dtype=np.int32),
                np.arange(5000, 6024, dtype=np.int32),
            ),
            dedup=np.arange(7000, 7512, dtype=np.int32),
            eos=99,
        )
        self.tokens = PidOrderTokens(
            target_open=80,
            target_close=81,
            pid=self.pid_tokens,
        )
        self.codes = np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 10, 20, 30, -1],
                [1, 2, 3, 4, 5, 7, 11, 21, 31, 0],
            ],
            dtype=np.int32,
        )

    def ordered_path(self, row: int, order: str) -> list[int]:
        codes = self.codes[row]
        gid = [int(self.pid_tokens.gid[codes[index]]) for index in range(6)]
        sid = [
            int(self.pid_tokens.sid[index][codes[index + 6]]) for index in range(3)
        ]
        result = gid + sid if order == "gid_sid" else sid + gid
        if codes[9] >= 0:
            result.append(int(self.pid_tokens.dedup[codes[9]]))
        return result

    def test_both_orders_are_structurally_valid_and_canonicalize_equal(self) -> None:
        gid_sid = self.ordered_path(1, "gid_sid")
        sid_gid = self.ordered_path(1, "sid_gid")
        self.assertTrue(
            ordered_pid_is_valid(gid_sid, pid_order="gid_sid", tokens=self.pid_tokens)
        )
        self.assertTrue(
            ordered_pid_is_valid(sid_gid, pid_order="sid_gid", tokens=self.pid_tokens)
        )
        self.assertEqual(
            canonicalize_pid(gid_sid, pid_order="gid_sid"),
            canonicalize_pid(sid_gid, pid_order="sid_gid"),
        )

    def test_wrong_order_is_rejected(self) -> None:
        self.assertFalse(
            ordered_pid_is_valid(
                self.ordered_path(0, "sid_gid"),
                pid_order="gid_sid",
                tokens=self.pid_tokens,
            )
        )

    def test_constrained_wrapper_path_and_candidate_lookup(self) -> None:
        trie = CompactPidTrie(
            **build_compact_trie_arrays(
                self.codes,
                self.pid_tokens,
                pid_order="sid_gid",
            )
        )
        prompt = [1, 2, 3]
        internal = self.ordered_path(0, "sid_gid")
        constraint = PidOrderLegalPathConstraint(
            trie=trie,
            tokens=self.tokens,
            prompt_width=len(prompt),
        )
        self.assertEqual(constraint(0, np.asarray(prompt)), [80])
        prefix = prompt + [80]
        self.assertEqual(
            constraint(0, np.asarray(prefix)),
            sorted(
                {
                    self.ordered_path(index, "sid_gid")[0]
                    for index in range(len(self.codes))
                }
            ),
        )
        self.assertEqual(
            constraint(0, np.asarray(prefix + internal)),
            [81],
        )
        sequence = [80, *internal, 81, 99]
        candidate = parse_candidate(
            sequence,
            -1.0,
            pid_order="sid_gid",
            tokens=self.tokens,
            trie=trie,
        )
        self.assertEqual(candidate.poi_row, 0)
        self.assertIsNone(candidate.error)

    def test_invalid_wrapper_keeps_invalid_beam_slot(self) -> None:
        trie = CompactPidTrie(
            **build_compact_trie_arrays(self.codes, self.pid_tokens)
        )
        candidate = parse_candidate(
            [80, *self.ordered_path(0, "gid_sid"), 99],
            -2.0,
            pid_order="gid_sid",
            tokens=self.tokens,
            trie=trie,
        )
        self.assertIsNone(candidate.poi_row)
        self.assertEqual(candidate.error, "invalid_wrapper_or_length")

    def test_four_6000d_launcher_assigns_one_mode_per_gpu(self) -> None:
        launcher = (
            PROJECT_ROOT
            / "launchers"
            / "run_evaluate_tiger_pid_order_epoch3_fixed10k_4x6000d.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", launcher)
        self.assertIn('run_eval "${labels[0]}" 0 gid_sid unconstrained', launcher)
        self.assertIn('run_eval "${labels[1]}" 1 gid_sid constrained', launcher)
        self.assertIn('run_eval "${labels[2]}" 2 sid_gid unconstrained', launcher)
        self.assertIn('run_eval "${labels[3]}" 3 sid_gid constrained', launcher)
        self.assertIn("--expected-step 8967", launcher)
        self.assertIn("--per-device-eval-batch-size 32", launcher)
        self.assertIn("--chunk-size 500", launcher)
        self.assertIn("--cutoff-len 1024", launcher)
        self.assertIn("metrics.get(\"sample_count\") == 10000", launcher)
        self.assertNotIn("torchrun", launcher)

    def test_atomic_json_allows_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"

            def write(index: int) -> None:
                atomic_json(path, {"status": "passed", "writer": index})

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(write, range(64)))

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "passed")
            self.assertIn(payload["writer"], range(64))
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
