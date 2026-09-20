"""Compatibility checks for active catalog evaluation and the shared evaluator CLI."""
import contextlib
import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.genpoi.evaluate_active_retrieval import main
from scripts.sft.evaluate_retrieval import _trie_inputs, parse_args
from poi_gr.sft.evaluation import GenerativeEvalError


class ActiveEvaluationTest(unittest.TestCase):
    def test_dry_run_commands_match_existing_evaluator_and_fixed_protocol(self):
        args = ["evaluate_active_retrieval.py", "--valid-file", "valid.jsonl",
                "--reference-validation-subset", "reference.jsonl", "--checkpoint", "checkpoint-3",
                "--tokenizer", "tokenizer", "--trie-dir", "trie", "--head-dir", "head",
                "--output-dir", "output", "--expected-step", "3", "--dry-run"]
        output = io.StringIO()
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(output):
            self.assertEqual(main(), 0)
        command = shlex.split(output.getvalue().splitlines()[-1])
        with patch.object(sys, "argv", command[1:]):
            parsed = parse_args()
        self.assertEqual(parsed.mode, "valid-checkpoints")
        self.assertEqual(parsed.cutoff_len, 1024)
        self.assertEqual(parsed.expected_checkpoint_epochs, [3.0])
        self.assertEqual(parsed.num_beams, 10)
        self.assertIsNotNone(parsed.ssp_predictions_dir)

    def test_leaf_count_must_match_declared_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root / "mapping.parquet"
            mapping.touch()
            manifest = {"status": "completed", "leaf_count": 716245,
                        "input": {"poi_count": 716245, "pid_mapping": str(mapping)}}
            path = root / "trie_manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(_trie_inputs(root)[0], mapping)
            manifest["leaf_count"] = 716244
            path.write_text(json.dumps(manifest))
            with self.assertRaises(GenerativeEvalError):
                _trie_inputs(root)
