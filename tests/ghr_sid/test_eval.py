"""Tests for variable-length GHR identifier retrieval lookup."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from poi_gr.methods.ghr_sid.eval import (
    GhrEvalError,
    GhrIdIndex,
    GhrLegalPathConstraint,
    GhrPrefixIndex,
    GhrTokenIds,
    _aligned_codes_to_logical,
    _aligned_identifier_tokens,
    build_ghr_compact_trie,
    load_ghr_id_index,
    parse_generated_base_bucket,
    parse_generated_candidate,
)
from poi_gr.pid.trie import sha256_file


class GhrEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.codes = np.asarray(
            [
                [0, 1, 2, -1],
                [0, 1, 3, 4],
                [5, 2, 1, -1],
            ],
            dtype=np.int32,
        )
        self.lengths = np.asarray([3, 4, 3], dtype=np.uint8)
        self.poi_ids = np.asarray([101, 102, 103], dtype=np.int64)
        self.index = GhrIdIndex(
            codes=self.codes,
            lengths=self.lengths,
            poi_ids=self.poi_ids,
        )
        logical_to_token = tuple(range(200, 206))
        self.tokens = GhrTokenIds(
            target_open=100,
            target_close=101,
            logical_to_token=logical_to_token,
            token_to_logical={value: code for code, value in enumerate(logical_to_token)},
            eos=2,
            minimum_identifier_length=3,
            maximum_identifier_length=4,
        )

    def test_variable_length_lookup_is_exact(self) -> None:
        self.assertEqual(self.index.lookup((0, 1, 2)), 0)
        self.assertEqual(self.index.lookup((0, 1, 3, 4)), 1)
        self.assertEqual(self.index.lookup((5, 2, 1)), 2)
        self.assertEqual(self.index.lookup((0, 1, 2, 4)), -1)
        self.assertEqual(self.index.poi_id(1), "102")

    def test_valid_generated_candidate_maps_to_poi(self) -> None:
        candidate = parse_generated_candidate(
            [100, 200, 201, 202, 101, 2],
            0.5,
            tokens=self.tokens,
            index=self.index,
        )
        self.assertEqual(candidate.codes, (0, 1, 2))
        self.assertEqual(candidate.poi_row, 0)
        self.assertIsNone(candidate.error)

    def test_invalid_beams_keep_explicit_error_types(self) -> None:
        missing_eos = parse_generated_candidate(
            [100, 200, 201, 202, 101],
            0.5,
            tokens=self.tokens,
            index=self.index,
        )
        invalid_token = parse_generated_candidate(
            [100, 200, 201, 999, 101, 2],
            0.4,
            tokens=self.tokens,
            index=self.index,
        )
        not_in_corpus = parse_generated_candidate(
            [100, 200, 203, 202, 101, 2],
            0.3,
            tokens=self.tokens,
            index=self.index,
        )
        self.assertEqual(missing_eos.error, "missing_eos")
        self.assertEqual(invalid_token.error, "invalid_identifier_token")
        self.assertEqual(not_in_corpus.error, "identifier_not_in_corpus")

    def test_base_bucket_is_independent_of_suffix_and_full_id_validity(self) -> None:
        prefix_index = GhrPrefixIndex(self.codes)
        self.assertEqual(prefix_index.bucket_size((0, 1, 2)), 1)
        self.assertEqual(prefix_index.bucket_size((0, 1, 3)), 1)
        self.assertEqual(prefix_index.bucket_size((0, 1, 4)), 0)
        self.assertEqual(
            parse_generated_base_bucket(
                [100, 200, 201, 202, 999],
                tokens=self.tokens,
            ),
            (0, 1, 2),
        )
        self.assertIsNone(
            parse_generated_base_bucket(
                [100, 200, 999, 202],
                tokens=self.tokens,
            )
        )

    def test_padding_must_be_minus_one(self) -> None:
        invalid = self.codes.copy()
        invalid[0, 3] = 0
        with self.assertRaises(GhrEvalError):
            GhrIdIndex(codes=invalid, lengths=self.lengths, poi_ids=self.poi_ids)

    def test_legal_path_constraint_supports_variable_length_paths(self) -> None:
        trie = build_ghr_compact_trie(self.index)
        constraint = GhrLegalPathConstraint(
            trie=trie,
            tokens=self.tokens,
            prompt_width=3,
        )
        self.assertEqual(constraint.allowed_next(()), [100])
        self.assertEqual(constraint.allowed_next((100,)), [200, 205])
        self.assertEqual(constraint.allowed_next((100, 200)), [201])
        self.assertEqual(constraint.allowed_next((100, 200, 201)), [202, 203])
        self.assertEqual(constraint.allowed_next((100, 200, 201, 202)), [101])
        self.assertEqual(
            constraint.allowed_next((100, 200, 201, 203)),
            [204],
        )
        self.assertEqual(
            constraint.allowed_next((100, 200, 201, 202, 101)),
            [2],
        )
        self.assertEqual(
            constraint.allowed_next((100, 200, 201, 202, 101, 2, 2)),
            [2],
        )
        with self.assertRaisesRegex(GhrEvalError, "不在冻结语料库"):
            constraint.allowed_next((100, 200, 202))

    def test_fixed_aligned_collision_artifacts_load_as_logical_codes(self) -> None:
        raw_codes = np.asarray(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], dtype=np.int32
        )
        expected = np.asarray(
            [[1, 1026, 2051, 3076, 3109], [1, 1026, 2051, 3076, 3110]],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(_aligned_codes_to_logical(raw_codes), expected)
        tokens = _aligned_identifier_tokens()
        self.assertEqual(len(tokens), 3136)
        self.assertEqual(tokens[0], "<S1_0>")
        self.assertEqual(tokens[-1], "<R2_31>")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            codes_path = root / "identifier_codes.npy"
            mapping_path = root / "poi_identifier_mapping.parquet"
            metrics_path = root / "metrics.json"
            np.save(codes_path, raw_codes, allow_pickle=False)
            pq.write_table(
                pa.table(
                    {
                        "poi_id": ["101", "102"],
                        "s1": raw_codes[:, 0],
                        "s2": raw_codes[:, 1],
                        "s3": raw_codes[:, 2],
                        "r1": raw_codes[:, 3],
                        "r2": raw_codes[:, 4],
                    }
                ),
                mapping_path,
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "base_tiger": {"poi_count": 2},
                        "identifier": {
                            "layers": 5,
                            "distinct_count": 2,
                            "within_bucket_duplicate_pair_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            artifacts = {}
            for path in (codes_path, mapping_path, metrics_path):
                artifacts[path.name] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ghr-aligned-collision-quantizer-v1",
                        "status": "completed",
                        "protocol": {"base_layers": 3, "suffix_layers": 2},
                        "artifacts": artifacts,
                    }
                ),
                encoding="utf-8",
            )

            index, metadata = load_ghr_id_index(root)
            self.assertEqual(index.lookup(expected[0]), 0)
            self.assertEqual(index.poi_id(1), "102")
            self.assertEqual(metadata["format"], "fixed_five_layer_aligned_collision")


if __name__ == "__main__":
    unittest.main()
