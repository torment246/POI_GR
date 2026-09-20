"""Tests for exact ordered embedding subsetting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.subset import EmbeddingSubsetError, build_embedding_subset


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


class EmbeddingSubsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        values = np.asarray(
            [[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]], dtype=np.float16
        )
        np.save(self.source / "embeddings.npy", values)
        write_ids(self.source / "poi_ids.jsonl", ["a", "b", "c", "d", "e"])
        (self.source / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "input": {"total_rows": 5, "text_field": "text"},
                    "model": {"backend": "fake", "embedding_dim": 2},
                    "output": {"shape": [5, 2], "dtype": "float16"},
                }
            ),
            encoding="utf-8",
        )
        self.target_manifest = self.root / "catalog_manifest.json"
        self.target_manifest.write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, target_ids: list[str], expected_rows: int) -> Path:
        target = self.root / "target_ids.jsonl"
        write_ids(target, target_ids)
        output = self.root / "output"
        build_embedding_subset(
            source_dir=self.source,
            target_poi_ids=target,
            target_manifest=self.target_manifest,
            output_dir=output,
            expected_rows=expected_rows,
            project_root=self.root,
            copy_chunk_rows=2,
        )
        return output

    def test_exact_ordered_subset(self) -> None:
        output = self.build(["b", "d", "e"], 3)
        expected = np.load(self.source / "embeddings.npy")[[1, 3, 4]]
        actual = np.load(output / "embeddings.npy")
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            (output / "poi_ids.jsonl").read_bytes(),
            (self.root / "target_ids.jsonl").read_bytes(),
        )
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["output"]["shape"], [3, 2])
        self.assertTrue(manifest["derivation"]["exact_vector_copy"])
        self.assertEqual(
            manifest["derivation"]["copy_strategy"],
            "contiguous_source_range_then_memory_gather",
        )
        self.assertTrue(manifest["metrics"]["vectors_all_finite"])

    def test_rejects_target_that_is_not_an_ordered_subsequence(self) -> None:
        target = self.root / "target_ids.jsonl"
        write_ids(target, ["d", "b"])
        output = self.root / "output"
        with self.assertRaisesRegex(EmbeddingSubsetError, "有序子序列"):
            build_embedding_subset(
                source_dir=self.source,
                target_poi_ids=target,
                target_manifest=self.target_manifest,
                output_dir=output,
                expected_rows=2,
                project_root=self.root,
                copy_chunk_rows=2,
            )
        self.assertFalse(output.exists())

    def test_rejects_existing_output(self) -> None:
        target = self.root / "target_ids.jsonl"
        write_ids(target, ["a"])
        output = self.root / "output"
        output.mkdir()
        with self.assertRaisesRegex(EmbeddingSubsetError, "拒绝覆盖"):
            build_embedding_subset(
                source_dir=self.source,
                target_poi_ids=target,
                target_manifest=self.target_manifest,
                output_dir=output,
                expected_rows=1,
                project_root=self.root,
            )
