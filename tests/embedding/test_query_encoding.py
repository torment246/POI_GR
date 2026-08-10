"""Tests for aligned and resumable Train Query encoding."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poi_gr.embedding.core import ModelConfig, OutputConfig
from poi_gr.embedding.query_encoding import (
    TrainQueryEncodingConfig,
    iter_query_catalog,
    run_train_query_encoding,
    validate_train_query_embeddings,
)
from poi_gr.embedding.query_shards import build_train_query_shards
from poi_gr.embedding.query_stats import build_train_query_stats


def _sample(query: str, poi_id: str) -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"<QUERY>{query}</QUERY>\n"
                    "<USER_GID><G_1><G_2><G_3><G_4><G_5><G_6></USER_GID>"
                ),
            },
            {"role": "assistant", "content": "<S1_1><S2_2><S3_3>"},
        ],
        "target_poi_id": poi_id,
        "split": "train",
    }


def _write_source(sft_dir: Path) -> None:
    records = [
        _sample("北京南站", "p1"),
        _sample("停车场", "p1"),
        _sample("北京南站", "p2"),
        _sample("合生汇B1", "p3"),
    ]
    train_path = sft_dir / "train.jsonl"
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    train_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(train_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "sft-main-data-v1",
        "status": "completed",
        "outputs": {"train.jsonl": {"rows": len(records), "sha256": digest}},
    }
    (sft_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


class FakeEncoder:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        values = np.asarray(
            [[len(text), index + 1, 1.0] for index, text in enumerate(sentences)],
            dtype=np.float32,
        )
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values


class QueryEncodingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        sft_dir = self.root / "sft"
        sft_dir.mkdir()
        _write_source(sft_dir)
        query_shards_dir = self.root / "query_shards"
        build_train_query_shards(
            sft_dir,
            query_shards_dir,
            num_shards=4,
            buffer_rows_per_shard=2,
            show_progress=False,
        )
        self.query_stats_dir = self.root / "query_stats"
        stats = build_train_query_stats(
            query_shards_dir,
            self.query_stats_dir,
            poi_partitions=4,
            poi_buffer_rows=2,
            read_batch_rows=2,
            show_progress=False,
        )
        stats_sha256 = hashlib.sha256(stats.manifest_path.read_bytes()).hexdigest()
        self.output_dir = self.root / "query_embeddings"
        self.config = TrainQueryEncodingConfig(
            job_name="query-encoding-test",
            query_stats_dir=self.query_stats_dir,
            expected_stats_manifest_sha256=stats_sha256,
            expected_queries=stats.unique_queries,
            source_model_config=self.root / "embedding.yaml",
            model=ModelConfig(
                path=self.root / "model",
                device="cpu",
                batch_size=2,
                encode_buffer_size=2,
                max_seq_length=32,
                torch_dtype="float32",
                attention=None,
                padding_side="right",
                normalize_embeddings=True,
                truncate_dim=None,
                prompt_name=None,
            ),
            output=OutputConfig(
                dir=self.output_dir,
                embedding_dtype="float16",
                checkpoint_interval_batches=1,
                resume=True,
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_catalog_iterator_preserves_query_id_and_resume_order(self) -> None:
        queries = list(iter_query_catalog(self.query_stats_dir, read_batch_rows=2))
        self.assertEqual(len(queries), 3)
        self.assertEqual(
            list(
                iter_query_catalog(
                    self.query_stats_dir,
                    start_row=1,
                    stop_row=3,
                    read_batch_rows=1,
                )
            ),
            queries[1:],
        )

    def test_run_writes_aligned_artifact_and_reuses_completed_output(self) -> None:
        encoder = FakeEncoder()
        result = run_train_query_encoding(
            self.config,
            project_root=self.root,
            show_progress=False,
            encoder_loader=lambda _: (encoder, "cpu", 0.01),
        )
        self.assertFalse(result.reused)
        self.assertTrue(encoder.eval_called)
        embeddings = np.load(result.embeddings_path, mmap_mode="r")
        self.assertEqual(embeddings.shape, (3, 3))
        self.assertEqual(embeddings.dtype, np.dtype("float16"))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["output"]["row_mapping"], "row_index equals query_id")
        self.assertTrue(manifest["validation"]["all_finite"])

        validated = validate_train_query_embeddings(
            self.output_dir,
            expected_stats_manifest_sha256=(self.config.expected_stats_manifest_sha256),
        )
        self.assertTrue(validated.reused)

        def fail_loader(_: ModelConfig) -> tuple[FakeEncoder, str, float]:
            raise AssertionError("completed output should be reused before model load")

        reused = run_train_query_encoding(
            self.config,
            project_root=self.root,
            show_progress=False,
            encoder_loader=fail_loader,
        )
        self.assertTrue(reused.reused)


if __name__ == "__main__":
    unittest.main()
