"""Lightweight tests for the POI embedding pipeline."""

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

from poi_gr.embedding import (
    ConfigError,
    DataConfig,
    DataValidationError,
    EmbeddingJobConfig,
    ModelConfig,
    OutputConfig,
    apply_overrides,
    discover_input_files,
    encode_texts_to_npy,
    iter_texts,
    scan_and_prepare_input,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        self.call_sizes.append(len(sentences))
        return np.asarray(
            [[len(text), index + 1, 1.0] for index, text in enumerate(sentences)],
            dtype=np.float32,
        )


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class EmbeddingPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        (self.input_dir / "_SUCCESS").touch()
        write_jsonl(
            self.input_dir / "part-00001.json",
            [{"poi_id": "p3", "text": "ccc"}],
        )
        write_jsonl(
            self.input_dir / "part-00000.json",
            [
                {"poi_id": "p1", "text": "a"},
                {"poi_id": "p2", "text": "bb"},
            ],
        )
        self.data_config = DataConfig(
            input_dir=self.input_dir,
            file_pattern="part-*.json",
            success_marker="_SUCCESS",
            id_field="poi_id",
            text_field="text",
            expected_rows=3,
            check_unique_ids=True,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_scan_preserves_sorted_file_and_row_order(self) -> None:
        files = discover_input_files(self.data_config)
        ids_path = self.output_dir / "poi_ids.jsonl"
        prepared = scan_and_prepare_input(files, self.data_config, ids_path)

        self.assertEqual([path.name for path in files], ["part-00000.json", "part-00001.json"])
        self.assertEqual(prepared.total_rows, 3)
        self.assertEqual(list(iter_texts(prepared, self.data_config)), ["a", "bb", "ccc"])
        ids = [json.loads(line) for line in ids_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(ids, ["p1", "p2", "p3"])

    def test_duplicate_poi_id_is_rejected(self) -> None:
        write_jsonl(
            self.input_dir / "part-00001.json",
            [{"poi_id": "p2", "text": "duplicate"}],
        )
        files = discover_input_files(self.data_config)
        with self.assertRaisesRegex(DataValidationError, "重复"):
            scan_and_prepare_input(
                files,
                self.data_config,
                self.output_dir / "poi_ids.jsonl",
            )

    def test_encode_writes_expected_memmap_shape_and_dtype(self) -> None:
        self.output_dir.mkdir()
        model_config = ModelConfig(
            path=self.root / "model",
            device="cpu",
            batch_size=2,
            encode_buffer_size=4,
            max_seq_length=32,
            torch_dtype="float32",
            attention=None,
            padding_side="left",
            normalize_embeddings=False,
            truncate_dim=None,
            prompt_name=None,
        )
        output_config = OutputConfig(
            dir=self.output_dir,
            embedding_dtype="float16",
            checkpoint_interval_batches=1,
            resume=True,
        )
        encoder = FakeEncoder()
        result = encode_texts_to_npy(
            encoder,
            ["a", "bb", "ccc"],
            embeddings_path=self.output_dir / "embeddings.npy",
            progress_path=self.output_dir / "progress.json",
            total_rows=3,
            embedding_dim=3,
            start_row=0,
            model_config=model_config,
            output_config=output_config,
            signature="test-signature",
            show_progress=False,
        )

        embeddings = np.load(self.output_dir / "embeddings.npy", mmap_mode="r")
        self.assertEqual(embeddings.shape, (3, 3))
        self.assertEqual(embeddings.dtype, np.dtype("float16"))
        np.testing.assert_array_equal(embeddings[:, 0], np.asarray([1, 2, 3]))
        progress = json.loads((self.output_dir / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(result["rows_encoded_this_run"], 3)
        self.assertEqual(encoder.call_sizes, [3])

    def test_encode_can_resume_without_overwriting_completed_rows(self) -> None:
        self.output_dir.mkdir()
        embeddings_path = self.output_dir / "embeddings.npy"
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float16,
            shape=(3, 3),
        )
        embeddings[:2] = np.asarray([[10, 10, 10], [20, 20, 20]], dtype=np.float16)
        embeddings.flush()
        del embeddings

        model_config = ModelConfig(
            path=self.root / "model",
            device="cpu",
            batch_size=2,
            encode_buffer_size=4,
            max_seq_length=32,
            torch_dtype="float32",
            attention=None,
            padding_side="left",
            normalize_embeddings=False,
            truncate_dim=None,
            prompt_name=None,
        )
        output_config = OutputConfig(
            dir=self.output_dir,
            embedding_dtype="float16",
            checkpoint_interval_batches=1,
            resume=True,
        )
        encode_texts_to_npy(
            FakeEncoder(),
            ["ccc"],
            embeddings_path=embeddings_path,
            progress_path=self.output_dir / "progress.json",
            total_rows=3,
            embedding_dim=3,
            start_row=2,
            model_config=model_config,
            output_config=output_config,
            signature="resume-signature",
            show_progress=False,
        )

        resumed = np.load(embeddings_path, mmap_mode="r")
        np.testing.assert_array_equal(resumed[0], np.asarray([10, 10, 10]))
        np.testing.assert_array_equal(resumed[1], np.asarray([20, 20, 20]))
        np.testing.assert_array_equal(resumed[2], np.asarray([3, 1, 1]))

    def test_batch_size_cannot_exceed_encode_buffer(self) -> None:
        model_config = ModelConfig(
            path=self.root / "model",
            device="cpu",
            batch_size=2,
            encode_buffer_size=4,
            max_seq_length=32,
            torch_dtype="float32",
            attention=None,
            padding_side="left",
            normalize_embeddings=False,
            truncate_dim=None,
            prompt_name=None,
        )
        output_config = OutputConfig(
            dir=self.output_dir,
            embedding_dtype="float16",
            checkpoint_interval_batches=1,
            resume=True,
        )
        job_config = EmbeddingJobConfig(
            job_name="test",
            data=self.data_config,
            model=model_config,
            output=output_config,
        )

        with self.assertRaisesRegex(ConfigError, "不能小于"):
            apply_overrides(job_config, batch_size=5)


if __name__ == "__main__":
    unittest.main()
