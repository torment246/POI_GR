"""Synthetic tests for the isolated TIGER-Joint data path."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint import (  # noqa: E402
    DATA_SCHEMA_VERSION,
    DYNAMIC_HISTORY_IDENTIFIER,
    DYNAMIC_TARGET_IDENTIFIER,
    DynamicPreflightTemplate,
    HistoryWindow,
    PoiEmbeddingStore,
    TigerJointPreparationError,
    build_fresh_initialization_contract,
    build_joint_special_tokens,
    build_sid_free_training_data,
    collate_dynamic_examples,
    combine_dynamic_preflight_splits,
    load_dynamic_token_ids,
    parse_sid_free_record,
    raw_order_to_sid_free_sample,
    scan_dynamic_preflight_split,
    tokenize_dynamic_record,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402
from poi_gr.sft.data import TimeSplit  # noqa: E402


class _FakeTokenizer:
    def __init__(self) -> None:
        tokens = (
            "<POI_SID>",
            "</POI_SID>",
            "<CURRENT>",
            "</CURRENT>",
            "<TARGET_POI>",
            "</TARGET_POI>",
            "<S1_0>",
            "<S2_0>",
            "<S3_0>",
            "<|im_start|>",
            "<|im_end|>",
        )
        self.vocabulary = {
            token: token_id for token_id, token in enumerate(tokens, start=10)
        }
        self.ordered_tokens = sorted(tokens, key=len, reverse=True)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocabulary.get(token, -1)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("implicit tokens must be disabled")
        result: list[int] = []
        position = 0
        while position < len(text):
            matched = next(
                (
                    token
                    for token in self.ordered_tokens
                    if text.startswith(token, position)
                ),
                None,
            )
            if matched is None:
                result.append(1000 + ord(text[position]))
                position += 1
            else:
                result.append(self.vocabulary[matched])
                position += len(matched)
        return result

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        return_length: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, list[Any]]:
        if (
            add_special_tokens
            or not return_length
            or return_attention_mask
            or return_token_type_ids
            or padding
            or truncation
        ):
            raise AssertionError("unexpected tokenizer options")
        input_ids = [self.encode(text, add_special_tokens=False) for text in texts]
        return {
            "input_ids": input_ids,
            "length": [len(values) for values in input_ids],
        }


class _FakeFormatter:
    def __init__(self, prefix: str, suffix: str) -> None:
        self.prefix = prefix
        self.suffix = suffix

    def apply(self, *, content: str = "", idx: str = "") -> list[str]:
        del idx
        return [f"{self.prefix}{content}{self.suffix}"]


class _FakeEmptyFormatter:
    def apply(self) -> list[str]:
        return []


class _FakeTemplate:
    def __init__(self) -> None:
        self.format_prefix = _FakeEmptyFormatter()
        self.format_user = _FakeFormatter(
            "<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n",
        )
        self.format_assistant = _FakeFormatter("", "<|im_end|>\n")

    def encode_oneturn(
        self,
        tokenizer: _FakeTokenizer,
        messages: list[dict[str, str]],
        *,
        system: None,
        tools: None,
    ) -> tuple[list[int], list[int]]:
        if system is not None or tools is not None:
            raise AssertionError("unexpected system/tools")
        source = (
            "<|im_start|>user\n"
            + messages[0]["content"]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        target = messages[1]["content"] + "<|im_end|>\n"
        return (
            tokenizer.encode(source, add_special_tokens=False),
            tokenizer.encode(target, add_special_tokens=False),
        )


def _write_embedding_artifacts(root: Path) -> tuple[Path, np.ndarray]:
    embedding_dir = root / "embeddings"
    embedding_dir.mkdir()
    poi_ids_path = embedding_dir / "poi_ids.jsonl"
    poi_ids_path.write_text(
        "".join(json.dumps(value) + "\n" for value in ("poi-a", "poi-b", "poi-c")),
        encoding="utf-8",
    )
    embeddings = np.arange(18, dtype=np.float16).reshape(3, 6)
    np.save(embedding_dir / "embeddings.npy", embeddings, allow_pickle=False)
    manifest = {
        "status": "completed",
        "signature": "a" * 64,
        "output": {
            "shape": [3, 6],
            "dtype": "float16",
            "embeddings": "embeddings.npy",
            "poi_ids": "poi_ids.jsonl",
        },
    }
    (embedding_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return embedding_dir, embeddings


def _sid_free_record(split: str = "train") -> dict[str, Any]:
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "sample_id": f"{split}-1",
        "split": split,
        "history_length": 2,
        "history_poi_rows": [0, 1],
        "target_poi_row": 2,
        "target_poi_id": "poi-c",
        "messages": [
            {
                "role": "user",
                "content": (
                    "<HISTORY>"
                    + DYNAMIC_HISTORY_IDENTIFIER
                    + DYNAMIC_HISTORY_IDENTIFIER
                    + "</HISTORY><CURRENT>coffee</CURRENT>"
                ),
            },
            {"role": "assistant", "content": DYNAMIC_TARGET_IDENTIFIER},
        ],
    }


def _raw_order(source_dt: str, poi_id: str = "poi-c") -> dict[str, Any]:
    day = f"{source_dt[:4]}-{source_dt[4:6]}-{source_dt[6:]}"
    return {
        "order_id": f"order-{source_dt}",
        "searchid": f"search-{source_dt}",
        "passenger_id": "user-1",
        "query": "coffee",
        "disp_lng": 116.4,
        "disp_lat": 39.9,
        "create_time": f"{day} 12:00:00",
        "source_dt": source_dt,
        "poi_id": poi_id,
        "history_length": 1,
        "history_sequence": [
            {
                "event_time": "2026-06-01T10:00:00+08:00",
                "order_id": "history-order",
                "searchid": "history-search",
                "query": "tea",
                "disp_lng": 116.3,
                "disp_lat": 39.8,
                "poi_id": "poi-a",
            }
        ],
    }


TIME_SPLIT = TimeSplit(
    train_start=date(2026, 7, 1),
    train_end=date(2026, 7, 12),
    valid_date=date(2026, 7, 13),
    test_date=date(2026, 7, 14),
)
HISTORY_WINDOW = HistoryWindow(date(2026, 4, 1), date(2026, 6, 30))


class TigerJointPreparationTest(unittest.TestCase):
    def test_bge_catalog_is_the_only_identity_to_row_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            embedding_dir, embeddings = _write_embedding_artifacts(
                Path(temporary_directory)
            )
            store = PoiEmbeddingStore.from_directory(embedding_dir)

        self.assertEqual(store.row_for_poi_id("poi-b"), 1)
        self.assertEqual(store.poi_count, 3)
        self.assertEqual(len(store.poi_ids_sha256), 64)
        np.testing.assert_array_equal(
            store.gather([2, 0]), embeddings[[2, 0]].astype(np.float32)
        )

    def test_raw_order_becomes_rows_and_unassigned_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            embedding_dir, _ = _write_embedding_artifacts(Path(temporary_directory))
            store = PoiEmbeddingStore.from_directory(embedding_dir)
            sample = raw_order_to_sid_free_sample(
                _raw_order("20260701"),
                relative_path="part-0",
                line_number=1,
                split=TIME_SPLIT,
                history_window=HISTORY_WINDOW,
                embedding_store=store,
            )

        assert sample is not None
        self.assertEqual(sample["history_poi_rows"], [0])
        self.assertEqual(sample["target_poi_row"], 2)
        serialized = json.dumps(sample)
        self.assertNotIn("POI_TIGER_ID", serialized)
        self.assertNotIn("<C_", serialized)
        self.assertEqual(
            sample["messages"][0]["content"].count(DYNAMIC_HISTORY_IDENTIFIER),
            1,
        )

    def test_old_tiger_record_is_rejected_instead_of_converted(self) -> None:
        old = _sid_free_record()
        old["schema_version"] = "tiger-map-search-sft-data-v1"
        old["target_tiger_id_key"] = "1-2-3|c0"
        old["messages"][0]["content"] = (
            "<CURRENT>x</CURRENT><POI_TIGER_ID><S1_1><S2_2><S3_3><C_0></POI_TIGER_ID>"
        )
        with self.assertRaisesRegex(TigerJointPreparationError, "旧 TIGER JSONL"):
            parse_sid_free_record(old, expected_split="train")

    def test_tokenize_and_collate_row_based_record(self) -> None:
        tokenizer = _FakeTokenizer()
        template = _FakeTemplate()
        token_ids = load_dynamic_token_ids(tokenizer)
        with tempfile.TemporaryDirectory() as temporary_directory:
            embedding_dir, embeddings = _write_embedding_artifacts(
                Path(temporary_directory)
            )
            store = PoiEmbeddingStore.from_directory(embedding_dir)
            text_record = parse_sid_free_record(
                _sid_free_record(), expected_split="train"
            )
            example = tokenize_dynamic_record(
                text_record,
                tokenizer=tokenizer,
                template=template,
                token_ids=token_ids,
            )
            batch = collate_dynamic_examples(
                [example],
                embedding_store=store,
                catalog_rows=[0, 2],
                pad_token_id=0,
                device="cpu",
            )

        self.assertEqual(len(example.history_sid_positions), 2)
        self.assertEqual(example.labels[example.query_state_position], -100)
        np.testing.assert_array_equal(
            batch.history_embeddings.numpy(), embeddings[[0, 1]].astype(np.float32)
        )
        np.testing.assert_array_equal(
            batch.target_embeddings.numpy(), embeddings[[2]].astype(np.float32)
        )

    def test_streaming_preflight_accepts_only_sid_free_schema(self) -> None:
        tokenizer = _FakeTokenizer()
        template = _FakeTemplate()
        token_ids = load_dynamic_token_ids(tokenizer)
        contract = DynamicPreflightTemplate.from_runtime(
            tokenizer=tokenizer,
            template=template,
            token_ids=token_ids,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_stats = []
            for split, rows in (("train", 2), ("valid", 1)):
                records = []
                for index in range(rows):
                    record = _sid_free_record(split)
                    record["sample_id"] = f"{split}-{index}"
                    records.append(record)
                path = root / f"{split}.jsonl"
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                split_stats.append(
                    scan_dynamic_preflight_split(
                        name=split,
                        path=path,
                        expected_rows=rows,
                        expected_sha256=sha256_file(path),
                        tokenizer=tokenizer,
                        template_contract=contract,
                        batch_size=2,
                        progress_every=1,
                    )
                )
            combined = combine_dynamic_preflight_splits(split_stats)

        self.assertEqual(combined["rows"], 3)
        self.assertEqual(combined["history_events"], 6)
        self.assertEqual(combined["poi_row_references"], 9)
        self.assertEqual(combined["dynamic_sid_slots"], 27)
        self.assertTrue(combined["formal_zero_gate_passed"])

    def test_fresh_initialization_contract_has_no_legacy_source(self) -> None:
        contract = build_fresh_initialization_contract(row_count=100)

        self.assertEqual(contract["legacy_artifacts_loaded"], [])
        self.assertFalse(contract["fresh_rqvae"]["checkpoint_loaded"])
        self.assertFalse(contract["fresh_qwen"]["old_tiger_vocab_loaded"])
        self.assertEqual(contract["construction_order"][1], "construct_fresh_rqvae")

    def test_fresh_vocabulary_has_three_levels_and_no_collision(self) -> None:
        inventory = build_joint_special_tokens(codebook_sizes=(4, 3, 2))

        self.assertEqual(inventory["sid_token_capacities"], [4, 3, 2])
        self.assertEqual(inventory["collision_tokens"], [])
        self.assertNotIn("<POI_TIGER_ID>", inventory["additional_tokens"])
        self.assertFalse(
            any(token.startswith("<C_") for token in inventory["additional_tokens"])
        )

    def test_builder_writes_train_valid_only_and_declares_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            embedding_dir, _ = _write_embedding_artifacts(root)
            orders_dir = root / "orders"
            orders_dir.mkdir()
            (orders_dir / "_SUCCESS").touch()
            records = [
                _raw_order("20260701"),
                _raw_order("20260713", poi_id="poi-b"),
                _raw_order("20260714"),
            ]
            (orders_dir / "part-00000").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output_dir = root / "sid-free"
            result = build_sid_free_training_data(
                orders_dir,
                embedding_dir,
                output_dir,
                TIME_SPLIT,
                HISTORY_WINDOW,
            )

            train_record = json.loads(
                (output_dir / "train.jsonl").read_text(encoding="utf-8")
            )
            test_jsonl_exists = (output_dir / "test.jsonl").exists()

        self.assertFalse(test_jsonl_exists)
        self.assertFalse(result.manifest["isolation"]["old_sid_mapping_loaded"])
        self.assertEqual(train_record["schema_version"], DATA_SCHEMA_VERSION)
        self.assertEqual(
            result.stats["test_record_skipped_before_behavior_formatting"], 1
        )


if __name__ == "__main__":
    unittest.main()
