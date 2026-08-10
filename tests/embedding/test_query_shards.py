"""Tests for deterministic Train Query hash sharding."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from poi_gr.embedding.query_shards import (
    QueryShardError,
    build_train_query_shards,
    stable_query_shard,
    validate_query_shards,
)


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


def _write_source(sft_dir: Path, records: list[dict[str, object]]) -> None:
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


class QueryShardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sft_dir = self.root / "sft"
        self.sft_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_preserves_rows_and_colocates_equal_queries(self) -> None:
        records = [
            _sample("北京南站", "p1"),
            _sample("合生汇B1层", "p2"),
            _sample("北京南站", "p3"),
            _sample(" 409号楼 ", "p4"),
        ]
        _write_source(self.sft_dir, records)
        output_dir = self.root / "query_shards"
        result = build_train_query_shards(
            self.sft_dir,
            output_dir,
            num_shards=4,
            buffer_rows_per_shard=2,
            show_progress=False,
        )
        self.assertEqual(result.total_rows, 4)
        self.assertFalse(result.reused)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["contract"]["query_normalization"], "none")
        self.assertEqual(manifest["sharding"]["total_rows"], 4)

        observed: list[tuple[int, str, str]] = []
        for file_info in manifest["sharding"]["files"]:
            table = pq.read_table(output_dir / file_info["file"])
            for row in table.to_pylist():
                observed.append(
                    (file_info["shard_id"], row["query"], row["target_poi_id"])
                )
        self.assertEqual(len(observed), 4)
        beijing_shards = {
            shard_id for shard_id, query, _ in observed if query == "北京南站"
        }
        self.assertEqual(beijing_shards, {stable_query_shard("北京南站", 4)})
        self.assertIn((stable_query_shard(" 409号楼 ", 4), " 409号楼 ", "p4"), observed)

        validated = validate_query_shards(output_dir)
        self.assertEqual(validated.total_rows, 4)
        reused = build_train_query_shards(
            self.sft_dir,
            output_dir,
            num_shards=4,
            buffer_rows_per_shard=2,
            show_progress=False,
        )
        self.assertTrue(reused.reused)

    def test_rejects_non_train_record(self) -> None:
        record = _sample("北京南站", "p1")
        record["split"] = "valid"
        _write_source(self.sft_dir, [record])
        with self.assertRaisesRegex(QueryShardError, "split 不是 train"):
            build_train_query_shards(
                self.sft_dir,
                self.root / "query_shards",
                num_shards=2,
                buffer_rows_per_shard=1,
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main()
