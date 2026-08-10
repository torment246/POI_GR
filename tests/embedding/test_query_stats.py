"""Tests for bounded-memory Train Query and POI statistics."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from poi_gr.embedding.query_shards import build_train_query_shards
from poi_gr.embedding.query_stats import (
    build_train_query_stats,
    stable_poi_partition,
    validate_train_query_stats,
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


def _read_all(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("part-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


class QueryStatsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sft_dir = self.root / "sft"
        self.sft_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_frequency_df_and_poi_coverage(self) -> None:
        records = [
            _sample("北京南站", "p1"),
            _sample("北京南站", "p1"),
            _sample("北京南站", "p2"),
            _sample("停车场", "p1"),
            _sample("停车场", "p1"),
            _sample("合生汇B1", "p3"),
        ]
        _write_source(self.sft_dir, records)
        query_shards_dir = self.root / "query_shards"
        build_train_query_shards(
            self.sft_dir,
            query_shards_dir,
            num_shards=4,
            buffer_rows_per_shard=2,
            show_progress=False,
        )
        output_dir = self.root / "query_stats"
        result = build_train_query_stats(
            query_shards_dir,
            output_dir,
            poi_partitions=4,
            poi_buffer_rows=2,
            read_batch_rows=2,
            show_progress=False,
        )

        self.assertEqual(result.source_rows, 6)
        self.assertEqual(result.unique_queries, 3)
        self.assertEqual(result.unique_query_poi_pairs, 4)
        self.assertEqual(result.covered_pois, 3)
        self.assertFalse(result.reused)

        catalog = _read_all(output_dir / "query_catalog")
        by_query = {row["query"]: row for row in catalog}
        self.assertEqual(sorted(row["query_id"] for row in catalog), [0, 1, 2])
        self.assertEqual(by_query["北京南站"]["train_order_count"], 3)
        self.assertEqual(by_query["北京南站"]["poi_df"], 2)
        self.assertEqual(by_query["停车场"]["train_order_count"], 2)
        self.assertEqual(by_query["停车场"]["poi_df"], 1)

        query_poi = _read_all(output_dir / "query_poi")
        pairs = {
            (row["query_id"], row["target_poi_id"]): row["train_order_count"]
            for row in query_poi
        }
        beijing_id = by_query["北京南站"]["query_id"]
        self.assertEqual(pairs[(beijing_id, "p1")], 2)
        self.assertEqual(pairs[(beijing_id, "p2")], 1)

        poi_stats = {
            row["target_poi_id"]: row for row in _read_all(output_dir / "poi_stats")
        }
        self.assertEqual(poi_stats["p1"]["train_order_count"], 4)
        self.assertEqual(poi_stats["p1"]["unique_query_count"], 2)
        self.assertEqual(poi_stats["p2"]["train_order_count"], 1)
        self.assertEqual(poi_stats["p3"]["unique_query_count"], 1)

        validated = validate_train_query_stats(output_dir)
        self.assertTrue(validated.reused)
        reused = build_train_query_stats(
            query_shards_dir,
            output_dir,
            poi_partitions=4,
            poi_buffer_rows=2,
            read_batch_rows=2,
            show_progress=False,
        )
        self.assertTrue(reused.reused)

    def test_poi_partition_is_stable(self) -> None:
        first = stable_poi_partition("2156309362375659522", 256)
        self.assertEqual(
            first,
            stable_poi_partition("2156309362375659522", 256),
        )
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 256)


if __name__ == "__main__":
    unittest.main()
