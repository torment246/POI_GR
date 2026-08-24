import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from poi_gr.pid.trie import sha256_file
from poi_gr.sft.generalization_subset import (
    SUBSET_NAMES,
    build_generalization_validation_suite,
    classify_generalization_row,
)


def record(index: int, query: str, target_poi_id: str) -> dict:
    user_gid = "".join(f"<G_{value}>" for value in "wx4g0b")
    target_gid = "".join(f"<G_{value}>" for value in "wx4g1c")
    target = target_gid + "<S1_1><S2_2><S3_3>"
    return {
        "sample_id": f"sample-{index}",
        "messages": [
            {
                "role": "user",
                "content": (
                    "<CURRENT>\n"
                    f"<USER_GID>{user_gid}</USER_GID>\n"
                    f"<QUERY>{query}</QUERY>\n"
                    "</CURRENT>"
                ),
            },
            {"role": "assistant", "content": target},
        ],
        "order_id": f"order-{index}",
        "searchid": f"search-{index}",
        "target_poi_id": target_poi_id,
        "target_pid_key": target,
        "requires_dedup": False,
        "history_length": 0,
        "split": "valid",
    }


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def write_reference_manifest(path: Path, data_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "output_sha256": sha256_file(data_path),
            }
        ),
        encoding="utf-8",
    )


class GeneralizationSubsetTest(unittest.TestCase):
    def test_mutually_exclusive_classification(self) -> None:
        self.assertEqual(
            classify_generalization_row(
                query_seen=True,
                pair_seen=False,
                target_frequency=6,
                long_tail_max_frequency=5,
            ),
            "seen_query_unseen_pair",
        )
        self.assertEqual(
            classify_generalization_row(
                query_seen=False,
                pair_seen=False,
                target_frequency=6,
                long_tail_max_frequency=5,
            ),
            "unseen_query_seen_target",
        )
        self.assertEqual(
            classify_generalization_row(
                query_seen=True,
                pair_seen=True,
                target_frequency=2,
                long_tail_max_frequency=5,
            ),
            "long_tail_target",
        )
        self.assertEqual(
            classify_generalization_row(
                query_seen=False,
                pair_seen=False,
                target_frequency=0,
                long_tail_max_frequency=5,
            ),
            "cold_target",
        )

    def test_builds_four_disjoint_subsets_and_preserves_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / "train_shards"
            train_dir.mkdir()
            train_rows = (
                [("已见查询甲", "poi-a")] * 3
                + [("已见查询乙", "poi-b")] * 3
                + [("长尾查询甲", "poi-l1")]
                + [("长尾查询乙", "poi-l2")] * 2
            )
            shard = train_dir / "part-00000-of-00001.parquet"
            pq.write_table(
                pa.table(
                    {
                        "query": [value[0] for value in train_rows],
                        "target_poi_id": [value[1] for value in train_rows],
                    }
                ),
                shard,
                compression="zstd",
            )
            train_manifest = {
                "schema_version": "train-query-hash-shards-v1",
                "status": "completed",
                "contract": {
                    "split": "train",
                    "output_fields": ["query", "target_poi_id"],
                },
                "sharding": {
                    "total_rows": len(train_rows),
                    "files": [
                        {
                            "file": shard.name,
                            "rows": len(train_rows),
                            "sha256": sha256_file(shard),
                        }
                    ],
                },
            }
            (train_dir / "manifest.json").write_text(
                json.dumps(train_manifest), encoding="utf-8"
            )

            values = [
                record(0, "已见查询甲", "poi-b"),
                record(1, "已见查询乙", "poi-a"),
                record(2, "全新查询甲", "poi-a"),
                record(3, "全新查询乙", "poi-b"),
                record(4, "任意查询甲", "poi-l1"),
                record(5, "任意查询乙", "poi-l2"),
                record(6, "冷启动查询甲", "poi-c1"),
                record(7, "冷启动查询乙", "poi-c2"),
                record(8, "随机参考查询", "poi-r"),
                record(9, "地理参考查询", "poi-g"),
            ]
            valid_file = root / "valid.jsonl"
            write_jsonl(valid_file, values)
            random_file = root / "random.jsonl"
            geo_file = root / "geo.jsonl"
            write_jsonl(random_file, [values[8], values[9]])
            write_jsonl(geo_file, [values[9], values[8]])
            random_manifest = root / "random_manifest.json"
            geo_manifest = root / "geo_manifest.json"
            write_reference_manifest(random_manifest, random_file)
            write_reference_manifest(geo_manifest, geo_file)

            output_dir = root / "output"
            suite = build_generalization_validation_suite(
                valid_file,
                train_dir,
                random_file,
                random_manifest,
                geo_file,
                geo_manifest,
                output_dir,
                source_rows=len(values),
                source_sha256=sha256_file(valid_file),
                subset_size=2,
                long_tail_max_frequency=2,
            )
            self.assertEqual(set(suite.manifest["subsets"]), set(SUBSET_NAMES))
            self.assertEqual(
                suite.manifest["business_key_overlap_matrix"]["random_traffic"][
                    "geo_complex"
                ],
                2,
            )
            selected_keys: set[tuple[str, str]] = set()
            for subset_name in SUBSET_NAMES:
                spec = suite.manifest["subsets"][subset_name]
                self.assertEqual(spec["output_rows"], 2)
                rows = [
                    json.loads(line)
                    for line in Path(spec["output_file"])
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                keys = {(row["order_id"], row["searchid"]) for row in rows}
                self.assertFalse(keys & selected_keys)
                selected_keys.update(keys)
            reused = build_generalization_validation_suite(
                valid_file,
                train_dir,
                random_file,
                random_manifest,
                geo_file,
                geo_manifest,
                output_dir,
                source_rows=len(values),
                source_sha256=sha256_file(valid_file),
                subset_size=2,
                long_tail_max_frequency=2,
            )
            self.assertEqual(reused.manifest, suite.manifest)


if __name__ == "__main__":
    unittest.main()
