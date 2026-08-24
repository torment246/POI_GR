"""Integration tests for the frozen QGR-SID M2-A lexical proxy."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.proxy import (  # noqa: E402
    evaluate_lexical_relation_proxy,
    profile_temporal_split,
    validate_proxy_output,
)
from poi_gr.methods.tiger.identifier import sha256_file  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class LexicalRelationProxyTest(unittest.TestCase):
    def _build_fixture(
        self, root: Path
    ) -> tuple[Path, Path, Path, Path, Path]:
        poi_dir = root / "poi"
        order_dir = root / "orders"
        identifier_dir = root / "identifier"
        poi_dir.mkdir()
        order_dir.mkdir()
        identifier_dir.mkdir()

        pois = [
            {"poi_id": "100", "displayname": "唯一地点"},
            {"poi_id": "101", "displayname": "春风小区1号楼"},
            {"poi_id": "102", "displayname": "春风小区2号楼"},
            {"poi_id": "103", "displayname": "春风小区3号楼"},
        ]
        poi_path = poi_dir / "part-00000.json"
        with poi_path.open("w", encoding="utf-8") as handle:
            for poi in pois:
                handle.write(
                    json.dumps(poi, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )

        mapping_path = identifier_dir / "poi_tiger_id_mapping.parquet"
        pq.write_table(
            pa.table(
                {
                    "poi_id": pa.array(["100", "101", "102", "103"]),
                    "s1": pa.array([0, 1, 1, 1], type=pa.int32()),
                    "s2": pa.array([0, 2, 2, 2], type=pa.int32()),
                    "s3": pa.array([0, 3, 3, 3], type=pa.int32()),
                    "base_sid_bucket_size": pa.array(
                        [1, 3, 3, 3], type=pa.int32()
                    ),
                }
            ),
            mapping_path,
        )
        write_json(
            identifier_dir / "tiger_id_manifest.json",
            {
                "schema_version": "tiger-item-identifier-v1",
                "status": "completed",
                "mapping": {
                    "path": mapping_path.name,
                    "rows": 4,
                    "sha256": sha256_file(mapping_path),
                },
            },
        )
        write_json(
            identifier_dir / "metrics.json",
            {
                "base_sid_colliding_poi_count": 3,
                "base_sid_collision_bucket_count": 1,
            },
        )

        early_targets = ["101"] * 12 + ["102"] * 4 + ["103"] * 2
        holdout_targets = ["102", "103"]
        order_path = order_dir / "part-00000.json"
        with order_path.open("w", encoding="utf-8") as handle:
            for index, poi_id in enumerate(
                early_targets + holdout_targets, start=1
            ):
                minute = index
                record = {
                    "order_id": f"o{index}",
                    "searchid": f"s{index}",
                    "query": f"春风小区{int(poi_id) - 100}号楼",
                    "create_time": f"2026-07-01 00:{minute:02d}:00",
                    "source_dt": "20260701",
                    "poi_id": poi_id,
                }
                handle.write(
                    json.dumps(
                        record, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )
        sft_manifest_path = root / "sft_manifest.json"
        write_json(
            sft_manifest_path,
            {
                "schema_version": "sft-main-data-v1",
                "status": "completed",
                "orders": {
                    "files": [
                        {
                            "relative_path": order_path.name,
                            "rows": 20,
                            "sha256": sha256_file(order_path),
                        }
                    ]
                },
                "outputs": {"train.jsonl": {"rows": 20}},
            },
        )
        m1_manifest_path = root / "m1_manifest.json"
        write_json(
            m1_manifest_path,
            {
                "schema_version": "qgr-sid-numeric-relation-audit-v1",
                "status": "completed",
            },
        )
        return (
            poi_dir,
            order_dir,
            identifier_dir,
            sft_manifest_path,
            m1_manifest_path,
        )

    def test_temporal_profile_uses_exact_event_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, order_dir, _, _, _ = self._build_fixture(root)
            split = profile_temporal_split(order_dir, early_ratio=0.9)
            self.assertEqual(split.train_order_count, 20)
            self.assertEqual(split.early_target_count, 18)
            self.assertEqual(split.holdout_target_count, 2)
            self.assertEqual(split.cutoff_minute, "2026-07-01 00:18")

    def test_full_proxy_is_reproducible_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                poi_dir,
                order_dir,
                identifier_dir,
                sft_manifest_path,
                m1_manifest_path,
            ) = self._build_fixture(root)
            output_dir = root / "proxy"
            result = evaluate_lexical_relation_proxy(
                project_root=PROJECT_ROOT,
                poi_dir=poi_dir,
                order_dir=order_dir,
                sft_manifest_path=sft_manifest_path,
                identifier_dir=identifier_dir,
                m1_manifest_path=m1_manifest_path,
                output_dir=output_dir,
                batch_rows=2,
                examples_per_kind=2,
            )
            self.assertEqual(
                result.metrics["holdout"]["collision_holdout_order_count"], 2
            )
            ranking = result.metrics["holdout"]["ranking"]
            self.assertGreater(
                ranking["query_guided"]["hr_at_1"],
                ranking["popularity"]["hr_at_1"],
            )
            self.assertEqual(
                result.metrics["temporal_split"]["early_target_count"], 18
            )
            validation = validate_proxy_output(output_dir)
            self.assertEqual(validation["status"], "validated")
            self.assertEqual(validation["checked_output_count"], 14)


if __name__ == "__main__":
    unittest.main()
