"""Tests for the streaming QGR-SID relation coverage audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.audit import (  # noqa: E402
    QgrSidAuditError,
    audit_numeric_relations,
    validate_audit_output,
)
from poi_gr.methods.tiger.identifier import sha256_file  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class NumericRelationAuditTest(unittest.TestCase):
    def _build_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        poi_dir = root / "poi"
        identifier_dir = root / "identifier"
        query_dir = root / "query"
        poi_dir.mkdir()
        identifier_dir.mkdir()
        query_dir.mkdir()

        pois = [
            {"poi_id": "p0", "displayname": "唯一POI", "category_code": "1"},
            {
                "poi_id": "p1",
                "displayname": "小区1号楼",
                "category_code": "2",
                "category": "楼栋号",
            },
            {
                "poi_id": "p2",
                "displayname": "小区2056号楼",
                "category_code": "2",
                "category": "楼栋号",
            },
            {
                "poi_id": "p3",
                "displayname": "没有数字关系",
                "category_code": "3",
                "category": "设施",
            },
        ]
        with (poi_dir / "part-00000.json").open("w", encoding="utf-8") as handle:
            for poi in pois:
                handle.write(json.dumps(poi, ensure_ascii=False) + "\n")

        mapping_path = identifier_dir / "poi_tiger_id_mapping.parquet"
        pq.write_table(
            pa.table(
                {
                    "poi_id": pa.array(["p0", "p1", "p2", "p3"]),
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

        arrays = {
            "covered_poi_rows": np.asarray([1, 2, 3], dtype=np.int64),
            "train_order_count": np.asarray([10, 5, 1], dtype=np.int64),
            "unique_query_count": np.asarray([3, 2, 1], dtype=np.int32),
        }
        contracts = {}
        for name, array in arrays.items():
            path = query_dir / f"{name}.npy"
            np.save(path, array)
            contracts[name] = {
                "file": path.name,
                "sha256": sha256_file(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        write_json(
            query_dir / "manifest.json",
            {
                "schema_version": "query-poi-aggregates-v1",
                "status": "completed",
                "outputs": contracts,
            },
        )
        return poi_dir, identifier_dir, query_dir

    def test_full_audit_aligns_rows_and_reports_unweighted_and_query_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poi_dir, identifier_dir, query_dir = self._build_fixture(root)
            output_dir = root / "audit"
            result = audit_numeric_relations(
                poi_dir=poi_dir,
                identifier_dir=identifier_dir,
                output_dir=output_dir,
                query_aggregate_dir=query_dir,
                value_max=1055,
                batch_rows=2,
                examples_per_relation=2,
            )
            metrics = result.metrics
            self.assertEqual(metrics["status"], "completed")
            self.assertEqual(metrics["scanned_poi_count"], 4)
            self.assertEqual(metrics["colliding_poi_count"], 3)
            self.assertEqual(metrics["stable_lexical_relation_poi_count"], 2)
            self.assertEqual(metrics["in_vocab_relation_poi_count"], 1)
            self.assertEqual(
                metrics["per_relation"]["R_BUILDING"]["out_of_vocab_poi_count"],
                1,
            )
            weighted = metrics["query_weighted_coverage"]
            self.assertEqual(weighted["collision_train_order_count"], 16)
            self.assertEqual(weighted["in_vocab_relation_train_order_count"], 10)
            self.assertTrue((output_dir / "_SUCCESS").is_file())
            self.assertEqual(validate_audit_output(output_dir)["status"], "validated")

    def test_row_alignment_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poi_dir, identifier_dir, _ = self._build_fixture(root)
            raw_path = poi_dir / "part-00000.json"
            rows = raw_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["poi_id"] = "wrong"
            rows[0] = json.dumps(first, ensure_ascii=False)
            raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(QgrSidAuditError, "行对齐失败"):
                audit_numeric_relations(
                    poi_dir=poi_dir,
                    identifier_dir=identifier_dir,
                    output_dir=root / "audit",
                    batch_rows=2,
                )


if __name__ == "__main__":
    unittest.main()
