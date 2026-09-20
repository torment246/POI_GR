"""Synthetic tests for Geohash6 Geo-Semantic PID construction."""

from __future__ import annotations

import hashlib
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

from poi_gr.pid.geohash import (
    GeohashPidError,
    build_geohash_pid,
    compose_pid,
    encode_geohash,
    geohash_to_tokens,
    tokens_to_geohash,
)
from poi_gr.sid.evaluation import compute_basic_metrics


class GeohashPidTest(unittest.TestCase):
    def test_active_512_requires_explicit_matching_capacity(self) -> None:
        manifest_path, poi_path = self.write_fixture()
        manifest = json.loads(manifest_path.read_text())
        manifest["codebook_sizes"] = [512, 512, 512]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(GeohashPidError):
            build_geohash_pid(manifest_path, poi_path, self.root / "wrong")
        result = build_geohash_pid(manifest_path, poi_path, self.root / "active", sid_codebook_size=512)
        self.assertEqual(result.manifest["pid_codes"]["codebook_sizes"], [32] * 6 + [512] * 3)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_fixture(
        self,
        *,
        coordinates: list[tuple[float, float]] | None = None,
        metadata_order: list[int] | None = None,
        experiment_id: str = "BJ-RQVAE-1024x3",
    ) -> tuple[Path, Path]:
        sid_codes = np.asarray(
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [4, 5, 6],
            ],
            dtype=np.int32,
        )
        poi_ids = ["p1", "p2", "p3", "p4"]
        if coordinates is None:
            coordinates = [
                (116.3970, 39.9080),
                (121.4737, 31.2304),
                (116.3971, 39.9081),
                (116.5000, 39.9000),
            ]
        if metadata_order is None:
            metadata_order = list(range(len(poi_ids)))

        sid_path = self.root / "sid_codes.npy"
        ids_path = self.root / "poi_ids.jsonl"
        manifest_path = self.root / "sid_manifest.json"
        metrics_path = self.root / "metrics.json"
        poi_path = self.root / "pois.jsonl"
        np.save(sid_path, sid_codes, allow_pickle=False)
        ids_path.write_text(
            "".join(json.dumps(poi_id) + "\n" for poi_id in poi_ids),
            encoding="utf-8",
        )
        ids_hash = hashlib.sha256(ids_path.read_bytes()).hexdigest()
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "sid-input-v1",
                    "experiment_id": experiment_id,
                    "method": "vanilla_rqvae",
                    "checkpoint": {
                        "epoch": 20,
                        "path": "checkpoint_epoch_20.pt",
                        "sha256": "synthetic-checkpoint",
                    },
                    "sid_codes": {
                        "path": sid_path.name,
                        "shape": list(sid_codes.shape),
                        "dtype": str(sid_codes.dtype),
                    },
                    "poi_ids": {
                        "path": ids_path.name,
                        "sha256": ids_hash,
                    },
                    "codebook_sizes": [1024, 1024, 1024],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        basic, _, _, _ = compute_basic_metrics(sid_codes)
        metrics_path.write_text(
            json.dumps(
                {
                    "basic": basic,
                    "inputs": {"poi_data": str(poi_path)},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = []
        for index in metadata_order:
            longitude, latitude = coordinates[index]
            records.append(
                {
                    "poi_id": poi_ids[index],
                    "displayname": f"合成POI-{index + 1}",
                    "address": f"合成地址-{index + 1}",
                    "category": "合成类别",
                    "category_code": "100000",
                    "lng": longitude,
                    "lat": latitude,
                    "layer": index,
                    "click_score": float(index),
                }
            )
        poi_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return manifest_path, poi_path

    def test_standard_geohash_encoding(self) -> None:
        self.assertEqual(encode_geohash(-5.6, 42.6, 6), "ezs42e")

    def test_geohash_prefix_is_stable(self) -> None:
        full = encode_geohash(116.397, 39.908, 6)
        self.assertEqual(encode_geohash(116.397, 39.908, 4), full[:4])
        self.assertEqual(encode_geohash(116.397, 39.908, 5), full[:5])

    def test_gid_token_conversion_is_reversible(self) -> None:
        value = "wx4g0b"
        tokens = geohash_to_tokens(value)
        self.assertEqual(tokens.dtype, np.uint8)
        self.assertEqual(tokens_to_geohash(tokens), value)

    def test_pid_order_is_gid_then_sid(self) -> None:
        gid = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        sid = np.asarray([[101, 202, 303]], dtype=np.int32)
        pid = compose_pid(gid, sid)
        np.testing.assert_array_equal(
            pid, np.asarray([[1, 2, 3, 4, 5, 6, 101, 202, 303]])
        )
        self.assertEqual(pid.dtype, np.int32)

    def test_gid_splits_distant_same_sid_and_retains_local_collision(self) -> None:
        manifest_path, poi_path = self.write_fixture()
        result = build_geohash_pid(
            manifest_path,
            poi_path,
            self.root / "output",
        )
        resolution = result.metrics["collision_resolution"]
        self.assertEqual(resolution["sid_colliding_poi_count"], 3)
        self.assertEqual(resolution["sid_colliding_poi_became_unique_count"], 1)
        self.assertEqual(resolution["residual_colliding_poi_count"], 2)
        self.assertEqual(resolution["residual_pid_bucket_count"], 1)
        self.assertEqual(resolution["maximum_residual_pid_bucket_size"], 2)

    def test_accepts_compatible_sid_from_another_experiment(self) -> None:
        manifest_path, poi_path = self.write_fixture(
            experiment_id="GenPOI-BGE-M3-GeoPE-1024x3"
        )
        result = build_geohash_pid(
            manifest_path,
            poi_path,
            self.root / "genpoi-output",
        )
        self.assertEqual(
            result.manifest["sid_source"]["experiment_id"],
            "GenPOI-BGE-M3-GeoPE-1024x3",
        )

    def test_poi_id_and_sid_row_mapping_must_match(self) -> None:
        manifest_path, poi_path = self.write_fixture(metadata_order=[1, 0, 2, 3])
        with self.assertRaisesRegex(GeohashPidError, "行映射不一致"):
            build_geohash_pid(
                manifest_path,
                poi_path,
                self.root / "misaligned-output",
            )

    def test_invalid_coordinates_are_rejected_without_outputs(self) -> None:
        coordinates = [
            (116.3970, 39.9080),
            (121.4737, 31.2304),
            (float("nan"), 39.9081),
            (116.5000, 91.0),
        ]
        manifest_path, poi_path = self.write_fixture(coordinates=coordinates)
        output_dir = self.root / "invalid-output"
        with self.assertRaisesRegex(GeohashPidError, "lng 非法"):
            build_geohash_pid(manifest_path, poi_path, output_dir)
        self.assertFalse(output_dir.exists())

    def test_manifest_recovers_complete_pid_definition(self) -> None:
        manifest_path, poi_path = self.write_fixture()
        output_dir = self.root / "manifest-output"
        build_geohash_pid(manifest_path, poi_path, output_dir)
        manifest = json.loads(
            (output_dir / "pid_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], "geohash-pid-v1")
        self.assertEqual(manifest["poi_count"], 4)
        self.assertEqual(manifest["geohash"]["alphabet"], "0123456789bcdefghjkmnpqrstuvwxyz")
        self.assertEqual(manifest["geohash"]["length"], 6)
        self.assertEqual(manifest["geohash"]["coordinate_conversion"], "none")
        self.assertEqual(
            manifest["pid_codes"]["token_order"],
            ["G1", "G2", "G3", "G4", "G5", "G6", "S1", "S2", "S3"],
        )
        self.assertEqual(
            manifest["pid_codes"]["codebook_sizes"],
            [32, 32, 32, 32, 32, 32, 1024, 1024, 1024],
        )
        pid = np.load(output_dir / manifest["pid_codes"]["path"])
        gid = np.load(output_dir / manifest["gid_codes"]["path"])
        sid = np.load(self.root / "sid_codes.npy")
        np.testing.assert_array_equal(pid[:, :6], gid)
        np.testing.assert_array_equal(pid[:, 6:], sid)
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {
                "gid_codes.npy",
                "pid_codes.npy",
                "pid_manifest.json",
                "metrics.json",
                "residual_collision_cases.jsonl",
                "comparison.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
