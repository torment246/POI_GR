"""Synthetic tests for GenPOI Geographic Position Embedding."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.genpoi_geope import (
    GenpoiGeoPEError,
    apply_geope_rotation,
    build_genpoi_geope_embeddings,
    center_and_l2_normalize_embeddings,
    compute_bearing_angles,
    fit_reference_points,
)


class GenpoiGeoPETest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_fixture(
        self,
        *,
        name: str = "default",
        mismatched_id: bool = False,
    ) -> tuple[Path, Path]:
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(32, 16)).astype(np.float32)
        embeddings /= np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )
        embeddings = embeddings.astype(np.float16)

        fixture_root = self.root / name
        fixture_root.mkdir()
        embedding_dir = fixture_root / "embedding"
        embedding_dir.mkdir()
        np.save(
            embedding_dir / "embeddings.npy",
            embeddings,
            allow_pickle=False,
        )
        (embedding_dir / "poi_ids.jsonl").write_text(
            "".join(json.dumps(f"poi-{index}") + "\n" for index in range(32)),
            encoding="utf-8",
        )
        (embedding_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "input": {
                        "total_rows": 32,
                        "fingerprint": "synthetic-bge",
                    },
                    "output": {
                        "shape": [32, 16],
                        "dtype": "float16",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        poi_dir = fixture_root / "pois"
        poi_dir.mkdir()
        (poi_dir / "_SUCCESS").touch()
        records = []
        for index in range(32):
            poi_id = (
                "wrong-id"
                if mismatched_id and index == 7
                else f"poi-{index}"
            )
            records.append(
                {
                    "poi_id": poi_id,
                    "lng": 116.0 + (index % 8) * 0.01,
                    "lat": 39.0 + (index // 8) * 0.01,
                }
            )
        (poi_dir / "part-00000.json").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return embedding_dir, poi_dir

    def test_rotation_matches_formula_and_preserves_norm(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        coordinates = np.asarray([[1.0, 0.0]], dtype=np.float64)
        references = np.asarray([[0.0, 0.0]], dtype=np.float64)
        angles = compute_bearing_angles(coordinates, references)
        self.assertAlmostEqual(float(angles[0, 0]), 0.0)
        rotated = apply_geope_rotation(
            embeddings,
            coordinates,
            references,
        )
        np.testing.assert_allclose(rotated, embeddings, atol=1e-6)

        north = np.asarray([[0.0, 1.0]], dtype=np.float64)
        rotated_north = apply_geope_rotation(
            embeddings,
            north,
            references,
        )
        np.testing.assert_allclose(
            rotated_north,
            [[0.0, 1.0, -1.0, 0.0]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(rotated_north, axis=1),
            np.linalg.norm(embeddings, axis=1),
            atol=1e-6,
        )

    def test_reference_points_are_deterministic_and_sorted(self) -> None:
        coordinates = np.asarray(
            [
                [116.0, 39.0],
                [116.01, 39.0],
                [116.5, 39.5],
                [116.51, 39.5],
            ],
            dtype=np.float64,
        )
        first = fit_reference_points(
            coordinates,
            reference_count=2,
            seed=42,
            n_init=2,
        )
        second = fit_reference_points(
            coordinates,
            reference_count=2,
            seed=42,
            n_init=2,
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[1:, 0] >= first[:-1, 0]))

    def test_center_and_l2_normalization_matches_formula(self) -> None:
        embeddings = np.asarray(
            [[2.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        source_mean = np.asarray([1.0, 1.0], dtype=np.float32)
        actual = center_and_l2_normalize_embeddings(
            embeddings,
            source_mean,
        )
        expected = np.asarray(
            [[1.0, -1.0], [-1.0, 1.0]],
            dtype=np.float32,
        ) / np.sqrt(2.0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(actual, axis=1),
            np.ones(2),
            atol=1e-6,
        )

    def test_local_projection_uses_physical_longitude_scale(self) -> None:
        coordinates = np.asarray(
            [
                [116.0, 40.0],
                [116.1, 40.0],
                [116.0, 40.1],
                [116.1, 40.1],
            ],
            dtype=np.float64,
        )
        references = fit_reference_points(
            coordinates,
            reference_count=2,
            seed=42,
            n_init=2,
            coordinate_projection="local_equirectangular_km",
        )
        self.assertEqual(references.shape, (2, 2))
        self.assertTrue(np.isfinite(references).all())

    def test_full_builder_preserves_alignment_and_is_repeatable(self) -> None:
        embedding_dir, poi_dir = self.write_fixture()
        first_dir = self.root / "first"
        second_dir = self.root / "second"
        first = build_genpoi_geope_embeddings(
            embedding_dir,
            poi_dir,
            first_dir,
            reference_count=4,
            kmeans_n_init=2,
            chunk_rows=7,
        )
        second = build_genpoi_geope_embeddings(
            embedding_dir,
            poi_dir,
            second_dir,
            reference_count=4,
            kmeans_n_init=2,
            chunk_rows=7,
        )
        self.assertEqual(first.output_hashes, second.output_hashes)
        self.assertEqual(
            {path.name for path in first_dir.iterdir()},
            {
                "embeddings.npy",
                "poi_ids.jsonl",
                "reference_points.npy",
                "manifest.json",
            },
        )
        manifest = first.manifest
        self.assertEqual(manifest["input"]["scope"], "full")
        self.assertTrue(
            manifest["input"]["poi_data"][
                "row_order_matches_embedding_ids"
            ]
        )
        self.assertEqual(manifest["output"]["shape"], [32, 16])
        self.assertEqual(manifest["geope"]["segment_dim"], 4)
        self.assertEqual(
            manifest["geope"]["kmeans_input"],
            "local_equirectangular_km",
        )
        self.assertEqual(
            manifest["geope"]["reference_fit_rows"],
            32,
        )
        self.assertLess(
            manifest["metrics"]["max_absolute_norm_delta_float32"],
            1e-5,
        )
        self.assertGreater(
            manifest["metrics"]["mean_vector_l2_delta_float32"],
            0.1,
        )

    def test_prefix_smoke_and_mismatched_ids(self) -> None:
        embedding_dir, poi_dir = self.write_fixture()
        output_dir = self.root / "smoke"
        result = build_genpoi_geope_embeddings(
            embedding_dir,
            poi_dir,
            output_dir,
            reference_count=4,
            kmeans_n_init=2,
            max_rows=20,
        )
        self.assertEqual(result.manifest["input"]["scope"], "prefix_smoke")
        output = np.load(output_dir / "embeddings.npy")
        self.assertEqual(output.shape, (20, 16))

        mismatched_embedding, mismatched_poi = self.write_fixture(
            name="mismatched",
            mismatched_id=True,
        )
        with self.assertRaisesRegex(GenpoiGeoPEError, "行映射不一致"):
            build_genpoi_geope_embeddings(
                mismatched_embedding,
                mismatched_poi,
                self.root / "mismatch",
                reference_count=4,
                kmeans_n_init=2,
            )

    def test_centered_builder_reuses_reference_points(self) -> None:
        embedding_dir, poi_dir = self.write_fixture()
        baseline_dir = self.root / "baseline"
        build_genpoi_geope_embeddings(
            embedding_dir,
            poi_dir,
            baseline_dir,
            reference_count=4,
            kmeans_n_init=2,
        )
        centered_dir = self.root / "centered"
        result = build_genpoi_geope_embeddings(
            embedding_dir,
            poi_dir,
            centered_dir,
            reference_count=4,
            reference_points_path=baseline_dir / "reference_points.npy",
            embedding_preprocessing="global_mean_center_l2",
            chunk_rows=7,
        )

        source = np.asarray(
            np.load(embedding_dir / "embeddings.npy"),
            dtype=np.float32,
        )
        source_mean = source.mean(axis=0, dtype=np.float64).astype(np.float32)
        coordinates = np.asarray(
            [
                [116.0 + (index % 8) * 0.01, 39.0 + (index // 8) * 0.01]
                for index in range(32)
            ],
            dtype=np.float64,
        )
        reference_points = np.load(baseline_dir / "reference_points.npy")
        expected = apply_geope_rotation(
            center_and_l2_normalize_embeddings(source, source_mean),
            coordinates,
            reference_points,
        ).astype(np.float16)
        actual = np.load(centered_dir / "embeddings.npy")
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            np.load(centered_dir / "reference_points.npy"),
            reference_points,
        )
        np.testing.assert_array_equal(
            np.load(centered_dir / "source_mean.npy"),
            source_mean,
        )
        self.assertEqual(
            result.manifest["geope"]["reference_point_method"],
            "provided",
        )
        self.assertEqual(
            result.manifest["geope"]["embedding_preprocessing"]["mode"],
            "global_mean_center_l2",
        )
        self.assertLess(
            result.manifest["metrics"][
                "max_absolute_rotation_norm_delta_float32"
            ],
            1e-5,
        )

    def test_invalid_segment_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(GenpoiGeoPEError, "每个 GeoPE 分段"):
            apply_geope_rotation(
                np.ones((2, 12), dtype=np.float32),
                np.asarray([[0.0, 0.0], [1.0, 1.0]]),
                np.asarray(
                    [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]
                ),
            )


if __name__ == "__main__":
    unittest.main()
