#!/usr/bin/env python3
"""Run the bounded-memory R3 Query-Predictable RQ-KMeans screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sid.query_predictable_rqkmeans import (  # noqa: E402
    QueryPredictableRQKMeansError,
    QueryPredictableScreenConfig,
    run_query_predictable_screen,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sid/query_predictable_rqkmeans_bge_m3_1024x3.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        config = QueryPredictableScreenConfig(
            content_embeddings=_path(raw["data"]["content_embeddings"]),
            fused_embeddings=_path(raw["data"]["fused_embeddings"]),
            query_aggregates=_path(raw["data"]["query_aggregates"]),
            covered_poi_rows=_path(raw["data"]["covered_poi_rows"]),
            category_indices=_path(raw["data"]["category_indices"]),
            category_means=_path(raw["data"]["category_means"]),
            reference_dir=_path(raw["initialization"]["reference_dir"]),
            sample_cache_dir=_path(raw["initialization"]["sample_cache_dir"]),
            output_dir=_path(raw["output_dir"]),
            codebook_sizes=tuple(int(value) for value in raw["quantizer"]["codebook_sizes"]),
            query_weights=tuple(float(value) for value in raw["quantizer"]["query_weights"]),
            beta=float(raw["data"]["beta"]),
            fusion_alpha=float(raw["data"]["fusion_alpha"]),
            iterations=int(raw["quantizer"]["iterations"]),
            chunk_rows=int(raw["runtime"]["chunk_rows"]),
            scan_chunk_rows=int(raw["runtime"]["scan_chunk_rows"]),
            content_top_t=(
                None
                if raw["quantizer"].get("content_top_t") is None
                else int(raw["quantizer"]["content_top_t"])
            ),
            expected_rows=int(raw["expected"]["rows"]),
            expected_dimension=int(raw["expected"]["dimension"]),
            expected_sample_indices_sha256=str(
                raw["expected"]["sample_indices_sha256"]
            ),
            routing_mode=str(raw["quantizer"].get("routing_mode", "lloyd")),
            query_support_tau=float(
                raw["quantizer"].get("query_support_tau", 20.0)
            ),
            validation_query_embeddings=(
                None
                if raw.get("validation") is None
                else _path(raw["validation"]["query_embeddings"])
            ),
            validation_query_mapping=(
                None
                if raw.get("validation") is None
                else _path(raw["validation"]["query_mapping"])
            ),
            poi_ids=(
                None
                if raw.get("validation") is None
                else _path(raw["validation"]["poi_ids"])
            ),
            expected_validation_rows=(
                None
                if raw.get("validation") is None
                else int(raw["validation"]["expected_rows"])
            ),
            expected_validation_query_sha256=(
                None
                if raw.get("validation") is None
                else str(raw["validation"]["expected_query_embeddings_sha256"])
            ),
            expected_validation_mapping_sha256=(
                None
                if raw.get("validation") is None
                else str(raw["validation"]["expected_query_mapping_sha256"])
            ),
            expected_poi_ids_sha256=(
                None
                if raw.get("validation") is None
                else str(raw["validation"]["expected_poi_ids_sha256"])
            ),
        )
        result = run_query_predictable_screen(config)
    except (OSError, ValueError, KeyError, TypeError, QueryPredictableRQKMeansError) as error:
        print(f"QD-RQ R3 screen failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
