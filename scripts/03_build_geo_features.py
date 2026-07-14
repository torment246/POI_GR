#!/usr/bin/env python3
"""Build MobilityBench POI geographic features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import format_rate, joined_examples  # noqa: E402
from geo_features import (  # noqa: E402
    build_raw_geo_features,
    encode_geohashes,
    invalid_latlon_mask,
    standardize_features,
)
from text_normalize import clean_text  # noqa: E402


SID_COLUMNS = ["row_id", "poi_id", "name", "address", "city", "lat", "lon", "category_l1", "tags"]
EMBEDDING_META_REQUIRED_COLUMNS = ["row_id", "poi_id", "lat", "lon"]
EMBEDDING_META_OPTIONAL_COLUMNS = ["name", "address", "city"]
GEO_META_COLUMNS = ["row_id", "poi_id", "name", "address", "city", "lat", "lon", "geohash5", "geohash6", "geohash7"]
MANUAL_SAMPLE_COLUMNS = ["sample_group", *GEO_META_COLUMNS]
REPORT_SAMPLE_COLUMNS = ["row_id", "poi_id", "name", "city", "address", "lat", "lon", "geohash6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid-input", default="data/sid/poi_sid_train.parquet", help="Input POI SID parquet.")
    parser.add_argument(
        "--embedding-meta",
        default="data/embeddings/poi_embedding_meta.parquet",
        help="Input POI embedding metadata parquet.",
    )
    parser.add_argument("--out-features", default="data/geo/poi_geo_features.npy", help="Output geo feature npy.")
    parser.add_argument("--out-meta", default="data/geo/poi_geo_meta.parquet", help="Output geo metadata parquet.")
    parser.add_argument("--stats-out", default="outputs/geo/geo_feature_stats.json", help="Output feature stats JSON.")
    parser.add_argument("--anchors-out", default="outputs/geo/geo_anchors.npy", help="Output KMeans anchor npy.")
    parser.add_argument("--report", default="outputs/reports/poi_geo_quality_report.md", help="Output quality report.")
    parser.add_argument("--report-dir", default=None, help="Directory for extra CSV reports.")
    parser.add_argument("--n-anchors", type=int, default=16, help="Number of spatial anchors.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for KMeans and sampling.")
    parser.add_argument("--qrels", default="data/processed/mobilitybench/qrels.csv", help="Optional qrels CSV.")
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit("Failed to read parquet. Please install pyarrow: pip install pyarrow") from exc


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise SystemExit("Failed to write parquet. Please install pyarrow: pip install pyarrow") from exc
    except ValueError as exc:
        if "pyarrow" in str(exc).lower() or "fastparquet" in str(exc).lower():
            raise SystemExit("Failed to write parquet. Please install pyarrow: pip install pyarrow") from exc
        raise


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def add_optional_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def clean_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(clean_text)
    return df


def series_as_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(clean_text)


def assert_unique_non_empty_poi_id(df: pd.DataFrame, label: str) -> None:
    empty_count = int(series_as_text(df["poi_id"]).eq("").sum())
    if empty_count:
        raise ValueError(f"{label} contains empty poi_id rows: {empty_count}")
    duplicate_count = int(series_as_text(df["poi_id"]).duplicated(keep="first").sum())
    if duplicate_count:
        examples = series_as_text(df.loc[series_as_text(df["poi_id"]).duplicated(keep=False), "poi_id"]).head(5).tolist()
        raise ValueError(f"{label} contains duplicate poi_id rows: {duplicate_count}, examples={examples}")


def aligned_sid_to_embedding_meta(sid: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_columns(sid, SID_COLUMNS, "poi_sid_train.parquet")
    ensure_columns(meta, EMBEDDING_META_REQUIRED_COLUMNS, "poi_embedding_meta.parquet")
    meta = add_optional_columns(meta.copy(), EMBEDDING_META_OPTIONAL_COLUMNS)
    sid = sid.copy()

    sid["poi_id"] = series_as_text(sid["poi_id"])
    meta["poi_id"] = series_as_text(meta["poi_id"])
    clean_string_columns(sid, ["name", "address", "city"])
    clean_string_columns(meta, ["name", "address", "city"])

    assert_unique_non_empty_poi_id(sid, "poi_sid_train.parquet")
    assert_unique_non_empty_poi_id(meta, "poi_embedding_meta.parquet")

    sid_ids = set(sid["poi_id"])
    meta_ids = set(meta["poi_id"])
    sid_minus_meta = sid_ids - meta_ids
    meta_minus_sid = meta_ids - sid_ids
    if sid_minus_meta or meta_minus_sid:
        raise ValueError(
            "poi_sid_train.parquet and poi_embedding_meta.parquet have different poi_id sets: "
            f"sid_minus_meta={len(sid_minus_meta)}, meta_minus_sid={len(meta_minus_sid)}"
        )

    initial_row_order_poi_aligned = sid["poi_id"].tolist() == meta["poi_id"].tolist()
    initial_row_id_aligned = series_as_text(sid["row_id"]).tolist() == series_as_text(meta["row_id"]).tolist()

    sid_part = sid[["poi_id", "row_id", "name", "address", "city", "lat", "lon"]].copy()
    meta_part = meta[["poi_id", "row_id", "name", "address", "city", "lat", "lon"]].copy()
    meta_part["_embedding_order"] = np.arange(len(meta_part), dtype=np.int64)
    merged = meta_part.merge(sid_part, on="poi_id", how="left", suffixes=("_meta", "_sid"), sort=False)
    merged = merged.sort_values("_embedding_order").reset_index(drop=True)

    row_id_meta = series_as_text(merged["row_id_meta"])
    row_id_sid = series_as_text(merged["row_id_sid"])
    row_id_mismatch = row_id_meta.ne(row_id_sid)
    if bool(row_id_mismatch.any()):
        examples = merged.loc[row_id_mismatch, ["poi_id", "row_id_meta", "row_id_sid"]].head(5).to_dict("records")
        raise ValueError(f"row_id/poi_id mapping mismatch between SID and embedding meta, examples={examples}")

    for source in ["meta", "sid"]:
        merged[f"lat_{source}"] = pd.to_numeric(merged[f"lat_{source}"], errors="coerce")
        merged[f"lon_{source}"] = pd.to_numeric(merged[f"lon_{source}"], errors="coerce")

    latlon_same = np.isclose(merged["lat_meta"], merged["lat_sid"], rtol=0.0, atol=1e-8, equal_nan=True) & np.isclose(
        merged["lon_meta"], merged["lon_sid"], rtol=0.0, atol=1e-8, equal_nan=True
    )
    if not bool(np.all(latlon_same)):
        examples = merged.loc[~latlon_same, ["poi_id", "lat_meta", "lon_meta", "lat_sid", "lon_sid"]].head(5).to_dict("records")
        raise ValueError(f"lat/lon mismatch between SID and embedding meta, examples={examples}")

    def choose_text(meta_col: str, sid_col: str) -> pd.Series:
        meta_text = series_as_text(merged[meta_col])
        sid_text = series_as_text(merged[sid_col])
        return meta_text.where(meta_text.ne(""), sid_text)

    aligned = pd.DataFrame(
        {
            "row_id": merged["row_id_meta"],
            "poi_id": merged["poi_id"],
            "name": choose_text("name_meta", "name_sid"),
            "address": choose_text("address_meta", "address_sid"),
            "city": choose_text("city_meta", "city_sid"),
            "lat": merged["lat_meta"],
            "lon": merged["lon_meta"],
        }
    )
    diagnostics = {
        "input_sid_initial_row_order_aligned_with_embedding_meta": initial_row_order_poi_aligned,
        "input_sid_initial_row_id_aligned_with_embedding_meta": initial_row_id_aligned,
    }
    return aligned, diagnostics


def validate_latlon_or_raise(df: pd.DataFrame) -> tuple[int, int, int]:
    lat_missing_count = int(df["lat"].isna().sum())
    lon_missing_count = int(df["lon"].isna().sum())
    invalid_mask = invalid_latlon_mask(df["lat"].to_numpy(dtype=np.float64), df["lon"].to_numpy(dtype=np.float64))
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        examples = df.loc[invalid_mask, ["row_id", "poi_id", "name", "lat", "lon"]].head(10).to_dict("records")
        raise ValueError(
            f"Invalid lat/lon rows found after previous cleaning: count={invalid_count}, "
            f"lat_missing={lat_missing_count}, lon_missing={lon_missing_count}, examples={examples}"
        )
    return lat_missing_count, lon_missing_count, invalid_count


def fit_kmeans_anchors(coords_norm: np.ndarray, n_anchors: int, random_state: int) -> np.ndarray:
    if n_anchors <= 0:
        raise ValueError(f"n_anchors must be positive, got {n_anchors}")
    if coords_norm.shape[0] < n_anchors:
        raise ValueError(f"n_anchors ({n_anchors}) cannot exceed POI count ({coords_norm.shape[0]})")
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise SystemExit("scikit-learn is required. Please install scikit-learn: pip install scikit-learn") from exc

    kmeans = KMeans(n_clusters=n_anchors, random_state=random_state, n_init=10)
    kmeans.fit(coords_norm)
    centers = kmeans.cluster_centers_.astype(np.float64)
    return np.column_stack([centers[:, 0] * 90.0, centers[:, 1] * 180.0])


def top_counts_text(series: pd.Series, limit: int = 10) -> str:
    counts = series.value_counts().head(limit)
    if counts.empty:
        return "- none"
    return "\n".join(f"- {value}: {int(count)}" for value, count in counts.items())


def build_geohash6_distribution(geo_meta: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for geohash6, count in geo_meta["geohash6"].value_counts().head(limit).items():
        subset = geo_meta.loc[geo_meta["geohash6"].eq(geohash6)]
        rows.append(
            {
                "geohash6": geohash6,
                "count": int(count),
                "example_names": joined_examples(subset["name"], limit=5),
                "example_cities": joined_examples(subset["city"], limit=5),
            }
        )
    return pd.DataFrame(rows, columns=["geohash6", "count", "example_names", "example_cities"])


def build_anchor_assignment_distribution(
    geo_meta: pd.DataFrame,
    anchor_assignment: np.ndarray,
    anchors: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(geo_meta)
    for anchor_id in range(anchors.shape[0]):
        mask = anchor_assignment == anchor_id
        subset = geo_meta.loc[mask]
        count = int(mask.sum())
        rows.append(
            {
                "anchor_id": anchor_id,
                "count": count,
                "ratio": float(count / total) if total else 0.0,
                "anchor_lat": float(anchors[anchor_id, 0]),
                "anchor_lon": float(anchors[anchor_id, 1]),
                "example_names": joined_examples(subset["name"], limit=5),
                "example_cities": joined_examples(subset["city"], limit=5),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["anchor_id", "count", "ratio", "anchor_lat", "anchor_lon", "example_names", "example_cities"],
    )


def sample_group(source: pd.DataFrame, n: int, group: str, random_state: int, random_sample: bool = True) -> pd.DataFrame:
    if source.empty or n <= 0:
        return pd.DataFrame(columns=MANUAL_SAMPLE_COLUMNS)
    if random_sample and len(source) > n:
        sampled = source.sample(n=n, random_state=random_state)
    else:
        sampled = source.head(n)
    sampled = sampled.copy()
    sampled.insert(0, "sample_group", group)
    return sampled.reindex(columns=MANUAL_SAMPLE_COLUMNS)


def boundary_source(geo_meta: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    ranks = pd.DataFrame(
        {
            "lat_min_rank": geo_meta["lat"].rank(method="first", ascending=True),
            "lat_max_rank": geo_meta["lat"].rank(method="first", ascending=False),
            "lon_min_rank": geo_meta["lon"].rank(method="first", ascending=True),
            "lon_max_rank": geo_meta["lon"].rank(method="first", ascending=False),
        },
        index=geo_meta.index,
    )
    score = ranks.min(axis=1)
    return geo_meta.assign(_boundary_score=score).sort_values("_boundary_score").drop(columns=["_boundary_score"]).head(limit)


def build_manual_check_samples(
    geo_meta: pd.DataFrame,
    qrels_path: Path,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, int], bool]:
    samples = [
        sample_group(geo_meta, 100, "random", random_state, random_sample=True),
        sample_group(
            geo_meta.loc[geo_meta["geohash5"].isin(geo_meta["geohash5"].value_counts().head(10).index)],
            100,
            "high_frequency_geohash5",
            random_state + 1,
            random_sample=True,
        ),
        sample_group(boundary_source(geo_meta, 50), 50, "coordinate_boundary", random_state + 2, random_sample=False),
    ]

    qrels_used = False
    if qrels_path.exists():
        qrels = pd.read_csv(qrels_path, dtype=str)
        if "poi_id" in qrels.columns:
            qrel_ids = set(series_as_text(qrels["poi_id"]))
            qrel_source = geo_meta.loc[geo_meta["poi_id"].isin(qrel_ids)]
            samples.append(sample_group(qrel_source, 50, "qrels_poi", random_state + 3, random_sample=True))
            qrels_used = True

    manual = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=MANUAL_SAMPLE_COLUMNS)
    group_counts = manual["sample_group"].value_counts().to_dict() if not manual.empty else {}
    return manual, {str(k): int(v) for k, v in group_counts.items()}, qrels_used


def stats_to_jsonable(value: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=np.float64).tolist()]


def write_feature_stats(
    path: Path,
    names: list[str],
    raw_features: np.ndarray,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    raw_feature_std: np.ndarray,
    n_anchors: int,
) -> dict[str, Any]:
    stats = {
        "feature_names": names,
        "feature_mean": stats_to_jsonable(feature_mean),
        "feature_std": stats_to_jsonable(feature_std),
        "raw_feature_std": stats_to_jsonable(raw_feature_std),
        "raw_feature_min": stats_to_jsonable(np.min(raw_features, axis=0)),
        "raw_feature_max": stats_to_jsonable(np.max(raw_features, axis=0)),
        "standardized_feature_mean": stats_to_jsonable(np.mean(features.astype(np.float64), axis=0)),
        "standardized_feature_std": stats_to_jsonable(np.std(features.astype(np.float64), axis=0)),
        "n_rows": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "n_anchors": int(n_anchors),
    }
    ensure_parent_dir(path)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def validate_outputs(
    geo_meta: pd.DataFrame,
    embedding_meta: pd.DataFrame,
    features: np.ndarray,
    expected_dim: int,
    raw_feature_std: np.ndarray,
) -> None:
    if features.shape[0] != len(embedding_meta):
        raise ValueError(f"Geo feature rows ({features.shape[0]}) != embedding meta rows ({len(embedding_meta)})")
    if features.shape[0] != len(geo_meta):
        raise ValueError(f"Geo feature rows ({features.shape[0]}) != geo meta rows ({len(geo_meta)})")
    if features.shape[1] != expected_dim:
        raise ValueError(f"Geo feature dim ({features.shape[1]}) != expected dim ({expected_dim})")
    if not geo_meta["poi_id"].is_unique:
        raise ValueError("poi_id must be unique in geo meta")
    if geo_meta["poi_id"].tolist() != series_as_text(embedding_meta["poi_id"]).tolist():
        raise ValueError("poi_geo_meta poi_id order is not aligned with embedding meta")
    if series_as_text(geo_meta["row_id"]).tolist() != series_as_text(embedding_meta["row_id"]).tolist():
        raise ValueError("poi_geo_meta row_id order is not aligned with embedding meta")
    for precision in [5, 6, 7]:
        column = f"geohash{precision}"
        if bool(geo_meta[column].fillna("").astype(str).str.strip().eq("").any()):
            raise ValueError(f"{column} contains empty values")
    if np.isnan(features).any():
        raise ValueError("Geo features contain NaN")
    if np.isinf(features).any():
        raise ValueError("Geo features contain Inf")
    means = features.astype(np.float64).mean(axis=0)
    stds = features.astype(np.float64).std(axis=0)
    variable_mask = raw_feature_std >= 1e-12
    if float(np.max(np.abs(means))) > 1e-3:
        raise ValueError(f"Standardized geo feature mean is not close to 0, max_abs={float(np.max(np.abs(means))):.8f}")
    if bool(variable_mask.any()) and not np.all((stds[variable_mask] > 0.999) & (stds[variable_mask] < 1.001)):
        raise ValueError(
            "Standardized geo feature std is not close to 1, "
            f"min={float(stds[variable_mask].min()):.8f}, max={float(stds[variable_mask].max()):.8f}"
        )


def markdown_anchor_table(anchor_distribution: pd.DataFrame) -> str:
    lines = ["| anchor_id | count | ratio | anchor_lat | anchor_lon |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, row in anchor_distribution.iterrows():
        lines.append(
            f"| {int(row['anchor_id'])} | {int(row['count'])} | {float(row['ratio']):.4f} | "
            f"{float(row['anchor_lat']):.6f} | {float(row['anchor_lon']):.6f} |"
        )
    return "\n".join(lines)


def markdown_manual_samples(geo_meta: pd.DataFrame, n: int = 20) -> str:
    rows = geo_meta.reindex(columns=REPORT_SAMPLE_COLUMNS).head(n)
    lines = ["| row_id | poi_id | name | city | address | lat | lon | geohash6 |", "| --- | --- | --- | --- | --- | ---: | ---: | --- |"]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['row_id']} | {row['poi_id']} | {row['name']} | {row['city']} | {row['address']} | "
            f"{float(row['lat']):.6f} | {float(row['lon']):.6f} | {row['geohash6']} |"
        )
    return "\n".join(lines)


def build_report(
    sid_rows: int,
    embedding_rows: int,
    geo_meta: pd.DataFrame,
    features: np.ndarray,
    lat_missing_count: int,
    lon_missing_count: int,
    invalid_latlon_count: int,
    row_order_aligned: bool,
    row_id_aligned: bool,
    diagnostics: dict[str, Any],
    anchors: np.ndarray,
    anchor_distribution: pd.DataFrame,
    min_anchor_distance: np.ndarray,
    stats: dict[str, Any],
    manual_group_counts: dict[str, int],
    qrels_used: bool,
) -> str:
    total = len(geo_meta)
    lat = geo_meta["lat"].astype(float)
    lon = geo_meta["lon"].astype(float)
    standardized_means = np.asarray(stats["standardized_feature_mean"], dtype=np.float64)
    standardized_stds = np.asarray(stats["standardized_feature_std"], dtype=np.float64)
    has_nan = bool(np.isnan(features).any())
    has_inf = bool(np.isinf(features).any())

    geohash_lines = [
        f"- geohash5_non_empty_count / rate: {int(geo_meta['geohash5'].ne('').sum())} / {format_rate(int(geo_meta['geohash5'].ne('').sum()), total)}",
        f"- geohash6_non_empty_count / rate: {int(geo_meta['geohash6'].ne('').sum())} / {format_rate(int(geo_meta['geohash6'].ne('').sum()), total)}",
        f"- geohash7_non_empty_count / rate: {int(geo_meta['geohash7'].ne('').sum())} / {format_rate(int(geo_meta['geohash7'].ne('').sum()), total)}",
        f"- unique_geohash5: {int(geo_meta['geohash5'].nunique())}",
        f"- unique_geohash6: {int(geo_meta['geohash6'].nunique())}",
        f"- unique_geohash7: {int(geo_meta['geohash7'].nunique())}",
        "- top_geohash5:",
        top_counts_text(geo_meta["geohash5"], 10),
        "- top_geohash6:",
        top_counts_text(geo_meta["geohash6"], 10),
    ]

    return "\n".join(
        [
            "# POI Geo Feature Quality Report",
            "",
            "## Basic Statistics",
            f"- input_sid_rows: {sid_rows}",
            f"- input_embedding_meta_rows: {embedding_rows}",
            f"- output_geo_rows: {len(geo_meta)}",
            f"- output_geo_features_shape: {list(features.shape)}",
            f"- unique_poi_id: {int(geo_meta['poi_id'].nunique())}",
            f"- row_order_aligned_with_embedding_meta: {row_order_aligned}",
            f"- row_id_aligned_with_embedding_meta: {row_id_aligned}",
            f"- input_sid_initial_row_order_aligned_with_embedding_meta: {diagnostics['input_sid_initial_row_order_aligned_with_embedding_meta']}",
            f"- input_sid_initial_row_id_aligned_with_embedding_meta: {diagnostics['input_sid_initial_row_id_aligned_with_embedding_meta']}",
            f"- lat_missing_count: {lat_missing_count}",
            f"- lon_missing_count: {lon_missing_count}",
            f"- invalid_latlon_count: {invalid_latlon_count}",
            "",
            "## Coordinate Range",
            f"- lat_min: {float(lat.min()):.6f}",
            f"- lat_max: {float(lat.max()):.6f}",
            f"- lat_mean: {float(lat.mean()):.6f}",
            f"- lat_p50: {float(lat.quantile(0.50)):.6f}",
            f"- lon_min: {float(lon.min()):.6f}",
            f"- lon_max: {float(lon.max()):.6f}",
            f"- lon_mean: {float(lon.mean()):.6f}",
            f"- lon_p50: {float(lon.quantile(0.50)):.6f}",
            "",
            "## Geohash Coverage",
            *geohash_lines,
            "",
            "## Anchor Summary",
            f"- n_anchors: {anchors.shape[0]}",
            "- anchor coordinates table:",
            markdown_anchor_table(anchor_distribution),
            "- anchor assignment distribution:",
            "\n".join(
                f"  - anchor_{int(row['anchor_id'])}: {int(row['count'])} / {float(row['ratio']):.4f}"
                for _, row in anchor_distribution.iterrows()
            ),
            f"- min_anchor_distance_km_mean: {float(np.mean(min_anchor_distance)):.6f}",
            f"- min_anchor_distance_km_p50: {float(np.quantile(min_anchor_distance, 0.50)):.6f}",
            f"- min_anchor_distance_km_p95: {float(np.quantile(min_anchor_distance, 0.95)):.6f}",
            "",
            "## Feature Numeric Checks",
            f"- feature_dim: {features.shape[1]}",
            f"- has_nan: {has_nan}",
            f"- has_inf: {has_inf}",
            f"- standardized_mean_abs_max: {float(np.max(np.abs(standardized_means))):.8f}",
            f"- standardized_std_min: {float(np.min(standardized_stds)):.8f}",
            f"- standardized_std_max: {float(np.max(standardized_stds)):.8f}",
            "",
            "## Extra CSV Reports",
            "- poi_geo_manual_check_samples.csv",
            "- geohash6_distribution.csv",
            "- anchor_assignment_distribution.csv",
            f"- qrels_sample_used: {qrels_used}",
            f"- manual_sample_group_counts: {manual_group_counts}",
            "",
            "## Manual Samples",
            markdown_manual_samples(geo_meta, 20),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    sid_path = Path(args.sid_input)
    embedding_meta_path = Path(args.embedding_meta)
    out_features_path = Path(args.out_features)
    out_meta_path = Path(args.out_meta)
    stats_path = Path(args.stats_out)
    anchors_path = Path(args.anchors_out)
    report_path = Path(args.report)
    report_dir = Path(args.report_dir) if args.report_dir else report_path.parent
    qrels_path = Path(args.qrels)

    sid = read_parquet(sid_path, "poi_sid_train.parquet")
    embedding_meta = read_parquet(embedding_meta_path, "poi_embedding_meta.parquet")
    aligned, diagnostics = aligned_sid_to_embedding_meta(sid, embedding_meta)
    lat_missing_count, lon_missing_count, invalid_latlon_count = validate_latlon_or_raise(aligned)

    lat = aligned["lat"].to_numpy(dtype=np.float64)
    lon = aligned["lon"].to_numpy(dtype=np.float64)
    geo_meta = aligned.copy()
    geo_meta["geohash5"] = encode_geohashes(lat, lon, precision=5)
    geo_meta["geohash6"] = encode_geohashes(lat, lon, precision=6)
    geo_meta["geohash7"] = encode_geohashes(lat, lon, precision=7)
    geo_meta = geo_meta.reindex(columns=GEO_META_COLUMNS)

    coords_norm = np.column_stack([lat / 90.0, lon / 180.0])
    anchors = fit_kmeans_anchors(coords_norm, args.n_anchors, args.random_state)
    raw_features, names, distance_matrix = build_raw_geo_features(lat, lon, anchors)
    features, feature_mean, feature_std, raw_feature_std = standardize_features(raw_features)
    min_anchor_distance = distance_matrix.min(axis=1)
    anchor_assignment = distance_matrix.argmin(axis=1)

    expected_dim = 6 + args.n_anchors * 3
    validate_outputs(geo_meta, embedding_meta, features, expected_dim, raw_feature_std)

    anchor_distribution = build_anchor_assignment_distribution(geo_meta, anchor_assignment, anchors)
    geohash6_distribution = build_geohash6_distribution(geo_meta, limit=500)
    manual_samples, manual_group_counts, qrels_used = build_manual_check_samples(geo_meta, qrels_path, args.random_state)

    ensure_parent_dir(out_features_path)
    np.save(out_features_path, features.astype(np.float32, copy=False))
    write_parquet(geo_meta, out_meta_path)
    ensure_parent_dir(anchors_path)
    np.save(anchors_path, anchors.astype(np.float32, copy=False))
    stats = write_feature_stats(
        stats_path,
        names,
        raw_features,
        features,
        feature_mean,
        feature_std,
        raw_feature_std,
        args.n_anchors,
    )

    ensure_dir(report_dir)
    write_csv(manual_samples, report_dir / "poi_geo_manual_check_samples.csv")
    write_csv(geohash6_distribution, report_dir / "geohash6_distribution.csv")
    write_csv(anchor_distribution, report_dir / "anchor_assignment_distribution.csv")

    row_order_aligned = geo_meta["poi_id"].tolist() == series_as_text(embedding_meta["poi_id"]).tolist()
    row_id_aligned = series_as_text(geo_meta["row_id"]).tolist() == series_as_text(embedding_meta["row_id"]).tolist()
    report = build_report(
        sid_rows=len(sid),
        embedding_rows=len(embedding_meta),
        geo_meta=geo_meta,
        features=features,
        lat_missing_count=lat_missing_count,
        lon_missing_count=lon_missing_count,
        invalid_latlon_count=invalid_latlon_count,
        row_order_aligned=row_order_aligned,
        row_id_aligned=row_id_aligned,
        diagnostics=diagnostics,
        anchors=anchors,
        anchor_distribution=anchor_distribution,
        min_anchor_distance=min_anchor_distance,
        stats=stats,
        manual_group_counts=manual_group_counts,
        qrels_used=qrels_used,
    )
    ensure_parent_dir(report_path)
    report_path.write_text(report, encoding="utf-8")

    print(f"input_poi_count: {len(embedding_meta)}")
    print(f"geo_feature_shape: {list(features.shape)}")
    print(f"geohash6_unique_count: {int(geo_meta['geohash6'].nunique())}")
    print(f"aligned_with_embedding_meta: {row_order_aligned and row_id_aligned}")
    print(f"has_nan: {bool(np.isnan(features).any())}")
    print(f"has_inf: {bool(np.isinf(features).any())}")
    print(f"report_output: {report_path}")
    print("next_step: train RQ-VAE semantic and geo_fused versions")


if __name__ == "__main__":
    main()
