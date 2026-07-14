#!/usr/bin/env python3
"""Build MobilityBench POI SID training data with raw JSON-style POI text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import ensure_columns, ensure_dir, ensure_parent_dir, format_rate, joined_examples  # noqa: E402
from text_normalize import clean_text, is_empty_text  # noqa: E402


BASE_COLUMNS = [
    "poi_id",
    "name",
    "address",
    "city",
    "district",
    "business_area",
    "lat",
    "lon",
    "category_l1",
    "category_l2",
    "brand",
    "tags",
    "rating",
    "price",
    "open_hours",
    "source",
    "raw_id",
]

JSON_TEXT_COLUMNS = [
    "name",
    "address",
    "city",
    "district",
    "business_area",
    "category_l1",
    "category_l2",
    "brand",
    "tags",
    "rating",
    "price",
    "open_hours",
]

OUTPUT_COLUMNS = [
    "row_id",
    *BASE_COLUMNS,
    "parse_warning",
    "poi_json",
    "poi_text",
]

PREVIEW_COLUMNS = [
    "row_id",
    "poi_id",
    "name",
    "address",
    "city",
    "lat",
    "lon",
    "category_l1",
    "category_l2",
    "brand",
    "tags",
    "parse_warning",
    "poi_json",
    "poi_text",
]

MANUAL_SAMPLE_COLUMNS = [
    "sample_group",
    "row_id",
    "poi_id",
    "name",
    "address",
    "city",
    "lat",
    "lon",
    "category_l1",
    "category_l2",
    "brand",
    "tags",
    "parse_warning",
    "poi_json",
    "poi_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pois", default="data/processed/mobilitybench/pois.csv", help="Input POI CSV.")
    parser.add_argument("--out", default="data/sid/poi_sid_train.parquet", help="Output parquet path.")
    parser.add_argument(
        "--preview-out",
        default="data/sid/poi_sid_train_preview.csv",
        help="Output preview CSV path.",
    )
    parser.add_argument("--report-dir", default="outputs/reports/poi_sid", help="Directory for generated coverage reports.")
    parser.add_argument("--type-rules", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--brand-rules", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_and_clean_pois(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path, dtype=object)
    stats = {"raw_rows": len(raw)}
    df = ensure_columns(raw.copy(), BASE_COLUMNS)

    for column in BASE_COLUMNS:
        df[column] = df[column].map(clean_text)

    stats["raw_missing_poi_id_rows"] = int(df["poi_id"].map(is_empty_text).sum())
    stats["raw_missing_name_rows"] = int(df["name"].map(is_empty_text).sum())
    stats["raw_missing_address_rows"] = int(df["address"].map(is_empty_text).sum())
    stats["raw_missing_lat_rows"] = int(df["lat"].map(is_empty_text).sum())
    stats["raw_missing_lon_rows"] = int(df["lon"].map(is_empty_text).sum())

    empty_poi_id = df["poi_id"].map(is_empty_text)
    stats["removed_empty_poi_id_rows"] = int(empty_poi_id.sum())
    df = df.loc[~empty_poi_id].copy()

    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    valid_latlon = lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)
    stats["removed_invalid_latlon_rows"] = int((~valid_latlon).sum())
    df = df.loc[valid_latlon].copy()
    df["lat"] = lat.loc[valid_latlon].astype(float).values
    df["lon"] = lon.loc[valid_latlon].astype(float).values

    duplicate_poi_id = df.duplicated("poi_id", keep="first")
    stats["removed_duplicate_poi_id_rows"] = int(duplicate_poi_id.sum())
    df = df.loc[~duplicate_poi_id].copy()
    df = df.reset_index(drop=True)
    df["row_id"] = range(len(df))
    return df, stats


def build_parse_warning(row: dict[str, Any]) -> str:
    warnings: list[str] = []
    if is_empty_text(row.get("name")):
        warnings.append("missing_name")
    if is_empty_text(row.get("address")):
        warnings.append("missing_address")
    if is_empty_text(row.get("city")):
        warnings.append("missing_city")
    if is_empty_text(row.get("category_l1")):
        warnings.append("missing_category_l1")
    if is_empty_text(row.get("tags")):
        warnings.append("missing_tags")
    return ";".join(warnings)


def build_poi_json_object(row: dict[str, Any]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for column in JSON_TEXT_COLUMNS:
        value = clean_text(row.get(column))
        if value:
            obj[column] = value

    obj["location"] = {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
    }
    return obj


def compact_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_output_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        poi_obj = build_poi_json_object(row)
        poi_json = compact_json(poi_obj)
        row["parse_warning"] = build_parse_warning(row)
        row["poi_json"] = poi_json
        row["poi_text"] = poi_json
        records.append(row)
    return pd.DataFrame(records).reindex(columns=OUTPUT_COLUMNS)


def warning_counter(df: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in df["parse_warning"].fillna("").astype(str):
        for warning in value.split(";"):
            if warning:
                counter[warning] += 1
    return counter


def warning_counts_text(counter: Counter[str], limit: int = 30) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- {name}: {count}" for name, count in counter.most_common(limit))


def json_field_coverage(df: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    total = len(df)
    for column in JSON_TEXT_COLUMNS:
        count = int(df[column].map(lambda x: not is_empty_text(x)).sum())
        rows.append(f"- {column}_non_empty_count / rate: {count} / {format_rate(count, total)}")
    rows.append(f"- location_non_empty_count / rate: {total} / {format_rate(total, total)}")
    return rows


def top_category_table(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["category_l1"].map(lambda x: not is_empty_text(x))].copy()
    if subset.empty:
        return pd.DataFrame(columns=["category_l1", "count", "example_names"])
    grouped = (
        subset.groupby("category_l1", dropna=False)
        .agg(count=("category_l1", "size"), example_names=("name", joined_examples))
        .reset_index()
        .sort_values(["count", "category_l1"], ascending=[False, True])
        .head(200)
    )
    return grouped[["category_l1", "count", "example_names"]]


def sample_group(df: pd.DataFrame, mask: pd.Series, group: str, n: int, seed: int) -> tuple[pd.DataFrame, int]:
    subset = df.loc[mask].copy()
    take = min(n, len(subset))
    if take:
        subset = subset.sample(n=take, random_state=seed)
    else:
        subset = subset.head(0)
    subset.insert(0, "sample_group", group)
    return subset.reindex(columns=MANUAL_SAMPLE_COLUMNS), take


def write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    ensure_parent_dir(path)
    df.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def write_reports(df: pd.DataFrame, stats: dict[str, int], report_dir: Path) -> dict[str, int]:
    ensure_dir(report_dir)
    top_category_table(df).to_csv(report_dir / "top_raw_category_l1.csv", index=False, encoding="utf-8-sig")

    complete_basic = (
        df["name"].map(lambda x: not is_empty_text(x))
        & df["address"].map(lambda x: not is_empty_text(x))
        & df["category_l1"].map(lambda x: not is_empty_text(x))
        & df["city"].map(lambda x: not is_empty_text(x))
    )
    manual_specs = [
        ("missing_category_l1", df["parse_warning"].str.contains("missing_category_l1", na=False), 80, 21),
        ("missing_city", df["parse_warning"].str.contains("missing_city", na=False), 80, 22),
        ("complete_basic_fields", complete_basic, 80, 23),
        ("random", pd.Series([True] * len(df), index=df.index), 60, 24),
    ]
    manual_parts: list[pd.DataFrame] = []
    manual_counts: dict[str, int] = {}
    for group, mask, n, seed in manual_specs:
        sampled, count = sample_group(df, mask, group, n, seed)
        manual_parts.append(sampled)
        manual_counts[group] = count
    manual_df = pd.concat(manual_parts, ignore_index=True) if manual_parts else pd.DataFrame(columns=MANUAL_SAMPLE_COLUMNS)
    write_csv(manual_df, report_dir / "poi_sid_train_manual_check_samples.csv", MANUAL_SAMPLE_COLUMNS)

    output_rows = len(df)
    warning_counts = warning_counter(df)
    empty_poi_json = int(df["poi_json"].map(is_empty_text).sum())
    empty_poi_text = int(df["poi_text"].map(is_empty_text).sum())

    report_lines = [
        "# POI JSON Coverage Report",
        "",
        "## Basic Statistics",
        f"- raw_rows: {stats['raw_rows']}",
        f"- output_rows: {output_rows}",
        f"- removed_empty_poi_id_rows: {stats['removed_empty_poi_id_rows']}",
        f"- removed_invalid_latlon_rows: {stats['removed_invalid_latlon_rows']}",
        f"- removed_duplicate_poi_id_rows: {stats['removed_duplicate_poi_id_rows']}",
        f"- unique_poi_id: {int(df['poi_id'].nunique()) if output_rows else 0}",
        f"- empty_poi_json_count: {empty_poi_json}",
        f"- empty_poi_text_count: {empty_poi_text}",
        f"- raw_missing_poi_id_rows: {stats['raw_missing_poi_id_rows']}",
        f"- raw_missing_name_rows: {stats['raw_missing_name_rows']}",
        f"- raw_missing_address_rows: {stats['raw_missing_address_rows']}",
        f"- raw_missing_lat_rows: {stats['raw_missing_lat_rows']}",
        f"- raw_missing_lon_rows: {stats['raw_missing_lon_rows']}",
        "",
        "## Missing Fields",
    ]
    for field in ["name", "address", "city", "category_l1", "tags"]:
        count = int(df[field].map(is_empty_text).sum())
        report_lines.append(f"- {field}_missing_count / rate: {count} / {format_rate(count, output_rows)}")

    report_lines.extend(
        [
            "",
            "## JSON Field Coverage",
            *json_field_coverage(df),
            "",
            "## Parse Warnings",
            warning_counts_text(warning_counts, 30),
            "",
            "## Manual Check Samples",
        ]
    )
    requested_counts = {
        "missing_category_l1": 80,
        "missing_city": 80,
        "complete_basic_fields": 80,
        "random": 60,
    }
    for group, requested in requested_counts.items():
        report_lines.append(f"- {group}: {manual_counts.get(group, 0)} / requested {requested}")

    report_path = report_dir / "poi_parse_coverage_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "output_rows": output_rows,
        "empty_poi_json": empty_poi_json,
        "empty_poi_text": empty_poi_text,
    }


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


def main() -> None:
    args = parse_args()
    pois_path = Path(args.pois)
    out_path = Path(args.out)
    preview_path = Path(args.preview_out)
    report_dir = Path(args.report_dir)

    df, stats = read_and_clean_pois(pois_path)
    if stats["removed_empty_poi_id_rows"]:
        print(f"[WARN] removed empty poi_id rows: {stats['removed_empty_poi_id_rows']}")
    if stats["removed_invalid_latlon_rows"]:
        print(f"[WARN] removed invalid lat/lon rows: {stats['removed_invalid_latlon_rows']}")
    if stats["removed_duplicate_poi_id_rows"]:
        print(f"[WARN] removed duplicate poi_id rows: {stats['removed_duplicate_poi_id_rows']}")
    if stats["raw_missing_name_rows"]:
        print(f"[WARN] raw missing name rows: {stats['raw_missing_name_rows']}")
    if stats["raw_missing_address_rows"]:
        print(f"[WARN] raw missing address rows: {stats['raw_missing_address_rows']}")
    if stats["raw_missing_lat_rows"] or stats["raw_missing_lon_rows"]:
        print(f"[WARN] raw missing lat/lon rows: {stats['raw_missing_lat_rows']} / {stats['raw_missing_lon_rows']}")

    output = build_output_table(df)
    write_parquet(output, out_path)
    write_csv(output.head(5000), preview_path, PREVIEW_COLUMNS)
    report_stats = write_reports(output, stats, report_dir)

    print(f"raw_rows: {stats['raw_rows']}")
    print(f"output_rows: {report_stats['output_rows']}")
    print(f"removed_empty_poi_id_rows: {stats['removed_empty_poi_id_rows']}")
    print(f"removed_invalid_latlon_rows: {stats['removed_invalid_latlon_rows']}")
    print(f"removed_duplicate_poi_id_rows: {stats['removed_duplicate_poi_id_rows']}")
    print(f"empty_poi_json_count: {report_stats['empty_poi_json']}")
    print(f"empty_poi_text_count: {report_stats['empty_poi_text']}")
    print(f"report: {report_dir / 'poi_parse_coverage_report.md'}")


if __name__ == "__main__":
    main()
