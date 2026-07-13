#!/usr/bin/env python3
"""Prepare LLM4POI and Yelp MVP data artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poi_genret.schema import (  # noqa: E402
    INTERACTION_COLUMNS,
    POI_COLUMNS,
    QREL_COLUMNS,
    QUERY_COLUMNS,
    write_csv,
)
from poi_genret.process_utils import stable_hash  # noqa: E402


CITY_LABELS = {"nyc": "NYC", "tky": "Tokyo", "ca": "California"}


def log(message: str) -> None:
    print(f"[prepare_mvp_data] {message}")


def compact_json(value: Any) -> str:
    if value in (None, "", {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def split_categories(categories: Any) -> list[str]:
    if not categories:
        return []
    return [part.strip() for part in str(categories).split(",") if part.strip()]


def make_query_id(source: str, poi_id: str, query_type: str, query_text: str) -> str:
    return f"{source}_q_{stable_hash(poi_id + '|' + query_type + '|' + query_text, length=16)}"


def find_llm4poi_files(raw_dir: Path, legacy_nyc: Path | None = None) -> list[Path]:
    files = sorted((raw_dir / "LLM4POI").glob("*/preprocessed/train_sample.csv"))
    if files:
        return files
    if legacy_nyc and legacy_nyc.exists():
        return [legacy_nyc]
    log(f"[WARN] no LLM4POI train_sample.csv found under {raw_dir / 'LLM4POI'}")
    return []


def read_llm4poi_file(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city_code = path.parts[-3].lower() if len(path.parts) >= 3 else "unknown"
    city = CITY_LABELS.get(city_code, city_code.upper())
    df = pd.read_csv(path, nrows=max_rows)
    if df.empty:
        return empty_outputs()

    required = {"UserId", "PoiId", "Latitude", "Longitude", "PoiCategoryName", "UTCTimeOffsetEpoch"}
    missing = sorted(required.difference(df.columns))
    if missing:
        log(f"[WARN] skip {path}: missing columns {missing}")
        return empty_outputs()

    df["poi_id_out"] = f"llm4poi:{city_code}:" + df["PoiId"].astype(str)
    df["source_out"] = "llm4poi"
    df["timestamp_out"] = pd.to_datetime(df["UTCTimeOffsetEpoch"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    category_id_col = "PoiCategoryId" if "PoiCategoryId" in df.columns else "PoiCategoryName"

    poi_df = (
        df.sort_values(["PoiId", "UTCTimeOffsetEpoch"])
        .groupby("PoiId", as_index=False)
        .agg(
            poi_id=("poi_id_out", "first"),
            lat=("Latitude", "median"),
            lon=("Longitude", "median"),
            category_l1=("PoiCategoryName", "first"),
            category_l2=(category_id_col, "first"),
        )
    )
    poi_df["name"] = ""
    poi_df["address"] = ""
    poi_df["city"] = city
    poi_df["district"] = ""
    poi_df["business_area"] = ""
    poi_df["brand"] = ""
    poi_df["tags"] = poi_df["category_l1"]
    poi_df["rating"] = ""
    poi_df["price"] = ""
    poi_df["open_hours"] = ""
    poi_df["source"] = "llm4poi"
    poi_df = poi_df[POI_COLUMNS]

    interactions = pd.DataFrame(
        {
            "user_id": f"llm4poi:{city_code}:" + df["UserId"].astype(str),
            "poi_id": df["poi_id_out"],
            "timestamp": df["timestamp_out"],
            "lat": df["Latitude"],
            "lon": df["Longitude"],
            "action_type": "checkin",
            "source": "llm4poi",
        }
    )

    template_queries: list[dict[str, Any]] = []
    template_qrels: list[dict[str, Any]] = []
    for row in poi_df.fillna("").itertuples(index=False):
        category = str(row.category_l1)
        if not category:
            continue
        query_text = f"search for {category} in {city}"
        query_id = make_query_id("llm4poi", row.poi_id, "template_poi", query_text)
        template_queries.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "city": city,
                "user_lat": "",
                "user_lon": "",
                "timestamp": "",
                "query_type": "template_poi",
                "source": "llm4poi",
            }
        )
        template_qrels.append({"query_id": query_id, "poi_id": row.poi_id, "label": 1, "source": "llm4poi"})

    behavior_queries, behavior_qrels = build_llm4poi_behavior_queries(df, city, city_code)
    queries = pd.DataFrame(template_queries + behavior_queries, columns=QUERY_COLUMNS)
    qrels = pd.DataFrame(template_qrels + behavior_qrels, columns=QREL_COLUMNS)
    log(f"read {path}: rows={len(df)} pois={len(poi_df)} queries={len(queries)}")
    return poi_df, interactions[INTERACTION_COLUMNS], queries, qrels


def build_llm4poi_behavior_queries(df: pd.DataFrame, city: str, city_code: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_col = "trajectory_id" if "trajectory_id" in df.columns else "pseudo_session_trajectory_id"
    if group_col not in df.columns:
        return [], []

    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    ordered = df.sort_values([group_col, "UTCTimeOffsetEpoch"]).fillna("")
    for _, group in ordered.groupby(group_col, sort=False):
        if len(group) < 2:
            continue
        rows = list(group.itertuples(index=False))
        for prev, target in zip(rows[:-1], rows[1:]):
            prev_cat = str(getattr(prev, "PoiCategoryName", "POI"))
            target_poi_id = str(getattr(target, "poi_id_out"))
            timestamp = getattr(target, "timestamp_out", "")
            query_text = f"next POI after visiting {prev_cat} in {city}"
            raw = f"{city_code}|{getattr(target, group_col)}|{getattr(target, 'check_ins_id', '')}|{target_poi_id}"
            query_id = f"llm4poi_b_{stable_hash(raw, length=16)}"
            queries.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "city": city,
                    "user_lat": getattr(prev, "Latitude", ""),
                    "user_lon": getattr(prev, "Longitude", ""),
                    "timestamp": timestamp,
                    "query_type": "behavior_poi",
                    "source": "llm4poi",
                }
            )
            qrels.append({"query_id": query_id, "poi_id": target_poi_id, "label": 1, "source": "llm4poi"})
    return queries, qrels


def empty_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=POI_COLUMNS),
        pd.DataFrame(columns=INTERACTION_COLUMNS),
        pd.DataFrame(columns=QUERY_COLUMNS),
        pd.DataFrame(columns=QREL_COLUMNS),
    )


def read_llm4poi(raw_dir: Path, legacy_nyc: Path | None, max_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = [read_llm4poi_file(path, max_rows=max_rows) for path in find_llm4poi_files(raw_dir, legacy_nyc)]
    if not outputs:
        return empty_outputs()
    return tuple(pd.concat(parts, ignore_index=True) for parts in zip(*outputs))  # type: ignore[return-value]


def yelp_price(attributes: Any) -> str:
    if not isinstance(attributes, dict):
        return ""
    value = attributes.get("RestaurantsPriceRange2")
    return "" if value is None else str(value)


def yelp_business_to_poi(obj: dict[str, Any]) -> dict[str, Any]:
    categories = split_categories(obj.get("categories"))
    state = obj.get("state") or ""
    city = obj.get("city") or ""
    city_value = f"{city}, {state}".strip(", ")
    return {
        "poi_id": f"yelp:{obj.get('business_id', '')}",
        "name": obj.get("name") or "",
        "address": obj.get("address") or "",
        "city": city_value,
        "district": state,
        "business_area": obj.get("postal_code") or "",
        "lat": obj.get("latitude", ""),
        "lon": obj.get("longitude", ""),
        "category_l1": categories[0] if categories else "",
        "category_l2": categories[1] if len(categories) > 1 else "",
        "brand": "",
        "tags": "|".join(categories),
        "rating": obj.get("stars", ""),
        "price": yelp_price(obj.get("attributes")),
        "open_hours": compact_json(obj.get("hours")),
        "source": "yelp",
    }


def build_yelp_queries(poi: dict[str, Any], is_open: Any) -> list[dict[str, str]]:
    name = str(poi["name"]).strip()
    city = str(poi["city"]).strip()
    category = str(poi["category_l1"]).strip()
    rating = poi.get("rating", "")

    templates: list[tuple[str, str]] = []
    if name:
        templates.append(("poi_search", f"find {name}"))
    if category and city:
        templates.append(("template_poi", f"search for {category} in {city}"))
    if category:
        templates.append(("nearby_search", f"open {category} near me"))
    if category and city and rating not in ("", None):
        templates.append(("template_poi", f"high rated {category} in {city}"))
    if category and str(is_open) == "1":
        templates.append(("nearby_search", f"open {category} near me now"))

    rows = []
    for query_type, query_text in templates:
        rows.append(
            {
                "query_id": make_query_id("yelp", str(poi["poi_id"]), query_type, query_text),
                "query_text": query_text,
                "city": city,
                "user_lat": "",
                "user_lon": "",
                "timestamp": "",
                "query_type": query_type,
                "source": "yelp",
            }
        )
    return rows


def read_yelp_businesses(path: Path, max_businesses: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pois: list[dict[str, Any]] = []
    queries: list[dict[str, str]] = []
    qrels: list[dict[str, Any]] = []

    if not path.exists():
        log(f"[WARN] Yelp business file not found: {path}")
        return pd.DataFrame(columns=POI_COLUMNS), pd.DataFrame(columns=QUERY_COLUMNS), pd.DataFrame(columns=QREL_COLUMNS)

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_businesses is not None and idx >= max_businesses:
                break
            obj = json.loads(line)
            poi = yelp_business_to_poi(obj)
            if not poi["poi_id"] or poi["poi_id"] == "yelp:":
                continue
            pois.append(poi)
            for query in build_yelp_queries(poi, obj.get("is_open")):
                queries.append(query)
                qrels.append({"query_id": query["query_id"], "poi_id": poi["poi_id"], "label": 1, "source": "yelp"})

    log(f"read {path}: businesses={len(pois)} queries={len(queries)}")
    return (
        pd.DataFrame(pois, columns=POI_COLUMNS),
        pd.DataFrame(queries, columns=QUERY_COLUMNS),
        pd.DataFrame(qrels, columns=QREL_COLUMNS),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/data_report", help="Accepted for CLI consistency.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional per-file row limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=42, help="Accepted for CLI consistency.")
    parser.add_argument(
        "--llm4poi-nyc",
        type=Path,
        default=ROOT / "data/raw/LLM4POI/nyc/preprocessed/train_sample.csv",
        help="Legacy option. Used only if --raw-dir has no LLM4POI train_sample.csv files.",
    )
    parser.add_argument(
        "--yelp-business",
        type=Path,
        default=ROOT / "data/raw/yelp/yelp_academic_dataset_business.json",
    )
    parser.add_argument(
        "--max-yelp-businesses",
        type=int,
        default=None,
        help="Legacy smoke-test limit. Overrides --max-rows for Yelp when set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    llm_pois, interactions, llm_queries, llm_qrels = read_llm4poi(args.raw_dir, args.llm4poi_nyc, args.max_rows)
    yelp_limit = args.max_yelp_businesses if args.max_yelp_businesses is not None else args.max_rows
    yelp_pois, yelp_queries, yelp_qrels = read_yelp_businesses(args.yelp_business, yelp_limit)

    pois = pd.concat([llm_pois, yelp_pois], ignore_index=True)
    queries = pd.concat([llm_queries, yelp_queries], ignore_index=True)
    qrels = pd.concat([llm_qrels, yelp_qrels], ignore_index=True)

    write_csv(pois, args.output_dir / "pois.csv", POI_COLUMNS)
    write_csv(interactions, args.output_dir / "interactions.csv", INTERACTION_COLUMNS)
    write_csv(queries, args.output_dir / "queries.csv", QUERY_COLUMNS)
    write_csv(qrels, args.output_dir / "qrels.csv", QREL_COLUMNS)

    summary = {
        "pois": len(pois),
        "interactions": len(interactions),
        "queries": len(queries),
        "qrels": len(qrels),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
