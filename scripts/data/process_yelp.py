#!/usr/bin/env python3
"""Process Yelp as an independent template-based POI retrieval dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from poi_genret.process_utils import (  # noqa: E402
    clean,
    compact_json,
    log,
    parse_ratio,
    split_keys_random,
    stable_hash,
    write_manifest,
    write_split_files,
)
from poi_genret.schema import (  # noqa: E402
    INTERACTION_COLUMNS,
    MOBILITY_CANDIDATE_COLUMNS,
    POI_COLUMNS,
    QREL_COLUMNS,
    QUERY_COLUMNS,
    write_csv,
)


def split_categories(categories: Any) -> list[str]:
    if not categories:
        return []
    return [part.strip() for part in str(categories).split(",") if part.strip()]


def get_price(attributes: Any) -> str:
    if isinstance(attributes, dict) and attributes.get("RestaurantsPriceRange2") is not None:
        return str(attributes["RestaurantsPriceRange2"])
    return ""


def business_to_poi(obj: dict[str, Any]) -> dict[str, Any]:
    categories = split_categories(obj.get("categories"))
    city = clean(obj.get("city"))
    state = clean(obj.get("state"))
    raw_id = clean(obj.get("business_id"))
    return {
        "poi_id": f"yelp:{raw_id}",
        "name": clean(obj.get("name")),
        "address": clean(obj.get("address")),
        "city": f"{city}, {state}".strip(", "),
        "district": state,
        "business_area": clean(obj.get("postal_code")),
        "lat": clean(obj.get("latitude")),
        "lon": clean(obj.get("longitude")),
        "category_l1": categories[0] if categories else "",
        "category_l2": categories[1] if len(categories) > 1 else "",
        "brand": "",
        "tags": "|".join(categories),
        "rating": clean(obj.get("stars")),
        "price": get_price(obj.get("attributes")),
        "open_hours": compact_json(obj.get("hours")),
        "source": "yelp",
        "raw_id": raw_id,
    }


def build_queries(poi: dict[str, Any], obj: dict[str, Any], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    name = poi["name"]
    city = poi["city"]
    category = poi["category_l1"]
    rating = poi["rating"]
    templates: list[str] = []
    if name:
        templates.append(f"find {name}")
    if category and city:
        templates.append(f"search for {category} in {city}")
        templates.append(f"{category} near {city}")
        templates.append(f"high rated {category} in {city}")
    if category:
        templates.append(f"open {category} near me")
    templates = templates[:5]
    extra_context = compact_json(
        {
            "business_id": poi["raw_id"],
            "name": name,
            "categories": poi["tags"],
            "rating": rating,
            "is_open": obj.get("is_open"),
        }
    )
    queries = []
    qrels = []
    for idx, query_text in enumerate(templates):
        query_id = "yelp_q_" + stable_hash(f"{poi['poi_id']}|{idx}|{query_text}", 18)
        queries.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "city": city,
                "user_lat": "",
                "user_lon": "",
                "timestamp": "",
                "query_type": "template_poi_search",
                "source": "yelp",
                "raw_query_id": f"{poi['raw_id']}:{idx}",
                "extra_context": extra_context,
                "split": split,
            }
        )
        qrels.append({"query_id": query_id, "poi_id": poi["poi_id"], "label": 1, "source": "yelp"})
    return queries, qrels


def process_yelp(raw_dir: Path, output_dir: Path, split_ratio_text: str, seed: int, max_rows: int | None) -> dict[str, Any]:
    path = raw_dir / "yelp/yelp_academic_dataset_business.json"
    out = output_dir / "yelp"
    out.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        log("process_yelp", f"[WARN] file not found: {path}")
        return {}

    split_ratio = parse_ratio(split_ratio_text)
    raw_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            obj = json.loads(line)
            raw_id = clean(obj.get("business_id"))
            if raw_id:
                raw_ids.append(raw_id)
    key_to_split = split_keys_random(raw_ids, split_ratio, seed)

    pois: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            obj = json.loads(line)
            poi = business_to_poi(obj)
            if not poi["raw_id"]:
                continue
            split = key_to_split.get(poi["raw_id"], "train")
            pois.append(poi)
            q, r = build_queries(poi, obj, split)
            queries.extend(q)
            qrels.extend(r)

    pois_df = pd.DataFrame(pois, columns=POI_COLUMNS)
    queries_df = pd.DataFrame(queries)
    qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)
    write_csv(pois_df, out / "pois.csv", POI_COLUMNS)
    write_csv(pd.DataFrame(columns=INTERACTION_COLUMNS), out / "interactions.csv", INTERACTION_COLUMNS)
    write_csv(queries_df.drop(columns=["split"], errors="ignore"), out / "queries.csv", QUERY_COLUMNS)
    write_csv(qrels_df, out / "qrels.csv", QREL_COLUMNS)
    write_csv(pd.DataFrame(columns=MOBILITY_CANDIDATE_COLUMNS), out / "candidates.csv", MOBILITY_CANDIDATE_COLUMNS)
    counts = write_split_files(out, queries_df, qrels_df, QUERY_COLUMNS, QREL_COLUMNS)
    write_manifest(
        out,
        "Yelp",
        "yelp",
        "random_by_poi_id",
        split_ratio_text,
        seed,
        counts,
        False,
        "Template queries are split by business_id to avoid POI label leakage across splits.",
    )

    summary = {
        "dataset_name": "Yelp",
        "source": "yelp",
        "scenario": "template-based POI retrieval",
        "pois": len(pois_df),
        "queries": len(queries_df),
        "qrels": len(qrels_df),
        "interactions": 0,
        "cities": int(pois_df["city"].nunique()) if not pois_df.empty else 0,
        "categories": int(pois_df["category_l1"].nunique()) if not pois_df.empty else 0,
        "split": counts,
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("process_yelp", f"read {len(pois_df)} businesses")
    log("process_yelp", f"write queries={len(queries_df)} qrels={len(qrels_df)}")
    log("process_yelp", f"train/dev/test={counts['train_queries']}/{counts['dev_queries']}/{counts['test_queries']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/data_report")
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_yelp(args.raw_dir, args.output_dir, args.split_ratio, args.seed, args.max_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
