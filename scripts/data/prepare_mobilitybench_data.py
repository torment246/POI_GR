#!/usr/bin/env python3
"""Extract MobilityBench POI Search and Nearby Search data for evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from poi_genret.schema import (  # noqa: E402
    MOBILITY_CANDIDATE_COLUMNS,
    POI_COLUMNS,
    QREL_COLUMNS,
    QUERY_COLUMNS,
    read_csv_if_exists,
    write_csv,
)
from poi_genret.process_utils import stable_hash  # noqa: E402


SANDBOX_FILES = [
    "nested_poi_data.json",
    "search_around_poi_sandbox.json",
    "reverse_geocoding_sandbox.json",
]
CHINA_LAT_MIN = 16.0
CHINA_LAT_MAX = 54.0
CHINA_LON_MIN = 73.0
CHINA_LON_MAX = 135.0


def log(message: str) -> None:
    print(f"[prepare_mobilitybench_data] {message}")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def split_location(value: Any) -> tuple[str, str]:
    text = clean(value)
    if not text or "," not in text:
        return "", ""
    first, second = [part.strip() for part in text.split(",", 1)]
    try:
        a = float(first)
        b = float(second)
    except ValueError:
        return "", ""
    # Amap-style locations are usually lon,lat. Some reverse-geocoding keys are lat,lon.
    if abs(a) <= 90 and abs(b) > 90:
        lat, lon = a, b
    else:
        lon, lat = a, b
    return f"{lat:.6f}", f"{lon:.6f}"


def in_china_bbox(lat: object, lon: object) -> bool:
    try:
        lat_value = float(clean(lat))
        lon_value = float(clean(lon))
    except ValueError:
        return False
    return CHINA_LAT_MIN <= lat_value <= CHINA_LAT_MAX and CHINA_LON_MIN <= lon_value <= CHINA_LON_MAX


def candidate_id(name: str, address: str, lat: str, lon: str) -> str:
    return "mobilitybench:" + stable_hash("|".join([name, address, lat, lon]), length=16)


def looks_like_structured_context(text: object) -> bool:
    value = clean(text)
    if not value:
        return False
    if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
        return True
    if re.search(r"['\"]?(user_current_location|current_location|location|lat|lon|latitude|longitude)['\"]?\s*:", value, re.I):
        return True
    return bool(re.search(r"\d{2,3}\.\d{4,}\s*,\s*\d{1,2}\.\d{4,}", value))


def clean_category(value: object) -> str:
    text = clean(value)
    return "" if looks_like_structured_context(text) else text


def infer_city_from_address(address: object) -> str:
    text = clean(address)
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text)
    for municipality in ["北京市", "上海市", "天津市", "重庆市"]:
        if normalized.startswith(municipality):
            return municipality
    parts = [part for part in normalized.split(" ") if part]
    if len(parts) >= 2 and (parts[0].endswith("省") or parts[0].endswith("自治区")):
        candidate = parts[1]
        if candidate.endswith(("市", "州", "地区", "盟")):
            return candidate
    if parts and parts[0].endswith(("市", "州", "地区", "盟")):
        return parts[0]
    match = re.search(r"(?:省|自治区)\s*([\u4e00-\u9fff]{2,12}(?:市|州|地区|盟))", normalized)
    if match:
        return match.group(1)
    return ""


def normalize_candidate(
    obj: dict[str, Any],
    query_id: str = "",
    candidate_source: str = "",
    source_file: str = "",
    fallback_city: str = "",
    fallback_category: str = "",
) -> dict[str, str] | None:
    name = clean(obj.get("name") or obj.get("poi_name") or obj.get("title"))
    address = clean(obj.get("address") or obj.get("formatted_address") or obj.get("addr"))
    location = obj.get("location") or obj.get("loc")
    lat = clean(obj.get("lat") or obj.get("latitude"))
    lon = clean(obj.get("lng") or obj.get("lon") or obj.get("longitude"))
    if location and (not lat or not lon):
        lat, lon = split_location(location)
    if not (name or address) or not (lat and lon):
        return None
    category = clean_category(obj.get("category") or obj.get("type") or obj.get("keywords") or fallback_category)
    city = clean(obj.get("city")) or infer_city_from_address(address) or clean(fallback_city)
    distance = clean(obj.get("distance"))
    return {
        "query_id": query_id,
        "candidate_poi_id": candidate_id(name, address, lat, lon),
        "name": name,
        "address": address,
        "city": city,
        "lat": lat,
        "lon": lon,
        "category": category,
        "distance": distance,
        "candidate_source": candidate_source,
        "source_file": source_file,
    }


def candidate_to_poi(candidate: dict[str, str]) -> dict[str, str]:
    return {
        "poi_id": candidate["candidate_poi_id"],
        "name": candidate["name"],
        "address": candidate["address"],
        "city": candidate["city"],
        "district": "",
        "business_area": "",
        "lat": candidate["lat"],
        "lon": candidate["lon"],
        "category_l1": candidate["category"],
        "category_l2": "",
        "brand": "",
        "tags": candidate["category"],
        "rating": "",
        "price": "",
        "open_hours": "",
        "source": "mobilitybench",
    }


def recursive_candidates(
    value: Any,
    query_id: str = "",
    candidate_source: str = "",
    source_file: str = "",
    fallback_city: str = "",
    fallback_category: str = "",
) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        candidate = normalize_candidate(value, query_id, candidate_source, source_file, fallback_city, fallback_category)
        if candidate:
            found.append(candidate)
        for key, child in value.items():
            next_city = fallback_city
            next_category = fallback_category
            if isinstance(key, str):
                if key.endswith("市") or key.endswith("区") or key.endswith("县"):
                    next_city = key
                elif not fallback_category and len(key) <= 20:
                    next_category = key
            found.extend(recursive_candidates(child, query_id, candidate_source, source_file, next_city, next_category))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_candidates(child, query_id, candidate_source, source_file, fallback_city, fallback_category))
    return found


def extract_candidates_from_text(
    text: Any,
    query_id: str,
    candidate_source: str,
    source_file: str,
    fallback_city: str = "",
    fallback_category: str = "",
) -> list[dict[str, str]]:
    raw = clean(text)
    if not raw:
        return []
    candidates: list[dict[str, str]] = []
    for chunk in re.findall(r"\{[^{}]*['\"](?:name|poi_name|title)['\"][^{}]*\}", raw, flags=re.S):
        candidate = normalize_candidate(
            {
                "name": extract_key_value(chunk, ["name", "poi_name", "title"]),
                "address": extract_key_value(chunk, ["address", "formatted_address", "addr"]),
                "location": extract_key_value(chunk, ["location"]),
                "lat": extract_key_value(chunk, ["lat", "latitude"]),
                "lon": extract_key_value(chunk, ["lng", "lon", "longitude"]),
                "category": extract_key_value(chunk, ["category", "type", "keywords"]),
                "city": extract_key_value(chunk, ["city"]),
            },
            query_id,
            candidate_source,
            source_file,
            fallback_city,
            fallback_category,
        )
        if candidate:
            candidates.append(candidate)

    if not candidates:
        pattern = re.compile(
            r"['\"]location['\"]:\s*['\"]([^'\"]+)['\"].{0,800}?['\"]name['\"]:\s*['\"]([^'\"]+)['\"].{0,800}?['\"]address['\"]:\s*['\"]([^'\"]*)['\"]",
            re.S,
        )
        for location, name, address in pattern.findall(raw):
            lat, lon = split_location(location)
            candidate = normalize_candidate(
                {"name": name, "address": address, "lat": lat, "lon": lon, "category": fallback_category},
                query_id,
                candidate_source,
                source_file,
                fallback_city,
                fallback_category,
            )
            if candidate:
                candidates.append(candidate)
    return dedupe_candidates(candidates)


def extract_key_value(text: str, keys: list[str]) -> str:
    key_pattern = "|".join(re.escape(key) for key in keys)
    match = re.search(rf"['\"](?:{key_pattern})['\"]\s*:\s*(['\"])(.*?)\1", text, flags=re.S)
    return clean(match.group(2)) if match else ""


def dedupe_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for candidate in candidates:
        key = candidate["candidate_poi_id"] + "|" + candidate.get("query_id", "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def classify_row(row: pd.Series) -> str | None:
    task = clean(row.get("task_scenario")).lower()
    tool_text = " ".join([clean(row.get("tool_list")), clean(row.get("tools_list"))]).lower()
    if task == "nearby search":
        return "nearby_search"
    if task == "poi search":
        return "poi_search"
    if task:
        return None
    if "nearby_poi_query" in tool_text or "search_around_poi" in tool_text:
        return "nearby_search"
    if "poi_query" in tool_text:
        return "poi_search"
    return None


def user_lat_lon(row: pd.Series) -> tuple[str, str]:
    return split_location(row.get("user_loc"))


def build_query(row: pd.Series, query_id: str, query_type: str) -> dict[str, str]:
    user_lat, user_lon = user_lat_lon(row)
    return {
        "query_id": query_id,
        "query_text": clean(row.get("query")),
        "city": clean(row.get("city") or row.get("user_city") or row.get("slot_city")),
        "user_lat": user_lat,
        "user_lon": user_lon,
        "timestamp": clean(row.get("time")),
        "query_type": query_type,
        "source": "mobilitybench",
    }


def extract_row_candidates(row: pd.Series, query_id: str, source_file: str, query_type: str) -> list[dict[str, str]]:
    city = clean(row.get("city") or row.get("user_city") or row.get("slot_city"))
    category = clean(row.get("near_poi_info") or row.get("key_info"))
    fields = ["poi_result", "near_poi_ans", "std_end", "location_ans", "ans_loc"]
    candidates: list[dict[str, str]] = []
    for field in fields:
        candidates.extend(extract_candidates_from_text(row.get(field), query_id, field, source_file, city, category))
    if query_type == "nearby_search":
        for candidate in candidates:
            if not candidate["category"] and category:
                candidate["category"] = category
    return dedupe_candidates(candidates)


def process_csv(path: Path, max_rows: int | None = None) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]], Counter]:
    queries: list[dict[str, str]] = []
    qrels: list[dict[str, Any]] = []
    candidates: list[dict[str, str]] = []
    counters: Counter = Counter()

    df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=max_rows)
    log(f"columns for {path.name}: {list(df.columns)}")
    counters["episodes"] += len(df)
    for idx, row in df.iterrows():
        counters[f"task::{clean(row.get('task_scenario')) or 'missing'}"] += 1
        counters[f"intent::{clean(row.get('intent_family')) or 'missing'}"] += 1
        query_type = classify_row(row)
        if query_type is None:
            continue
        counters[query_type] += 1
        query_id = "mobilitybench_q_" + stable_hash(f"{path.name}|{idx}|{clean(row.get('query'))}|{query_type}", length=16)
        queries.append(build_query(row, query_id, query_type))
        row_candidates = extract_row_candidates(row, query_id, path.name, query_type)
        candidates.extend(row_candidates)
        if query_type == "poi_search" and len(row_candidates) == 1:
            qrels.append({"query_id": query_id, "poi_id": row_candidates[0]["candidate_poi_id"], "label": 1, "source": "mobilitybench"})
            counters["unique_qrels"] += 1
        elif row_candidates:
            counters["candidate_only_queries"] += 1
        else:
            counters["no_candidate_queries"] += 1
    log(f"read {path}: episodes={len(df)} poi_nearby_queries={len(queries)} candidates={len(candidates)} qrels={len(qrels)}")
    return queries, qrels, candidates, counters


def read_sandbox_candidates(raw_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sandbox_dir = raw_dir / "mobilitybench/sandbox"
    candidates: list[dict[str, str]] = []
    pois: list[dict[str, str]] = []
    for name in SANDBOX_FILES:
        path = sandbox_dir / name
        if not path.exists():
            log(f"[WARN] sandbox file not found: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        extracted = recursive_candidates(obj, candidate_source="sandbox", source_file=name)
        candidates.extend(extracted)
        log(f"parsed sandbox {name}: candidates={len(extracted)}")
    candidates = dedupe_candidates(candidates)
    before_filter = len(candidates)
    candidates = [candidate for candidate in candidates if in_china_bbox(candidate.get("lat"), candidate.get("lon"))]
    filtered = before_filter - len(candidates)
    if filtered:
        log(f"filtered sandbox candidates outside approximate China bbox: {filtered}")
    seen_pois: set[str] = set()
    for candidate in candidates:
        poi = candidate_to_poi(candidate)
        if poi["poi_id"] in seen_pois:
            continue
        seen_pois.add(poi["poi_id"])
        pois.append(poi)
    return candidates, pois


def task_distribution_rows(counters: Counter) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counters.items()):
        if "::" in key:
            kind, value = key.split("::", 1)
            rows.append({"distribution_type": kind, "value": value, "count": count})
    rows.extend(
        [
            {"distribution_type": "query_type", "value": "poi_search", "count": counters.get("poi_search", 0)},
            {"distribution_type": "query_type", "value": "nearby_search", "count": counters.get("nearby_search", 0)},
            {"distribution_type": "parse", "value": "unique_qrels", "count": counters.get("unique_qrels", 0)},
            {"distribution_type": "parse", "value": "candidate_only_queries", "count": counters.get("candidate_only_queries", 0)},
            {"distribution_type": "parse", "value": "no_candidate_queries", "count": counters.get("no_candidate_queries", 0)},
        ]
    )
    return rows


def merge_into_unified(output_dir: Path, mobility_pois: pd.DataFrame, mobility_queries: pd.DataFrame, mobility_qrels: pd.DataFrame) -> dict[str, Any]:
    pois = read_csv_if_exists(output_dir / "pois.csv", POI_COLUMNS)
    queries = read_csv_if_exists(output_dir / "queries.csv", QUERY_COLUMNS)
    qrels = read_csv_if_exists(output_dir / "qrels.csv", QREL_COLUMNS)

    merged_pois = pd.concat([pois, mobility_pois], ignore_index=True).drop_duplicates("poi_id", keep="first")
    merged_queries = pd.concat([queries, mobility_queries], ignore_index=True).drop_duplicates("query_id", keep="first")
    merged_qrels = pd.concat([qrels, mobility_qrels], ignore_index=True).drop_duplicates(["query_id", "poi_id"], keep="first")

    write_csv(merged_pois, output_dir / "pois.csv", POI_COLUMNS)
    write_csv(merged_queries, output_dir / "queries.csv", QUERY_COLUMNS)
    write_csv(merged_qrels, output_dir / "qrels.csv", QREL_COLUMNS)
    return {
        "unified_pois": len(merged_pois),
        "unified_queries": len(merged_queries),
        "unified_qrels": len(merged_qrels),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/data_report", help="Accepted for CLI consistency.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional per-CSV row limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=42, help="Accepted for CLI consistency.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted((args.raw_dir / "mobilitybench/datasets").glob("all_data_benchmark_50000*.csv"))
    if not csv_files:
        log(f"[WARN] no MobilityBench CSV files found under {args.raw_dir / 'mobilitybench/datasets'}")

    all_queries: list[dict[str, str]] = []
    all_qrels: list[dict[str, Any]] = []
    all_candidates: list[dict[str, str]] = []
    total_counters: Counter = Counter()
    seen_query_keys: set[tuple[str, str, str]] = set()

    for path in csv_files:
        queries, qrels, candidates, counters = process_csv(path, args.max_rows)
        total_counters.update(counters)
        for query in queries:
            key = (query["query_text"], query["city"], query["query_type"])
            if key in seen_query_keys:
                continue
            seen_query_keys.add(key)
            all_queries.append(query)
        all_qrels.extend(qrels)
        all_candidates.extend(candidates)

    sandbox_candidates, sandbox_pois = read_sandbox_candidates(args.raw_dir)
    all_candidates = dedupe_candidates(all_candidates + sandbox_candidates)
    kept_query_ids = {query["query_id"] for query in all_queries}
    all_qrels = [qrel for qrel in all_qrels if qrel["query_id"] in kept_query_ids]

    candidate_pois = [candidate_to_poi(candidate) for candidate in all_candidates]
    poi_by_id = {poi["poi_id"]: poi for poi in candidate_pois + sandbox_pois}

    mobility_queries = pd.DataFrame(all_queries, columns=QUERY_COLUMNS)
    mobility_qrels = pd.DataFrame(all_qrels, columns=QREL_COLUMNS)
    mobility_pois = pd.DataFrame(poi_by_id.values(), columns=POI_COLUMNS)

    write_csv(mobility_queries, args.output_dir / "mobilitybench_poi_nearby_queries.csv", QUERY_COLUMNS)
    write_csv(mobility_qrels, args.output_dir / "mobilitybench_poi_nearby_qrels.csv", QREL_COLUMNS)
    write_csv(pd.DataFrame(all_candidates, columns=MOBILITY_CANDIDATE_COLUMNS), args.output_dir / "mobilitybench_poi_candidates.csv", MOBILITY_CANDIDATE_COLUMNS)
    write_csv(pd.DataFrame(task_distribution_rows(total_counters)), args.output_dir / "mobilitybench_task_distribution.csv")
    write_csv(mobility_pois, args.output_dir / "mobilitybench_pois.csv", POI_COLUMNS)
    unified_stats = merge_into_unified(args.output_dir, mobility_pois, mobility_queries, mobility_qrels)

    total_queries = len(all_queries)
    parse_rate = (len({qrel["query_id"] for qrel in all_qrels}) / total_queries) if total_queries else 0.0
    print(
        json.dumps(
            {
                "mobilitybench_queries": total_queries,
                "mobilitybench_qrels": len(all_qrels),
                "mobilitybench_candidates": len(all_candidates),
                "mobilitybench_pois": len(poi_by_id),
                "poi_search": sum(1 for q in all_queries if q["query_type"] == "poi_search"),
                "nearby_search": sum(1 for q in all_queries if q["query_type"] == "nearby_search"),
                "unique_target_parse_rate": parse_rate,
                **unified_stats,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
