#!/usr/bin/env python3
"""Process MobilityBench POI/Nearby subsets as an independent evaluation dataset."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
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
    split_location,
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

POI_TASKS = {"poi search"}
NEARBY_TASKS = {"nearby search"}
CHINA_LAT_MIN = 16.0
CHINA_LAT_MAX = 54.0
CHINA_LON_MIN = 73.0
CHINA_LON_MAX = 135.0
EXCLUDED_TASK_KEYWORDS = {
    "planning",
    "route",
    "weather",
    "traffic",
    "geolocation",
    "arrival",
    "departure",
}


def classify_row(row: pd.Series) -> str | None:
    task = clean(row.get("task_scenario")).lower()
    if task in NEARBY_TASKS:
        return "nearby_search"
    if task in POI_TASKS:
        return "poi_search"
    if task:
        return None

    # Fallback only when task_scenario is missing. Do not treat query_poi alone
    # as POI Search because route-planning tasks also call query_poi.
    intent = clean(row.get("intent_family")).lower()
    tools = " ".join([clean(row.get("tool_list")), clean(row.get("tools_list"))]).lower()
    if any(keyword in intent or keyword in tools for keyword in EXCLUDED_TASK_KEYWORDS):
        return None
    if "nearby_poi_query" in tools or "search_around_poi" in tools:
        return "nearby_search"
    if "information retrieval" in intent and "poi_query" in tools:
        return "poi_search"
    return None


def valid_coord(lat: object, lon: object) -> bool:
    try:
        la = float(clean(lat))
        lo = float(clean(lon))
    except ValueError:
        return False
    return -90 <= la <= 90 and -180 <= lo <= 180


def in_china_bbox(lat: object, lon: object) -> bool:
    try:
        la = float(clean(lat))
        lo = float(clean(lon))
    except ValueError:
        return False
    return CHINA_LAT_MIN <= la <= CHINA_LAT_MAX and CHINA_LON_MIN <= lo <= CHINA_LON_MAX


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def candidate_id(name: str, address: str, lat: str, lon: str) -> str:
    return "mobilitybench:" + stable_hash("|".join([name, address, lat, lon]), 18)


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


def normalize_candidate(obj: dict[str, Any], fallback_city: str = "", fallback_category: str = "") -> dict[str, str] | None:
    name = clean(obj.get("name") or obj.get("poi_name") or obj.get("title"))
    address = clean(obj.get("address") or obj.get("formatted_address") or obj.get("addr"))
    lat = clean(obj.get("lat") or obj.get("latitude"))
    lon = clean(obj.get("lng") or obj.get("lon") or obj.get("longitude"))
    if (not lat or not lon) and obj.get("location"):
        lat, lon = split_location(obj.get("location"))
    if not (name or address) or not (lat and lon):
        return None
    category = clean_category(obj.get("category") or obj.get("type") or obj.get("keywords") or fallback_category)
    city = clean(obj.get("city")) or infer_city_from_address(address) or clean(fallback_city)
    raw = stable_hash("|".join([name, address, lat, lon]), 18)
    return {
        "poi_id": "mobilitybench:" + raw,
        "raw_id": raw,
        "name": name,
        "address": address,
        "city": city,
        "lat": lat,
        "lon": lon,
        "category": category,
        "distance": clean(obj.get("distance")),
    }


def candidate_to_poi(candidate: dict[str, str]) -> dict[str, str]:
    return {
        "poi_id": candidate["poi_id"],
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
        "raw_id": candidate["raw_id"],
    }


def extract_from_text(text: Any, fallback_city: str = "", fallback_category: str = "") -> list[dict[str, str]]:
    raw = clean(text)
    if not raw:
        return []
    out: list[dict[str, str]] = []
    for chunk in re.findall(r"\{[^{}]*['\"](?:name|poi_name|title)['\"][^{}]*\}", raw, flags=re.S):
        obj = {
            "name": extract_key_value(chunk, ["name", "poi_name", "title"]),
            "address": extract_key_value(chunk, ["address", "formatted_address", "addr"]),
            "location": extract_key_value(chunk, ["location"]),
            "lat": extract_key_value(chunk, ["lat", "latitude"]),
            "lon": extract_key_value(chunk, ["lng", "lon", "longitude"]),
            "category": extract_key_value(chunk, ["category", "type", "keywords"]),
            "city": extract_key_value(chunk, ["city"]),
        }
        candidate = normalize_candidate(obj, fallback_city, fallback_category)
        if candidate:
            out.append(candidate)

    if not out:
        pattern = re.compile(
            r"['\"]location['\"]:\s*['\"]([^'\"]+)['\"].{0,800}?['\"]name['\"]:\s*['\"]([^'\"]+)['\"].{0,800}?['\"]address['\"]:\s*['\"]([^'\"]*)['\"]",
            re.S,
        )
        for location, name, address in pattern.findall(raw):
            lat, lon = split_location(location)
            candidate = normalize_candidate(
                {"name": name, "address": address, "lat": lat, "lon": lon, "category": fallback_category},
                fallback_city,
                fallback_category,
            )
            if candidate:
                out.append(candidate)
    return dedupe_candidates(out)


def extract_key_value(text: str, keys: list[str]) -> str:
    key_pattern = "|".join(re.escape(key) for key in keys)
    match = re.search(rf"['\"](?:{key_pattern})['\"]\s*:\s*(['\"])(.*?)\1", text, flags=re.S)
    return clean(match.group(2)) if match else ""


def recursive_sandbox(value: Any, fallback_city: str = "", fallback_category: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        cand = normalize_candidate(value, fallback_city, fallback_category)
        if cand:
            found.append(cand)
        for key, child in value.items():
            next_city = fallback_city
            next_category = fallback_category
            if isinstance(key, str):
                if key.endswith("市") or key.endswith("区") or key.endswith("县"):
                    next_city = key
                elif not next_category and len(key) <= 20:
                    next_category = key
            found.extend(recursive_sandbox(child, next_city, next_category))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_sandbox(child, fallback_city, fallback_category))
    return found


def dedupe_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for cand in candidates:
        key = cand["poi_id"]
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def normalize_query_text(text: object) -> str:
    return re.sub(r"\s+", "", clean(text).lower())


def parse_context_value(value: object) -> object:
    if isinstance(value, dict):
        return {k: parse_context_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [parse_context_value(v) for v in value]
    text = clean(value)
    if not text:
        return ""
    if text[:1] not in "{[":
        return text
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if parsed is value:
            continue
        if isinstance(parsed, (dict, list)):
            return parse_context_value(parsed)
        return parsed
    return text


def context_dict(row: pd.Series) -> dict[str, object]:
    parsed = parse_context_value(row.get("context"))
    return parsed if isinstance(parsed, dict) else {}


def first_context_value(ctx: dict[str, object], keys: list[str]) -> str:
    for key in keys:
        value = clean(ctx.get(key))
        if value and value.lower() != "nan":
            return value
    return ""


def city_from_context(row: pd.Series, ctx: dict[str, object]) -> str:
    context_city = first_context_value(ctx, ["address_l3", "city", "user_city", "slot_city", "adcode"])
    if context_city:
        return context_city
    return clean(row.get("city") or row.get("user_city") or row.get("slot_city"))


def user_location_from_context(row: pd.Series, ctx: dict[str, object]) -> tuple[str, str]:
    loc = first_context_value(ctx, ["user_loc", "user_current_location", "current_location", "location"])
    if loc:
        return split_location(loc)
    return split_location(row.get("user_loc"))


def extract_query_candidates(row: pd.Series, query_type: str) -> list[tuple[str, dict[str, str]]]:
    ctx = context_dict(row)
    city = city_from_context(row, ctx)
    category = clean(row.get("near_poi_info") or row.get("key_info") or row.get("near_poi"))
    sources = ["near_poi_ans", "poi_result"] if query_type == "nearby_search" else ["poi_result", "location_ans", "std_end"]
    pairs: list[tuple[str, dict[str, str]]] = []
    for source in sources:
        for cand in extract_from_text(row.get(source), city, category):
            pairs.append((source, cand))
    seen: set[str] = set()
    out = []
    for source, cand in pairs:
        if cand["poi_id"] in seen:
            continue
        seen.add(cand["poi_id"])
        out.append((source, cand))
    return out


def build_query(row: pd.Series, query_id: str, query_type: str, split: str) -> dict[str, Any]:
    ctx = context_dict(row)
    user_lat, user_lon = user_location_from_context(row, ctx)
    city = city_from_context(row, ctx)
    tool_list = clean(row.get("tool_list") or row.get("tools_list"))
    context = {
        "context": parse_context_value(row.get("context")),
        "task_scenario": clean(row.get("task_scenario")),
        "intent_family": clean(row.get("intent_family")),
        "tool_list": tool_list,
        "has_tool_list": bool(tool_list),
    }
    return {
        "query_id": query_id,
        "query_text": clean(row.get("query")),
        "city": city,
        "user_lat": user_lat,
        "user_lon": user_lon,
        "timestamp": clean(row.get("time")),
        "query_type": query_type,
        "source": "mobilitybench",
        "raw_query_id": clean(row.get("_raw_query_id")),
        "extra_context": compact_json(context),
        "split": split,
    }


def process_csvs(
    raw_dir: Path, split_ratio_text: str, seed: int, max_rows: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_ratio = parse_ratio(split_ratio_text)
    csv_files = sorted((raw_dir / "mobilitybench/datasets").glob("all_data_benchmark_50000*.csv"))
    all_rows: list[pd.DataFrame] = []
    task_counts: Counter = Counter()
    total_episodes = 0
    for path in csv_files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=max_rows)
        log("process_mobilitybench", f"{path.name}: columns={list(df.columns)}")
        log("process_mobilitybench", f"{path.name}: read {len(df)} rows")
        total_episodes += len(df)
        task_counts.update(df.get("task_scenario", pd.Series(dtype=str)).replace("", "missing").tolist())
        df["_source_file"] = path.name
        df["_source_row"] = range(len(df))
        all_rows.append(df)
    if not all_rows:
        return empty_frames()
    raw = pd.concat(all_rows, ignore_index=True)
    raw["_query_type"] = raw.apply(classify_row, axis=1)
    raw = raw[raw["_query_type"].notna()].copy()
    city_series = raw["city"].astype(str) if "city" in raw.columns else pd.Series([""] * len(raw))
    raw["_dedupe_key"] = raw["query"].astype(str) + "|" + city_series + "|" + raw["_query_type"].astype(str)
    raw = raw.drop_duplicates("_dedupe_key", keep="first").reset_index(drop=True)
    city_series = raw["city"].astype(str) if "city" in raw.columns else pd.Series([""] * len(raw))
    split_keys = raw["query"].map(normalize_query_text).tolist()
    key_to_split = split_keys_random(split_keys, split_ratio, seed)

    query_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    poi_by_id: dict[str, dict[str, str]] = {}
    candidate_rows: list[dict[str, Any]] = []
    tool_intent_rows: list[dict[str, Any]] = []

    for idx, row in raw.iterrows():
        query_type = clean(row["_query_type"])
        raw_query_id = f"{row['_source_file']}:{row['_source_row']}"
        row["_raw_query_id"] = raw_query_id
        query_id = "mobilitybench_q_" + stable_hash(f"{raw_query_id}|{clean(row.get('query'))}|{query_type}", 18)
        split_key = normalize_query_text(row.get("query"))
        split = key_to_split.get(split_key, "train")
        query = build_query(row, query_id, query_type, split)
        pairs = extract_query_candidates(row, query_type)
        query_rows.append(query)
        for rank, (candidate_source, candidate) in enumerate(pairs, start=1):
            poi_by_id[candidate["poi_id"]] = candidate_to_poi(candidate)
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "poi_id": candidate["poi_id"],
                    "rank": rank,
                    "score": "",
                    "candidate_source": candidate_source,
                    "source": "mobilitybench",
                }
            )
        if query_type == "poi_search" and len(pairs) == 1:
            qrel_rows.append({"query_id": query_id, "poi_id": pairs[0][1]["poi_id"], "label": 1, "source": "mobilitybench"})
        elif pairs:
            tool_intent_rows.append(query)

    meta = {
        "total_episodes": total_episodes,
        "task_distribution": dict(task_counts),
    }
    return (
        pd.DataFrame(poi_by_id.values(), columns=POI_COLUMNS),
        pd.DataFrame(query_rows),
        pd.DataFrame(qrel_rows, columns=QREL_COLUMNS),
        pd.DataFrame(candidate_rows, columns=MOBILITY_CANDIDATE_COLUMNS),
        pd.DataFrame(tool_intent_rows),
        meta,
    )


def empty_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return (
        pd.DataFrame(columns=POI_COLUMNS),
        pd.DataFrame(columns=QUERY_COLUMNS),
        pd.DataFrame(columns=QREL_COLUMNS),
        pd.DataFrame(columns=MOBILITY_CANDIDATE_COLUMNS),
        pd.DataFrame(columns=QUERY_COLUMNS),
        {"total_episodes": 0, "task_distribution": {}},
    )


def read_sandbox(raw_dir: Path) -> pd.DataFrame:
    sandbox_dir = raw_dir / "mobilitybench/sandbox"
    poi_by_id: dict[str, dict[str, str]] = {}
    for path in sorted(sandbox_dir.glob("*.json")):
        if path.name not in {"nested_poi_data.json", "search_around_poi_sandbox.json", "reverse_geocoding_sandbox.json"}:
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        candidates = recursive_sandbox(obj)
        log("process_mobilitybench", f"{path.name}: parsed sandbox candidates={len(candidates)}")
        for candidate in candidates:
            poi_by_id[candidate["poi_id"]] = candidate_to_poi(candidate)
    return pd.DataFrame(poi_by_id.values(), columns=POI_COLUMNS)


def count_non_china_bbox_pois(pois: pd.DataFrame) -> int:
    if pois.empty:
        return 0
    return int((~pois.apply(lambda row: in_china_bbox(row.get("lat"), row.get("lon")), axis=1)).sum())


def filter_non_china_bbox_pois(pois: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    if pois.empty:
        return pois, 0
    mask = pois.apply(lambda row: in_china_bbox(row.get("lat"), row.get("lon")), axis=1)
    removed = int((~mask).sum())
    if removed:
        samples = pois.loc[~mask, ["poi_id", "name", "address", "lat", "lon"]].head(5).to_dict("records")
        log("process_mobilitybench", f"{label}: filtered non-China bbox POIs={removed}, examples={samples}")
    return pois.loc[mask].copy(), removed


def write_tool_intent_splits(output_dir: Path, tool_intent: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for split in ["train", "dev", "test"]:
        df = tool_intent[tool_intent["split"] == split].drop(columns=["split"], errors="ignore") if not tool_intent.empty else pd.DataFrame(columns=QUERY_COLUMNS)
        write_csv(df, output_dir / f"{split}_tool_intent_queries.csv", QUERY_COLUMNS)
        counts[f"{split}_tool_intent_queries"] = len(df)
    return counts


def build_nearby_spatial_quality(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    pois: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if queries.empty or candidates.empty or pois.empty:
        return pd.DataFrame(columns=QUERY_COLUMNS), pd.DataFrame()

    nearby_queries = queries[queries["query_type"] == "nearby_search"].copy()
    joined = nearby_queries.merge(candidates, on="query_id", how="inner")
    joined = joined.merge(
        pois[["poi_id", "name", "address", "lat", "lon"]],
        on="poi_id",
        how="left",
        suffixes=("", "_poi"),
    )
    abnormal_rows: list[dict[str, Any]] = []
    evaluated_query_ids: set[str] = set()
    abnormal_query_ids: set[str] = set()

    for _, row in joined.iterrows():
        if not valid_coord(row.get("user_lat"), row.get("user_lon")):
            continue
        if not valid_coord(row.get("lat"), row.get("lon")):
            continue
        evaluated_query_ids.add(clean(row.get("query_id")))
        distance = haversine_km(
            float(row["user_lat"]),
            float(row["user_lon"]),
            float(row["lat"]),
            float(row["lon"]),
        )
        if distance > 50:
            query_id = clean(row.get("query_id"))
            abnormal_query_ids.add(query_id)
            abnormal_rows.append(
                {
                    "query_id": query_id,
                    "query_text": clean(row.get("query_text")),
                    "city": clean(row.get("city")),
                    "user_lat": clean(row.get("user_lat")),
                    "user_lon": clean(row.get("user_lon")),
                    "poi_id": clean(row.get("poi_id")),
                    "poi_name": clean(row.get("name")),
                    "poi_address": clean(row.get("address")),
                    "poi_lat": clean(row.get("lat")),
                    "poi_lon": clean(row.get("lon")),
                    "distance_km": distance,
                }
            )

    clean_ids = evaluated_query_ids.difference(abnormal_query_ids)
    clean_nearby = nearby_queries[nearby_queries["query_id"].isin(clean_ids)].drop(columns=["split"], errors="ignore")
    return clean_nearby, pd.DataFrame(abnormal_rows)


def process_mobilitybench(raw_dir: Path, output_dir: Path, split_ratio_text: str, seed: int, max_rows: int | None) -> dict[str, Any]:
    out = output_dir / "mobilitybench"
    out.mkdir(parents=True, exist_ok=True)
    row_pois, queries, qrels, candidates, tool_intent, meta = process_csvs(raw_dir, split_ratio_text, seed, max_rows)
    sandbox_pois = read_sandbox(raw_dir)
    sandbox_pois_raw = len(sandbox_pois)
    sandbox_pois, sandbox_filtered_non_china = filter_non_china_bbox_pois(sandbox_pois, "sandbox_poi_catalog")
    row_pois_non_china = count_non_china_bbox_pois(row_pois)
    if row_pois_non_china:
        log(
            "process_mobilitybench",
            f"[WARN] strict POI/Nearby extracted POIs outside China bbox={row_pois_non_china}; "
            "kept to avoid breaking qrels/candidates references.",
        )
    pois = pd.concat([row_pois, sandbox_pois], ignore_index=True).drop_duplicates("poi_id", keep="first")
    candidates = candidates[candidates["query_id"].astype(str).str.strip() != ""].drop_duplicates(["query_id", "poi_id", "candidate_source"], keep="first")
    clean_nearby, nearby_abnormal = build_nearby_spatial_quality(queries, candidates, pois)

    write_csv(pois, out / "pois.csv", POI_COLUMNS)
    write_csv(sandbox_pois, out / "poi_catalog_from_sandbox.csv", POI_COLUMNS)
    write_csv(pd.DataFrame(columns=INTERACTION_COLUMNS), out / "interactions.csv", INTERACTION_COLUMNS)
    write_csv(queries.drop(columns=["split"], errors="ignore"), out / "queries.csv", QUERY_COLUMNS)
    write_csv(qrels, out / "qrels.csv", QREL_COLUMNS)
    write_csv(candidates, out / "candidates.csv", MOBILITY_CANDIDATE_COLUMNS)
    write_csv(clean_nearby, out / "clean_nearby_queries.csv", QUERY_COLUMNS)
    nearby_abnormal.to_csv(out / "nearby_distance_abnormal_samples.csv", index=False, encoding="utf-8")
    write_csv(tool_intent.drop(columns=["split"], errors="ignore") if not tool_intent.empty else pd.DataFrame(columns=QUERY_COLUMNS), out / "tool_intent_queries.csv", QUERY_COLUMNS)
    counts = write_split_files(out, queries, qrels, QUERY_COLUMNS, QREL_COLUMNS)
    counts.update(write_tool_intent_splits(out, tool_intent))
    write_manifest(
        out,
        "MobilityBench",
        "mobilitybench",
        "stratified_random_by_query_type_city_query",
        split_ratio_text,
        seed,
        counts,
        False,
        "Only strict POI Search and Nearby Search rows from the two main benchmark CSVs are retained. POI Query is excluded. qrels.csv contains only unique-target POI Search labels; Nearby Search rows are kept with candidates/tool_intent evidence. Sandbox POI catalog rows outside an approximate China bbox are filtered.",
    )
    summary = {
        "dataset_name": "MobilityBench",
        "source": "mobilitybench",
        "scenario": "Chinese real-map POI Search / Nearby Search",
        "total_episodes": meta["total_episodes"],
        "pois": len(pois),
        "queries": len(queries),
        "qrels": len(qrels),
        "qrels_poi_search": len(qrels),
        "qrels_nearby_search": 0,
        "interactions": 0,
        "candidates": len(candidates),
        "sandbox_poi_catalog": len(sandbox_pois),
        "sandbox_poi_catalog_raw": sandbox_pois_raw,
        "sandbox_poi_catalog_filtered_non_china_bbox": sandbox_filtered_non_china,
        "row_poi_non_china_bbox": row_pois_non_china,
        "poi_search": int((queries["query_type"] == "poi_search").sum()) if not queries.empty else 0,
        "nearby_search": int((queries["query_type"] == "nearby_search").sum()) if not queries.empty else 0,
        "clean_nearby_queries": len(clean_nearby),
        "nearby_distance_abnormal_queries": int(nearby_abnormal["query_id"].nunique()) if not nearby_abnormal.empty else 0,
        "nearby_distance_abnormal_candidates": len(nearby_abnormal),
        "tool_intent_queries": len(tool_intent),
        "split": counts,
        "task_distribution": meta["task_distribution"],
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"task_scenario": k, "count": v} for k, v in meta["task_distribution"].items()]).to_csv(
        out / "task_distribution.csv", index=False, encoding="utf-8"
    )
    log("process_mobilitybench", f"write queries={len(queries)} qrels={len(qrels)} candidates={len(candidates)}")
    log("process_mobilitybench", f"train/dev/test={counts['train_queries']}/{counts['dev_queries']}/{counts['test_queries']}")
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
    summary = process_mobilitybench(args.raw_dir, args.output_dir, args.split_ratio, args.seed, args.max_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
