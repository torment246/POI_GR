#!/usr/bin/env python3
"""Validate processed MobilityBench POI/Nearby data quality.

This script is read-only with respect to data/processed/mobilitybench. It writes
quality reports under outputs/data_report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


QUERY_COLUMNS = [
    "query_id",
    "query_text",
    "city",
    "user_lat",
    "user_lon",
    "timestamp",
    "query_type",
    "source",
    "raw_query_id",
    "extra_context",
]
QREL_COLUMNS = ["query_id", "poi_id", "label", "source"]
POI_COLUMNS = [
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
CANDIDATE_COLUMNS = ["query_id", "poi_id", "rank", "score", "candidate_source", "source"]
POI_TASKS = {"poi search"}
NEARBY_TASKS = {"nearby search"}
EXCLUDED_TASK_KEYWORDS = {
    "planning",
    "route",
    "weather",
    "traffic",
    "geolocation",
    "arrival",
    "departure",
}


def log(message: str) -> None:
    print(f"[validate_mobilitybench_data] {message}")


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        log(f"[WARN] missing file: {path}")
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def load_extra_context(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_tool_names(tool_text: str) -> list[str]:
    if not tool_text:
        return []
    names = re.findall(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", tool_text)
    if names:
        return names
    fallback = []
    for name in ["nearby_poi_query", "poi_query", "search_around_poi", "query_poi", "weather_query", "traffic_info"]:
        if name in tool_text:
            fallback.append(name)
    return fallback


def classify_raw_row(row: pd.Series) -> str | None:
    task = clean(row.get("task_scenario")).lower()
    if task in NEARBY_TASKS:
        return "nearby_search"
    if task in POI_TASKS:
        return "poi_search"
    if task:
        return None

    tool_text = " ".join([clean(row.get("tool_list")), clean(row.get("tools_list"))]).lower()
    intent = clean(row.get("intent_family")).lower()
    tool_names = set(extract_tool_names(tool_text))
    if any(keyword in intent or keyword in tool_text for keyword in EXCLUDED_TASK_KEYWORDS):
        return None
    if "nearby_poi_query" in tool_names or "search_around_poi" in tool_names:
        return "nearby_search"
    if "information retrieval" in intent and "poi_query" in tool_names:
        return "poi_search"
    return None


def normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", "", clean(text).lower())


def missing_rate(df: pd.DataFrame, table: str) -> list[dict[str, Any]]:
    rows = []
    if df.empty:
        return rows
    for col in df.columns:
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        rows.append(
            {
                "section": "missing_rate",
                "table": table,
                "metric": col,
                "value": int(missing.sum()),
                "rate": float(missing.mean()),
            }
        )
    return rows


def check_required(df: pd.DataFrame, table: str, required: list[str]) -> list[dict[str, Any]]:
    rows = []
    for col in required:
        exists = col in df.columns
        empty_count = int((df[col].astype(str).str.strip() == "").sum()) if exists and not df.empty else None
        rows.append(
            {
                "section": "schema",
                "table": table,
                "metric": col,
                "value": "present" if exists else "missing",
                "rate": "" if empty_count is None else empty_count / len(df) if len(df) else 0,
            }
        )
    return rows


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def valid_coord(lat: Any, lon: Any) -> bool:
    try:
        la = float(lat)
        lo = float(lon)
    except (TypeError, ValueError):
        return False
    return -90 <= la <= 90 and -180 <= lo <= 180


def raw_stats(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], Counter, Counter]:
    csv_files = sorted((raw_dir / "mobilitybench/datasets").glob("all_data_benchmark_50000*.csv"))
    frames = []
    task_counter: Counter = Counter()
    tool_counter: Counter = Counter()
    total = 0
    for path in csv_files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        log(f"raw {path.name}: rows={len(df)} columns={list(df.columns)}")
        df["_source_file"] = path.name
        df["_source_row"] = range(len(df))
        total += len(df)
        if "task_scenario" in df.columns:
            task_counter.update(df["task_scenario"].replace("", "missing").tolist())
        for col in ["tool_list", "tools_list"]:
            if col in df.columns:
                for text in df[col].astype(str):
                    names = extract_tool_names(text)
                    if names:
                        tool_counter.update(names)
                    elif text and text != "nan":
                        tool_counter["unparsed_tool_list"] += 1
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not raw.empty:
        raw["_strict_query_type"] = raw.apply(classify_raw_row, axis=1)
    strict_counts = raw["_strict_query_type"].value_counts(dropna=True).to_dict() if "_strict_query_type" in raw else {}
    return raw, {"raw_episodes": total, **strict_counts}, task_counter, tool_counter


def build_raw_lookup(raw: pd.DataFrame) -> dict[str, pd.Series]:
    if raw.empty:
        return {}
    return {f"{row['_source_file']}:{row['_source_row']}": row for _, row in raw.iterrows()}


def raw_excerpt(row: pd.Series | None) -> str:
    if row is None:
        return ""
    fields = ["poi_result", "near_poi_ans", "std_end", "location_ans", "route_ans"]
    parts = []
    for field in fields:
        value = clean(row.get(field))
        if value:
            parts.append(f"{field}: {value[:500]}")
    return " | ".join(parts)[:1200]


def top5_candidate_names(query_id: str, candidates: pd.DataFrame, pois: pd.DataFrame) -> str:
    if candidates.empty or pois.empty:
        return ""
    poi_lookup = pois.set_index("poi_id")
    rows = candidates[candidates["query_id"] == query_id].copy()
    if rows.empty:
        return ""
    rows["rank_num"] = pd.to_numeric(rows.get("rank", ""), errors="coerce")
    rows = rows.sort_values("rank_num", na_position="last").head(5)
    names = []
    for poi_id in rows["poi_id"]:
        if poi_id in poi_lookup.index:
            name = clean(poi_lookup.loc[poi_id].get("name"))
            names.append(name or poi_id)
        else:
            names.append(poi_id)
    return " | ".join(names)


def make_manual_samples(
    report_dir: Path,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    pois: pd.DataFrame,
    candidates: pd.DataFrame,
    tool_intent: pd.DataFrame,
    raw_lookup: dict[str, pd.Series],
    qrel_not_in_candidates: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    poi_lookup = pois.set_index("poi_id") if not pois.empty else pd.DataFrame()
    qrel_queries = queries.merge(qrels, on="query_id", how="inner", suffixes=("", "_qrel"))

    buckets = []
    buckets.append(("poi_search_with_qrels", qrel_queries[qrel_queries["query_type"] == "poi_search"], 50))
    buckets.append(("nearby_search_with_qrels", qrel_queries[qrel_queries["query_type"] == "nearby_search"], 50))
    buckets.append(("tool_intent", tool_intent.copy(), 30))
    buckets.append(("qrel_not_in_candidates", qrel_not_in_candidates.copy(), 30))

    sample_rows = []
    for bucket_name, df, n in buckets:
        if df.empty:
            continue
        sample = df.sample(n=min(n, len(df)), random_state=seed)
        for _, row in sample.iterrows():
            query_id = clean(row.get("query_id"))
            qrel_poi_id = clean(row.get("poi_id"))
            qrel = poi_lookup.loc[qrel_poi_id] if qrel_poi_id and not poi_lookup.empty and qrel_poi_id in poi_lookup.index else {}
            raw_id = clean(row.get("raw_query_id"))
            raw_row = raw_lookup.get(raw_id)
            sample_rows.append(
                {
                    "sample_bucket": bucket_name,
                    "query_id": query_id,
                    "query_text": clean(row.get("query_text")),
                    "city": clean(row.get("city")),
                    "user_loc": ",".join([clean(row.get("user_lon")), clean(row.get("user_lat"))]).strip(","),
                    "query_type": clean(row.get("query_type")),
                    "task_scenario": clean(load_extra_context(clean(row.get("extra_context"))).get("task_scenario")),
                    "tool_list": clean(load_extra_context(clean(row.get("extra_context"))).get("tool_list")),
                    "qrel_poi_name": clean(qrel.get("name") if isinstance(qrel, pd.Series) else ""),
                    "qrel_poi_address": clean(qrel.get("address") if isinstance(qrel, pd.Series) else ""),
                    "qrel_lat": clean(qrel.get("lat") if isinstance(qrel, pd.Series) else ""),
                    "qrel_lon": clean(qrel.get("lon") if isinstance(qrel, pd.Series) else ""),
                    "top5_candidate_names": top5_candidate_names(query_id, candidates, pois),
                    "raw_result_excerpt": raw_excerpt(raw_row),
                    "manual_label": "",
                    "manual_comment": "",
                }
            )
    out = pd.DataFrame(sample_rows)
    out.to_csv(report_dir / "mobilitybench_manual_check_samples.csv", index=False, encoding="utf-8")
    return out


def write_markdown(path: Path, sections: dict[str, Any]) -> None:
    def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
        if df.empty:
            return "No data."
        view = df.head(max_rows).copy()
        cols = list(view.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = [str(row[col]).replace("|", "\\|").replace("\n", " ") for col in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# MobilityBench Quality Report",
        "",
        "This report validates processed MobilityBench data without modifying any processed dataset files.",
        "",
        "## Summary",
        "",
    ]
    for key, value in sections["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Raw Task Scenario Distribution", ""])
    lines.append(md_table(sections["task_distribution"], 30))
    lines.extend(["", "## Raw Tool Distribution", ""])
    lines.append(md_table(sections["tool_distribution"], 30))
    lines.extend(["", "## Schema And Missing Rates", ""])
    lines.append(md_table(sections["missing_summary"], 80))
    lines.extend(["", "## Reference Integrity", ""])
    lines.append(md_table(sections["integrity"]))
    lines.extend(["", "## Qrels Quality", ""])
    lines.append(md_table(sections["qrels_quality"]))
    lines.extend(["", "## Candidate Quality", ""])
    lines.append(md_table(sections["candidate_quality"]))
    lines.extend(["", "## Split Quality", ""])
    lines.append(md_table(sections["split_quality"]))
    lines.extend(["", "## Notable Samples", ""])
    lines.append("Manual check samples are written to `mobilitybench_manual_check_samples.csv`.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/mobilitybench")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/data_report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    raw, raw_count_summary, task_counter, tool_counter = raw_stats(args.raw_dir)
    raw_lookup = build_raw_lookup(raw)

    queries = read_csv(args.processed_dir / "queries.csv", QUERY_COLUMNS)
    qrels = read_csv(args.processed_dir / "qrels.csv", QREL_COLUMNS)
    pois = read_csv(args.processed_dir / "pois.csv", POI_COLUMNS)
    candidates = read_csv(args.processed_dir / "candidates.csv", CANDIDATE_COLUMNS)
    train_queries = read_csv(args.processed_dir / "train_queries.csv", QUERY_COLUMNS)
    dev_queries = read_csv(args.processed_dir / "dev_queries.csv", QUERY_COLUMNS)
    test_queries = read_csv(args.processed_dir / "test_queries.csv", QUERY_COLUMNS)
    train_qrels = read_csv(args.processed_dir / "train_qrels.csv", QREL_COLUMNS)
    dev_qrels = read_csv(args.processed_dir / "dev_qrels.csv", QREL_COLUMNS)
    test_qrels = read_csv(args.processed_dir / "test_qrels.csv", QREL_COLUMNS)
    tool_intent = read_csv(args.processed_dir / "tool_intent_queries.csv", QUERY_COLUMNS)
    clean_nearby_queries = read_csv(args.processed_dir / "clean_nearby_queries.csv", QUERY_COLUMNS)

    summary_rows: list[dict[str, Any]] = []
    for table, df, cols in [
        ("queries", queries, QUERY_COLUMNS),
        ("qrels", qrels, QREL_COLUMNS),
        ("pois", pois, POI_COLUMNS),
        ("candidates", candidates, CANDIDATE_COLUMNS),
    ]:
        summary_rows.extend(check_required(df, table, cols))
        summary_rows.extend(missing_rate(df, table))

    processed_type_counts = queries["query_type"].value_counts().to_dict() if "query_type" in queries else {}
    invalid_query_types = queries[~queries["query_type"].isin(["poi_search", "nearby_search"])] if "query_type" in queries else queries

    context_df = queries[["query_id", "query_text", "query_type", "city", "raw_query_id", "extra_context"]].copy() if not queries.empty else pd.DataFrame()
    if not context_df.empty:
        context_df["task_scenario"] = context_df["extra_context"].map(lambda x: clean(load_extra_context(x).get("task_scenario")))
        context_df["tool_list"] = context_df["extra_context"].map(lambda x: clean(load_extra_context(x).get("tool_list")))
        non_poi_mask = context_df["task_scenario"].str.contains(
            "Planning|Route|Weather|Traffic|Geolocation|Arrival|Departure",
            case=False,
            na=False,
        )
        non_poi_mixed = context_df[non_poi_mask]
    else:
        non_poi_mixed = pd.DataFrame()

    query_ids = set(queries["query_id"]) if "query_id" in queries else set()
    poi_ids = set(pois["poi_id"]) if "poi_id" in pois else set()
    qrel_orphan_query = qrels[~qrels["query_id"].isin(query_ids)] if not qrels.empty else qrels
    qrel_orphan_poi = qrels[~qrels["poi_id"].isin(poi_ids)] if not qrels.empty else qrels
    candidate_orphan_query = candidates[(candidates["query_id"] != "") & ~candidates["query_id"].isin(query_ids)] if not candidates.empty else candidates
    candidate_orphan_poi = candidates[~candidates["poi_id"].isin(poi_ids)] if not candidates.empty else candidates

    qrel_counts = qrels.groupby("query_id").size() if not qrels.empty else pd.Series(dtype=int)
    candidate_pairs = set(zip(candidates["query_id"], candidates["poi_id"])) if not candidates.empty else set()
    qrel_in_candidate_mask = qrels.apply(lambda r: (r["query_id"], r["poi_id"]) in candidate_pairs, axis=1) if not qrels.empty else pd.Series(dtype=bool)
    qrel_in_candidates_rate = float(qrel_in_candidate_mask.mean()) if len(qrel_in_candidate_mask) else 0.0
    qrel_not_in_candidates = qrels[~qrel_in_candidate_mask].merge(queries, on="query_id", how="left") if not qrels.empty else pd.DataFrame()

    candidate_counts = candidates[candidates["query_id"] != ""].groupby("query_id").size() if not candidates.empty else pd.Series(dtype=int)
    dup_candidates = candidates[candidates["query_id"] != ""].duplicated(["query_id", "poi_id"], keep=False) if not candidates.empty else pd.Series(dtype=bool)
    duplicate_candidate_rate = float(dup_candidates.mean()) if len(dup_candidates) else 0.0

    cand_pois = candidates.merge(pois, on="poi_id", how="left", suffixes=("", "_poi")) if not candidates.empty else pd.DataFrame()
    candidate_attr_missing = []
    for col in ["name", "address", "lat", "lon", "category_l1"]:
        if col in cand_pois:
            miss = cand_pois[col].astype(str).str.strip() == ""
            candidate_attr_missing.append({"metric": f"candidate_{col}_missing_rate", "value": float(miss.mean()) if len(miss) else 0.0})
    abnormal_coord = cand_pois[~cand_pois.apply(lambda r: valid_coord(r.get("lat"), r.get("lon")), axis=1)] if not cand_pois.empty else pd.DataFrame()
    empty_name = cand_pois[cand_pois.get("name", pd.Series(dtype=str)).astype(str).str.strip() == ""] if not cand_pois.empty else pd.DataFrame()

    nearby = queries[queries["query_type"] == "nearby_search"].merge(candidates[candidates["query_id"] != ""], on="query_id", how="inner")
    nearby = nearby.merge(pois[["poi_id", "name", "address", "lat", "lon"]], on="poi_id", how="left") if not nearby.empty else nearby
    distances = []
    abnormal_nearby_rows = []
    for _, row in nearby.iterrows():
        if not valid_coord(row.get("user_lat"), row.get("user_lon")) or not valid_coord(row.get("lat"), row.get("lon")):
            continue
        dist = haversine_km(float(row["user_lat"]), float(row["user_lon"]), float(row["lat"]), float(row["lon"]))
        distances.append(dist)
        if dist > 50:
            abnormal_nearby_rows.append(
                {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "poi_id": row["poi_id"],
                    "poi_name": row.get("name", ""),
                    "distance_km": dist,
                }
            )
    distance_series = pd.Series(distances, dtype=float)
    abnormal_nearby = pd.DataFrame(abnormal_nearby_rows)

    split_sets = {
        "train": set(train_queries["query_id"]) if "query_id" in train_queries else set(),
        "dev": set(dev_queries["query_id"]) if "query_id" in dev_queries else set(),
        "test": set(test_queries["query_id"]) if "query_id" in test_queries else set(),
    }
    query_split_overlap = (
        len(split_sets["train"] & split_sets["dev"])
        + len(split_sets["train"] & split_sets["test"])
        + len(split_sets["dev"] & split_sets["test"])
    )
    norm_sets = {
        "train": set(train_queries["query_text"].map(normalize_query_text)) if "query_text" in train_queries else set(),
        "dev": set(dev_queries["query_text"].map(normalize_query_text)) if "query_text" in dev_queries else set(),
        "test": set(test_queries["query_text"].map(normalize_query_text)) if "query_text" in test_queries else set(),
    }
    normalized_query_overlap = (
        len(norm_sets["train"] & norm_sets["dev"])
        + len(norm_sets["train"] & norm_sets["test"])
        + len(norm_sets["dev"] & norm_sets["test"])
    )
    split_qrel_violations = {
        "train": int((~train_qrels["query_id"].isin(split_sets["train"])).sum()) if not train_qrels.empty else 0,
        "dev": int((~dev_qrels["query_id"].isin(split_sets["dev"])).sum()) if not dev_qrels.empty else 0,
        "test": int((~test_qrels["query_id"].isin(split_sets["test"])).sum()) if not test_qrels.empty else 0,
    }
    split_dist_rows = []
    for split, df in [("train", train_queries), ("dev", dev_queries), ("test", test_queries)]:
        if df.empty:
            continue
        for (qt, city), count in df.groupby(["query_type", "city"]).size().sort_values(ascending=False).head(20).items():
            split_dist_rows.append({"split": split, "query_type": qt, "city": city, "count": int(count)})
    split_distribution = pd.DataFrame(split_dist_rows)

    integrity = pd.DataFrame(
        [
            {"metric": "qrels_query_ref_valid_rate", "value": 1 - len(qrel_orphan_query) / len(qrels) if len(qrels) else 1.0},
            {"metric": "qrels_poi_ref_valid_rate", "value": 1 - len(qrel_orphan_poi) / len(qrels) if len(qrels) else 1.0},
            {"metric": "orphan_qrels_query_count", "value": len(qrel_orphan_query)},
            {"metric": "orphan_qrels_poi_count", "value": len(qrel_orphan_poi)},
            {"metric": "orphan_candidates_query_count", "value": len(candidate_orphan_query)},
            {"metric": "orphan_candidates_poi_count", "value": len(candidate_orphan_poi)},
        ]
    )
    qrels_quality = pd.DataFrame(
        [
            {"metric": "queries_with_qrels", "value": int(qrel_counts.size)},
            {"metric": "qrel_count_min", "value": int(qrel_counts.min()) if len(qrel_counts) else 0},
            {"metric": "qrel_count_p50", "value": float(qrel_counts.quantile(0.5)) if len(qrel_counts) else 0},
            {"metric": "qrel_count_max", "value": int(qrel_counts.max()) if len(qrel_counts) else 0},
            {"metric": "qrel_in_candidates_rate", "value": qrel_in_candidates_rate},
            {"metric": "qrel_not_in_candidates_count", "value": len(qrel_not_in_candidates)},
        ]
    )
    candidate_quality = pd.DataFrame(
        [
            {"metric": "queries_with_candidates", "value": int(candidate_counts.size)},
            {"metric": "candidate_count_min", "value": int(candidate_counts.min()) if len(candidate_counts) else 0},
            {"metric": "candidate_count_p50", "value": float(candidate_counts.quantile(0.5)) if len(candidate_counts) else 0},
            {"metric": "candidate_count_p95", "value": float(candidate_counts.quantile(0.95)) if len(candidate_counts) else 0},
            {"metric": "candidate_count_max", "value": int(candidate_counts.max()) if len(candidate_counts) else 0},
            {"metric": "duplicate_candidate_rate", "value": duplicate_candidate_rate},
            {"metric": "empty_candidate_name_count", "value": len(empty_name)},
            {"metric": "abnormal_candidate_coord_count", "value": len(abnormal_coord)},
            *candidate_attr_missing,
        ]
    )
    split_quality = pd.DataFrame(
        [
            {"metric": "query_split_overlap", "value": query_split_overlap},
            {"metric": "normalized_query_overlap", "value": normalized_query_overlap},
            {"metric": "train_qrel_split_violations", "value": split_qrel_violations["train"]},
            {"metric": "dev_qrel_split_violations", "value": split_qrel_violations["dev"]},
            {"metric": "test_qrel_split_violations", "value": split_qrel_violations["test"]},
        ]
    )

    task_distribution = pd.DataFrame(task_counter.most_common(), columns=["task_scenario", "count"])
    tool_distribution = pd.DataFrame(tool_counter.most_common(), columns=["tool", "count"])
    missing_summary = pd.DataFrame(summary_rows)
    quality_rows = []
    quality_rows.extend(summary_rows)
    for name, df in [
        ("integrity", integrity),
        ("qrels_quality", qrels_quality),
        ("candidate_quality", candidate_quality),
        ("split_quality", split_quality),
    ]:
        for _, row in df.iterrows():
            quality_rows.append({"section": name, "table": "", "metric": row["metric"], "value": row["value"], "rate": ""})
    quality_rows.extend(
        [
            {"section": "raw", "table": "raw", "metric": "raw_episodes", "value": raw_count_summary.get("raw_episodes", 0), "rate": ""},
            {"section": "raw", "table": "raw", "metric": "raw_strict_poi_search", "value": raw_count_summary.get("poi_search", 0), "rate": ""},
            {"section": "raw", "table": "raw", "metric": "raw_strict_nearby_search", "value": raw_count_summary.get("nearby_search", 0), "rate": ""},
            {"section": "processed", "table": "queries", "metric": "processed_queries", "value": len(queries), "rate": ""},
            {"section": "processed", "table": "queries", "metric": "processed_poi_search", "value": processed_type_counts.get("poi_search", 0), "rate": ""},
            {"section": "processed", "table": "queries", "metric": "processed_nearby_search", "value": processed_type_counts.get("nearby_search", 0), "rate": ""},
            {"section": "processed", "table": "queries", "metric": "invalid_query_type_count", "value": len(invalid_query_types), "rate": ""},
            {"section": "processed", "table": "queries", "metric": "non_poi_task_mixed_count", "value": len(non_poi_mixed), "rate": ""},
            {"section": "nearby", "table": "distance", "metric": "nearby_distance_count", "value": len(distance_series), "rate": ""},
            {"section": "nearby", "table": "distance", "metric": "nearby_distance_p50_km", "value": float(distance_series.quantile(0.5)) if len(distance_series) else "", "rate": ""},
            {"section": "nearby", "table": "distance", "metric": "nearby_distance_p95_km", "value": float(distance_series.quantile(0.95)) if len(distance_series) else "", "rate": ""},
            {"section": "nearby", "table": "distance", "metric": "nearby_distance_abnormal_gt50km_count", "value": len(abnormal_nearby), "rate": ""},
            {"section": "nearby", "table": "clean_subset", "metric": "clean_nearby_queries", "value": len(clean_nearby_queries), "rate": ""},
        ]
    )
    quality_summary = pd.DataFrame(quality_rows)
    quality_summary.to_csv(args.report_dir / "mobilitybench_quality_summary.csv", index=False, encoding="utf-8")

    manual = make_manual_samples(
        args.report_dir,
        queries,
        qrels,
        pois,
        candidates,
        tool_intent,
        raw_lookup,
        qrel_not_in_candidates,
        args.seed,
    )

    sections = {
        "summary": {
            "raw episodes": raw_count_summary.get("raw_episodes", 0),
            "raw strict poi_search": raw_count_summary.get("poi_search", 0),
            "raw strict nearby_search": raw_count_summary.get("nearby_search", 0),
            "processed queries": len(queries),
            "processed poi_search": processed_type_counts.get("poi_search", 0),
            "processed nearby_search": processed_type_counts.get("nearby_search", 0),
            "qrels": len(qrels),
            "pois": len(pois),
            "candidates": len(candidates),
            "invalid query type count": len(invalid_query_types),
            "non-POI task mixed count": len(non_poi_mixed),
            "qrel in candidates rate": f"{qrel_in_candidates_rate:.6f}",
            "nearby distance abnormal >50km": len(abnormal_nearby),
            "clean nearby queries": len(clean_nearby_queries),
            "manual samples": len(manual),
        },
        "task_distribution": task_distribution,
        "tool_distribution": tool_distribution,
        "missing_summary": missing_summary,
        "integrity": integrity,
        "qrels_quality": qrels_quality,
        "candidate_quality": candidate_quality,
        "split_quality": split_quality,
    }
    write_markdown(args.report_dir / "mobilitybench_quality_report.md", sections)

    # Sample detail files are useful when the headline counts are non-zero.
    qrel_orphan_query.head(50).to_csv(args.report_dir / "mobilitybench_orphan_qrels_query_samples.csv", index=False, encoding="utf-8")
    qrel_orphan_poi.head(50).to_csv(args.report_dir / "mobilitybench_orphan_qrels_poi_samples.csv", index=False, encoding="utf-8")
    candidate_orphan_query.head(50).to_csv(args.report_dir / "mobilitybench_orphan_candidates_query_samples.csv", index=False, encoding="utf-8")
    candidate_orphan_poi.head(50).to_csv(args.report_dir / "mobilitybench_orphan_candidates_poi_samples.csv", index=False, encoding="utf-8")
    qrel_not_in_candidates.head(50).to_csv(args.report_dir / "mobilitybench_qrel_not_in_candidates_samples.csv", index=False, encoding="utf-8")
    abnormal_nearby.head(50).to_csv(args.report_dir / "mobilitybench_nearby_distance_abnormal_samples.csv", index=False, encoding="utf-8")
    split_distribution.to_csv(args.report_dir / "mobilitybench_split_query_type_city_distribution.csv", index=False, encoding="utf-8")

    print("[MobilityBench Quality Summary]")
    print(f"raw episodes = {raw_count_summary.get('raw_episodes', 0)}")
    print(f"processed queries = {len(queries)}")
    print(f"poi_search = {processed_type_counts.get('poi_search', 0)}")
    print(f"nearby_search = {processed_type_counts.get('nearby_search', 0)}")
    print(f"qrels = {len(qrels)}")
    print(f"candidates = {len(candidates)}")
    print(f"qrels query ref valid rate = {1 - len(qrel_orphan_query) / len(qrels) if len(qrels) else 1.0:.6f}")
    print(f"qrels poi ref valid rate = {1 - len(qrel_orphan_poi) / len(qrels) if len(qrels) else 1.0:.6f}")
    print(f"qrel in candidates rate = {qrel_in_candidates_rate:.6f}")
    print(f"query split overlap = {query_split_overlap}")
    print(f"normalized query overlap = {normalized_query_overlap}")
    print(f"nearby distance abnormal count = {len(abnormal_nearby)}")
    print(f"manual check samples = {len(manual)}")


if __name__ == "__main__":
    main()
