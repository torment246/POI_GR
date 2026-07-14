#!/usr/bin/env python3
"""Process LLM4POI sub-datasets as independent next-POI retrieval datasets."""

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
    assign_split_by_order,
    clean,
    compact_json,
    log,
    parse_ratio,
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


DATASET_MAP = {
    "nyc": ("Foursquare-NYC", "NYC"),
    "foursquare-nyc": ("Foursquare-NYC", "NYC"),
    "tky": ("Foursquare-TKY", "Tokyo"),
    "foursquare-tky": ("Foursquare-TKY", "Tokyo"),
    "ca": ("Gowalla-CA", "California"),
    "gowalla-ca": ("Gowalla-CA", "California"),
}


def find_dataset_files(raw_dir: Path) -> list[tuple[str, str, Path]]:
    base = raw_dir / "LLM4POI"
    if not base.exists():
        log("process_llm4poi", f"[WARN] raw directory not found: {base}")
        return []
    found: list[tuple[str, str, Path]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        key = child.name.lower()
        if key not in DATASET_MAP:
            continue
        dataset_name, city = DATASET_MAP[key]
        candidates = sorted(child.glob("**/*.csv"))
        train_like = [p for p in candidates if "train_sample" in p.name.lower()]
        selected = train_like[0] if train_like else (candidates[0] if candidates else None)
        if selected is None:
            log("process_llm4poi", f"[WARN] no CSV found for {dataset_name}: {child}")
            continue
        found.append((dataset_name, city, selected))
    return found


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def process_one(dataset_name: str, city: str, path: Path, output_root: Path, split_ratio_text: str, seed: int, max_rows: int | None) -> dict[str, Any]:
    split_ratio = parse_ratio(split_ratio_text)
    df = pd.read_csv(path, nrows=max_rows)
    log("process_llm4poi", f"{dataset_name}: columns={list(df.columns)}")
    log("process_llm4poi", f"{dataset_name}: read {len(df)} rows from {path}")

    user_col = first_existing(df, ["UserId", "user_id"])
    poi_col = first_existing(df, ["PoiId", "poi_id"])
    cat_col = first_existing(df, ["PoiCategoryName", "category", "category_name"])
    cat_id_col = first_existing(df, ["PoiCategoryId", "category_id"])
    lat_col = first_existing(df, ["Latitude", "lat", "latitude"])
    lon_col = first_existing(df, ["Longitude", "lon", "longitude", "lng"])
    ts_col = first_existing(df, ["UTCTimeOffsetEpoch", "timestamp", "time"])
    traj_col = first_existing(df, ["trajectory_id", "pseudo_session_trajectory_id", "session_id"])
    required = [user_col, poi_col, cat_col, lat_col, lon_col, ts_col]
    if any(col is None for col in required):
        missing = [name for name, col in zip(["user", "poi", "category", "lat", "lon", "timestamp"], required) if col is None]
        raise ValueError(f"{dataset_name} missing required fields: {missing}")

    df = df.copy()
    if traj_col is None:
        traj_col = "_trajectory_id"
        df[traj_col] = df[user_col].astype(str)
    df["_raw_poi_id"] = df[poi_col].astype(str)
    df["_poi_id"] = "llm4poi:" + dataset_name + ":" + df["_raw_poi_id"]
    df["_user_id"] = "llm4poi:" + dataset_name + ":" + df[user_col].astype(str)
    df["_trajectory_id"] = dataset_name + ":" + df[traj_col].astype(str)
    if ts_col == "UTCTimeOffsetEpoch":
        df["_timestamp"] = pd.to_datetime(df[ts_col], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df["_ts_sort"] = pd.to_numeric(df[ts_col], errors="coerce")
    else:
        parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df["_timestamp"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna(df[ts_col].astype(str))
        df["_ts_sort"] = parsed.astype("int64", errors="ignore")

    pois = (
        df.sort_values(["_raw_poi_id", "_ts_sort"])
        .groupby("_raw_poi_id", as_index=False)
        .agg(
            poi_id=("_poi_id", "first"),
            raw_id=("_raw_poi_id", "first"),
            lat=(lat_col, "median"),
            lon=(lon_col, "median"),
            category_l1=(cat_col, "first"),
            category_l2=(cat_id_col if cat_id_col else cat_col, "first"),
        )
    )
    pois["name"] = ""
    pois["address"] = ""
    pois["city"] = city
    pois["district"] = ""
    pois["business_area"] = ""
    pois["brand"] = ""
    pois["tags"] = pois["category_l1"]
    pois["rating"] = ""
    pois["price"] = ""
    pois["open_hours"] = ""
    pois["source"] = "llm4poi"
    pois = pois[POI_COLUMNS]

    df = df.sort_values(["_user_id", "_trajectory_id", "_ts_sort"]).reset_index(drop=True)
    interactions = pd.DataFrame(
        {
            "interaction_id": [
                "llm4poi_i_" + stable_hash(f"{dataset_name}|{i}|{poi_id}|{timestamp}", 18)
                for i, (poi_id, timestamp) in enumerate(zip(df["_poi_id"], df["_timestamp"]))
            ],
            "user_id": df["_user_id"],
            "poi_id": df["_poi_id"],
            "timestamp": df["_timestamp"],
            "lat": df[lat_col],
            "lon": df[lon_col],
            "action_type": "checkin",
            "trajectory_id": df["_trajectory_id"],
            "source": "llm4poi",
        }
    )

    query_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    for trajectory_id, group in df.groupby("_trajectory_id", sort=False):
        records = group.to_dict("records")
        sample_indices = list(range(1, len(records)))
        split_by_pos = dict(zip(sample_indices, assign_split_by_order(len(sample_indices), split_ratio)))
        for pos in sample_indices:
            history = records[max(0, pos - 5) : pos]
            target = records[pos]
            history_pois = [clean(r.get("_poi_id")) for r in history]
            history_categories = [clean(r.get(cat_col)) for r in history]
            history_times = [clean(r.get("_timestamp")) for r in history]
            query_id = "llm4poi_q_" + stable_hash(f"{dataset_name}|{trajectory_id}|{pos}|{target.get('_poi_id')}", 18)
            extra_context = {
                "dataset": dataset_name,
                "trajectory_id": trajectory_id,
                "history_poi_ids": history_pois,
                "history_categories": history_categories,
                "history_timestamps": history_times,
            }
            query_rows.append(
                {
                    "query_id": query_id,
                    "query_text": "next POI after visiting categories: " + ", ".join([c for c in history_categories if c][-3:]),
                    "city": city,
                    "user_lat": clean(history[-1].get(lat_col)) if history else "",
                    "user_lon": clean(history[-1].get(lon_col)) if history else "",
                    "timestamp": clean(target.get("_timestamp")),
                    "query_type": "next_poi",
                    "source": "llm4poi",
                    "raw_query_id": f"{trajectory_id}:{pos}",
                    "extra_context": compact_json(extra_context),
                    "split": split_by_pos[pos],
                }
            )
            qrel_rows.append({"query_id": query_id, "poi_id": clean(target.get("_poi_id")), "label": 1, "source": "llm4poi"})

    queries = pd.DataFrame(query_rows)
    qrels = pd.DataFrame(qrel_rows, columns=QREL_COLUMNS)
    output_dir = output_root / "llm4poi" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pois, output_dir / "pois.csv", POI_COLUMNS)
    write_csv(interactions, output_dir / "interactions.csv", INTERACTION_COLUMNS)
    write_csv(queries.drop(columns=["split"], errors="ignore"), output_dir / "queries.csv", QUERY_COLUMNS)
    write_csv(qrels, output_dir / "qrels.csv", QREL_COLUMNS)
    write_csv(pd.DataFrame(columns=MOBILITY_CANDIDATE_COLUMNS), output_dir / "candidates.csv", MOBILITY_CANDIDATE_COLUMNS)
    counts = write_split_files(output_dir, queries, qrels, QUERY_COLUMNS, QREL_COLUMNS)
    write_manifest(
        output_dir,
        dataset_name,
        "llm4poi",
        "time_order_within_trajectory",
        split_ratio_text,
        seed,
        counts,
        False,
        "Only train_sample-style files were available; split was built by temporal order inside each trajectory.",
    )

    summary = {
        "dataset_name": dataset_name,
        "source": "llm4poi",
        "scenario": "next POI / trajectory-based retrieval",
        "pois": len(pois),
        "users": int(df["_user_id"].nunique()),
        "interactions": len(interactions),
        "queries": len(queries),
        "qrels": len(qrels),
        "categories": int(pois["category_l1"].nunique()),
        "lat_min": float(pd.to_numeric(pois["lat"], errors="coerce").min()),
        "lat_max": float(pd.to_numeric(pois["lat"], errors="coerce").max()),
        "lon_min": float(pd.to_numeric(pois["lon"], errors="coerce").min()),
        "lon_max": float(pd.to_numeric(pois["lon"], errors="coerce").max()),
        "avg_trajectory_length": float(df.groupby("_trajectory_id").size().mean()),
        "split": counts,
        "whether_original_split_used": False,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("process_llm4poi", f"{dataset_name}: write pois={len(pois)} interactions={len(interactions)} queries={len(queries)}")
    log("process_llm4poi", f"{dataset_name}: train/dev/test={counts['train_queries']}/{counts['dev_queries']}/{counts['test_queries']}")
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
    summaries = []
    for dataset_name, city, path in find_dataset_files(args.raw_dir):
        summaries.append(process_one(dataset_name, city, path, args.output_dir, args.split_ratio, args.seed, args.max_rows))
    print(json.dumps({"datasets": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
