#!/usr/bin/env python3
"""Run the full data preparation pipeline for all MVP datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from poi_genret.schema import QREL_COLUMNS, QUERY_COLUMNS, read_csv_if_exists, write_csv  # noqa: E402


def log(message: str) -> None:
    print(f"[prepare_all_datasets] {message}")


def run_step(args: list[str]) -> None:
    log("running: " + " ".join(args))
    subprocess.run(args, check=True)


def write_splits(output_dir: Path, seed: int) -> dict[str, Any]:
    queries = read_csv_if_exists(output_dir / "queries.csv", QUERY_COLUMNS)
    qrels = read_csv_if_exists(output_dir / "qrels.csv", QREL_COLUMNS)
    if queries.empty:
        manifest = {"error": "queries.csv is empty"}
        (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    train_dev = queries[queries["source"].isin(["llm4poi", "yelp"])].copy()
    test = queries[queries["source"] == "mobilitybench"].copy()
    dev_query_ids = set(train_dev["query_id"].drop_duplicates().sample(frac=0.1, random_state=seed).tolist()) if len(train_dev) else set()

    dev_queries = train_dev[train_dev["query_id"].isin(dev_query_ids)].copy()
    train_queries = train_dev[~train_dev["query_id"].isin(dev_query_ids)].copy()
    test_query_ids = set(test["query_id"].tolist())
    qrel_query_ids = set(qrels["query_id"].tolist()) if not qrels.empty else set()
    test_retrieval_query_ids = test_query_ids.intersection(qrel_query_ids)
    test_tool_intent_query_ids = test_query_ids.difference(qrel_query_ids)

    train_qrels = qrels[qrels["query_id"].isin(set(train_queries["query_id"]))].copy() if not qrels.empty else pd.DataFrame(columns=QREL_COLUMNS)
    dev_qrels = qrels[qrels["query_id"].isin(dev_query_ids)].copy() if not qrels.empty else pd.DataFrame(columns=QREL_COLUMNS)
    test_qrels = qrels[qrels["query_id"].isin(test_query_ids)].copy() if not qrels.empty else pd.DataFrame(columns=QREL_COLUMNS)

    write_csv(train_queries, output_dir / "train_queries.csv", QUERY_COLUMNS)
    write_csv(dev_queries, output_dir / "dev_queries.csv", QUERY_COLUMNS)
    write_csv(test, output_dir / "test_mobilitybench_queries.csv", QUERY_COLUMNS)
    write_csv(train_qrels, output_dir / "train_qrels.csv", QREL_COLUMNS)
    write_csv(dev_qrels, output_dir / "dev_qrels.csv", QREL_COLUMNS)
    write_csv(test_qrels, output_dir / "test_mobilitybench_qrels.csv", QREL_COLUMNS)

    manifest = {
        "seed": seed,
        "train": {
            "description": "Yelp template queries and LLM4POI behavior/template queries.",
            "queries": len(train_queries),
            "qrels": len(train_qrels),
            "sources": sorted(train_queries["source"].unique().tolist()) if len(train_queries) else [],
        },
        "dev": {
            "description": "10% random query_id sample from Yelp and LLM4POI.",
            "queries": len(dev_queries),
            "qrels": len(dev_qrels),
            "sources": sorted(dev_queries["source"].unique().tolist()) if len(dev_queries) else [],
        },
        "test": {
            "description": "MobilityBench POI Search and Nearby Search.",
            "queries": len(test),
            "qrels": len(test_qrels),
            "sources": sorted(test["source"].unique().tolist()) if len(test) else [],
            "test_retrieval": {
                "description": "MobilityBench queries with explicit parsed qrels.",
                "queries": len(test_retrieval_query_ids),
            },
            "test_tool_intent": {
                "description": "MobilityBench queries with tool/candidate evidence but no unique qrel.",
                "queries": len(test_tool_intent_query_ids),
            },
        },
        "files": {
            "train_queries": str(output_dir / "train_queries.csv"),
            "dev_queries": str(output_dir / "dev_queries.csv"),
            "test_mobilitybench_queries": str(output_dir / "test_mobilitybench_queries.csv"),
            "train_qrels": str(output_dir / "train_qrels.csv"),
            "dev_qrels": str(output_dir / "dev_qrels.csv"),
            "test_mobilitybench_qrels": str(output_dir / "test_mobilitybench_qrels.csv"),
        },
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/data_report")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = ["--raw-dir", str(args.raw_dir), "--output-dir", str(args.output_dir), "--report-dir", str(args.report_dir), "--seed", str(args.seed)]
    max_rows = ["--max-rows", str(args.max_rows)] if args.max_rows is not None else []

    run_step([sys.executable, str(ROOT / "scripts/data/prepare_mvp_data.py"), *common, *max_rows])
    run_step([sys.executable, str(ROOT / "scripts/data/prepare_mobilitybench_data.py"), *common, *max_rows])
    manifest = write_splits(args.output_dir, args.seed)
    run_step([sys.executable, str(ROOT / "scripts/data/analyze_datasets.py"), *common, *max_rows])

    report_path = args.report_dir / "dataset_report.md"
    summary = {
        "split_manifest": manifest,
        "report": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
