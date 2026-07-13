#!/usr/bin/env python3
"""Evaluate a POI retrieval run with Recall@K, MRR@K, and NDCG@K."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poi_genret.evaluation import evaluate_run, rows_to_qrels, rows_to_run


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=ROOT / "data/processed/qrels.csv")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="CSV with query_id,poi_id and optional score or rank columns.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qrels = rows_to_qrels(read_csv_rows(args.qrels))
    run = rows_to_run(read_csv_rows(args.run))
    print(json.dumps(evaluate_run(qrels, run, args.ks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

