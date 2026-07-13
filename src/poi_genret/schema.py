"""Shared schemas for the POI generative retrieval data layer."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


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

INTERACTION_COLUMNS = [
    "interaction_id",
    "user_id",
    "poi_id",
    "timestamp",
    "lat",
    "lon",
    "action_type",
    "trajectory_id",
    "source",
]

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

MOBILITY_CANDIDATE_COLUMNS = [
    "query_id",
    "poi_id",
    "rank",
    "score",
    "candidate_source",
    "source",
]


def write_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df.reindex(columns=columns)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def read_csv_if_exists(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path, dtype=str, keep_default_na=False)
