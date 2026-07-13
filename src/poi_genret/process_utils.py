"""Shared helpers for dataset-only processing. No semantic ID logic lives here."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def compact_json(value: Any) -> str:
    if value in (None, "", {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return compact_json(value)


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
    if abs(a) <= 90 and abs(b) > 90:
        lat, lon = a, b
    else:
        lon, lat = a, b
    return f"{lat:.6f}", f"{lon:.6f}"


def parse_ratio(text: str) -> tuple[float, float, float]:
    parts = [float(part) for part in re.split(r"[:,/]", text) if part.strip()]
    if len(parts) != 3:
        raise ValueError("--split-ratio must have three values, e.g. 0.8,0.1,0.1")
    total = sum(parts)
    if total <= 0:
        raise ValueError("--split-ratio values must be positive")
    return parts[0] / total, parts[1] / total, parts[2] / total


def assign_split_by_order(n: int, split_ratio: tuple[float, float, float]) -> list[str]:
    train_ratio, dev_ratio, _ = split_ratio
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    if n >= 3:
        train_end = max(1, min(train_end, n - 2))
        dev_end = max(train_end + 1, min(dev_end, n - 1))
    return ["train" if i < train_end else "dev" if i < dev_end else "test" for i in range(n)]


def split_keys_random(keys: list[str], split_ratio: tuple[float, float, float], seed: int) -> dict[str, str]:
    unique = pd.Series(keys).drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    splits = assign_split_by_order(len(unique), split_ratio)
    return dict(zip(unique, splits))


def write_split_files(
    output_dir: Path,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    query_columns: list[str],
    qrel_columns: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ["train", "dev", "test"]:
        split_queries = queries[queries["split"] == split].drop(columns=["split"], errors="ignore")
        query_ids = set(split_queries["query_id"].tolist()) if "query_id" in split_queries.columns else set()
        split_qrels = qrels[qrels["query_id"].isin(query_ids)] if not qrels.empty else pd.DataFrame(columns=qrel_columns)
        split_queries.reindex(columns=query_columns).to_csv(output_dir / f"{split}_queries.csv", index=False, encoding="utf-8")
        split_qrels.reindex(columns=qrel_columns).to_csv(output_dir / f"{split}_qrels.csv", index=False, encoding="utf-8")
        counts[f"{split}_queries"] = len(split_queries)
        counts[f"{split}_qrels"] = len(split_qrels)
    return counts


def write_manifest(
    output_dir: Path,
    dataset_name: str,
    source: str,
    split_method: str,
    split_ratio: str,
    seed: int,
    counts: dict[str, int],
    original_split_used: bool,
    notes: str,
) -> None:
    manifest = {
        "dataset_name": dataset_name,
        "source": source,
        "split_method": split_method,
        "split_ratio": split_ratio,
        "seed": seed,
        "train_count": counts.get("train_queries", 0),
        "dev_count": counts.get("dev_queries", 0),
        "test_count": counts.get("test_queries", 0),
        "train_qrels": counts.get("train_qrels", 0),
        "dev_qrels": counts.get("dev_qrels", 0),
        "test_qrels": counts.get("test_qrels", 0),
        "whether_original_split_used": original_split_used,
        "notes": notes,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

