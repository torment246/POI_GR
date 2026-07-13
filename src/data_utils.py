"""Small data-processing utilities for POI SID training table construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_columns(df: pd.DataFrame, columns: list[str], default: str = "") -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = default
    return df


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML rules must be a mapping: {path}")
    return dict(data)


def write_yaml_if_missing(path: Path, data: Mapping[str, Any]) -> None:
    if path.exists():
        return
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, allow_unicode=True, sort_keys=False)


def format_rate(count: int | float, total: int | float) -> str:
    if not total:
        return "0.0000"
    return f"{float(count) / float(total):.4f}"


def value_counts_text(series: pd.Series, limit: int = 50) -> str:
    counts = series.fillna("").astype(str).value_counts(dropna=False).head(limit)
    if counts.empty:
        return "- none"
    return "\n".join(f"- {idx or '<empty>'}: {int(count)}" for idx, count in counts.items())


def joined_examples(names: pd.Series, limit: int = 5) -> str:
    examples = []
    seen = set()
    for value in names.fillna("").astype(str):
        value = value.strip()
        if not value or value in seen:
            continue
        examples.append(value)
        seen.add(value)
        if len(examples) >= limit:
            break
    return " | ".join(examples)
