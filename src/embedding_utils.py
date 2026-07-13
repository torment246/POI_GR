"""Utilities for POI text embedding construction and validation."""

from __future__ import annotations

import json
import os
import re
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from text_normalize import clean_text


TEXT_FIELDS = ["name", "category_l1", "category_l2", "brand", "tags", "city", "address"]
EXCLUDED_JSON_KEYS = {"location", "lat", "lon", "latitude", "longitude"}
LABELS = {
    "name": "名称",
    "category_l1": "类别",
    "category_l2": "二级类别",
    "brand": "品牌",
    "tags": "标签",
    "city": "城市",
    "address": "地址",
}
OPTIONAL_FIELDS = ["category_l1", "category_l2", "brand", "tags", "city"]


def is_meaningful_text(value: object) -> bool:
    text = clean_text(value)
    if not text or text.casefold() in {"unk", "nan", "none", "null", "<na>"}:
        return False
    if looks_like_structured_location_value(text):
        return False
    return True


def looks_like_structured_location_value(text: str) -> bool:
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        return True
    if re.search(r"['\"]?(user_current_location|current_location|location|latitude|longitude|lat|lon)['\"]?\s*:", stripped, re.I):
        return True
    return bool(re.search(r"\d{2,3}\.\d{4,}\s*,\s*\d{1,2}\.\d{4,}", stripped))


def parse_json_object(value: object) -> dict[str, Any] | None:
    text = clean_text(value)
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError, TypeError):
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def remove_location_keys(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        key_text = str(key)
        if key_text.casefold() in EXCLUDED_JSON_KEYS:
            continue
        if isinstance(value, dict):
            nested = remove_location_keys(value)
            if nested:
                out[key_text] = nested
            continue
        out[key_text] = value
    return out


def row_text_values(row: pd.Series | dict[str, Any]) -> tuple[dict[str, str], bool]:
    parsed = parse_json_object(row.get("poi_json"))
    source = remove_location_keys(parsed) if parsed is not None else row
    used_json = parsed is not None

    values: dict[str, str] = {}
    for field in TEXT_FIELDS:
        value = clean_text(source.get(field))
        if is_meaningful_text(value):
            values[field] = value
        else:
            values[field] = ""
    return values, used_json


def build_embedding_text(row: pd.Series | dict[str, Any]) -> str:
    values, _ = row_text_values(row)
    lines = [f"{LABELS['name']}：{values['name']}"]
    for field in OPTIONAL_FIELDS:
        if values[field]:
            lines.append(f"{LABELS[field]}：{values[field]}")
    lines.append(f"{LABELS['address']}：{values['address']}")
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("embedding_text is empty after construction")
    return text


def embedding_text_has_forbidden_location_tokens(text: object) -> bool:
    lowered = clean_text(text).casefold()
    if not lowered:
        return False
    if re.search(r"['\"]?(user_current_location|current_location|location|latitude|longitude|lat|lon)['\"]?\s*[:：]", lowered):
        return True
    return bool(re.search(r"\d{2,3}\.\d{4,}\s*,\s*\d{1,2}\.\d{4,}", lowered))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sentence_transformer_model(model_path: Path, device: str):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Embedding model path does not exist: {model_path}. "
            "Use --model to specify a local model directory."
        )

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required. Please install sentence-transformers.") from exc

    try:
        return SentenceTransformer(str(model_path), device=device, local_files_only=True)
    except TypeError:
        try:
            return SentenceTransformer(str(model_path), device=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load local embedding model from {model_path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load local embedding model from {model_path}: {exc}") from exc


def set_model_max_length(model: Any, max_length: int | None) -> None:
    if max_length and max_length > 0 and hasattr(model, "max_seq_length"):
        model.max_seq_length = int(max_length)


def l2_norms(embeddings: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embeddings.astype(np.float32, copy=False), axis=1)


def validate_embeddings(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    input_rows: int,
    normalize: bool,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D matrix, got shape={embeddings.shape}")
    if embeddings.shape[0] != len(meta):
        raise ValueError(f"Embedding rows ({embeddings.shape[0]}) != meta rows ({len(meta)})")
    if len(meta) != input_rows:
        raise ValueError(f"Meta rows ({len(meta)}) != input rows ({input_rows})")
    if not meta["poi_id"].is_unique:
        raise ValueError("poi_id must be unique in embedding meta")
    empty_text = meta["embedding_text"].fillna("").astype(str).str.strip().eq("")
    if bool(empty_text.any()):
        raise ValueError(f"embedding_text has empty rows: {int(empty_text.sum())}")
    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN")
    if np.isinf(embeddings).any():
        raise ValueError("Embeddings contain Inf")
    forbidden = meta["embedding_text"].map(embedding_text_has_forbidden_location_tokens)
    if bool(forbidden.any()):
        examples = meta.loc[forbidden, "poi_id"].head(5).tolist()
        raise ValueError(f"embedding_text contains location/lat/lon field names, examples={examples}")
    if normalize:
        mean_norm = float(l2_norms(embeddings).mean())
        if not 0.98 <= mean_norm <= 1.02:
            raise ValueError(f"Normalized embedding mean L2 norm should be near 1, got {mean_norm:.6f}")


def format_rate(count: int | float, total: int | float) -> str:
    if not total:
        return "0.0000"
    return f"{float(count) / float(total):.4f}"
