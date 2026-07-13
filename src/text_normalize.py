"""Text normalization helpers for POI cleaning."""

from __future__ import annotations

import math
import re
from typing import Any


EMPTY_TEXT_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "<na>",
    "n/a",
    "na",
    "nil",
    "无",
    "暂无",
    "未知",
    "未填写",
    "空",
    "unknown",
}


def normalize_space(text: str) -> str:
    """Trim text and collapse consecutive whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(x: object) -> str:
    """Convert obvious null-like values to empty strings and normalize spaces."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except TypeError:
        pass

    text = normalize_space(str(x))
    if text.casefold() in EMPTY_TEXT_VALUES:
        return ""
    return text


def is_empty_text(text: object) -> bool:
    """Return True when a value is empty after POI text normalization."""
    return clean_text(text) == ""
