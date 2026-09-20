"""Conservative and deterministic query normalization."""

from __future__ import annotations

import re
import unicodedata


NORMALIZATION_VERSION = "conservative_qg_v1"
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "、": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "－": "-",
        "…": "...",
    }
)


class QueryNormalizationError(ValueError):
    """Raised when a Query cannot produce a valid normalized key."""


def normalize_query(query: str) -> str:
    """Normalize surface variants without deleting location-bearing words."""

    if not isinstance(query, str):
        raise QueryNormalizationError("Query 必须是字符串")
    normalized = unicodedata.normalize("NFKC", query)
    normalized = normalized.lower().translate(_PUNCTUATION_TRANSLATION)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if not normalized:
        raise QueryNormalizationError("归一化后的 Query 为空")
    return normalized
