"""Define the GenPOI proximity-level classification and SSP data contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CURRENT_PATTERN = re.compile(r"<CURRENT>(.*?)</CURRENT>", re.DOTALL)
QUERY_PATTERN = re.compile(r"<QUERY>(.*?)</QUERY>", re.DOTALL)
USER_GID_PATTERN = re.compile(r"<USER_GID>(.*?)</USER_GID>", re.DOTALL)
GID_TOKEN_PATTERN = re.compile(r"<G_([0-9bcdefghjkmnpqrstuvwxyz])>")


class GenPoiProximityError(ValueError):
    """Raised when a GenPOI sample cannot define a proximity label."""


@dataclass(frozen=True)
class ProximityExample:
    query: str
    user_gid: tuple[str, ...]
    target_gid: tuple[str, ...]
    label: int
    sample_id: str


def common_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Return the number of equal leading tokens."""

    return next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )


def effective_prefix_length(proximity_level: int, *, gamma: int = 2) -> int:
    """Return the number of user GID tokens prefilled by GenPOI SSP."""

    if not 0 <= proximity_level <= 6:
        raise GenPoiProximityError("proximity_level 必须位于 [0, 6]")
    if gamma < 0:
        raise GenPoiProximityError("gamma 不得为负数")
    return max(proximity_level - gamma, 0)


def _role_content(record: Mapping[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise GenPoiProximityError("messages 必须是 list")
    matches = [
        item.get("content")
        for item in messages
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise GenPoiProximityError(f"messages 必须恰好包含一个 {role} content")
    return matches[0]


def extract_current_query_and_gid(user_content: str) -> tuple[str, tuple[str, ...]]:
    """Extract only inference-available Query and user GID from CURRENT."""

    current_matches = CURRENT_PATTERN.findall(user_content)
    if len(current_matches) != 1:
        raise GenPoiProximityError("user content 必须恰好包含一个 CURRENT 区块")
    current = current_matches[0]
    query_matches = QUERY_PATTERN.findall(current)
    user_gid_matches = USER_GID_PATTERN.findall(current)
    if len(query_matches) != 1 or len(user_gid_matches) != 1:
        raise GenPoiProximityError("CURRENT 必须恰好包含一个 QUERY 和 USER_GID")
    query = query_matches[0].strip()
    if not query:
        raise GenPoiProximityError("CURRENT QUERY 不能为空")
    user_gid = tuple(GID_TOKEN_PATTERN.findall(user_gid_matches[0]))
    if len(user_gid) != 6:
        raise GenPoiProximityError("用户 GID 必须恰好包含 6 个 Token")
    return query, user_gid


def extract_proximity_example(
    record: Mapping[str, Any],
    *,
    expected_split: str | None = None,
) -> ProximityExample:
    """Extract query and a 0..6 GID common-prefix label from one SFT sample."""

    split = record.get("split")
    if expected_split is not None and split != expected_split:
        raise GenPoiProximityError(
            f"样本 split {split!r} 与预期 {expected_split!r} 不一致"
        )
    user_content = _role_content(record, "user")
    assistant_content = _role_content(record, "assistant")
    query, user_gid = extract_current_query_and_gid(user_content)
    target_gid = tuple(GID_TOKEN_PATTERN.findall(assistant_content)[:6])
    if len(target_gid) != 6:
        raise GenPoiProximityError("目标 GID 必须恰好包含 6 个 Token")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise GenPoiProximityError("sample_id 必须是非空字符串")
    return ProximityExample(
        query=query,
        user_gid=user_gid,
        target_gid=target_gid,
        label=common_prefix_length(user_gid, target_gid),
        sample_id=sample_id,
    )


def prefilled_gid_tokens(
    user_gid: Sequence[str],
    proximity_level: int,
    *,
    gamma: int = 2,
) -> tuple[str, ...]:
    """Return serialized GID tokens used to prefill constrained decoding."""

    if len(user_gid) != 6:
        raise GenPoiProximityError("user_gid 必须包含 6 个 Geohash 字符")
    prefix_length = effective_prefix_length(proximity_level, gamma=gamma)
    return tuple(f"<G_{value}>" for value in user_gid[:prefix_length])
