"""Risk-pair schema and structural validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import BeamRiskError


RISK_PAIR_SCHEMA_VERSION = "beamrisk-pair-v1"
RISK_TYPES = frozenset(("first_prune", "final_rank"))


def business_key(order_id: Any, searchid: Any) -> str:
    if not isinstance(order_id, str) or not order_id:
        raise BeamRiskError("order_id 必须是非空字符串")
    if not isinstance(searchid, str) or not searchid:
        raise BeamRiskError("searchid 必须是非空字符串")
    return f"{order_id}\x1f{searchid}"


def _token_list(value: Any, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in value
        )
    ):
        raise BeamRiskError(f"{name} 必须是非空、非负整数 Token 列表")
    return tuple(int(token) for token in value)


@dataclass(frozen=True)
class RiskPair:
    sample_id: str
    order_id: str
    searchid: str
    target_poi_id: str
    risk_type: str
    first_prune_depth: int | None
    prompt_token_ids: tuple[int, ...]
    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]
    reference_positive_score: float
    reference_negative_score: float
    reference_margin: float
    negative_structure_valid: bool
    negative_catalog_expandable: bool
    negative_poi_id: str | None
    strict_duplicate_filtered: bool

    @property
    def key(self) -> str:
        return business_key(self.order_id, self.searchid)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RiskPair":
        if value.get("schema_version") != RISK_PAIR_SCHEMA_VERSION:
            raise BeamRiskError("risk pair schema_version 无效")
        risk_type = value.get("risk_type")
        if risk_type not in RISK_TYPES:
            raise BeamRiskError(f"risk_type 无效：{risk_type}")
        depth = value.get("first_prune_depth")
        if risk_type == "first_prune":
            if not isinstance(depth, int) or isinstance(depth, bool) or depth not in (1, 2, 3, 4):
                raise BeamRiskError("first_prune 的深度必须是 1/2/3/4")
        elif depth is not None:
            raise BeamRiskError("final_rank 的 first_prune_depth 必须为空")
        prompt = _token_list(value.get("prompt_token_ids"), "prompt_token_ids")
        positive = _token_list(value.get("positive_token_ids"), "positive_token_ids")
        negative = _token_list(value.get("negative_token_ids"), "negative_token_ids")
        if len(positive) != len(negative):
            raise BeamRiskError("正负路径 Token 长度必须相同")
        scores: list[float] = []
        for name in (
            "reference_positive_score",
            "reference_negative_score",
            "reference_margin",
        ):
            raw = value.get(name)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise BeamRiskError(f"{name} 必须是有限数值")
            score = float(raw)
            if not math.isfinite(score):
                raise BeamRiskError(f"{name} 必须是有限数值")
            scores.append(score)
        if not math.isclose(scores[2], scores[1] - scores[0], abs_tol=2e-4):
            raise BeamRiskError("reference_margin 必须等于 negative-positive")
        strings: dict[str, str] = {}
        for name in ("sample_id", "order_id", "searchid", "target_poi_id"):
            raw = value.get(name)
            if not isinstance(raw, str) or not raw:
                raise BeamRiskError(f"{name} 必须是非空字符串")
            strings[name] = raw
        bools: dict[str, bool] = {}
        for name in (
            "negative_structure_valid",
            "negative_catalog_expandable",
            "strict_duplicate_filtered",
        ):
            raw = value.get(name)
            if not isinstance(raw, bool):
                raise BeamRiskError(f"{name} 必须是 bool")
            bools[name] = raw
        negative_poi_id = value.get("negative_poi_id")
        if negative_poi_id is not None and (
            not isinstance(negative_poi_id, str) or not negative_poi_id
        ):
            raise BeamRiskError("negative_poi_id 必须为空或非空字符串")
        if bools["negative_catalog_expandable"] != (negative_poi_id is not None):
            raise BeamRiskError("negative_catalog_expandable 与 negative_poi_id 不一致")
        if bools["negative_catalog_expandable"] and not bools["negative_structure_valid"]:
            raise BeamRiskError("目录可展开的负路径必须先满足 Token 结构合法")
        return cls(
            **strings,
            risk_type=risk_type,
            first_prune_depth=depth,
            prompt_token_ids=prompt,
            positive_token_ids=positive,
            negative_token_ids=negative,
            reference_positive_score=scores[0],
            reference_negative_score=scores[1],
            reference_margin=scores[2],
            negative_structure_valid=bools["negative_structure_valid"],
            negative_catalog_expandable=bools["negative_catalog_expandable"],
            negative_poi_id=negative_poi_id,
            strict_duplicate_filtered=bools["strict_duplicate_filtered"],
        )
