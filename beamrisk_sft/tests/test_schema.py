import pytest

from beamrisk_sft.errors import BeamRiskError
from beamrisk_sft.schema import RISK_PAIR_SCHEMA_VERSION, RiskPair


def pair_value() -> dict:
    return {
        "schema_version": RISK_PAIR_SCHEMA_VERSION,
        "sample_id": "a" * 64,
        "order_id": "o1",
        "searchid": "s1",
        "target_poi_id": "p1",
        "risk_type": "first_prune",
        "first_prune_depth": 2,
        "prompt_token_ids": [1, 2],
        "positive_token_ids": [3, 4, 5],
        "negative_token_ids": [3, 7, 8],
        "reference_positive_score": -3.0,
        "reference_negative_score": -2.0,
        "reference_margin": 1.0,
        "negative_structure_valid": True,
        "negative_catalog_expandable": False,
        "negative_poi_id": None,
        "strict_duplicate_filtered": False,
    }


def test_pair_schema_accepts_one_mutually_exclusive_risk() -> None:
    pair = RiskPair.from_mapping(pair_value())
    assert pair.risk_type == "first_prune"
    assert pair.first_prune_depth == 2
    assert len(pair.positive_token_ids) == len(pair.negative_token_ids)


def test_pair_schema_rejects_mismatched_paths_and_margin() -> None:
    value = pair_value()
    value["negative_token_ids"] = [3, 7]
    with pytest.raises(BeamRiskError):
        RiskPair.from_mapping(value)
    value = pair_value()
    value["reference_margin"] = 0.5
    with pytest.raises(BeamRiskError):
        RiskPair.from_mapping(value)


def test_final_rank_cannot_carry_prune_depth() -> None:
    value = pair_value()
    value["risk_type"] = "final_rank"
    with pytest.raises(BeamRiskError):
        RiskPair.from_mapping(value)


def test_catalog_expandable_path_must_be_structurally_valid() -> None:
    value = pair_value()
    value["negative_structure_valid"] = False
    value["negative_catalog_expandable"] = True
    value["negative_poi_id"] = "p2"
    with pytest.raises(BeamRiskError, match="Token 结构合法"):
        RiskPair.from_mapping(value)
