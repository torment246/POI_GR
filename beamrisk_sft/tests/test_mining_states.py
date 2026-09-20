import json
from pathlib import Path

import pytest

from beamrisk_sft.errors import BeamRiskError
from beamrisk_sft.mining import MINING_SCHEMA_VERSION, _load_state_map


def _state(sample_id: str, value: str) -> dict:
    return {
        "schema_version": MINING_SCHEMA_VERSION,
        "sample_id": sample_id,
        "state": value,
    }


def test_state_map_counts_disjoint_rank_files(tmp_path: Path) -> None:
    paths = [tmp_path / "rank0.jsonl", tmp_path / "rank1.jsonl"]
    paths[0].write_text(
        json.dumps(_state("a" * 64, "miss@10")) + "\n",
        encoding="utf-8",
    )
    paths[1].write_text(
        json.dumps(_state("b" * 64, "hit@1")) + "\n",
        encoding="utf-8",
    )
    states, counts = _load_state_map(paths, expected_rows=2)
    assert states == {"a" * 64: "miss@10", "b" * 64: "hit@1"}
    assert counts == {"miss@10": 1, "hit@1": 1}


def test_state_map_rejects_cross_rank_duplicate(tmp_path: Path) -> None:
    paths = [tmp_path / "rank0.jsonl", tmp_path / "rank1.jsonl"]
    payload = json.dumps(_state("a" * 64, "miss@10")) + "\n"
    for path in paths:
        path.write_text(payload, encoding="utf-8")
    with pytest.raises(BeamRiskError, match="重复"):
        _load_state_map(paths, expected_rows=2)
