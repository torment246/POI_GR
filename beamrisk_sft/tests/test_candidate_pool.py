import hashlib
import json
from pathlib import Path

import pytest

from beamrisk_sft.candidate_pool import _scan_smallest_hash_lines
from beamrisk_sft.errors import BeamRiskError


def _write_source(path: Path, sample_ids: list[str]) -> str:
    payload = b"".join(
        (
            json.dumps(
                {
                    "sample_id": sample_id,
                    "order_id": f"o{index}",
                    "searchid": f"s{index}",
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for index, sample_id in enumerate(sample_ids)
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_candidate_scan_selects_exact_smallest_ids_and_checks_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    digest = _write_source(
        source,
        [f"{value:064x}" for value in (9, 2, 7, 1, 5)],
    )
    selected = _scan_smallest_hash_lines(
        source,
        expected_rows=5,
        expected_sha256=digest,
        pool_size=3,
    )
    assert [item.sample_id for item in selected] == [
        f"{value:064x}" for value in (1, 2, 5)
    ]


def test_candidate_scan_rejects_manifest_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write_source(source, [f"{value:064x}" for value in (1, 2)])
    with pytest.raises(BeamRiskError, match="SHA256"):
        _scan_smallest_hash_lines(
            source,
            expected_rows=2,
            expected_sha256="0" * 64,
            pool_size=1,
        )
