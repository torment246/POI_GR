from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from beamrisk_sft.catalog import build_tiger_key_to_poi
from beamrisk_sft.errors import BeamRiskError


def test_tiger_key_lookup_uses_current_pyarrow_contract(tmp_path: Path) -> None:
    path = tmp_path / "mapping.parquet"
    pq.write_table(
        pa.table(
            {
                "poi_id": ["p1", "p2"],
                "tiger_id_key": ["1-2-3|c0", "4-5-6|c1"],
            }
        ),
        path,
    )
    assert build_tiger_key_to_poi(path, {"4-5-6|c1"}) == {
        "4-5-6|c1": "p2"
    }


def test_tiger_key_lookup_rejects_missing_path(tmp_path: Path) -> None:
    path = tmp_path / "mapping.parquet"
    pq.write_table(
        pa.table({"poi_id": ["p1"], "tiger_id_key": ["1-2-3|c0"]}),
        path,
    )
    with pytest.raises(BeamRiskError, match="缺少候选路径"):
        build_tiger_key_to_poi(path, {"9-9-9|c9"})


def test_tiger_key_lookup_can_return_only_catalog_expandable_paths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mapping.parquet"
    pq.write_table(
        pa.table({"poi_id": ["p1"], "tiger_id_key": ["1-2-3|c0"]}),
        path,
    )
    assert build_tiger_key_to_poi(
        path,
        {"1-2-3|c0", "9-9-9|c0"},
        require_all=False,
    ) == {"1-2-3|c0": "p1"}
