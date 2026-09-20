from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from beamrisk_sft.config import load_config
from beamrisk_sft.io import iter_jsonl, sha256_file
from beamrisk_sft.mining import (
    MINING_SCHEMA_VERSION,
    _checkpoint_identity,
    finalize_mined_parts,
)
from beamrisk_sft.schema import RISK_PAIR_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[2]


def _raw_final_rank_pair(rank: int, negative_key: str) -> dict:
    return {
        "schema_version": RISK_PAIR_SCHEMA_VERSION,
        "sample_id": f"{rank + 1:064x}",
        "order_id": f"order-{rank}",
        "searchid": f"search-{rank}",
        "target_poi_id": f"target-{rank}",
        "risk_type": "final_rank",
        "first_prune_depth": None,
        "gold_rank_at_depth": 2,
        "boundary_rank": 1,
        "prompt_token_ids": [1, 2],
        "positive_token_ids": [3, 4],
        "negative_token_ids": [5, 6],
        "reference_positive_score": -3.0,
        "reference_negative_score": -2.0,
        "reference_margin": 1.0,
        "negative_structure_valid": True,
        "negative_catalog_expandable": False,
        "negative_poi_id": None,
        "negative_tiger_id_key": negative_key,
        "strict_duplicate_filtered": False,
    }


def test_finalizer_keeps_structurally_valid_unmapped_generated_paths(
    tmp_path: Path,
) -> None:
    original = load_config(ROOT / "beamrisk_sft" / "configs" / "main_v1.yaml")
    mapping = tmp_path / "identifier" / "mapping.parquet"
    mapping.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "poi_id": ["negative-0"],
                "tiger_id_key": ["1-2-3|c0"],
            }
        ),
        mapping,
    )
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    signatures = [
        {
            "poi_id": "target-0",
            "displayname": "目标",
            "address": "目标地址",
            "alias": "",
            "lng": 116.1,
            "lat": 39.1,
        },
        {
            "poi_id": "negative-0",
            "displayname": "负例",
            "address": "负例地址",
            "alias": "",
            "lng": 116.2,
            "lat": 39.2,
        },
    ]
    (catalog / "part-000.json").write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in signatures),
        encoding="utf-8",
    )
    mining_root = tmp_path / "mining"
    config = replace(
        original,
        paths=replace(
            original.paths,
            tiger_identifier_dir=mapping.parent,
            tiger_mapping=mapping,
            poi_catalog_dir=catalog,
            mining_root=mining_root,
        ),
        mining=replace(
            original.mining,
            target_pair_count=4,
            minimum_pair_count=1,
            per_rank_pair_cap=1,
        ),
    )
    checkpoint = tmp_path / "checkpoint-5571"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"synthetic model")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"epoch": 1.0, "global_step": 5_571}),
        encoding="utf-8",
    )
    checkpoint_identity = _checkpoint_identity(checkpoint)
    parts_dir = config.mining_dir(1) / "parts"
    parts_dir.mkdir(parents=True)
    for rank in range(4):
        negative_key = "1-2-3|c0" if rank == 0 else f"9-9-{rank}|c0"
        pair_path = parts_dir / f"risk_pairs_rank_{rank:03d}.jsonl"
        state_path = parts_dir / f"states_rank_{rank:03d}.jsonl"
        pair_path.write_text(
            json.dumps(_raw_final_rank_pair(rank, negative_key)) + "\n",
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": MINING_SCHEMA_VERSION,
                    "sample_id": f"{rank + 1:064x}",
                    "state": "hit@10_not@1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (parts_dir / f"manifest_rank_{rank:03d}.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "reference_checkpoint": checkpoint_identity,
                    "output": str(pair_path),
                    "output_sha256": sha256_file(pair_path),
                    "states_output": str(state_path),
                    "states_rows": 1,
                    "states_sha256": sha256_file(state_path),
                }
            ),
            encoding="utf-8",
        )

    result = finalize_mined_parts(
        config,
        reference_epoch=1,
        checkpoint=checkpoint,
        resume=False,
    )
    assert result["pairs"] == 4
    assert result["catalog_expandable_final_rank_negatives"] == 1
    assert result["catalog_unexpandable_final_rank_negatives"] == 3
    assert result["strict_duplicate_filtered"] == 0
    rows = [value for _, value in iter_jsonl(config.mining_dir(1) / "risk_pairs.jsonl")]
    assert sum(bool(value["negative_catalog_expandable"]) for value in rows) == 1
    assert sum(value["negative_poi_id"] is None for value in rows) == 3

