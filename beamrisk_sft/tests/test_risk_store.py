from pathlib import Path

from datasets import Dataset

from beamrisk_sft.risk_store import RiskPairStore
from beamrisk_sft.schema import RISK_PAIR_SCHEMA_VERSION


def record(index: int) -> dict:
    return {
        "schema_version": RISK_PAIR_SCHEMA_VERSION,
        "sample_id": f"{index:064x}",
        "order_id": f"o{index}",
        "searchid": f"s{index}",
        "target_poi_id": f"p{index}",
        "risk_type": "first_prune" if index % 2 == 0 else "final_rank",
        "first_prune_depth": 1 if index % 2 == 0 else None,
        "prompt_token_ids": [1, index + 2],
        "positive_token_ids": [3, 4],
        "negative_token_ids": [5, 6],
        "reference_positive_score": -3.0,
        "reference_negative_score": -2.0,
        "reference_margin": 1.0,
        "negative_structure_valid": True,
        "negative_catalog_expandable": False,
        "negative_poi_id": None,
        "strict_duplicate_filtered": False,
    }


def test_ddp_batches_are_disjoint_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "dataset"
    Dataset.from_list([record(index) for index in range(17)]).save_to_disk(path)
    store = RiskPairStore(path, seed=42, expected_rows=17)
    batches = [
        store.batch_for_optimizer_step(
            optimizer_step=4,
            interval=4,
            global_batch_size=8,
            rank=rank,
            world_size=4,
        )
        for rank in range(4)
    ]
    assert all(len(batch) == 2 for batch in batches)
    prompts = [prompt for batch in batches for prompt in batch.prompts]
    assert len(set(prompts)) == 8
    repeated = store.batch_for_optimizer_step(
        optimizer_step=4,
        interval=4,
        global_batch_size=8,
        rank=0,
        world_size=4,
    )
    assert repeated == batches[0]
