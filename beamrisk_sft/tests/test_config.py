from pathlib import Path

from beamrisk_sft.config import load_config
from beamrisk_sft.io import implementation_sha256


ROOT = Path(__file__).resolve().parents[2]


def test_main_config_freezes_tiger_protocol_and_managed_outputs() -> None:
    config = load_config(ROOT / "beamrisk_sft" / "configs" / "main_v1.yaml")
    assert config.training.world_size == 4
    assert config.training.epochs == 3
    assert config.training.expected_global_batch_size == 512
    assert config.training.expected_optimizer_steps_per_epoch == 5571
    assert config.risk.global_batch_size == 128
    assert config.risk.weight == 0.2
    assert config.mining.beam_size == 10
    assert config.paths.model.name == "Qwen3-0.6B-TIGER-Vocab-v1"
    assert config.paths.raw_valid.name == "valid.jsonl"
    assert config.paths.tiger_mapping.parent == config.paths.tiger_identifier_dir
    managed = ROOT / "beamrisk_sft"
    for path in (
        config.paths.candidate_pool_dir,
        config.paths.sft_output_dir,
        config.paths.mining_root,
        config.paths.eval_output_dir,
    ):
        path.relative_to(managed)
    fingerprint = implementation_sha256(ROOT)
    assert len(fingerprint) == 64
    int(fingerprint, 16)
