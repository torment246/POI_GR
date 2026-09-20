from pathlib import Path

from beamrisk_sft.smoke import run_trainer_smoke


def test_real_trainer_applies_one_risk_event_at_optimizer_step_four(
    tmp_path: Path,
) -> None:
    result = run_trainer_smoke(
        tmp_path / "trainer_smoke",
        expected_world_size=1,
        device="cpu",
    )
    assert result is not None
    assert result["status"] == "passed"
    assert result["optimizer_steps"] == 4
    assert result["rank_reports"][0]["risk_events"] == 1
    assert result["rank_reports"][0]["local_risk_pairs"] == 2
    assert result["rank_reports"][0]["weights_changed"] is True

