import json
from dataclasses import replace
from pathlib import Path

import pytest

from beamrisk_sft.errors import BeamRiskError
from beamrisk_sft.config import load_config
from beamrisk_sft.evaluation import _load_eval_result, summarize_final_evaluation


ROOT = Path(__file__).resolve().parents[2]


def _payload(checkpoint: Path, *, constrained: bool) -> dict:
    return {
        "status": "completed",
        "evaluation_subset": {
            "output_rows": 10_000,
            "output_sha256": "a" * 64,
        },
        "results": [
            {
                "status": "completed",
                "config": {
                    "checkpoint": str(checkpoint),
                    "epoch": 3.0,
                    "num_beams": 10,
                    "rows": 10_000,
                    "legal_path_constraint": constrained,
                },
                "metrics": {
                    "sample_count": 10_000,
                    "hr@1": 0.5,
                    "hr@3": 0.7,
                    "hr@5": 0.8,
                    "hr@10": 0.9,
                    "ndcg@10": 0.68,
                    "valid_id_rate": 1.0 if constrained else 0.75,
                },
                "performance": {},
                "run_dir": "/tmp/run",
            }
        ],
    }


@pytest.mark.parametrize("constrained", [False, True])
def test_final_eval_result_contract(tmp_path: Path, constrained: bool) -> None:
    checkpoint = tmp_path / "checkpoint-16713"
    checkpoint.mkdir()
    path = tmp_path / f"result-{constrained}.json"
    path.write_text(
        json.dumps(_payload(checkpoint, constrained=constrained)),
        encoding="utf-8",
    )
    result, subset = _load_eval_result(
        path,
        checkpoint=checkpoint,
        legal_path_constraint=constrained,
    )
    assert result["metrics"]["hr@10"] == 0.9
    assert subset["output_rows"] == 10_000


def test_constrained_eval_requires_all_paths_valid(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-16713"
    checkpoint.mkdir()
    payload = _payload(checkpoint, constrained=True)
    payload["results"][0]["metrics"]["valid_id_rate"] = 0.99
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BeamRiskError, match="valid_id_rate"):
        _load_eval_result(
            path,
            checkpoint=checkpoint,
            legal_path_constraint=True,
        )


def test_final_summary_includes_training_and_both_protocols(tmp_path: Path) -> None:
    original = load_config(ROOT / "beamrisk_sft" / "configs" / "main_v1.yaml")
    sft_output = tmp_path / "sft"
    eval_output = tmp_path / "eval"
    config = replace(
        original,
        paths=replace(
            original.paths,
            sft_output_dir=sft_output,
            eval_output_dir=eval_output,
        ),
    )
    stages = sft_output / "stages"
    stages.mkdir(parents=True)
    checkpoints: list[Path] = []
    for epoch, loss in ((1, 0.4), (2, 0.3), (3, 0.28)):
        step = 5_571 * epoch
        checkpoint = sft_output / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "log_history": [{"step": step, "eval_loss": loss}],
                }
            ),
            encoding="utf-8",
        )
        (stages / f"epoch_{epoch}.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "target_epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "risk_enabled": epoch >= 2,
                    "beamrisk_summary": {
                        "global_pairs": 128 if epoch >= 2 else 0
                    },
                }
            ),
            encoding="utf-8",
        )
        checkpoints.append(checkpoint)
    unconstrained_path = tmp_path / "unconstrained.json"
    constrained_path = tmp_path / "constrained.json"
    unconstrained_path.write_text(
        json.dumps(_payload(checkpoints[-1], constrained=False)),
        encoding="utf-8",
    )
    constrained_path.write_text(
        json.dumps(_payload(checkpoints[-1], constrained=True)),
        encoding="utf-8",
    )
    result = summarize_final_evaluation(
        config,
        final_checkpoint=checkpoints[-1],
        unconstrained_result_path=unconstrained_path,
        constrained_result_path=constrained_path,
    )
    assert len(result["training"]) == 3
    assert result["training"][-1]["validation_loss"] == 0.28
    assert (eval_output / "final_eval_summary.json").is_file()
    assert (eval_output / "final_eval_summary.md").is_file()
