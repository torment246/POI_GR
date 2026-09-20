"""LLaMA-Factory Trainer extension that appends BeamRisk gradients."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

from .config import RiskConfig
from .errors import BeamRiskError
from .loss import score_and_compute_beam_risk_loss
from .risk_store import RiskPairStore


class BeamRiskTrainer(CustomSeq2SeqTrainer):
    """Preserve the standard CE step and periodically append one risk batch."""

    def __init__(
        self,
        *args: Any,
        risk_dataset_path: Path | None = None,
        risk_dataset_rows: int = 0,
        risk_config: RiskConfig | None = None,
        risk_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._beamrisk_config = risk_config
        self._beamrisk_store = (
            RiskPairStore(
                risk_dataset_path,
                seed=risk_seed,
                expected_rows=risk_dataset_rows,
            )
            if risk_dataset_path is not None
            else None
        )
        if (self._beamrisk_store is None) != (risk_config is None):
            raise BeamRiskError("risk dataset 与 risk config 必须同时启用或关闭")
        self._beamrisk_last_optimizer_step: int | None = None
        self._beamrisk_loss_sum = 0.0
        self._beamrisk_margin_sum = 0.0
        self._beamrisk_pairs = 0
        self._beamrisk_events = 0
        self._beamrisk_log_type_loss_sums: Counter[str] = Counter()
        self._beamrisk_log_type_pairs: Counter[str] = Counter()
        self._beamrisk_type_counts: Counter[str] = Counter()
        self._beamrisk_type_loss_sums: Counter[str] = Counter()
        self._beamrisk_total_pairs = 0
        self._beamrisk_total_events = 0

    def _should_inject(self, upcoming_optimizer_step: int) -> bool:
        config = self._beamrisk_config
        return (
            config is not None
            and config.weight > 0
            and upcoming_optimizer_step % config.interval_optimizer_steps == 0
            and self._beamrisk_last_optimizer_step != upcoming_optimizer_step
        )

    def _inject_beamrisk(self, model: Any, optimizer_step: int) -> None:
        config = self._beamrisk_config
        store = self._beamrisk_store
        if config is None or store is None:
            return
        # Transformers 4.52 performs gradient accumulation itself and creates
        # Accelerator with accumulation_steps=1.  Consequently this one-shot
        # auxiliary backward has exactly the configured lambda scale.  Fail
        # loudly if a future Trainer moves accumulation back into Accelerator;
        # silently dividing BeamRisk by the CE accumulation factor would change
        # the preregistered experiment.
        accelerator_accumulation = int(
            self.accelerator.gradient_accumulation_steps
        )
        if accelerator_accumulation != 1:
            raise BeamRiskError(
                "BeamRisk 要求 Trainer 显式梯度累积且 Accelerator "
                f"gradient_accumulation_steps=1，实际 {accelerator_accumulation}"
            )
        rank = int(self.args.process_index)
        world_size = int(self.args.world_size)
        batch = store.batch_for_optimizer_step(
            optimizer_step=optimizer_step,
            interval=config.interval_optimizer_steps,
            global_batch_size=config.global_batch_size,
            rank=rank,
            world_size=world_size,
        )
        if len(batch) != config.per_device_batch_size:
            raise BeamRiskError(
                f"per-rank risk batch {len(batch)} != {config.per_device_batch_size}"
            )
        pad_token_id = int(self.processing_class.pad_token_id)
        local_loss_sum = 0.0
        local_margin_sum = 0.0
        for start in range(0, len(batch), config.per_device_micro_batch_size):
            end = min(start + config.per_device_micro_batch_size, len(batch))
            chunk = batch.slice(start, end)
            with self.compute_loss_context_manager():
                output = score_and_compute_beam_risk_loss(
                    model,
                    chunk.prompts,
                    chunk.positives,
                    chunk.negatives,
                    pad_token_id=pad_token_id,
                    temperature=config.temperature,
                    margin=config.margin,
                    device=self.args.device,
                )
                chunk_weight = len(chunk) / len(batch)
                weighted_loss = config.weight * chunk_weight * output.loss
            self.accelerator.backward(weighted_loss)
            detached_losses = output.per_pair_loss.detach().float().cpu().tolist()
            local_loss_sum += float(sum(detached_losses))
            local_margin_sum += float(output.margins.detach().sum().item())
            for risk_type, pair_loss in zip(chunk.risk_types, detached_losses):
                self._beamrisk_log_type_loss_sums[risk_type] += float(pair_loss)
                self._beamrisk_log_type_pairs[risk_type] += 1
                self._beamrisk_type_loss_sums[risk_type] += float(pair_loss)
        self._beamrisk_last_optimizer_step = optimizer_step
        self._beamrisk_loss_sum += local_loss_sum
        self._beamrisk_margin_sum += local_margin_sum
        self._beamrisk_pairs += len(batch)
        self._beamrisk_events += 1
        self._beamrisk_total_pairs += len(batch)
        self._beamrisk_total_events += 1
        self._beamrisk_type_counts.update(batch.risk_types)

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Any = None,
    ) -> torch.Tensor:
        main_loss = super().training_step(
            model,
            inputs,
            num_items_in_batch=num_items_in_batch,
        )
        upcoming_optimizer_step = int(self.state.global_step) + 1
        if self._should_inject(upcoming_optimizer_step):
            self._inject_beamrisk(model, upcoming_optimizer_step)
        return main_loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        enriched = dict(logs)
        if self._beamrisk_pairs:
            enriched["beamrisk_loss"] = self._beamrisk_loss_sum / self._beamrisk_pairs
            enriched["beamrisk_margin"] = self._beamrisk_margin_sum / self._beamrisk_pairs
            enriched["beamrisk_events"] = float(self._beamrisk_events)
            enriched["beamrisk_pairs"] = float(self._beamrisk_pairs * int(self.args.world_size))
            for risk_type, metric_name in (
                ("first_prune", "beamrisk_survive_loss"),
                ("final_rank", "beamrisk_rank_loss"),
            ):
                type_pairs = self._beamrisk_log_type_pairs[risk_type]
                if type_pairs:
                    enriched[metric_name] = (
                        self._beamrisk_log_type_loss_sums[risk_type] / type_pairs
                    )
                    enriched[f"{metric_name}_pairs"] = float(
                        type_pairs * int(self.args.world_size)
                    )
            self._beamrisk_loss_sum = 0.0
            self._beamrisk_margin_sum = 0.0
            self._beamrisk_pairs = 0
            self._beamrisk_events = 0
            self._beamrisk_log_type_loss_sums.clear()
            self._beamrisk_log_type_pairs.clear()
        super().log(enriched, start_time=start_time)

    def beamrisk_summary(self) -> dict[str, Any]:
        return {
            "enabled": self._beamrisk_store is not None,
            "local_pairs": self._beamrisk_total_pairs,
            "global_pairs": self._beamrisk_total_pairs * int(self.args.world_size),
            "events": self._beamrisk_total_events,
            "local_risk_type_counts": dict(sorted(self._beamrisk_type_counts.items())),
            "local_mean_loss_by_risk_type": {
                risk_type: self._beamrisk_type_loss_sums[risk_type] / count
                for risk_type, count in sorted(self._beamrisk_type_counts.items())
                if count
            },
        }
