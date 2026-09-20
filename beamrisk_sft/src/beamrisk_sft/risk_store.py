"""Memory-mapped deterministic DDP risk-pair batches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from datasets import load_from_disk

from .errors import BeamRiskError
from .schema import RiskPair


@dataclass(frozen=True)
class RiskBatch:
    prompts: tuple[tuple[int, ...], ...]
    positives: tuple[tuple[int, ...], ...]
    negatives: tuple[tuple[int, ...], ...]
    risk_types: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.prompts)

    def slice(self, start: int, end: int) -> "RiskBatch":
        return RiskBatch(
            prompts=self.prompts[start:end],
            positives=self.positives[start:end],
            negatives=self.negatives[start:end],
            risk_types=self.risk_types[start:end],
        )


class RiskPairStore:
    def __init__(
        self,
        dataset_path: Path,
        *,
        seed: int,
        expected_rows: int,
    ) -> None:
        self.dataset_path = dataset_path.resolve()
        self.dataset = load_from_disk(str(self.dataset_path))
        self.rows = len(self.dataset)
        if self.rows != expected_rows or self.rows <= 0:
            raise BeamRiskError(
                f"Risk dataset 行数 {self.rows} != manifest {expected_rows}"
            )
        self.seed = int(seed)
        self.stride = 104_729
        while math.gcd(self.stride, self.rows) != 1:
            self.stride += 2

    def _global_indices(self, event_index: int, global_batch_size: int) -> list[int]:
        if event_index < 0 or global_batch_size <= 0:
            raise BeamRiskError("risk event/global batch 参数无效")
        offset = (self.seed * 1_000_003 + event_index * global_batch_size) % self.rows
        return [
            (offset + index * self.stride) % self.rows
            for index in range(global_batch_size)
        ]

    def batch_for_optimizer_step(
        self,
        *,
        optimizer_step: int,
        interval: int,
        global_batch_size: int,
        rank: int,
        world_size: int,
    ) -> RiskBatch:
        if optimizer_step <= 0 or optimizer_step % interval != 0:
            raise BeamRiskError("只允许为命中的 optimizer step 取 risk batch")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise BeamRiskError("risk DDP rank/world_size 无效")
        if global_batch_size % world_size:
            raise BeamRiskError("risk global batch 必须整除 world_size")
        event_index = optimizer_step // interval - 1
        global_indices = self._global_indices(event_index, global_batch_size)
        local_indices = global_indices[rank::world_size]
        rows = [self.dataset[index] for index in local_indices]
        parsed = [RiskPair.from_mapping(row) for row in rows]
        return RiskBatch(
            prompts=tuple(pair.prompt_token_ids for pair in parsed),
            positives=tuple(pair.positive_token_ids for pair in parsed),
            negatives=tuple(pair.negative_token_ids for pair in parsed),
            risk_types=tuple(pair.risk_type for pair in parsed),
        )
