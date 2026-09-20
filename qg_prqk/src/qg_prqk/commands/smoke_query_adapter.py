"""Overfit the QG Query Adapter on deterministic synthetic vectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F


from qg_prqk.config import load_config
from qg_prqk.adapters.model import (
    ResidualQueryAdapter,
    fit_query_adapter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用合成向量验证 Query Adapter 可过拟合，不读取业务数据。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-as-epochs", type=int, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, seed=args.seed)
    torch.manual_seed(args.seed)
    rows = 48
    dimension = 16
    negatives_per_query = 8
    positives = F.normalize(torch.randn(rows, dimension), dim=1)
    rotation = torch.linalg.qr(torch.randn(dimension, dimension)).Q
    raw_queries = F.normalize(positives @ rotation + 0.03 * torch.randn_like(positives), dim=1)
    negative_rows = torch.stack(
        [
            torch.tensor(
                [(index + offset + 1) % rows for offset in range(negatives_per_query)]
            )
            for index in range(rows)
        ]
    )
    negatives = positives[negative_rows]
    valid_mask = torch.ones((rows, negatives_per_query), dtype=torch.bool)
    weights = torch.ones(rows)
    adapter_config = replace(
        config.query_adapter,
        bottleneck=16,
        dropout=0.0,
        learning_rate=2e-2,
        weight_decay=0.0,
        epochs=args.steps_as_epochs,
        batch_size=40,
    )
    model = ResidualQueryAdapter(
        dimension,
        adapter_config.bottleneck,
        residual_scale=adapter_config.residual_scale,
        dropout=adapter_config.dropout,
    )
    result = fit_query_adapter(
        model,
        raw_queries,
        positives,
        negatives,
        valid_mask,
        weights,
        target_poi_rows=torch.arange(rows),
        reasonable_positive_rows=[() for _ in range(rows)],
        train_indices=list(range(40)),
        dev_indices=list(range(40, 48)),
        config=adapter_config,
        device=torch.device("cpu"),
        batch_size=40,
    )
    payload = {
        "status": "passed"
        if result.epoch_losses[-1] < result.epoch_losses[0]
        else "failed",
        "synthetic_only": True,
        "first_epoch_loss": result.epoch_losses[0],
        "last_epoch_loss": result.epoch_losses[-1],
        "raw_metrics": result.raw_metrics,
        "adapted_metrics": result.adapted_metrics,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
