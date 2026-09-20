#!/usr/bin/env python3
"""Two-rank CPU smoke for the TIGER-Joint DDP forward boundary."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from poi_gr.methods.tiger_joint import (  # noqa: E402
    HuggingFaceCausalLMJointAdapter,
    JointTigerTrainingModule,
    SidTokenLayout,
)
from poi_gr.sid.rqvae import RQVAE  # noqa: E402
from tests.tiger_joint.test_training import (  # noqa: E402
    _TinyCausalLM,
    _build_synthetic_batch,
)


def main() -> int:
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("DDP contract smoke 固定要求两个 rank")
    torch.manual_seed(42)
    batch = _build_synthetic_batch()
    if rank:
        batch = replace(
            batch,
            target_embeddings=batch.target_embeddings + 0.05,
            catalog_embeddings=batch.catalog_embeddings - 0.03,
        )
    generator = HuggingFaceCausalLMJointAdapter(
        _TinyCausalLM(vocab_size=40, hidden_size=8)
    )
    rqvae = RQVAE(
        input_dim=6,
        hidden_dim=5,
        latent_dim=4,
        codebook_sizes=(4, 4, 4),
    )
    projection = nn.Linear(8, 4)
    module = JointTigerTrainingModule(
        generator=generator,
        rqvae=rqvae,
        query_projection=projection,
        token_layout=SidTokenLayout(
            (
                tuple(range(20, 24)),
                tuple(range(24, 28)),
                tuple(range(28, 32)),
            )
        ),
        alignment_weight=0.25,
        rq_weight=0.5,
        temperature=0.2,
    )
    ddp_module = DistributedDataParallel(
        module,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.Adam(ddp_module.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    with ddp_module.no_sync():
        first_output = ddp_module(batch)
        (first_output.total_loss / 2).backward()
    second_output = ddp_module(batch)
    (second_output.total_loss / 2).backward()
    gradient_groups = {
        "generator": generator.causal_lm.model.embedding.weight.grad,
        "projection": projection.weight.grad,
        "rq_encoder": rqvae.encoder[0].weight.grad,
        "rq_codebook": rqvae.quantizer.codebooks[0].weight.grad,
    }
    if any(
        value is None or not torch.isfinite(value).all() or not value.abs().sum()
        for value in gradient_groups.values()
    ):
        raise RuntimeError("DDP 后存在缺失、非有限或零梯度")
    optimizer.step()
    flattened = torch.cat(
        [parameter.detach().reshape(-1) for parameter in module.parameters()]
    )
    gathered = [torch.empty_like(flattened) for _ in range(world_size)]
    dist.all_gather(gathered, flattened)
    torch.testing.assert_close(gathered[0], gathered[1], rtol=0.0, atol=0.0)
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "completed",
                    "world_size": world_size,
                    "parameters_synchronized": True,
                    "gradient_accumulation_no_sync": True,
                    "all_gradient_groups_finite_positive": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
