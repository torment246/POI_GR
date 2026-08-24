"""MMBERT recall checkpoint adapter for the shared embedding pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class MMBertRecallEncoder:
    """Expose the trained 128-dimensional recall head through ``encode``."""

    pooling_description = "attention_mask_mean+linear_768_to_128"

    def __init__(
        self,
        *,
        tokenizer: Any,
        encoder: Any,
        projection: Any,
        device: str,
        max_seq_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.projection = projection
        self.device = device
        self.max_seq_length = max_seq_length

    def eval(self) -> "MMBertRecallEncoder":
        self.encoder.eval()
        self.projection.eval()
        return self

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.projection.out_features)

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        prompt_name: str | None = None,
        **_: Any,
    ) -> np.ndarray:
        """Encode texts with the checkpoint's mean-pool and projection head."""

        import torch
        import torch.nn.functional as functional

        if show_progress_bar:
            raise ValueError("MMBertRecallEncoder 不在内部显示进度条")
        if not convert_to_numpy:
            raise ValueError("MMBertRecallEncoder 当前只支持 NumPy 输出")
        if prompt_name is not None:
            raise ValueError("MMBertRecallEncoder 不支持 prompt_name")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")

        all_vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(sentences), batch_size):
                batch = list(sentences[start : start + batch_size])
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in tokens.items()
                }
                outputs = self.encoder(**tokens)
                hidden = outputs.last_hidden_state.float()
                attention_mask = tokens["attention_mask"]
                expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
                pooled = (hidden * expanded_mask).sum(dim=1) / expanded_mask.sum(
                    dim=1
                ).clamp_min(1e-9)
                projected = self.projection(pooled + 1e-9)
                if normalize_embeddings:
                    projected = functional.normalize(projected, p=2, dim=1)
                all_vectors.append(projected.cpu().numpy())

        if not all_vectors:
            return np.empty(
                (0, self.get_sentence_embedding_dimension()), dtype=np.float32
            )
        return np.concatenate(all_vectors, axis=0)


def _load_projection_state(path: Path) -> dict[str, Any]:
    import torch

    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        load_kwargs["weights_only"] = True
    if "mmap" in parameters:
        load_kwargs["mmap"] = True
    state = torch.load(path, **load_kwargs)
    required = ("projection.weight", "projection.bias")
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"checkpoint 缺少投影参数：{missing}")
    return {key.removeprefix("projection."): state[key] for key in required}


def load_mmbert_recall_encoder(
    *,
    model_path: Path,
    device: str,
    max_seq_length: int,
    torch_dtype: Any,
    attention: str | None,
    padding_side: str,
) -> MMBertRecallEncoder:
    """Load the encoder and the trained 128-dimensional projection head."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    full_state_path = model_path / "pytorch_model.bin"
    if not full_state_path.is_file():
        raise RuntimeError(f"缺少完整 encoder+projection 权重：{full_state_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = padding_side
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if attention:
        model_kwargs["attn_implementation"] = attention
    encoder = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)

    projection_state = _load_projection_state(full_state_path)
    projection_weight = projection_state["weight"]
    projection_bias = projection_state["bias"]
    if projection_weight.ndim != 2 or projection_bias.ndim != 1:
        raise RuntimeError("projection 参数维度非法")
    output_dim, input_dim = map(int, projection_weight.shape)
    if input_dim != int(encoder.config.hidden_size):
        raise RuntimeError(
            f"projection 输入维度 {input_dim} != encoder hidden_size "
            f"{encoder.config.hidden_size}"
        )
    if projection_bias.shape != (output_dim,):
        raise RuntimeError("projection bias 与 weight 形状不一致")
    if output_dim != 128:
        raise RuntimeError(f"本实验只接受训练好的 128 维输出，实际为 {output_dim}")

    projection = torch.nn.Linear(input_dim, output_dim, dtype=torch.float32)
    projection.load_state_dict(projection_state, strict=True)
    projection.to(device)
    del projection_state, projection_weight, projection_bias

    return MMBertRecallEncoder(
        tokenizer=tokenizer,
        encoder=encoder,
        projection=projection,
        device=device,
        max_seq_length=max_seq_length,
    ).eval()
