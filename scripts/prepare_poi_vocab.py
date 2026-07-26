#!/usr/bin/env python
"""Extend a local Qwen tokenizer with deterministic POI/PID tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "poi-vocab-v1"
EXPECTED_TOKEN_COUNT = 3620
TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class PoiVocabError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_named_files(directory: Path, names: Sequence[str]) -> str:
    """Hash a named artifact set including relative names and contents."""

    digest = hashlib.sha256()
    found = False
    for name in sorted(names):
        path = directory / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    if not found:
        raise PoiVocabError(f"目录中没有可哈希的 tokenizer 文件：{directory}")
    return digest.hexdigest()


def load_requested_tokens(path: Path) -> list[str]:
    """Load and validate the ordered token contract."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PoiVocabError(f"Token 文件读取失败：{path}") from error
    if not isinstance(payload, dict):
        raise PoiVocabError("special_tokens.json 必须是 JSON object")
    tokens = payload.get("additional_special_tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise PoiVocabError("additional_special_tokens 必须是字符串列表")
    if payload.get("token_count") != EXPECTED_TOKEN_COUNT or len(tokens) != EXPECTED_TOKEN_COUNT:
        raise PoiVocabError(
            f"新增 Token 数必须为 {EXPECTED_TOKEN_COUNT}，实际为 {len(tokens)}"
        )
    if len(tokens) != len(set(tokens)):
        raise PoiVocabError("新增 Token 列表存在重复项")
    if "<D_-1>" in tokens:
        raise PoiVocabError("不得注册 <D_-1>")
    return tokens


def verify_atomic_tokens(tokenizer: Any, tokens: Sequence[str], mapping: dict[str, int]) -> None:
    """Verify every requested token is ordinary, atomic and round-trip stable."""

    token_ids = list(mapping.values())
    if len(token_ids) != len(set(token_ids)):
        raise PoiVocabError("新增 Token ID 存在重复")
    for token in tokens:
        token_id = mapping[token]
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise PoiVocabError(f"Token 不是原子编码：{token} -> {encoded}")
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != token:
            raise PoiVocabError(f"Token round-trip 不一致：{token} -> {decoded}")
        added_token = tokenizer.added_tokens_decoder.get(token_id)
        if added_token is None or bool(added_token.special):
            raise PoiVocabError(f"Token 未按普通 Token 注册：{token}")


def _base_model_weight(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("model*.safetensors"))
    if len(candidates) != 1:
        raise PoiVocabError(
            f"基础模型必须恰好包含一个 model*.safetensors，实际为 {len(candidates)}"
        )
    return candidates[0]


def base_fingerprints(model_dir: Path) -> dict[str, str]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise PoiVocabError(f"缺少模型配置：{config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PoiVocabError(f"模型配置不是合法 JSON：{config_path}") from error
    if config.get("model_type") != "qwen3":
        raise PoiVocabError(f"模型类型必须为 qwen3，实际为 {config.get('model_type')}")
    architectures = config.get("architectures")
    if architectures != ["Qwen3ForCausalLM"]:
        raise PoiVocabError(f"模型必须为 Qwen3ForCausalLM，实际为 {architectures}")
    return {
        "config_sha256": sha256_file(config_path),
        "model_sha256": sha256_file(_base_model_weight(model_dir)),
        "tokenizer_sha256": sha256_named_files(model_dir, TOKENIZER_FILES),
    }


def _mapping_payload(
    *,
    model_dir: Path,
    tokens_path: Path,
    original_vocab_size: int,
    original_model_vocab_size: int,
    tokenizer: Any,
    tokens: Sequence[str],
    fingerprints: dict[str, str],
    extended_tokenizer_sha256: str,
) -> dict[str, Any]:
    mapping = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}
    return {
        "schema_version": SCHEMA_VERSION,
        "original_model_path": str(model_dir.resolve()),
        "original_model_sha256": fingerprints["model_sha256"],
        "original_config_sha256": fingerprints["config_sha256"],
        "original_tokenizer_sha256": fingerprints["tokenizer_sha256"],
        "token_source_path": str(tokens_path.resolve()),
        "token_source_sha256": sha256_file(tokens_path),
        "original_vocab_size": original_vocab_size,
        "original_model_vocab_size": original_model_vocab_size,
        "new_vocab_size": len(tokenizer),
        "added_token_count": len(tokens),
        "extended_tokenizer_sha256": extended_tokenizer_sha256,
        "tokens": mapping,
    }


def validate_existing_output(
    output_dir: Path,
    model_dir: Path,
    tokens_path: Path,
    tokens: Sequence[str],
) -> dict[str, Any]:
    """Validate an existing expanded model without overwriting it."""

    from transformers import AutoTokenizer

    mapping_path = output_dir / "poi_token_mapping.json"
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PoiVocabError(
            f"扩词表目录已存在但映射文件无效，拒绝覆盖：{mapping_path}"
        ) from error
    fingerprints = base_fingerprints(model_dir)
    expected = {
        "original_model_sha256": fingerprints["model_sha256"],
        "original_config_sha256": fingerprints["config_sha256"],
        "original_tokenizer_sha256": fingerprints["tokenizer_sha256"],
        "token_source_sha256": sha256_file(tokens_path),
        "added_token_count": EXPECTED_TOKEN_COUNT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PoiVocabError(f"现有扩词表目录与当前输入不一致：{key}")
    persisted_tokens = payload.get("tokens")
    if not isinstance(persisted_tokens, dict) or set(persisted_tokens) != set(tokens):
        raise PoiVocabError("现有 Token 集合与当前 Token 表不一致")

    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    mapping = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}
    if mapping != payload["tokens"]:
        raise PoiVocabError("现有 tokenizer 的 Token ID 与映射文件不一致")
    if len(tokenizer) != payload.get("new_vocab_size"):
        raise PoiVocabError("现有 tokenizer 词表大小与映射文件不一致")
    current_hash = sha256_named_files(output_dir, TOKENIZER_FILES)
    if current_hash != payload.get("extended_tokenizer_sha256"):
        raise PoiVocabError("现有 tokenizer 文件哈希与映射文件不一致")
    verify_atomic_tokens(tokenizer, tokens, mapping)
    return payload


def prepare_poi_vocab(model_dir: Path, tokens_path: Path, output_dir: Path) -> dict[str, Any]:
    """Create or validate the deterministic expanded Qwen model."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_dir = model_dir.resolve()
    tokens_path = tokens_path.resolve()
    output_dir = output_dir.resolve()
    tokens = load_requested_tokens(tokens_path)
    if output_dir.exists():
        return validate_existing_output(output_dir, model_dir, tokens_path, tokens)

    fingerprints = base_fingerprints(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    original_vocab_size = len(tokenizer)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    original_model_vocab_size = int(config["vocab_size"])
    added_count = tokenizer.add_tokens(list(tokens), special_tokens=False)
    if added_count != EXPECTED_TOKEN_COUNT:
        raise PoiVocabError(
            f"tokenizer.add_tokens 必须新增 {EXPECTED_TOKEN_COUNT} 个，实际为 {added_count}"
        )
    mapping = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}
    verify_atomic_tokens(tokenizer, tokens, mapping)

    temp_dir = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if temp_dir.exists():
        raise PoiVocabError(f"临时目录已存在：{temp_dir}")
    temp_dir.mkdir(parents=True)
    try:
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if not bool(model.config.tie_word_embeddings):
            raise PoiVocabError("基础模型未配置输入 Embedding 与 LM Head 绑定")
        if input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr():
            raise PoiVocabError("扩词表后输入 Embedding 与 LM Head 未保持绑定")
        if input_embeddings.num_embeddings != len(tokenizer):
            raise PoiVocabError("扩词表后模型 Embedding 行数与 tokenizer 不一致")
        model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reloaded = AutoTokenizer.from_pretrained(temp_dir, local_files_only=True)
        reloaded_mapping = {
            token: int(reloaded.convert_tokens_to_ids(token)) for token in tokens
        }
        if reloaded_mapping != mapping or len(reloaded) != len(tokenizer):
            raise PoiVocabError("保存后重新加载的词表大小或 Token ID 发生变化")
        verify_atomic_tokens(reloaded, tokens, reloaded_mapping)
        extended_hash = sha256_named_files(temp_dir, TOKENIZER_FILES)
        payload = _mapping_payload(
            model_dir=model_dir,
            tokens_path=tokens_path,
            original_vocab_size=original_vocab_size,
            original_model_vocab_size=original_model_vocab_size,
            tokenizer=reloaded,
            tokens=tokens,
            fingerprints=fingerprints,
            extended_tokenizer_sha256=extended_hash,
        )
        (temp_dir / "poi_token_mapping.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为本地 Qwen3-0.6B 添加普通、原子的 POI/PID Token。"
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="基础生成模型目录")
    parser.add_argument("--tokens", type=Path, required=True, help="special_tokens.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="扩词表模型输出目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = prepare_poi_vocab(args.model_dir, args.tokens, args.output_dir)
    print(
        "词表扩展校验通过："
        f"{payload['original_vocab_size']} -> {payload['new_vocab_size']}，"
        f"新增 {payload['added_token_count']} 个普通原子 Token"
    )


if __name__ == "__main__":
    main()
