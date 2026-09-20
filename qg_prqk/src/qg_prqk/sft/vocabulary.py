"""Create the shared SFT vocabulary from frozen data metadata."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from qg_prqk.artifacts import sha256_file, utc_now


VOCAB_SCHEMA_VERSION = "qg-prqk-sft-vocab-v1"
MAPPING_FILENAME = "qg_prqk_token_mapping.json"
TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class SftVocabError(ValueError):
    """Raised when the shared QG-PRQK vocabulary contract is violated."""


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftVocabError(f"{name} 不是合法 JSON：{path}") from error
    if not isinstance(payload, dict):
        raise SftVocabError(f"{name} 必须是 JSON object")
    return payload


def _hash_named_files(directory: Path, names: Sequence[str]) -> str:
    import hashlib

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
        raise SftVocabError(f"目录中没有 tokenizer 文件：{directory}")
    return digest.hexdigest()


def load_shared_tokens(gid_tokens_path: Path, nogid_tokens_path: Path) -> list[str]:
    """Validate that both variants use one exactly ordered token inventory."""

    payloads = [
        _load_json(gid_tokens_path, "GID special_tokens"),
        _load_json(nogid_tokens_path, "NoGID special_tokens"),
    ]
    token_lists: list[list[str]] = []
    for payload in payloads:
        tokens = payload.get("additional_special_tokens")
        count = payload.get("token_count")
        if (
            not isinstance(tokens, list)
            or not all(isinstance(token, str) and token for token in tokens)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count != len(tokens)
        ):
            raise SftVocabError("special_tokens 的 Token 列表或数量非法")
        if len(tokens) != len(set(tokens)) or "<D_-1>" in tokens:
            raise SftVocabError("Token 表重复或包含非法 <D_-1>")
        token_lists.append(tokens)
    if token_lists[0] != token_lists[1]:
        raise SftVocabError("GID 与 NoGID 必须使用完全相同且同序的 Token 表")
    return token_lists[0]


def _base_fingerprints(model_dir: Path) -> dict[str, str]:
    config_path = model_dir / "config.json"
    config = _load_json(config_path, "基础模型 config")
    if (
        config.get("model_type") != "qwen3"
        or config.get("architectures") != ["Qwen3ForCausalLM"]
    ):
        raise SftVocabError("基础模型必须是 Qwen3ForCausalLM")
    weights = sorted(model_dir.glob("model*.safetensors"))
    if len(weights) != 1:
        raise SftVocabError("基础模型必须恰好包含一个 model*.safetensors")
    return {
        "config_sha256": sha256_file(config_path),
        "model_sha256": sha256_file(weights[0]),
        "tokenizer_sha256": _hash_named_files(model_dir, TOKENIZER_FILES),
    }


def _verify_atomic_tokens(
    tokenizer: Any, tokens: Sequence[str], mapping: Mapping[str, int]
) -> None:
    token_ids = list(mapping.values())
    if len(token_ids) != len(set(token_ids)):
        raise SftVocabError("新增 Token ID 存在重复")
    for token in tokens:
        token_id = mapping[token]
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise SftVocabError(f"Token 不是原子编码：{token}")
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != token:
            raise SftVocabError(f"Token round-trip 不一致：{token}")
        added = tokenizer.added_tokens_decoder.get(token_id)
        if added is None or bool(added.special):
            raise SftVocabError(f"Token 未按普通 Token 注册：{token}")


def _validate_existing(
    *,
    output_dir: Path,
    model_dir: Path,
    token_paths: Sequence[Path],
    tokens: Sequence[str],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    payload = _load_json(output_dir / MAPPING_FILENAME, "现有扩词表映射")
    fingerprints = _base_fingerprints(model_dir)
    expected = {
        "schema_version": VOCAB_SCHEMA_VERSION,
        "base_model_sha256": fingerprints["model_sha256"],
        "base_config_sha256": fingerprints["config_sha256"],
        "base_tokenizer_sha256": fingerprints["tokenizer_sha256"],
        "token_source_sha256": [sha256_file(path) for path in token_paths],
        "added_token_count": len(tokens),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SftVocabError(f"现有扩词表与当前输入不一致：{key}")
    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    mapping = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens
    }
    if mapping != payload.get("tokens"):
        raise SftVocabError("现有 tokenizer 的 Token ID 与映射不一致")
    if _hash_named_files(output_dir, TOKENIZER_FILES) != payload.get(
        "extended_tokenizer_sha256"
    ):
        raise SftVocabError("现有 tokenizer 文件哈希不一致")
    _verify_atomic_tokens(tokenizer, tokens, mapping)
    return payload


def prepare_shared_sft_vocab(
    *,
    model_dir: Path,
    gid_tokens_path: Path,
    nogid_tokens_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Expand one local Qwen model for both SFT variants without overwrite."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_dir = model_dir.resolve()
    token_paths = [gid_tokens_path.resolve(), nogid_tokens_path.resolve()]
    output_dir = output_dir.resolve()
    tokens = load_shared_tokens(*token_paths)
    if output_dir.exists():
        return _validate_existing(
            output_dir=output_dir,
            model_dir=model_dir,
            token_paths=token_paths,
            tokens=tokens,
        )

    fingerprints = _base_fingerprints(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    original_vocab_size = len(tokenizer)
    added_count = tokenizer.add_tokens(list(tokens), special_tokens=False)
    if added_count != len(tokens):
        raise SftVocabError(
            f"期望新增 {len(tokens)} 个 Token，实际新增 {added_count} 个"
        )
    mapping = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens
    }
    _verify_atomic_tokens(tokenizer, tokens, mapping)

    temporary = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if temporary.exists():
        raise SftVocabError(f"临时目录已存在：{temporary}")
    temporary.mkdir(parents=True)
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
        if not bool(model.config.tie_word_embeddings):
            raise SftVocabError("模型未配置输入 Embedding 与 LM Head 绑定")
        if (
            model.get_input_embeddings().weight.data_ptr()
            != model.get_output_embeddings().weight.data_ptr()
        ):
            raise SftVocabError("扩词表后输入 Embedding 与 LM Head 未保持绑定")
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reloaded = AutoTokenizer.from_pretrained(temporary, local_files_only=True)
        reloaded_mapping = {
            token: int(reloaded.convert_tokens_to_ids(token)) for token in tokens
        }
        if reloaded_mapping != mapping or len(reloaded) != len(tokenizer):
            raise SftVocabError("保存后 Token ID 或词表大小发生变化")
        _verify_atomic_tokens(reloaded, tokens, reloaded_mapping)
        payload = {
            "schema_version": VOCAB_SCHEMA_VERSION,
            "status": "completed",
            "built_at": utc_now(),
            "shared_by_variants": ["a4_gid_parent", "a4_nogid"],
            "base_model_path": str(model_dir),
            "base_model_sha256": fingerprints["model_sha256"],
            "base_config_sha256": fingerprints["config_sha256"],
            "base_tokenizer_sha256": fingerprints["tokenizer_sha256"],
            "token_source_paths": [str(path) for path in token_paths],
            "token_source_sha256": [sha256_file(path) for path in token_paths],
            "original_vocab_size": original_vocab_size,
            "new_vocab_size": len(reloaded),
            "added_token_count": len(tokens),
            "ordinary_tokens": True,
            "extended_tokenizer_sha256": _hash_named_files(
                temporary, TOKENIZER_FILES
            ),
            "tokens": reloaded_mapping,
        }
        (temporary / MAPPING_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
        return payload
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
