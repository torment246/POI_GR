#!/usr/bin/env python3
"""Run position-wise teacher-forcing diagnostics for generative POI IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from poi_gr.methods.gnpr.eval import (  # noqa: E402
    GnprEvalError,
    load_gnpr_token_ids,
    parse_target_codes as parse_gnpr_target,
)
from poi_gr.methods.ghr_sid.eval import (  # noqa: E402
    GhrEvalError,
    load_ghr_id_index,
    load_ghr_token_ids,
)
from poi_gr.methods.tiger.eval import (  # noqa: E402
    TigerEvalError,
    load_tiger_token_ids,
    parse_target_codes as parse_tiger_target,
)
from poi_gr.pid.trie import PidTrieError, load_pid_token_ids, sha256_file  # noqa: E402
from poi_gr.sft.evaluation import (  # noqa: E402
    GenerativeEvalError,
    encode_prompt_like_training,
    load_generation_model,
    load_lf_tokenizer_and_template,
)
from poi_gr.sft.teacher_forcing import (  # noqa: E402
    TeacherForcingError,
    empty_teacher_forcing_metrics,
    finalize_teacher_forcing_metrics,
    score_target_logits,
    update_teacher_forcing_metrics,
)


@dataclass(frozen=True)
class EncodedExample:
    sample_id: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    position_names: tuple[str, ...]
    semantic_indices: tuple[int, int, int]
    identifier_indices: tuple[int, ...]
    group: str
    context_indices: tuple[int, ...] = ()
    conditional_tail_indices: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在冻结评测集上统计 SID 各目标位置的 teacher-forcing 准确率。"
    )
    parser.add_argument(
        "--method",
        choices=("tiger", "gnpr", "genpoi", "ghr"),
        required=True,
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--token-mapping-filename",
        default="tiger_token_mapping.json",
        help="TIGER/RQ-KMeans Tokenizer 目录内的 Token 映射文件名。",
    )
    parser.add_argument(
        "--token-capacities",
        type=int,
        nargs=4,
        default=(1024, 1024, 1024, 306),
        metavar=("S1", "S2", "S3", "C"),
        help="TIGER/RQ-KMeans 四个位置的 Token 容量。",
    )
    parser.add_argument(
        "--identifier-dir",
        type=Path,
        help="GHR identifier 目录；--method ghr 时必填。",
    )
    parser.add_argument(
        "--token-source-file",
        type=Path,
        help="GHR SFT special_tokens.json；--method ghr 时必填。",
    )
    parser.add_argument(
        "--tiger-c-positive-reference-file",
        type=Path,
        help=(
            "可选 TIGER 固定 10k Messages；仅用于 GHR，并严格筛选同业务键下"
            "目标 collision code > 0 的样本。"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=float, default=3.0)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(32, 16, 8, 4, 2),
        default=16,
    )
    parser.add_argument("--checkpoint-rows", type=int, default=1000)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--skip-model-hash", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TeacherForcingError(
                    f"评测数据第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(value, dict):
                raise TeacherForcingError(f"评测数据第 {line_number} 行必须为 object")
            records.append(value)
    return records


def _record_alignment_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    values = tuple(record.get(name) for name in ("sample_id", "order_id", "searchid"))
    if not all(isinstance(value, str) and value for value in values):
        raise TeacherForcingError("评测样本缺少 sample_id/order_id/searchid 对齐键")
    return values  # type: ignore[return-value]


def _ordered_alignment_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update("\t".join(_record_alignment_key(record)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_tiger_c_positive_records(
    records: Sequence[dict[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
    *,
    token_capacities: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select rows whose aligned TIGER target uses a non-zero collision code."""

    if len(records) != 10_000 or len(reference_records) != 10_000:
        raise TeacherForcingError("TIGER C>0 筛选要求两侧输入均恰好为 10,000 条")
    capacities = tuple(int(value) for value in token_capacities)
    if len(capacities) != 4:
        raise TeacherForcingError("TIGER Token 容量必须恰好包含四层")

    selected: list[dict[str, Any]] = []
    for row_number, (record, reference) in enumerate(
        zip(records, reference_records),
        start=1,
    ):
        if _record_alignment_key(record) != _record_alignment_key(reference):
            raise TeacherForcingError(f"TIGER 参考集第 {row_number} 行业务键未对齐")
        _, target_content, _ = assistant_content(reference)
        codes = parse_tiger_target(target_content, capacities)
        if codes[3] > 0:
            selected.append(record)

    if len(selected) != 2_356:
        raise TeacherForcingError(
            f"固定 10k 的 TIGER C>0 样本应为 2,356 条，实际为 {len(selected):,} 条"
        )
    return selected, {
        "selection": "aligned TIGER target collision code > 0",
        "source_rows": len(records),
        "selected_rows": len(selected),
        "selected_alignment_sha256": _ordered_alignment_sha256(selected),
    }


def assistant_content(record: Mapping[str, Any]) -> tuple[str, str, str]:
    messages = record.get("messages")
    if (
        record.get("split") != "valid"
        or not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise TeacherForcingError("评测样本必须是 valid user+assistant Messages")
    user_content = messages[0].get("content")
    target_content = messages[1].get("content")
    sample_id = record.get("sample_id")
    if not all(isinstance(value, str) and value for value in (user_content, target_content, sample_id)):
        raise TeacherForcingError("Messages content 与 sample_id 必须是非空字符串")
    return user_content, target_content, sample_id


def encode_tiger_example(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
) -> EncodedExample:
    user_content, target_content, sample_id = assistant_content(record)
    capacities = (
        len(tokens.s1),
        len(tokens.s2),
        len(tokens.s3),
        len(tokens.collision),
    )
    codes = parse_tiger_target(target_content, capacities)
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected = [
        tokens.target_open,
        tokens.s1[codes[0]],
        tokens.s2[codes[1]],
        tokens.s3[codes[2]],
        tokens.collision[codes[3]],
        tokens.target_close,
    ]
    if target_ids != expected:
        raise TeacherForcingError(f"TIGER 目标 Token 编码不一致：{sample_id}")
    return EncodedExample(
        sample_id=sample_id,
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_ids=tuple((*expected, tokens.eos)),
        position_names=(
            "target_open",
            "sid_1",
            "sid_2",
            "sid_3",
            "disambiguation",
            "target_close",
            "eos",
        ),
        semantic_indices=(1, 2, 3),
        identifier_indices=(1, 2, 3, 4),
        group=(
            "collision_suffix_zero"
            if codes[3] == 0
            else "collision_suffix_nonzero"
        ),
    )


def encode_gnpr_example(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
) -> EncodedExample:
    user_content, target_content, sample_id = assistant_content(record)
    codes = parse_gnpr_target(target_content)
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected = [
        tokens.target_open,
        tokens.a[codes[0]],
        tokens.b[codes[1]],
        tokens.c[codes[2]],
    ]
    if codes[3] >= 0:
        expected.extend((tokens.dedup[codes[3]], tokens.target_close))
        position_names = (
            "target_open",
            "sid_1",
            "sid_2",
            "sid_3",
            "disambiguation",
            "target_close",
            "eos",
        )
        identifier_indices = (1, 2, 3, 4)
        group = "collision"
    else:
        expected.append(tokens.target_close)
        position_names = (
            "target_open",
            "sid_1",
            "sid_2",
            "sid_3",
            "target_close",
            "eos",
        )
        identifier_indices = (1, 2, 3)
        group = "singleton"
    if target_ids != expected:
        raise TeacherForcingError(f"GNPR 目标 Token 编码不一致：{sample_id}")
    return EncodedExample(
        sample_id=sample_id,
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_ids=tuple((*expected, tokens.eos)),
        position_names=position_names,
        semantic_indices=(1, 2, 3),
        identifier_indices=identifier_indices,
        group=group,
    )


def _is_member(token_id: int, values: Sequence[int]) -> bool:
    return int(values[0]) <= token_id <= int(values[-1])


def encode_genpoi_example(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
) -> EncodedExample:
    user_content, target_content, sample_id = assistant_content(record)
    requires_dedup = record.get("requires_dedup")
    if not isinstance(requires_dedup, bool):
        raise TeacherForcingError("GenPOI requires_dedup 必须为 bool")
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected_length = 10 if requires_dedup else 9
    if len(target_ids) != expected_length:
        raise TeacherForcingError(
            f"GenPOI 目标 PID 长度 {len(target_ids)} != {expected_length}：{sample_id}"
        )
    if any(not _is_member(token_id, tokens.gid) for token_id in target_ids[:6]):
        raise TeacherForcingError(f"GenPOI GID Token 编码不一致：{sample_id}")
    if any(
        not _is_member(target_ids[6 + level], tokens.sid[level])
        for level in range(3)
    ):
        raise TeacherForcingError(f"GenPOI SID Token 编码不一致：{sample_id}")
    if requires_dedup and not _is_member(target_ids[9], tokens.dedup):
        raise TeacherForcingError(f"GenPOI Dedup Token 编码不一致：{sample_id}")

    position_names = (
        "gid_1",
        "gid_2",
        "gid_3",
        "gid_4",
        "gid_5",
        "gid_6",
        "sid_1",
        "sid_2",
        "sid_3",
        *(("disambiguation",) if requires_dedup else ()),
        "eos",
    )
    identifier_indices = tuple(range(expected_length))
    return EncodedExample(
        sample_id=sample_id,
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_ids=tuple((*target_ids, tokens.eos)),
        position_names=position_names,
        semantic_indices=(6, 7, 8),
        identifier_indices=identifier_indices,
        group="dedup" if requires_dedup else "singleton",
        context_indices=(0, 1, 2, 3, 4, 5),
    )


def encode_ghr_example(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
    collision_prefixes: set[tuple[int, int, int]] | None = None,
) -> EncodedExample:
    """Encode one GHR target while preserving its frozen TIGER prefix."""

    user_content, target_content, sample_id = assistant_content(record)
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    if (
        len(target_ids) < 5
        or target_ids[0] != tokens.target_open
        or target_ids[-1] != tokens.target_close
    ):
        raise TeacherForcingError(f"GHR 目标结构非法：{sample_id}")
    try:
        logical_codes = tuple(
            tokens.token_to_logical[int(token_id)] for token_id in target_ids[1:-1]
        )
    except KeyError as error:
        raise TeacherForcingError(f"GHR 目标包含非法 identifier Token：{sample_id}") from error
    if not (
        tokens.minimum_identifier_length
        <= len(logical_codes)
        <= tokens.maximum_identifier_length
    ):
        raise TeacherForcingError(f"GHR identifier 长度越界：{sample_id}")
    if len(logical_codes) < 3:
        raise TeacherForcingError(f"GHR identifier 缺少 TIGER 三层前缀：{sample_id}")
    expected = [
        tokens.target_open,
        *(tokens.logical_to_token[code] for code in logical_codes),
        tokens.target_close,
    ]
    if target_ids != expected:
        raise TeacherForcingError(f"GHR 目标 Token 编码不一致：{sample_id}")

    prefix = tuple(int(value) for value in logical_codes[:3])
    if collision_prefixes is None:
        group = f"identifier_length_{len(logical_codes)}"
    else:
        group = "base_collision" if prefix in collision_prefixes else "base_singleton"
    suffix_count = len(logical_codes) - 3
    return EncodedExample(
        sample_id=sample_id,
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_ids=tuple((*expected, tokens.eos)),
        position_names=(
            "target_open",
            "sid_1",
            "sid_2",
            "sid_3",
            *(f"suffix_{position}" for position in range(1, suffix_count + 1)),
            "target_close",
            "eos",
        ),
        semantic_indices=(1, 2, 3),
        identifier_indices=tuple(range(1, len(logical_codes) + 1)),
        group=group,
        conditional_tail_indices=tuple(range(4, len(expected))),
    )


def encode_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
    ghr_collision_prefixes: set[tuple[int, int, int]] | None = None,
) -> list[EncodedExample]:
    encoders = {
        "tiger": encode_tiger_example,
        "gnpr": encode_gnpr_example,
        "genpoi": encode_genpoi_example,
        "ghr": encode_ghr_example,
    }
    try:
        encoder = encoders[method]
    except KeyError as error:
        raise TeacherForcingError(f"未知 teacher-forcing 方法：{method}") from error
    examples: list[EncodedExample] = []
    for record in records:
        kwargs: dict[str, Any] = {}
        if method == "ghr":
            kwargs["collision_prefixes"] = ghr_collision_prefixes
        examples.append(
            encoder(
                record,
                tokenizer=tokenizer,
                template=template,
                tokens=tokens,
                cutoff_len=cutoff_len,
                **kwargs,
            )
        )
    return examples


def evaluate_batch(
    examples: Sequence[EncodedExample],
    *,
    model: Any,
    tokenizer: Any,
    metrics: dict[str, Any],
) -> None:
    import torch

    combined = [list((*example.prompt_ids, *example.target_ids)) for example in examples]
    maximum = max(len(values) for values in combined)
    input_ids = torch.full(
        (len(examples), maximum),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    batch_indices: list[int] = []
    prediction_positions: list[int] = []
    target_ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    for batch_index, (example, values) in enumerate(zip(examples, combined)):
        padding = maximum - len(values)
        input_ids[batch_index, padding:] = torch.tensor(values, device=model.device)
        attention_mask[batch_index, padding:] = 1
        start = len(target_ids)
        for target_index, token_id in enumerate(example.target_ids):
            batch_indices.append(batch_index)
            prediction_positions.append(
                padding + len(example.prompt_ids) - 1 + target_index
            )
            target_ids.append(int(token_id))
        offsets.append((start, len(target_ids)))

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        selected = output.logits[
            torch.tensor(batch_indices, device=model.device),
            torch.tensor(prediction_positions, device=model.device),
        ]
        top1, top10, nll = score_target_logits(
            selected,
            torch.tensor(target_ids, device=model.device),
        )
    top1_values = top1.cpu().tolist()
    top10_values = top10.cpu().tolist()
    nll_values = nll.cpu().tolist()
    predicted_ids = selected.argmax(dim=-1).cpu().tolist()
    for example, (start, stop) in zip(examples, offsets):
        update_teacher_forcing_metrics(
            metrics,
            position_names=example.position_names,
            top1_correct=top1_values[start:stop],
            top10_correct=top10_values[start:stop],
            nll_values=nll_values[start:stop],
            semantic_indices=example.semantic_indices,
            identifier_indices=example.identifier_indices,
            group=example.group,
            context_indices=example.context_indices,
            conditional_tail_indices=example.conditional_tail_indices,
        )
        for position_name, predicted_id in zip(
            example.position_names,
            predicted_ids[start:stop],
        ):
            if position_name not in {"target_close", "eos"}:
                continue
            predicted_token = tokenizer.convert_ids_to_tokens(int(predicted_id))
            if not isinstance(predicted_token, str):
                predicted_token = f"<TOKEN_ID_{int(predicted_id)}>"
            position_counts = metrics.setdefault(
                "structural_top1_predictions",
                {},
            ).setdefault(position_name, {"all": {}, "groups": {}})
            position_counts["all"][predicted_token] = (
                position_counts["all"].get(predicted_token, 0) + 1
            )
            group_counts = position_counts["groups"].setdefault(
                example.group,
                {},
            )
            group_counts[predicted_token] = group_counts.get(predicted_token, 0) + 1
    del output, selected, input_ids, attention_mask


def validate_checkpoint(checkpoint: Path, expected_epoch: float) -> dict[str, Any]:
    state_path = checkpoint / "trainer_state.json"
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    for path in (state_path, config_path, model_path):
        if not path.is_file():
            raise TeacherForcingError(f"checkpoint 缺少文件：{path.name}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    epoch = float(state.get("epoch", -1))
    if abs(epoch - expected_epoch) > 0.001:
        raise TeacherForcingError(f"checkpoint epoch {epoch} != {expected_epoch}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {"epoch": epoch, "vocab_size": int(config["vocab_size"])}


def main() -> int:
    args = parse_args()
    try:
        data_file = resolve(args.data_file)
        checkpoint = resolve(args.checkpoint)
        tokenizer_path = resolve(args.tokenizer)
        output_dir = resolve(args.output_dir)
        identifier_dir = (
            resolve(args.identifier_dir) if args.identifier_dir is not None else None
        )
        token_source_file = (
            resolve(args.token_source_file)
            if args.token_source_file is not None
            else None
        )
        tiger_c_positive_reference_file = (
            resolve(args.tiger_c_positive_reference_file)
            if args.tiger_c_positive_reference_file is not None
            else None
        )
        if args.method == "ghr" and (
            identifier_dir is None or token_source_file is None
        ):
            raise TeacherForcingError(
                "--method ghr 必须同时提供 --identifier-dir 与 --token-source-file"
            )
        if tiger_c_positive_reference_file is not None and args.method != "ghr":
            raise TeacherForcingError(
                "--tiger-c-positive-reference-file 当前只适用于 --method ghr"
            )
        if args.checkpoint_rows <= 0:
            raise TeacherForcingError("--checkpoint-rows 必须为正整数")
        if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
            raise TeacherForcingError("--smoke-limit 必须位于 [1, 100]")
        records = load_records(data_file)
        if len(records) != 10_000:
            raise TeacherForcingError("正式 teacher-forcing 数据必须恰好为 10,000 条")
        selection_metadata: dict[str, Any] | None = None
        if tiger_c_positive_reference_file is not None:
            reference_records = load_records(tiger_c_positive_reference_file)
            records, selection_metadata = select_tiger_c_positive_records(
                records,
                reference_records,
                token_capacities=args.token_capacities,
            )
            selection_metadata.update(
                {
                    "reference_file": str(tiger_c_positive_reference_file),
                    "reference_sha256": sha256_file(tiger_c_positive_reference_file),
                }
            )
        if args.smoke_limit is not None:
            records = records[: args.smoke_limit]
        checkpoint_metadata = validate_checkpoint(checkpoint, args.expected_epoch)
        tokenizer, template = load_lf_tokenizer_and_template(
            tokenizer_path,
            project_root=PROJECT_ROOT,
        )
        if args.method == "tiger":
            tokens, token_metadata = load_tiger_token_ids(
                tokenizer_path,
                tokenizer,
                token_capacities=args.token_capacities,
                mapping_filename=args.token_mapping_filename,
            )
        elif args.method == "gnpr":
            tokens, token_metadata = load_gnpr_token_ids(tokenizer_path, tokenizer)
        elif args.method == "ghr":
            assert identifier_dir is not None and token_source_file is not None
            tokens, token_metadata = load_ghr_token_ids(
                tokenizer_path,
                tokenizer,
                identifier_dir=identifier_dir,
                token_source_path=token_source_file,
            )
        else:
            tokens, token_metadata = load_pid_token_ids(tokenizer_path)
        if len(tokenizer) != checkpoint_metadata["vocab_size"]:
            raise TeacherForcingError("checkpoint 与 tokenizer 词表大小不一致")

        ghr_collision_prefixes: set[tuple[int, int, int]] | None = None
        ghr_identifier_metadata: dict[str, Any] | None = None
        if args.method == "ghr":
            assert identifier_dir is not None
            ghr_index, ghr_identifier_metadata = load_ghr_id_index(identifier_dir)
            prefixes = np.asarray(ghr_index.codes[:, :3], dtype=np.int32)
            _, inverse, counts = np.unique(
                prefixes,
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            ghr_collision_prefixes = {
                tuple(int(value) for value in prefixes[row])
                for row in np.flatnonzero(counts[inverse] > 1)
            }

        model_hash = (
            None
            if args.skip_model_hash
            else sha256_file(checkpoint / "model.safetensors")
        )
        config = {
            "schema_version": "sid-teacher-forcing-config-v1",
            "method": args.method,
            "data_file": str(data_file),
            "data_sha256": sha256_file(data_file),
            "rows": len(records),
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": checkpoint_metadata["epoch"],
            "checkpoint_sha256": model_hash,
            "tokenizer": token_metadata,
            "ghr_identifier": ghr_identifier_metadata,
            "sample_selection": selection_metadata,
            "cutoff_len": args.cutoff_len,
            "initial_batch_size": args.batch_size,
            "teacher_forcing": True,
            "beam_search": False,
            "target_positions_only": True,
            "script_sha256": sha256_file(Path(__file__)),
        }
        signature = canonical_sha256(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("signature") != signature:
                raise TeacherForcingError("已有 teacher-forcing 运行配置变化")
        else:
            progress = {
                "schema_version": "sid-teacher-forcing-progress-v1",
                "status": "running",
                "signature": signature,
                "config": config,
                "next_line": 0,
                "metrics": empty_teacher_forcing_metrics(),
                "inference_seconds": 0.0,
                "actual_batch_size": args.batch_size,
                "peak_memory_bytes": 0,
            }
            atomic_json(progress_path, progress)

        model = load_generation_model(
            checkpoint,
            expected_vocab_size=len(tokenizer),
        )
        import torch

        torch.cuda.reset_peak_memory_stats(model.device)
        allowed = [value for value in (32, 16, 8, 4, 2) if value <= args.batch_size]
        batch_size = min(int(progress["actual_batch_size"]), allowed[0])
        next_report = (
            (int(progress["next_line"]) // args.checkpoint_rows) + 1
        ) * args.checkpoint_rows
        while int(progress["next_line"]) < len(records):
            start = int(progress["next_line"])
            stop = min(start + batch_size, len(records))
            examples = encode_batch(
                records[start:stop],
                method=args.method,
                tokenizer=tokenizer,
                template=template,
                tokens=tokens,
                cutoff_len=args.cutoff_len,
                ghr_collision_prefixes=ghr_collision_prefixes,
            )
            started = time.monotonic()
            try:
                evaluate_batch(
                    examples,
                    model=model,
                    tokenizer=tokenizer,
                    metrics=progress["metrics"],
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                smaller = [value for value in allowed if value < batch_size]
                if not smaller:
                    raise TeacherForcingError("batch_size=2 仍然 OOM")
                batch_size = smaller[0]
                progress["actual_batch_size"] = batch_size
                continue
            progress["inference_seconds"] += time.monotonic() - started
            progress["next_line"] = stop
            progress["actual_batch_size"] = batch_size
            progress["peak_memory_bytes"] = max(
                int(progress["peak_memory_bytes"]),
                int(torch.cuda.max_memory_allocated(model.device)),
            )
            if stop >= next_report or stop == len(records):
                atomic_json(progress_path, progress)
                print(f"[{args.method}] {stop:,}/{len(records):,}", flush=True)
                next_report += args.checkpoint_rows

        result = {
            "schema_version": "sid-teacher-forcing-result-v1",
            "status": "completed",
            "config": config,
            "metrics": finalize_teacher_forcing_metrics(progress["metrics"]),
            "structural_top1_predictions": progress["metrics"].get(
                "structural_top1_predictions",
                {},
            ),
            "performance": {
                "inference_seconds": progress["inference_seconds"],
                "samples_per_second": len(records) / progress["inference_seconds"],
                "peak_memory_bytes": progress["peak_memory_bytes"],
                "actual_batch_size": progress["actual_batch_size"],
            },
        }
        progress["status"] = "completed"
        atomic_json(progress_path, progress)
        atomic_json(output_dir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (
        TeacherForcingError,
        GenerativeEvalError,
        PidTrieError,
        TigerEvalError,
        GnprEvalError,
        GhrEvalError,
        OSError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
