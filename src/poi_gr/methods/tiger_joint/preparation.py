"""Prepare strictly SID-free TIGER-Joint records for dynamic training."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from poi_gr.methods.tiger_joint.catalog import PoiEmbeddingStore
from poi_gr.methods.tiger_joint.data import (
    DynamicSidTemplate,
    JointTigerBatch,
    SidTokenLayout,
    TigerJointDataError,
)
from poi_gr.methods.tiger_joint.dataset import (
    DATA_SCHEMA_VERSION,
    DYNAMIC_HISTORY_IDENTIFIER,
    DYNAMIC_TARGET_IDENTIFIER,
    build_joint_special_tokens,
)


DYNAMIC_HISTORY_IDENTIFIER_RE = re.compile(re.escape(DYNAMIC_HISTORY_IDENTIFIER))
ANY_SID_TOKEN_RE = re.compile(r"<(?:S[123]|C|D)_\d+>")
FORBIDDEN_LEGACY_MARKERS = (
    "<POI_TIGER_ID>",
    "</POI_TIGER_ID>",
    "<C_",
    "<D_",
)
FORBIDDEN_LEGACY_FIELDS = {
    "target_tiger_id_key",
    "tiger_id",
    "tiger_ids",
    "identifier_codes",
    "collision_code",
}


class TigerJointPreparationError(TigerJointDataError):
    """Raised when SID-free data cannot form a dynamic example."""


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise TigerJointPreparationError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointPreparationError(f"{name} 不是合法 JSON：{path}") from error
    if not isinstance(value, dict):
        raise TigerJointPreparationError(f"{name} 必须是 JSON object：{path}")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TigerJointPreparationError(f"{name} 必须是正整数")
    return int(value)


def _require_nonnegative_row(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise TigerJointPreparationError(f"{name} 必须是非负整数 BGE 行号")
    return int(value)


@dataclass(frozen=True)
class DynamicTextRecord:
    """A SID-free behavior sample carrying only explicit BGE row identities."""

    sample_id: str
    split: str
    user_content: str
    target_content: str
    history_poi_rows: tuple[int, ...]
    target_poi_row: int


@dataclass(frozen=True)
class DynamicTokenIds:
    """Atomic structural and placeholder token IDs used to locate slots."""

    history_open: int
    history_close: int
    current_close: int
    target_open: int
    target_close: int
    placeholders: tuple[int, int, int]


@dataclass(frozen=True)
class DynamicTigerExample:
    """One unpadded tokenized sample with BGE rows and dynamic SID positions."""

    sample_id: str
    split: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    query_state_position: int
    history_poi_rows: tuple[int, ...]
    history_sid_positions: tuple[tuple[int, int, int], ...]
    target_poi_row: int
    target_sid_positions: tuple[int, int, int]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def parse_sid_free_record(
    record: Mapping[str, Any],
    *,
    expected_split: str,
) -> DynamicTextRecord:
    """Validate the new row-based schema without interpreting old identifiers."""

    if expected_split not in {"train", "valid"}:
        raise TigerJointPreparationError("联合训练准备只允许 train/valid")
    if record.get("schema_version") != DATA_SCHEMA_VERSION:
        raise TigerJointPreparationError(
            "只接受 tiger-joint-sid-free-data-v1；旧 TIGER JSONL 禁止进入"
        )
    legacy_fields = FORBIDDEN_LEGACY_FIELDS.intersection(record)
    if legacy_fields:
        raise TigerJointPreparationError(
            "SID-free 样本包含旧 identifier 字段：" + ", ".join(sorted(legacy_fields))
        )
    if record.get("split") != expected_split:
        raise TigerJointPreparationError("样本 split 与输入文件不一致")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise TigerJointPreparationError("样本缺少 sample_id")
    history_length = record.get("history_length")
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or not 0 <= history_length <= 10
    ):
        raise TigerJointPreparationError("history_length 必须位于 0～10")
    raw_history_rows = record.get("history_poi_rows")
    if (
        not isinstance(raw_history_rows, list)
        or len(raw_history_rows) != history_length
    ):
        raise TigerJointPreparationError(
            "history_poi_rows 数量必须与 history_length 一致"
        )
    history_poi_rows = tuple(
        _require_nonnegative_row(value, f"history_poi_rows[{index}]")
        for index, value in enumerate(raw_history_rows)
    )
    target_poi_row = _require_nonnegative_row(
        record.get("target_poi_row"), "target_poi_row"
    )
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], dict)
        or not isinstance(messages[1], dict)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise TigerJointPreparationError("样本 Messages 必须是 user+assistant")
    user_content = str(messages[0]["content"])
    target_content = str(messages[1]["content"])
    combined_text = user_content + target_content
    if any(marker in combined_text for marker in FORBIDDEN_LEGACY_MARKERS):
        raise TigerJointPreparationError("SID-free 模板包含旧 TIGER SID/C 标记")
    if user_content.count(DYNAMIC_HISTORY_IDENTIFIER) != history_length:
        raise TigerJointPreparationError("历史动态槽位数量与 history_length 不一致")
    if (
        user_content.count("<POI_SID>") != history_length
        or user_content.count("</POI_SID>") != history_length
    ):
        raise TigerJointPreparationError("历史 POI_SID wrapper 数量无效")
    if user_content.count("<CURRENT>") != 1 or user_content.count("</CURRENT>") != 1:
        raise TigerJointPreparationError("CURRENT wrapper 必须恰好出现一次")
    without_dynamic_slots = DYNAMIC_HISTORY_IDENTIFIER_RE.sub("", user_content)
    if ANY_SID_TOKEN_RE.search(without_dynamic_slots):
        raise TigerJointPreparationError("动态 User 模板的槽位外仍包含 SID Token")
    if target_content != DYNAMIC_TARGET_IDENTIFIER:
        raise TigerJointPreparationError("Assistant 必须是严格的动态三级目标模板")
    return DynamicTextRecord(
        sample_id=sample_id,
        split=expected_split,
        user_content=user_content,
        target_content=target_content,
        history_poi_rows=history_poi_rows,
        target_poi_row=target_poi_row,
    )


def parse_sid_free_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_split: str,
) -> list[DynamicTextRecord]:
    """Validate one bounded row-based record batch in input order."""

    return [
        parse_sid_free_record(record, expected_split=expected_split)
        for record in records
    ]


def _require_atomic_token(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or encoded != [token_id]
    ):
        raise TigerJointPreparationError(f"Tokenizer Token 不是原子项：{token}")
    return token_id


def load_dynamic_token_ids(tokenizer: Any) -> DynamicTokenIds:
    """Load the fresh structural tokens used by dynamic templates."""

    return DynamicTokenIds(
        history_open=_require_atomic_token(tokenizer, "<POI_SID>"),
        history_close=_require_atomic_token(tokenizer, "</POI_SID>"),
        current_close=_require_atomic_token(tokenizer, "</CURRENT>"),
        target_open=_require_atomic_token(tokenizer, "<TARGET_POI>"),
        target_close=_require_atomic_token(tokenizer, "</TARGET_POI>"),
        placeholders=(
            _require_atomic_token(tokenizer, "<S1_0>"),
            _require_atomic_token(tokenizer, "<S2_0>"),
            _require_atomic_token(tokenizer, "<S3_0>"),
        ),
    )


def build_sid_token_layout(
    tokenizer: Any,
    codebook_sizes: Sequence[int],
) -> SidTokenLayout:
    """Resolve all fresh S1/S2/S3 vocabulary IDs in code order."""

    if len(codebook_sizes) != 3:
        raise TigerJointPreparationError("codebook_sizes 必须恰好包含三层")
    levels: list[tuple[int, ...]] = []
    for level_index, raw_size in enumerate(codebook_sizes, start=1):
        size = _require_positive_int(raw_size, f"codebook_sizes[{level_index - 1}]")
        levels.append(
            tuple(
                _require_atomic_token(tokenizer, f"<S{level_index}_{code}>")
                for code in range(size)
            )
        )
    return SidTokenLayout(levels)


def _find_wrapped_slots(
    token_ids: Sequence[int],
    *,
    open_token: int,
    close_token: int,
    placeholders: tuple[int, int, int],
    name: str,
) -> tuple[tuple[int, int, int], ...]:
    open_positions = [
        index for index, token_id in enumerate(token_ids) if token_id == open_token
    ]
    slots: list[tuple[int, int, int]] = []
    expected = (*placeholders, close_token)
    for open_position in open_positions:
        end = open_position + 5
        if (
            end > len(token_ids)
            or tuple(token_ids[open_position + 1 : end]) != expected
        ):
            raise TigerJointPreparationError(f"{name} 动态槽位结构无效")
        slots.append((open_position + 1, open_position + 2, open_position + 3))
    return tuple(slots)


def tokenize_dynamic_record(
    text_record: DynamicTextRecord,
    *,
    tokenizer: Any,
    template: Any,
    token_ids: DynamicTokenIds,
    cutoff_len: int = 1024,
) -> DynamicTigerExample:
    """Tokenize exactly like qwen3_nothink and retain every dynamic position."""

    if cutoff_len != 1024:
        raise TigerJointPreparationError("新的历史联合训练 cutoff_len 必须是 1024")
    source_ids, target_ids = template.encode_oneturn(
        tokenizer,
        [
            {"role": "user", "content": text_record.user_content},
            {"role": "assistant", "content": text_record.target_content},
        ],
        system=None,
        tools=None,
    )
    if not isinstance(source_ids, list) or not isinstance(target_ids, list):
        raise TigerJointPreparationError("qwen3_nothink 未返回 Token ID list")
    input_ids = tuple(int(value) for value in (*source_ids, *target_ids))
    if len(input_ids) > cutoff_len:
        raise TigerJointPreparationError(
            f"完整 Source+Target 长度 {len(input_ids)} 超过 1024"
        )
    history_positions = _find_wrapped_slots(
        source_ids,
        open_token=token_ids.history_open,
        close_token=token_ids.history_close,
        placeholders=token_ids.placeholders,
        name="历史",
    )
    if len(history_positions) != len(text_record.history_poi_rows):
        raise TigerJointPreparationError("Tokenized 历史槽位数量与 POI 行数不一致")
    current_positions = [
        index
        for index, token_id in enumerate(source_ids)
        if token_id == token_ids.current_close
    ]
    if len(current_positions) != 1:
        raise TigerJointPreparationError("Tokenized </CURRENT> 必须恰好出现一次")
    relative_targets = _find_wrapped_slots(
        target_ids,
        open_token=token_ids.target_open,
        close_token=token_ids.target_close,
        placeholders=token_ids.placeholders,
        name="目标",
    )
    if len(relative_targets) != 1:
        raise TigerJointPreparationError("Tokenized 目标槽位必须恰好出现一次")
    target_positions = tuple(
        len(source_ids) + position for position in relative_targets[0]
    )
    labels = tuple([-100] * len(source_ids) + [int(value) for value in target_ids])
    example = DynamicTigerExample(
        sample_id=text_record.sample_id,
        split=text_record.split,
        input_ids=input_ids,
        labels=labels,
        query_state_position=current_positions[0],
        history_poi_rows=text_record.history_poi_rows,
        history_sid_positions=history_positions,
        target_poi_row=text_record.target_poi_row,
        target_sid_positions=(
            target_positions[0],
            target_positions[1],
            target_positions[2],
        ),
    )
    _validate_single_example(example, cutoff_len=cutoff_len)
    return example


def _validate_single_example(
    example: DynamicTigerExample,
    *,
    cutoff_len: int,
) -> None:
    length = example.sequence_length
    DynamicSidTemplate(
        input_ids=torch.tensor([example.input_ids], dtype=torch.long),
        labels=torch.tensor([example.labels], dtype=torch.long),
        attention_mask=torch.ones((1, length), dtype=torch.long),
        query_state_positions=torch.tensor(
            [example.query_state_position], dtype=torch.long
        ),
        history_sample_indices=torch.zeros(
            len(example.history_poi_rows), dtype=torch.long
        ),
        history_poi_rows=torch.tensor(example.history_poi_rows, dtype=torch.long),
        history_sid_positions=torch.tensor(
            example.history_sid_positions, dtype=torch.long
        ).reshape(-1, 3),
        target_poi_rows=torch.tensor([example.target_poi_row], dtype=torch.long),
        target_sid_positions=torch.tensor(
            [example.target_sid_positions], dtype=torch.long
        ),
        cutoff_len=cutoff_len,
    )


def load_dynamic_examples(
    path: Path,
    *,
    split: str,
    max_rows: int,
    tokenizer: Any,
    template: Any,
    token_ids: DynamicTokenIds,
    cutoff_len: int = 1024,
) -> list[DynamicTigerExample]:
    """Stream a bounded SID-free Train/Valid prefix for smoke testing."""

    if max_rows <= 0:
        raise TigerJointPreparationError("max_rows 必须大于 0")
    if path.name == "test.jsonl":
        raise TigerJointPreparationError("联合训练禁止读取 test.jsonl")
    examples: list[DynamicTigerExample] = []
    with path.resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TigerJointPreparationError(
                    f"{split} 第 {line_number} 行 JSON 解析失败"
                ) from error
            if not isinstance(record, dict):
                raise TigerJointPreparationError(
                    f"{split} 第 {line_number} 行必须是 JSON object"
                )
            examples.append(
                tokenize_dynamic_record(
                    parse_sid_free_record(record, expected_split=split),
                    tokenizer=tokenizer,
                    template=template,
                    token_ids=token_ids,
                    cutoff_len=cutoff_len,
                )
            )
            if len(examples) == max_rows:
                break
    if len(examples) != max_rows:
        raise TigerJointPreparationError(
            f"{split} 只读取到 {len(examples)} 行，少于要求的 {max_rows}"
        )
    return examples


def collate_dynamic_examples(
    examples: Sequence[DynamicTigerExample],
    *,
    embedding_store: PoiEmbeddingStore,
    catalog_rows: Sequence[int] | np.ndarray,
    pad_token_id: int,
    device: torch.device | str,
    cutoff_len: int = 1024,
) -> JointTigerBatch:
    """Right-pad examples and gather bounded batches from the BGE mmap."""

    if not examples:
        raise TigerJointPreparationError("行为 batch 不能为空")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TigerJointPreparationError("pad_token_id 必须是整数")
    max_length = max(example.sequence_length for example in examples)
    if max_length > cutoff_len:
        raise TigerJointPreparationError("行为 batch 包含超过 cutoff_len 的样本")
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    query_positions: list[int] = []
    target_rows: list[int] = []
    target_positions: list[tuple[int, int, int]] = []
    history_sample_indices: list[int] = []
    history_rows: list[int] = []
    history_positions: list[tuple[int, int, int]] = []
    for sample_index, example in enumerate(examples):
        length = example.sequence_length
        input_ids[sample_index, :length] = torch.tensor(
            example.input_ids, dtype=torch.long
        )
        labels[sample_index, :length] = torch.tensor(example.labels, dtype=torch.long)
        attention_mask[sample_index, :length] = 1
        query_positions.append(example.query_state_position)
        target_rows.append(example.target_poi_row)
        target_positions.append(example.target_sid_positions)
        history_sample_indices.extend([sample_index] * len(example.history_poi_rows))
        history_rows.extend(example.history_poi_rows)
        history_positions.extend(example.history_sid_positions)

    catalog_array = np.ascontiguousarray(catalog_rows, dtype=np.int64)
    resolved_device = torch.device(device)
    template = DynamicSidTemplate(
        input_ids=input_ids.to(resolved_device),
        labels=labels.to(resolved_device),
        attention_mask=attention_mask.to(resolved_device),
        query_state_positions=torch.tensor(
            query_positions, dtype=torch.long, device=resolved_device
        ),
        history_sample_indices=torch.tensor(
            history_sample_indices, dtype=torch.long, device=resolved_device
        ),
        history_poi_rows=torch.tensor(
            history_rows, dtype=torch.long, device=resolved_device
        ),
        history_sid_positions=torch.tensor(
            history_positions, dtype=torch.long, device=resolved_device
        ).reshape(-1, 3),
        target_poi_rows=torch.tensor(
            target_rows, dtype=torch.long, device=resolved_device
        ),
        target_sid_positions=torch.tensor(
            target_positions, dtype=torch.long, device=resolved_device
        ),
        cutoff_len=cutoff_len,
    )
    return JointTigerBatch(
        template=template,
        history_embeddings=torch.from_numpy(embedding_store.gather(history_rows)).to(
            resolved_device
        ),
        target_embeddings=torch.from_numpy(embedding_store.gather(target_rows)).to(
            resolved_device
        ),
        catalog_poi_rows=torch.from_numpy(catalog_array).to(resolved_device),
        catalog_embeddings=torch.from_numpy(embedding_store.gather(catalog_array)).to(
            resolved_device
        ),
    )


def load_fresh_tokenizer_and_template(
    base_model_dir: Path,
    *,
    project_root: Path,
    codebook_sizes: Sequence[int] = (1024, 1024, 1024),
) -> tuple[Any, Any]:
    """Extend vanilla Qwen in memory with the new collision-free vocabulary."""

    factory_source = project_root.resolve() / "third_party" / "LLaMA-Factory" / "src"
    if not factory_source.is_dir():
        raise TigerJointPreparationError("缺少本地 LLaMA-Factory source")
    if str(factory_source) not in sys.path:
        sys.path.insert(0, str(factory_source))
    try:
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.model.patcher import patch_tokenizer
        from transformers.models.qwen2.tokenization_qwen2_fast import (
            Qwen2TokenizerFast,
        )
    except ImportError as error:
        raise TigerJointPreparationError("无法导入本地 Qwen/LLaMA-Factory") from error

    base_model_dir = base_model_dir.resolve()
    tokenizer_config = _load_json_object(
        base_model_dir / "tokenizer_config.json", "Tokenizer config"
    )
    if tokenizer_config.get("tokenizer_class") != "Qwen2Tokenizer":
        raise TigerJointPreparationError("联合训练只接受 vanilla Qwen2 tokenizer")
    if base_model_dir.name != "Qwen3-0.6B":
        raise TigerJointPreparationError(
            "fresh 联合训练 tokenizer 必须直接来自 models/Qwen3-0.6B"
        )
    model_args = ModelArguments(
        model_name_or_path=str(base_model_dir),
        use_fast_tokenizer=True,
        trust_remote_code=False,
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        base_model_dir,
        local_files_only=True,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
    )
    patch_tokenizer(tokenizer, model_args)
    inventory = build_joint_special_tokens(codebook_sizes=codebook_sizes)
    tokens = inventory["additional_tokens"]
    original_vocab_size = len(tokenizer)
    added_count = tokenizer.add_tokens(tokens, special_tokens=False)
    if added_count != len(tokens) or len(tokenizer) != original_vocab_size + len(
        tokens
    ):
        raise TigerJointPreparationError(
            "vanilla Qwen 已含联合 Token 或 fresh vocabulary 添加不完整"
        )
    if any(token.startswith("<C_") for token in tokenizer.get_added_vocab()):
        raise TigerJointPreparationError("fresh tokenizer 意外包含旧 C Token")
    data_args = DataArguments(
        template="qwen3_nothink",
        train_on_prompt=False,
        cutoff_len=1024,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, template
