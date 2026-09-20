"""Exact full-catalog prefix constraints for both variable-length QG Final IDs."""

from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache
from typing import Any

from qg_prqk.sft.evaluation import FinalIdIndex
from qg_prqk.sft.evaluation_data import SftEvaluationError


class FinalIdPrefixIndex:
    """Represent a trie as sorted terminal paths with bounded prefix caching."""

    def __init__(self, index: FinalIdIndex) -> None:
        if not index.row_by_tokens or len(index.row_by_tokens) != index.poi_count:
            raise SftEvaluationError("合法路径目录为空或 POI 数量不一致")
        self.eos = index.eos
        self.paths = sorted(
            (index.target_open, *tokens, index.target_close, index.eos)
            for tokens in index.row_by_tokens
        )
        self.max_length = max(map(len, self.paths))
        self.upper_token = max(max(path) for path in self.paths) + 1
        if len(set((index.target_open, index.target_close, index.eos))) != 3:
            raise SftEvaluationError("目标边界与 EOS 必须互不相同")
        for path in self.paths:
            if len(path) not in (index.base_width + 3, index.base_width + 4):
                raise SftEvaluationError("Final ID 长度不符合可选末位 Dedup 协议")
            if any(
                t in path[1:-2]
                for t in (index.target_open, index.target_close, index.eos)
            ):
                raise SftEvaluationError("内部 ID 不能包含目标边界或 EOS")
        self.allowed_next = lru_cache(maxsize=65536)(self._allowed_next)

    def _allowed_next(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        if self.eos in prefix:
            stop = prefix.index(self.eos) + 1
            path = prefix[:stop]
            row = bisect_left(self.paths, path)
            if (
                row >= len(self.paths)
                or self.paths[row] != path
                or any(t != self.eos for t in prefix[stop:])
            ):
                raise SftEvaluationError("EOS 前不是完整合法 ID，或 EOS 后有非法 Token")
            return (self.eos,)
        row = bisect_left(self.paths, prefix)
        end = bisect_left(self.paths, (*prefix, self.upper_token))
        if row == end:
            raise SftEvaluationError("生成前缀不在全目录中，禁止无约束回退")
        children = []
        while row < end:
            token = self.paths[row][len(prefix)]
            children.append(token)
            # Skip the entire child subtree instead of scanning its POIs.
            row = bisect_left(self.paths, (*prefix, token + 1), row, end)
        return tuple(children)


class PrefixConstraint:
    """Ignore left-padded prompts and constrain only newly generated tokens."""

    def __init__(self, trie: FinalIdPrefixIndex, prompt_width: int) -> None:
        if prompt_width <= 0:
            raise SftEvaluationError("prompt_width 必须为正数")
        self.trie = trie
        self.prompt_width = prompt_width

    def __call__(self, _batch_id: int, input_ids: Any) -> list[int]:
        values = input_ids[self.prompt_width :]
        if hasattr(values, "tolist"):
            values = values.tolist()
        return list(self.trie.allowed_next(tuple(int(v) for v in values)))


class ConstrainedGenerationModel:
    """Add one generation constraint while reusing the frozen evaluator unchanged."""

    def __init__(self, model: Any, trie: FinalIdPrefixIndex) -> None:
        self.model = model
        self.trie = trie
        self.device = model.device

    def generate(self, **kwargs: Any) -> Any:
        if (
            "prefix_allowed_tokens_fn" in kwargs
            or kwargs["max_new_tokens"] < self.trie.max_length
        ):
            raise SftEvaluationError("重复的路径约束或不足以完整闭合的生成长度")
        return self.model.generate(
            **kwargs,
            prefix_allowed_tokens_fn=PrefixConstraint(
                self.trie, kwargs["input_ids"].shape[1]
            ),
        )
