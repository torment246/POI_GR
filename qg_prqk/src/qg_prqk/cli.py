"""Single public command-line interface for QG-PRQK."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Command:
    """A lazily loaded public command."""

    module: str
    summary: str


COMMAND_GROUPS: tuple[tuple[str, tuple[tuple[str, Command], ...]], ...] = (
    (
        "数据与 Query 监督",
        (
            ("preflight", Command("preflight", "核验冻结输入与字段契约")),
            ("build-active-poi", Command("active_poi", "按冻结行序构建 active POI 资产")),
            ("build-query-stats", Command("query_statistics", "构建 Train-only Query–POI 统计")),
            ("build-query-supervision", Command("query_supervision", "构建 D0/D1/D2/D3 Query 监督")),
            ("cache-query-embeddings", Command("query_embeddings", "缓存归一化 Query embedding")),
            ("build-query-graph", Command("query_graph", "构建 Query 图与分层 Query view")),
        ),
    ),
    (
        "Query Adapter",
        (
            ("smoke-query-adapter", Command("smoke_query_adapter", "用合成向量检查 Adapter 可训练性")),
            ("train-query-adapter", Command("train_query_adapter", "从预构建张量训练 Adapter")),
            ("gate-query-adapter", Command("query_adapter_gate", "运行历史 50k 工程 Gate")),
            ("select-query-adapter", Command("select_query_adapter", "全量选择并重训 D3 Adapter")),
        ),
    ),
    (
        "Semantic ID",
        (
            ("inspect-sid-config", Command("inspect_sid_config", "只读检查 SID 配置继承与哈希")),
            ("inspect-relational-inputs", Command("inspect_relational_inputs", "只读检查 S1/S2/S3 输入 header")),
            ("build-base-codebook", Command("base_codebook", "训练 POI-only PRQ-KMeans 基础码本")),
            ("build-relational-codebook", Command("relational_codebook", "用 Query/类别图细化 S1/S2")),
            ("evaluate-relational-codebook", Command("evaluate_relational_codebook", "比较基础与关系化 S1/S2")),
            ("build-local-codebook", Command("local_codebook", "在 GID parent 内训练局部 S3")),
            ("build-local-codebook-nogid", Command("local_codebook_nogid", "在 S1/S2 parent 内训练无 GID 的 S3")),
            ("evaluate-sid", Command("evaluate_sid", "评估并发布 canonical 三层 SID")),
            ("evaluate-sid-nogid", Command("evaluate_sid_nogid", "评估无 GID 三层 SID 对照")),
            ("build-final-identifiers", Command("final_identifiers", "追加末位 dedup 并构建唯一 Final ID")),
            ("visualize-sid", Command("visualize_sid", "生成或复核三种 SID 可视化")),
            ("visualize-sid-categories", Command("visualize_sid_categories", "共同五类 t-SNE 与类别/区域前缀图")),
        ),
    ),
    (
        "SFT",
        (
            ("build-sft-data", Command("sft_data", "构建两种 Final ID 的 history10 Messages")),
            ("prepare-sft-vocab", Command("sft_vocabulary", "构建两分支共享扩展词表")),
            ("prepare-sft-cache", Command("sft_cache", "全量零截断预检并构建 Tokenized Cache")),
            ("train-sft", Command("train_sft", "通过 LLaMA-Factory 训练指定 SFT 分支")),
            ("evaluate-sft", Command("evaluate_sft", "双卡评测两个 SFT：固定 10k 与四类泛化集")),
            ("evaluate-sft-test", Command("evaluate_sft_test", "双卡评测两个 SFT：最后一天全量 Test")),
        ),
    ),
    (
        "冻结诊断",
        (
            ("diagnose-base-codebook", Command("diagnose_base_codebook", "复核基础码本 hard endpoint")),
            ("diagnose-base-topk", Command("diagnose_base_topk", "复核基础码本 Top-k refinement")),
            ("diagnose-relational-codebook", Command("diagnose_relational_codebook", "运行 S1/S2 因素归因")),
            ("diagnose-relational-s2", Command("diagnose_relational_s2", "运行 full S2-only 归因")),
        ),
    ),
)


def _commands() -> dict[str, Command]:
    return {
        name: command
        for _, commands in COMMAND_GROUPS
        for name, command in commands
    }


def format_help() -> str:
    """Return concise help grouped by method responsibility."""

    lines = [
        "用法：python -m qg_prqk <command> [参数]",
        "",
        "QG-PRQK 统一入口。阶段编号仅保留在历史实验协议中。",
    ]
    for group, commands in COMMAND_GROUPS:
        lines.extend(("", f"{group}："))
        width = max(len(name) for name, _ in commands)
        lines.extend(
            f"  {name:<{width}}  {command.summary}" for name, command in commands
        )
    lines.extend(("", "运行 `python -m qg_prqk <command> --help` 查看子命令参数。"))
    return "\n".join(lines)


def _load_handler(command: Command) -> Callable[[list[str] | None], int]:
    module = importlib.import_module(f"qg_prqk.commands.{command.module}")
    handler = getattr(module, "main", None)
    if not callable(handler):
        raise RuntimeError(f"命令模块缺少 main()：{module.__name__}")
    return handler


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one semantic command without duplicating its argument parser."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(format_help())
        return 0
    name, *remaining = arguments
    command = _commands().get(name)
    if command is None:
        print(f"未知命令：{name}\n\n{format_help()}", file=sys.stderr)
        return 2
    original_program = sys.argv[0]
    try:
        sys.argv[0] = f"python -m qg_prqk {name}"
        return _load_handler(command)(remaining)
    finally:
        sys.argv[0] = original_program


if __name__ == "__main__":
    raise SystemExit(main())
