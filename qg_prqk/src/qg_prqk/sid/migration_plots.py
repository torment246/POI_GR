"""Render observed prefix changes and membership flows; never select or modify cases."""

from __future__ import annotations

import html
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from qg_prqk.sid.category_region_plots import _save


VARIANTS = ("A4_GID_PARENT", "A4_NOGID_S1S2_PARENT")
VARIANT_TITLES = ("A0 → A4 GID-parent", "A0 → A4 NoGID")
OUTCOME_LABELS = {
    "improved": "提高",
    "tied": "持平",
    "degraded": "降低",
    "not_comparable": "含单例，不可比",
}
OUTCOME_COLORS = {
    "improved": "#329680",
    "tied": "#b7c3d0",
    "degraded": "#ca7067",
    "not_comparable": "#e8ebee",
}
LEVELS = ("S1", "S1/S2", "S1/S2/S3")


def _outcome_bar(axis: Any, y: float, counts: dict[str, int], total: int) -> None:
    left = 0.0
    for outcome, color in OUTCOME_COLORS.items():
        width = counts[outcome] * 100 / total
        axis.barh(y, width, left=left, height=0.65, color=color, edgecolor="white")
        if width >= 9:
            axis.text(
                left + width / 2,
                y,
                f"{width:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
        left += width


def _example_card(
    axis: Any, case: dict[str, Any] | None, depth: int, outcome: str
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(
        f"{LEVELS[depth - 1]} · 类别同伴一致率{OUTCOME_LABELS[outcome]}",
        loc="left",
        fontsize=13,
        color=OUTCOME_COLORS[outcome] if outcome != "tied" else "#50627a",
    )
    if case is None:
        axis.text(
            0.05, 0.5, "该分组没有符合样例范围的候选\n未改变筛选条件补图", fontsize=12
        )
        return
    anchor = case["anchor"]
    title = textwrap.fill(anchor["displayname"], width=26)
    axis.text(0, 0.96, title, va="top", fontsize=11, color="#24384b")
    axis.text(
        0,
        0.77,
        f"{case['before_prefix']}  →  {case['after_prefix']}",
        fontsize=11,
        color="#24384b",
    )
    category = anchor["category"].split(":", 1)[0] if depth == 1 else anchor["category"]
    axis.text(0, 0.70, f"锚点类别：{category[:32]}", fontsize=9, color="#66717e")
    retained = case["groups"]["retained"]["count"]
    for y, size, other, color, label in (
        (
            0.58,
            case["before_size"],
            case["groups"]["removed"]["count"],
            "#ca7067",
            "A0",
        ),
        (0.40, case["after_size"], case["groups"]["added"]["count"], "#329680", "A4"),
    ):
        axis.text(0, y + 0.065, f"{label} 桶 n={size:,}", fontsize=9)
        shared_width = retained / size
        axis.barh(y, shared_width, height=0.095, color="#487bac", edgecolor="white")
        axis.barh(
            y,
            other / size,
            left=shared_width,
            height=0.095,
            color=color,
            edgecolor="white",
        )
        if shared_width > 0.15:
            axis.text(
                shared_width / 2,
                y,
                str(retained),
                ha="center",
                va="center",
                fontsize=10,
                color="white",
            )
        if other / size > 0.15:
            axis.text(
                shared_width + other / size / 2,
                y,
                str(other),
                ha="center",
                va="center",
                fontsize=10,
                color="white",
            )
    cat, geo = case["category"], case["region"]
    axis.text(
        0, 0.27, f"同类其他 POI：{cat['before']:.1%} → {cat['after']:.1%}", fontsize=11
    )
    axis.text(
        0,
        0.19,
        f"同网格其他 POI：{geo['before']:.1%} → {geo['after']:.1%}",
        fontsize=11,
    )
    axis.text(
        0,
        0.10,
        f"保留 {retained:,} / 移出 {case['groups']['removed']['count']:,} / 移入 {case['groups']['added']['count']:,}",
        fontsize=10,
        color="#435365",
    )
    axis.text(
        0,
        0.02,
        f"本组有 {case['eligible_poi']:,} 个合格锚点；固定种子抽取 1 个",
        fontsize=8.5,
        color="#66717e",
    )


def draw_migration_figures(
    settings: dict[str, Any],
    root: Path,
    data: dict[str, Any],
    examples: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    """Render a full-catalog overview and one three-level case sheet per A4 variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    font = root / settings["font_path"]
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    with plt.rc_context(
        {"font.family": family, "axes.unicode_minus": False, "pdf.fonttype": 42}
    ):
        figure, axes = plt.subplots(2, 3, figsize=(23, 12))
        for method_index, name in enumerate(VARIANTS):
            levels = data["methods"][name]
            total = data["poi_rows"]
            axis = axes[method_index, 0]
            y = np.arange(3)[::-1]
            for offset, key, color, label in (
                (0.18, "prefix_changed_count", "#487bac", "前缀码号改变"),
                (-0.18, "membership_changed_count", "#c28c3d", "同桶成员集合改变"),
            ):
                rates = [x[key] * 100 / total for x in levels]
                axis.barh(y + offset, rates, height=0.30, color=color, label=label)
                for position, value in zip(y + offset, rates):
                    axis.text(
                        value + 1, position, f"{value:.2f}%", va="center", fontsize=10
                    )
            axis.set_yticks(y, LEVELS)
            axis.set_xlim(0, 115)
            axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
            axis.set_title(
                f"{VARIANT_TITLES[method_index]}\n编码与同伴集合变化（全库 POI）",
                fontsize=15,
                loc="left",
                pad=15,
            )
            axis.legend(loc="lower right", frameon=False, fontsize=10)
            for spine in axis.spines.values():
                spine.set_visible(False)

            axis = axes[method_index, 1]
            labels = []
            for i, level in enumerate(levels):
                for j, key in enumerate(("category", "region")):
                    _outcome_bar(axis, 5 - i * 2 - j, level[key]["counts"], total)
                    labels.append(
                        f"{LEVELS[i]} {'类别' if key == 'category' else '区域'}"
                    )
            axis.set_yticks(np.arange(6)[::-1], labels, fontsize=9)
            axis.set_xlim(0, 100)
            axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
            axis.set_title(
                "同类/同网格的其他 POI 比例变化\n提高、持平、降低及不可比（全库分母）",
                loc="left",
                fontsize=14,
                pad=15,
            )
            for spine in axis.spines.values():
                spine.set_visible(False)

            axis = axes[method_index, 2]
            axis.axis("off")
            transitions = levels[2]["singleton_transitions"]
            rows = [
                [
                    transitions["singleton_to_singleton"],
                    transitions["singleton_to_collision"],
                ],
                [
                    transitions["collision_to_singleton"],
                    transitions["collision_to_collision"],
                ],
            ]
            cells = [
                [f"{value:,} 个 POI\n占全库 {value / total:.2%}" for value in row]
                for row in rows
            ]
            tab = axis.table(
                cellText=cells,
                rowLabels=["A0 单例", "A0 碰撞"],
                colLabels=["A4 单例", "A4 碰撞"],
                cellLoc="center",
                bbox=[0.12, 0.24, 0.88, 0.60],
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(11)
            for (r, col), cell in tab.get_celld().items():
                cell.set_edgecolor("#dce2e8")
                if r == 0 or col == -1:
                    cell.set_facecolor("#eef3f8")
            tab[(1, 1)].set_facecolor("#f5dedb")
            tab[(2, 0)].set_facecolor("#d6eee5")
            axis.set_title(
                "完整三位 SID：碰撞状态迁移\n按 POI 计数，不是桶数",
                fontsize=14,
                loc="left",
                pad=15,
            )
            axis.text(
                0.12,
                0.10,
                "绿：原碰撞 POI 变单例；红：原单例 POI 新增碰撞\n两者必须同时看，不把单例当作语义纯度改善。",
                fontsize=10,
                color="#556575",
            )
        figure.suptitle("A0 → A4 到底改变了什么？", fontsize=23, y=0.99)
        handles = [
            Patch(facecolor=color, label=OUTCOME_LABELS[key])
            for key, color in OUTCOME_COLORS.items()
        ]
        figure.legend(
            handles=handles,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.53, 0.951),
        )
        figure.text(
            0.5,
            0.02,
            "同伴比例剔除自身：S1 看粗类别，后两层看细类别；区域为 Geohash5。任一侧是单例则不可比。本图仅统计 SID，不含 GID/dedup，不是检索效果。",
            ha="center",
            fontsize=10,
            color="#66717e",
        )
        figure.subplots_adjust(
            left=0.045, right=0.98, top=0.865, bottom=0.08, wspace=0.32, hspace=0.40
        )
        _save(figure, output, "sid_migration_overview", settings["projection"]["dpi"])
        plt.close(figure)

        for method_index, name in enumerate(VARIANTS):
            figure, axes = plt.subplots(3, 3, figsize=(21, 14))
            for row, outcome in enumerate(("improved", "tied", "degraded")):
                for column, depth in enumerate((1, 2, 3)):
                    case = next(
                        (
                            x
                            for x in examples[name]
                            if x["depth"] == depth
                            and x["category"]["outcome"] == outcome
                        ),
                        None,
                    )
                    _example_card(axes[row, column], case, depth, outcome)
            figure.suptitle(
                f"{VARIANT_TITLES[method_index]}：沿同一锚点看移出、保留、移入",
                fontsize=22,
                y=0.99,
            )
            figure.text(
                0.5,
                0.954,
                "蓝＝前后共有成员（含锚点） / 红＝离开锚点新桶的成员 / 绿＝新进入锚点桶的成员；每条按自身桶大小归一化，数字为实际 POI 数。",
                ha="center",
                fontsize=11,
                color="#556575",
            )
            figure.text(
                0.5,
                0.929,
                "按类别同伴一致率提高 / 持平 / 降低分组；同时列区域变化，不做综合胜负判断。案例必须换前缀且换成员、前后均非单例；S2/S3 桶各不超过 20。",
                ha="center",
                fontsize=10,
                color="#66717e",
            )
            figure.text(
                0.5,
                0.02,
                f"三个组内均按 seed={data['parameters']['seed']} 确定性抽样，不取最大增益。成员名称与去向见 migration_examples.html；完整成员行号和 POI ID 见 memberships.npz。",
                ha="center",
                fontsize=10,
                color="#66717e",
            )
            figure.subplots_adjust(
                left=0.025, right=0.98, top=0.87, bottom=0.065, wspace=0.14, hspace=0.22
            )
            stem = (
                "sid_migration_gid_examples"
                if method_index == 0
                else "sid_migration_nogid_examples"
            )
            _save(figure, output, stem, settings["projection"]["dpi"])
            plt.close(figure)


def migration_examples_html(examples: dict[str, list[dict[str, Any]]]) -> str:
    """Present bounded, escaped POI previews and precise old/new prefixes for membership changes."""
    lines = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>A0 到 A4 的桶成员迁移</title>",
        "<style>body{font-family:sans-serif;margin:24px;color:#25384b}table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "td,th{border:1px solid #dce1e7;padding:7px;text-align:left}th{background:#eef3f8}h2{margin-top:42px}</style>",
        "<h1>A0 → A4：桶成员保留、移出、移入</h1><p>案例按类别同伴一致率提高、持平、降低分组抽样，不按最大增益选择。"
        "同伴比例排除锚点自身；区域单独展示。预览每组最多 5 个 POI，<a href='memberships.npz'>完整成员行号/ID</a>另存，未删减统计。"
        "“移出/移入”均相对于该锚点前后的两个桶；前后桶编码可能不同。这不是最终检索优劣判断。</p>",
    ]
    for index, name in enumerate(VARIANTS):
        lines.append(f"<h2>{VARIANT_TITLES[index]}</h2>")
        for case in examples[name]:
            cat, region = case["category"], case["region"]
            lines.append(
                f"<h3>{LEVELS[case['depth'] - 1]} 类别一致率{OUTCOME_LABELS[cat['outcome']]}：{html.escape(case['anchor']['displayname'])}</h3>"
            )
            lines.append(
                f"<p>POI {html.escape(str(case['anchor']['poi_id']))} · {html.escape(case['anchor']['category'])}<br>"
                f"{case['before_prefix']}（{case['before_size']} POI）→ {case['after_prefix']}（{case['after_size']} POI）<br>"
                f"同类其他 POI 比例 {cat['before']:.2%} → {cat['after']:.2%}；同网格 {region['before']:.2%} → {region['after']:.2%}</p>"
            )
            for key, title in (
                ("retained", "保留"),
                ("removed", "移出"),
                ("added", "移入"),
            ):
                group = case["groups"][key]
                lines.append(
                    f"<h4>{title} {group['count']} 个；展示 {len(group['preview'])} 个</h4>"
                )
                lines.append(
                    "<table><tr><th>POI ID</th><th>名称</th><th>完整类别</th><th>网格</th><th>A0 前缀</th><th>A4 前缀</th></tr>"
                )
                for member in group["preview"]:
                    values = [
                        member["poi_id"],
                        member["displayname"],
                        member["category"],
                        member["geohash5"],
                        member["old_prefix"],
                        member["new_prefix"],
                    ]
                    lines.append(
                        "<tr>"
                        + "".join(
                            f"<td>{html.escape(str(value))}</td>" for value in values
                        )
                        + "</tr>"
                    )
                lines.append("</table>")
    return "\n".join(lines) + "</html>\n"
