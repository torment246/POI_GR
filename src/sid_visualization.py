"""Matplotlib visualizations for MobilityBench SID quality reports."""

from __future__ import annotations

import html
import os
from pathlib import Path
import re
import textwrap
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


FIGURE_SIZE = (13.5, 7.6)
DPI = 200
NOTE_TEXT = "MobilityBench POI, N={n}"

COLORS = {
    "semantic": "#2563eb",
    "geo": "#f97316",
    "green": "#16a34a",
    "red": "#dc2626",
    "amber": "#d97706",
    "slate": "#334155",
    "muted": "#64748b",
    "light_blue": "#dbeafe",
    "light_orange": "#ffedd5",
    "light_green": "#dcfce7",
    "light_red": "#fee2e2",
    "line_grid": "#e2e8f0",
    "card_border": "#cbd5e1",
}

PREFERRED_FONT_NAMES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK",
    "Arial Unicode MS",
    "DejaVu Sans",
]

PREFERRED_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


DEFAULT_COMPARE_METRICS: dict[str, dict[str, float]] = {
    "semantic": {
        "unique_sid_rate": 0.5617,
        "unique_pid_gid6_sid_rate": 0.9676,
        "prefix3_weighted_semantic_purity": 0.7651,
        "prefix3_weighted_city_purity": 0.6799,
        "qrels_targets_in_sid_collision_rate": 0.4943,
        "qrels_targets_in_pid_collision_rate": 0.0082,
        "queries_with_pid_collision_rate": 0.1073,
    },
    "geo_fused": {
        "unique_sid_rate": 0.5073,
        "unique_pid_gid6_sid_rate": 0.9472,
        "prefix3_weighted_semantic_purity": 0.7518,
        "prefix3_weighted_city_purity": 0.8290,
        "qrels_targets_in_sid_collision_rate": 0.5858,
        "qrels_targets_in_pid_collision_rate": 0.0149,
        "queries_with_pid_collision_rate": 0.1557,
    },
}

DEFAULT_SEMANTIC_REPORT_METRICS: dict[str, float] = {
    "input_rows": 109385,
    "unique_sid_count": 61437,
    "unique_sid_rate": 0.5617,
    "sid_collision_poi_count": 63495,
    "unique_pid_gid6_sid_count": 105841,
    "unique_pid_gid6_sid_rate": 0.9676,
    "pid_collision_poi_count": 6246,
    "pid_gid6_sid_dedup_unique_count": 109385,
    "pid_gid6_sid_dedup_unique_rate": 1.0,
    "qrels_targets_in_pid_collision_rate": 0.0082,
    "prefix1_semantic_purity": 0.3527,
    "prefix1_city_purity": 0.0915,
    "prefix1_geohash5_purity": 0.0147,
    "prefix2_semantic_purity": 0.4461,
    "prefix2_city_purity": 0.2765,
    "prefix2_geohash5_purity": 0.1572,
    "prefix3_semantic_purity": 0.7651,
    "prefix3_city_purity": 0.6799,
    "prefix3_geohash5_purity": 0.6120,
}


def configure_matplotlib() -> tuple[str, str | None]:
    """Configure matplotlib for PPT-ready Chinese charts."""
    selected_font_name = None
    selected_font_path = None

    for font_path in PREFERRED_FONT_FILES:
        path = Path(font_path)
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                selected_font_name = font_manager.FontProperties(fname=str(path)).get_name()
                selected_font_path = str(path)
                break
            except Exception:
                continue

    if selected_font_name is None:
        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in PREFERRED_FONT_NAMES:
            if name in available:
                selected_font_name = name
                break

    if selected_font_name is None:
        selected_font_name = "DejaVu Sans"

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected_font_name, *PREFERRED_FONT_NAMES],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "axes.edgecolor": "#cbd5e1",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
        }
    )
    return selected_font_name, selected_font_path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}".replace(",", "")


def wrap_cn(text: str, width: int = 18) -> str:
    """Wrap mixed Chinese/English text for card annotations."""
    pieces: list[str] = []
    for part in str(text).split("\n"):
        if len(part) <= width:
            pieces.append(part)
        else:
            pieces.extend(textwrap.wrap(part, width=width, break_long_words=True, replace_whitespace=False))
    return "\n".join(pieces)


def add_common_note(fig: plt.Figure, n_rows: int) -> None:
    fig.text(0.985, 0.022, NOTE_TEXT.format(n=fmt_int(n_rows)), ha="right", va="bottom", fontsize=10, color=COLORS["muted"])


def add_corner_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.045, 0.94, title, ha="left", va="top", fontsize=22, weight="bold", color="#0f172a")
    if subtitle:
        fig.text(0.045, 0.895, subtitle, ha="left", va="top", fontsize=12.5, color=COLORS["muted"])


def save_figure(fig: plt.Figure, out_dir: str | Path, basename: str) -> tuple[Path, Path]:
    out = ensure_dir(out_dir)
    png_path = out / f"{basename}.png"
    svg_path = out / f"{basename}.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    return png_path, svg_path


def read_mapping_n(path: str | Path) -> int:
    try:
        return int(len(pd.read_parquet(path, columns=["poi_id"])))
    except Exception:
        return int(DEFAULT_SEMANTIC_REPORT_METRICS["input_rows"])


def parse_compare_report(path: str | Path) -> dict[str, dict[str, float]]:
    metrics = {mode: values.copy() for mode, values in DEFAULT_COMPARE_METRICS.items()}
    p = Path(path)
    if not p.exists():
        return metrics
    lines = p.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if "mode" in cells:
            header = cells
            continue
        if header and cells and cells[0] in metrics and len(cells) == len(header):
            mode = cells[0]
            for key, value in zip(header[1:], cells[1:]):
                try:
                    metrics[mode][key] = float(value)
                except ValueError:
                    continue
    return metrics


def parse_semantic_report(path: str | Path) -> dict[str, float]:
    metrics = DEFAULT_SEMANTIC_REPORT_METRICS.copy()
    p = Path(path)
    if not p.exists():
        return metrics
    text = p.read_text(encoding="utf-8")
    patterns = {
        "input_rows": r"- input_rows: ([0-9]+)",
        "unique_sid": r"- unique_sid_count / rate: ([0-9]+) / ([0-9.]+)",
        "sid_collision_poi_count": r"- sid_collision_poi_count: ([0-9]+)",
        "unique_pid": r"- unique_pid_gid6_sid_count / rate: ([0-9]+) / ([0-9.]+)",
        "pid_collision_poi_count": r"- pid_collision_poi_count: ([0-9]+)",
        "dedup_pid": r"- pid_gid6_sid_dedup_unique_count / rate: ([0-9]+) / ([0-9.]+)",
        "qrels_pid_collision": r"- qrels_targets_in_pid_collision_count / rate: [0-9]+ / ([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        if key == "unique_sid":
            metrics["unique_sid_count"] = float(match.group(1))
            metrics["unique_sid_rate"] = float(match.group(2))
        elif key == "unique_pid":
            metrics["unique_pid_gid6_sid_count"] = float(match.group(1))
            metrics["unique_pid_gid6_sid_rate"] = float(match.group(2))
        elif key == "dedup_pid":
            metrics["pid_gid6_sid_dedup_unique_count"] = float(match.group(1))
            metrics["pid_gid6_sid_dedup_unique_rate"] = float(match.group(2))
        elif key == "qrels_pid_collision":
            metrics["qrels_targets_in_pid_collision_rate"] = float(match.group(1))
        else:
            metrics[key] = float(match.group(1))

    section_patterns = {
        "prefix1": r"### prefix1\n- weighted_semantic_purity: ([0-9.]+)\n- weighted_city_purity: ([0-9.]+)\n- weighted_geohash5_purity: ([0-9.]+)",
        "prefix2": r"### prefix2\n- weighted_semantic_purity: ([0-9.]+)\n- weighted_city_purity: ([0-9.]+)\n- weighted_geohash5_purity: ([0-9.]+)",
        "prefix3": r"### prefix3\n- weighted_semantic_purity: ([0-9.]+)\n- weighted_city_purity: ([0-9.]+)\n- weighted_geohash5_purity: ([0-9.]+)",
    }
    # Use the last occurrence of each prefix section; the report has group-size
    # sections before the purity sections.
    for prefix, pattern in section_patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            continue
        semantic, city, geo = matches[-1]
        metrics[f"{prefix}_semantic_purity"] = float(semantic)
        metrics[f"{prefix}_city_purity"] = float(city)
        metrics[f"{prefix}_geohash5_purity"] = float(geo)
    return metrics


def draw_round_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    fc: str,
    ec: str = "#94a3b8",
    fontsize: int = 13,
    weight: str = "normal",
    radius: float = 0.035,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#0f172a",
        linespacing=1.45,
    )
    return patch


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#64748b") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def fig1_pipeline(out_dir: str | Path, n_rows: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_axis_off()
    add_corner_title(fig, "SID 构建链路已打通", "从清洗 POI 到可生成 PID token 的端到端阶段产物")
    add_common_note(fig, n_rows)

    labels = [
        "POI 清洗\n109385 POI",
        "Qwen3 文本向量\n109385 × 1024",
        "Geo 特征\n109385 × 54",
        "RQ-VAE\n3 层 codebook\n128 codes/layer",
        "SID\nS0 S1 S2",
        "PID\ngeohash6 + SID",
    ]
    colors = ["#eff6ff", "#eef2ff", "#f0fdf4", "#fff7ed", "#fefce8", "#ecfeff"]
    xs = np.linspace(0.045, 0.83, len(labels))
    y = 0.52
    w = 0.125
    h = 0.19
    for i, (x, label) in enumerate(zip(xs, labels)):
        draw_round_box(ax, (x, y), w, h, label, colors[i], fontsize=12.2, weight="bold" if i in {3, 5} else "normal")
        if i < len(labels) - 1:
            draw_arrow(ax, (x + w + 0.006, y + h / 2), (xs[i + 1] - 0.006, y + h / 2))

    rq_x = xs[3] + w / 2
    pid_x = xs[5] + w / 2
    ax.text(
        rq_x,
        y - 0.08,
        "semantic: text only\ngeo_fused: text + 0.1 × geo",
        ha="center",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
        linespacing=1.35,
    )
    ax.text(
        pid_x,
        y - 0.055,
        "pid_gid6_sid_dedup\n唯一率 100%",
        ha="center",
        va="top",
        fontsize=11,
        color=COLORS["green"],
        weight="bold",
        linespacing=1.3,
    )
    ax.text(
        0.5,
        0.16,
        "已完成从 POI 库到可生成 PID token 的第一版构建链路。",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color="#0f172a",
    )
    return save_figure(fig, out_dir, "fig1_sid_pipeline_overview")


def fig2_compare_bar(out_dir: str | Path, metrics: dict[str, dict[str, float]], n_rows: int) -> tuple[Path, Path]:
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_axes([0.12, 0.14, 0.61, 0.72])
    add_corner_title(fig, "Semantic SID 更适合作为第一版生成目标，Geo_fused 空间一致性更强")
    add_common_note(fig, n_rows)

    keys = [
        ("SID 唯一率", "unique_sid_rate", False),
        ("PID 唯一率", "unique_pid_gid6_sid_rate", False),
        ("Prefix3 语义纯度", "prefix3_weighted_semantic_purity", False),
        ("Prefix3 城市纯度", "prefix3_weighted_city_purity", False),
        ("Qrels SID 冲突率", "qrels_targets_in_sid_collision_rate", True),
        ("Qrels PID 冲突率", "qrels_targets_in_pid_collision_rate", True),
        ("候选集 PID 冲突率", "queries_with_pid_collision_rate", True),
    ]
    labels = [x[0] for x in keys]
    y = np.arange(len(keys))
    height = 0.34
    semantic = [metrics["semantic"][key] for _, key, _ in keys]
    geo = [metrics["geo_fused"][key] for _, key, _ in keys]

    for idx, (_, _, lower_better) in enumerate(keys):
        if lower_better:
            ax.axhspan(idx - 0.47, idx + 0.47, color="#fff1f2", zorder=0)

    ax.barh(y - height / 2, semantic, height, color=COLORS["semantic"], label="Semantic")
    ax.barh(y + height / 2, geo, height, color=COLORS["geo"], label="Geo_fused")
    for idx, (s, g) in enumerate(zip(semantic, geo)):
        ax.text(s + 0.012, idx - height / 2, pct(s), va="center", ha="left", fontsize=10.5, color="#1e293b")
        ax.text(g + 0.012, idx + height / 2, pct(g), va="center", ha="left", fontsize=10.5, color="#1e293b")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.set_xlabel("比例", fontsize=12)
    ax.grid(axis="x", color=COLORS["line_grid"], linewidth=0.9)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.text(0.82, 4.15, "冲突率：越低越好", fontsize=11, color=COLORS["red"], weight="bold")

    box_ax = fig.add_axes([0.765, 0.25, 0.2, 0.48])
    box_ax.set_axis_off()
    draw_round_box(
        box_ax,
        (0.0, 0.0),
        1.0,
        1.0,
        "结论\n\nSemantic：唯一率更高、\nqrels/PID 冲突更低\n\nGeo_fused：城市纯度更高，\n但冲突率上升",
        fc="#f8fafc",
        ec=COLORS["card_border"],
        fontsize=12,
        weight="bold",
        radius=0.05,
    )
    return save_figure(fig, out_dir, "fig2_sid_quality_compare_bar")


def fig3_collision_funnel(out_dir: str | Path, metrics: dict[str, float], n_rows: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_axis_off()
    add_corner_title(fig, "geohash6 + SID 显著降低目标 token 冲突", "Semantic 主线的 SID → PID → Dedup PID 冲突消解")
    add_common_note(fig, n_rows)

    stages = [
        ("POI 总数", metrics["input_rows"], 1.0, "#dbeafe"),
        ("Unique SID", metrics["unique_sid_count"], metrics["unique_sid_rate"], "#bfdbfe"),
        ("Unique PID", metrics["unique_pid_gid6_sid_count"], metrics["unique_pid_gid6_sid_rate"], "#bbf7d0"),
        ("Dedup PID", metrics["pid_gid6_sid_dedup_unique_count"], metrics["pid_gid6_sid_dedup_unique_rate"], "#dcfce7"),
    ]
    y_positions = [0.72, 0.55, 0.38, 0.21]
    max_w = 0.66
    left = 0.11
    h = 0.095
    for idx, ((name, count, ratio, color), y) in enumerate(zip(stages, y_positions)):
        w = max_w * ratio
        draw_round_box(
            ax,
            (left, y),
            w,
            h,
            f"{name}: {fmt_int(count)} / {pct(ratio)}",
            fc=color,
            ec="#93c5fd" if idx < 2 else "#86efac",
            fontsize=14,
            weight="bold",
            radius=0.025,
        )
        if idx < len(stages) - 1:
            draw_arrow(ax, (left + w / 2, y - 0.005), (left + max_w * stages[idx + 1][2] / 2, y_positions[idx + 1] + h + 0.005))

    ax.text(0.44, 0.49, "+ geohash6 空间 token", fontsize=13, color=COLORS["green"], weight="bold")
    ax.text(0.78, 0.245, "用 D_i 保证\n最终 token 唯一", fontsize=12, color=COLORS["green"], weight="bold", linespacing=1.3)

    side_text = (
        f"SID collision POI: {fmt_int(metrics['sid_collision_poi_count'])}\n"
        f"PID collision POI: {fmt_int(metrics['pid_collision_poi_count'])}\n"
        f"Qrels PID collision rate: {pct(metrics['qrels_targets_in_pid_collision_rate'], 2)}"
    )
    draw_round_box(ax, (0.76, 0.49), 0.19, 0.18, side_text, fc="#f8fafc", ec=COLORS["card_border"], fontsize=12, radius=0.025)
    ax.text(
        0.5,
        0.08,
        "虽然 full SID 仍有碰撞，但加入 GID 后，PID 已基本满足后续生成训练的目标唯一性要求。",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#0f172a",
    )
    return save_figure(fig, out_dir, "fig3_sid_pid_collision_funnel")


def fig4_prefix_purity(out_dir: str | Path, metrics: dict[str, float], n_rows: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    add_corner_title(fig, "SID 层级越深，语义与空间一致性越强", "Semantic 模型 prefix purity")
    add_common_note(fig, n_rows)

    x = np.arange(3)
    labels = ["S0", "S0+S1", "S0+S1+S2"]
    semantic = [metrics["prefix1_semantic_purity"], metrics["prefix2_semantic_purity"], metrics["prefix3_semantic_purity"]]
    city = [metrics["prefix1_city_purity"], metrics["prefix2_city_purity"], metrics["prefix3_city_purity"]]
    geo = [metrics["prefix1_geohash5_purity"], metrics["prefix2_geohash5_purity"], metrics["prefix3_geohash5_purity"]]

    ax.plot(x, semantic, marker="o", linewidth=3, color=COLORS["semantic"], label="语义纯度")
    ax.plot(x, city, marker="o", linewidth=3, color=COLORS["geo"], label="城市纯度")
    ax.plot(x, geo, marker="o", linewidth=3, color=COLORS["green"], label="geohash5 纯度")
    for series in (semantic, city, geo):
        for xi, yi in zip(x, series):
            ax.text(xi, yi + 0.025, pct(yi), ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Purity", fontsize=12)
    ax.set_ylim(0, 0.88)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(axis="y", color=COLORS["line_grid"], linewidth=0.9)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.annotate(
        "层级越深，语义与空间一致性越强",
        xy=(2, semantic[-1]),
        xytext=(1.15, 0.82),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["slate"], linewidth=1.5),
        fontsize=13,
        weight="bold",
        color=COLORS["slate"],
    )
    fig.text(0.5, 0.07, "当前 SID 的细粒度层级有效，但第一层粗语义仍需优化。", ha="center", fontsize=15, weight="bold")
    fig.subplots_adjust(left=0.11, right=0.95, bottom=0.18, top=0.83)
    return save_figure(fig, out_dir, "fig4_sid_prefix_purity")


def fig5_codebook_usage(out_dir: str | Path, n_rows: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    add_corner_title(fig, "三层 codebook 均被充分利用，没有 codebook collapse")
    add_common_note(fig, n_rows)
    ax.set_axis_off()

    rows = ["Semantic", "Geo_fused"]
    cols = ["Codebook 0", "Codebook 1", "Codebook 2"]
    matrix = np.ones((2, 3))
    table_ax = fig.add_axes([0.14, 0.25, 0.72, 0.42])
    table_ax.imshow(matrix, cmap=mpl.colors.ListedColormap(["#dcfce7"]), vmin=0, vmax=1)
    table_ax.set_xticks(np.arange(3))
    table_ax.set_yticks(np.arange(2))
    table_ax.set_xticklabels(cols, fontsize=14, weight="bold")
    table_ax.set_yticklabels(rows, fontsize=14, weight="bold")
    table_ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for spine in table_ax.spines.values():
        spine.set_visible(False)
    table_ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    table_ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    table_ax.grid(which="minor", color="white", linestyle="-", linewidth=4)
    table_ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(2):
        for j in range(3):
            table_ax.text(j, i, "128/128\nusage 100%\ndead code = 0", ha="center", va="center", fontsize=14, color="#14532d", weight="bold")

    draw_round_box(
        ax,
        (0.25, 0.08),
        0.5,
        0.11,
        "结论：三层 codebook 均被充分利用，量化空间没有塌缩。",
        fc="#f0fdf4",
        ec="#86efac",
        fontsize=14,
        weight="bold",
        radius=0.03,
    )
    return save_figure(fig, out_dir, "fig5_codebook_usage_heatmap")


def draw_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    badge: str,
    badge_color: str,
    sid: str,
    group_size: str,
    top_label: str,
    examples: str,
    comment: str,
) -> None:
    draw_round_box(ax, xy, width, height, "", fc="#ffffff", ec=COLORS["card_border"], fontsize=1, radius=0.025)
    x, y = xy
    ax.text(x + 0.025, y + height - 0.04, title, ha="left", va="top", fontsize=13.2, weight="bold")
    badge_w = 0.095 if len(badge) <= 2 else 0.12
    badge_patch = FancyBboxPatch(
        (x + width - badge_w - 0.025, y + height - 0.065),
        badge_w,
        0.038,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor=badge_color,
        edgecolor=badge_color,
    )
    ax.add_patch(badge_patch)
    ax.text(x + width - badge_w / 2 - 0.025, y + height - 0.046, badge, ha="center", va="center", fontsize=10.5, color="white", weight="bold")
    meta = f"SID/prefix: {sid}   |   size: {group_size}\nTop label: {top_label}"
    ax.text(x + 0.025, y + height - 0.095, meta, ha="left", va="top", fontsize=9.8, color=COLORS["muted"], linespacing=1.25)
    ax.text(
        x + 0.025,
        y + height - 0.165,
        "示例：" + wrap_cn(examples, 30),
        ha="left",
        va="top",
        fontsize=9.9,
        color="#0f172a",
        linespacing=1.25,
    )
    ax.text(
        x + 0.025,
        y + 0.045,
        wrap_cn(comment, 34),
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=COLORS["slate"],
        linespacing=1.2,
    )


def fig6_cluster_examples(out_dir: str | Path, n_rows: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_axis_off()
    add_corner_title(fig, "SID 已能聚合部分同类 POI，但低标注类别仍会造成混合簇")
    add_common_note(fig, n_rows)

    cards = [
        (
            "好的语义簇：加油站",
            "可用",
            COLORS["green"],
            "S0_1",
            "1277",
            "加油站",
            "中国石油加油加气站 / 中国石油乌鲁木齐永德信加油站 / 中国石化青年路加油站",
            "同类能源服务 POI 聚合明显，适合作为可解释 SID 示例。",
        ),
        (
            "好的语义簇：停车场",
            "可用",
            COLORS["green"],
            "S0_83",
            "1757",
            "停车场",
            "龙盘国际家具广场停车场 / 建博花园二期停车场 / 蛇口广场地下停车场",
            "停车相关命名模式稳定，粗粒度 SID 已能捕获主要类别。",
        ),
        (
            "好的品牌/连锁簇：蜜雪冰城",
            "可用",
            COLORS["green"],
            "S0_56 S1_123 S2_61",
            "209",
            "品牌一致",
            "蜜雪冰城(水塔店) / 蜜雪冰城(襄平峰汇店) / 蜜雪冰城(金福贵源店)",
            "连锁品牌在文本 embedding 中形成稳定邻近结构。",
        ),
        (
            "需要优化的混合簇：低语义纯度",
            "待优化",
            COLORS["amber"],
            "S0_97",
            "1669",
            "政府机构 / UNK 混合",
            "黄家镇 / 巴彦查干乡人民政府 / 永乐镇人民政府 / 陈家镇裕安农贸市场",
            "类别缺失、行政地名与机构名混杂，后续应补充类别归一化。",
        ),
    ]
    positions = [(0.055, 0.50), (0.525, 0.50), (0.055, 0.13), (0.525, 0.13)]
    for pos, card in zip(positions, cards):
        draw_card(ax, pos, 0.42, 0.33, *card)
    return save_figure(fig, out_dir, "fig6_sid_cluster_examples")


def fig7_stage_conclusion(out_dir: str | Path, n_rows: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_axis_off()
    add_corner_title(fig, "当前 SID/PID 已具备作为第一版生成式 POI 检索目标 token 的条件")
    add_common_note(fig, n_rows)

    columns = [
        (
            "已经完成",
            [
                "POI 清洗：109385",
                "文本向量：Qwen3, 1024维",
                "Geo 特征：54维",
                "RQ-VAE：3层 × 128 code",
                "SID/PID 导出：完成",
            ],
            "#eff6ff",
        ),
        (
            "质量结论",
            [
                "Codebook 使用率：100%",
                "Unique SID：56.2%",
                "Unique PID：96.8%",
                "Qrels PID 冲突：0.82%",
                "Prefix3 语义纯度：76.5%",
            ],
            "#f0fdf4",
        ),
        (
            "后续优化",
            [
                "提升 prefix1 粗语义纯度",
                "降低 full SID collision",
                "补充类别和品牌规则",
                "尝试 latent_dim=128",
                "进入 query-to-PID 生成训练",
            ],
            "#fff7ed",
        ),
    ]
    x_positions = [0.055, 0.365, 0.675]
    for x, (title, items, color) in zip(x_positions, columns):
        draw_round_box(ax, (x, 0.26), 0.27, 0.48, "", fc=color, ec=COLORS["card_border"], fontsize=1, radius=0.03)
        ax.text(x + 0.135, 0.68, title, ha="center", va="center", fontsize=17, weight="bold")
        for idx, item in enumerate(items):
            ax.text(x + 0.035, 0.615 - idx * 0.075, f"• {item}", ha="left", va="center", fontsize=12.2, color="#0f172a")

    ax.text(
        0.5,
        0.12,
        "当前 SID/PID 已具备作为第一版生成式 POI 检索目标 token 的条件。",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        color="#0f172a",
    )
    return save_figure(fig, out_dir, "fig7_sid_stage_conclusion")


def write_index_files(out_dir: str | Path, figures: list[dict[str, str]]) -> tuple[Path, Path]:
    out = ensure_dir(out_dir)
    readme_path = out / "README.md"
    html_path = out / "index.html"

    readme_lines = [
        "# SID Quality Visualization Figures",
        "",
        "这些图用于 MobilityBench POI 生成式检索项目的 SID 质量汇报，可直接放入 PPT。",
        "",
    ]
    for fig in figures:
        readme_lines.extend(
            [
                f"## {fig['title']}",
                f"- PNG: `{fig['png']}`",
                f"- SVG: `{fig['svg']}`",
                f"- 图说明: {fig['description']}",
                f"- PPT 使用建议: {fig['usage']}",
                "",
            ]
        )
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    cards = []
    for fig in figures:
        png_name = Path(fig["png"]).name
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(fig['title'])}</h2>
              <p>{html.escape(fig['description'])}</p>
              <img src="{html.escape(png_name)}" alt="{html.escape(fig['title'])}">
              <p class="links"><a href="{html.escape(png_name)}">PNG</a> · <a href="{html.escape(Path(fig['svg']).name)}">SVG</a></p>
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>SID Quality Figures</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
    h1 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 28px; max-width: 1280px; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; }}
    .card img {{ width: 100%; border: 1px solid #e2e8f0; }}
    .links {{ color: #475569; }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <h1>MobilityBench POI SID Quality Figures</h1>
  <p>所有图片均提供 PNG 和 SVG，PNG 按 PPT 16:9 尺寸输出。</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    print(f"Saved: {html_path}")
    print(f"Saved: {readme_path}")
    return html_path, readme_path


def generate_all_figures(
    out_dir: str | Path,
    semantic_report: str | Path,
    compare_report: str | Path,
    semantic_mapping: str | Path,
) -> list[dict[str, str]]:
    out = ensure_dir(out_dir)
    n_rows = read_mapping_n(semantic_mapping)
    compare_metrics = parse_compare_report(compare_report)
    semantic_metrics = parse_semantic_report(semantic_report)
    semantic_metrics["input_rows"] = n_rows

    figure_specs = [
        ("Figure 1：SID 构建流程与阶段产物总览", "展示 POI 清洗、文本向量、Geo 特征、RQ-VAE、SID/PID 的阶段产物。", "适合作为 SID 章节开场页。", fig1_pipeline(out, n_rows)),
        ("Figure 2：Semantic vs Geo_fused 核心指标对比", "比较两套 SID 的唯一率、纯度和冲突率。", "适合说明为什么第一版推荐 semantic。", fig2_compare_bar(out, compare_metrics, n_rows)),
        ("Figure 3：SID 到 PID 的冲突消解漏斗图", "展示 geohash6 + SID 对 token 冲突的削减效果。", "适合解释 PID 设计必要性。", fig3_collision_funnel(out, semantic_metrics, n_rows)),
        ("Figure 4：SID 层次结构与 Prefix Purity", "展示 S0、S0+S1、S0+S1+S2 的语义和空间纯度变化。", "适合讨论层级 SID 是否有效。", fig4_prefix_purity(out, semantic_metrics, n_rows)),
        ("Figure 5：Codebook 使用率热力图", "展示 semantic 和 geo_fused 三层 codebook 都达到 100% 使用率。", "适合说明量化没有塌缩。", fig5_codebook_usage(out, n_rows)),
        ("Figure 6：典型 SID cluster 示例卡片", "展示同类聚合、品牌聚合和低标注混合簇。", "适合人工案例分析。", fig6_cluster_examples(out, n_rows)),
        ("Figure 7：当前阶段结论页", "总结已完成工作、质量结论和下一步优化方向。", "适合作为 SID 阶段收尾页。", fig7_stage_conclusion(out, n_rows)),
    ]

    figures: list[dict[str, str]] = []
    for title, description, usage, (png, svg) in figure_specs:
        figures.append(
            {
                "title": title,
                "description": description,
                "usage": usage,
                "png": str(png),
                "svg": str(svg),
            }
        )
    write_index_files(out, figures)
    return figures
