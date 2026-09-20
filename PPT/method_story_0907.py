"""Method-first 0907 presentation for BeamRisk-SFT and QG-PRQK."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


INK = "111827"
BLACK = "000000"
NAVY = "1A3A6B"
MUTED = "64748B"
LINE = "CBD5E1"
LIGHT = "F8FAFC"
BLUE = "EAF2FF"
WHITE = "FFFFFF"
FORMULA_DIR = Path(__file__).resolve().parents[1] / "outputs/ppt_0907/formulas"
MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "outputs/ppt_0907/mplconfig"


def render_latex_formula(latex: str, font_size: float) -> Path:
    """Render one or more LaTeX math lines to a transparent high-DPI PNG."""
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
    digest = sha256(f"v2|{font_size}|{latex}".encode("utf-8")).hexdigest()[:20]
    output = FORMULA_DIR / f"{digest}.png"
    if output.is_file():
        return output

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    display = "\n".join(f"${line}$" for line in latex.splitlines())
    figure = Figure(figsize=(11.0, 2.8), dpi=320)
    FigureCanvasAgg(figure)
    figure.text(0.0, 0.0, display, fontsize=font_size, color="#111827",
                ha="left", va="bottom", linespacing=1.30)
    figure.savefig(output, dpi=320, transparent=True, bbox_inches="tight",
                   pad_inches=0.025)
    return output


class Canvas:
    """Native PowerPoint primitives for paper-style method explanation."""

    def __init__(self, deck, title: str, claim: str, source: str):
        self.deck = deck
        self.slide = deck.slide(title, claim, source)

    def text(self, text: str, x: float, y: float, w: float, h: float,
             *, size: float = 16, bold: bool = False, color: str = INK,
             center: bool = False):
        return self.deck.text(
            self.slide, text, y, h, x=x, w=w, size=size, color=color,
            bold=bold, align=PP_ALIGN.CENTER if center else PP_ALIGN.LEFT,
        )

    def node(self, text: str, x: float, y: float, w: float, h: float,
             *, emphasis: bool = False, size: float = 16, bold: bool = False,
             rounded: bool = False):
        shape_type = (
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if rounded
            else MSO_AUTO_SHAPE_TYPE.RECTANGLE
        )
        shape = self.slide.shapes.add_shape(
            shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.name = "method_node"
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(BLUE if emphasis else LIGHT)
        shape.line.color.rgb = RGBColor.from_string(LINE)
        shape.line.width = Pt(0.75)
        self.deck_style(shape, text, size=size, bold=bold)
        return shape

    def token(self, text: str, x: float, y: float, *, selected: bool = False,
              w: float = 0.70, h: float = 0.50, size: float = 15):
        shape = self.slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h),
        )
        shape.name = "method_token"
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(BLUE if selected else WHITE)
        shape.line.color.rgb = RGBColor.from_string(NAVY if selected else LINE)
        shape.line.width = Pt(1.4 if selected else 0.7)
        self.deck_style(shape, text, size=size, bold=selected)
        return shape

    def deck_style(self, shape, text: str, *, size: float, bold: bool) -> None:
        from build_0907 import style_frame

        style_frame(shape.text_frame, text, size=size, color=INK, bold=bold,
                    align=PP_ALIGN.CENTER)
        frame = shape.text_frame
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        frame.margin_left = frame.margin_right = Inches(0.08)
        frame.margin_top = frame.margin_bottom = Inches(0.04)
        for paragraph in frame.paragraphs:
            paragraph.space_after = Pt(0)

    def arrow(self, points: list[tuple[float, float]], *, color: str = MUTED,
              width: float = 1.15, dashed: bool = False, head: bool = True):
        for index, (start, end) in enumerate(zip(points, points[1:])):
            connector = self.slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT,
                Inches(start[0]), Inches(start[1]), Inches(end[0]), Inches(end[1]),
            )
            connector.name = "method_edge"
            connector.line.color.rgb = RGBColor.from_string(color)
            connector.line.width = Pt(width)
            line = connector.line._get_or_add_ln()
            if dashed:
                dash = OxmlElement("a:prstDash")
                dash.set("val", "dash")
                line.append(dash)
            if head and index == len(points) - 2:
                tail = OxmlElement("a:tailEnd")
                tail.set("type", "triangle")
                tail.set("w", "med")
                tail.set("len", "med")
                line.append(tail)
        return connector

    def line(self, x1: float, y1: float, x2: float, y2: float,
             *, color: str = LINE, width: float = 0.8, dashed: bool = False):
        return self.arrow([(x1, y1), (x2, y2)], color=color, width=width,
                          dashed=dashed, head=False)

    def dot(self, x: float, y: float, *, selected: bool = False, size: float = 0.17):
        shape = self.slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x), Inches(y), Inches(size), Inches(size)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(NAVY if selected else MUTED)
        shape.line.fill.background()
        return shape

    def label(self, title: str, text: str, x: float, y: float, w: float,
              *, h: float = 1.05, emphasis: bool = False):
        self.text(title, x, y, w, 0.43, size=18, bold=True, color=BLACK)
        self.line(x, y + 0.43, x + w, y + 0.43, color=NAVY if emphasis else LINE,
                  width=1.2 if emphasis else 0.8)
        self.text(text, x, y + 0.50, w, h - 0.50, size=15.5)

    def equation(self, latex: str, x: float, y: float, w: float, h: float,
                 *, emphasis: bool = False, size: float = 20):
        background = self.slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h),
        )
        background.name = "method_formula_bg"
        background.fill.solid()
        background.fill.fore_color.rgb = RGBColor.from_string(BLUE if emphasis else LIGHT)
        background.line.fill.background()

        formula_path = render_latex_formula(latex, size)
        with Image.open(formula_path) as image:
            image_w, image_h = image.size
        pad_x, pad_y = 0.13, 0.08
        box_w, box_h = max(0.1, w - 2 * pad_x), max(0.1, h - 2 * pad_y)
        image_aspect = image_w / image_h
        box_aspect = box_w / box_h
        if image_aspect >= box_aspect:
            pic_w = box_w
            pic_h = pic_w / image_aspect
        else:
            pic_h = box_h
            pic_w = pic_h * image_aspect
        pic_x = x + (w - pic_w) / 2
        pic_y = y + (h - pic_h) / 2
        picture = self.slide.shapes.add_picture(
            str(formula_path), Inches(pic_x), Inches(pic_y),
            width=Inches(pic_w), height=Inches(pic_h),
        )
        picture.name = "latex_formula"
        picture._element.nvPicPr.cNvPr.set("descr", f"LaTeX: {latex}")
        return picture

    def note(self, text: str, y: float = 6.12, h: float = 0.66,
             *, color: str = INK):
        return self.text(text, 0.65, y, 12.0, h, size=15.5, color=color)


def connect_row(c: Canvas, labels: list[str], y: float, *, x: float = 0.66,
                w: float = 12.0, h: float = 0.90, emphasis: int | None = None,
                size: float = 16):
    gap = 0.44
    node_w = (w - gap * (len(labels) - 1)) / len(labels)
    for index, label in enumerate(labels):
        left = x + index * (node_w + gap)
        c.node(label, left, y, node_w, h, emphasis=index == emphasis, size=size)
        if index:
            c.arrow([(left - gap + 0.05, y + h / 2), (left - 0.05, y + h / 2)])


def cover(deck) -> None:
    from build_0907 import style_frame

    deck.cover()
    slide = deck.prs.slides[0]
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.text == "阶段实验与方法进展":
            style_frame(shape.text_frame, "BeamRisk-SFT 与 QG-PRQK", 22, BLACK,
                        align=PP_ALIGN.CENTER)
        if shape.text == "2026.08.25—09.07  |  胡丹":
            style_frame(shape.text_frame, "方法设计与计算细节  |  2026.09.07  |  胡丹",
                        16, MUTED, align=PP_ALIGN.CENTER)


def overview(deck) -> None:
    c = Canvas(deck, "本次聚焦：两个训练—使用失配",
               "一个发生在 SID 构建端，一个发生在生成器训练端；两条创新线分别处理。",
               "TIGER / PRQ-KMeans / RIPOR；本项目 BeamRisk-SFT 与 QG-PRQK 方法文档")
    connect_row(c, ["POI 文本与类别\n冻结向量", "SID Tokenizer\nS1 → S2 → S3", "Query + 位置 + 历史\n生成模型 SFT", "Beam Search\nTop-K POI"], 2.05)
    c.arrow([(3.78, 3.02), (3.78, 3.44), (2.14, 3.44), (2.14, 3.78)], color=NAVY)
    c.node("失配 A：Tokenizer 优化 POI 几何\n但下游要让真实 Query 容易生成", 0.65, 3.80, 4.55, 1.02,
           emphasis=True, size=17)
    c.arrow([(9.43, 3.02), (9.43, 3.44), (10.62, 3.44), (10.62, 3.78)], color=NAVY)
    c.node("失配 B：SFT 优化 gold-prefix CE\n但评测依赖模型自身 Beam 的生存与排序", 8.08, 3.80, 4.55, 1.02,
           emphasis=True, size=17)
    c.label("QG-PRQK", "在离线 SID 构建中加入 Query 粒度、类别层次与局部地理。",
            0.65, 5.06, 5.65, h=1.15, emphasis=True)
    c.label("BeamRisk-SFT", "固定 SID，使用真实 Beam 轨迹对齐生存边界和最终排序。",
            6.98, 5.06, 5.65, h=1.15, emphasis=True)


def beam_observation(deck) -> None:
    c = Canvas(deck, "实验首先暴露：低 CE 不保证前缀活在 Beam 中",
               "Teacher-Forcing 看的是正确历史下的局部分类；自由生成会沿模型自己的前缀逐步剪枝。",
               "beamrisk_sft/README.md §4；TIGER 固定 10k 逐层诊断")
    c.text("训练：Teacher-Forcing", 0.65, 1.96, 5.45, 0.40, size=19, bold=True,
           color=BLACK)
    connect_row(c, ["gold S1", "gold S2", "gold S3", "gold C"], 2.66,
                x=0.65, w=5.45, h=0.72, size=15)
    c.text("每一步都喂入 gold 前缀；CE 能评估 token，却没有观察错误分支。",
           0.65, 3.64, 5.45, 0.72, size=16)
    c.line(6.50, 1.95, 6.50, 5.64, color=LINE)
    c.text("评测：模型自己的 Beam", 6.98, 1.96, 5.65, 0.40, size=19, bold=True,
           color=BLACK)
    columns = [7.05, 8.47, 9.89, 11.31]
    for x, label in zip(columns, ["S1", "S2", "S3", "C"]):
        c.text(label, x, 2.45, 0.8, 0.32, size=14, color=MUTED, center=True)
    # Gold path: it becomes the fourth candidate at S2 and is pruned from K=3.
    c.token("g1", columns[0], 3.05, selected=True)
    for y, token in [(2.82, "a2"), (3.45, "b2"), (4.08, "c2")]:
        c.token(token, columns[1], y)
        c.arrow([(columns[0]+0.70, 3.30), (columns[1]-0.04, y+0.25)], width=0.75)
    c.token("g2", columns[1], 4.71, selected=True)
    c.arrow([(columns[0]+0.70, 3.30), (columns[1]-0.04, 4.96)], color=NAVY,
            width=2.0, dashed=True)
    for y, token in [(2.82, "a3"), (3.45, "b3"), (4.08, "c3")]:
        c.token(token, columns[2], y)
        c.arrow([(columns[1]+0.70, y+0.25), (columns[2]-0.04, y+0.25)], width=0.75)
    c.arrow([(columns[2]+0.70, 2.82+0.25), (columns[3]-0.04, 2.82+0.25)], width=0.75)
    c.token("aC", columns[3], 2.82)
    c.text("K=3 的真实边界", 9.61, 5.01, 2.63, 0.35, size=15, color=NAVY,
           bold=True)
    c.arrow([(9.45, 5.18), (8.91, 5.18), (8.91, 4.33)], color=NAVY, width=1.4)
    c.note("关键差异：局部 token 概率提高，可能仍不足以让 gold 跨过第 K 名边界。合法 Trie 能消除无效路径，却不能替代合法候选内部的前缀排序。")


def beam_related(deck) -> None:
    c = Canvas(deck, "相关工作已经关注前缀，但没有直接给出地图 POI 的解法",
               "论文的共同启发是让训练看到模型前缀；我们的缺口在真实 Beam 边界、固定 SID 和标签噪声。",
               "Wiseman & Rush, EMNLP 2016；GLEN, EMNLP 2023；RIPOR, SIGIR 2024；Zhou et al., EMNLP 2023")
    rows = [
        (1.95, "Beam-Search Optimization", "训练时运行 Beam；gold 掉出时对第 K 名前缀做 margin 更新",
         "面向通用 seq2seq；需要搜索内更新，未处理 POI 目录关系与单点击多相关"),
        (3.16, "RIPOR / GLEN", "前缀排序、渐进训练与 prefix-aware dynamic negatives",
         "依赖检索负例或相关性教师；并非当前 Beam=10 的首个因果剪枝边界"),
        (4.37, "RL from relevance feedback", "对完整生成序列给 reward，再优化检索排序",
         "采样成本与方差更高；地图单点击标签会把站点—入口等合理候选当负例"),
    ]
    for y, name, mechanism, gap in rows:
        c.text(name, 0.65, y, 2.72, 0.70, size=17, bold=True, color=BLACK)
        c.arrow([(3.18, y+0.41), (3.73, y+0.41)], color=NAVY)
        c.text(mechanism, 3.92, y, 3.88, 0.77, size=15.5)
        c.arrow([(7.90, y+0.41), (8.42, y+0.41)], color=MUTED)
        c.text(gap, 8.62, y, 4.02, 0.77, size=15.5)
        c.line(0.65, y+0.91, 12.63, y+0.91)
    c.note("设计选择：不重建 TIGER SID、不增加线上模块，也不先引入 reward model；只把当前生成器实际遇到的 Beam 风险变成一个可审计的辅助训练流。")


def beam_framework(deck) -> None:
    c = Canvas(deck, "BeamRisk-SFT：主 CE 不动，旁路对齐真实 Beam 风险",
               "当前模型自己产生边界样本；风险候选按 epoch 冻结，避免每个训练 batch 内运行 Beam。",
               "beamrisk_sft/README.md §6–8；tracing.py / mining.py / loss.py / trainer.py")
    connect_row(c, ["同一扩词基座\nEpoch 1 标准 CE", "Train-only 请求\n真实 Beam=10", "状态分类\n首次剪枝 / 最终排序", "冻结风险 pair\n下一 epoch 注入"], 2.10,
                emphasis=2)
    c.arrow([(11.09, 3.02), (11.09, 3.46), (8.65, 3.46), (8.65, 3.82)], color=NAVY)
    c.node("风险旁路：重新累计正负路径分数\n每个请求最多一个 pair；候选选择 stop-gradient",
           6.48, 3.84, 5.10, 1.05, emphasis=True, size=17)
    c.node("主干：完整训练集标准 CE\n保留 Query / GID / 历史 / 固定 [S1,S2,S3,C]",
           0.65, 3.84, 5.15, 1.05, size=17)
    c.arrow([(5.80, 4.37), (6.45, 4.37)], color=NAVY)
    c.node("共享 Qwen 参数更新\n线上仍用原 Beam Search", 4.45, 5.25, 4.75, 0.78,
           emphasis=True, size=17)
    c.arrow([(3.23, 4.89), (3.23, 5.64), (4.42, 5.64)], color=MUTED)
    c.arrow([(9.03, 4.89), (9.03, 5.64), (9.23, 5.64)], color=MUTED)
    c.note("创新对象是 SFT 训练目标，而不是 SID、Tokenizer、Trie 或推理打分。这样才能把收益归因到“训练看见真实 Beam 风险”。")


def beam_first_prune(deck) -> None:
    c = Canvas(deck, "机制一：只在首次剪枝的因果位置纠偏",
               "如果 gold 在深度 t* 首次掉出 Beam，只比较同深度 gold 前缀与实际第 K 名边界。",
               "Beam-Search Optimization 的第 K 名边界思想；BeamRisk-SFT classify_beam_risk()")
    c.text("示意：K=3，gold 在 S2 首次掉出", 0.65, 1.93, 5.75, 0.41,
           size=18, bold=True, color=BLACK)
    level_x = [1.20, 2.75, 4.30, 5.85]
    for x, label in zip(level_x, ["S1", "S2", "S3", "C"]):
        c.text(label, x, 2.46, 0.72, 0.36, size=14, color=MUTED, center=True)
    c.token("g1", level_x[0], 3.40, selected=True)
    for y, token in [(2.92, "a2"), (3.55, "b2"), (4.18, "c2")]:
        c.token(token, level_x[1], y)
        c.arrow([(level_x[0]+0.70, 3.65), (level_x[1]-0.04, y+0.25)], width=0.7)
    c.token("g2", level_x[1], 4.81, selected=True)
    c.arrow([(level_x[0]+0.70, 3.65), (level_x[1]-0.04, 5.06)], color=NAVY,
            width=2.0, dashed=True)
    for y, token in [(2.92, "a3"), (3.55, "b3"), (4.18, "c3")]:
        c.token(token, level_x[2], y)
        c.arrow([(level_x[1]+0.70, y+0.25), (level_x[2]-0.04, y+0.25)], width=0.7)
    c.text("正样本 g_t*", 0.76, 5.30, 1.70, 0.35, color=NAVY, bold=True)
    c.text("负样本 b_t*：真实第 K 名", 2.68, 5.30, 3.28, 0.35, color=MUTED)
    c.line(6.56, 1.94, 6.56, 5.76)
    c.text("路径累计分数", 7.02, 2.02, 5.61, 0.40, size=18, bold=True,
           color=BLACK)
    c.equation(r"\operatorname{Score}(g_t\mid x)=\sum_{j=1}^{t}\log P_\theta(y_j\mid x,y_{<j})",
               7.02, 2.66, 5.61, 0.84, emphasis=True, size=19)
    c.text("只重算两条同长度路径，保证分数可比较。\n不对 t* 之后的错误重复造 pair，避免一个请求产生多份高度相关梯度。",
           7.02, 3.84, 5.61, 1.04, size=16)
    c.equation(r"\Delta_{\mathrm{survive}}=\operatorname{Score}(b_{t^\star}\mid x)-\operatorname{Score}(g_{t^\star}\mid x)",
               7.02, 5.06, 5.61, 0.73, size=18)
    c.note("如果失败发生在 S1 之前的格式 token，记录为 pre-SID prune，只做诊断、不训练 SID 风险。", 6.23, 0.52)


def beam_states(deck) -> None:
    c = Canvas(deck, "机制二：先存活，再排序；三种状态互斥",
               "每个请求只解决当前最接近检索指标的一个问题，避免从多个前缀重复施加风险梯度。",
               "beamrisk_sft/README.md §6.3–6.6")
    c.node("当前模型对请求 x\n运行真实 Beam=10", 0.65, 3.06, 2.85, 1.00,
           emphasis=True, size=17)
    branches = [
        (1.92, "gold 未进 Top-10", "Survive", "gold 首剪枝前缀\nvs 当层第 10 名", "目标：HR@10"),
        (3.42, "gold 在第 2–10 名", "Rank", "完整 gold SID\nvs 当前 rank-1 错误 SID", "目标：HR@1 / NDCG"),
        (4.92, "gold 已经 rank-1", "CE only", "不构造风险 pair\n防止过度优化已正确样本", "目标：保持"),
    ]
    for y, state, risk, pair, goal in branches:
        c.node(state, 4.35, y, 2.38, 0.78, size=16)
        c.node(risk, 7.35, y, 1.56, 0.78, emphasis=risk != "CE only",
               size=16, bold=True)
        c.text(pair, 9.20, y-0.06, 2.34, 0.92, size=14.7)
        c.text(goal, 11.65, y-0.03, 0.98, 0.84, size=13.2, color=MUTED,
               center=True)
        c.arrow([(3.50, 3.56), (3.92, 3.56), (3.92, y+0.39), (4.32, y+0.39)])
        c.arrow([(6.73, y+0.39), (7.32, y+0.39)])
        c.arrow([(8.91, y+0.39), (9.17, y+0.39)])
    c.note("完整路径负例只在严格目录等价检查通过后使用；名称、地址、别名、坐标完全相同的重复实体跳过风险 loss，但仍参加 CE。")


def beam_math(deck) -> None:
    c = Canvas(deck, "BeamRisk v1 的计算：同一模型、两条路径、一个 margin",
               "风险候选是离散 stop-gradient；训练时用当前参数重新计算 gold 与 negative 的累计 log 概率。",
               "beamrisk_sft/src/beamrisk_sft/scoring.py 与 loss.py；v1 已实现")
    c.equation(r"\operatorname{Score}(p\mid x)=\sum_{j=1}^{|p|}\log P_\theta(p_j\mid x,p_{<j})",
               0.65, 2.02, 5.80, 0.83, emphasis=True, size=21)
    c.equation(r"\Delta=\operatorname{Score}(p^-\mid x)-\operatorname{Score}(p^+\mid x)",
               6.83, 2.02, 5.80, 0.83, emphasis=True, size=21)
    c.text("miss@10", 0.65, 3.35, 1.36, 0.36, size=16, bold=True, color=NAVY)
    c.text("negative = 首剪枝深度的第 K 名边界前缀", 2.08, 3.35, 4.61, 0.45,
           size=16)
    c.text("hit@10_not@1", 6.83, 3.35, 2.06, 0.36, size=16, bold=True,
           color=NAVY)
    c.text("negative = 当前 Beam 第 1 名错误完整 SID", 8.94, 3.31, 3.69, 0.66,
           size=15.5)
    c.equation(r"L_{\mathrm{risk}}^{(v1)}=\operatorname{softplus}\!\left(\frac{\Delta+\mu}{\tau}\right)",
               1.37, 4.28, 4.90, 0.83, size=20)
    c.equation(r"L_{\mathrm{total}}=L_{\mathrm{CE}}+\lambda_{\mathrm{risk}}L_{\mathrm{risk}}", 7.08, 4.28, 4.90, 0.83,
               size=20)
    c.text("v1：τ=1，margin=0，λ_risk=0.2；每 4 个 optimizer step 注入一次风险 batch。",
           0.65, 5.42, 12.0, 0.48, size=16)
    c.note("softplus 的关键隐患：即使 Δ≤0，即 gold 已超过 negative，梯度仍不为零；后续会持续扩大不必要的间隔。", 6.16, 0.56, color=NAVY)


def beam_revision(deck) -> None:
    c = Canvas(deck, "v1 复盘：问题来自目标形状与梯度耦合",
               "当前负结果不支持原配置重跑；它给出了 BeamRisk v2 必须同时修正的四个机制。",
               "beamrisk_sft/BEAMRISK_V1_RESULT_ANALYSIS.md；以下 v2 为设计草案，未实现")
    items = [
        ("无界 softplus", "越过边界后仍继续推大 margin", "零边界 Huber-hinge\n满足条件即停止风险梯度"),
        ("跨桶 final-rank", "大部分 pair 改动 S1/S2/S3 语义树", "分层门控\n先 first-prune；排序仅保留同桶 pair"),
        ("合并后统一裁剪", "风险梯度过强时连带压缩主 CE", "独立梯度预算\n风险范数限制为 CE 的固定比例"),
        ("单点击假负例", "站点—入口等关系 POI 可能都合理", "关系保护 / 软负例\n不把合理近等价实体强行压低"),
    ]
    for i, (problem, effect, fix) in enumerate(items):
        y = 1.93 + i * 1.05
        c.text(problem, 0.65, y, 2.16, 0.36, size=17, bold=True, color=BLACK)
        c.text(effect, 2.76, y, 3.58, 0.65, size=15.5)
        c.arrow([(6.35, y+0.34), (6.92, y+0.34)], color=NAVY, width=1.4)
        c.node(fix, 7.12, y-0.13, 5.51, 0.86, emphasis=True, size=15.5)
    c.equation(r"L_{\mathrm{boundary}}=\phi(\Delta+\mu),\qquad \phi(z)=0\ \mathrm{for}\ z\leq0",
               2.18, 6.03, 9.00, 0.64, emphasis=True, size=19)


def beam_v2(deck) -> None:
    c = Canvas(deck, "BeamRisk v2 草案：边界截断、分层门控、独立预算",
               "先保证 gold 跨过真实 Beam 边界，再在不会破坏语义前缀的范围内优化最终排序。",
               "由 v1 结果分析凝练；尚未修改代码、未运行")
    c.text("1  零边界平滑 hinge", 0.65, 1.95, 3.25, 0.38, size=18, bold=True,
           color=BLACK)
    c.equation(r"z=\Delta+\mu"
               "\n" r"\phi(z)=0,\qquad z\leq0"
               "\n" r"\phi(z)=\frac{z^2}{2\delta},\qquad 0<z\leq\delta"
               "\n" r"\phi(z)=z-\frac{\delta}{2},\qquad z>\delta",
               0.65, 2.51, 3.85, 2.03, emphasis=True, size=17)
    c.text("gold 达到 margin 后梯度严格为 0；δ 控制边界附近的平滑程度。",
           0.65, 4.78, 3.85, 0.70, size=15.5)
    c.line(4.87, 1.95, 4.87, 5.69)
    c.text("2  分层风险门控", 5.23, 1.95, 3.12, 0.38, size=18, bold=True,
           color=BLACK)
    c.node("Epoch 2\n只学 first-prune 生存", 5.23, 2.64, 3.07, 0.87,
           emphasis=True)
    c.node("Epoch 3\n刷新 first-prune\n+ 同 S1-S3 的 C 排序", 5.23, 3.92, 3.07, 1.13)
    c.arrow([(6.77, 3.51), (6.77, 3.89)], color=NAVY)
    c.line(8.68, 1.95, 8.68, 5.69)
    c.text("3  风险梯度独立预算", 9.04, 1.95, 3.59, 0.38, size=18, bold=True,
           color=BLACK)
    c.equation(r"g_{\mathrm{risk}}\leftarrow\min\!\left(1,\rho\frac{\|g_{\mathrm{CE}}\|_2}{\|g_{\mathrm{risk}}\|_2}\right)g_{\mathrm{risk}}",
               9.04, 2.71, 3.59, 1.21, size=17)
    c.text("候选起点 ρ=0.1–0.2；先单独缩放风险梯度，再与 CE 合并并执行全局裁剪。",
           9.04, 4.28, 3.59, 1.03, size=15.5)
    c.note("v2 的可证伪门禁：S1–S3 的自由生成生存率先不退化；风险 margin 不再无限增大；关系保护样本不被主动压低。", 6.12, 0.64)


def beam_algorithm(deck) -> None:
    c = Canvas(deck, "BeamRisk 的完整训练算法：策略建立—挖掘—风险对齐",
               "从相同扩词基座独立训练三轮；500k 只是挖掘候选池，主 CE 每轮仍遍历全量 Train。",
               "beamrisk_sft/README.md §7–9；v1 流程已实现，v2 门控部分待实现")
    connect_row(c, ["初始化\nQwen + TIGER tokens", "Epoch 1\n全量标准 CE", "Mine #1\nTrain-only Beam=10", "Epoch 2\nCE + Survive", "Mine #2\n刷新风险", "Epoch 3\nCE + 分层风险"],
                2.10, h=0.92, emphasis=2, size=15)
    c.text("每次挖掘保存", 0.65, 3.56, 2.16, 0.38, size=18, bold=True,
           color=BLACK)
    c.text("每层 live Beam 前缀与分数；gold rank；首剪枝深度；第 K 名边界；最终 rank；合法性和目录可展开属性。",
           2.93, 3.56, 9.70, 0.72, size=15.5)
    c.line(0.65, 4.50, 12.63, 4.50)
    c.text("风险 pair 进入训练前", 0.65, 4.80, 2.35, 0.68, size=18, bold=True,
           color=BLACK)
    c.text("重算当前正负路径分数；过滤相同路径、结构非法、目录不可展开和已满足边界样本；保护目录等价与关系型近等价实体。",
           3.08, 4.82, 9.55, 0.76, size=15.5)
    c.note("评测仍只看固定 10k 的无约束 Beam=10 主指标；合法路径约束作为并列诊断，不改变训练目标。线上推理路径与 TIGER 完全一致。")


def qg_observation(deck) -> None:
    c = Canvas(deck, "SID 实验暴露：结构更好，不代表 Query 更容易生成",
               "QG-PRQK 不再追求单一静态指标，而把已有负向证据直接写进 Tokenizer 的约束。",
               "QG_PRQK_METHOD_SPEC.md §3；本项目 Embedding、RQ-KMeans、类别与 Geo 消融")
    items = [
        ("静态碰撞减少", "但 SFT 未同步提升", "SID 目标必须显式考虑 Query 可预测性"),
        ("Query 聚合能改善连续召回", "但逐 POI 动态融合不稳定", "保留独立 Query view，不覆盖 POI 向量"),
        ("S1 硬绑定 category", "大类被压进一个根节点，路径坍塌", "类别只做可分裂的软结构锚点"),
        ("粗层加入绝对地理", "会干扰跨区域语义组织", "Geo 只在给定 GID/S1/S2 的 S3 局部消歧"),
    ]
    for i, (evidence, conflict, design) in enumerate(items):
        y = 1.93 + i * 1.07
        c.text(evidence, 0.65, y-0.04, 2.48, 0.66, size=16.5, bold=True, color=BLACK)
        c.text(conflict, 3.08, y, 3.11, 0.63, size=15.5)
        c.arrow([(6.23, y+0.34), (6.80, y+0.34)], color=NAVY)
        c.node(design, 7.00, y-0.14, 5.63, 0.88, emphasis=True, size=15.5)
    c.note("核心判断：可生成性不是 POI embedding 的附属指标，而是 SID 离散分配本身应优化的对象。")


def qg_related(deck) -> None:
    c = Canvas(deck, "论文给了内容、Query、类别与 Geo，但仍缺粒度感知的统一设计",
               "QG-PRQK 不是复刻某一篇论文，而是组合其有效机制并针对地图 POI 的失败模式重新约束。",
               "TIGER, NeurIPS 2023；CQ-SID, arXiv:2605.14434；PRQ-KMeans, arXiv:2608.24207；GenPOI, 2026")
    items = [
        ("TIGER", "POI 内容 → RQ-VAE → 分层 SID", "Tokenizer 不看真实 Query 分布"),
        ("PRQ-KMeans", "去全局方向 + soft centroid + projection residual", "只建模单一实体视图，不对齐 Query token"),
        ("CQ-SID", "首层类别约束 + 高置信 Query–Item Bi-InfoNCE", "硬 category index；Query 对都按精确监督处理"),
        ("GenPOI", "文本与 Geo 表示、GID 与生成约束", "绝对地理解决位置编码，未定义 Query 可监督到哪一层"),
    ]
    for i, (paper, mechanism, gap) in enumerate(items):
        y = 1.88 + i * 1.03
        c.text(paper, 0.65, y, 1.76, 0.37, size=17, bold=True, color=BLACK)
        c.text(mechanism, 2.52, y, 4.16, 0.64, size=15.5)
        c.arrow([(6.73, y+0.31), (7.24, y+0.31)], color=NAVY)
        c.text(gap, 7.46, y, 5.17, 0.64, size=15.5)
        c.line(0.65, y+0.79, 12.63, y+0.79)
    c.node("我们的缺口：Query depth + 双视图共享 token + 类别软锚点 + S3 局部地理",
           1.26, 6.08, 10.76, 0.60, emphasis=True, size=18)


def qg_framework(deck) -> None:
    c = Canvas(deck, "QG-PRQK：类别层次锚定的 Query–Geo 渐进残差量化",
               "Query 先判监督深度，再以独立视图参与对应层分配；三层量化后补确定性去重形成完整检索 ID。",
               "QG_PRQK_METHOD_SPEC.md v2.1-CAT；主码本 512×3")
    c.node("冻结 active POI\nBGE 内容 + 类别 + 坐标", 0.65, 2.02, 2.32, 0.90,
           size=15.5)
    c.node("POI residual view\n去全局方向", 3.43, 2.02, 2.17, 0.90,
           size=15.5)
    c.arrow([(2.97, 2.47), (3.40, 2.47)], color=NAVY)

    c.node("训练集 Query–POI 边\n频次 / 类别集中度 / 熵", 0.65, 3.56, 2.32, 1.02,
           size=15.0)
    c.node("监督深度与可靠性\nD0 / D1 / D2 / D3", 3.43, 3.34, 2.17, 0.90,
           emphasis=True, size=15.0)
    c.node("D3 Exact Adapter\n独立 Query residual view", 3.43, 4.55, 2.17, 0.90,
           size=14.8)
    c.arrow([(2.97, 4.07), (3.40, 3.79)], color=NAVY)
    c.arrow([(2.97, 4.07), (3.40, 5.00)], color=NAVY)

    c.node("S1\n双质心共享\n离散 token\n+ 粗类软锚点", 6.12, 2.09, 1.77, 1.35,
           emphasis=True, size=14.7)
    c.node("S2\n双视图残差\n+ 条件细类锚点", 8.35, 2.09, 1.77, 1.35,
           emphasis=True, size=14.7)
    c.node("S3\nD3 双视图残差\n+ 局部 Geo / 困难图", 10.58, 2.09, 1.77, 1.35,
           emphasis=True, size=14.2)
    c.arrow([(5.60, 2.47), (6.09, 2.47)], color=NAVY)
    c.arrow([(5.60, 3.79), (5.86, 3.79), (5.86, 2.77), (6.09, 2.77)], color=NAVY)
    c.arrow([(5.60, 5.00), (5.94, 5.00), (5.94, 3.07), (6.09, 3.07)], color=NAVY)
    c.arrow([(7.89, 2.77), (8.32, 2.77)], color=NAVY)
    c.arrow([(10.12, 2.77), (10.55, 2.77)], color=NAVY)
    c.text("projection residual", 7.15, 3.62, 4.22, 0.34, size=15,
           bold=True, color=NAVY, center=True)
    c.arrow([(6.52, 4.08), (11.95, 4.08)], color=NAVY, width=1.5)

    c.node("QG SID = [S1, S2, S3]", 8.17, 4.42, 4.18, 0.65,
           emphasis=True, size=17, bold=True)
    c.arrow([(11.46, 3.44), (11.46, 4.39)], color=NAVY)
    c.node("残留碰撞追加确定性 C\nFinal PID = [S1,S2,S3,C]",
           8.17, 5.36, 4.18, 0.62, size=14.8)
    c.arrow([(10.26, 5.07), (10.26, 5.33)], color=NAVY)
    c.text("D0 不入图；D1→S1；D2→S1–S2；D3→S1–S3。\n类别只约束 S1/S2；相对 Geo 只在 GID6/S1/S2 父组内约束 S3。",
           0.65, 5.66, 6.96, 0.62, size=14.8)
    c.note("第四块补全完整检索 ID：QG-PRQK 输出三层语义 SID；仅对语义不可分的残留实体追加确定性 C。", 6.28, 0.44)


def qg_depth(deck) -> None:
    c = Canvas(deck, "机制一：Query 能确定多少语义，就监督到多少层",
               "同一个 Query 的目标分布先按 coarse / fine / exact 聚合，再由集中度与熵决定监督深度。",
               "QG_PRQK_METHOD_SPEC.md §4；统计仅使用 Train")
    examples = [
        ("附近好吃的", "D0", "目标跨类别且分散", "不监督 SID"),
        ("火锅", "D1", "粗类别集中", "S1"),
        ("川味火锅", "D2", "细类别集中", "S1 + S2"),
        ("海底捞西单店", "D3", "Exact POI 高置信", "S1 + S2 + S3"),
    ]
    for i, (query, depth, reason, layers) in enumerate(examples):
        y = 1.92 + i * 0.87
        c.text(f"“{query}”", 0.65, y-0.03, 2.48, 0.45, size=17, bold=True,
               color=BLACK)
        c.token(depth, 3.15, y-0.04, selected=depth != "D0", w=0.86, h=0.48)
        c.text(reason, 4.32, y, 3.12, 0.47, size=15.5)
        c.arrow([(7.45, y+0.25), (8.02, y+0.25)], color=NAVY)
        c.text(layers, 8.24, y, 1.62, 0.39, size=16, bold=True, color=NAVY)
        for j, label in enumerate(["S1", "S2", "S3"]):
            selected = label in layers
            c.token(label, 10.15+j*0.78, y-0.04, selected=selected,
                    w=0.63, h=0.48, size=14)
    c.equation(r"P_l(c\mid q)=\sum_{p\in c}P(p\mid q),\qquad C_l(q)=\max_c P_l(c\mid q)"
               "\n" r"H_l(q)=-\frac{\sum_c P_l(c\mid q)\log P_l(c\mid q)}{\log|\mathcal{C}_l|}",
               0.65, 5.45, 7.40, 1.02, emphasis=True, size=18)
    c.text("D3 沿用 Exact-Core；其余 Query 先判 D2，再判 D1。\n粗类 Query 不会被迫区分具体门店。",
           8.47, 5.45, 4.16, 0.99, size=15.5)


def qg_reliability(deck) -> None:
    c = Canvas(deck, "Query–POI 图：边权同时表达频次、集中度与歧义",
               "Query 可连接多个真实 POI，但每层的总边质量被可靠性限制，避免热门粗 Query 主导聚类。",
               "QG_PRQK_METHOD_SPEC.md §4.4、§6；P4 medium 已完成数据与路由验收")
    c.node("Query q\nsupport / concentration / entropy", 0.65, 3.03, 2.85, 1.02,
           emphasis=True, size=16)
    poi_y = [2.10, 3.43, 4.76]
    for i, y in enumerate(poi_y, 1):
        c.node(f"POI p{i}\n真实 Train 发单边", 5.08, y, 2.43, 0.78, size=15.5)
        c.arrow([(3.50, 3.54), (4.25, 3.54), (4.25, y+0.39), (5.05, y+0.39)],
                color=NAVY if i == 1 else MUTED, width=1.4 if i == 1 else 0.9)
    c.text("layer mask", 7.86, 2.10, 1.61, 0.32, size=15, bold=True, color=BLACK)
    c.text("D1 [1,0,0]\nD2 [1,1,0]\nD3 [1,1,1]", 7.86, 2.62, 1.83, 1.28,
           size=16)
    c.line(9.91, 1.96, 9.91, 5.88)
    c.equation(r"r_l(q)=\frac{\log(1+\min(n_q,20))}{\log 21}"
               "\n" r"\cdot C_l(q)^\gamma\left(1-H_l(q)\right)",
               10.23, 2.09, 2.40, 1.48, emphasis=True, size=16)
    c.equation(r"\omega_l(q,p)=\frac{n(q,p)}{\sum_{p'}n(q,p')}"
               "\n" r"w_l(q,p)=r_l(q)\,\omega_l(q,p)",
               10.23, 4.18, 2.40, 1.19, size=15.5)
    c.equation(r"\sum_p\omega_l(q,p)=1", 10.23, 5.58, 2.40, 0.43,
               emphasis=True, size=14)
    c.note("D1/D2 只保留主导类别内的真实发单边；D3 使用 Exact-Core 主目标。false-negative targets 只做保护，不扩成 S3 强正边。")


def qg_adapter(deck) -> None:
    c = Canvas(deck, "机制二：Exact Query Adapter 只校正 D3 Query view",
               "短 Query 与长 POI 文本不必共享同一几何；保持 POI 向量冻结，用小残差网络校正精确 Query。",
               "QG_PRQK_METHOD_SPEC.md §5；query_adapter.py；P3A-FULL 已完成")
    connect_row(c, ["Raw BGE(q)\n1,024D", "LayerNorm\nLinear 1024→64", "GELU + Dropout\nLinear 64→1024", "Residual add\nL2 normalize"],
                2.05, emphasis=3)
    c.arrow([(1.77, 2.97), (1.77, 3.43), (11.16, 3.43), (11.16, 3.00)],
            color=NAVY, width=1.7)
    c.text("identity 路径", 5.16, 3.57, 2.42, 0.35, size=15, color=NAVY,
           bold=True, center=True)
    c.node("正样本\nD3 Exact POI", 0.65, 4.34, 2.25, 0.80, emphasis=True)
    c.node("16 个固定困难负例\n语义6 / 词法4 / 局部地理6", 3.47, 4.34, 3.55, 0.80)
    c.node("动态 in-batch 负例\n目标与合理正例全部 mask", 7.59, 4.34, 3.50, 0.80)
    c.arrow([(11.09, 4.74), (11.57, 4.74)], color=NAVY)
    c.equation(r"L_{\mathrm{NCE}}=\sum_q\widehat{w}_q\left[\log\sum_j\exp\!\left(\frac{s(q,p_j)}{\tau}\right)-\frac{s(q,p^+)}{\tau}\right]",
               1.33, 5.38, 10.65, 0.70, emphasis=True, size=18)
    c.note("输出层零初始化，从 identity 开始。Adapter 只服务离线 SID Tokenizer，不是线上 Qwen 前处理，也不改原始 SFT Query。", 6.20, 0.64)


def qg_dual_view(deck) -> None:
    c = Canvas(deck, "机制三：POI 与 Query 双质心，共享同一个离散 token",
               "同一 token k 对应 POI 中心 U_l,k 与 Query 中心 V_l,k；通过图一致性对齐分配，而不是强行共用向量中心。",
               "QG_PRQK_METHOD_SPEC.md §7.1–7.7")
    c.node("POI residual r_p^(l−1)", 0.65, 2.23, 2.75, 0.77, size=16)
    c.node("POI codebook U_l\n512 个中心", 4.00, 2.09, 2.67, 1.02,
           emphasis=True)
    c.node("Query residual r_q^(l−1)\n仅 d(q)≥l", 0.65, 4.45, 2.75, 0.92, size=16)
    c.node("Query codebook V_l\n512 个中心", 4.00, 4.40, 2.67, 1.02,
           emphasis=True)
    c.arrow([(3.40, 2.61), (3.97, 2.61)], color=NAVY)
    c.arrow([(3.40, 4.91), (3.97, 4.91)], color=NAVY)
    c.node("共享 token k\nI[t_q ≠ s_p]", 7.39, 3.18, 2.05, 1.02,
           emphasis=True, size=17, bold=True)
    c.arrow([(6.67, 2.61), (7.02, 2.61), (7.02, 3.69), (7.36, 3.69)])
    c.arrow([(6.67, 4.91), (7.02, 4.91), (7.02, 3.91), (7.36, 3.91)])
    c.equation(r"\mathcal{J}_l=\sum_p d(r_p,u_{l,s_p})"
               "\n" r"+\alpha_l\sum_q d(r_q,v_{l,t_q})"
               "\n" r"+\lambda_l\sum_{(q,p)}w_l(q,p)\,\mathbf{1}[t_q\neq s_p]"
               "\n" r"+\eta_l\sum_p D_{\mathrm{cat},l}(p,s_p)",
               10.01, 2.16, 2.62, 2.44, emphasis=True, size=16)
    c.text("稀疏 Query 中心收缩", 9.83, 4.95, 2.80, 0.35, size=16, bold=True,
           color=BLACK)
    c.equation(r"v_{l,k}\leftarrow\operatorname{norm}\!\left(\frac{\sum_{q:t_q=k}r_q+\tau u_{l,k}}{n_{q,k}+\tau}\right)", 9.61, 5.43, 3.02, 0.62,
               size=14.5)
    c.note("S1 使用 D1/D2/D3；S2 只使用 D2/D3；S3 只使用 D3。POI 与 Query assignment、中心分开更新，token index 保持共享。")


def qg_category_residual(deck) -> None:
    c = Canvas(deck, "类别是软锚点，residual 是逐层传递的未解释信息",
               "类别只鼓励路径更纯，不把 CategoryID 硬编码成 S1；projection residual 避免下一层重复解释当前方向。",
               "CQ-SID 的 category constraint；PRQ-KMeans projection residual；QG_PRQK_METHOD_SPEC.md §7")
    c.text("S1：粗类别软代价", 0.65, 1.97, 3.40, 0.40, size=18, bold=True,
           color=BLACK)
    c.equation(r"P(c\mid k)=\frac{n(k,c)+\alpha_1P_{\mathrm{global}}(c)}{n(k)+\alpha_1}"
               "\n" r"D_{\mathrm{cat},1}(p,k)=-\frac{\log P(c_{\mathrm{coarse}}(p)\mid k)}{\log|\mathcal{C}_{\mathrm{coarse}}|}",
               0.65, 2.54, 5.66, 1.18, emphasis=True, size=17)
    c.text("一个粗类可以占用多个 S1 code；避免“大类压进单一根节点”。",
           0.65, 3.94, 5.66, 0.63, size=15.5)
    c.text("S2：条件细类别软代价", 0.65, 4.85, 3.76, 0.40, size=18,
           bold=True, color=BLACK)
    c.equation(r"D_{\mathrm{cat},2}(p,k)\propto-\log P\!\left(c_{\mathrm{fine}}(p)\mid S_1(p),k\right)",
               0.65, 5.42, 5.66, 0.65, size=17)
    c.line(6.69, 1.95, 6.69, 6.12)
    c.text("逐层 projection residual", 7.08, 1.97, 4.45, 0.40, size=18,
           bold=True, color=BLACK)
    c.node("当前 residual r", 7.08, 2.68, 2.14, 0.72)
    c.node("选中中心 c_k", 10.28, 2.68, 2.14, 0.72, emphasis=True)
    c.arrow([(9.22, 3.04), (10.25, 3.04)], color=NAVY)
    c.equation(r"\operatorname{Proj}_{c_k}(r)=\frac{r^\top c_k}{\|c_k\|_2^2}\,c_k", 7.08, 3.80, 5.34, 0.70,
               emphasis=True, size=18)
    c.equation(r"r_{\mathrm{next}}=\operatorname{norm}\!\left(r-\operatorname{Proj}_{c_k}(r)\right)", 7.08, 4.78, 5.34, 0.70,
               emphasis=True, size=18)
    c.text("POI 沿 U_l 中心递推；Query 沿 V_l 中心递推。\nD1 到 S1 停，D2 到 S2 停，D3 继续到 S3。",
           7.08, 5.65, 5.34, 0.72, size=15.5)


def qg_geo(deck) -> None:
    c = Canvas(deck, "S3：在固定父组内用局部地理和困难实体做消歧",
               "绝对位置由 GID 提供；Geo 只描述同一 GID6/S1/S2 父组内部的相对结构。",
               "QG_PRQK_METHOD_SPEC.md §8；当前为设计阶段，P7 未实现")
    c.node("parent = (GID6, S1, S2)", 0.65, 3.12, 2.75, 0.85,
           emphasis=True, size=17)
    c.node("局部 Geo view\ncell dx/dy\nlog(1+d), sin/cos bearing", 4.02, 1.95, 3.28, 1.33,
           size=15.5)
    c.node("困难实体图\n同细类 + 同/邻 GID6\n名称或 BGE 近似；Top-20", 4.02, 3.74, 3.28, 1.33,
           size=15.5)
    c.node("POI residual + D3 Query residual", 4.02, 5.49, 3.28, 0.65,
           size=15)
    c.arrow([(3.40, 3.54), (3.72, 3.54), (3.72, 2.61), (3.99, 2.61)])
    c.arrow([(3.40, 3.54), (3.72, 3.54), (3.72, 4.41), (3.99, 4.41)])
    c.node("S3 assignment", 8.20, 3.02, 2.22, 1.05, emphasis=True,
           size=18, bold=True)
    for y in (2.61, 4.41, 5.81):
        c.arrow([(7.30, y), (7.73, y), (7.73, 3.55), (8.17, 3.55)], color=NAVY)
    c.text("单例父组 Geo gate=0；保护 false-negative targets；不强制组内每个 POI 使用不同 S3。",
           8.20, 4.47, 4.43, 0.70, size=15.2)
    c.equation(r"\mathcal{J}_3=\mathcal{J}_{3,\mathrm{dual}}+\gamma_{\mathrm{geo}}D_{\mathrm{local}}+\rho_{\mathrm{hard}}C_{\mathrm{hard}}",
               8.20, 5.43, 4.43, 0.67, emphasis=True, size=16)
    c.note("经纬度不拼入 BGE，Geo 不进入 S1/S2。这里也不是用户当前位置输入：用户 GID 仍作为生成模型的请求条件。", 6.20, 0.55)


def qg_algorithm(deck) -> None:
    c = Canvas(deck, "QG-PRQK 的构建算法：逐层交替分配，而不是反向传播",
               "每层从 POI-only PRQ-KMeans 初始化，再交替更新 POI/Query assignment、中心和类别分布。",
               "QG_PRQK_METHOD_SPEC.md §7.6、§9；P5–P8 尚未运行")
    connect_row(c, ["A0：POI-only PRQK\n初始化 U_l, s_p", "分配 Query\n初始化 V_l, t_q", "更新 POI assignment\n内容 + 图 + 类别", "更新 Query assignment\n距离 + 图", "更新 U/V 与类别分布\nQuery 中心收缩", "收敛后生成 residual\n进入下一层"],
                2.04, h=1.12, emphasis=0, size=13.7)
    c.arrow([(11.61, 3.17), (11.61, 3.55), (4.78, 3.55), (4.78, 3.19)],
            color=NAVY, width=1.4)
    c.text("每层循环", 7.50, 3.70, 1.60, 0.35, size=15, bold=True,
           color=NAVY, center=True)
    c.label("Assignment 代价", "POI：内容距离 + 图不一致 + 类别代价\nQuery：查询距离 + 图不一致",
            0.65, 4.35, 5.70, h=1.28)
    c.label("停止条件", "目标相对改善 <1e-4；POI 变化 <0.1%；Query <0.2%；\n连续两轮满足，或最多 30 轮。",
            6.93, 4.35, 5.70, h=1.28)
    c.note("RQ-KMeans 没有 optimizer / learning rate，但必须记录复合 objective、各项 distortion、图不一致率、类别 CE、活跃码和桶分布。")


def status_and_relation(deck) -> None:
    c = Canvas(deck, "实现边界：两条方法正好位于检索链路的上下游",
               "QG-PRQK 先改善可生成 SID；BeamRisk-SFT 再改善固定 SID 的生成策略，必须分阶段归因。",
               "QG 实施状态 2026-09-07；BeamRisk v1 结果分析与 v2 草案")
    c.text("QG-PRQK：Tokenizer 侧", 0.65, 1.92, 4.15, 0.40, size=19,
           bold=True, color=BLACK)
    connect_row(c, ["已完成\nQuery stats / depth", "已完成\nD3 Adapter", "已完成\nP4 medium 图数据", "等待审核\nP5–P8 SID"],
                2.55, x=0.65, w=12.0, h=0.86, emphasis=2, size=14.8)
    c.arrow([(11.11, 3.43), (11.11, 3.82), (2.17, 3.82), (2.17, 4.19)],
            color=NAVY, width=1.4)
    c.text("BeamRisk-SFT：Generator 侧", 0.65, 4.20, 4.42, 0.40, size=19,
           bold=True, color=BLACK)
    connect_row(c, ["固定已验收 SID", "v1 已完成\n真实 Beam 风险旁路", "v1 复盘冻结", "v2 草案\n边界截断 / 门控 / 预算"],
                4.82, x=0.65, w=12.0, h=0.86, emphasis=3, size=14.8)
    c.note("组合顺序：先独立验收 QG 静态 SID 与 Prefix Probe；获批后使用同一 SFT 协议建立普通 CE 基线；只有基线稳定，才评 BeamRisk v2。", 6.14, 0.61)


def contribution(deck) -> None:
    c = Canvas(deck, "方法创新点与可证伪边界",
               "创新不是模块堆叠，而是把两类训练—使用失配变成可计算、可审计、可逐步验证的目标。",
               "本页仅总结方法主张；未把未完成阶段或内部验证写成最终提升")
    c.label("BeamRisk-SFT", "真实 Beam 的首个因果剪枝点\nSurvive-then-Rank 互斥状态\n训练旁路，不增加线上推理模块",
            0.65, 2.05, 5.64, h=1.72, emphasis=True)
    c.label("QG-PRQK", "Query 可确定粒度控制监督深度\nPOI/Query 双质心共享 token\n类别软锚点 + S3 局部地理",
            6.98, 2.05, 5.64, h=1.72, emphasis=True)
    c.text("必须证明什么", 0.65, 4.27, 2.30, 0.46, size=19, bold=True,
           color=BLACK)
    c.text("BeamRisk：gold 的自由生成前缀生存与最终排序改善，同时 CE 主干不被风险梯度破坏。\nQG：Query–POI token 一致性、类别结构、局部消歧和 Prefix Probe 同时改善，不能只看唯一率或类别纯度。",
           0.65, 4.90, 12.0, 0.96, size=16)
    c.note("最终判定仍回到同协议生成式 HR / NDCG。当前状态：BeamRisk v1 已复盘，v2 未实现；QG 已完成 P4 medium，尚无完整 SID 或 SFT。", 6.12, 0.66)


def references(deck) -> None:
    c = Canvas(deck, "参考工作",
               "方法组织参考优秀论文汇报常见的“反例—相关工作缺口—总框架—模块放大—公式—算法—边界”结构。",
               "公开论文与本地 papers/；访问核对日期 2026-09-07")
    left = [
        "Rajput et al.  Recommender Systems with Generative Retrieval (TIGER). NeurIPS 2023.",
        "Wiseman & Rush.  Sequence-to-Sequence Learning as Beam-Search Optimization. EMNLP 2016.",
        "Lee et al.  GLEN: Generative Retrieval via Lexical Index Learning. EMNLP 2023.",
        "Zeng et al.  Scalable and Effective Generative Information Retrieval (RIPOR). SIGIR 2024.",
    ]
    right = [
        "Zhou et al.  Enhancing Generative Retrieval with Reinforcement Learning from Relevance Feedback. EMNLP 2023.",
        "Zhu et al.  Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL. arXiv:2605.14434.",
        "Luo et al.  PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization. arXiv:2608.24207v2.",
        "Chen et al.  Revisiting General Map Search via Generative POI Retrieval (GenPOI). 2026.",
    ]
    c.text("Beam / Prefix Optimization", 0.65, 1.98, 5.70, 0.40, size=19,
           bold=True, color=BLACK)
    c.line(0.65, 2.49, 6.02, 2.49, color=NAVY, width=1.1)
    for i, text in enumerate(left):
        c.text(text, 0.65, 2.70+i*0.82, 5.70, 0.82, size=14.5)
    c.text("Semantic ID / Map POI", 6.98, 1.98, 5.65, 0.40, size=19,
           bold=True, color=BLACK)
    c.line(6.98, 2.49, 12.63, 2.49, color=NAVY, width=1.1)
    for i, text in enumerate(right):
        c.text(text, 6.98, 2.70+i*0.82, 5.65, 0.82, size=14.5)
    c.note("本次 PPT 不展示此前各 SID / SFT 实验结果表；仅保留为方法动机所必需的定性发现。所有未运行内容均标为设计草案或待实现。", 6.17, 0.64)


def normalize_titles(deck) -> None:
    """Match the template: white content-page titles on its navy master band."""
    for slide_index, slide in enumerate(deck.prs.slides):
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            runs = [run for paragraph in shape.text_frame.paragraphs for run in paragraph.runs]
            if slide_index > 0 and shape.name.startswith("Title"):
                for run in runs:
                    run.font.color.rgb = RGBColor(255, 255, 255)
            elif any(run.font.size is not None and run.font.size.pt >= 20 for run in runs):
                for run in runs:
                    run.font.color.rgb = RGBColor(0, 0, 0)


def build_method_story(deck) -> None:
    cover(deck)
    overview(deck)
    beam_observation(deck)
    beam_related(deck)
    beam_framework(deck)
    beam_first_prune(deck)
    beam_states(deck)
    beam_math(deck)
    beam_revision(deck)
    beam_v2(deck)
    beam_algorithm(deck)
    qg_observation(deck)
    qg_related(deck)
    qg_framework(deck)
    qg_depth(deck)
    qg_reliability(deck)
    qg_adapter(deck)
    qg_dual_view(deck)
    qg_category_residual(deck)
    qg_geo(deck)
    qg_algorithm(deck)
    status_and_relation(deck)
    contribution(deck)
    references(deck)
    normalize_titles(deck)
