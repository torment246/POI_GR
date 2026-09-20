# QG-PRQK 实现状态模板

> 本模板采用 v2.1-CAT 阶段与术语。真实进度始终以同目录下 `QG_PRQK_IMPLEMENTATION_STATUS.md` 为准。

```yaml
PROJECT: QG-PRQK
STATUS: NOT_STARTED
CURRENT_PHASE: M1
LAST_COMPLETED_PHASE: NONE
NEXT_ACTION: 仅在 canonical 状态文档不存在时执行 M1，核对 v2.1-CAT 方法、配置与仓库真实资产；不实现算法。
METHOD_VERSION: v2.1-CAT
PRIMARY_CODEBOOK_SIZES: [512, 512, 512]
P2_FULL_QUERY_STATS: UNKNOWN
V2_1_CATEGORY_MIGRATION: NOT_STARTED
CATEGORY_AUDIT: UNKNOWN
COARSE_CATEGORY_BINDING: category_code[0:2]
FINE_CATEGORY_BINDING: category_code
P2_5_CAT_QUERY_DEPTH: NOT_STARTED
P3_EXACT_ADAPTER: UNKNOWN
P4_CAT_QUERY_GRAPH: NOT_STARTED
P5_CAT_POI_ONLY_PRQK: NOT_STARTED
P6_CAT_S1_S2: NOT_STARTED
P7_CAT_S3_GEO: NOT_STARTED
P8_CAT_STATIC_EVAL: NOT_STARTED
RUNTIME_CONFIG_VERSION: UNKNOWN
LAST_UPDATED: null
GIT_COMMIT_AT_UPDATE: null
WORKTREE_STATE: UNKNOWN
```

---

## 1. 冻结决策

- 范围：北京单城、精确原始 `poi_id` 检索。
- POI embedding：冻结复用现有北京 BGE-M3，首轮不重新生成、不加类别文本、不做 concat。
- 类别绑定：Fine Category=`category_code` 完整 6 位；Coarse Category=`category_code[0:2]`；`category` 仅作可读路径。
- Query 粒度：D3 Exact-Core 监督 S1/S2/S3；D2 Fine-Category 监督 S1/S2；D1 Coarse-Category 只监督 S1；D0 不参与 SID。
- SID：POI/Query 独立 residual view、双质心共享 token、Query–POI 层级图约束和 projection residual。
- 类别作用：S1 使用 coarse category 软代价，S2 使用 conditioned fine category 软代价；必须 warm-up，不硬映射 SID。sample 只验证工程闭环，方法判断与调参只依据 full。
- S3：D3 Query residual + GID6 局部 Geo + same-fine-category 困难边；不加类别分类损失。
- 禁止项：单向量融合、ESS per-POI gate、Family/Brand/Entity-Group、名称实体族启发式及不存在的 brand/canonical/category_l1/l2 输入。
- 主容量：`512 × 512 × 512`（2026-09-07 用户确认；BGE 向量仍为 1024 维）。
- Base PID：`[G1..G6,S1,S2,S3]`；只对碰撞项追加已有确定性 Dedup ID。
- 第一版停止点：P8-CAT 完整 SID 与静态验收后 `HOLD_FOR_REVIEW`；Final PID、Qwen SFT、外部基线、Order-B 和其他消融全部延后确认。
- 代码与输出隔离：全部 QG 项目代码、文档、配置、测试和新产物位于 `qg_prqk/`；根目录冻结资产只读。

---

## 2. REPO_MAP

待 M1 根据真实仓库填写。必须区分 QG 内运行入口、仓库内只读复制来源和冻结数据输入。

---

## 3. DATA_AND_ARTIFACTS

待 M1/P2.5-CAT 填写：

- 北京 POI embedding、POI 行映射与 hash：
- P2 `query_stats`、`targets`、false-negative 保护与 manifest：
- Fine category index/vocab、行序、覆盖率与 hash：
- Fine→Coarse 确定性映射：
- QG 输出 namespace：
- Train-only、Validation/Test 禁读约束：

---

## 4. ASSUMPTION_CONFLICTS

不得静默修改方法，至少持续核对：

- `REAL_POI_FIELDS_ONLY`：只使用真实字段；Fine/Coarse 按上述 `category_code` 绑定。
- `NO_FAMILY_HEURISTICS`：不得恢复 Family/Brand/Entity-Group 或名称启发式。
- `CATEGORY_IS_SOFT_STRUCTURE_ONLY`：类别不进入 BGE、不 concat、不硬设 SID。
- `NO_CANONICAL_MAPPING_ASSUMPTION`：没有真实 canonical duplicate/equivalent mapping 时不得写成依赖。
- `P3_REAL_STATE`：以 checkpoint/cache/log 实物判断完成度；代码存在不等于训练完成。
- `P8_HOLD`：P8-CAT 后先请用户审查，不自动生成 PID、启动 SFT 或跑外部基线。
- `QG_SELF_CONTAINED_CODE_AND_OUTPUTS`：QG 运行代码与输出都位于 `qg_prqk/`。

---

## 5. 阶段记录

### M1 — v2.1-CAT 文档、配置与真实进度迁移

- 状态：NOT_STARTED
- 修改文件：
- 审核结论：
- 事实修正：
- 验证：
- 阻塞项：

### P2.5-CAT — 粗/细类别 Query 粒度固化

- 状态：NOT_STARTED
- 输入：只读 P2 artifact、冻结 category mapping 与必要 POI metadata；不重读原始 Train，不读 Validation/Test。
- 输出：
- 阈值与敏感性：
- 测试：
- 阻塞项：

### P3A — Exact Adapter 真实 Gate

- 状态：NOT_STARTED
- 约束：只在 P3 真实状态为 `CODE_ONLY` 时保留；最多 50,000 条 D3，比较 Adapter 与 identity fallback。

### P4-CAT — D1/D2/D3 层级 Query 图与 embedding

- 状态：NOT_STARTED

### P5-CAT — POI-only PRQK 内部 A0

- 状态：NOT_STARTED

### P6-CAT — S1/S2 类别软约束双视图 PRQK

- 状态：NOT_STARTED

### P7-CAT — S3 D3 Query + Local Geo 困难图细化

- 状态：NOT_STARTED

### P8-CAT — 完整 SID 静态评测与 Prefix Probe

- 状态：NOT_STARTED
- 完成后：`HOLD_FOR_REVIEW`

### P8-CAT 后延后项

- 状态：DEFERRED
- 范围：Final PID/Dedup、Qwen SFT、外部基线、Order-B、组件与数据窗口消融。

---

## 6. 当前 NEXT_ACTION 详细说明

只描述一个可独立执行、验证和停止的阶段，并写清输入、输出、禁读数据、运行规模、验收标准与需要用户确认的 Gate。canonical 状态文档存在时不得用本模板覆盖其真实进度。
