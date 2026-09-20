# QG-PRQK 分阶段执行计划

## v3.0 当前执行导航（2026-09-19）

用户已确认 v3.0 方向，并要求复核后准备实施、复用现有代码、根据实验结果及时改方案。2026-09-19 已完成公式边界检查与实现映射，尚未新增 v3 生产代码或运行训练。算法和公式统一维护在 [v3.0 主规范](QG_PRQK_METHOD_SPEC.md)，本文负责代码切片和执行决策；实时完成度以[实施状态](../experiments/QG_PRQK_IMPLEMENTATION_STATUS.md)为准。

| 阶段 | 执行内容 | 当前状态 |
|---|---|---|
| V3-DOC | 新规范、v2.1 归档、公式复核与代码规划 | 已完成；10 项 CPU 合成数学检查通过，不代表实现或检索验收 |
| V3-DATA | 全量 Query/Pair、标注、路由、搜索分布字段契约及合成验证 | 下一最小开发步骤，尚未实现 |
| V3-GRAIN | Train 内标注、粒度分类头、校准与独立验收 | 未开始 |
| V3-PROFILE | 补齐全量 Raw BGE、冻结参照、构造 POI 搜索分布及证据量 | 未开始 |
| V3-QUANT | 同一 POI assignment 的 U/V、父组条件残差、SID 验收 | 未开始 |
| V3-SFT / V3-EVAL | Query 开/关配对的真实 SFT 与五组 Validation 双解码 | 未开始 |
| V3-GEO | 后续独立验证真实距离冲突项 | 延后 |

阶段是依赖关系和核验范围，不是要求一次实现全部功能。每次优先完成当前可独立检查的小闭环；按用户已确认的方向推进，不重复要求已获得的设计确认。长训练及全量任务按当次启动授权和资源安排执行，本次没有启动。现有 `qg_prqk_v2_1_*` 配置仍只代表旧算法，v3 不得直接继承其 Query 筛选、Adapter policy、自由 Query assignment 或源码签名。

### 先复用的代码与需要隔离的部分

| 现有位置（相对 `qg_prqk/src/qg_prqk/`） | 复用方式 | 需要处理的边界 |
|---|---|---|
| `data/query_statistics.py`、`query_normalization.py` | 直接读取已有 P2 Query/完整 Pair 与归一化版本 | 不为新方案重扫 758 万订单；不使用旧 `retained_for_sid` 筛选；源目录不匹配直接报错 |
| `data/query_embeddings.py` | 直接调用 `iter_normalized_queries`、`load_text_encoder` 等已有能力 | 迭代器已覆盖全部 Query；先盘点 cache 的真实行映射、空间和参数，再补缺失，不能仅因文件名相似复用 |
| `sid/base_quantizer.py` | 复用内容变换、确定性初始化/空码处理与 `projection_residual` | 空码重初始化同步同一行的 V；旧 `_soft_centroid_update` 仅按内容算权重，不直接复用为联合更新 |
| `sid/relational_quantizer.py` | 直接调用 `build_category_cost_lookup` | 不调用旧自由 Query assignment、Query 质心收缩或整套 `fit` |
| `sid/geo.py`、`sid/local_data.py` | 复用 GID、parent、局部 Geo 特征和困难边定义 | 重建各候选的新父组资产；旧 P6/P7 loader 不接收新 schema；最大 parent 的构图临时内存先检查 |
| `sid/local_quantizer.py` | 复用地理质心和困难边代价规则 | 旧 S3 内容 Top-32 与全码方案不同；旧惩罚函数有全 N×K 分配，改为分块适配 |
| `sid/identifiers.py`、`sft/a0_gid_data.py`、现有 SFT/评测模块 | 复用 Final ID、当前/历史 ID 替换、词表/缓存/预检、生成与指标 | `remap_record` 等含 A0 专属元数据；薄适配显式写 v3 来源，沿用非标识输入与评测业务键 |

暂不搬目录、统一所有配置类或重写 CLI。允许在 `qg_prqk` 内先调用已有内部函数；确有接口不符时增加小适配，尽量不改冻结 v2 的实现文件和源码签名。最终验证方法后，再合并重复代码、整理命名和公开接口。新的结果必须使用独立 schema/配置/目录，不能覆盖历史结果。

### 可独立交付的代码切片

以下文件名为拟定位置，按实际依赖逐步创建，不预建空文件或通用框架。

| 顺序 | 最小交付与拟定文件 | 输入 → 输出 | 本步通过条件 |
|---|---|---|---|
| 1：V3-DATA | `data/query_routing.py`；对应 `test_query_routing.py` | P2 字段定义、Query/POI 行映射、标注/概率样例 → 有校验的路由与 Pair 读取接口 | 完整正边保留；重复/缺失/错位 ID 报错；概率有限、gate 单调；质量与深度分开；不启动训练 |
| 2：V3-PROFILE 核心 | `data/search_profiles.py`；对应 `test_search_profiles.py` | 合成路由、稀疏参照分布、完整 Pair → POI `h/M/m/b` | 手算与流式聚合一致；先缩放总量再路由；多目标、零证据、低质量及大量微小深层 gate 均正确 |
| 3：V3-QUANT 核心 | `sid/search_quantizer.py`；对应 `test_search_quantizer.py` | 冻结内容/搜索视图、类别、父组 → U/V、唯一 POI assignment、残差与目标轨迹 | 同成员更新；零搜索权重回到同求解器对照；零条件残差中立；修改父层使下层缓存失效；先 hard-only |
| 4：V3-GRAIN 与真实 PROFILE | `adapters/granularity.py`，按需增加标签/训练命令；复用 Raw 编码 | Train 弱标签和真实人工集 → 冻结粒度头/校准器 → 全量路由、参照和 POI 分布 | Query 分组隔离；真实深层误路由/校准验收；全量行数和覆盖核对；原始与变换后的缓存不混用 |
| 5：全目录 SID | 小型独立 `sid/search_config.py` 与必要 CLI 适配，单份首轮实验 YAML | 验收过的输入和配置 → Query 开/关两套 SID | 合成、小目录 smoke、内存估算通过后执行 full；新配对同初始化、候选域、日程和停止规则 |
| 6：真实 SFT/评测 | 复用现有 SFT pipeline 和平台骨架，仅加 v3 来源适配 | SID → Final ID → 当前/历史映射 → 词表/预检/cache → Qwen SFT → 五组 Validation 双解码 | 非标识输入一致；1024 零截断；完整 POI 指标与逐请求配对结果；Test 仅用于最终确认 |

第 1 步是下一最小开发任务。第 2、3 步可以先使用合成路由验证，真实标签准备不阻塞量化核心；第 4 步标注规范也可提前准备。默认正式主候选仍采用学得的全量软路由。若真实标签成为主要等待项，可以明确命名“旧路由对照”来先检验搜索分布/共享分配，但要单独记录旧筛选范围和深度语义，不把它算作完整 v3 或新粒度模型验收。

粒度头先采用冻结 BGE 加小分类头，先证明标签定义和深层误路由可控；没有证据时不扩展成独立意图识别工程。参照点 M、稀疏 top-k 和损失初始尺度在第 2、3 步确定工程起点，不声称是最优值。先固定层间比例、检查一个整体搜索强度，不展开三层权重与全部模块的笛卡尔网格。

### 实验顺序与如何据结果改方案

首轮目标是尽快得到 `v3-query-off` 与 `v3-query` 的真实 SFT 配对。两边都保留类别/旧地理项、使用全 512 候选；不能拿新模型对旧 A0 的差异全部归给 Query。参照、路由与量化的静态检查用于发现实现问题，不能替代 Qwen 生成结果，也不要求所有静态指标先超过 A0 才进入 SFT。

| 实际观察 | 优先检查与下一次单因素改动 |
|---|---|
| 粒度模型把大量宽泛/上下文 Query 送入 S3 | 看独立人工集的混淆与校准；先修标签/校准或减弱深层路由，再决定是否升级模型 |
| S3 多数搜索残差变为零，或深层生成退步 | 报告单例父组/有效权重分布；固定其他项比较关闭父组中心化，不坚持原公式 |
| 搜索目标改善、Qwen S1 或完整 POI 指标下降 | 先查搜索与内容/类别项的尺度及分配变化，再单独减小搜索强度；不继续追求图一致率 |
| 热目标受益、冷目标受损 | 分层看有/无 Query 的内容失真和候选分配；先调整搜索强度或证据权重，不同时改粒度与地理 |
| SID 桶更容易命中，但完整 POI 没提升 | 查 GID/SID/Dedup 的条件正确率和碰撞负担；以完整 POI 指标判断，之后再隔离地理项 |
| 固定 10k 与泛化集方向不一致或差异很小 | 看配对增减命中、区间和切片样本量；必要时增加 seed/配对样本，不仅凭小数点差异宣称有效 |
| 同配置完整 SFT 配对没有收益 | 先定位覆盖、路由、表示、中心化、尺度中的一个因素，形成下一假设；不一次叠加所有新模块 |

每轮在同一方法实验文档记录真实配置、来源、指标、失败和下一假设；公式变更同步主规范，旧产物不追溯改名。新增一组对照前说明它回答哪个问题。常规修正沿用现有授权与任务范围；资源运行安排按当次任务落实，不为每个文档小修重复请求确认。

本次除 10 项一次性数学检查外，复用了下列现有轻量测试验证工程基础：全量 Query 迭代/模拟编码、类别代价、投影残差、历史与目标 ID 替换，共 10 项通过；`compileall` 与文档差异检查通过。它们不包含新量化器或真实模型训练。可在仓库根复现：

```bash
mkdir -p outputs/tmp/v3rv
PYTHONPATH=qg_prqk/src:qg_prqk/tests \
TMPDIR=/ofs/map_search/hudan/poi_genret/outputs/tmp/v3rv \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
/ofs/map_search/hudan/envs/poi-gr/bin/python -m unittest \
  test_query_embeddings.QueryEmbeddingsTest \
  test_relational_codebook.RelationalCodebookCategoryCostTest \
  test_base_codebook.BaseCodebookQuantizerTest.test_projection_residual_is_orthogonal \
  test_a0_gid_sft.A0GidSftTest.test_remaps_every_history_and_target_without_changing_context -v
```

## v2.1-CAT 历史执行计划原文

以下为原阶段状态机，算法解释对应 [v2.1-CAT 归档](QG_PRQK_V2_1_CAT_SPEC.md)。其中“当前实际进度”“尚未启动”及 HOLD 条款均指当时阶段，不能覆盖最新实施状态与用户授权。原文保留供复现已完成实验；v3.0 不沿用其阶段编号。

> 本文记录方法形成过程中的阶段状态机。当前源码已改为 `data / adapters / sid / sft` 领域结构，统一从 `qg_prqk/scripts/qg_prqk.py` 进入；P3A/P4/P5/P6/P7/P8 仅保留为历史协议编号，源码对应关系见 `qg_prqk/README.md`。

> 文档状态：canonical 执行计划 v2.1-CAT。  
> 当前实际进度：P3A-FULL、P4–P8 canonical 链路、NoGID 受控变体与三方比较均已完成。用户已批准两个 SFT 分支；P9 双 Final ID/全量 history10 Messages与 P10 共同扩词表/双 Train-Valid Tokenized Cache 已完成并验证，当前停在 `HOLD_FOR_QG_PRQK_DUAL_SFT_LAUNCH_REVIEW`。\
> 当前目标：只在 v2 双视图方法上加入全量 POI 粗/细类别软监督，完成第一版完整 SID。  
> 第一版终点：P8-CAT 后的 `HOLD_FOR_REVIEW` 已履行；P9 是用户另行批准的新阶段，不追溯修改 P0–P8 状态机。

---

## 1. Codex 工作协议

每次只执行状态文档 `NEXT_ACTION` 指定的一个阶段。固定流程：

1. 完整阅读仓库 `AGENTS.md`、`skills/poi-genret-workflow/SKILL.md`、canonical 方法文档、执行计划和状态文档；
2. 检查 `git status` 和用户未提交改动；
3. 核对输入 artifact manifest、schema、行数和 SHA256；
4. 实现最小闭环；
5. 单元测试；
6. sample/smoke；
7. 允许时再执行 full run；
8. 按实际运行情况更新状态文档、唯一方法实验文档和 artifact manifest；纯代码/文档整理不得伪装成实验；
9. 写唯一 `NEXT_ACTION`；
10. 停止。

禁止一次执行多个阶段、覆盖历史产物、静默改变方法、未测试直接全量运行或伪造未执行结果。

全部 QG 源码、脚本、配置、测试、方法级训练/评测入口、文档和新产物必须分别位于 `qg_prqk/` 的对应目录。复用仓库项目代码时先复制到 `qg_prqk/` 并登记来源；运行时不得导入 `src/poi_gr` 或调用仓库其他方法脚本。仓库根的冻结 BGE、数据和模型仅只读引用。不得自动 commit/push，不得修改或清理用户已有改动。

---

## 2. 状态迁移规则

文档迁移会话必须根据实际状态将进度映射为：

```yaml
METHOD_VERSION: v2.1-CAT
P2_FULL_QUERY_STATS: COMPLETED
P2_5_CAT_QUERY_DEPTH: COMPLETED
P3_EXACT_ADAPTER: CODE_ONLY | GATE_COMPLETED | FULL_IN_PROGRESS | FULL_COMPLETED | BLOCKED
P4_CAT_QUERY_GRAPH: NOT_STARTED | SAMPLE_COMPLETED_AND_VALIDATED | MEDIUM_COMPLETED_AND_VALIDATED | FULL_COMPLETED_AND_VALIDATED
P5_CAT_POI_ONLY_PRQK: NOT_STARTED | SAMPLE_COMPLETED_AND_VALIDATED | MEDIUM_100K_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED | MEDIUM_100K_DIAGNOSTIC_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED | MEDIUM_100K_PARAMETER_DIAGNOSTIC_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED | HARD60_TOPK5_SAMPLE_READY | HARD60_TOPK5_SAMPLE_COMPLETED_AND_VALIDATED | MEDIUM_100K_COMPLETED_AND_VALIDATED | MEDIUM_500K_COMPLETED_AND_VALIDATED | FULL_COMPLETED_AND_VALIDATED
P6_CAT_S1_S2: NOT_STARTED | INPUT_CONTRACT_COMPLETED_SAMPLE_NOT_STARTED | SAMPLE_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED | SAMPLE_ATTRIBUTION_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED | FULL_READY | FULL_IN_PROGRESS | FULL_COMPLETED_AND_VALIDATED | FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED | FULL_COMPLETED_AND_VALIDATED_S2_DIAGNOSTIC_REVIEW_REQUIRED
P7_CAT_S3_GEO: NOT_STARTED | SAMPLE_COMPLETED_AND_VALIDATED | FULL_COMPLETED_AND_VALIDATED
P8_CAT_STATIC_EVAL: NOT_STARTED | FULL_COMPLETED_AND_VALIDATED | FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED
```

P2 全量指标必须原样保留：

```text
Train rows: 7,586,410
normalized unique Query: 1,372,661
unique Query–POI pair: 2,022,443
D3 exact-core Query: 291,590
D3 exact-core covered POI: 153,349
false-negative pair: 557,639
all Query-covered POI: 491,213
```

当前执行值为 `P3_EXACT_ADAPTER: FULL_COMPLETED`、`P4_CAT_QUERY_GRAPH: FULL_COMPLETED_AND_VALIDATED`、`P5_CAT_POI_ONLY_PRQK: FULL_COMPLETED_AND_VALIDATED`、`P6_CAT_S1_S2: FULL_COMPLETED_AND_VALIDATED`、`P7_CAT_S3_GEO: FULL_COMPLETED_AND_VALIDATED`、`P8_CAT_STATIC_EVAL: FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED`。P8 full 已发布 A4 SID 候选并通过独立 validator；其 `HOLD_FOR_REVIEW` 停止点已履行，用户随后另行批准 P9 双分支 SFT 数据准备。

2026-09-11 追加的 NoGID 受控分支也已完成：`P7_NOGID_S1S2_PARENT_S3: FULL_COMPLETED_AND_VALIDATED`、`P8_NOGID_S1S2_PARENT_COMPARISON: FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED`。三方统一 `(S1,S2)` S3 候选后，NoGID 在 distinct SID、max bucket、S3/cumulative Prefix Probe 上均略差于原 A4，因此不替换 canonical 分支；其 `HOLD_FOR_NOGID_S3_REVIEW` 随后由用户以“两个分支都构造 SFT 数据”履行。

2026-09-06 子状态：完整 D3 Query cache 已复用，SELECT 3 epoch 后选中 epoch 2，FINAL 从共同初始 state 完成两轮全量重训，`--validate-only` 通过。初始化沿用用户在 2026-09-05 确认的“同结构/同 identity、建模前 seed=42 新建 state、SELECT/FINAL 逐值共用”。历史 Gate 未保存初始 state 且建模后才设置 seed，不声称新模型与历史 Gate 初始权重逐值相同。完整结果只在实施状态文档的 `EXP-20260905-02` 维护。

P3A-FULL 当时的独立执行协议要求：

```text
完成 active POI P3A-FULL 后设置 HOLD_FOR_P3A_FULL_REVIEW；不得自动进入 P4-CAT。
```

上述停止点均已履行。2026-09-07 用户确认继续原 v2.1-CAT 完整方法并将容量改为 `[512,512,512]`，不是改成 Query-only 或普通 additive RQK。配置迁移和 P4-CAT sample/50k/full 均已完成，随后用户确认启动 SID 构建；P5 10k sample 通过后完成确定性嵌套的 100k medium。100k 三层 hard fit 均达到 30 轮上限且 Top-k refinement 逐轮恶化 hard distortion；第一轮独立诊断证实 hard60-off 在 43/39/33 轮收敛，但 Top-k 存在误差–均衡权衡。2026-09-08 的配对诊断进一步证实 Top-k5 牺牲 distortion、改善码均衡和路径唯一率，用户据此确认 `max_iter=60 + Top-k5/beta=15/5轮`。新协议 10k 通过后，用户于 2026-09-09 明确覆盖原 Gate 顺序，要求跳过新协议 100k/500k 直接构建 full；实现以显式开关和 sample manifest 哈希记录该单次授权，默认 Gate 保护未放宽。716,245 POI full 已完成并独立验收；此后 P6 sample、同闭包 P5 A0 对照和六分支归因也已完成，当前状态见 P6 章节。

active POI 数据修订不改变算法，仅替换后续阶段的冻结候选宇宙：

```text
catalog rows: 716,245
catalog: data/beijing_poi_active_order14d_history10_20260715_json/
BGE/category rows: qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/
frozen P2.5/P3 namespace: qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/
P4-P8 namespace: qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/
```

后续配置入口为 `qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml`，加载器为 `qg_prqk/src/qg_prqk/sid/pipeline_config.py`。它锁定冻结上游配置哈希、P2.5/D3 cache/FULL/FINAL 来源与 D1/D2 Raw BGE、D3 FINAL policy；只改变下游容量和输出目录。上游文件名中的 `1024x3` 不改名，BGE 1024 维不变，不重新编码或训练 Adapter。当前只读核验命令为 `qg_prqk/scripts/qg_prqk.py inspect-sid-config`；正式 Query 图构建仍须另行验收实际输入产物。

该 catalog 的成员由 Train/Validation/Test 当前目标和 history 中出现过的 POI ID 构成，只用于保证冻结候选库可回表；训练、holdout 划分和 checkpoint 选择仍只读取 Train-derived D3/P2 数据，不读取业务 Validation/Test Query 或标签。

---

## 3. v2.1-CAT 阶段总览

| Phase | 名称 | 目标 | 是否允许全量 |
|---|---|---|---:|
| M1 | v2.1-CAT 文档迁移 | 更新 canonical 方法、计划、状态和配置 | 否 |
| P2.5-CAT | 类别 Query 粒度标注 | active POI 上重建 D3 Exact、D2 细类、D1 粗类、D0 | 是，现有统计后处理 |
| P3A | Exact Adapter | 旧 50k Gate 保留；active 闭集执行 FULL-SELECT/FULL-FINAL | P3A-FULL 已获确认 |
| P4-CAT | Query 图与 embedding | 构建 D1/D2/D3 节点、边、layer mask 和缓存 | sample/medium 验收、另行确认后 full |
| P5-CAT | POI-only PRQK 512×3 | 建立主方法内部 A0 与初始化 codebook | sample/medium 后 full |
| P6-CAT | 类别+Query S1/S2 | 粗/细类别软锚定、双质心、逐层 residual | sample smoke 后直接 full；只按 full 调参 |
| P7-CAT | D3 Query + Geo S3 | 精确实体层与局部地图消歧 | sample smoke 后直接 full；只按 full 调参 |
| P8-CAT | 静态评测与 SID 候选 | 比较 A0/A4，写完整指标与 manifest | full eval |

P8-CAT 后必须停止。Final PID、Qwen SFT、历史外部基线、Order-B 和其他消融不属于本轮自动执行链。

---

# M1：v2.1-CAT 文档迁移

## 目标

只修改文档和状态；不执行数据处理、embedding、聚类或训练。

## 必须完成

1. 阅读 v2.1-CAT 更新包、v2 canonical 文档和真实状态；
2. 更新 canonical：
   - `qg_prqk/docs/methods/QG_PRQK_METHOD_SPEC.md`
   - `qg_prqk/docs/methods/QG_PRQK_CODEX_EXECUTION_PLAN.md`
   - `qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md`
3. 保留原 P0/P1/P2/P3 历史记录；
4. 将 D1/D2 更新为 Coarse/Fine Category，删除 Family 依赖；
5. 绑定完整 `category_code` 和前 2 位粗类，登记已有类别 mapping；
6. 加入 S1/S2 类别软代价与 warm-up；
7. 根据真实产物保持 P3=`CODE_ONLY`，保留 P3A；
8. 将 `NEXT_ACTION` 设置为 P2.5-CAT；
9. 不改算法代码、不运行任何任务。

## 验收

- canonical 方法版本为 v2.1-CAT；
- P2 全量结果未丢失；
- P3 状态与实际一致；
- diff 仅含预期文档和 v2.1-CAT 配置；
- M1 完成时唯一下一步为 P2.5-CAT；该阶段现已完成。

M1 只记录为方案迁移，不创建实验编号，不修改 Python 算法代码或任何历史 artifact。

---

# P2.5-CAT：类别 Query 粒度标注

## 输入

- 已完成的 `qg_prqk/outputs/qg_prqk_1024x3_v1_1/query_stats/` 及 manifest；
- active POI 行序的 716,245 行 `category_indices.npy` 和共享的 402 类 category vocab；
- active BGE `poi_ids.jsonl` 行映射与必要的北京 active POI `category/category_code` metadata。

类别绑定固定为：完整 6 位 `category_code` 是 `fine_category_id`，前 2 位是 `coarse_category_id`，`category` 只作可读路径；现有 vocab 为 402 个细类、19 个粗类。当前没有 brand/family/chain/parent/canonical 字段，本版不构造或推断它们。

## 禁止

- 不重新读取原始 7,586,410 行 Train；
- 不读取 Valid/Test Query、订单标签或统计量；active POI 候选成员资格是已冻结上游输入，不属于 Query 监督；
- 不重新运行 P2；
- 不训练 Adapter；
- 不修改已有 D3 exact-core 定义。
- 不使用名称启发式构造类别或实体族；
- 不把类别加入 BGE 文本或 embedding。

## 实现步骤

1. 核对 P2 manifest/schema/hash，不改写任何 P2 文件；
2. 校验 active 类别 mapping 的 716,245 行、active BGE 行序、SHA256、全量非空、6 位数字格式和 `fine -> coarse` 唯一映射；
3. 将既有 `n(q,p)` 聚合到 coarse/fine category，计算 `C_coarse/H_coarse` 与 `C_fine/H_fine`；
4. 将现有 `is_high_confidence=true` Query 原样固定为 D3；
5. 对剩余 Query 先按 fine category 判 D2，再按 coarse category 判 D1，其余为 D0；
6. D2 只保留主导 fine category 内的真实发单边，D1 只保留主导 coarse category 内的真实发单边；
7. 计算 reliability、layer mask 和按 Query 归一的 edge weight；
8. 输出小范围阈值敏感性和 D0/D1/D2/D3 人工样例；
9. 写完整 manifest，并证明未读取原始 Train/Valid/Test。

## 产物

```text
qg_prqk/outputs/<v2-run>/query_depth/
  category_mapping.parquet
  query_category_depth.parquet
  query_category_stats.parquet
  query_poi_layer_edges.parquet
  p2_5_cat_metrics.json
  p2_5_cat_report.md
  manifest.json
```

## 必须报告

- D0/D1/D2/D3 Query 数与占比；
- 各层覆盖 POI；
- coarse/fine category 数量、缺失率和一致性；
- Query count、concentration 与 entropy 分布；
- 每个 Query 保留边权的归一化检查；
- 20 条 D1、20 条 D2、20 条 D3、20 条 D0；
- 阈值小范围敏感性；
- 未读取 Valid/Test 的 manifest 证据；
- 不得只报告通过测试。

P2.5-CAT 完成当时的唯一下一步为 P3A，不得跳过当时 `CODE_ONLY` 的真实 Adapter Gate；该 Gate 后续已于 2026-09-03 完成。

## 历史全库完成结果（2026-09-02，只读保留）

- 全量 `category_mapping.parquet` 为 2,337,178 行，402 fine / 19 coarse，缺失、非法 code、BGE 行序错位和 fine→coarse 冲突均为 0；
- 1,372,661 个 Query 的 D0/D1/D2/D3 数分别为 1,029,782 / 25,639 / 25,650 / 291,590，D3 与冻结 P2 完全一致；
- S1/S2/S3 覆盖 POI 分别为 210,965 / 188,570 / 153,349，层级边总计 1,204,570 行；
- Query–层概率和与边权和最大误差分别为 `5.7732e-15` 与 `5.5511e-15`；
- sample 1,000、medium 50,000、full 和独立 `--validate-only` 均通过；full 耗时 261.08 秒、峰值 RSS 471.57 MiB；
- manifest SHA256 为 `61cd7cf8e4bc783fe81bab9ce4ee87fcf4bab3274f25a036963931dbb811027a`，输出位于 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat/query_depth/`。

该结果基于旧 2,337,178 条候选库，不能作为 active P3A-FULL 的 P2.5 输入。active 版本必须写入 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_depth/`，不得覆盖上述历史产物。

## active 数据修订完成结果（2026-09-04）

- 716,245 条 active POI 的 BGE/category 精确行子集已发布到 `qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/` 并通过独立逐值复核；
- 全局 402 个 fine ID 保持稳定，active 实际出现 397 个 fine、全部 19 个 coarse；5 个零支持 fine 只在词表中保留，不补 POI、不重编号；
- active P2.5 输出 716,245 行 category mapping、1,372,661 个 Query 和 1,204,570 条层级边；D0/D1/D2/D3 仍为 1,029,782/25,639/25,650/291,590；
- active P2.5 manifest SHA256=`5f203b94d567433f89ff25a7755516b211544b9be2c484ac28e2243ccd7ce55f`，独立 `--validate-only` 通过；后续 P3/P4 只绑定该 manifest。

---

# P3A：Exact Query Adapter 审计与补全

## 决策逻辑

1. P3A 从 `CODE_ONLY` 起步，先确定性抽取最多 50,000 条 D3 Query 跑真实 Gate；该 Gate 已于 2026-09-03 完成并通过；
2. 旧 Gate 的代码、配置、checkpoint、manifest、日志和产物全部保留，不覆盖；它只作为 active holdout 上的对比模型；
3. 从未进入 Gate 的 D3 中按 seed 42 和 Query ID BLAKE2b 顺序确定性冻结 20,000 条 Train-only internal holdout，不读取业务 Validation/Test；
4. FULL-SELECT 按 2026-09-05 用户确认的同结构 identity、建模前 seed=42 新初始状态重新训练，使用其余 271,590 条 D3，固定 3 epoch；不声称逐值复现历史 Gate 初始权重。每个 epoch 在同一 holdout 和 716,245 条 active POI 上执行精确检索；
5. 主 checkpoint 按 Recall@10 选择，Recall@1、困难子集 Recall@10 依次 tie-break；同时报告 Recall@1/5/10/20、MRR@10 与 Query embedding cosine drift；
6. 同一 holdout 统一比较 Raw BGE Query、历史 50k Gate Adapter 和 FULL-SELECT Adapter；
7. 选出 best epoch 后，再从同一 identity 初始状态使用全部 291,590 条 D3 固定 epoch 重训 FULL-FINAL，不再选模；
8. 完成后设置 `HOLD_FOR_P3A_FULL_REVIEW`；若 Adapter 无提升，记录真实结果并等待评审，不自行进入 P4-CAT。

## 第一版不做

- 不将 D1/D2 当作单 POI exact positive；
- 不立即训练 hierarchical multi-positive Adapter；
- 不修改 POI embedding。

## 输出

```text
qg_prqk/outputs/<v2-run>/query_adapter_exact/
  query_adapter_exact.pt
  adapted_query_embeddings_d3.*
  raw_vs_adapted_eval.*
  manifest.json
```

历史 Gate 与 active P3A-FULL 已分别按当时协议停止并完成用户确认；2026-09-07 用户确认继续完整方法后，已完成 P4-CAT sample/medium 并等待审核，不能跳过 full 图验收直接聚类。

用户现已选择 full 路径。active FULL 输出固定写入独立 namespace：

```text
qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_adapter_exact_full/
  internal_holdout/
  comparison/
  select/
  final/query_adapter_exact_final.pt
  manifest.json
```

P3A-FULL 第一版 Query view policy 固定为 D1=Raw BGE、D2=Raw BGE、D3=Final Adapted Query；本阶段不验证或修改 D1/D2 表示。

## 历史 50k Gate 完成结果（2026-09-03，只读保留）

- `EXP-20260903-01` 使用 BLAKE2b Query ID hash 从 291,590 条 D3 确定性选择 50,000 条，稳定切分 47,500/2,500 Train/internal-dev，未读取正式 Validation/Test；
- POI 2,337,178×1,024 使用单卡 FAISS GPU `IndexFlatIP` 精确 Top-100；POI add 与 Query search 前都执行 float32 L2 归一化；
- 每 Query 固定 6/4/6 个 semantic/lexical/local-geo 负例，外加动态 in-batch，P2 false-negative mask 生效；
- overall Recall@10 `66.00%→71.24%`、margin `-0.032193→-0.021226`；困难子集 Recall@10 `45.51%→54.10%`、margin `-0.078656→-0.072205`；
- 严格 Gate 通过，50k 产物选择 Adapter view；正式输出和独立 validator 均完成，manifest SHA256=`12a29f1ff90d16f619231da18b99916da0afc7a2dfa4d138a18b9bb4bac96c2f`；
- 该 Gate 的候选库仍是旧 2,337,178 条全库；其历史指标不与 active FULL 指标混写。active P3A-FULL 已完成，后续授权状态以文档顶部和 canonical 状态文档为准。

---

# P4-CAT：类别层级 Query 图与 Query embedding

## 目标

将 D1/D2/D3 Query 组织为独立 Query 节点，而不是融合进 POI embedding。

## 实现

1. 按 query_id/文本/编码配置验证并复用已完成的 D3 Raw cache；D1/D2 缺失部分编码，D3 仅应用 FINAL Adapter，不重跑已完成 BGE 编码；
2. 严格按 P3A-FULL policy 使用 D1/D2 Raw BGE 与 D3 Final Adapter；50k Gate checkpoint 不作为 P4 默认 Query view；
3. 构建 Query 节点索引；
4. 构建 Query–POI 边和 layer mask；
5. 每 Query 每层保留边的条件概率之和为 1，最终边权之和等于该层 reliability；复用 P2.5 定义，不再次归一而抹去可靠性；
6. 对超级头部 Query 做 capped support；
7. 保留 false-negative mask 供 Adapter/诊断；
8. 支持分片、resume、SHA256 和 sample limit。

## 当前已完成的小闭环（2026-09-07）

`scripts/build_query_graph.py` 与 `p4_data.py/p4_query_graph.py/p4_cli.py` 已完成 v1 1,000-query sample、v2 50,000-query medium 与 v3 342,879-query full。sample/medium 原样保留，medium 复用 sample，full 则以冻结 manifest SHA256 严格复用 medium 前 50,000 行。19 项 P4 合成测试、当时全部 100 项回归通过；三档真实构建、独立 validator 与完成 resume 均通过。该历史停止点已履行，P5 active POI full 也已完成并独立验收。

当前 CLI 通过 `--gate` 明确区分 sample≤1000、medium≤50000 与固定 full=342879。medium/full 使用 8192 行缓存块、256 计算 batch；full 通过 D3 mmap 分块读取、Arrow 分块来源签名和 Arrow 聚合控制峰值。构建/恢复 full 必须显式绑定 medium manifest 路径与冻结 SHA256；独立 validator 从完成 manifest 读取前缀来源。v1/v2 复核继续使用各自原快照，不改旧源码合同；v3 独立输出 `query_graph/full_342879/`。物理合同见方法规范 §6.5–6.7，真实指标、环境、命令、来源哈希和局限在实施状态文档维护。

## 不再将“每 POI 最多 4 个 Query prototype”作为主数据结构

如全量 Query 节点过大，可将 prototype compression 作为可配置工程压缩，但必须有“不压缩”小样本对照，且不能改变 Query 总质量归一原则。

---

# P5-CAT：POI-only PRQ-KMeans 512×3

## 目标

建立完全不含 Query/Category/Geo 的 POI-only `A0`，作为 v2.1-CAT 主方法的内部正确性对照和双视图初始化；本阶段不是历史外部基线实验。

## 实现

- 当前 BGE POI embedding；
- 主方法固定 global-direction removal，并对 POI/Query 应用同一 POI-fitted transform；
- spherical/cosine assignment；
- projection residual；
- 512×3；
- 旧 Gate 每层 min 8 / max 30 iter；新 P5 试运行协议为 min 8 / max 60 iter；
- objective 与 assignment change 双停止；
- Top-k refinement 受控开关；
- 输出 `U1/U2/U3`、POI assignments、residuals 和 manifest。

## Gate

先 sample，再 100k/500k，Gate 通过后才允许全量。只检查实现退化性质、目标收敛、利用率、residual energy 和资源；历史普通 BGE-RQK 指标最多作只读 sanity reference，不重跑、不做同协议 SFT、不形成正式公平比较。

## 当前状态（2026-09-09）

- 10k sample 已完成并通过独立 validator；三层 code 利用率均为 100%，hard fit 分别在 16/15/16 轮收敛，零 residual 行为 0。
- sample distinct SID=9,934/10,000（99.34%），collision excess=66、最大桶=4；这些结果只用于 sample 结构 Gate。
- 10k 成功 manifest SHA256=`c9d18df7d1029c9b80442965f31943a31cf4c3ef076de2aa6b9fd5b6d437f4e2`；其历史停止点已履行。
- 100k medium 成功构建并独立验收，manifest SHA256=`8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0`；三层 code 利用率均为 100%，distinct SID=96.24%，但三层 hard fit 都达到 `max_iter=30` 且固定 Top-k refinement 逐轮恶化 hard distortion。
- 不覆盖的 100k hard endpoint 受控诊断已完成：hard30-off 仍全部未收敛，hard60-off 在 43/39/33 轮收敛；原 Top-k5 使 distortion 更差，但 distinct SID 为 96.240%，高于 hard30/60-off 的 95.786%/95.722%。
- 不覆盖的 hard60+Top-k5 配对诊断已完成：每层从同一次 hard endpoint 原位分叉，Top-k5 的 distortion 增量为 `0.007237/0.004646/0.003592`，但 Kish ESS 三层均提高、Gini 三层均降低；完整路径 distinct SID=`96,171`，相对 hard60-off 增加 449。
- 用户确认的 `max_iter=60 + Top-k5/beta=15/5轮` 新 10k 已构建并独立验收：三层在 16/15/16 轮收敛、512 code 全激活，manifest SHA256=`29a918fb5d3b3727d393ad31dc69e6f95f61fcd49a3036af8d1981473784f4d7`。
- 2026-09-09 用户明确授权跳过新协议 100k/500k，由上述 sample 直接构建全部 716,245 active POI；full manifest 显式绑定该 sample 与授权范围。三层在 57/52/57 轮收敛、512 code 全激活，distinct SID=`606,045`（84.6142%）、collision excess=`110,200`、最大桶=208，独立 validator 退出 0。manifest SHA256=`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`。该停止点随后已履行：用户确认 P6/P7 hard 上限为 60；真实 P6 仍等待 sample/medium 规模确认。

---

# P6-CAT：类别锚定的 Query 粒度感知双视图 S1/S2

## 目标

实现 POI/Query 双质心、共享 token index、层级 graph alignment、逐层 Query residual，以及 S1 coarse/S2 fine 类别软代价。

## S1

使用 D1/D2/D3 Query，并加入 `P(coarse_category|S1)` 的 Dirichlet 平滑软代价；`alpha=32`，类别启动权重 `0.08`。

## S2

仅使用 D2/D3 Query；Query 输入必须是 S1 residual。类别项必须使用 `P(fine_category|S1,S2)` 条件路径分布；`alpha=16`，类别启动权重 `0.08`。

## 迭代

每轮：

1. POI assignment；
2. Query assignment；
3. 更新 POI centroids `U_l`；
4. 更新 Query centroids `V_l`；
5. Query centroid shrinkage；
6. 更新 coarse/fine category 分布；
7. 记录 objective、POI/Query assignment change、alignment disagreement 和 category CE；
8. 收敛后分别计算 POI/Query projection residual。

## Warm-up 与受控配置

```text
iteration 1-2: category weight=0，Query/graph 为目标值的 50%
iteration 3-6: 线性 ramp 到目标值
iteration 7+: 目标值
```

`0.05/0.10` Query/graph 与 `0.08` 类别权重是第一版 full 的冻结起点。sample 只检查实现、数据流、显存和产物合同，不再以 sample 指标决定参数是否可进入 full。

hard alternating 上限由用户于 2026-09-09 确认为 60；Top-k5/β=15/5 轮、停止阈值、容量、权重和 warm-up 不变。实现必须使用独立 P6/P7 overlay，禁止原地修改旧 30 轮 downstream YAML。P6 Gate 必须构造 Query 图闭包：先选基础 POI 与 Query，再保留这些 Query 的全部 S1/S2 边，并将所有边目标 POI 并入 POI 集，不能丢边后重归一。

历史 canonical-v1 保留既有确定性 Gate：sample=P5 10k 基础 POI + P4 1k Query 前缀（闭包后 11,459 POI），medium=P5 100k 基础 POI + P4 50k Query 前缀（闭包后 148,860 POI）。sample 已完成并验证；medium 从未启动，也不删除。

用户随后批准只在相同 sample 上做归因，不改 canonical。六分支结果显示：只关闭 graph 后 distortion 为 `0.5889/0.5822`；只把 Query 质心 POI 先验由 `tau=32` 降到近零后为 `0.2624/0.2309`，但 Query active code 增至 `492/500`。canonical 中占用码的 Query 支持中位数仅为 2，Query-weighted POI prior mass 达 `85.20%/90.85%`；因此 1k Gate 的主要来源是稀疏支持下的固定 prior，graph 次之，Top-k5 边际影响较小，category 基本无关。近零 prior 是过拟合式因果探针而非候选参数。

2026-09-09 用户进一步确认统一规模策略：小规模只负责把代码、数据流和产物合同跑通；通过后直接跑全量，方法指标和参数修改只根据全量结果判断。新 `hard60_topk5_direct_full_v2` 合同绑定已完成 sample manifest，显式记录跳过 medium；full 使用全部 716,245 active POI、342,879 个 P4 Query 和全部 912,980 条 S1/S2 边，算法权重、`tau=32`、warm-up、Top-k5、容量、seed 和停止条件均不变。2026-09-11 用户接受 P6 S2 诊断作为留存，并将该规模策略明确授权到 P7；P7 sample smoke 与 direct full 现均已完成。

2026-09-10 用户确认先诊断 S2。实现冻结正式 P6-S1 endpoint，只在 317,240 个 S2 Query、398,480 条完整 S2 边和全部 active POI 上运行六个 S2 分支。四点分解表明上游 S1 residual 不是退化源；canonical 相对 graph-off 增加 Query distortion `0.110462`、换取图一致率 `+51.7950pp`，graph-off 的 `28.4846%` 又低于 P5 A0 的 `33.9990%`，所以不得直接关闭 graph。`tau→1e-6` 只改善 `0.002110`；Top-k5 改善 distortion `0.011019` 并增加 1,794 个 distinct prefix；category 只增加 distortion `0.000403` 并提高 fine path purity `1.4316pp`。下一步若调整，只应在用户确认后对 S2 graph weight 做 full 多目标权衡，不修改其他因素。

## 验收

- sample 只要求代码、完整边闭包、Query depth mask、双视图 residual、512 code 合法范围、产物与独立 validator 跑通；sample 指标不再作为 full 准入或调参依据；
- full 统一报告 Query residual distortion、S1/S2 图一致率、coarse/fine 类别结构、code 利用率、unique/bucket 结构和无 Query POI 分桶；
- 参数是否保留或修改只根据 full 的同口径 P5 A0 对照决定；
- full 完成后的 S2 诊断已履行并由用户接受为留存；正式 P6 不修改。P7 已在后续明确授权下完成，不把诊断分支写回 canonical P6。

---

# P7-CAT：D3 Query + Geo-aware S3

## 目标

S3 使用 D3 Query residual + POI residual + GID6 局部 Geo + same-fine-category local hard graph。

## 关键约束

- D1/D2 Query 不参与 S3；
- S3 Query 必须使用 S1/S2 后 residual；
- Geo 不进入 S1/S2；
- S3 不加入类别分类代价，fine category 只筛选困难边；
- hard edge 只在固定 `(GID6,S1,S2)` parent 内要求 same fine category 和较高 BGE/name/address/geo 复合相似度；因此第一版是同 GID6，不向相邻 cell 扩展；
- 当前没有 canonical duplicate mapping，只排除 self-loop 与 P2 false-negative protected targets；
- GID 不是 residual codebook；
- 第一版 parent key 为 `(GID6,S1,S2)`；
- 同时生成 GID4/5/6 桶统计供后续消融。

## Gate

- `Acc(S3 | GID,S1,S2)`；
- local hard collision；
- semantic distortion；
- exact/bucket uniqueness；
- 同商场/医院/车站子 POI 子集。

## 当前状态（2026-09-11）

- sample smoke 已完成：11,459 POI、832 D3 Query、376 条困难边；首次 validator schema 失败和修正后成功产物均保留，成功 manifest SHA256=`5a18914d4236476dc2c0f037afa2699897938416faaa099d72a873b40419ea68`。
- full 已完成并由冻结源码独立复核：716,245 POI、291,590 D3 Query、746,034 条困难边，hard alternating 第 40 轮收敛，POI/Query 512 个 S3 code 全激活。
- full 加权条件 S3 一致率=`85.2967%`，困难实体加权碰撞率=`17.5233%`；纯 SID distinct=`617,887`（86.2675%）、最大桶=95，`GID6+SID` distinct ratio=`93.8637%`、最大桶=56。
- full manifest SHA256=`dd395ab0d11321106455ad05b18503af9b7d3dda7cb7fd85e531655cece2133c`；构建和独立 validator 均退出 0。该停止点已履行，用户随后确认进入 P8。

---

# P8-CAT：静态评测与 SID 发布候选

## 第一版只比较

```text
A0: POI-only PRQK 512×3 initialization
A4: Query + Category + Geo full
```

必须报告 codebook/路径、coarse/fine category、D1/D2/D3 Query 对齐、Prefix Probe、GID6 局部碰撞、Distinct SID 和桶分布。A0 是内部初始化，不是历史 BGE-RQK 外部基线。无论结果如何，结束后都设置：

```text
NEXT_ACTION: HOLD_FOR_REVIEW
```

不得自动进入 Final PID、Qwen SFT 或其他消融。

## 当前结果（2026-09-11）

- sample/full 均完成并由冻结源码独立复核；canonical full manifest SHA256=`c66ea6c5fac499ed7c377b9f2ae7d73c3fde333c7842f0d79e9c84b141c82ab3`。
- A4 相对 A0：distinct SID `+11,842`，最大桶 `208→95`，coarse/fine purity 提高 `10.4784/6.0707pp`，同 GID6 pairwise separation 提高 `0.0369pp`。
- content-only Prefix Probe 的 weighted S1/S2/S3 为 A0 `29.9614%/39.2334%/90.1722%`、A4 `21.0191%/36.2871%/89.7239%`；cumulative S1–S3 为 `12.7183%/7.7743%`。结构改善而 Query-only 可预测性下降，结论为 `REVIEW_REQUIRED`。
- 当前唯一状态为 `HOLD_FOR_REVIEW`。不得自动修改 graph/query/category/Geo 权重，也不得启动本节后的任何延后项。

---

# P8-CAT 后的延后项

## 追加受控分支：NoGID `(S1,S2)` parent S3（已完成）

- P7 NoGID 冻结 P6 S1/S2，不读/不算/不输出 GID，输出三位 `[S1,S2,S3]`；full manifest SHA256=`a6ce4ba8c20f79073e5a41fa2e516b26b2bdeb7c2d7c56e260f4a94f46bf1d6c`。
- P8 将 A0/原 A4/NoGID A4 全部统一到 `(S1,S2)` S3 候选；NoGID 相对原 A4 distinct `-4,689`、max bucket `+64`、S3 probe `-0.4403pp`、cumulative `-0.0572pp`，manifest SHA256=`b22ab413558b1e147542135fd554fdf0b7a71082bdd1842fc787aa2b16ed4ed8`。
- 该分支结论为 `REVIEW_REQUIRED`，只是已完成的受控变体，未改写 canonical v2.1-CAT。该停止点随后已履行：用户要求两个分支都进入 SFT 数据准备，而不是用 NoGID 替换 canonical A4。

## P9：双 Final ID 与 history10 SFT 数据（已完成）

- `P9-FINAL-ID`：从冻结的原 A4 GID-parent SID 和 NoGID SID 分别构造 `GID6+SID3+[D]` 与 `SID3+[D]`。Dedup 按完整 Base ID 桶和 `poi_id` 字典序确定性分配，单例省略 D，两者最终均全局唯一。
- `P9-SFT-DATA`：一次扫描冻结的 8,790,513 条 history10 源语料，成对写出两个分支；只替换 history/target POI identifier，保留 Query、请求 GID、用户、时间切分和顺序。Train/Valid/Test=`7,586,410/597,421/606,682`。
- 两分支共用 3,743 个普通原子 Token 的有序清单；GID-parent/NoGID SFT manifest SHA256 分别为 `c05810df785952b5e4412d4e67d31324587e2e02fb940ebb005331d9ea09e114` / `b173fe1411b6043727bd138a88b93bb9456b2bff89522d1c46b75e6304d21e65`。
- 工程 smoke、全量源哈希、全量逐行转换、首尾边界抽样、Ruff、compileall、定向单测和代码隔离必须通过。完成后固定停止在 `HOLD_FOR_QG_PRQK_SFT_DATA_REVIEW`。

## P10：共同词表、双 Cache 与两个 Qwen SFT（训练准备已完成）

P9 审核后按以下顺序继续：

1. 从同一个基础 Qwen3-0.6B 和两份逐值相同的 Token 清单构造一个共享扩词表模型；已完成并验证；
2. 分别全量扫描 GID-parent 与 NoGID 的 Train/Valid，固定 `cutoff_len=1024`，零超长且零目标截断后才构建各自不可覆盖 Cache；已完成，两分支超长/target truncation 均为 0；
3. 两个训练入口使用相同初始模型、3 epoch、global batch 512 和其余冻结超参数，差异只允许是数据注册名、Cache 和输出目录；两条 launcher `--dry-run` 已通过，正式训练待用户启动；
4. Test 不进入预检 Cache 或模型更新；本阶段不自动进入端到端生成评测、外部基线、Order-B 或其他消融。

当前不得把“训练脚本可启动”写成“训练已运行”。共享扩词表和两套 Cache 为 `COMPLETED_AND_VALIDATED`；两个 SFT checkpoint 和训练指标仍为 `NOT_STARTED`。状态固定为 `HOLD_FOR_QG_PRQK_DUAL_SFT_LAUNCH_REVIEW`。

---

## 4. 每阶段结束验收

1. 检查 `git diff`、`git status`，确认没有覆盖用户改动；
2. 代码阶段运行 compileall、相关 unit test、`--help` 和最小 sample/smoke；
3. 核对所有新输出都位于 `qg_prqk/outputs/`，并校验 schema、行数、manifest 与 checksum；
4. 在 `qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md` 记录真实状态、命令、结果、限制和唯一 `NEXT_ACTION`；
5. 只有正式实验才创建或追加 `qg_prqk/docs/experiments/QG_PRQK.md`，实验编号使用 `EXP-YYYYMMDD-NN`；
6. 完成当前阶段后停止，不顺带进入下一阶段。

最终回复只说明本阶段修改、核心数据流、验证结果、当前限制/下一步，以及是否 commit/push。
