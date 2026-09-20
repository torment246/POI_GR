> 归档说明（2026-09-19）：以下为 v3.0 登记前的 v2.1-CAT 规范原文，正文完整保留。
> 文中的“当前”“尚未启动”及阶段停止点均指当时状态；实际完成情况以[实施状态](../experiments/QG_PRQK_IMPLEMENTATION_STATUS.md)和[实验记录](../experiments/QG_PRQK.md)为准。
> 新方案与版本差异见 [v3.0 主规范](QG_PRQK_METHOD_SPEC.md)。

---

# QG-PRQK v2.1-CAT：类别层次锚定的查询粒度感知 Query–Geo 渐进残差语义 ID

> 文档状态：canonical 方法规范 v2.1-CAT。  
> 当前任务：北京单城、精确原始 `poi_id` 生成式检索。  
> 当前进度：P3A-FULL、P4–P8、NoGID `(S1,S2)` parent S3 受控变体与三方比较均已完成；用户随后明确批准同时保留 A4 GID-parent 和纯三位 NoGID 两个 SFT 分支。两套 Final ID、8,790,513 条逐行对齐的 history10 语料、共同扩词表和两套 Train/Valid Tokenized Cache 已完成，当前停在 `HOLD_FOR_QG_PRQK_DUAL_SFT_LAUNCH_REVIEW`；两个 Qwen SFT 尚未启动。\
> 量化器：RQ-KMeans / Projection Residual Quantization。  
> 主码本容量：`[512, 512, 512]`（2026-09-07 用户确认）。\
> POI 主视图：716,245 条 active POI 对应的冻结 BGE-M3 精确行子集。  
> Query 主视图：D1/D2 固定 Raw BGE；D3 使用已完成的 FULL-FINAL Adapted Query view。\
> 第一版完成边界：构建并静态验收完整 SID；该 P8-CAT 停止点已经履行，后续 Final ID/SFT 数据阶段由用户另行明确批准。

---

## 源码导航约定

当前实现按领域组织在 `src/qg_prqk/data/`、`adapters/`、`sid/` 和 `sft/`，统一入口为 `qg_prqk/scripts/qg_prqk.py`。P3A/P4/P5/P6/P7/P8 是实现期间的历史阶段编号，只用于解释冻结配置、manifest、输出目录和下文的实验演进，不再对应当前源码文件名。完整对应表见 `qg_prqk/README.md`；历史复核使用 `outputs/run_control/**/source_snapshot/` 中的原始代码。

---

## 1. 研究问题与最终创新点

传统流程为：

```text
POI text -> POI embedding -> RQ-KMeans -> SID -> Query-to-PID SFT
```

该流程存在四个问题：

1. **量化目标与查询生成目标不一致。**普通 RQ-KMeans 只优化 POI 连续表示的量化误差，不能保证真实用户 Query 容易生成这些 SID。
2. **真实 Query 的可识别粒度不同。**“火锅”通常只稳定表达粗类别，较具体的业态词可表达细类别，完整店名或地址才可能稳定表达具体 POI。将全部发单 Query 都视作精确单 POI 监督会制造冲突；只保留精确 Query 又会浪费大量类别语义。
3. **冻结 BGE 文本没有显式类别字段。**现有 POI `text` 只由名称、地址和别名组成；Train Query 覆盖 491,213 个 POI，因此仅靠内容向量和行为边不能为 active POI 闭集提供稳定的粗细语义锚点。
4. **地图 POI 需要局部地理消歧。**同类、同名或同一高密区域内的 POI 往往文本相似且坐标接近，需要在显式 GID 之外利用局部地理结构完成最终区分。

因此，QG-PRQK v2.1-CAT 的核心创新定义为：

> **类别层次锚定的查询粒度感知 Query–Geo 渐进残差量化。**全量 POI 类别层级分别为 S1/S2 提供粗细结构锚点；Train Query 的粗类别、细类别与精确 POI 目标分布决定其可靠监督深度；POI 内容视图与 Query 视图维护独立质心但共享离散 token，并沿共享路径分别逐层计算 residual；S3 再结合 D3 Query、确定性 GID 和局部地图困难关系完成精确 POI 消歧。

压缩表示为：

```text
Category hierarchy -> S1/S2 structural anchor
Query target distribution -> supervision_depth d(q)

(POI residual 0, Query residual 0, coarse category) -> S1
(POI residual 1, Query residual 1, fine category)   -> S2
(POI residual 2, Query residual 2, Local Geo) -> S3
```

其中 Query 仅在 `layer <= d(q)` 时生效。

---

## 2. 冻结决策

### 2.1 当前必须保持不变

1. 任务仍为精确单 POI 检索，不引入泛搜、多正样本返回或可变长度 SID。
2. POI 初始向量使用当前效果最好的 BGE POI embedding；不得切换为 E4 POI embedding。
3. BGE POI embedding、已有 POI `text`、L2 归一化和 POI 行映射沿用冻结产物；当前产物无 PCA，不得要求不存在的 PCA artifact。
4. SID 使用 RQ-KMeans，不使用 RQ-VAE。
5. 三层码本统一为：

   ```text
   K1 = 512
   K2 = 512
   K3 = 512
   ```

6. GID 继续使用现有 Geohash G1–G6，不重新学习。
7. 类别不拼入 POI `text`，不与 POI embedding concat，也不触发 POI 重编码。
8. SID 第一版只执行到 P8-CAT 静态验收；是否生成 Final PID 或进入 Qwen3-0.6B SFT，必须在 `HOLD_FOR_REVIEW` 后由用户另行确认。
9. 当前不做品牌、连锁、实体族、父子主体、RL、PRM、可变深度、动态 POI decoder 或在线 SID 更新。

2026-09-07 容量修订只影响下游量化器：每层 token index 为 `0..511`，三层 SID 长度不变；POI/Query embedding 仍为 1024 维。理论组合数为 `512^3=134,217,728`，不等于实际可用桶数，也不保证无碰撞。类别监督、D1/D2/D3、双视图图对齐、POI 拟合公共方向去除、projection residual、S3 Geo/困难图和原有权重均保留；P2/P2.5/Adapter 不重跑。容量变化的实际效果须由 P5–P8 的利用率、碰撞和 Prefix Probe 验收，不能从库缩小直接推断效果更好。

### 2.2 PID 输出顺序不作为 SID 构建阶段变量

同一个固定 SID 后续仍可做两种顺序消融：

```text
Order-A: G1 G2 G3 G4 G5 G6 S1 S2 S3
Order-B: S1 S2 G1 G2 G3 G4 G5 G6 S3
```

v2.1-CAT 第一轮不执行任一顺序的 Qwen SFT；P8-CAT 只发布 SID 候选并停止。若后续获准进入 SFT，第一轮仍只使用 Order-A；Order-B 不改变 SID，只能作为后续受控消融。

### 2.3 冻结输入与真实字段契约

北京首轮只读以下现有输入：

```text
POI catalog : data/beijing_poi_active_order14d_history10_20260715_json/
POI BGE     : qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/embeddings.npy
POI row map : qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/poi_ids.jsonl
Train orders: data/sft/beijing_order_main_v1/train.jsonl
P2 stats    : qg_prqk/outputs/qg_prqk_1024x3_v1_1/query_stats/
```

active POI 闭集共有 716,245 条，由北京 14 天 Train/Validation/Test 的当前目标与 history 中出现过的 POI ID 构成，用途仅是冻结可检索候选宇宙并保证后续业务评测的目标可回表。候选成员资格可以覆盖三种业务 split，但 QG 训练和选模不得读取 Validation/Test 的 Query 文本、订单标签或统计量；因此“候选 POI 是否可被检索”与“Query 监督是否泄漏”是两个独立合同。

POI catalog 实际字段为：

```text
address, alias, area, category, category_code, city, click_score,
displayname, lat, layer, lng, poi_id, source_dt, text
```

本方法允许使用 `poi_id, displayname, alias, category, category_code, address, lat, lng, city, source_dt, text`。`area/layer/click_score` 只可做 schema 审计，当前不进入层级构造、困难负样本、困难图或聚类。仓库当前不存在 `brand`、`brand_id`、`chain_id`、`parent_poi_id`、`entity_group_id`、`canonical_poi_id`、`landmark` 或物理 `category_l1/l2/l3` 列，任何实现都不得假设这些字段存在。

类别层级只绑定已有 `category_code`：完整 6 位 code 是 `fine_category_id`，前 2 位是 `coarse_category_id`，`category` 仅作可读路径。QG 从 2,337,178 行历史冻结输入中按 active POI 行序精确切出 716,245 行 BGE 与类别 index；保留全局 402 个细类稳定 ID，active 实际出现 397 个细类并覆盖全部 19 个粗类。P2.5-CAT 必须重新校验 schema、行数、hash、逐行类别非空合法和 `fine -> coarse` 唯一映射；词表中的 5 个零支持细类保留但不补入非 active POI，也不重编号。

POI embedding 固定为 716,245 行、1024 维、float16、已归一化的北京 BGE-M3 active 行子集；它必须与原 2,337,178 行冻结向量逐值一致，只改变候选行集合和行号，不重新编码文本。POI ID 映射必须逐行校验。当前只做北京，因此不新增其他城市 embedding。

### 2.4 数据切分与防泄漏

```text
Train      2026-07-01—2026-07-12   7,586,410
Validation 2026-07-13                 597,421
Test       2026-07-14                 606,682
```

Query 统计、层级标注、Adapter 和 SID 构建只能使用 Train。内部开发集必须从 Train 确定性切分；正式 Validation/Test 的 Query、订单标签与统计量不得参与构建、阈值校准或选参。唯一例外是上游已冻结的 active POI 候选成员资格可由三种 split 的目标/history POI ID 构成，它不提供 Query 监督。P2.5-CAT 只能读取现有 Train-only P2 Parquet、active 类别映射与必要的 active POI metadata，不得再次扫描原始订单文件。

### 2.5 代码、文档和输出边界

- 本方法全部源码、脚本、配置、测试和方法级训练/评测入口必须位于 `qg_prqk/`；复用仓库实现时先复制并登记 provenance，不得在运行时导入 `src/poi_gr` 或调用仓库其他方法脚本。
- QG 专属方法、状态和实验文档只保存在 `qg_prqk/docs/`；不复制到仓库根 `docs/`。
- QG 新产物只写入 `qg_prqk/outputs/`；仓库根 `data/`、`models/`、`outputs/` 中的冻结输入只读。
- PyTorch、FAISS、Transformers 和 LLaMA-Factory 是锁定版本的外部依赖，不复制完整源码。
- `overwrite=false` 为默认值；全量任务必须支持 manifest、checksum、断点续跑和独立版本目录。

### 2.6 外部基线的执行顺序

POI-only PRQ-KMeans 是 v2.1-CAT 主方法内部的初始化和 P8-CAT 静态对照，不等同于历史普通 BGE-RQK 外部基线。当前只执行到完整类别+Query+Geo SID 的 P8-CAT；历史普通 BGE-RQK 的复用或重跑、Qwen SFT、Order-B 和其他组件消融全部留到 `HOLD_FOR_REVIEW` 之后，并在执行前向用户确认。

---

## 3. 已有实验证据对方法的约束

### 3.1 Query 有价值，但不能做逐 POI 动态整体融合

已有 Query-Augmented POI Embedding 实验表明：

- Query 等权聚合可显著提升连续向量检索；
- 按 Query 频次与区分度加权进一步提升；
- 按 POI 的 Kish ESS 动态设置不同 Query 融合强度会明显下降；
- 固定整体强度、改善 Query 表示本身更稳定。

因此 v2 必须遵守：

```text
保留：Query 内部可靠性权重、共享 Query Adapter、层级 Query 监督
删除：ESS -> per-POI gate -> per-POI lambda
删除：将 POI 与 Query 聚合为一个候选级动态融合向量再做 RQ-KMeans
```

### 3.2 P2 全量产物作为 D3 Exact Core 保留

当前全量 P2 已得到：

```text
Train rows                    7,586,410
normalized unique Query       1,372,661
unique Query–POI pair         2,022,443
D3 exact-core Query           291,590
D3 exact-core covered POI     153,349
false-negative pair           557,639
all POI with Train Query      491,213
```

这些产物不得重跑、覆盖或删除。原“高置信 Query”在 v2 中正式解释为：

```text
D3-Exact-Core：能够可靠监督到具体 POI/S3 的 Query 核心集合
```

未进入 D3 的 Query 不是垃圾数据；P2.5 将继续判断它们能否提供 D1/D2 监督。

P2 产物继续使用既有 schema 与字段名，其中 `is_high_confidence=true` 在 v2 中只是 `D3-Exact-Core` 的语义别名，不修改、不迁移原 Parquet。冻结阈值为：

```yaml
exact_core:
  min_query_count: 2
  min_pair_count: 2
  min_top1_share: 0.75
  min_margin: 0.25
  max_normalized_entropy: 0.55
false_negative_mask:
  min_pair_count: 2
  min_share: 0.10
```

正式 P2 manifest SHA256 为 `25238e011dc0b162d4e27082f4bc918d1a2e64aa3ac0a46e415bee4ab07cd162`，并已确认 `limit=null`、`is_prefix_sample=false`、`valid_and_test_read=false`。v2 不重跑、覆盖、移动或删除该产物。

### 3.3 P3 实际完成度

P3A 已完成 50,000 条 D3 Query 的正式 Train-only Gate，状态为 `GATE_COMPLETED`，不是 full。该历史 Gate 使用 2,337,178 条旧全库候选，包含确定性 D3 选样、BGE Query cache、GPU 精确 Top-100、四源困难负例、P2 false-negative mask、Query-only Adapter、独立精确重检、overall/头尾/困难子集评测、checkpoint、日志和 manifest；全部代码、配置和产物继续只读保留。

正式 Gate 在 2,500 条内部 dev 上将 overall Recall@10 从 66.00% 提升到 71.24%，将 raw Recall@1 未命中的 1,560 条困难子集 Recall@10 从 45.51% 提升到 54.10%，两者 mean hard margin 也同时改善。完整指标与指纹记录在 `docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md`，历史产物位于 `outputs/qg_prqk_1024x3_v2_1_cat/query_adapter_exact/`。用户已确认 P3A-FULL：在排除这 50,000 条 Gate Query 后确定性抽取 20,000 条 Train-only D3 internal holdout；FULL-SELECT 从同一 identity 初始 Adapter 在其余 D3 上重新训练，每轮在 716,245 条 active POI 闭集上精确检索选 best epoch；随后从同一初始状态用全部 291,590 条 D3 固定 best epoch 重新训练 FULL-FINAL。Gate checkpoint 只参加同一 holdout 对比，不作为训练续点。

### 3.4 类别资产与硬绑定负向证据

已有项目资产确认完整 `category_code` 覆盖历史全库 2,337,178 个 POI，共 402 个细类；编码均为 6 位数字，前 2 位形成 19 个粗类。QG 必须将它按 active POI 严格 BGE 行子序切为 716,245 行，并在 active P2.5-CAT 产物中重新登记来源、行号和 hash；历史全库类别资产只作不可变来源。

同时，已有 E4 硬类别 `402×1024×2048` 实验已经证明不能令 `S1=category_code`：其 category purity 虽为 100%，SID 唯一率只有 53.1864%，碰撞 POI 占 60.2040%，最大桶 601，主要失败模式是“楼栋号”等大类被压入单一根节点。因此 v2.1-CAT 只允许使用可分裂到多个 code 的类别软 assignment 代价，并在 sample/medium Gate 检查 code collapse、桶结构和 Prefix Probe；不得恢复硬类别编码。

---

## 4. 类别契约与 Query 监督粒度

### 4.1 两级类别定义

Query 监督深度不能由旧 SID 定义，避免循环监督。v2.1-CAT 只使用全量 POI 的既有类别编码构造外部层级：

```text
L1 / Coarse Category：category_code 前 2 位，共 19 类
L2 / Fine Category  ：完整 6 位 category_code，共 402 类
L3 / Exact POI      ：最终评测使用的原始 poi_id
```

`category` 字符串只用于展示可读路径，不作为不稳定的主键。既有映射资产为：

```text
category indices : qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/category_indices.npy
category vocab   : outputs/embeddings/gnpr_sid/history10_pluscode6_top10_v1/category_vocab.parquet
row order        : qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/poi_ids.jsonl
```

active 冻结输入映射为 716,245 行 `int32`，合法范围 0–401，实际出现 397 个 index 且无 invalid category；缺失的全局细类为 `142411、261311、261410、261412、271029`，19 个粗类全部出现。active P2.5-CAT 必须在 QG manifest 中重新校验 hash、行序、逐行类别覆盖、6 位数字格式和 `fine_category_id -> coarse_category_id` 唯一性，并发布独立的 `category_mapping.parquet`。旧 2,337,178 行 P2.5 结果保留为历史产物，不作为后续 P3/P4 输入。`99` 等未知/其他编码是合法类别，不能静默删除。

本版不构造 Family/Brand/Chain/Parent 层，不读取或推断这些字段，也不通过名称规则生成实体族。Exact ID 固定为原始 `poi_id`；当前没有 canonical 映射，不自行合并等价 POI。

### 4.2 Query 类别分布

由 P2 的 Train-only `n(q,p)` 与类别映射计算：

```text
P_exact(p|q)  : exact POI distribution
P_fine(c|q)   : aggregate P_exact by fine_category_id
P_coarse(c|q) : aggregate P_exact by coarse_category_id
```

对粗、细类别分别计算：

```text
C_l(q) = max target concentration at level l
H_l(q) = normalized entropy at level l
```

只有一个目标类别时归一化熵显式记为 0，不允许除零。

### 4.3 监督深度

为每个 Query 赋值：

```text
D3 Exact POI       -> supervise S1, S2, S3
D2 Fine Category   -> supervise S1, S2
D1 Coarse Category -> supervise S1
D0 Contextual      -> do not supervise SID
```

第一版固定启动阈值：

```yaml
query_depth:
  d3:
    use_existing_p2_exact_core: true
  d2_fine_category:
    min_query_count: 3
    min_concentration: 0.80
    max_normalized_entropy: 0.40
  d1_coarse_category:
    min_query_count: 3
    min_concentration: 0.85
    max_normalized_entropy: 0.35
```

规则：

1. P2 `is_high_confidence=true` 原样判为 D3，不重新定义；
2. 非 D3 且细类别满足条件时判 D2；
3. 非 D2/D3 且粗类别满足条件时判 D1；
4. 其余为 D0。

P2.5-CAT 必须同时输出 Train-only 小范围阈值敏感性和人工样例，但第一版正式构建使用上述冻结启动值，不读取 Validation/Test 调参。

### 4.4 有效 Query–POI 边与 Query 权重

D3 仅用现有 Exact-Core top1 POI 作为强边；P2 false-negative targets 继续用于负样本保护，不扩成 S3 强边。D2 只保留属于主导 `fine_category_id` 的真实发单 POI 边，D1 只保留属于主导 `coarse_category_id` 的真实发单 POI 边。

Query 权重是“层级可靠性权重”，不是 POI 级动态融合系数。

对层级 `l`：

```text
reliability_l(q)
  = capped_log_support(q, cap=20)
  * C_l(q)^gamma
  * (1 - H_l(q))

capped_log_support(q, cap)
  = log1p(min(count(q), cap)) / log1p(cap)
```

因此第一版 `capped_log_support` 与 `reliability_l` 均位于 `[0,1]`。该定义已显式进入 v2.1 配置、P2.5 manifest 和单元测试，不再依赖实现侧隐含解释。

对一个 Query 的关联 POI 边进行归一化：

```text
sum_p omega(q,p) = 1
```

防止一个关联多个 POI 的粗 Query 产生多倍总质量。禁止使用：

```text
Kish ESS -> a_p -> lambda_{p,l}
```

Kish ESS 只用于诊断、Query 多样性报告和可选压缩策略。

---

## 5. P3 Query Adapter 的角色

### 5.1 第一版训练数据

第一版 P3 只使用 D3-Exact-Core：

```text
Query -> exact positive POI
```

使用 P2 false-negative mask 避免把真实合理目标误作负样本。

### 5.2 模型

```text
BGE Query embedding
-> LayerNorm
-> Linear(d, bottleneck)
-> GELU
-> Dropout(0.05)
-> Linear(bottleneck, d)
-> residual add
-> L2 normalize
```

BGE 主干和 POI embedding 全部冻结。默认 `d=1024`、`bottleneck=64`，输出层零初始化，使 Adapter 从 identity 开始。

2026-09-06 完成状态：FULL-SELECT 选定 epoch 2，FULL-FINAL 已从共同初始状态使用全部 D3 固定重训两轮；最终只标记为 D3 默认 Query view，尚未进入后续 SID 阶段。实验数字与来源哈希统一见实施状态文档的 `EXP-20260905-02`。

第一版负样本固定为动态 in-batch、POI BGE ANN、基于 `displayname/alias/category/category_code/address` 的 lexical-metadata，以及基于类目与距离的 local-geo 四类。`area/layer/click_score` 不参与；目标 POI、batch 内重复正目标和 P2 false-negative target 均必须 mask。

### 5.3 用途

P3 输出 Query view：

```text
r_q^0 = Adapter(BGE(q))
```

它不是线上 Qwen 前处理器，只服务于离线 SID Tokenizer。

P3A 50,000 条历史 Gate 已按本节契约完成：旧 2,337,178 条候选库的精确 Top-100 复核中，Adapter 在 overall 与困难子集均优于 raw BGE。P3A-FULL 不从该 checkpoint 续训，而是在 active POI 闭集下使用同结构、同 identity 输出的新初始状态重新执行 FULL-SELECT/FULL-FINAL。2026-09-05 用户确认：在创建 Adapter 前固定 seed=42，保存一份初始 state，SELECT/FINAL 逐值共用；历史 Gate 初始 state 未保存且建模后才设置 seed，因此不声称新 state 与历史 Gate 的初始随机权重逐值相同。该澄清不改变其余算法或超参数。最终只有 `query_adapter_exact_final` 可标记为 D3 默认 Query view；在 P3A-FULL 评审前不得进入 P4-CAT。

### 5.4 D1/D2 Query 的处理

第一轮固定使用以下 Query view，不在 P3A-FULL 中验证或修改 D1/D2：

```text
D1: Raw BGE Query view
D2: Raw BGE Query view
D3: Final Adapted Query view
```

“层级多正样本 Adapter”作为后续消融，不是首次完整方法的前置条件：

```text
P3-Exact      : D3 only
P3-Hierarchical: D3 exact + D2 fine-category multi-positive + D1 coarse-category multi-positive
```

---

## 6. Query–POI 层级监督图

### 6.1 节点

```text
POI nodes   : all POIs with frozen BGE content embeddings
Query nodes : D1/D2/D3 unique normalized queries
```

### 6.2 边

对每个 Query–POI pair 至少保存：

```text
query_id
poi_id
pair_count
P(p|q)
query_depth
layer_mask
edge_weight
coarse_category_id
fine_category_id
hierarchy_source
```

层掩码：

```text
D1 -> [1,0,0]
D2 -> [1,1,0]
D3 -> [1,1,1]
D0 -> [0,0,0]
```

条件概率按 Query 在该层保留的目标边集合归一：

```text
omega_l(q,p) = count(q,p) / sum_{p' in retained(q,l)} count(q,p')
w_l(q,p) = reliability_l(q) * omega_l(q,p)
```

每 Query 每层 `sum omega_l=1`，最终 `sum w_l=reliability_l(q)`，不能再把最终边权重新归一为 1 而抹去可靠性。这与已冻结的 P2.5 实现一致，不是新权重设计。每个 Query 的总边质量受 capped support 约束，防止热门 Query 支配聚类。边必须能够追溯到现有 P2 `query_poi_pairs`，不得重新扫描原始订单来生成另一套计数。

### 6.3 D3 边

D3 以现有 exact-core 主目标为主要强边；P2 false-negative target 不作为 hard negative。是否作为弱正边由配置控制，第一版仅用于负样本 mask，不参与 S3 强对齐。

### 6.4 D1/D2 类别边

D1/D2 Query 可连接多个主导类别内的真实发单 POI，但只在允许层级产生 alignment regularization：

- D1 Coarse-Category 只影响 S1；
- D2 Fine-Category 影响 S1/S2；
- 不得影响 S3。

### 6.5 P4 首个 sample 的冻结数据与运行契约（v1）

首次实现入口为 `qg_prqk/scripts/build_query_graph.py`，当时只开放 `--limit 1..1000`，按 P2.5 全局 `query_id` 顺序取前 1,000 个 D1/D2/D3 节点；D0 跳过，不重新抽取或修改 P3A holdout。这是工程 sample，不是随机效果评测集，也不是 P2.5“前 1,000 个全部 Query”的旧 sample。该版本完整代码已冻结于 `qg_prqk/outputs/run_control/p4_cat_active_512/sample_1000_attempt01/source_snapshot/`；复核旧 sample 须使用该快照的 `scripts/build_query_graph.py`，不能改写其 manifest 来适配新源码。

- `p4_data.py`：从冻结 manifest 的文件清单读取 P2.5 深度/层级边、active POI mapping 和 P2 false-negative mask；只校验并读取本样本涉及的 Query/FN 分片，mapping 流式校验全行序。原始订单及业务 Validation/Test 不读取。
- D3 的 `query_id/normalized_query` 先和 FULL selection 对齐，再校验整个 selection 的 ID/文本顺序哈希与已完成 cache 合同一致；只读取被引用的原子 D3 cache 块，逐文件复核 SHA256，禁止重编码 D3。
- 节点使用独立连续 `node_row`；保留原 `query_id`、文本、深度、三位 `layer_mask`、support、逐层 reliability、类别、view 名称和 D3 cache 行。边保留 P2.5 的原字段与权重，另加 `node_row/poi_row_index`；FN 单独保存，不变成 D3 次要强对齐边。
- D1/D2 只调用已冻结 BGE Query 编码器，保持 max length 128、batch 256、BF16、无 instruction、float16/L2 输出；D3 只应用 role/config/epoch/初始化来源均匹配的 FULL-FINAL Adapter。BGE 主干和 Adapter 均不更新参数。
- 每 256 个节点原子提交一对 raw/view NPY 与 SHA256，再推进 progress；完整输出按节点顺序流式合并。`--resume` 验证配置、输入、源码与已提交块后跳过已完成编码；无 journal 的文件只保留为 unconfirmed，不能据预分配大小认定完成。
- `--validate-only` 重读冻结输入，逐项比较节点/边/FN、shape/dtype/有限值/L2 norm、块与合并矩阵、D1/D2 raw=view、D3 raw=旧 cache，以及 manifest/成功标记和指标；这是数据/路由验收，不是新的召回评测或 Adapter 模型选择。

本小步输出固定在 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/sample_001000/`（其他 limit 使用对应六位数字）：`query_nodes.parquet`、`query_poi_edges.parquet`、`false_negative_mask.parquet`、`raw_query_embeddings.npy`、`query_embeddings.npy`、raw/view 分块、`config_resolved.json`、`progress.json`、`manifest.json` 和 `_SUCCESS`。两份完整向量的 shape 均为 `[sample_query_rows,1024]`；全部 716,245 个 POI 节点仍引用原 active 行映射，不只保留有边的 POI。

P4 产物是公共方向去除前的分层 Query view；同一个 POI-fitted 公共方向变换及逐层 projection residual 留在 P5–P7 执行，不能在 P4 提前拟合或重复应用。sample 验收后停在 `HOLD_FOR_P4_SAMPLE_REVIEW`，medium/full 与 P5 不由此入口自动启动。

### 6.6 P4 medium 的规模与复用契约（v2）

sample 经用户确认继续后，当时 v2 入口增加显式 `--gate medium --limit 50000`；依然按相同顺序取 active Query 前缀，sample 上限 1,000、medium 上限 50,000，当时不开放 full。模型、batch size=256、max length=128、精度、归一化、类别/边权与 view policy 均不变。medium 仍是工程规模验收，不是随机评测或选模。

- 新 manifest schema 为 `qg-prqk-p4-query-graph-v2`，记录 gate、来源 sample manifest 路径/冻结 SHA256/复用行数；旧 sample 保持 v1 与原源码快照。当前源码不放松签名校验以兼容旧目录。
- medium 构建/恢复必须显式传 `--reuse-sample` 与 `--reuse-sample-sha256`。先检查成功标记、冻结 manifest、全部 artifact 哈希、图/FN/文本/行序与当前前缀一致、源文件哈希、配置和 Raw/D3 cache；再逐值复用旧 sample 的 Raw 和最终 view，不重复编码其中的 D1/D2，也不重复应用其中的 D3 Adapter。
- 每 8,192 个节点提交一对 Raw/view 缓存块（继承既有 encode buffer），BGE/Adapter 实际计算仍最多每批 256。输出逐块写入和顺序合并，向量校验及 drift 计算最多处理 8,192 行 float32 临时矩阵；不建立 Query×POI 两两矩阵。
- 图表在明确的 50k 上限内保留内存表示；来源签名保持原 JSON 字节合同，但逐段更新 SHA256，避免再拼一个大型 JSON 字符串。D3 Raw 工作矩阵最多约 97.66 MiB，结构数据与模型的实际峰值 RSS 在运行 manifest 记录，不宣称当前实现已通过全量内存验收。
- 新产物位于 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/medium_050000/`，不覆盖 `sample_001000/`。独立 validator 重算图/view 指标，并检查 medium 前缀与 sample 两份向量逐值一致；完成目录 resume 不加载模型。

验收后停止于 `HOLD_FOR_P4_MEDIUM_REVIEW`，不自动进入 full、P5、SID、Final PID、SFT 或基线。

### 6.7 P4 full 的有界内存与 medium 复用合同（v3）

medium 验收并经用户确认后，full 入口只允许固定 `--gate full --limit 342879`，即 active P2.5 中全部 D1/D2/D3 Query；行数过多或不足都直接拒绝。模型、batch size=256、max length=128、精度、归一化、Query view policy、类别/边权与 false-negative 定义不变，本轮只扩展数据规模。

- full 构建/恢复必须显式传入 `--reuse-medium` 和外部冻结的 medium manifest SHA256。系统先核对 medium 的成功标记、manifest、全部 artifact/块哈希、图/FN/文本/行序、配置与来源，再将前 50,000 行 Raw/view 逐值复用；不链式重新依赖 sample 推测当前前缀。
- D3 Raw 不再创建 `[342879,1024]` 工作矩阵，而是按当前 8,192 行节点区间，从已验 SHA256 的冻结 D3 NPY 分块通过 mmap 向量化选行。D1/D2 文本只在 medium 之后的未完成行做 BGE；D3 只对未复用行应用 FINAL Adapter。
- v3 来源签名对 Arrow table 按 8,192 行 RecordBatch 序列化后逐块更新 SHA256，不将全量节点/边/FN 转换为巨型 Python JSON。图守恒指标通过 Arrow group-by 和列计算复算；不构建 Query×POI 矩阵。
- full 仍保留 Query 图 Arrow 表的有界内存表示，真实 dry-run 在 342,879 Query/1,204,570 边上峰值约 4.54 GiB；正式构建+内联 validator 外层峰值约 7.07 GiB，无 OOM/Swap。该数据规模已实测，不等价于任意规模均为流式。
- 输出固定为 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/full_342879/`，包含 42 对原子 Raw/view 分块、两份完整 float16 矩阵、图/FN/config/progress/manifest 和 `_SUCCESS`。validator 要求 `is_full=true`，并复算前 50,000 行与 medium 的两种向量逐值一致。

2026-09-07 真实 full 已完成：D1/D2/D3=`25,639/25,650/291,590`，S1/S2/S3 边=`514,500/398,480/291,590`，FN=`376,701`，覆盖边上 POI=`210,965`；新编码 D1/D2=`43,751`。构建、独立 `--validate-only` 与屏蔽 GPU 的完成态 `--resume` 均退出 0。P4 完成后停在 `HOLD_FOR_P4_FULL_REVIEW`；它提供 SID 构建的完整 Query 侧前置，但 P5 聚类不由 P4 入口自动启动。

---

## 7. 双视图图正则渐进 RQ-KMeans

### 7.1 两类残差视图

POI 内容残差：

```text
r_p^0 = normalized BGE POI embedding after global-direction removal
```

Query 残差：

```text
r_q^0 = normalized depth-specific Query view under the same POI-fitted global-direction transform
```

其中 D1/D2 view 为 Raw BGE，D3 view 为 FULL-FINAL Adapter。主方法固定公共方向去除（配置键 `prqk.remove_global_direction=true`）：只在冻结 POI catalog 上估计公共方向，对 POI 与 Query 应用同一线性去除后分别 L2 normalize。关闭该变换只属于 P8-CAT 人工审查后的可选消融，不能在首轮实现中自行切换。

公共方向和变换固定定义为：

```text
x_hat = x / ||x||
mu = (1 / N_poi) * sum_p x_hat_p
T(x) = normalize(x_hat - [(x_hat · mu) / (mu · mu)] * mu)
```

`mu` 必须使用完整冻结 active POI catalog 的全部 716,245 行估计；sample/medium 只限制参与 K-Means 拟合的 POI 行，不得用选样子集重估 `mu`。P5–P7 对 Query 使用同一个已冻结 `mu`，不得从 Query 重新估计公共方向。

### 7.2 每层双质心、共享 token index

每层 `l` 有：

```text
U_l = {u_l,0 ... u_l,511}  # POI content centroids
V_l = {v_l,0 ... v_l,511}  # Query centroids
```

`u_l,k` 与 `v_l,k` 对应同一个 token `S_l=k`，但不要求短 Query 和长 POI 共用完全相同的向量中心。

### 7.3 S1 粗类别软代价

根据上一轮 POI assignment 统计：

```text
n(k,c) = number of POIs assigned to S1=k and coarse_category=c
P(c|k) = [n(k,c) + alpha_1 * P_global(c)] / [n(k) + alpha_1]
alpha_1 = 32
```

对候选 S1 code `k` 的类别代价为：

```text
D_cat1(p,k)
  = -log P(coarse_category(p)|k)
    / log(max(num_coarse_categories, 2))
```

该代价只鼓励一个 S1 code 的类别分布更集中，允许同一粗类别占用多个 S1 code；绝不把 code id 硬映射为 category id。

### 7.4 S2 条件细类别软代价

S2 token 可在不同 S1 父节点下复用，因此不得使用 `P(fine_category|S2)`，必须统计完整前缀：

```text
P(c_fine | s1,s2)
  = [n(s1,s2,c_fine) + alpha_2 * P(c_fine|s1)]
    / [n(s1,s2) + alpha_2]
alpha_2 = 16
```

未出现的 `(s1,s2)` 回退到 `P(c_fine|s1)`，再无统计时回退全局分布。候选 S2 code 的代价为：

```text
D_cat2(p,k)
  = -log P(fine_category(p) | S1(p), candidate_S2=k)
    / log(max(num_fine_categories_under_coarse, 2))
```

### 7.5 分层复合目标

```text
J1 = POI distortion
   + 0.05 * D1/D2/D3 Query distortion
   + 0.05 * D1/D2/D3 graph disagreement
   + 0.08 * coarse-category cost

J2 = POI residual distortion
   + 0.10 * D2/D3 Query residual distortion
   + 0.10 * D2/D3 graph disagreement
   + 0.08 * fine-category path cost
```

形式化的共享部分为：

```text
sum_p d_p(r_p^{l-1}, u_l,s_p)
+ alpha_l * sum_{q:d(q)>=l} d_q(r_q^{l-1}, v_l,t_q)
+ lambda_l * sum_{(q,p):d(q)>=l} w_qp * I[t_q != s_p]
+ eta_l * sum_p D_cat_l(p,s_p),  l in {1,2}
```

其中 `s_p` 与 `t_q` 分别是 POI/Query token；D1/D2/D3 通过层掩码控制作用深度。类别权重 `0.08` 是第一版 full 的冻结起点。sample 只承担工程闭环验证，权重是否调整必须根据 full 对照结果决定，不能依据稀疏 sample 单独调参。

### 7.6 交替优化

每层采用以下迭代：

1. 使用 POI-only PRQ-KMeans 质心和 assignment 初始化 `U_l, s_p`；
2. 将 Query 分配到 `V_l`，初始化 `t_q`；
3. 固定 Query assignment 和当前类别分布，更新 POI assignment：

   ```text
   content distance + graph alignment penalty + category cost
   ```

4. 固定 POI assignment，更新 Query assignment：

   ```text
   query distance + graph alignment penalty
   ```

5. 分别更新 `U_l` 与 `V_l`；
6. 对 Query 质心做 content-prior shrinkage，防止稀疏 code 不稳定；
7. S1 更新 `P(coarse|S1)`，S2 更新 `P(fine|S1,S2)`；
8. 重复直到目标和 assignment 收敛。

POI-only PRQ-KMeans 必须先独立通过 sample/full 正确性核验，但它只是 v2.1-CAT 的内部初始化 `A0`；历史 BGE-RQK 的重跑、同协议 SFT 或正式外部比较不在 P8-CAT 前执行。每轮 assignment 使用固定 seed、稳定 POI/Query 顺序和较小 code id 的平局规则；空 code 使用确定性的 farthest reinit。

### 7.7 Query 质心稀疏收缩

对 Query support 较少的 token：

```text
v_l,k <- normalize((sum assigned query residuals + tau * u_l,k) / (n_q,k + tau))
```

首轮 `tau=32`。这是 code-level 统计收缩，不是 per-POI 动态融合。

### 7.8 类别与 Query 权重渐进打开

为避免 POI-only 初始化被类别或 Query 项突然破坏：

```text
iteration 1-2: category weight = 0；Query/graph 为目标值的 50%
iteration 3-6: 线性 ramp 到目标值
iteration 7+:  使用目标值
```

必须逐轮记录 content distortion、Query distortion、graph disagreement、category CE、active codes 和桶分布。sample 若出现代码错误、非法 assignment、缺边或 code/路径灾难性坍塌则不得发布 full；sample 的普通指标升降不作为参数选择依据。full 的同口径指标才用于方法判断和后续调参。

### 7.9 Projection Residual

POI 与 Query 分别沿各自质心做 residual：

```text
r_p^l = normalize(r_p^{l-1} - Proj_{u_l,s_p}(r_p^{l-1}))
r_q^l = normalize(r_q^{l-1} - Proj_{v_l,t_q}(r_q^{l-1}))
```

关键规则：

- S2 使用 Query 的 S1 residual，不能重复使用原始 Query embedding；
- S3 使用 Query 的 S1/S2 residual；
- D1 Query 在 S1 后停止，不再创建 S2/S3 residual；
- D2 Query 在 S2 后停止；
- D3 Query保留到 S3。

### 7.10 第一版启动权重

首轮建议弱到中等强度：

```yaml
dual_view_rqk:
  query_distortion_weight: [0.05, 0.10, 0.20]
  graph_alignment_weight:  [0.05, 0.10, 0.20]
  category_weight:         [0.08, 0.08, 0.00]
```

这些是第一版 full 启动值，不声称最优。P6-CAT 的 sample 只在固定 Query depth、码本容量和 Geo-off 条件下检查实现闭环；正式方法判断和后续参数修改只使用 full 的同口径对照。其他消融开关实现后仍留到 P8-CAT 人工审查之后决定。

---

## 8. S3：Exact Query + Local Geo 精确实体消歧

### 8.1 GID 的角色

G1–G6 是确定性绝对地理位置，不参与语义 residual 减法。S3 的地理视图只表达：

```text
在给定 GID 和 S1/S2 后，局部父组内部的相对位置与实体冲突
```

S3 只使用 D3 Exact-Core Query 的 S1/S2 后 residual；D1/D2 不进入 S3。S3 不再加入类别分类代价，fine category 仅用于筛选局部困难边。

### 8.2 局部父组

第一版使用：

```text
parent_key = (GID6, S1, S2)
```

同时离线统计 `(GID4/5/6, S1, S2)` 的桶分布，为后续 GID 粒度消融准备数据，但不在首轮同时修改 GID 长度。

### 8.3 Geo feature

将 `lng/lat` 转成北京局部米制坐标后，对 POI 构造局部特征：

```text
cell-relative dx, dy
log(1 + parent-group distance)
sin(parent-group bearing), cos(parent-group bearing)
```

各维仅在 Train catalog 上做 robust standardization（median/IQR；退化维回退 mean/std），再 L2 normalize。singleton 父组的组内相对特征和 geo gate 置 0。不得将原始经纬度字符串拼入 BGE，也不得让 S1/S2 使用 Geo feature。

### 8.4 S3 三视图目标与困难边

S3 在双视图目标上增加：

```text
geo distortion
local hard-entity collision penalty
```

形式：

```text
J_3 = J_3_dual_view
    + gamma_geo * local_geo_distortion
    + rho_hard * local_confusion_collision
```

困难候选只在同一完整 `(GID6,S1,S2)` 父组内构建，优先要求：

```text
same fine_category_id
AND same GID6（由固定 parent_key 保证）
AND composite(BGE, displayname/alias, category, address, geo) >= 0.60
```

复合分数权重固定为 BGE/name-alias/category/address/geo=`0.35/0.30/0.15/0.10/0.10`。每个 POI 最多保留 Top-20，并保存 similarity 与 relation flags。第一版不扩展相邻 GID6；该扩展只可作为用户确认后的独立消融。`area/layer/click_score` 不参与；当前没有 brand、family、parent 或 canonical duplicate 映射，不得声称据此构边或构造评测等价类。边必须排除同一 `poi_id` 自环，并排除 P2 false-negative protected targets；若以后获得经过验证的等价 POI 映射，只能在用户确认后的独立版本中接入。

第一版启动权重为 Query distortion `0.20`、graph alignment `0.20`、Geo `0.10`、hard collision `0.05`；singleton 的局部 Geo 项强制为 0。S3 初始 hard assignment 后只在有界候选 code 内做确定性局部细化；不得为了静态唯一率强制父组内每个 POI 使用不同 S3。

---

## 9. RQ-KMeans 迭代与收敛

### 9.1 每层默认迭代

```yaml
hard_alternating_fit:
  min_iter: 8
  max_iter: 60  # 2026-09-09 用户确认用于 P6/P7；旧 30 轮配置只读保留
  objective_rel_tol: 1.0e-4
  poi_assignment_change_tol: 1.0e-3
  query_assignment_change_tol: 2.0e-3
  patience: 2
```

首版固定在 hard fit 收敛后做 5 轮 Top-k centroid refinement，`topk=5`、`beta=15`。对每个 residual 只保留 cosine 最近的 5 个质心，并在该集合内计算：

```text
w_ik = exp(beta * cos(r_i, c_k))
       / sum_{j in TopK(i)} exp(beta * cos(r_i, c_j))
c_k <- normalize(sum_{i: k in TopK(i)} w_ik * r_i)
```

没有 soft support 的 code 使用当前 hard assignment 中的确定性 farthest row 重置；5 轮结束后必须重新做一次 hard cosine assignment，较小 code id 处理精确平局。该 refinement 只软化质心更新，不产生多 token assignment。

### 9.2 RQ-KMeans 没有梯度 loss，但有优化目标

不存在：

```text
backward / optimizer / learning rate
```

必须记录：

```text
objective
relative objective improvement
POI assignment change
Query assignment change
active codes
cluster size distribution
alignment disagreement rate
category CE / path CE
Geo distortion
hard collision
```

### 9.3 停止条件

连续两轮同时满足：

```text
relative objective improvement < 1e-4
POI assignment change < 0.1%
Query assignment change < 0.2%
```

或达到当前阶段配置的轮数上限。P5 旧 Gate 的上限为 30；2026-09-08 用户确认的新 P5 试运行协议将上限改为 60。2026-09-09 用户进一步明确确认 P6/P7 复合目标也使用 `max_iter=60`；该变更通过独立 overlay 落地，其余权重、warm-up、Top-k5、projection residual 和停止阈值不变，旧 30 轮配置与历史 manifest 不修改。

复合 objective 不要求每轮严格单调；若连续 3 轮明显升高，必须停止并标记失败。算法收敛不等于 Tokenizer 有效，还必须通过静态评测和 Prefix Probe；v2.1-CAT 本轮不以最终 SFT 作为 P8 Gate。

### 9.4 100k 收敛与 Top-k 配对诊断证据及 P5 试运行决策

2026-09-07 的 P5 100k medium 触发审核后，已在相同选样、seed、初始化、公共方向和 projection residual 上运行独立诊断。原 100k Top-k5 分支、hard30+Top-k-off 和 hard60+Top-k-off 的三层 final distortion 分别为：

| 分支 | S1 | S2 | S3 | 收敛轮次 |
|---|---:|---:|---:|---|
| 原 hard30+Top-k5 | 0.474177 | 0.559833 | 0.600312 | hard 三层均未收敛 |
| hard30+Top-k-off | 0.466934 | 0.557356 | 0.597651 | 三层均未收敛 |
| hard60+Top-k-off | 0.466731 | 0.556393 | 0.597649 | 43 / 39 / 33 |
| hard60+Top-k5 | 0.473968 | 0.560269 | 0.598943 | 43 / 40 / 35 |

这证明在 100k 规模下 `max_iter=30` 不足，且原 Top-k5 相对其自身 hard30 trace 的 distortion 三层均恶化。但评估不能只看 distortion：原 Top-k5、hard30-off、hard60-off 的 distinct SID 依次为 `96,240/95,786/95,722`，collision excess 为 `3,760/4,214/4,278`，原 Top-k5 的码均衡也更好。因此当前证据表明存在误差–均衡/碰撞权衡，不能自动删除 Top-k。

2026-09-08 补充诊断没有再次拟合一条近似 hard60 路径，而是每层只运行一次 hard60，再从逐值相同的 codebook/assignment 端点原位分叉 Top-k5。直接配对结果显示，Top-k5 在 S1/S2/S3 分别增加 distortion `0.007237/0.004646/0.003592`，同时把 Kish ESS 从 `401.23/392.27/410.87` 提高到 `432.72/427.77/435.17`，Gini 从 `0.2840/0.2881/0.2637` 降到 `0.2377/0.2449/0.2324`。因此 Top-k 的作用应定义为显式码均衡正则，而不是 distortion 改善步骤。

完整 hard60+Top-k5 路径的 distinct SID=`96,171`、collision excess=`3,829`、最大桶=22；相对 hard60-off 的 `95,722/4,278/23`，增加 449 个唯一 SID并减少 449 个冗余碰撞。每层 pre-Top-k 与 Top-k5 是严格直接配对；hard60-off 与 Top-k5 的 S2/S3 因上一层 projection residual 不同，仍只能解释为完整管线分支比较。

旧下游 YAML `qg_prqk_v2_1_category_active_512x3.yaml` 仍保持 `max_iter=30` 和 Top-k5，两个诊断 role 均不是 canonical checkpoint，也不得覆盖。用户于 2026-09-08 确认采用 `max_iter=60` 并保留 Top-k5 做 P5 试运行；新入口 `qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml` 严格绑定旧 YAML 与三份诊断 manifest，只覆盖 P5 hard `max_iter`，输出到独立 namespace。新 10k Gate 在 16/15/16 轮收敛且无结构退化。用户于 2026-09-09 明确授权跳过该协议的 100k/500k，由已验收 sample 直接构建 full；该例外通过显式 CLI 开关、sample manifest SHA 和 `P5_A0_ONLY` 范围写入 full 合同，默认 Gate 顺序未放宽。716,245 POI full 三层在 57/52/57 轮收敛，512 code 全激活，distinct SID=`606,045`（84.6142%），manifest SHA256=`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`，独立 validator 通过。

用户随后确认 P6/P7 的 hard alternating 上限同样设为 60。历史入口 `qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml` 绑定旧 downstream signature、P4 full 与 P5 full manifest，只修改 `max_iter`。用户当时确认 sample=`10,000` 基础 POI + 前 `1,000` Query、medium=`100,000` 基础 POI + 前 `50,000` Query，并保留每个 Query 的全部 S1/S2 边及其 target POI 闭包；实际规模为 sample `11,459 POI/2,610边`、medium `148,860 POI/134,265边`。历史 v1 配置及未运行 medium 合同继续只读保留。

P6 sample 的首次启动因跨 warm-up 权重比较 objective 被保护性终止；失败目录和日志完整保留。修正为仅在第 6 轮 warm-up 结束后累计 objective increase streak 后，同协议 attempt02 完成：S1/S2 hard 在 12/9 轮收敛，512 个 POI code 均全激活。相对同闭包 P5 A0，图加权一致率从 `28.88%/32.48%` 提高到 `94.44%/89.76%`，S1 coarse purity 提高 `3.55pp`、S1+S2 fine path purity 提高 `0.30pp`；但 Query cosine distortion 从 `0.6217/0.5870` 上升到 `0.6831/0.6546`，distinct prefix 从 `10,403` 小幅降到 `10,387`。因此 sample 工程结构正常，但“Query distortion 改善”门禁未通过，必须人工评审后才能决定是否调整权重或进入 medium。

同一闭包上的六分支诊断进一步拆解了该差值。canonical Query 质心每个占用码的 Query 支持中位数仅为 2，而 `tau=32` 令 Query-weighted POI prior mass 达到 S1/S2 `85.20%/90.85%`。只关闭 graph 时 distortion 降为 `0.5889/0.5822`；只把 prior 降到近零时降为 `0.2624/0.2309`，同时图一致率仍为 `63.73%/60.79%`；Top-k5 只造成 `0.0164/0.0082` 的边际 distortion，category 对 Query distortion 的影响约为千分之一。近零 prior 分支激活 `492/500` 个 Query code，接近小样本记忆，不能冻结为正式参数。该证据只解释稀疏 sample，不作为参数选择结论。

2026-09-09 用户明确把后续规模协议改为“sample smoke → full”：小规模只检查代码、数据和产物闭环，跳过 medium；所有方法指标和参数调整只以 full 为依据。新增且不覆盖历史配置的 `qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml` 绑定 P6 sample manifest SHA256=`10f40347f60e38289cb9b0405787433e176c0ab051b718773d9e3e2ab36a612a`，full 固定使用 716,245 POI、342,879 Query、912,980 条 S1/S2 边；除规模和 Gate 顺序外不改算法。full 已完成并独立验证。随后完成的 full S2-only 受控诊断冻结正式 S1 endpoint，只重跑 S2 六分支；结果把 `+0.028162` 总退化分解为 S1 residual `-0.022802`、S2 质心适配 `-0.053880` 与 coupled assignment `+0.104844`。graph-off 虽把 distortion 降至 `0.510165`，图一致率却降至 `28.4846%`，因此诊断不授权直接移除图监督或改写正式 P6。

2026-09-11 用户接受上述 P6 诊断作为留存并确认继续 P7，不修改 P6 权重。P7 使用独立 `qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml`：冻结 P6 S1/S2，只让 D3 residual 进入 S3；Query/graph/Geo/hard 权重为 `0.20/0.20/0.10/0.05`。困难边只在固定 `(GID6,S1,S2)` parent 内构造，因此第一版是同 GID6、不扩展相邻 cell；same fine category 只作候选筛选，不形成 S3 分类代价。sample smoke 后 direct full 已完成，full hard 在第 40 轮收敛，条件 S3 加权一致率=`85.2967%`、困难边加权碰撞率=`17.5233%`、纯 SID distinct=`617,887`（86.2675%），manifest SHA256=`dd395ab0d11321106455ad05b18503af9b7d3dda7cb7fd85e531655cece2133c`。这些是 P7 结构指标，完整 A0/A4 方法结论仍由 P8 给出。

---

## 10. 评测与 Gate

### 10.1 数据质量

- D0/D1/D2/D3 Query 数量；
- 各层覆盖 POI 数；
- coarse/fine category 数量、覆盖率与 `fine -> coarse` 一致性；
- 每层 concentration/entropy 分布；
- 人工抽样 20–50 条。

### 10.2 SID 静态指标

- 利用率、Kish ESS、Gini、路径熵；
- residual energy；
- unique SID / PID；
- bucket mean/p50/p90/p95/p99/max；
- Query–POI token disagreement；
- D1/D2/D3 各层前缀一致率；
- `Purity/NMI(coarse_category, S1)`；
- `Purity/NMI(fine_category, S1+S2)`；
- 每个 coarse category 使用的 S1 code 数与每个 fine category 使用的 S1+S2 path 数；
- 同 GID 局部实体分离率。

### 10.3 Prefix Probe

至少评估：

```text
Query -> S1
Query + correct S1 -> S2
Query + correct GID/S1/S2 -> S3
cumulative S1-S3
```

并按 D1/D2/D3 分桶。

### 10.4 P8-CAT 第一版比较与停止条件

本轮只比较：

```text
A0: POI-only PRQK 512×3 initialization
A4: Query + Category + Geo full
```

A0 是主方法必需的内部初始化，不是历史普通 BGE-RQK 外部基线；比较只在 A4 完整 SID 跑完后的 P8-CAT 进行。必须联合查看类别结构、Query disagreement、Prefix Probe、局部碰撞、Distinct SID、桶分布和 residual energy，不能仅凭类别纯度更高宣布成功。

若类别结构改善但 Query 可预测性或桶结构明显恶化，标记 `REVIEW_REQUIRED`。P8-CAT 无论结果为 `COMPLETED`、`REVIEW_REQUIRED` 还是 `FAILED`，唯一下一步都必须是 `HOLD_FOR_REVIEW`，不得自动启动 Final PID、Qwen SFT、外部基线或其他消融。

2026-09-11 已按该口径完成 P8。A4 相对 A0 的 distinct SID 增加 11,842、最大桶从 208 降至 95，coarse/fine purity 分别提高 10.4784/6.0707pp，同 GID6 pairwise separation 也改善；但冻结的 content-only Prefix Probe weighted S1/S2/S3 从 A0 的 `29.9614%/39.2334%/90.1722%` 降为 A4 的 `21.0191%/36.2871%/89.7239%`，autoregressive cumulative S1–S3 从 `12.7183%` 降为 `7.7743%`。因此正式结论为 `REVIEW_REQUIRED`，状态停在 `HOLD_FOR_REVIEW`。该结论不自动修改本节算法定义；完整口径和产物见 `EXP-20260911-02`。

#### 10.4.1 NoGID `(S1,S2)` parent S3 受控变体

2026-09-11 用户要求追加一个纯三位 SID 变体：冻结 P6 S1/S2，S3 只以 `(S1,S2)` 为 parent，不读取、不计算、不输出 GID6/geohash，POI/Query SID 均为 `[S1,S2,S3]`。该变体沿用 P5 S3 初始化、D3 Query 图、projection residual、Query/graph/Geo/hard 权重、warm-up、Top-k5 和 60 轮上限；连续 Geo 改为相对 `(S1,S2)` parent 中心的软特征，困难边候选改为 `(S1,S2,fine category)`。原 same-GID 常数贡献 `0.10` 移除后，筛选阈值从 `0.60` 等价平移到语义分数 `0.50`，不改其他组件权重。

full 结果为 distinct SID=`613,198/716,245`，max bucket=`159`，S3 加权训练一致率=`84.8905%`。为排除候选 parent 口径的不公平，P8 将 A0、原 GID-parent A4 和 NoGID A4 的 S3 probe 都改为 correct `(S1,S2)` 候选；三者 weighted S3 为 `67.7142%/66.2934%/65.8531%`，cumulative S1–S3 为 `9.7802%/5.8439%/5.7867%`。NoGID 相对原 A4 减少 4,689 个 distinct、max bucket `95→159`，S3 probe 下降 `0.4403pp`。因此工程变体成立，但实验不支持它替换原 GID-parent A4；本节不修改 canonical v2.1-CAT 定义，状态为 `HOLD_FOR_NOGID_S3_REVIEW`。完整口径见 `EXP-20260911-03`。

---

## 11. 第一版 SID 产物与后续边界

### 11.1 本轮 SID

```text
SID(p)     = [S1, S2, S3]
```

P8-CAT 发布三层 SID、codebooks、assignments、bucket index、静态指标和完整 manifest，然后停止。不得在聚类过程中强制 SID 唯一，也不得为了唯一率把类别软约束改成硬编码。

### 11.2 双 Final ID 与 SFT 协议

P8-CAT 和 NoGID 对照的停止点均已履行。2026-09-11 用户明确批准进入 SFT，并要求同时构造 A4 GID-parent 与第一种 NoGID 两个分支。两者使用不同 Final ID，但严格共用相同订单、Query、请求 GID、用户、history 事件、日期切分和样本顺序：

```text
GID-parent Base ID = [G1, G2, G3, G4, G5, G6, S1, S2, S3]
GID-parent Final ID = G1 G2 G3 G4 G5 G6 S1 S2 S3 [D]

NoGID Base ID = [S1, S2, S3]
NoGID Final ID = S1 S2 S3 [D]
```

这里的 NoGID 只表示 history/target 的 POI 标识符不含 GID；订单请求位置 `<USER_GID>` 仍是两个分支共同的输入上下文，不得删除。history POI 与 target POI 在各自分支必须使用同一种 Final ID。Dedup 只在完整 Base ID 碰撞桶内按原始 `poi_id` 字典序从 0 确定性分配，始终位于最后；单例不追加 D。

全量结果中，GID-parent 的 Base ID distinct=`672,294/716,245`、最大桶=`56`，使用 `D0..D55`；NoGID 的 Base ID distinct=`613,198/716,245`、最大桶=`159`，使用 `D0..D158`。两者追加 D 后都达到 `716,245/716,245` 唯一。为保证公平初始化，两个分支共用由两份完全相同的 `special_tokens.json` 生成的一个 3,743 Token 普通原子词表；NoGID 未使用的更长目标不会触发单独词表或单独初始化。

Qwen3-0.6B 第一轮继续采用 Order-A/history10、`qwen3_nothink`、`cutoff_len=1024`、packing、`train_on_prompt=false`、3 epoch 和 global batch 512。正式训练前必须分别对两个分支的 Train/Valid 做全量 Token 预检，`over_cutoff_count=0` 且 `target_truncated_count=0` 后才允许构建 Cache 和启动训练。Test 可以保留为后续评测语料，但不得注册进 Tokenized Cache 或模型更新。本阶段只完成 Final ID、全量 Messages、词表/Cache 代码与两个训练入口；扩词表、Cache、SFT 和端到端评测仍需下一最小闭环执行。

### 11.3 必须发布的核心产物

各阶段的具体物理格式由实现前的数据契约确定，但至少包含：

```text
config_resolved.yaml
manifest.json
category_mapping.parquet
query_category_depth.parquet
query_category_stats.parquet
query_poi_layer_edges.parquet
query_embeddings.*
query_adapter_exact.pt
poi_codebooks_s1_s2_s3.*
query_codebooks_s1_s2_s3.*
poi_assignments_s1_s2_s3.*
query_assignments_s1_s2_s3.*
category_distribution_s1.*
category_distribution_s1_s2.*
residual_diagnostics.*
hard_entity_edges.*
poi_sid.parquet
sid_bucket_index.parquet
static_metrics.json
prefix_probe_metrics.*
report.md
```

大型定长矩阵优先 float16/mmap NPY，centroid/聚合用 float32，assignment 用 int32，结构化表优先 Parquet。禁止对 71.6 万 active POI 构建全量两两矩阵；全量阶段必须分片、可 resume、原子提交，并记录输入/输出 schema、行数、SHA256、配置、seed、源码提交和 worktree 状态。

### 11.4 v2.1-CAT 配置

P4–P8 的不可变 base 配置为 `qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml`，新输出 namespace 为 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/`。该配置通过独立 schema 和 `src/qg_prqk/sid/pipeline_config.py` 继承冻结 active P3A-FULL/类别配置，校验上游 YAML 哈希，仅覆盖主码本容量和下游输出路径，显式登记 P2.5、D3 cache、FULL/FINAL manifest 与 FINAL checkpoint 路径/哈希。P6/P7 的历史 hard60 Gate 配置继续保留；P6 full 使用独立 direct-full-v2 配置，P7 full 使用独立 direct-full-v1 配置，两者都显式绑定已完成 sample 并记录跳过 medium。当前分别通过统一入口的 `build-relational-codebook`、`evaluate-relational-codebook`、`diagnose-relational-codebook`、`diagnose-relational-s2` 和 `build-local-codebook` 子命令执行；诊断不替代 canonical checkpoint。历史实验的旧脚本名和源码仍保存在相应冻结快照中。

已完成的 active P2.5、D3 Query cache、P3A-FULL 仍位于 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/`。这些上游产物不依赖 SID 容量，原配置、目录名、checkpoint 和 manifest 均原样只读保留，不改成 512、不重跑。旧全库配置、50k Gate、旧 FULL 局部产物与 P2 v1.1 同样保留。P2.5 的旧 loader 不支持新的下游 schema，不能用新 YAML 误启动 P2.5；后续阶段须显式使用新的严格配置入口。

---

## 12. 工程验收与风险保护

- 每个新阶段先通过 unit test、合成样例和 sample smoke；工程闭环通过后按已确认协议直接 full，方法指标和调参只依据 full。
- Query 边总质量按 Query 归一并 cap，避免头部 Query 主导；完全无 Query 的 POI 保持 content-only 路径。
- D1/D2 不得泄漏到 S3，Geo 不得泄漏到 S1/S2；逐层 residual 和 layer mask 必须有断言。
- 类别只进入 S1/S2 assignment 软代价，不进入 BGE，不硬映射为 SID；S2 类别统计必须条件化于 `(S1,S2)` 完整前缀。
- false-negative mask 必须在 Adapter 负样本、S3 hard edge 和相关诊断中生效。
- 若 Adapter、双视图或 Geo Gate 无提升，保留 identity/query-off/geo-off 回退并记录真实结果，不伪造成功。
- 静态唯一率不能替代可预测性；必须联合看 Prefix Probe 和最终 Exact/Bucket HR。
- 任何真实字段、阈值或运行资源存在不确定时，先记录审计证据并向用户确认，不自行改变冻结口径。
- P8-CAT 完成后必须设置 `NEXT_ACTION: HOLD_FOR_REVIEW`，不得自动进入 SFT。

---

## 13. Query 时间窗口扩展

12 天 Train 数据先完成 v2.1-CAT SID。30/60/90 天只作为 `HOLD_FOR_REVIEW` 后的后续数据规模实验：

1. 收集测试日前的完整聚合统计，不是每个 POI 随机取最近 1–3 条；
2. 计算 12/30/60/90 天的 Any/D1/D2/D3 POI 覆盖曲线；
3. 可增加时间衰减计数，但保留原始累计次数用于置信估计；
4. 所有窗口严格截止在正式 valid/test 之前；
5. 三个月数据不得与首轮方法变量同时改变。

完全无 Query 的 POI 继续使用 content-only RQ-KMeans + Geo-aware S3，不视为异常。

---

## 14. v2 -> v2.1-CAT 迁移结果

### 删除

- 将 POI embedding 与聚合 Query embedding 融合为单个向量后再做 RQ-KMeans；
- `Kish ESS -> per-POI query gate -> lambda_{p,l}`；
- D2 Family/Brand/Entity-Group 及名称启发式；
- 将类别拼入 BGE、与 POI embedding concat 或令 `S1=CategoryID`；
- S3 类别分类代价；
- P8 后自动进入 Final PID/Qwen SFT。

### 保留

- P2 全量 Query–POI 统计、D3 exact-core 和 false-negative mask；
- P3 Query-only Adapter 代码及迁移当时的 `CODE_ONLY` 状态（现已完成 FULL，见文档顶部）；
- BGE POI 主视图；
- RQ-KMeans / projection residual；
- 迁移当时的 1024×3（后续由用户于 2026-09-07 改为 512×3）；
- POI/Query 独立 residual view、双质心共享 token 和逐层 Query residual；
- D3 Exact-Core 与 false-negative mask；
- S3 局部地理与困难实体。

### 新增

- 全量 POI 粗/细类别结构锚点；
- D1 Coarse-Category、D2 Fine-Category、D3 Exact POI、D0；
- S1 `P(coarse_category|S1)` 软代价；
- S2 `P(fine_category|S1,S2)` 条件软代价；
- 类别与 Query 权重 warm-up/ramp；
- Query–POI 层级监督图；
- P8-CAT 的 A0/A4 静态比较与 `HOLD_FOR_REVIEW` 停止点。
