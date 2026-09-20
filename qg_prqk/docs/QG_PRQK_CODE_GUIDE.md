# QG-PRQK 方法与代码导读

本文档说明已经实现的 **v2.1-CAT** 数据流、源码职责和历史实验，对应 [v2.1-CAT 规范归档](methods/QG_PRQK_V2_1_CAT_SPEC.md)。2026-09-19 已确认但尚未实现的 [v3.0 主规范](methods/QG_PRQK_METHOD_SPEC.md)单独维护新算法、版本差异和模块改造计划；本文中的 Query 图/双 assignment 不代表 v3.0。实验细节以 `experiments/QG_PRQK.md` 为准，实时状态以 `experiments/QG_PRQK_IMPLEMENTATION_STATUS.md` 为准。

当前状态为 `DESIGN_APPROVED_NOT_IMPLEMENTED`，指 v3.0 文档已登记、代码未开始。既有 A0-GID 三轮 SFT、五集双解码、A0/A4 配对 Qwen 诊断和 A0 全量 Test 双解码已验收；全量 Test 约束 A4−A0 的 HR@1/HR@10/NDCG@10 仅为 `+0.0125/−0.0157/+0.0183pp`。已完成的模型和评测不因新方案重新标为待运行。完整来源见实验记录与 README。

## 1. 建议阅读顺序

第一次进入目录时，按下面顺序阅读即可：

1. `README.md`：目录入口、最短命令和当前状态；
2. 本文档：完整方法、实验进展和代码索引；
3. `methods/QG_PRQK_METHOD_SPEC.md`：v3.0 方案与实施步骤；复核已有代码和结果时另读 `methods/QG_PRQK_V2_1_CAT_SPEC.md`；
4. `experiments/QG_PRQK.md`：P5 以后每次正式实验的详细记录；
5. `experiments/QG_PRQK_IMPLEMENTATION_STATUS.md`：P0–P4 历史、资产哈希、失败恢复和当前下一步；
6. `QG_PRQK_CODEX_COMMANDS.md`：历史复核、恢复及长命令。

阅读当前代码时从 `scripts/qg_prqk.py` 进入，再看 `src/qg_prqk/cli.py` 的命令路由；不要从历史 P 编号猜文件名。历史配置、manifest 和输出目录仍保留 P3/P4/P5 等编号，是为了保持已完成实验可复核。

**码本配置入口不要混用：**当前 A0/A4 的基础配置在 `sid/pipeline_config.py`，常量 `CURRENT_SID_CODEBOOK_SIZES=(512,512,512)`，读取 `configs/qg_prqk_v2_1_category_active_512x3.yaml`。根模块 `config.py` 仅保留历史 v1.1 合同，常量已明确命名为 `LEGACY_V1_CODEBOOK_SIZES`；`load_legacy_config` 是历史加载器，`load_config` 只是兼容别名。当前下游继承冻结上游，再显式覆盖为 512×3，不修改历史 YAML、签名和模型。1024 维 BGE 与 1024 cutoff 均不是码本容量。

## 2. v2.1-CAT 已实现的方法全景

### 2.1 一句话定义

QG-PRQK 是一个类别层次锚定、Query 粒度感知、Query–POI 双视图的 Projection-Residual RQ-KMeans 方法：它先从 Train 订单中判断 Query 能可靠监督到哪一层，再用 Query 图和类别软代价重组 POI 的 `S1/S2`，最后用 Exact Query、局部地理与困难实体关系构建 `S3`。

### 2.2 数据流

```text
北京 active POI（716,245）
  ├─ 冻结 POI BGE-M3 [716245, 1024]
  ├─ 类别、名称、地址、经纬度
  └─ POI-only Projection-Residual RQ-KMeans
             │
Train orders（7,586,410；不读取业务 Validation/Test）
  └─ Query 归一化与 Query–POI 统计
       └─ D0 / D1 / D2 / D3 监督深度
            ├─ D1/D2：Raw BGE Query view
            └─ D3：BGE + Query Adapter
                     │
                     └─ Query–POI 分层图
                              │
POI-only A0 ──────────────────┘
  └─ Query 图 + S1 粗类软代价 + S2 细类软代价
       └─ 关系化 S1/S2
            ├─ GID-parent S3：parent=(GID6,S1,S2)
            └─ NoGID S3：parent=(S1,S2)
                    │
                    └─ 静态评测与三方可视化
                         └─ 末位确定性 Dedup，得到唯一 Final ID
                              └─ history10 Messages
                                   └─ 共享扩词表 + 分支 Tokenized Cache
                                        └─ Qwen3-0.6B LLaMA-Factory SFT（三轮完成）
                                             └─ 固定 10k + 四类泛化生成评测（两版十项完成）
```

### 2.3 冻结输入与边界

| 项目 | 当前合同 |
|---|---|
| POI 候选库 | 北京 active POI，716,245 行；用于保证后续评测目标可检索 |
| POI 表示 | 冻结 BGE-M3，1024 维，L2 归一化；不使用 E4/PCA |
| Query/SID 监督 | 只来自 Train；业务 Validation/Test 不参与统计、训练或选模 |
| POI 字段 | 使用 `poi_id/displayname/alias/category/category_code/address/lat/lng` 等实际存在字段 |
| 未使用字段 | `area/layer/click_score` 不进入当前方法 |
| 类别定义 | `category_code[0:2]` 为 19 个 coarse category；完整 `category_code` 为全局 402 个 fine category，active 实际出现 397 个 |
| SID 容量 | `512 × 512 × 512`，每层 token 为 0–511 |
| 代码边界 | 项目实现全部位于 `qg_prqk/`，运行时不导入根目录 `src/poi_gr` |
| 输出边界 | QG 新产物全部写入 `qg_prqk/outputs/`；根 `outputs/` 仅可作为冻结只读来源 |

active POI 的成员资格可以覆盖 Train/Validation/Test 中出现过的目标和历史 POI，因为它只定义可检索候选集合；Query 统计、Adapter、图、码本与选参仍严格 Train-only。

## 3. 各方法组件

### 3.1 Query 归一化与监督深度

Query 先做保守且确定性的 NFKC、英文小写、常见标点和空白归一化，再聚合 `n(q,p)`、`n(q)`、top1/top2、margin 和熵。P2 的高置信 Query 原样作为 D3 Exact-core；其他 Query 根据订单目标的类别分布依次判为 D2、D1 或 D0。

| 深度 | 语义 | 进入的 SID 层 | Query view |
|---|---|---|---|
| D0 | 上下文型或目标不稳定 | 不参与 SID 图监督 | 无 |
| D1 | Coarse-category Query | S1 | Raw BGE |
| D2 | Fine-category Query | S1、S2 | Raw BGE |
| D3 | Exact-core Query | S1、S2、S3 | Final adapted BGE |

类别只提供 assignment 的软代价，不拼接到 BGE，也不把类别 ID 硬编码成 S1/S2。D3 的合理多 POI 目标进入 false-negative mask，不被当成 Adapter hard negative。

### 3.2 D3 Query Adapter

Adapter 只改变 Query view，POI BGE 始终冻结。结构为：

```text
LayerNorm
  -> Linear(1024, 64)
  -> GELU
  -> Dropout(0.05)
  -> Linear(64, 1024)
  -> residual add
  -> L2 normalize
```

输出层零初始化，因此初始行为是 identity。训练使用 weighted、false-negative-masked InfoNCE；固定负例来自 semantic ANN、lexical metadata 和 local geo，另有动态 in-batch negatives。FULL-SELECT 在未进入旧 50k Gate 的 Train-only D3 中固定抽取 20,000 条 holdout，按 Recall@10、Recall@1、困难 Recall@10 选 epoch；FULL-FINAL 再从同一初始状态用全部 291,590 条 D3 固定重训 best epoch。

### 3.3 A0：POI-only 基础码本

A0 不使用 Query、类别或 Geo，只在冻结 active POI BGE 上执行三层 spherical/cosine Projection-Residual RQ-KMeans：

1. 在 POI 全库估计并去除公共方向，再对 POI/Query 使用同一变换；
2. 每层用 cosine K-Means 做 hard assignment；
3. 使用 Top-k=5、beta=15 的 5 轮质心细化作为码均衡正则；
4. 用 projection residual，而不是普通 `residual -= centroid`；
5. 三层 hard fit 最大 60 轮，码本大小均为 512。

A0 是完整方法的初始化和静态内部对照，不是外部业务基线。

### 3.4 A4 的 S1/S2：关系化双视图码本

POI residual 和 Query residual 各自维护质心，但同一层共享离散 token index。训练在内容几何、Query distortion、Query–POI 图一致性和类别软代价之间交替优化：

| 层 | Query distortion | 图对齐 | 类别软代价 | 类别条件 |
|---|---:|---:|---:|---|
| S1 | 0.05 | 0.05 | 0.08 | `P(coarse_category | S1)` |
| S2 | 0.10 | 0.10 | 0.08 | `P(fine_category | S1,S2)` |

类别前两轮关闭，第 3–6 轮线性 ramp；Query/graph 权重也按冻结 warm-up 逐步打开。Query 质心使用 `tau=32` 向 POI 质心收缩，降低稀疏 Query code 的不稳定性。D1 在 S1 后停止，D2 在 S2 后停止，只有 D3 residual 继续进入 S3。

### 3.5 A4 的 S3：两种 parent

两种 S3 都冻结正式 S1/S2，使用 D3 Query residual、Geo 连续特征、困难实体图、Top-k5 和有界局部细化。S3 的 Query/graph/Geo/hard 权重为 `0.20/0.20/0.10/0.05`，类别不再作为分类代价，只用于困难候选筛选。

| 变体 | parent | Geo 形式 | 困难图 | 三层 SID |
|---|---|---|---|---|
| A4 GID-parent | `(GID6,S1,S2)` | GID6 cell 和 parent 内局部相对位置 | 同 parent、同 fine category | `[S1,S2,S3]` |
| A4 NoGID | `(S1,S2)` | 仅相对 S1/S2 parent 中心，不计算 GID | 同 S1/S2 parent、同 fine category | `[S1,S2,S3]` |

GID-parent 困难边的 BGE/name/category/address/geo 权重为 `0.35/0.30/0.15/0.10/0.10`，阈值 0.60、每 POI Top-20；NoGID 移除恒定的 same-GID 0.10 后使用等价语义阈值 0.50。

### 3.6 静态评测

静态评测同时看四类信息，不能只凭某一个数宣布方法成功：

- 码本利用与碰撞：active codes、ESS、Gini、distinct SID、collision excess、最大桶；
- 类别结构：S1 coarse purity/NMI、S1+S2 fine purity/NMI；
- 图上训练对齐：分 D1/D2/D3 的 Query–POI token agreement；
- 脱离图目标的可预测性：teacher-forced 和 autoregressive content-only Prefix Probe。

这里的 Prefix Probe 不是 Qwen SFT，也不是业务 Validation/Test 检索。它只用冻结 Query view 对当前码本做最近码预测，用来发现“图一致性提高但 Query 自身难以预测 token”的风险。

新增的 GenPOI 风格可视化通过 `visualize-sid-categories` 单独运行：三版同五类、每类 200 个相同 POI；三层 POI 码向量分别归一化后等权串联，联合 PCA50+t-SNE，类别仅用于着色。另沿同五个固定锚点展示具体 S1/S1S2/S1S2S3 前缀的完整桶类别与 Geohash5 网格分布，同时输出全库 POI 加权纯度及剔除单例后的纯度。不改 SID，不读业务评测集，旧版 residual/热门 code 可视化保持原样。

### 3.7 Final ID 与 SFT

SID 本身不强制唯一。所有 SID 完成后，才在完整 Base ID 的碰撞桶内按 `poi_id` 字典序分配零基 Dedup token；单例不加 D，碰撞项的 D 永远位于最后。

| SFT 分支 | Base ID | Final ID | 请求侧位置 |
|---|---|---|---|
| GID-parent | `G1..G6,S1,S2,S3` | `G1..G6,S1,S2,S3,[D]` | 保留 `<USER_GID>` |
| NoGID | `S1,S2,S3` | `S1,S2,S3,[D]` | 仍保留 `<USER_GID>` |

两分支从同一 history10 语料逐行只替换 history/target identifier，共用一个新增 3,743 个普通原子 Token 的 Qwen3-0.6B 初始模型。训练协议固定为 LLaMA-Factory full SFT、4 GPU、3 epoch、global batch 512、LR 5e-5、BF16、cutoff 1024、packing、`train_on_prompt=false`。

2026-09-15 用户新增 A0-GID 对照：直接复用冻结 A0 三位 SID，前置与 A4 相同的 GID6，并重新分配可选末位 Dedup。SFT 沿用同一扩词初始模型和训练配置，仅替换历史/目标 identifier。`a0_gid_sft.py` 把全量准备搬到平台 CPU，在完整门禁通过后调用相同 LLaMA-Factory 四卡后端；不重新拟合量化器、不使用 A4 已训练 checkpoint，也不在准备阶段读取 Test。首次准备、已完成阶段复用与实际训练是三个不同状态，`--dry-run` 只表示小型合同检查通过。

## 4. 当前进展与结论

### 4.1 阶段进展

| 历史阶段 | 当前状态 | 核心结果 | 当前代码 |
|---|---|---|---|
| P0/P1 | 完成 | 数据、配置、哈希、输出和自包含代码边界已冻结 | `config.py`、`data/contracts.py`、`artifacts.py` |
| P2 | 完成 | 7,586,410 Train；1,372,661 normalized Query；2,022,443 Query–POI pair；291,590 D3 | `data/query_normalization.py`、`data/query_statistics.py` |
| P2.5-CAT active | 完成 | D0/D1/D2/D3=`1,029,782/25,639/25,650/291,590`；层级边 1,204,570 | `data/query_supervision*.py` |
| P3A 50k Gate | 完成 | Raw→Adapter R@10 `66.00%→71.24%`；困难 R@10 `45.51%→54.10%` | `adapters/gate.py` |
| P3A-FULL | 完成 | 20k holdout 上 SELECT epoch 2 的 R@1/R@10/困难 R@10=`53.955/77.100/55.5393%` | `adapters/selection.py` |
| P4 | 完成 | 342,879 Query、1,204,570 分层边、376,701 FN；D1/D2 raw、D3 adapted | `data/query_graph*.py` |
| P5/A0 | 完成 | 716,245 POI；三层 512 码全激活；distinct SID 606,045，最大桶 208 | `sid/base_*.py` |
| P6 | 完成并保留评审结论 | 图与类别结构显著改善；S2 Query distortion `+0.028162`，诊断定位为 graph-coupled assignment 的多目标代价 | `sid/relational_*.py` |
| P7 GID-parent | 完成 | distinct SID 617,887，最大桶 95；S3 加权图一致率 85.2967% | `sid/local_*.py`、`sid/geo.py` |
| P8 A0/A4 | 完成并保留评审结论 | A4 类别纯度、碰撞和局部分离更好，但 content-only cumulative Prefix Probe `12.7183%→7.7743%` | `sid/evaluation*.py` |
| NoGID P7/P8 | 完成并保留评审结论 | distinct 613,198、最大桶 159；同口径 S3/cumulative probe 略低于 GID-parent | `sid/nogid_*.py` |
| SID 可视化 | 完成 | 三方法无 code collapse；前缀加深时 BGE 语义相似度单调增加；A4 类别组织更强 | `sid/visualization*.py` |
| 同五类/类别区域可视化 | 完成 | 同 1,000 POI 联合 t-SNE、45 个具体前缀全桶组成；全库与剔除单例后的纯度均留存，两种 A4 图形接近 | `sid/category_region_visualization.py`、`sid/category_region_plots.py` |
| A0→A4 成员迁移 | 完成 | 全库精确变化及 18 个提高/持平/降低展示项；类别组织加强，但有新碰撞与标签语义不一致案例，不代表 Query 检索收益 | `sid/migration_analysis.py`、`sid/migration_plots.py` |
| P9 | 完成 | 两种 Final ID 均覆盖且唯一 716,245 POI；两套 Messages 各 8,790,513 行 | `sid/identifiers.py`、`sft/data.py` |
| P10 | 完成 | 共享词表 3,743 Token；两套 Train/Valid Cache 均零超长、零 target truncation | `sft/vocabulary.py`、`sft/preflight.py` |
| 双分支 SFT | 完成 | 4×A100，3 epoch；最终 step 8949/7404，两个退出码均为 0 | `sft/training.py`、`configs/sft/`、`launchers/` |
| 双分支 Validation 生成评测 | 十项完成 | GID / NoGID 固定 HR@10 为 85.19% / 84.67%，泛化宏平均为 46.6900% / 46.3325% | `sft/evaluation*.py`、`commands/evaluate_sft.py` |
| 双分支全量 Test | 脚本就绪、CPU 预检通过，未正式推理 | 最后一天各 606,682 条，复用同一 epoch-3 模型和 Beam=10 | `sft/full_test_data.py`、`sft/full_test_evaluation.py`、`commands/evaluate_sft_test.py` |
| A0-GID 四卡对照 | 三轮训练完成 | 同一 A4-GID 输入语料和初始模型；最终 `checkpoint-8955`，Validation loss `0.198839` | `sft/a0_gid_data.py`、`sft/a0_gid_pipeline.py` |

### 4.2 对“方法是否成功”的准确判断

- Query Adapter 已在固定 Train-only internal holdout 上稳定优于 Raw BGE，连续 Query view 学习是有效的。
- A4 SID 已完整构建，三层码本无坍塌，类别组织、distinct SID、最大桶和局部分离优于 A0，因此“结构化 SID 构建”是成功的。
- A4 的 content-only Prefix Probe 低于 A0，说明图监督得到的 token 不一定能由 Query embedding 的简单最近码规则直接预测；这是已记录的多目标权衡，不是 residual 数值退化。
- NoGID 是成立的受控分支，但当前静态证据不支持它替换 GID-parent；它被保留到 SFT 是为了做最终生成学习能力对比。
- 两个 SFT 与同协议 Validation 已完成：GID 整体数值略高，NoGID 冷目标更好；尚不能归因于单一模块或声称超过所有基线。下一项是用户确认的最后一天全量 Test，不据 Test 调整模型或选参。

## 5. 已编号实验与当前代码对应

P0–P4 中大量工作属于工程阶段或数据资产构建，没有分配正式效果实验号；其完整过程在实现状态文档第 12 节。下表汇总所有已经编号的 QG 运行记录。历史运行的逐字节复现必须使用对应 `outputs/run_control/**/source_snapshot/`，当前语义模块只表示同一职责在整理后位于哪里。

| 实验 | 状态 | 结果摘要 | 当前实现位置 |
|---|---|---|---|
| `EXP-20260903-01` | COMPLETED | 旧 2,337,178 POI 闭集的 50k D3 Gate 通过；不是 full | `adapters/{data,cache,gate,model,retrieval}.py` |
| `EXP-20260904-04` | INTERRUPTED，后完成 cache 恢复 | 首次 active FULL 在 163,840/291,590 Query 后中断，无训练/checkpoint；随后恢复为完整 291,590×1024 cache | `adapters/cache.py`、`data/query_embeddings.py` |
| `EXP-20260905-02` | COMPLETED | active FULL 选中 epoch 2，并从共同初始状态用全部 D3 重训 FINAL | `adapters/{selection,selection_config}.py` |
| `EXP-20260907-01` | COMPLETED | P5 10k sample 工程 Gate 通过 | `sid/{base_config,base_data,base_quantizer}.py` |
| `EXP-20260907-02` | REVIEW_REQUIRED | P5 100k hard30 未收敛；Top-k 提高均衡但恶化 hard distortion | `sid/base_quantizer.py` |
| `EXP-20260907-03` | REVIEW_REQUIRED | hard60-off 收敛，确认需要分离 hard endpoint 与 Top-k 作用 | `sid/base_diagnostics.py` |
| `EXP-20260908-03` | COMPLETED | 同 hard60 endpoint 配对验证 Top-k5：牺牲 distortion、改善 ESS/Gini/唯一数 | `sid/base_topk_diagnostics.py` |
| `EXP-20260908-04` | COMPLETED | 新 hard60+Top-k5 协议 10k smoke 通过 | `sid/base_quantizer.py` |
| `EXP-20260909-01` | COMPLETED | P5 active full 三层在 57/52/57 轮收敛，512 码全激活 | `sid/base_quantizer.py` |
| `EXP-20260909-02` | FAILED | P6 首次 sample 把 warm-up 前后不同目标直接比较，保护性停止且未发布成功产物 | `sid/relational_quantizer.py` |
| `EXP-20260909-03` | REVIEW_REQUIRED | 修正收敛判定后 sample 完成；结构改善、Query distortion 退化 | `sid/relational_quantizer.py`、`sid/relational_evaluation.py` |
| `EXP-20260909-04` | REVIEW_REQUIRED | sample 六分支归因：稀疏样本中 `tau=32` POI prior 为主、graph 次之 | `sid/relational_attribution.py` |
| `EXP-20260909-05` | REVIEW_REQUIRED | P6 full 完成；S1 distortion 微降、S2 上升 0.028162，图/类别显著改善 | `sid/relational_quantizer.py`、`sid/relational_evaluation.py` |
| `EXP-20260910-01` | REVIEW_REQUIRED | full S2 诊断确认主要代价来自 graph-coupled assignment，正式参数未改 | `sid/relational_s2_diagnostics.py` |
| `EXP-20260911-01` | COMPLETED | GID-parent S3 full 完成，distinct/max bucket=`617,887/95` | `sid/{local_config,local_data,geo,local_quantizer}.py` |
| `EXP-20260911-02` | REVIEW_REQUIRED | A4 静态结构优于 A0，但三层 content-only Prefix Probe 均下降 | `sid/{evaluation_config,evaluation_data,evaluation}.py` |
| `EXP-20260911-03` | REVIEW_REQUIRED | NoGID full 与三方可视化完成；NoGID 未优于 GID-parent | `sid/nogid_*.py`、`sid/visualization*.py` |
| `EXP-20260911-04` | COMPLETED | 双 Final ID 和双 8,790,513 行 Messages 完成 | `sid/identifiers.py`、`sft/data.py` |
| `EXP-20260912-01` | COMPLETED | 共享扩词表、双 Train/Valid 1024 Tokenized Cache 和 launcher dry-run 完成；未训练 | `sft/{vocabulary,preflight,training}.py` |
| `EXP-20260913-01` | COMPLETED | 双 4×A100 三轮 SFT 完成；最终 step 8949/7404，Valid loss 0.200681/0.339035 | `sft/training.py`；下一步 `sft/evaluation*.py` |
| `EXP-20260914-01` | COMPLETED | 双分支五组 Validation 十项完成；固定 HR@10 为 85.19% / 84.67%，冷目标 NoGID 更高 | `sft/evaluation*.py` |

`EXP-20260904-04` 是文档整理时对重复编号的修正：原 QG 记录误用了仓库已经分配给“active MMBERT Recall 128”的当日 03 号。该修正只保证仓库实验编号唯一，不改变 QG 运行时间、目录、配置、哈希或结论。

## 6. 源码逐文件说明

本节路径均相对于 `qg_prqk/`。当前 `src/qg_prqk/` 共 107 个 Python 文件，其中命令适配层 33 个；下面逐项覆盖全部实现文件。

### 6.1 入口与公共基础

| 文件 | 功能 |
|---|---|
| `scripts/qg_prqk.py` | 方法主链路的统一脚本入口；只把 `qg_prqk/src` 加入导入路径并调用统一 CLI |
| `scripts/visualize_sid_collisions.py` | 碰撞可视化的薄入口，仅调用 `sid/collision_visualization.py`；独立保留是为了不改变 V2 图已冻结的 CLI 来源哈希 |
| `scripts/analyze_sid_migration.py` | A0→两种 A4 迁移分析薄入口，只调用 `sid/migration_analysis.py`；同样不改已有图绑定的 CLI 哈希 |
| `scripts/a0_gid_sft.py` | A0-GID 平台准备/训练薄入口，支持 plan、prepare、hardware、train；不改已冻结的 A4 CLI 与评测来源指纹 |
| `scripts/evaluate_sft_constrained.py` | 双卡 GID/NoGID 约束解码薄入口，不改已冻结的统一 CLI；CPU 预检、自动 GPU smoke、十项评测与汇总 |
| `scripts/evaluate_a0_gid_sft.py` | POI-only PRQ-KMeans GID-first 的独立评测入口；`sft/a0_evaluation.py` 负责五组固定样本对齐、A0 全目录索引、单卡 A100 串行或双卡兼容调度与断点；复用已有生成/计分/Trie，不修改冻结 A4 入口 |
| `src/qg_prqk/__main__.py` | 支持设置 `PYTHONPATH` 后执行 `python -m qg_prqk` |
| `src/qg_prqk/cli.py` | 按领域注册并延迟加载所有子命令；不实现算法 |
| `src/qg_prqk/config.py` | 历史 v1.1 1024×3 合同及共享数据类；`load_legacy_config` / 兼容别名 `load_config`，非当前 SID 配置入口 |
| `src/qg_prqk/artifacts.py` | SHA256、UTC 时间、配置签名、原子 JSON/manifest 写入等公共产物逻辑 |
| `src/qg_prqk/hard_negative_mining.py` | Adapter 的 in-batch、semantic ANN、词法元数据、local-geo 负例生成与 FN 排除 |
| `src/qg_prqk/__init__.py` | 声明顶层 QG-PRQK Python 包 |
| `src/qg_prqk/data/__init__.py` | 声明数据准备包边界，不承载阶段逻辑 |
| `src/qg_prqk/adapters/__init__.py` | 声明 Query Adapter 包边界，不承载阶段逻辑 |
| `src/qg_prqk/sid/__init__.py` | 声明 Semantic ID 包边界，不承载阶段逻辑 |
| `src/qg_prqk/sft/__init__.py` | 声明 SFT 包边界，不承载阶段逻辑 |
| `src/qg_prqk/commands/__init__.py` | 声明 CLI command handler 包边界，不承载算法 |

### 6.2 `data/`：输入与 Query 数据

| 文件 | 功能 |
|---|---|
| `data/contracts.py` | 校验 POI、订单、split、Embedding 和 Final ID 的基础字段合同 |
| `data/active_poi.py` | 按冻结 active POI 行序，从全库 BGE/类别资产派生 716,245 行 QG 输入并逐值校验 |
| `data/query_normalization.py` | 保守、确定性的 Query 文本归一化 |
| `data/query_statistics.py` | 流式扫描 Train，分 shard 聚合 Query–POI 统计、D3、高歧义过滤和 false-negative mask |
| `data/query_supervision_config.py` | P2.5 类别资产、阈值、路径、敏感性设置及配置哈希合同 |
| `data/query_supervision.py` | 计算 D0/D1/D2/D3、层级可靠度、类别 mapping、分层 Query–POI 边并发布报告 |
| `data/query_embeddings.py` | 通用 Query BGE 编码、float16/L2 NPY、resume、manifest 和 validator |
| `data/query_graph_data.py` | P4 图节点/边/FN 的 schema、严格 loader、前缀复用与有界读取合同 |
| `data/query_graph.py` | 组装 D1/D2 Raw 与 D3 Final Adapter view，构建 sample/medium/full Query 图和恢复产物 |

### 6.3 `adapters/`：D3 Query Adapter

| 文件 | 功能 |
|---|---|
| `adapters/config.py` | 50k Gate 的 BGE、精确检索、Adapter 与评测配置合同 |
| `adapters/selection_config.py` | FULL 的 20k holdout、3 epoch 选模、共同初始化和 FINAL 重训合同 |
| `adapters/data.py` | 从 P2.5 选择 D3、稳定切分、映射 active POI、读取 POI metadata 与 FN |
| `adapters/cache.py` | D3 Query embedding 的原子分块、断点续跑、已验证前缀恢复和最终合并 |
| `adapters/model.py` | ResidualQueryAdapter、identity 初始化、masked weighted InfoNCE 和检索指标 |
| `adapters/retrieval.py` | FAISS `IndexFlatIP` 全库精确 cosine 检索及 Recall/MRR/困难子集评测 |
| `adapters/training.py` | 从预构建 tensor bundle 训练、验证并保存单个 Adapter |
| `adapters/gate.py` | 编排历史 50k D3 工程 Gate：选样、编码、负例、训练、raw/adapted 对比和发布 |
| `adapters/selection.py` | 编排 active FULL-SELECT、逐 epoch holdout 全库检索、best epoch 与 FULL-FINAL |

### 6.4 `sid/`：Semantic ID 主体

| 文件 | 功能 |
|---|---|
| `sid/pipeline_config.py` | 当前 512×3 SID 基础配置入口；`CURRENT_SID_CODEBOOK_SIZES` 与 `load_downstream_config`，继承 active P3A 并锁定上游哈希 |
| `sid/base_config.py` | P5 hard60+Top-k5 执行配置和参数选择证据校验 |
| `sid/base_data.py` | A0 输入、sample/medium/full 确定性选样、公共方向和输出路径 |
| `sid/base_quantizer.py` | POI-only spherical/cosine PRQ-KMeans、Top-k 细化、projection residual、恢复和验证 |
| `sid/base_diagnostics.py` | hard30/hard60 endpoint 受控诊断，不修改 canonical 产物 |
| `sid/base_topk_diagnostics.py` | 从逐值相同 hard60 endpoint 分叉 Top-k-off/Top-k5 并做配对比较 |
| `sid/relational_config.py` | P6 S1/S2 的图、类别、warm-up、收敛与 full Gate 配置合同 |
| `sid/relational_data.py` | 加载 POI/Query 双视图和图闭包，保证选中 Query 的目标 POI 全部在集合中 |
| `sid/relational_quantizer.py` | 双质心共享 token 的 S1/S2 交替优化、类别软代价和 projection residual |
| `sid/relational_evaluation.py` | P5 A0 与 P6 S1/S2 的公平闭包对比 |
| `sid/relational_attribution.py` | sample 上隔离 graph、prior、Top-k、category 的六分支归因 |
| `sid/relational_s2_diagnostics.py` | 冻结 full S1 endpoint，对 S2 residual、质心适配和 coupled assignment 做分解 |
| `sid/geo.py` | longitude-first Geohash、GID prefix、北京局部米制坐标、五维 Geo 和 robust 标准化 |
| `sid/local_config.py` | GID-parent S3 的 parent、困难图、Geo、局部细化与 full 配置合同 |
| `sid/local_data.py` | 加载冻结 S1/S2、D3 residual、POI 元数据，构建 `(GID6,S1,S2)` 困难图 |
| `sid/local_quantizer.py` | GID-parent S3 的双视图/Geo/hard 目标、Top-k5、有界局部细化与验证 |
| `sid/nogid_config.py` | NoGID S3 的 `(S1,S2)` parent 和禁止 GID 合同 |
| `sid/nogid_geo.py` | 不使用 GID 的 `(S1,S2)` parent key 与连续相对 Geo 特征 |
| `sid/nogid_data.py` | 构建 NoGID 输入闭包和 `(S1,S2,fine category)` 困难图 |
| `sid/nogid_quantizer.py` | 训练、发布和验证 NoGID S3 |
| `sid/evaluation_config.py` | A0/A4 canonical 静态评测配置与来源锁定 |
| `sid/evaluation_data.py` | 统一加载 A0/A4、P2.5、P4 和类别资产 |
| `sid/evaluation.py` | 码本/碰撞/类别/对齐/Prefix Probe/局部分离评测，并发布 canonical A4 SID |
| `sid/nogid_evaluation_config.py` | A0/GID-parent/NoGID 的公平三方比较合同 |
| `sid/nogid_evaluation.py` | 使用统一 `(S1,S2)` 候选口径比较三种三位 SID |
| `sid/identifiers.py` | 构造两种 Base ID，按完整碰撞桶追加末位 D，并发布唯一 POI mapping |
| `sid/visualization_config.py` | 三方可视化的来源、方法顺序、抽样和降维配置 |
| `sid/visualization.py` | 码分布、转移、residual t-SNE、exact-prefix 语义和可读样例；只读不改 SID |
| `sid/category_region_visualization.py` | 共同五类确定性抽样、选中码向量等权串联、联合 PCA/t-SNE、完整前缀桶类别/Geohash5 分布、全库与非单例纯度、来源哈希与发布核验 |
| `sid/category_region_plots.py` | 只消费已算好的坐标与计数，绘制三版同类 t-SNE 及 3×3 前缀组成图，输出 PNG/PDF；不按类别调整坐标 |
| `sid/collision_visualization.py` | 先筛选三版完整 SID 桶均为 3–20 个 POI 的共同锚点，再复用前缀计数/条形绘制；末层全部为真实碰撞，保留全部 POI 名称、类别、网格和坐标，不重算 t-SNE。入口为 `scripts/visualize_sid_collisions.py`，不改旧 V2 CLI 哈希 |
| `sid/migration_analysis.py` | 全库逐层统计 A0→A4 编码变化、同伴集合交并、单例/碰撞迁移；比较剔除自身的同类别/同网格比例，按提高/持平/降低确定性抽取案例。JSON 有界预览，NPZ 保存全部成员行号/ID，manifest 与只读验证绑定来源；不重训 |
| `sid/migration_plots.py` | 只消费迁移统计与案例，画全库概览、两版 3×3 保留/移出/移入图，并输出转义后的成员 HTML；不调整案例或 SID |

### 6.5 `sft/`：生成模型数据与训练接入

| 文件 | 功能 |
|---|---|
| `sft/data.py` | 从冻结 TIGER history10 反查 POI，一次扫描成对生成 GID-parent/NoGID Messages |
| `sft/vocabulary.py` | 校验两份 Token 清单同序，复制基础 Qwen 并生成共享扩词表模型 |
| `sft/preflight.py` | 只扫描 Train/Valid 的格式化长度，要求零超长/零 target 截断后构建不可覆盖 Cache |
| `sft/training.py` | 联检数据、共同模型、cache、cutoff、global batch 和输出路径，再调用 LLaMA-Factory |
| `sft/constrained_decoding.py` | 排序全目录终态路径与有界前缀缓存，精确约束可选 Dedup、目标边界和 EOS；model.generate 包装只注入 prefix callback，不改变原生成评分函数 |
| `sft/constrained_evaluation.py` | 继承冻结无约束计划与十组样本，核验配对、运行独立 chunk 断点与两卡 worker、转发平台日志，汇总约束指标及无约束差值；不读 Test |
| `sft/a0_gid_data.py` | 读取 A0 SID 与同行 A4 GID6，复用去重/序列化函数但发布 `a0_gid` 来源；流式替换 A4 Train/Valid 的历史和目标 ID，保持业务内容/顺序。保存 manifest、SHA256 和成功标记，已完成阶段快速校验复用，不读取 Test |
| `sft/a0_gid_pipeline.py` | 核对 A0/A4 训练配置仅有路径/identifier 差异，锁住平台单进程准备，复用既有 tokenizer/全量长度预检/packed Cache 构建器；缓存验收后启动相同 LLaMA-Factory 四卡 SFT。只按 CUDA/BF16/显存检查硬件，不按型号名称限制 |
| `sft/evaluation_data.py` | 核验冻结参考集/manifest，按业务主键一次扫描对齐五组 Validation，检查目标、CURRENT 请求与历史长度，不重新抽样 |
| `sft/evaluation.py` | 核验 epoch-3 权重/词表，建立全库 Final ID 精确查表，按训练模板编码并无约束生成，累计 HR/NDCG/MRR/合法率，支持 chunk 续跑与 OOM 降 batch |
| `sft/evaluation_suite.py` | CPU 准备双卡计划与样例预检，两个分支交错进入同一十项队列，最多两个单卡子进程并发；失败停止本套件子进程并保留断点。分别汇总每分支五组结果，固定 10k 不进入泛化宏平均 |
| `sft/full_test_data.py` | 从冻结 manifest 确认最后一天 Test，成对流式核验两版 606,682 行/哈希/业务键/当前请求；用 uint64 字节偏移惰性读取原 JSONL，不复制全量数据 |
| `sft/full_test_evaluation.py` | 冻结 epoch-3 权重，复用既有 QG 模板/Final ID/指标，执行全量 Test 分块推理、断点恢复、双卡各一分支调度、主日志转发和最终两版汇总；不改 Validation 源码指纹 |

评测数据流：已有五组参考样本的业务主键 → 各分支原始 `valid.jsonl` → 五组同序 Messages → 原 checkpoint tokenizer 与 `qwen3_nothink` → Beam=10 完整 Final ID → 716,245 POI 映射 → 精确 POI 指标。历史 GenPOI 参考集未提供辅助 `user_token` 字段，不把它强制作为连接键；有该字段时核对它，始终核对 Query/请求 GID 所在 CURRENT 块、history_length 与目标 POI。QG Messages 不会被参考集旧 ID 覆盖。

### 6.6 `commands/`：CLI 适配层

`commands/` 只负责参数解析、加载配置、调用领域函数并打印结果；算法不应写在这里。完整路由如下：

| 子命令 | command 文件 | 核心实现 |
|---|---|---|
| `preflight` | `commands/preflight.py` | `config.py`、`data/contracts.py` |
| `build-active-poi` | `commands/active_poi.py` | `data/active_poi.py` |
| `build-query-stats` | `commands/query_statistics.py` | `data/query_statistics.py` |
| `build-query-supervision` | `commands/query_supervision.py` | `data/query_supervision*.py` |
| `cache-query-embeddings` | `commands/query_embeddings.py` | `data/query_embeddings.py` |
| `build-query-graph` | `commands/query_graph.py` | `data/query_graph*.py` |
| `smoke-query-adapter` | `commands/smoke_query_adapter.py` | `adapters/model.py` |
| `train-query-adapter` | `commands/train_query_adapter.py` | `adapters/training.py` |
| `gate-query-adapter` | `commands/query_adapter_gate.py` | `adapters/gate.py` |
| `select-query-adapter` | `commands/select_query_adapter.py` | `adapters/selection.py` |
| `inspect-sid-config` | `commands/inspect_sid_config.py` | `sid/pipeline_config.py` |
| `inspect-relational-inputs` | `commands/inspect_relational_inputs.py` | `sid/relational_data.py` |
| `build-base-codebook` | `commands/base_codebook.py` | `sid/base_quantizer.py` |
| `build-relational-codebook` | `commands/relational_codebook.py` | `sid/relational_quantizer.py` |
| `evaluate-relational-codebook` | `commands/evaluate_relational_codebook.py` | `sid/relational_evaluation.py` |
| `build-local-codebook` | `commands/local_codebook.py` | `sid/local_quantizer.py` |
| `build-local-codebook-nogid` | `commands/local_codebook_nogid.py` | `sid/nogid_quantizer.py` |
| `evaluate-sid` | `commands/evaluate_sid.py` | `sid/evaluation.py` |
| `evaluate-sid-nogid` | `commands/evaluate_sid_nogid.py` | `sid/nogid_evaluation.py` |
| `build-final-identifiers` | `commands/final_identifiers.py` | `sid/identifiers.py` |
| `visualize-sid` | `commands/visualize_sid.py` | `sid/visualization.py` |
| `visualize-sid-categories` | `commands/visualize_sid_categories.py` | `sid/category_region_visualization.py`；构建/输入预检/已有输出复核 |
| `build-sft-data` | `commands/sft_data.py` | `sft/data.py` |
| `prepare-sft-vocab` | `commands/sft_vocabulary.py` | `sft/vocabulary.py` |
| `prepare-sft-cache` | `commands/sft_cache.py` | `sft/preflight.py` |
| `train-sft` | `commands/train_sft.py` | `sft/training.py` |
| `evaluate-sft` | `commands/evaluate_sft.py` | `sft/evaluation_data.py`、`sft/evaluation.py`、`sft/evaluation_suite.py` |
| `evaluate-sft-test` | `commands/evaluate_sft_test.py` | `sft/full_test_data.py`、`sft/full_test_evaluation.py`；独立 Test 入口，不放宽 Validation-only 命令 |
| `diagnose-base-codebook` | `commands/diagnose_base_codebook.py` | `sid/base_diagnostics.py` |
| `diagnose-base-topk` | `commands/diagnose_base_topk.py` | `sid/base_topk_diagnostics.py` |
| `diagnose-relational-codebook` | `commands/diagnose_relational_codebook.py` | `sid/relational_attribution.py` |
| `diagnose-relational-s2` | `commands/diagnose_relational_s2.py` | `sid/relational_s2_diagnostics.py` |

## 7. 配置文件说明

冻结配置文件名保留历史阶段号，不能为了美观改名，否则会破坏 manifest 里的路径和哈希引用。

| 配置 | 用途 | 当前角色 |
|---|---|---|
| `qg_prqk_1024x3.yaml` | P1–P3 初始方法、数据与 Adapter 参数 | 历史基础合同 |
| `qg_prqk_active_poi_assets_v1.yaml` | 从全库派生 active BGE/category | 已完成，只读 |
| `qg_prqk_v2_1_category_1024x3.yaml` | 历史全库 P2.5-CAT | 已完成历史版本 |
| `qg_prqk_v2_1_category_active_1024x3.yaml` | active P2.5-CAT | 已完成的上游来源 |
| `qg_prqk_p3a_1024x3_v1.yaml` | 历史全库 50k Adapter Gate | 已完成，保留 |
| `qg_prqk_p3a_active_1024x3_v1.yaml` | active Adapter 基础协议 | FULL 的 base 配置 |
| `qg_prqk_p3a_full_1024x3_v1.yaml` | 历史非 active FULL 协议 | 历史保留 |
| `qg_prqk_p3a_full_active_1024x3_v1.yaml` | active 20k holdout、SELECT/FINAL | 已完成，D3 默认 Query view 来源 |
| `qg_prqk_v2_1_category_active_512x3.yaml` | 继承 active 上游并把下游 SID 改为 512×3 | 当前下游基础合同 |
| `qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml` | A0 hard60+Top-k5 | 已完成 canonical A0 |
| `qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml` | 旧 sample/medium Gate 顺序 | 历史 sample/诊断只读 |
| `qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml` | P6 sample→full、跳过 medium | 已完成 canonical S1/S2 |
| `qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml` | GID-parent S3 | 已完成 canonical S3 |
| `qg_prqk_p7_nogid_s1s2_parent_hard60_topk5_direct_full_v1.yaml` | NoGID S3 | 已完成受控分支 |
| `qg_prqk_p8_a0_vs_a4_static_full_v1.yaml` | A0/A4 静态评测与发布 | 已完成，REVIEW_REQUIRED |
| `qg_prqk_p8_nogid_s1s2_parent_comparison_full_v1.yaml` | 三种 SID 公平比较 | 已完成，REVIEW_REQUIRED |
| `qg_prqk_sid_visualization_a0_a4_nogid_full_v1.yaml` | 三方 SID 可视化 | 已完成，只读 |
| `qg_prqk_sid_category_region_v2.yaml` | 通过 SHA256 引用旧配置的冻结来源；固定五类、抽样、等权码向量、Geohash5 和联合投影 | GenPOI 风格展示，旧图不覆盖 |
| `sft/a4_gid_parent_history10_v1.yaml` | GID-parent 四卡三轮 SFT | 已完成，checkpoint-8949 |
| `sft/a4_nogid_history10_v1.yaml` | NoGID 四卡三轮 SFT | 已完成，checkpoint-7404 |
| `sft/a0_gid_history10_v1.yaml` | A0-GID 四卡三轮 SFT，与 A4-GID 保持相同优化协议和初始模型 | 已完成，checkpoint-8955 |
| `sft/a0_gid_pipeline_v1.yaml` | 冻结 A0 SID、A4 GID/语料、共享词表来源；设置平台准备和缓存路径 | 不读取 Test，不在开发机运行全量准备 |
| `sft/dataset_info.json` | LLaMA-Factory Train/Valid 数据注册 | 不注册 Test |
| `sft/evaluation_epoch3_fixed10k_generalization_v1.yaml` | 固定五组 Validation、epoch 3、Beam=10 | 十项评测完成，来源和结果冻结 |
| `sft/evaluation_epoch3_constrained_fixed10k_generalization_v1.yaml` | 按 SHA256 继承两份完成的无约束计划，仅新增全目录前缀约束 | 双卡入口就绪、CPU 预检通过，GPU 未启动 |
| `sft/evaluation_epoch3_full_test_20260714_v1.yaml` | 通过哈希继承 Validation 模型/生成协议，单独绑定最后一天全量 Test | 双卡入口就绪、CPU 预检通过，不用于训练或选模 |
| `sft/evaluation_a0_gid_epoch3_dual_decode_v1.yaml` | 固定 A0-GID epoch 3、五组 Validation 与约束/无约束相同生成协议 | 单卡 A100 串行入口就绪，GPU 未启动 |
| `code_provenance.yaml` | 复制来源、源/目标 SHA256 和适配说明 | 代码隔离审计依据 |

## 8. 测试如何对应源码

`tests/` 当前有 54 个测试文件，全部使用合成数据，不提交真实业务数据。文件名与覆盖职责如下：

| 范围 | 测试文件 |
|---|---|
| 基础与隔离 | `test_artifacts.py`、`test_config.py`、`test_data_contracts.py`、`test_code_isolation.py`、`test_cli.py` |
| active/Query 统计 | `test_active_poi_assets.py`、`test_query_normalization.py`、`test_query_statistics.py`、`test_query_supervision.py`、`test_query_embeddings.py` |
| Query 图 | `test_query_graph_data.py`、`test_query_graph.py`、`test_query_graph_scaling.py` |
| Adapter | `test_adapter_training.py`、`test_hard_negative_mining.py`、`test_query_adapter.py`、`test_query_adapter_cache.py`、`test_query_adapter_config.py`、`test_query_adapter_data.py`、`test_query_adapter_gate.py`、`test_query_adapter_retrieval.py`、`test_query_adapter_selection_config.py`、`test_query_adapter_selection.py` |
| A0 | `test_base_codebook_config.py`、`test_base_codebook.py` |
| 关系化 S1/S2 | `test_relational_codebook_config.py`、`test_relational_codebook_data.py`、`test_relational_codebook.py`、`test_relational_codebook_attribution.py`、`test_relational_s2_diagnostics.py` |
| GID-parent S3 | `test_geo.py`、`test_local_codebook_config.py`、`test_local_codebook_data.py` |
| NoGID S3 | `test_nogid_codebook_config.py`、`test_nogid_codebook_data.py`、`test_nogid_sid_evaluation_config.py` |
| SID 评测/可视化 | `test_sid_pipeline_config.py`、`test_sid_evaluation_config.py`、`test_sid_evaluation.py`、`test_sid_visualization.py`、`test_sid_category_region_visualization.py`（同样本、码编号置换不变、完整桶计数、非单例纯度、A4 前两层一致）、`test_sid_collision_visualization.py`（三版共同碰撞条件、禁止退回单例、成员无删减、不同 ID、HTML 转义） |
| A0→A4 迁移 | `test_sid_migration.py`：10 项测试覆盖与暴力集合计算一致、码号置换、同码换成员、单例不可比、精确分数比较、总量守恒、分组抽样、全量成员与有界预览、输入错误和 HTML 转义 |
| A0-GID SFT | `test_a0_gid_sft.py`：完整历史/目标替换、CURRENT 与业务主键保留、非法 target/历史/split 拒绝、末位 Dedup、确定性 mapping、缺词停止、阶段复用/污染拒绝、并发锁、同 A4 初始化与配置；可按需执行真实 tokenizer 的八条合成数据 Cache smoke |
| 约束解码 | `test_sft_constrained.py`：两种长度的全部前缀与暴力集合一致、单例/碰撞闭合、左 padding、非法路径拒绝、真实小型 Qwen Beam=10、只注入 callback、smoke 失败阻断、双卡日志、短 TMPDIR、四类宏平均与无约束差值 |
| Final ID/SFT | `test_final_identifiers.py`、`test_sft_data.py`、`test_sft_vocabulary.py`、`test_sft_preflight.py`、`test_sft_training.py` |
| SFT 生成评测 | `test_sft_evaluation_data.py`、`test_sft_evaluation.py`、`test_sft_full_test_data.py`、`test_sft_full_test_evaluation.py`；覆盖冻结 split、全量配对、惰性读取、生成协议一致、断点/OOM、双卡调度、日志转发与汇总隔离 |

新增或修改领域逻辑时，应在同名测试文件补最小合成案例；命令参数只在 `commands/` 变化时补 `test_cli.py`。

## 9. 主要产物位置

本节路径均相对于 `qg_prqk/`。

| 产物 | 路径 |
|---|---|
| active POI BGE/category | `outputs/inputs/beijing_poi_active_bge_m3_v1/` |
| P2 Query 统计 | `outputs/qg_prqk_1024x3_v1_1/query_stats/` |
| active P2.5 Query 深度 | `outputs/qg_prqk_1024x3_v2_1_cat_active/query_depth/` |
| D3 Query cache / Adapter FULL | `outputs/qg_prqk_1024x3_v2_1_cat_active/query_cache_exact_full/`、`query_adapter_exact_full/` |
| P4 Query 图 | `outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/` |
| P5 A0 | `outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0_hard60_topk5_v1/` |
| P6 S1/S2 | `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_prqk_s1_s2_hard60_topk5_v1/` |
| GID-parent S3 | `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_geo_prqk_s3_hard60_topk5_v1/` |
| NoGID S3 | `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_geo_nogid_prqk_s3_s1s2_parent_hard60_topk5_v1/` |
| A0/A4 与 NoGID 静态评测 | `outputs/qg_prqk_512x3_v2_1_cat_active/p8_*_v1/` |
| SID 可视化 | `outputs/figures/qg_prqk_sid_a0_a4_nogid_full_v1/` |
| 同五类 t-SNE / 前缀类别区域图 | `outputs/figures/qg_prqk_sid_category_region_v2/`；两个 PNG/PDF、固定样本与坐标、完整分布指标和来源 manifest |
| 共同碰撞前缀案例 | `outputs/figures/qg_prqk_sid_collision_examples_v1/`；15 个非单例完整 SID 桶的 PNG/PDF、完整 POI 明细 HTML、全桶计数和来源 manifest |
| A0→A4 桶成员迁移 | `outputs/figures/qg_prqk_sid_migration_v1/`；全库概览、两版逐层提高/持平/降低案例 PNG/PDF、成员明细 HTML、完整成员 NPZ、JSON 统计与来源 manifest |
| 双 Final ID | `outputs/final_identifiers/a4_gid_parent_order_a_v1/`、`a4_nogid_sid3_v1/` |
| 双 Messages | `outputs/sft_data/a4_gid_parent_order_a_history10_v1/`、`a4_nogid_sid3_history10_v1/` |
| 共享扩词模型 | `outputs/models/Qwen3-0.6B-QGPRQK-A4-Vocab-v1/` |
| 双 Tokenized Cache | `outputs/sft_tokenized/a4_gid_parent_history10_v1/`、`a4_nogid_history10_v1/` |
| 双 SFT 最终 checkpoint | `outputs/sft/a4_gid_parent_history10_v1_gpu4_e3/checkpoint-8949/`、`outputs/sft/a4_nogid_history10_v1_gpu4_e3/checkpoint-7404/` |
| 双 SFT Validation 评测 | `outputs/eval/sft_epoch3_fixed10k_generalization_v1/<variant>/{data,results}/`；十项正式生成结果已完成 |
| 双 SFT 约束 Validation 评测 | `outputs/eval/sft_epoch3_constrained_fixed10k_generalization_v1/`；两版 plan 已验收，GPU 未运行，未来结果为分支内 `results/<subset>/` 与根 `summary.json` |
| 双 SFT 全量 Test 评测 | `outputs/eval/sft_epoch3_full_test_20260714_v1/`；两版各 606,682 条已完成并核验，原始 plan/断点/结果与总汇总保留 |
| A0-GID Final ID / Messages（已完成） | `outputs/final_identifiers/a0_gid_order_a_v1/`、`outputs/sft_data/a0_gid_order_a_history10_v1/`；只有 Train/Valid |
| A0-GID Cache / SFT（已完成） | `outputs/sft_tokenized/a0_gid_history10_v1/`、`outputs/sft/a0_gid_history10_v1_gpu4_e3/`；末轮 `checkpoint-8955` |
| A0-GID 双解码 Validation（入口就绪） | `outputs/eval/a0_gid_epoch3_dual_decode_v1/`；单卡 A100 平台自动准备共享缓存，先无约束五组、再约束五组；目前无正式 GPU 指标 |
| A0-GID 平台进度 | `outputs/run_control/sft_a0_gid_history10_v1_gpu4_e3/`；分阶段日志、退出码和缓存验收凭证 |
| 运行日志、PID、失败现场与源码快照 | `outputs/run_control/` |

阶段产物按各自 schema 核验 manifest 和 `_SUCCESS`；生成评测则要求退出码 0、`result/summary.status=completed`、完整样本数和来源哈希一致，不凭空要求不存在的 `_SUCCESS`。大文件存在或 NPY 达到预分配大小不代表成功。失败目录、旧 Gate、旧容量配置和源码快照是实验审计材料，不是当前默认输入。

## 10. 最小验证与下一步

查看所有当前命令：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py --help
```

运行轻量合成回归：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib:${LD_LIBRARY_PATH} \
MPLCONFIGDIR=qg_prqk/outputs/cache/matplotlib \
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  -m unittest discover -s qg_prqk/tests -v
```

检查两个已训练分支的评测输入，不启动 GPU 推理（会准备独立评测缓存）：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py evaluate-sft \
  --variant a4_gid_parent --dry-run

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py evaluate-sft \
  --variant a4_nogid --dry-run
```

下一步是在一张 A100 上运行 `run_evaluate_prqk_a0_gid_epoch3_dual_decode_1xa100.sh`；具体命令与产物说明见 `../README.md` 的“PRQ-KMeans 对照：单卡 A100 约束与无约束评测”。同一张卡先跑无约束五组，再跑约束五组，共用冻结数据和输出协议；不要与保留的双卡兼容入口同时启动。不要重跑 SID、覆盖任何已完成训练或读取 Test，也不根据五组评测选择 checkpoint。定向回归为 `tests/test_a0_evaluation.py`，覆盖 A0 标识来源、两种解码共享计分、断点、单卡串行、双卡兼容、硬件门禁和宏平均口径。
