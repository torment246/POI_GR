# 实验进展索引

本文件只维护正式实验的方法入口和简要索引。完整配置、命令、指标、产物与结论记录在对应方法文档中；同一论文复现不按 Embedding、SID、SFT 拆散，Query 增强向量创新作为独立研究线维护专门实验台账。

## 方法文档

| 方法 | 文档 | 当前范围 | 状态 |
|---|---|---|---|
| V1 | [V1 实验](experiments/V1.md) | 共享 Embedding 选型、RQ-VAE、GID/Dedup、SFT 和 Trie 评测 | 固定 10,000 条 Validation 最小闭环已完成 |
| Embedding 优化 | [Embedding 优化实验](experiments/EMBEDDING_OPTIMIZATION.md) | Train 正向 Query 聚合、异质性感知融合与向量召回消融 | E1—E4 已完成；最终冻结 E4/α=0.30/β=0.85 |
| RQ-KMeans | [RQ-KMeans SID 实验](experiments/RQKMEANS.md) | Embedding × Quantizer 对照、RQ-KMeans 参数筛选、TIGER RQ-VAE 公平对照与 30-bit 码本布局消融 | BGE/E4 三种 30-bit 布局已闭环；已冻结 E4-2048 主模型、E4-4096 容量上界和 BGE-2048 Embedding 消融 |
| TIGER | [TIGER 复现实验](experiments/TIGER.md) | RQ-VAE SID、collision token、历史序列 SFT 和无约束生成评测 | 三轮训练与固定子集评测已完成 |
| GNPR-SID | [GNPR-SID 复现实验](experiments/GNPR_SID.md) | 行为稀疏输入审计、全量 content-geo 输入、Diversity Loss RQ-VAE、条件 Dedup、SFT 和无约束生成评测 | 三轮训练与固定子集评测已完成 |
| GenPOI | [GenPOI 复现实验](experiments/GENPOI.md) | GeoPE32、唯一 PID、历史序列 SFT、SSP 与 TCG 评测 | 旧版与 Centered 修正版三轮闭环均完成 |

后续新论文或创新方法只有在第一次正式实验开始时才新增一个方法文档；单次训练、评测或消融继续追加到所属方法文档，不单独创建 Markdown。Embedding 优化文档是已确认创新线的统一台账，不为 E1—E4 分别建文件。

## 正式实验索引

| Experiment | 日期 | 方法 | 状态 | 核心结果 |
|---|---|---|---|---|
| `EXP-20260717-01` | 2026-07-17 | V1 | 已完成 | Qwen3-Embedding-0.6B 完成 2,337,178 条 POI 编码，吞吐 451.49 条/秒 |
| `EXP-20260718-01` | 2026-07-18 | V1 | 已完成 | Qwen3-Embedding-4B 完成 2,337,178 条 POI 编码，吞吐 114.86 条/秒 |
| `EXP-20260722-01` | 2026-07-22 | V1 | 已完成 | 0.6B 无 Instruction：Hit@10 28.25%，MRR@10 16.6907% |
| `EXP-20260722-02` | 2026-07-22 | V1 | 已完成 | 4B 无 Instruction：Hit@10 21.18%，低于相同口径的 0.6B |
| `EXP-20260723-01` | 2026-07-23 | V1 | 已完成 | 3 种码本容量 × 4 个固定 epoch 共 12 组北京全量 SID 评估完成；512×3/1024×3 epoch 20 的完整 Top 20 碰撞桶分析覆盖 5,982/4,037 条 POI，最大桶为 710/322 |
| `EXP-20260723-02` | 2026-07-23 | V1 | 已完成 | Geohash6 + 1024×3 epoch 20 SID 的 PID 唯一率为 84.89%，解决 48.45% 的原 SID 碰撞 POI；残余碰撞 POI为 557,154 |
| `EXP-20260723-03` | 2026-07-23 | V1 | 已完成 | 557,154 条碰撞 POI 按桶内 poi_id 稳定分配 Dedup Code，2,337,178 条最终 PID 全局唯一；映射确定性复跑哈希一致 |
| `EXP-20260723-04` | 2026-07-23 | V1 | 已完成 | 8,790,513 条订单按严格时间切分生成 Messages SFT 数据，PID 匹配率 100%；train/valid/test 为 7,586,410/597,421/606,682 |
| `EXP-20260724-01` | 2026-07-24—25 | V1 | 已完成 | Qwen3-0.6B 单卡 A100 全参数训练 2 epoch；四个完整 checkpoint 的 Validation Loss 为 0.39634/0.28160/0.23992/0.22749 |
| `EXP-20260725-01` | 2026-07-25 | V1 | 已完成（固定子集） | [固定 10,000 条 Validation、Beam=10 对比四个 checkpoint](experiments/V1.md#exp-20260725-01sft-eval-001final-pid-trie-约束生成式检索评测)；2.0 epoch 的 HR@1/HR@10/NDCG@10 为 0.4577/0.8483/0.6599 |
| `EXP-20260729-01` | 2026-07-29 | V1 | 已完成 | BGE-M3 旧固定 10,000 条无 Instruction 精确召回 Hit@10/20 为 32.78%/40.81%，三模型最优 |
| `EXP-20260730-01` | 2026-07-30 | TIGER | 已完成 | 三容量 20 epoch 和全量 SID 导出完成；唯一率为 48.09%/61.72%/71.68% |
| `EXP-20260730-02` | 2026-07-30 | TIGER | 已完成 | 追加 `C0–C305` 后，2,337,178 条四层 identifier 全局唯一且复跑映射哈希一致 |
| `EXP-20260730-03` | 2026-07-30 | TIGER | 已完成（训练准备） | 历史序列、Tokenizer、packed Cache、GPU smoke 和四卡训练配置通过 |
| `EXP-20260802-01` | 2026-08-02 | GenPOI | 已完成 | GeoPE32 三容量全量 SID 唯一率为 41.39%/58.00%/73.11%，选择 1024×3 |
| `EXP-20260802-02` | 2026-08-02 | GenPOI | 已完成 | Geohash6 后唯一率 85.7634%，再经 `D0–D215` 得到 2,337,178 条唯一 PID |
| `EXP-20260803-01` | 2026-08-03 | GenPOI | 已完成（训练准备） | 历史序列、3,626 个 Token、正式 Cache 和 GPU smoke 完成，三轮训练入口通过 |
| `EXP-20260804-01` | 2026-08-04 | TIGER | 已完成（固定子集） | [三轮无约束生成评测](experiments/TIGER.md#exp-20260804-01tiger-sft-eval-001三轮正式训练与固定-10000-条-validation-评测)中 epoch 3 的 HR@1/HR@10/NDCG@10 为 0.5187/0.8716/0.704190 |
| `EXP-20260804-02` | 2026-08-04 | GenPOI | 已完成（固定子集） | [三轮 TCG+SSP 评测](experiments/GENPOI.md#exp-20260804-02genpoi-sft-eval-001tcgssp-固定子集评测)中 epoch 3 的 HR@1/HR@10/NDCG@10 为 0.5085/0.8713/0.698751，PID 合法率 100% |
| `EXP-20260805-01` | 2026-08-05 | GNPR-SID | 已完成（输入审计未通过） | 703,306 条行为 POI 三容量完成；1024×3 唯一率 94.24%，但 Top 碰撞由 8192 用户哈希别名主导 |
| `EXP-20260805-02` | 2026-08-05 | GNPR-SID | 已中止（训练坍塌） | 等权 content-geo 在 epoch 2 的第三层仅使用 10/256 个码，监控 SID 唯一率降至 5.12% |
| `EXP-20260805-03` | 2026-08-05 | GNPR-SID | 已完成（诊断） | 加权 content-geo 两轮监控 SID 唯一率由 78.40% 升至 85.72%，未复现坍塌 |
| `EXP-20260805-04` | 2026-08-05 | GNPR-SID | 已中止 | 单卡任务完成至 epoch 5 后停止；其后发现未接入 Diversity Loss，相关平台中途输出已清空 |
| `EXP-20260805-05` | 2026-08-05 | GNPR-SID | 已完成（训练） | [带 Diversity Loss 的三容量四卡 20 epoch 正式训练](experiments/GNPR_SID.md#exp-20260805-05-gnpr-content-geo-三容量-diversity-loss-正式训练)退出码均为 0；512 监控 SID 唯一率 99.5550%，1024 从 epoch 17 起坍塌 |
| `EXP-20260805-06` | 2026-08-05 | GNPR-SID | 已完成 | [三容量 2,337,178 条全量 SID 评估](experiments/GNPR_SID.md#exp-20260805-06-gnpr-content-geo-三容量全量-sid-评估)唯一率为 42.64%/77.80%/3.13%，最终选择 512×3；1024 后期坍塌经全量结果确认 |
| `EXP-20260805-07` | 2026-08-05 | GNPR-SID | 已完成（复跑诊断） | [1024×3 epoch 16复跑与全量评估](experiments/GNPR_SID.md#exp-20260805-07-gnpr-1024x3-epoch-16-可复现性与全量评估)唯一率63.18%、碰撞POI 51.95%，前缀类别纯度较高但第三层失活且未稳定复现，仍选择512×3 |
| `EXP-20260805-08` | 2026-08-05 | GNPR-SID | 已完成（唯一标识） | [512×3 条件 Dedup Token 构建](experiments/GNPR_SID.md#exp-20260805-08-gnpr-512x3-论文-dedup-token-唯一标识)仅为 830,263 条碰撞 POI 追加 `D0-D222`，最终 2,337,178 条标识全局唯一且两次全量映射哈希一致 |
| `EXP-20260805-09` | 2026-08-05 | GNPR-SID | 已完成（训练准备） | [8,790,513 条无时间、无用户 ID 的地图检索 SFT 数据](experiments/GNPR_SID.md#exp-20260805-09-gnpr-地图检索-sft-全量数据与训练入口)、1,805 Token 扩词表和 packed Train/Valid 2,758,797/201,947 条 Cache 完成；目标截断为 0，GPU smoke 与正式 SFT 待运行 |
| `EXP-20260805-10` | 2026-08-05 | GenPOI | 已完成（修正版唯一标识） | [Centered GeoPE32 + TIGER-RQVAE 三容量与唯一 PID](experiments/GENPOI.md#exp-20260805-10-genpoi-centered-geope32-sid-修正版与唯一-pid)：1024 SID 唯一率65.7629%，Geohash6后79.6911%，再经 `D0-D280` 得到2,337,178条全局唯一PID |
| `EXP-20260806-01` | 2026-08-06 | GenPOI | 已完成（训练准备） | [Centered GeoPE32 新 PID 的全量 SFT 数据与缓存](experiments/GENPOI.md#exp-20260806-01-genpoi-centered-geope32-全量-sft-数据与缓存)：Train/Valid/Test 为7,586,410/597,421/606,682，packed Train/Valid 为2,935,702/214,860，四卡三轮入口已通过 |
| `EXP-20260807-01` | 2026-08-06—07 | GNPR-SID | 已完成（固定子集） | [三轮 SFT 与无约束生成评测](experiments/GNPR_SID.md#exp-20260807-01-gnpr-三轮-sft-与固定-10000-条-validation-无约束评测)中 epoch 3 最优，HR@1/HR@10/NDCG@10 为 49.66%/83.07%/67.2390%，Valid ID Rate 为 78.808% |
| `EXP-20260807-02` | 2026-08-07 | V1 | 已完成（固定子集） | [无 Train 订单泄漏的三模型精确召回](experiments/V1.md#exp-20260807-02固定-sft-validation-10k-三种-embedding-无泄漏精确召回)：BGE-M3 Hit@10/20 为 32.39%/40.25%，Qwen3-0.6B 为 28.26%/34.35%，Qwen3-4B 为 21.21%/24.94% |
| `EXP-20260807-03` | 2026-08-07 | GNPR-SID | 已完成（逐层诊断） | [固定 10,000 条 Teacher-Forcing 对比](experiments/GNPR_SID.md#exp-20260807-03-tigergnpr-sid-逐层-teacher-forcing-诊断)：GNPR 第一层 Top-1 比 TIGER 低 21.37pp，但第二、三层分别高 2.39/6.32pp；完整 identifier 差距 2.36pp，与自由生成 HR@1 差距 2.21pp 接近 |
| `EXP-20260807-04` | 2026-08-06—07 | GenPOI | 已完成（固定子集） | [Centered 三轮 SFT 与 TCG+SSP 评测](experiments/GENPOI.md#exp-20260807-04-genpoi-centered-三轮-sft-与固定-10000-条-tcgssp-评测)：epoch 3 的 HR@1/HR@10/NDCG@10 为 51.98%/87.83%/70.7430%，较旧版提高 1.13/0.70/0.8679pp，PID 合法率 100% |
| `EXP-20260807-05` | 2026-08-07 | Embedding 优化 | 已完成（全量精确召回） | [E1/E2 Train Query 增强](experiments/EMBEDDING_OPTIMIZATION.md#exp-20260807-05e1e2-train-query-增强全量精确召回)：E2/α=0.30 最优，Hit@10/MRR@10/NDCG@10 为 51.18%/27.7598%/33.2917%，相对 E0 提高 18.79/9.9971/12.0804pp |
| `EXP-20260807-06` | 2026-08-07 | Embedding 优化 | 已完成（负结果） | [E3 异质性感知逐 POI 融合](experiments/EMBEDDING_OPTIMIZATION.md#exp-20260807-06e3-异质性感知逐-poi-融合)：六组均低于 E2；最优 Hit@10/NDCG@10 为 41.13%/26.5432%，相对 E2/0.30 下降 10.05/6.7485pp，拒绝候选级动态 α 的单索引方案 |
| `EXP-20260808-01` | 2026-08-08 | Embedding 优化 | 已完成（正结果，边界最优） | [E4 类别公共 Query 方向残差](experiments/EMBEDDING_OPTIMIZATION.md#exp-20260808-01e4-类别公共-query-方向残差)：α=0.30、β=0.75 的 Hit@10/NDCG@10 为 55.82%/36.1412%，相对 E2 提高 4.64/2.8495pp；最优 β 位于预设上界 |
| `EXP-20260808-02` | 2026-08-08 | Embedding 优化 | 已完成（边界补充，最终冻结） | [E4 类别残差边界补充](experiments/EMBEDDING_OPTIMIZATION.md#exp-20260808-02e4-类别残差边界补充与最终冻结)：β=0.85 的 Hit@10/NDCG@10 为 56.37%/36.3976%，优于 0.75；β=1.00 出现回落，最终冻结 α=0.30、β=0.85 |
| `EXP-20260808-03` | 2026-08-08 | GenPOI | 已完成（逐层诊断） | [Centered GenPOI 固定 10,000 条 Teacher-Forcing](experiments/GENPOI.md#exp-20260808-03-centered-genpoi-固定-10000-条逐层-teacher-forcing-诊断)：gold GID6 条件下 S1/S2/S3 Top-1 为 83.97/88.67/89.27%，SID3 累计 69.90%；GID6 累计 73.12%，完整 PID Top-1 all 为 50.00% |
| `EXP-20260808-04` | 2026-08-08 | RQ-KMeans | 已完成（原始 BGE 全量 SID） | [RQ-KMeans 参数筛选与 30-bit 布局消融](experiments/RQKMEANS.md)：公平 `1024³` 唯一率/碰撞 POI 为 76.07%/36.15%，优于 TIGER 的 71.68%/40.79%；`4096×1024×256` 达到 81.89%/29.15%，最大桶 96，depth-2/3 micro-purity 为 83.36%/97.16% |
| `EXP-20260808-05` | 2026-08-08 | RQ-KMeans | 已完成（E4 + TIGER RQ-VAE） | [冻结 E4 + 对称 `1024³` TIGER RQ-VAE](experiments/RQKMEANS.md#exp-20260808-05-e4--tiger-rq-vae-1024³)：唯一率 74.3539%、碰撞 POI 37.4194%，相对 BGE+RQ-VAE 改善 2.6774/3.3700pp，S1 使用数 561→751；depth-1/2 micro-purity 降至 64.13%/74.88% |
| `EXP-20260808-06` | 2026-08-08 | RQ-KMeans | 已完成（E4 + RQ-KMeans，完整 `2×2`） | [冻结 E4 + 对称 `1024³` RQ-KMeans](experiments/RQKMEANS.md#exp-20260808-06-e4--rq-kmeans-1024³-与完整-2×2)：唯一率 76.1573%、碰撞 POI 35.7477%、最大桶 288；E4 在 RQ-KMeans 下唯一率仅增 0.0848pp，但最大桶由 467 降至 288，表明两类改进强重叠、互补收益集中在长尾碰撞 |
| `EXP-20260808-07` | 2026-08-08 | RQ-KMeans | 已完成（E4 非对称 30-bit 迁移） | [E4 + `4096×1024×256` RQ-KMeans](experiments/RQKMEANS.md#exp-20260808-07-e4--rq-kmeans-4096×1024×256)：唯一率 82.7331%、碰撞 POI 27.9655%，相对同布局 BGE 改善 0.8392/1.1857pp，相对 E4 `1024³` 改善 6.5758/7.7822pp；最大桶 125，冻结为当前纯 SID 结构最优 |
| `EXP-20260808-08` | 2026-08-08 | RQ-KMeans | 已完成（E4 工程折中与 SFT 候选冻结） | [E4 + `2048×1024×512` RQ-KMeans](experiments/RQKMEANS.md#exp-20260808-08e4-2048×1024×512-与-sft-候选冻结)：唯一率/碰撞 POI/最大桶为 79.8731%/31.4021%/143；相对 E4 `1024³` 改善 3.7158/4.3456pp且只增加 512 个基础 token，冻结为 SFT 主模型；E4-4096 为容量上界，BGE-2048 为同布局 Embedding 消融 |

## 记录规则

- 实验编号统一使用 `EXP-YYYYMMDD-NN`，同一天从 `01` 递增，跨阶段也不得重复。
- 新实验先在所属方法文档追加完整记录，再在本索引追加一行。
- 每条正式记录包含目标与假设、数据版本、代码状态、配置、命令、环境、核心指标、产物、结论和下一步。
- 只记录实际运行结果；失败或中止必须明确标注，不补写或猜测指标。
- 临时 smoke、性能探测和调试过程不单独建文档，只在正式实验中保留最终采用的配置及必要说明。

## 开发任务记录

| Task | 日期 | 范围 | 验证 | 状态 |
|---|---|---|---|---|
| `SID-EVAL-001` | 2026-07-22 | 统一 SID 输入校验、碰撞指标、码本利用率、前缀类别纯度与热点 Case | 7 个合成单元测试和 CLI smoke test 通过；未运行北京全量数据 | 已完成 |
