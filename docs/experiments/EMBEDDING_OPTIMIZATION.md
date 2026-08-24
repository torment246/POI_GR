# Embedding 优化实验

本文件专门记录 Query-Augmented POI Embedding 的协议、实现、消融和正式结果。V1 的三种原始编码器选型及完整第一版链路仍记录在 [V1 实验](V1.md)；本文件从冻结的 BGE-M3 E0 出发，只研究如何利用 Train 正向 `Query → POI` 日志离线增强 POI 向量。本文件的 E0—E4 是创新线内部编号，与 V1 历史编码器对照曾使用的 E1—E3 标签相互独立。

## 1. 研究边界

当前阶段保持以下约束：

- BGE-M3 参数完全冻结，不微调编码器；
- 只使用 2026-07-01—12 Train 的正向 Query–POI 样本；
- 不构造困难负样本，不进行对比学习、RL、GRPO 或 DPO；
- 固定 2026-07-13 SFT Validation 10,000 条评测 Query 的 BGE-M3 向量与行顺序；
- 所有实验只替换 POI 向量，Query 向量、候选库、相似度、Faiss 索引和 Top-K 不变；
- Embedding 获得稳定收益后才进入 RQ-KMeans/RQ-VAE 对比，本文件不提前记录 SID 结果。

核心问题是：训练日志中的用户表达能否补充 POI 名称、地址和别名没有显式覆盖的检索语义，并提升固定评测集上的整体精确召回。

## 2. 冻结评测协议与 E0

E0 的完整正式记录为 [V1 `EXP-20260807-02`](V1.md#exp-20260807-02固定-sft-validation-10k-三种-embedding-无泄漏精确召回)。本文件只保留后续实验必须复用的契约。

### 2.1 数据与检索契约

| 项目 | 冻结值 |
|---|---|
| 评测集 | `data/eval/embedding_retrieval/sft_validation_10k_v1/eval.jsonl` |
| 评测行数 / 唯一目标 POI | 10,000 / 7,520 |
| 评测集 SHA256 | `5557769526bd1a41a86adac5efb26eb2bf878fed8b2e2dad9f58afe93663e713` |
| Train / Eval 时间 | 2026-07-01—12 / 2026-07-13 |
| Train/Eval 业务键重叠 | `order_id=0`、`searchid=0`、复合键 `=0` |
| 候选 POI | 北京全量 2,337,178 条 |
| Query 编码器 | BGE-M3，CLS pooling，Right padding，无 Instruction，L2 Normalize |
| Query 向量 | `outputs/evaluation/sft_validation_10k_v1/query_embeddings/bge_m3_no_instruction.npy` |
| Query 向量 SHA256 | `98e2032b86b76cf42991dc1d1f4967ff456661fb32fbae185652163c85bf70e0` |
| Query 行映射 SHA256 | `d512aa206468ea672b473ce38c353eca703c64289ac890d9a935cd3c9b2f855f` |
| 检索 | GPU `GpuIndexFlatIP`，float32 精确内积，Top-20 |

### 2.2 E0 当前结果

| POI 向量 | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | NDCG@10* |
|---|---:|---:|---:|---:|---:|---:|
| 原始 BGE-M3 | 11.99% | 25.25% | 32.39% | 40.25% | 17.7627% | 21.2113% |

`*` NDCG@10 由保存的 `target_ranks` 独立补算；当前 `metrics.json` 尚未原生保存该指标。E0 召回结果 SHA256 为 `8052a40b64b6b875c61a596388a764644042bb701936a5394eda4d9ff4eb3a9c`。

## 3. Train 正向 Query 数据准备

训练侧哈希分片、Train-only 聚合统计、1,408,778 个唯一 Query 编码和 E1/E2 稀疏聚合均已完成。正式评测不保存 12 份全量融合向量，而是在每次构建 Faiss 索引前按 POI 行分块确定性融合；因此磁盘只保存两份 491,213 行的 Query 聚合向量和每组 Top-20 结果。

| 项目 | 结果 |
|---|---:|
| Train 文件 | `data/sft/beijing_order_main_v1/train.jsonl` |
| Train 行数 | 7,586,410 |
| Train SHA256 | `4ed3f3849e0beb10df500f013bc08b4663b0dca5df92dde9c8b3cb4a7ec9ffe4` |
| 哈希分片 | 256 个 ZSTD Parquet |
| 哈希规则 | 原始 Query UTF-8 的 BLAKE2b-64，`% 256` |
| 行守恒 / 哈希错位 | 7,586,410 / 0 |
| 唯一原始 Query | 1,408,778 |
| 唯一 Query–POI 对 | 2,063,175 |
| 有 Train Query 的 POI | 491,213（全目录 21.0174%） |
| 分片行数范围 | 19,734—100,193 |
| 原始分片产物 | `outputs/embeddings/query_augmented_bge_m3/train_query_shards_v1/`，84,734,776 bytes |
| 聚合统计产物 | `outputs/embeddings/query_augmented_bge_m3/train_query_stats_v1/`，56,586,021 bytes |
| 聚合耗时 / 峰值 RSS | 174.63 秒 / 221.80 MiB |
| 聚合文件 | Query 目录、Query–POI 计数、POI 覆盖各 256 个 ZSTD Parquet |

聚合产物满足四项守恒：Query 订单数和 POI 订单数均为 7,586,410；Query 的 `poi_df` 之和与 POI 的唯一 Query 数之和均为 2,063,175。`query_id` 是 `[0, 1,408,778)` 的连续 int64，顺序固定为 Query 哈希分片升序、分片内原始 Query 字典序。

Query 跨 POI 的 DF 分布：

| `df(q)` | 唯一 Query 数 |
|---|---:|
| 1 | 1,211,149 |
| 2—5 | 167,455 |
| 6—20 | 26,480 |
| >20 | 3,694 |

POI 的不同 Train Query 数及其在固定评测集中的覆盖：

| 不同 Train Query 数 | 全量覆盖 POI | 固定评测行数 |
|---|---:|---:|
| 0 | 1,845,965 | 307 |
| 1—5 | 396,668 | 1,535 |
| 6—20 | 81,191 | 2,919 |
| >20 | 13,354 | 5,239 |

固定评测的 7,520 个唯一目标 POI 中有 7,215 个具备 Train Query，覆盖率为 95.9441%；按 10,000 条评测行计算覆盖率为 96.93%。后续仍按固定 10,000 条整体评测，不把 307 条无覆盖样本设为单独验收项；实现中这些 POI 自然沿用原始 BGE 向量。

相关实现为 `src/poi_gr/embedding/query_shards.py`、`src/poi_gr/embedding/query_stats.py`、`scripts/embedding/build_train_query_shards.py`、`scripts/embedding/build_train_query_stats.py` 及对应测试。两段全量处理均采用逐分片读取、有界缓冲和 Parquet 压缩，不在内存中加载 758 万行明细，也不使用系统临时目录。

```bash
python scripts/embedding/build_train_query_stats.py
python scripts/embedding/build_train_query_stats.py --validate-only
```

## 4. 表示与实验矩阵

设 POI $i$ 的原始归一化 BGE 向量为 $c_i$，Train 中关联的不同原始 Query 集合为 $Q_i$，冻结编码器得到：

$$
e_q = \operatorname{Normalize}\left(\operatorname{BGE}(q)\right)
$$

Query 聚合向量统一写为：

$$
\bar{q}_i = \operatorname{Normalize}\left(
\frac{\sum_{q\in Q_i} w(q,i)e_q}
{\sum_{q\in Q_i} w(q,i)}
\right)
$$

最终 POI 向量为：

$$
z_i = \operatorname{Normalize}\left((1-\alpha_i)c_i+\alpha_i\bar{q}_i\right)
$$

若 $Q_i=\varnothing$，令 $z_i=c_i$；评测时仍统一计入整体指标，不单独选择参数或设门禁。

| 编号 | 方法 | 关键定义 | 状态 |
|---|---|---|---|
| E0 | 原始 BGE-M3 | $z_i=c_i$ | 已完成，正式结果在 V1 |
| E1 | 正向 Query 简单均值 | 不同 Query 等权，$w(q,i)=1$；扫描固定 $\alpha$ | 已完成，最佳 $\alpha=0.30$ |
| E2 | 频次与 DF 加权 | Train-only `count(q,i)` 与 `df(q)` 降低泛 Query 权重 | 已完成，最佳 $\alpha=0.30$ |
| E3 | Heterogeneity-Aware Fusion | 根据有效 Query 数与权重异质性自适应 $\alpha_i$；异质性越高，保留越强的内容锚点 | 已完成，假设未通过；最佳仍低于 E2 |
| E4 | Category Query Residual | 在 E2 上分离类别共性与 POI 特有 Query 残差，保持全库统一 $\alpha=0.30$ | 已完成；边界补充后冻结 $\beta=0.85$ |

E2 的第一版权重冻结为：

$$
w(q,i)=
\log\left(1+\operatorname{count}(q,i)\right)
\cdot
\log\left(\frac{N+1}{\operatorname{df}(q)+1}\right)
$$

其中 $N$ 为至少拥有一个 Train Query 的 POI 数。所有 `count`、`df`、类别统计和融合参数选择都只能使用 Train；Validation 只用于比较已经明确列出的离散配置，不反向生成 Query 权重。

E1/E2 分桶结果否定了“不同 Query 越多就应提高 Query 权重”的初始直觉：1—20 个不同 Query 的目标 POI 在较大 $\alpha$ 上仍受益，而 $>20$ 的头部 POI 在 $\alpha=0.20$ 达峰，继续提高会被大量异质、泛化 Query 拉离内容语义。因此 E3 改为先计算加权有效 Query 数：

$$
n_i^{\mathrm{eff}}=
\frac{\left(\sum_{q\in Q_i}w(q,i)\right)^2}
{\sum_{q\in Q_i}w(q,i)^2}
$$

再使用随异质性增加而增强内容锚定的候选形式：

$$
\alpha_i=\alpha_{\min}+
\frac{\alpha_{\max}-\alpha_{\min}}
{1+\left(n_i^{\mathrm{eff}}/\tau\right)^\gamma}
$$

其中 $\alpha_{\min}$、$\alpha_{\max}$、$\tau$ 和 $\gamma$ 在 `EXP-20260807-06` 开始前已冻结为六组小规模离散候选；无 Train Query 的 POI 仍令 $\alpha_i=0$。正式结果否定了把该式直接用于单一全库索引的方案：目标 POI 所属分桶的局部最优系数不能逐候选拼接成全局最优索引，因为每个 POI 同时也是其他 Query 的负候选。

E4 不再改变候选级融合系数。对达到最小支持度的类别 $k$，只用 Train Query 聚合计算等 POI 权重的原始类别中心：

$$
\mu_k=\frac{1}{|I_k|}\sum_{i\in I_k}\bar{q}_i,
\qquad |I_k|\ge 32
$$

支持度不足时令 $\mu_k=0$。随后构造类别残差 Query 表示并统一融合：

$$
r_i(\beta)=\operatorname{Normalize}\left(\bar{q}_i-\beta\mu_{\operatorname{cat}(i)}\right)
$$

$$
z_i(\beta)=\operatorname{Normalize}\left(0.70c_i+0.30r_i(\beta)\right)
$$

这样全库候选仍使用同一 $\alpha$，E4 只检验去除类别公共方向能否强化同类 POI 之间的可区分语义。

## 5. 公平比较与验收指标

每组正式实验必须满足：

1. 复用 E0 的 10,000 条 BGE Query 向量及行映射 SHA256；
2. 复用相同 2,337,178 条 POI ID 顺序和全量精确 `IndexFlatIP`；
3. 记录配置、Query 聚合统计、融合公式和召回结果 SHA256；若像 E1/E2 一样在线分块融合，则不要求额外落盘重复的全量融合向量；
4. 报告 Hit@1/3/5/10/20、MRR@10、NDCG@10；
5. 报告相对 E0 的绝对百分点变化和逐 Query 成对胜负；
6. 检查全部向量有限、L2 范数、重复向量率、近邻相似度分布及异常聚集；
7. 不读取 Test，不在不同实验间改变 Query 编码或检索精度。

必须同时报告以下子集，避免总体收益被头部流量掩盖：

- Train 不同 Query 数为 1—5、6—20、>20；
- 评测 Query 长度为 1—2、3—5、6—10、>10；
- 目标 POI 的 Train 订单频次分桶；
- 同名 POI、地址号牌型 POI 和连锁分店等后续可稳定构建的诊断集合。

## 6. 实施顺序

### 第一步：聚合统计与冻结 Query 编码

- 已完成：流式处理 256 个 Train Query 分片，生成唯一 Query、`count(q,i)`、`df(q)` 和 POI 覆盖统计；
- 已完成：建立固定评测目标的覆盖分桶；
- 已完成：实现每个唯一 Query 只编码一次的连续行序、float16 NPY、输入指纹和断点恢复契约；
- 已完成：在开发机 RTX A6000 上生成 `outputs/embeddings/query_augmented_bge_m3/train_query_embeddings_v1/embeddings.npy`，并通过独立 `--validate-only` 复检；
- 编码与后续聚合继续使用分片迭代和 mmap，不把 140 万 Query embedding 和 206 万 Query–POI 对同时装入 Python 对象。

冻结编码配置为 BGE-M3、无 Instruction、CLS pooling、Right padding、BF16 模型、float16 输出、`max_seq_length=128`、batch 256 和 buffer 8,192。`embeddings.npy` 的行号严格等于连续 `query_id`，预计 shape 为 `[1408778, 1024]`；每 20 个模型 batch 刷新断点。

```bash
bash launchers/run_encode_train_queries_bge_m3_1a100.sh
```

正式产物为 `[1408778, 1024]` 的 float16 NPY，SHA256 为 `e300e248a274a0febbcfb94b0b457f76916814564316029e258fd2bfe692ad67`，全部向量均为有限值。A6000 上模型加载 31.70 秒、编码 316.65 秒、平均 4,449.01 Query/s，总运行 378.53 秒；CUDA 峰值 allocated/reserved 显存约 1.76/2.45 GiB。运行时缓存和临时目录均位于项目 `outputs/` 下，未使用开发机系统临时目录。

### 第二步：E1 与 E2 最小闭环

- 已完成 E1 等权 Query 语义增量验证；
- 已完成 E2 频次可信度与跨 POI DF 降权验证，正式结果优于 E1；
- 两者均扫描固定 $\alpha\in\{0.05,0.10,0.20,0.30,0.40,0.50\}$，按相同 10,000 条 Query 和 2,337,178 个 POI 执行 float32 精确检索。

### 第三步：E3 与 E4

- E3 已完成并否定“在同一索引中逐 POI 改变融合系数”的方案；最优 E3 的 Hit@10/NDCG@10 仅为 41.13%/26.54%，低于 E2/0.30 的 51.18%/33.29%；
- E4 主实验与边界补充均已完成；最终 $\beta=0.85$ 的 Hit@10/NDCG@10 为 56.37%/36.40%，完整中心化 $\beta=1.00$ 已出现回落；
- 任一方法只有在总体 Hit@10/NDCG@10 稳定提升时，才进入量化器对比；分桶指标用于解释变化，不作为单独否决条件。

### 第四步：冻结最佳 Embedding

冻结最佳方法的配置、全量 POI 向量和指纹后，再比较 RQ-KMeans、现有 RQ-VAE 与必要的 balance/OPQ 消融。原始 BGE 与 E4 输入上的 RQ-VAE/RQ-KMeans 对称 `1024³` `2×2`、E4 五档 RQ-KMeans 30-bit 容量方向均已完成，结果记录在 [RQ-KMeans SID 实验](RQKMEANS.md)。首轮 SFT 使用 E4 后移-512、对称-1024、前移-2048确定容量方向；原始 BGE 消融顺延到胜出布局，避免同时改变 Embedding 和码本分配。量化器和碰撞后缀属于下一阶段 SID 方法实验，不把其结果混入本文件的向量召回结论。

## 7. 正式实验记录

后续每次已运行的 E1—E4 正式实验追加在本节，并在 `docs/EXPERIMENT_LOG.md` 增加唯一 `EXP-YYYYMMDD-NN` 索引。未运行的配置不填写结果，不用 smoke 指标代替正式全量精确召回。

### EXP-20260807-05：E1/E2 Train Query 增强全量精确召回

#### 目标与假设

在冻结 BGE-M3 Query 编码器和原始 POI 内容向量的前提下，验证 Train 正向 Query 是否能补充名称、地址和别名未显式覆盖的检索表达；进一步验证 E2 的订单频次与跨 POI DF 加权是否稳定优于 E1 等权均值。主选择指标预先固定为 NDCG@10，平局依次比较 Hit@10、MRR@10 和更小的 $\alpha$。

#### 数据、代码与配置

| 项目 | 冻结值 |
|---|---|
| Experiment | `EXP-20260807-05` |
| Git commit / 工作区 | `e4515195097dcb10aac0f3281ee7ddc0e482886c` / dirty |
| 配置 | `configs/embedding/embedding_query_augmented_eval_v1.yaml` |
| 配置 SHA256 | `8f1dafeff094bdc64b2353d06dc265c3bbaae3aa0d291add8d3c36533920603a` |
| 原始 POI 向量 SHA256 | `ab286a8932196f84bbda5dc1965202b9aedadbd770e3e38cf78e5e1a63184c90` |
| POI ID SHA256 | `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7` |
| 固定评测 Query 向量 SHA256 | `98e2032b86b76cf42991dc1d1f4967ff456661fb32fbae185652163c85bf70e0` |
| E0 召回结果 SHA256 | `8052a40b64b6b875c61a596388a764644042bb701936a5394eda4d9ff4eb3a9c` |
| E1/E2 聚合 manifest SHA256 | `57b21716b3c171f5b5641e1abb90d4e06217c227abb7402e2525d818d60547bd` |

E1/E2 聚合覆盖 491,213 个 POI、维度 1,024，使用 256 个 Query 哈希分片顺序累积。E1 聚合 SHA256 为 `d8c3fb59d96e048e2d2f78f5324969df6b76383d6b5611df7bd2c6b312e34e42`，E2 聚合 SHA256 为 `a580591e5aeaaa5f2df78c6f26d6fd68200178a1ca79c1fcec0eb78b5fbb367a`。全量聚合耗时 321.73 秒、峰值 RSS 6,877.01 MiB；两个输出均为 float16，完整有限值、POI 行唯一性、订单计数和不同 Query 计数守恒全部通过。实现只在 RAM 中保留 float32 累积器，最终产物和 staging 均位于项目 `outputs/`，未使用开发机系统临时目录。

正式评测在开发机单卡 RTX A6000 48GB 上顺序运行 12 组 `GpuIndexFlatIP`。每组只构建一个 float32 索引，分两批加入 1,000,000/1,337,178 个 POI，检索完成后立即释放；12 组索引构建与检索合计 908.91 秒，观测最大显存增量 10.43 GiB，最低剩余显存 34.31 GiB，退出后显存占用回到 0 MiB。正式命令为：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ofs/map_search/hudan/envs/poi-gr
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python scripts/embedding/evaluate_query_augmented.py \
  --config configs/embedding/embedding_query_augmented_eval_v1.yaml
```

#### 正式结果

| 方法 | $\alpha$ | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0 | 11.99% | 25.25% | 32.39% | 40.25% | 17.7627% | 21.2113% |
| E1 | 0.05 | 15.16% | 29.76% | 38.01% | 46.77% | 21.5645% | 25.4415% |
| E1 | 0.10 | 16.84% | 33.21% | 42.70% | 51.58% | 24.0693% | 28.4595% |
| E1 | 0.20 | 17.94% | 36.83% | 47.18% | 57.77% | 26.1315% | 31.0948% |
| **E1** | **0.30** | **17.51%** | **36.98%** | **48.27%** | **58.90%** | **26.0006%** | **31.2401%** |
| E1 | 0.40 | 16.48% | 36.03% | 47.24% | 58.40% | 24.7998% | 30.0744% |
| E1 | 0.50 | 15.47% | 34.14% | 46.18% | 57.71% | 23.6231% | 28.9037% |
| E2 | 0.05 | 15.40% | 30.28% | 38.82% | 47.53% | 21.9590% | 25.9322% |
| E2 | 0.10 | 17.52% | 34.68% | 43.85% | 53.26% | 24.9789% | 29.4391% |
| E2 | 0.20 | 19.21% | 39.17% | 49.69% | 59.77% | 27.8455% | 33.0041% |
| **E2** | **0.30** | **18.53%** | **40.02%** | **51.18%** | **62.10%** | **27.7598%** | **33.2917%** |
| E2 | 0.40 | 17.99% | 39.30% | 51.05% | 62.32% | 27.0275% | 32.6826% |
| E2 | 0.50 | 17.01% | 38.04% | 50.36% | 61.53% | 26.0950% | 31.7990% |

E1 最优点为 $\alpha=0.30$；E2 最优点同样为 $\alpha=0.30$，并成为本轮整体最优。E2 相对 E0 的 Hit@10、MRR@10、NDCG@10 分别提高 18.79、9.9971、12.0804 个百分点；相对 E1 最优分别再提高 2.91、1.7592、2.0516 个百分点。逐 Query 以未命中记作第 21 名时，E2 最优相对 E0 为 4,390 胜、4,912 平、698 负；E1 最优为 3,976 胜、5,238 平、786 负。

2,000 次固定种子成对 bootstrap 的 95% 区间均不跨 0：E2 相对 E0 的 Hit@10、MRR@10、NDCG@10 提升区间分别为 `[17.9600, 19.6205]`、`[9.5176, 10.5182]`、`[11.5826, 12.5811]` 个百分点；E2 相对 E1 的对应区间为 `[2.4900, 3.3100]`、`[1.5699, 1.9558]`、`[1.8734, 2.2365]` 个百分点。

#### 诊断与结论

- Validation 的业务订单键与 Train 完全不重叠，但历史地图检索中 Query 文本自然会重复。10,000 条评测中 8,399 条 Query 文本曾在 Train 出现，7,691 条精确 `Query–目标 POI` 对曾出现；这属于历史正反馈增强的目标场景，不等同于订单泄漏。
- 收益不只来自精确 Query 复现。在 1,601 条 Train 未见过相同 Query 文本的评测上，E2 仍把 Hit@10 从 55.59% 提高到 63.21%，NDCG@10 从 41.24% 提高到 49.98%；在 2,309 条未见过相同 `Query–目标 POI` 对的样本上，Hit@10 从 44.09% 提高到 52.14%，NDCG@10 从 31.22% 提高到 38.87%。
- E2 在全部六个固定 $\alpha$ 上均优于同系数 E1，证明订单频次与跨 POI DF 加权不是无效复杂化。
- 固定系数存在清晰峰值，$\alpha\ge 0.40$ 后主指标下降；Query 聚合不能取代内容向量，只能作为受控残差。
- 不同 Train Query 数分桶的最优强度不一致：1—5 与 6—20 桶在较大 $\alpha$ 上继续受益，$>20$ 桶在 $\alpha=0.20$ 达峰。这说明头部 POI 的大量异质 Query 会引入泛语义，下一步应做异质性感知收缩，而不是让 $\alpha_i$ 随覆盖数单调增加。
- 本轮只证明 Query 增强对连续向量精确召回有效，尚未证明它能改善 RQ-KMeans/RQ-VAE 的码本利用率、SID 唯一率或生成式检索指标。

#### 产物与下一步

正式汇总位于 `outputs/evaluation/sft_validation_10k_v1/query_augmented_bge_m3_v1/summary.json`，SHA256 为 `ca90786281d6dde5bd239140cd291e0eff5ff6190874f00c54bf6ad43bf913af`；精确 Query 重复与 bootstrap 诊断保存为同目录 `analysis.json`，SHA256 为 `6206b1295be2194fce6c37ee593690e07c3dabe158c5a878f4e1e231e705c638`。E1/E2 最优召回结果 SHA256 分别为 `56446d2a8c238993abd6507406329b6e8e68739f3ef8c3a43d111992a722e135` 和 `1b7b8b81e8b2c8807c22298b2de8eab33fa155ffad8b982a451b08c7d81fa964`。12 组结果、指标和运行 manifest 总计约 16 MiB，没有保存重复的全量融合向量。

E3 的正式负结果和 E4 的两次正式实验见后续各节。边界补充已在 $\beta=1.00$ 观察到回落，当前连续向量正式冻结为 E4/$(\alpha,\beta)=(0.30,0.85)$；下一步进入 RQ-KMeans 与现有 RQ-VAE 的同输入对比。

### EXP-20260807-06：E3 异质性感知逐 POI 融合

#### 目标与假设

E1/E2 的不同 Query 数分桶在不同固定 $\alpha$ 上达峰。E3 检验能否只用 Train 统计得到的 Kish 有效 Query 数 $n_i^{\mathrm{eff}}$，为每个有 Query 的 POI 分配较强或较弱的 Query 残差：权重越分散、有效 Query 数越大，$\alpha_i$ 越接近 $\alpha_{\min}$，从而保留更多内容语义。主选择指标仍预先固定为 NDCG@10，平局依次比较 Hit@10、MRR@10 和配置中的候选顺序。

#### 数据、配置与执行

| 项目 | 冻结值 |
|---|---|
| Experiment | `EXP-20260807-06` |
| Git commit / 工作区 | `e4515195097dcb10aac0f3281ee7ddc0e482886c` / dirty |
| 配置 | `configs/embedding/embedding_query_adaptive_eval_v1.yaml` |
| 配置 SHA256 | `053d7748028ec2c5b8aeb85edb1a333f9de23ae51287e728ad67904f1a1a24c2` |
| E1/E2 聚合 manifest SHA256 | `57b21716b3c171f5b5641e1abb90d4e06217c227abb7402e2525d818d60547bd` |
| E3 异质性特征 manifest SHA256 | `980c9e5b24cc86f4e5b2a9fae3946f287511d393bb3984c529cdc05e8073c9d0` |
| E2/0.30 对照召回 SHA256 | `1b7b8b81e8b2c8807c22298b2de8eab33fa155ffad8b982a451b08c7d81fa964` |

异质性特征覆盖 491,213 个 POI。$n_i^{\mathrm{eff}}$ 的最小值、中位数、P90、P99、最大值和均值分别为 1.0000、1.9879、7.7292、25.4273、691.5038 和 3.5738；E2 内容向量与 Query 聚合向量余弦的中位数为 0.6930。特征构建顺序复用 `covered_poi_rows.npy`，权重和相对冻结 E2 的最大误差为 $7.887\times10^{-7}$，完整 shape、dtype、有限值、边界与文件哈希校验均通过。

六组候选固定 $\alpha_{\min}=0.20$、$\alpha_{\max}=0.45$，扫描 $\tau\in\{5,10,20\}$ 与 $\gamma\in\{1,2\}$。正式评测在开发机单卡 RTX A6000 48GB 上顺序执行，仍使用 2,337,178 条 POI 的 float32 `GpuIndexFlatIP`。六组总运行 1,396.23 秒，其中建库 1,262.86 秒、检索 46.96 秒；最大显存增量 10.43 GiB、最低剩余 34.31 GiB，退出后显存回到 0 MiB。正式命令为：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ofs/map_search/hudan/envs/poi-gr
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/ofs/map_search/hudan/poi_genret/outputs/evaluation/sft_validation_10k_v1/query_adaptive_bge_m3_v1/runtime/tmp
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python scripts/embedding/evaluate_query_augmented.py \
  --config configs/embedding/embedding_query_adaptive_eval_v1.yaml
```

#### 正式结果

| 方法 | $\tau$ | $\gamma$ | 平均 $\alpha_i$ | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | - | - | 0 | 11.99% | 20.69% | 25.25% | 32.39% | 40.25% | 17.7627% | 21.2113% |
| **E2 固定对照** | - | - | **0.3000** | **18.53%** | **32.52%** | **40.02%** | **51.18%** | **62.10%** | **27.7598%** | **33.2917%** |
| E3 | 5 | 1 | 0.3682 | 13.80% | 23.43% | 29.50% | 38.53% | 48.61% | 20.5476% | 24.7763% |
| E3 | 10 | 1 | 0.3970 | 14.24% | 24.06% | 30.16% | 39.41% | 49.17% | 21.1013% | 25.4058% |
| **E3 组内最优** | **20** | **1** | **0.4180** | **14.85%** | **25.28%** | **31.61%** | **41.13%** | **50.65%** | **22.0580%** | **26.5432%** |
| E3 | 5 | 2 | 0.3884 | 12.15% | 20.84% | 26.39% | 34.68% | 44.20% | 18.2658% | 22.1225% |
| E3 | 10 | 2 | 0.4202 | 13.52% | 22.73% | 28.49% | 36.78% | 45.64% | 19.9045% | 23.8770% |
| E3 | 20 | 2 | 0.4378 | 14.91% | 25.02% | 31.01% | 39.72% | 48.87% | 21.8154% | 26.0384% |

E3 组内最优 `kish_t20_g1` 相对 E2/0.30 的 Hit@1/3/5/10/20、MRR@10 和 NDCG@10 分别下降 3.68、7.24、8.41、10.05、11.45、5.7018 和 6.7485 个百分点。逐 Query 以未命中记作第 21 名时，E3 相对 E2 仅 454 胜、6,009 平、3,537 负，因此该回退不是少量异常样本造成。

#### 诊断与结论

- E3 并非实现错位。异质性特征与 E2 聚合复用同一 `covered_poi_rows.npy` 行序并通过完整哈希校验；新增等价性测试确认，当逐 POI $\alpha$ 数组全部为 0.30 时，融合输出与标量 0.30 逐元素完全一致。
- E2 的分桶结果是在“全库统一一个 $\alpha$”条件下获得，不能解释为每个目标分桶可独立选择自己的 $\alpha_i$。每个 POI 既可能是正目标，也同时是其他 9,999 条 Query 的负候选；逐候选使用不同变换会改变全库的相对几何关系。
- 最优 E3 在 Train 不同 Query 数为 1—5 的 1,535 条样本上，Hit@10 从 E2 的 73.36% 小幅升至 73.75%；但在 $>20$ 的 5,239 条头部样本上从 39.24% 降至 22.60%，NDCG@10 从 22.96% 降至 12.77%。高异质目标降低 Query 权重后，会被仍以高 Query 权重编码的大量低异质候选压过。
- 因此拒绝“候选级动态 $\alpha_i$ + 单一全库余弦索引”作为 Embedding/SID 输入。若未来仍研究动态置信度，应放在固定 E2 首召回后的 Top-K 校准或重排中，不能直接替换全库向量。
- E3 本身不进入候选。后续 E4 已在统一 $\alpha$ 下对 Query 聚合本身做类别残差，并验证该修正能够改善全库相对几何关系。

#### 产物

正式结果位于 `outputs/evaluation/sft_validation_10k_v1/query_adaptive_bge_m3_v1/`。`summary.json` SHA256 为 `563b7acd680ddb7d737f2f4d83f2e4544da6264f8f1b24f627d7b984c087c40b`；组内最优 `kish_t20_g1/retrieval_results.npz` SHA256 为 `a7b073d76904831265c4235a0d350be53643bb76aa5bcbcb041b1dc6c947e2d1`。六组均保存独立指标、Top-20、逐目标 rank、实际 $\alpha_i$ 和运行 manifest，未保存六份全量融合 POI 向量。

### EXP-20260808-01：E4 类别公共 Query 方向残差

#### 目标与假设

E2 的 Query 聚合同时包含“餐饮、酒店、商场”等类别公共表达和具体 POI 的别名、楼牌与局部属性。E4 检验：在保持所有候选统一 $\alpha=0.30$ 的前提下，从每个 POI 的 E2 Query 聚合中减去一部分 Train-only 类别公共方向，能否缓解同类候选拥挤并保留 POI 特有检索语义。主选择指标在运行前固定为 NDCG@10，平局依次比较 Hit@10、MRR@10 与冻结候选顺序；$\beta$ 固定扫描 $\{0.10,0.25,0.50,0.75\}$。

#### 数据、类别统计与配置

| 项目 | 冻结值 |
|---|---|
| Experiment | `EXP-20260808-01` |
| Git commit / 工作区 | `e4515195097dcb10aac0f3281ee7ddc0e482886c` / dirty |
| 类别统计配置 | `configs/embedding/embedding_query_category_residual_bge_m3_v1.yaml` |
| 类别统计配置 SHA256 | `c23d4d049b5e48a591743e315b34832175673e275ab6d0ad6d6db362c79af678` |
| 正式评测配置 | `configs/embedding/embedding_query_category_residual_eval_v1.yaml` |
| 正式评测配置 SHA256 | `9ea5965ba70700ab359e2e5730adf543c1897676ed35cb5bd24ddb9e479e45be` |
| E4 类别统计 manifest SHA256 | `59004d8dd2dc74c059450eb858e2bbb63ffd94c88b53dc525c886007448f2d7f` |
| 类别行号 SHA256 | `5d895b0f79192fa4d82f6f158ded85d93d24e72e2e677f26475d08978c61e7a5` |
| E2/0.30 对照召回 SHA256 | `1b7b8b81e8b2c8807c22298b2de8eab33fa155ffad8b982a451b08c7d81fa964` |

类别输入来自已冻结的 GNPR 全量 POI 静态类别，共 402 类。491,213 个有 Train Query 聚合的 POI 覆盖 392 个非空类别；以 32 个覆盖 POI 为最小支持度后，290 个类别、489,968 个 POI 使用类别中心，剩余 1,245 个 POI 的类别中心置零。支持类别的原始中心范数最小值、中位数、P90、最大值和均值分别为 0.5911、0.6624、0.7272、0.8929 和 0.6694。

类别中心只保存一个 `[402,1024]` float32 NPY，检索器按 POI mmap/流式读取 E2 聚合并在线残差化，没有落盘四份约 1 GiB 的稠密候选向量。正式评测在单卡 RTX A6000 上顺序执行四组 2,337,178 POI 的 float32 `GpuIndexFlatIP`；四组总耗时 344.41 秒，最大显存增量 10.43 GiB、最低剩余 34.31 GiB，结束后显存占用为 0 MiB。

#### 正式结果

| 方法 | $\alpha$ | $\beta$ | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0 | - | 11.99% | 20.69% | 25.25% | 32.39% | 40.25% | 17.7627% | 21.2113% |
| E2 固定对照 | 0.30 | 0 | 18.53% | 32.52% | 40.02% | 51.18% | 62.10% | 27.7598% | 33.2917% |
| E4 | 0.30 | 0.10 | 19.00% | 32.52% | 40.41% | 51.47% | 62.42% | 28.0920% | 33.6112% |
| E4 | 0.30 | 0.25 | 19.22% | 33.12% | 41.10% | 52.06% | 63.17% | 28.4845% | 34.0571% |
| E4 | 0.30 | 0.50 | 19.63% | 34.23% | 42.13% | 54.03% | 65.22% | 29.3318% | 35.1638% |
| **E4 冻结候选最优** | **0.30** | **0.75** | **19.95%** | **34.94%** | **43.41%** | **55.82%** | **66.71%** | **30.0583%** | **36.1412%** |

最优 E4 相对 E2 的 Hit@1/3/5/10/20、MRR@10 和 NDCG@10 分别提高 1.42、2.42、3.39、4.64、4.61、2.2985 和 2.8495 个百分点。逐 Query 成对排名为 2,760 胜、6,453 平、787 负；相对 E0 为 4,854 胜、4,478 平、668 负。

#### 诊断与结论

- 四个 $\beta$ 的总体主指标单调上升，说明 E2 中确有较强的类别公共方向；将其部分去除后，同类候选之间更依赖 POI 特有 Query 语义。
- 收益主要集中在短 Query 与 Query 较多的头部 POI。长度 1—2 的 Hit@10 从 E2 的 31.92% 升至 37.86%，拥有 $>20$ 个不同 Train Query 的目标从 39.24% 升至 46.42%。
- 拥有 1—5 个不同 Train Query 的目标出现轻微回退，Hit@10/NDCG@10 从 73.36%/54.95% 变为 72.64%/54.46%；长度 $>10$ 的 NDCG@10 也从 56.39% 小幅降至 55.78%。E4 改善的是总体候选几何，不代表每个子集都受益。
- 307 条目标 POI 无 Train Query 的样本，其目标向量虽然保持 E0 不变，但 Hit@10 仍由 E2 的 32.57% 升至 39.41%，进一步说明移动负候选本身会改变检索结果；这与 E3 暴露的“候选间相互竞争”机制一致。
- 当前最优 $\beta=0.75$ 位于预设搜索上界。因此 E4 方向已通过，但本实验只能冻结“已评测候选内最优”，不能声称 0.75 是全局最优。若补充靠近完全类别中心化的候选，应作为预注册的后续 Validation 实验记录；最终论文结论仍需在未参与参数选择的 Test 上确认。

#### 产物与下一步

类别统计位于 `outputs/embeddings/query_augmented_bge_m3/query_category_residual_v1/`，其中 `category_query_means.npy` SHA256 为 `42b2f9800f39aec6ed7fb94019f4351d959c6ba8660b25fea04dd5bdaf047491`。正式召回位于 `outputs/evaluation/sft_validation_10k_v1/query_category_residual_bge_m3_v1/`；`summary.json` SHA256 为 `b76d20fbab642412ec8d1b962f6a288058f785f0338e9328b95e12f95f7a044b`，最优 `e4_beta_0p75/alpha_0p30/retrieval_results.npz` SHA256 为 `d9040ded85caad1a72d8f03e9e30cd4567d3e85f7323fdb09527b486d8bfd9b6`。

本实验当时将连续向量候选更新为 E4/$(\alpha,\beta)=(0.30,0.75)$。后续 `EXP-20260808-02` 已执行靠近 $\beta=1$ 的边界补充并将最终值冻结为 0.85；不再回到 E3 的候选级动态 $\alpha$。

### EXP-20260808-02：E4 类别残差边界补充与最终冻结

#### 目标与预注册候选

`EXP-20260808-01` 的四个指标点随 $\beta$ 单调上升，且最优点 0.75 位于预设上界。本实验在读取任何新增召回结果前固定两个候选：$\beta=0.85$ 用于观察 0.75 后的局部趋势，$\beta=1.00$ 对应完整减去类别均值。主指标继续使用 NDCG@10，平局顺序不变；全库融合系数仍固定为 $\alpha=0.30$，参考结果固定为 E4/0.75。

运行前只使用 Train 聚合检查完整中心化的数值安全性。491,213 个残差均为有限非零向量，范数最小值、P1、中位数、P99、最大值和均值分别为 0.2106、0.5728、0.7499、0.9277、1.0670 和 0.7530，因此允许 $\beta=1.00$ 进入正式评测。

#### 配置与执行

| 项目 | 冻结值 |
|---|---|
| Experiment | `EXP-20260808-02` |
| 类别统计配置 | `configs/embedding/embedding_query_category_residual_boundary_bge_m3_v1.yaml` |
| 类别统计配置 SHA256 | `a059a1371dd86e28d563895f2b8b5450e1f2fdf24d2c26b1ba63aeb8a5473db5` |
| 边界统计 manifest SHA256 | `ae65ec6bd8fda28228822a080fb65a131d933d68acc0ae6aae903c46f8b303d8` |
| 正式评测配置 | `configs/embedding/embedding_query_category_residual_boundary_eval_v1.yaml` |
| 正式评测配置 SHA256 | `7de9a8e8ec0d10d411c259b40ed9f8c62bce6d45faeff73b53774b210347ca85` |
| E4/0.75 参考召回 SHA256 | `d9040ded85caad1a72d8f03e9e30cd4567d3e85f7323fdb09527b486d8bfd9b6` |

边界统计重新执行完整输入哈希门禁，生成的 `category_query_means.npy` 与主 E4 逐字节相同，SHA256 均为 `42b2f9800f39aec6ed7fb94019f4351d959c6ba8660b25fea04dd5bdaf047491`。两组精确检索在开发机单卡 RTX A6000 上顺序执行，总运行 215.18 秒；最大显存增量 10.43 GiB、最低剩余 34.31 GiB，任务结束后显存占用回到 0 MiB。

#### 正式结果与最终选择

| 方法 | $\alpha$ | $\beta$ | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2 固定对照 | 0.30 | 0 | 18.53% | 32.52% | 40.02% | 51.18% | 62.10% | 27.7598% | 33.2917% |
| E4 主实验最优 | 0.30 | 0.75 | 19.95% | 34.94% | 43.41% | 55.82% | 66.71% | 30.0583% | 36.1412% |
| **E4 最终冻结** | **0.30** | **0.85** | **20.02%** | **34.92%** | **43.76%** | **56.37%** | **66.78%** | **30.2279%** | **36.3976%** |
| E4 完整中心化 | 0.30 | 1.00 | 19.88% | 34.88% | 43.78% | 56.27% | 65.97% | 30.0911% | 36.2709% |

$\beta=0.85$ 相对 0.75 的 Hit@1/3/5/10/20、MRR@10 和 NDCG@10 变化为 +0.07、-0.02、+0.35、+0.55、+0.07、+0.1696 和 +0.2564 个百分点；逐 Query 为 932 胜、8,556 平、512 负。相对 E2，其 Hit@10/NDCG@10 总提升达到 5.19/3.1059 个百分点。

$\beta=1.00$ 相对 0.85 的 Hit@10、Hit@20、MRR@10 和 NDCG@10 分别回落 0.10、0.81、0.1368 和 0.1267 个百分点，说明完全去除类别中心开始损失有用的类别语义。0.75、0.85、1.00 已在最优点两侧形成局部包围，不再继续增加或细扫 $\beta$。

分桶同样显示这一取舍：0.85 相对 0.75 继续改善长度 1—2 Query 的 Hit@10（37.86%→38.76%）和 $>20$ 个不同 Train Query 目标的 Hit@10（46.42%→47.47%），但长度 $>10$ 的 NDCG@10 从 55.78% 降至 55.08%。因此 0.85 是整体主指标最优，不代表每个子集均最优。

#### 产物与结论

正式汇总位于 `outputs/evaluation/sft_validation_10k_v1/query_category_residual_boundary_bge_m3_v1/summary.json`，SHA256 为 `b4ad04be5b39e11d44d45813c32c54c5e97e05bd07432cdc990a5e4591a59d31`；最终 0.85 召回 NPZ SHA256 为 `d54662d1cdc0c8932b21a1040e2f524ef875c6e4ceb4ff6aaac3546a48a80d3d`，1.00 召回 NPZ SHA256 为 `a197bfd628a019f544db52e078d31624b0be88fcbe2156e3e3aa97107e2ed7f9`。

E4 连续向量现正式冻结为 $\alpha=0.30,\beta=0.85$。后续量化器比较必须复用该表示、相同 POI 行序和输入指纹，不再利用当前 Validation 继续微调 $\alpha$ 或 $\beta$；最终论文指标仍需由未参与选择的 Test 确认。
