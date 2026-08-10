# RQ-KMeans SID 实验

本文件记录北京全量 POI 上的 RQ-KMeans 构建协议、参数筛选、与 TIGER RQ-VAE 的公平对照，以及相同 30-bit 容量下的非对称码本消融。原始 BGE-M3 与 E4 Query-Augmented Embedding 在对称 `1024³` 下的 RQ-VAE/RQ-KMeans `2×2` 已全部完成；两种输入的三档 30-bit RQ-KMeans 布局也已闭环，并据此冻结下游 SFT 候选。

## 1. 当前结论

1. 在完全相同的原始 BGE-M3 输入、POI 行序、三层 `1024×1024×1024` 码本和固定 Validation 划分下，标准贪心 RQ-KMeans 的全量唯一 SID 比例为 76.0725%，高于 TIGER RQ-VAE 的 71.6766%；碰撞 POI 比例从 40.7894% 降至 36.1501%。
2. 对称 RQ-KMeans 并非无代价地全面优于 RQ-VAE：depth-1/2/3 类别 micro-purity 比 TIGER 分别低 2.2555/0.3917/0.1881 个百分点，最大碰撞桶也从 306 增至 467。
3. 保持总编码容量均为 30 bit，把码本容量前移到第一层可以修复上述取舍。原始 BGE 的 `4096×1024×256` 唯一率达到 81.8938%，E4 输入进一步达到 82.7331%、碰撞 POI 降至 27.9655%，成为当前纯 SID 结构最优。
4. 从纯 SID 结构指标看，当前最优是 E4+`4096×1024×256`；从生成模型词表和第一 token 分类难度看，`2048×1024×512` 仍是重要的工程折中候选。二者必须再经过同一 SFT/生成评测后才能确定最终上线布局。
5. 主论文中的“RQ-KMeans vs RQ-VAE”必须继续使用对称 `1024³` 作为公平算法对照；两个非对称布局只能作为 RQ-KMeans 的容量分配消融，不能替换主对照。
6. 在完全相同的 TIGER `1024³` 协议下，把输入从原始 BGE 换为冻结 E4 后，唯一 SID 比例由 71.6766% 提升到 74.3539%，碰撞 POI 由 40.7894% 降到 37.4194%，第一层实际使用数由 561 增至 751；但 depth-1/2 类别 micro-purity 分别下降 5.1448/2.3896 个百分点。E4 的收益是“Query 意图区分度与码本展开”，不是静态类别树更纯。
7. 完整 `2×2` 表明 Embedding 与 Quantizer 的结构收益存在明显重叠，而非近似相加：E4 在 RQ-VAE 下把唯一率提高 2.6774 个百分点，在 RQ-KMeans 下只提高 0.0848 个百分点；但 E4 仍把 RQ-KMeans 最大碰撞桶从 467 降至 288。当前对称配置的全局最优是 E4+RQ-KMeans，唯一率 76.1573%、碰撞 POI 35.7477%，但其主要新增价值是修复长尾碰撞。
8. E4 的 `4096×1024×256` 迁移验证为正：相对同布局 BGE 唯一率提高 0.8392 个百分点、碰撞 POI 降低 1.1857 个百分点；相对 E4 `1024³` 唯一率提高 6.5758 个百分点。Embedding 与“容量前移”存在正交互，但最大桶仍由 BGE 的 96 增至 E4 的 125，长尾碰撞尚未被完全解决。
9. E4 `2048×1024×512` 的唯一率/碰撞 POI 为 79.8731%/31.4021%，最大桶为 143，是当前结构收益、基础词表规模和第一 token 难度之间的 Pareto 拐点。正式新增 SFT 候选冻结为 E4-2048 主模型、E4-4096 容量上界和 BGE-2048 同布局 Embedding 消融；E4-1024 被 E4-2048 支配，不进入新增正式 SFT。

## 2. 方法与公开实现依据

令原始 POI 向量为 $x\in\mathbb{R}^{1024}$，初始残差为 $r_0=x$。第 $m$ 层码本为 $C_m$，逐层执行：

$$
c_m=\arg\min_{j\in\{1,\ldots,K_m\}}
\left\|r_{m-1}-C_m[j]\right\|_2^2
$$

$$
r_m=r_{m-1}-C_m[c_m]
$$

最终 SID 为 $(c_1,c_2,c_3)$，重构向量为：

$$
\hat{x}=\sum_{m=1}^{3}C_m[c_m]
$$

主实验使用标准 sequential residual K-Means：每层只对上一层残差重新做 K-Means，不加入 balance penalty、类别监督、Query 监督或 OPQ。实现参考以下公开路径：

- [Faiss Additive Quantizers](https://github.com/facebookresearch/faiss/wiki/Additive-quantizers)：Residual Quantizer 的逐层残差定义、beam search 和 progressive-dimension 训练说明；
- [Faiss `ResidualQuantizer` API](https://faiss.ai/cpp_api/struct/structfaiss_1_1ResidualQuantizer.html)：`niter`、`nredo`、`max_beam_size`、refinement 和内存控制参数；
- [RecTokens](https://github.com/EdoardoBotta/RecTokens)：公开的流式/mini-batch RQ-KMeans 与 K-Means++ 路径；
- [OneRec 技术报告](https://arxiv.org/abs/2506.13695)：推荐系统中 RQ-KMeans 与 RQ-VAE 的重构、利用率和熵对照；
- OneSearch 本地论文：相同总容量下比较 `1024-1024-1024`、`2048-1024-512`、`4096-1024-256`，并提示全层强制 balance 可能损伤层级结构。

代码同时支持 Faiss 官方 `IndexResidualQuantizer` 的 progressive-dimension、beam 和 refinement，但该 CPU 路径在北京 1024 维数据上不适合做参数网格。本轮正式结果采用 Faiss GPU K-Means 组成的标准贪心 RQ，`max_beam_size=1`。

## 3. 冻结数据与公平协议

| 项目 | 冻结值 |
|---|---|
| POI 输入 | 原始 BGE-M3，`outputs/embeddings/beijing_poi_bge_m3/embeddings.npy` |
| POI 数 / 维度 | 2,337,178 / 1,024 |
| Embedding fingerprint | `d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071` |
| POI ID SHA256 | `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7` |
| Validation | 固定 1%，23,372 条 |
| Validation indices SHA256 | `409e476abf82cd609447c6c9a6f437d96ecd897a40c5e610a33a3bc996041249` |
| 正式 K-Means 样本 | Train-only 固定 500,000 条 |
| Sample indices SHA256 | `0f7f22d7a7543965d581bcc9098a833533dbbdbb701b867f54c76d0bf66d6eb7` |
| TIGER 对照 | `outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/evaluations/epoch_20/` |
| GPU | 开发机单卡 NVIDIA RTX A6000 48GB |
| Python 环境 | `/ofs/map_search/hudan/envs/poi-gr` |

RQ-KMeans 直接量化 1024 维 BGE 向量，不增加 latent/PCA 投影。全量 embedding 始终以 mmap 和 8,192 行分块读取；50 万训练样本残差仅在受控 RAM 中保存。实际观测进程峰值 RSS 约 7.1 GiB，GPU 显存远低于 48GB。所有缓存、日志和产物均位于项目 `outputs/`，未使用开发机系统临时目录。

## 4. 参数筛选

固定 Validation 上先比较迭代次数、样本量和随机初始化重复次数，再固定优化器参数比较码本布局。以下碰撞仅用于低成本筛选，最终结论以后续 2,337,178 条全量结果为准。

| 配置 | 唯一 SID | 碰撞 POI | 最大桶 | L1/L2/L3 利用率 | MSE/维 | 重构余弦 |
|---|---:|---:|---:|---:|---:|---:|
| `1024³, 100k, iter=10, nredo=1` | 99.0031% | 1.9382% | 5 | 98.44%/70.61%/42.09% | $2.11146\times10^{-4}$ | 0.885063 |
| `1024³, 100k, iter=20, nredo=1` | 99.0887% | 1.7585% | 5 | 98.44%/70.12%/42.97% | $2.08919\times10^{-4}$ | 0.886339 |
| `1024³, 500k, iter=20, nredo=1` | 99.0202% | 1.8484% | 6 | 99.80%/84.47%/76.56% | $1.92099\times10^{-4}$ | 0.896017 |
| `1024³, 500k, iter=20, nredo=3` | 99.0501% | 1.7927% | 5 | 99.80%/87.21%/77.44% | $1.91242\times10^{-4}$ | 0.896490 |
| `2048×1024×512, 500k, iter=20, nredo=3` | 99.3796% | 1.2151% | 4 | 98.34%/84.67%/80.47% | $1.88211\times10^{-4}$ | 0.898177 |
| `4096×1024×256, 500k, iter=20, nredo=3` | 99.5764% | 0.8386% | 3 | 92.97%/79.20%/78.52% | $1.83920\times10^{-4}$ | 0.900535 |

筛选结论：

- `iter=20` 相对 10 次仍有稳定收益，但第 20 次附近目标函数已进入明显的边际递减区间；不沿用 RQ-VAE 的“20 epoch”概念，二者只是数值相同、含义完全不同。
- 50 万样本显著改善重构、L2/L3 覆盖与熵，但碰撞率并不随重构误差单调变化，因此不能只按 MSE 选择 SID。
- `nredo=3` 相对 1 次重启带来小而稳定的 MSE、碰撞和深层利用率收益；这是一次性离线成本，正式配置保留 3 次重启。
- 固定优化器后，第一层容量从 1024 增至 2048/4096 同时改善重构与碰撞，说明北京全量 POI 的容量瓶颈主要位于语义主干 S1。

TIGER RQ-VAE epoch 20 在同一 Validation 上的重构 MSE/维为 $2.10859\times10^{-4}$，余弦为 0.885160；对称 RQ-KMeans `500k/iter20/nredo3` 分别为 $1.91242\times10^{-4}$ 和 0.896490。

## 5. 全量 SID 正式结果

三个 RQ-KMeans 布局均已对 2,337,178 条 POI 完成全量流式编码，并复用统一 SID evaluator 计算碰撞、逐层利用率/熵、前缀类别纯度与碰撞案例。

三个布局的总组合容量均相同：

$$
\log_2 1024+\log_2 1024+\log_2 1024=30
$$

$$
\log_2 2048+\log_2 1024+\log_2 512=30
$$

$$
\log_2 4096+\log_2 1024+\log_2 256=30
$$

### 5.1 碰撞、利用率与熵

| 方法 | 总 bit | 唯一 SID | 碰撞 POI | 最大桶 | L1/L2/L3 使用数 | L1/L2/L3 归一化熵 |
|---|---:|---:|---:|---:|---:|---:|
| TIGER RQ-VAE `1024³` | 30 | 71.6766% | 40.7894% | 306 | 561/1024/1024 | 0.8578/0.9693/0.9729 |
| RQ-KMeans `1024³` | 30 | 76.0725% | 36.1501% | 467 | 1024/1024/1024 | 0.9751/0.9378/0.9202 |
| RQ-KMeans `2048×1024×512` | 30 | 78.9451% | 32.7653% | 163 | 2048/1024/512 | 0.9761/0.9369/0.9124 |
| **RQ-KMeans `4096×1024×256`** | **30** | **81.8938%** | **29.1512%** | **96** | **4096/1024/256** | **0.9747/0.9298/0.9080** |

### 5.2 前缀类别纯度

| 方法 | Depth-1 micro | Depth-2 micro | Depth-3 micro | Depth-1 macro | Depth-2 macro | Depth-3 macro |
|---|---:|---:|---:|---:|---:|---:|
| TIGER RQ-VAE `1024³` | **69.2751%** | 77.2651% | 96.2830% | 61.8934% | 78.1823% | 98.2100% |
| RQ-KMeans `1024³` | 67.0196% | 76.8735% | 96.0949% | 64.4349% | 81.6343% | 98.1632% |
| RQ-KMeans `2048×1024×512` | 68.8996% | 80.2594% | 96.7587% | **65.6932%** | 85.4746% | 98.4364% |
| **RQ-KMeans `4096×1024×256`** | 68.8522% | **83.3590%** | **97.1629%** | 65.4931% | **88.6955%** | **98.6111%** |

对称 RQ-KMeans 的浅层类别 micro-purity 低于 TIGER，证明“直接 K-Means 不坍塌”不等于“层级语义自动最优”。非对称布局把更多组合容量放到 S1 后，depth-1 与 TIGER 的差距缩小至约 0.4 个百分点，同时显著提高 depth-2/3 纯度并减少碰撞。这是当前最关键的正结果。

## 6. 关于码本是否均匀与公平性

“RQ-VAE 是均匀码本、RQ-KMeans 是不均匀码本”不符合现有结果：

- TIGER RQ-VAE 第一层只使用 561/1024 个码，归一化熵为 0.8578，本身并不均匀；
- 对称 RQ-KMeans 全量三层均使用全部 1024 个码，但 L2/L3 熵为 0.9378/0.9202，频次仍然不均匀；
- 非对称 RQ-KMeans 也使用全部码，但深层熵随残差结构自然下降。

因此，公平主对照控制的是输入、层数、每层名义码本大小、总 bit、划分和评估协议；利用率与频率分布是量化器要比较的结果，不能在训练前强制做成相同。主实验不加入 balance loss。若后续验证 OneSearch 风格的 L3-only balance，应作为独立增强消融，并同时报告前缀纯度；全层平衡不进入当前主配置。

非对称布局与对称 RQ-VAE 不是“只改变算法”的公平对照，因此单独列为容量分配消融。三种 RQ-KMeans 布局虽同为 30 bit，但生成模型新增 token 数分别为 3,072、3,584 和 5,376；`4096×1024×256` 的 SID 结构最优，不代表其 SFT 的第一 token 分类和线上时延必然最优。

## 7. 实现、命令与产物

核心实现：

- `src/poi_gr/sid/rqkmeans.py`
- `scripts/sid/build_rqkmeans.py`
- `configs/sid/rqkmeans_bge_m3_1024x3.yaml`
- `tests/sid/test_rqkmeans.py`

正式优化器参数为 `sample_size=500000`、`iterations=20`、`nredo=3`、`backend=faiss_gpu`、`implementation=sequential_kmeans`。命令按布局只覆盖输出目录和 `--codebook-sizes`，不为每个实验生成额外 shell 脚本。

| 布局 | 完整产物目录 | SID SHA256 | metrics SHA256 |
|---|---|---|---|
| `1024³` | `outputs/sid/rqkmeans/bge_m3/screen/gpu_greedy_1024x3_s500k_i20_r3/` | `85eb5da0fc0f73e2d45fefc6a40d5a5d743f9be11948a0a7f1b7527049f7ae07` | `c60e686806c609e414e75d1fa71184e5077b940783499ea9e398cf6593f66869` |
| `2048×1024×512` | `outputs/sid/rqkmeans/bge_m3/screen/gpu_greedy_2048x1024x512_s500k_i20_r3/` | `de3601cc58d369380766a4cab586759ff38e482d5080edbce0907e68394712a0` | `0d6e14bcd228c648c84268a2eb7c638a73487ff2c45cbb71642f38b30bca9449` |
| `4096×1024×256` | `outputs/sid/rqkmeans/bge_m3/screen/gpu_greedy_4096x1024x256_s500k_i20_r3/` | `9a970f9532f13b1b2e14d8ecce002ec96355b33d2f1db7a1c0db6c002c748ddc` | `52c80cb4377c4766fff170e00fef5f9c73454dcefa2e4b3407673a1c8bef3fb4` |

本轮代码基于 Git commit `e4515195097dcb10aac0f3281ee7ddc0e482886c` 的 dirty 工作区执行。冻结配置 SHA256 为 `388e46782dcf1795b121290f4372e3aaa13466432d1df2e8e502edf50d221379`。

工程上额外修复了 OrangeFS 的两个性能问题：训练样本残差保留在受控 RAM，不使用项目盘 memmap 逐层更新；28 MiB 全量 SID 结果在内存中生成后一次原子写出并同步计算 NPY SHA，避免 memmap 强制刷盘和写后冷读。全量 embedding 仍严格流式读取。

## 8. EXP-20260808-05 E4 + TIGER RQ-VAE `1024³`

### 8.1 目标与冻结输入

目标是先补齐 Embedding × Quantizer `2×2` 中的 `E4+RQ-VAE`，隔离向量输入本身对 RQ-VAE 稳定性、码本利用率、碰撞和层级语义的影响。量化器严格沿用原始 BGE-TIGER 协议，只替换输入 Embedding 与实验输出目录。

冻结 E4 使用已在固定 10,000 条无泄漏 Validation 上选定的参数：

$$
r_i=\operatorname{Normalize}\left(q_i-0.85\mu_{\operatorname{cat}(i)}\right)
$$

$$
z_i=\operatorname{Normalize}\left(0.70c_i+0.30r_i\right)
$$

有 Train Query 聚合的 491,213 条 POI 使用上述公式；其余 1,845,965 条 POI 保持原始 BGE 向量 $c_i$。全量向量按 32,768 行流式融合并以标准顺序 `.npy` 写入项目目录，不使用系统 `/tmp`；OrangeFS 上的实际冻结耗时为 160.70 秒。

| 项目 | 冻结值 |
|---|---|
| E4 全量向量 | `outputs/embeddings/query_augmented_bge_m3/e4_alpha0p30_beta0p85_full_v1/embeddings.npy` |
| Shape / dtype | `[2337178, 1024]` / `float16` |
| E4 fingerprint | `ba7904ff5b542bb3c09d66b8feca2b0fb6f7f43b1eedce49351dfd8a3658f49e` |
| 原始 BGE SHA256 | `ab286a8932196f84bbda5dc1965202b9aedadbd770e3e38cf78e5e1a63184c90` |
| E2 聚合 manifest SHA256 | `57b21716b3c171f5b5641e1abb90d4e06217c227abb7402e2525d818d60547bd` |
| E4 类别残差 manifest SHA256 | `ae65ec6bd8fda28228822a080fb65a131d933d68acc0ae6aae903c46f8b303d8` |
| 类别行号 SHA256 | `5d895b0f79192fa4d82f6f158ded85d93d24e72e2e677f26475d08978c61e7a5` |

冻结后重新计算首个 8,192 行窗口，结果与在线评测公式转为 `float16` 后逐元素完全一致；窗口内 1,780 条覆盖 POI 完成融合，未覆盖 POI 与原始 BGE 逐元素完全一致。全量训练门禁再次确认 2,337,178 条向量均为有限值、POI ID 全量唯一且行数对齐。

### 8.2 训练协议与稳定性

配置为 `configs/sid/rqvae_tiger_e4_bge_m3_1024x3.yaml`。除输入向量、Embedding manifest、实验 ID 和输出目录外，所有协议字段与 `configs/sid/rqvae_tiger_bge_m3.yaml` 的 `TIGER-BGE-M3-1024x3` 完全一致：

- Encoder/Decoder 为 `1024→512→256→512→1024`，latent dim 256；
- 三层码本均为 1,024，Faiss GPU K-Means 使用固定 Train-only 500,000 条、20 iterations、seed 42；
- batch 4,096、block 32,768、Adam 学习率 $3\times10^{-4}$、20 epoch、无 Diversity Loss、无 early stopping；
- 固定 Validation 为 23,372 条，索引与 K-Means 样本 SHA256 均与原 BGE-TIGER 相同；
- 单卡 NVIDIA RTX A6000 48GB，训练用时 717.05 秒，训练与导出退出码均为 0。

三层 K-Means 初始化均使用 1,024 个码。训练至 epoch 20 未触发任何坍塌预警：

| 输入 | Val MSE/维 | Val cosine | Val L1/L2/L3 使用数 | Val L1/L2/L3 熵 | 监控 SID 唯一率 |
|---|---:|---:|---:|---:|---:|
| 原始 BGE + RQ-VAE | $2.10859\times10^{-4}$ | 0.885160 | 529/1018/1024 | 0.8571/0.9651/0.9698 | 98.5367% |
| E4 + RQ-VAE | $2.30769\times10^{-4}$ | 0.873316 | 715/1022/1022 | 0.9048/0.9660/0.9671 | 98.7549% |

E4 的重构误差比原始 BGE 更高、余弦低 1.1845 个百分点，但第一层利用率与熵明显更高。这说明“更低重构误差”与“更少离散碰撞”不是同一目标；E4 增加了输入分布的 Query 意图区分度，RQ-VAE 用更多 S1 code 表达该分布。

### 8.3 全量 SID 与前缀语义

| 输入 × Quantizer | 唯一 SID | 碰撞 POI | 碰撞冗余 | P99 / 最大桶 | L1/L2/L3 使用数 | L1/L2/L3 熵 |
|---|---:|---:|---:|---:|---:|---:|
| BGE × RQ-VAE `1024³` | 71.6766% | 40.7894% | 28.3234% | 7 / 306 | 561/1024/1024 | 0.8578/0.9693/0.9729 |
| **E4 × RQ-VAE `1024³`** | **74.3539%** | **37.4194%** | **25.6461%** | **6 / 334** | **751/1024/1024** | **0.9061/0.9693/0.9714** |
| BGE × RQ-KMeans `1024³` | 76.0725% | 36.1501% | 23.9275% | 6 / 467 | 1024/1024/1024 | 0.9751/0.9378/0.9202 |

E4 相对原始 BGE 的 RQ-VAE 增加 62,575 个不同 SID，唯一率提高 2.6774 个百分点，碰撞 POI 减少 78,764 条、比例降低 3.3700 个百分点；P99 桶由 7 降至 6，但最大极端桶由 306 增至 334。E4+RQ-VAE 仍比 BGE+对称 RQ-KMeans 少 1.7185 个百分点的唯一 SID，因此在 E4+RQ-KMeans 完成前不能判断 Embedding 与 Quantizer 是否存在正交叠加收益。

| 输入 × Quantizer | Depth-1 micro | Depth-2 micro | Depth-3 micro | Depth-1 macro | Depth-2 macro | Depth-3 macro |
|---|---:|---:|---:|---:|---:|---:|
| BGE × RQ-VAE `1024³` | 69.2751% | 77.2651% | 96.2830% | 61.8934% | 78.1823% | 98.2100% |
| E4 × RQ-VAE `1024³` | 64.1303% | 74.8755% | 95.9466% | 55.8665% | 77.8006% | 98.1072% |
| BGE × RQ-KMeans `1024³` | 67.0196% | 76.8735% | 96.0949% | 64.4349% | 81.6343% | 98.1632% |

E4 的静态 `category_code` 前缀纯度下降，尤其 depth 1 micro/macro 分别下降 5.1448/6.0269 个百分点。该现象与 E4 的定义一致：类别公共 Query 方向被显式去除，而训练 Query 还包含门牌、别名、连锁分店和局部意图等不等同于静态类别的信号。这里不能得出“E4 语义变差”的笼统结论；可以确定的是它优化了 Query→POI 连续召回和 SID 区分度，但没有优化静态类别树。最终价值必须由同一 SFT 的第一层 Teacher-Forcing 与端到端 HR/NDCG 决定。

### 8.4 产物与结论

| 产物 | 路径 / SHA256 |
|---|---|
| 流式冻结入口 | `scripts/embedding/materialize_query_augmented.py` / `0b026f27b307737f7239e2cde288c5acdb386771cc538d4a4095b777785d21a5` |
| RQ-VAE 配置 | `configs/sid/rqvae_tiger_e4_bge_m3_1024x3.yaml` / `dd2fb659e984defbf9069caca032b6297930d0ceb18e91d4eab914d0c42c6e34` |
| 训练目录 | `outputs/sid/tiger/e4_bge_m3/TIGER-E4-BGE-M3-1024x3/` |
| epoch 20 SID | `evaluations/epoch_20/sid_codes.npy` / `0c1cb86b714272900e25fbb66ae677390d170d40d57aef7aede2412a6f2cc6ef` |
| epoch 20 metrics | `evaluations/epoch_20/metrics.json` / `f74bfdb82bbaa601590aa51da3ea394d5c2f18a11c66448a33cb42adbcba5f4c` |

结论：`E4+RQ-VAE-1024³` 是稳定正结果，证明 Query 增强向量的连续召回收益能够部分传递到离散 SID 的码本展开和碰撞下降；但它同时暴露“Query 可检索语义”和“静态类别层级语义”之间的目标张力。该 checkpoint 保留为 `2×2` 主对照，不直接替换后续候选；最后一格 E4+对称 RQ-KMeans 见 EXP-20260808-06。

## 9. EXP-20260808-06 E4 + RQ-KMeans `1024³` 与完整 `2×2`

### 9.1 目标与协议

本实验补齐 Embedding × Quantizer 的最后一格，判断 E4 的结构收益在直接 RQ-KMeans 下是否仍然存在，以及 Embedding 改进与 Quantizer 改进能否近似叠加。配置为 `configs/sid/rqkmeans_e4_bge_m3_1024x3.yaml`，仅把原始 BGE 输入替换为 EXP-20260808-02 冻结的 E4 全量向量；其余字段与原始 BGE 对称 RQ-KMeans 保持一致：

- sequential residual K-Means，三层码本均为 1,024，`max_beam_size=1`；
- 固定 Train-only 500,000 条样本，`iterations=20`、`nredo=3`、seed 42；
- 固定 1% Validation 和相同 POI 行序，样本/Validation 索引 SHA256 均未改变；
- Faiss GPU 单卡训练，全量向量与 SID 均分块读取或生成，未使用系统 `/tmp`；
- 任务退出码为 0，三层训练样本均使用 1,024/1,024 个 code，全量产物状态为 `_SUCCESS`。

固定 Validation 上，E4+RQ-KMeans 的唯一率为 99.0715%，碰撞 POI 比例为 1.7371%；原始 BGE+RQ-KMeans 分别为 99.0501% 和 1.7927%。E4 的 L1/L2/L3 Validation 使用数为 1021/923/849，高于原始 BGE 的 1022/893/793 中后两层；但重构 MSE/维从 $1.91242\times10^{-4}$ 增至 $2.12047\times10^{-4}$，重构余弦从 0.896490 降至 0.884415。该趋势与 RQ-VAE 一致：E4 更易展开和区分，但不是更易重构。

### 9.2 北京全量 `2×2` 结果

| Embedding × Quantizer | 唯一 SID | 碰撞 POI | 碰撞冗余 | P99 / 最大桶 | L1/L2/L3 使用数 | L1/L2/L3 归一化熵 |
|---|---:|---:|---:|---:|---:|---:|
| BGE × RQ-VAE `1024³` | 71.6766% | 40.7894% | 28.3234% | 7 / 306 | 561/1024/1024 | 0.8578/0.9693/0.9729 |
| E4 × RQ-VAE `1024³` | 74.3539% | 37.4194% | 25.6461% | 6 / 334 | 751/1024/1024 | 0.9061/0.9693/0.9714 |
| BGE × RQ-KMeans `1024³` | 76.0725% | 36.1501% | 23.9275% | 6 / 467 | 1024/1024/1024 | 0.9751/0.9378/0.9202 |
| **E4 × RQ-KMeans `1024³`** | **76.1573%** | **35.7477%** | **23.8427%** | **6 / 288** | **1024/1024/1024** | **0.9706/0.9445/0.9273** |

E4+RQ-KMeans 产生 1,779,931 个不同 SID，是四个对称配置中的全局最优。相对 BGE+RQ-KMeans，它增加 1,982 个不同 SID，减少 9,405 条碰撞 POI，并把最大桶从 467 降至 288；P99 仍为 6。也就是说，E4 在 RQ-KMeans 上的平均收益很小，但对极端碰撞尾部有实质改善。

### 9.3 主效应与交互

以下均为同一北京全量集合上的确定性差值；每格只有一个固定 seed，故这里只做描述性交互分析，不宣称统计显著性。

| 改动 | 唯一 SID 变化 | 碰撞 POI 变化 |
|---|---:|---:|
| BGE→E4，Quantizer=RQ-VAE | +2.6774pp | -3.3700pp |
| BGE→E4，Quantizer=RQ-KMeans | +0.0848pp | -0.4024pp |
| RQ-VAE→RQ-KMeans，Embedding=BGE | +4.3959pp | -4.6393pp |
| RQ-VAE→RQ-KMeans，Embedding=E4 | +1.8033pp | -1.6717pp |

以唯一率为例，加性尺度上的交互量为：

$$
\Delta_{\mathrm{interaction}}
=(76.1573-76.0725)-(74.3539-71.6766)
=-2.5926\ \mathrm{pp}
$$

负交互不是“E4 失效”，而是两种方法主要在修复同一个结构瓶颈：E4 先提高输入的 Query 意图区分度并帮助码本展开；直接 RQ-KMeans 本身已经把 S1 使用率提高到 100%，因此留给 E4 的全局唯一率空间很小。E4 在 RQ-KMeans 下仍显著降低最大桶，说明二者尚有一部分互补性，但互补性集中在长尾碰撞而非整体平均指标。

### 9.4 前缀类别语义

| Embedding × Quantizer | Depth-1 micro | Depth-2 micro | Depth-3 micro | Depth-1 macro | Depth-2 macro | Depth-3 macro |
|---|---:|---:|---:|---:|---:|---:|
| BGE × RQ-VAE `1024³` | 69.2751% | 77.2651% | 96.2830% | 61.8934% | 78.1823% | 98.2100% |
| E4 × RQ-VAE `1024³` | 64.1303% | 74.8755% | 95.9466% | 55.8665% | 77.8006% | 98.1072% |
| BGE × RQ-KMeans `1024³` | 67.0196% | 76.8735% | 96.0949% | 64.4349% | 81.6343% | 98.1632% |
| E4 × RQ-KMeans `1024³` | 61.7668% | 74.1673% | 95.4859% | 57.7363% | 79.8975% | 97.9337% |

E4 在两个 Quantizer 下都让 depth-1 micro-purity 下降约 5.2 个百分点，说明静态类别纯度下降是输入目标变化，而不是 RQ-VAE 特有的训练问题。RQ-KMeans 自身相对 RQ-VAE 也略降 micro-purity。当前不能仅据此否定 E4：`category_code` 不覆盖门牌、别名、分店与 Query 意图，必须用相同 SFT 的逐层 Teacher-Forcing 和端到端 HR/NDCG 判断这些新方向是否更容易由 Query 预测。

### 9.5 产物与结论

| 产物 | 路径 / SHA256 |
|---|---|
| RQ-KMeans 配置 | `configs/sid/rqkmeans_e4_bge_m3_1024x3.yaml` / `61141339cee6d97e7dba76fd873dea3612d30c31c0cbef15124e9ec37f1ce977` |
| 完整产物目录 | `outputs/sid/rqkmeans/e4_bge_m3/gpu_greedy_1024x3_s500k_i20_r3/` |
| 全量 SID | `sid_codes.npy` / `43fa1f2fca5e2a7bab7a04186fe7dfba0e46683664f08383112a872748273eda` |
| 全量 metrics | `metrics.json` / `14ae8ea2745a943846b062633668327dbf17809e447476509452a76662feb351` |

结论：`2×2` 已完整闭环。E4 和 RQ-KMeans 都是正向结构改进，但收益强重叠，不能在论文中写成两个近似正交、可直接相加的模块。E4+RQ-KMeans 是对称 `1024³` 的最优组合；其相对 BGE+RQ-KMeans 的核心卖点应表述为“保持平均唯一率基本稳定，同时改善碰撞长尾并保留 Query 增强召回”，最终是否优于原始 BGE+RQ-KMeans 要由同协议 SFT 决定。

## 10. EXP-20260808-07 E4 + RQ-KMeans `4096×1024×256`

### 10.1 目标、输入与执行协议

目标是验证原始 BGE 上选出的 30-bit 最优容量分配能否迁移到 E4，而不是默认跨输入复用结论。输入继续使用冻结 E4/α=0.30/β=0.85，Embedding fingerprint 为 `ba7904ff5b542bb3c09d66b8feca2b0fb6f7f43b1eedce49351dfd8a3658f49e`；POI 行序、固定 1% Validation、Train-only 50 万 K-Means 样本和 seed 均与前序实验一致。

本轮复用 `configs/sid/rqkmeans_e4_bge_m3_1024x3.yaml`，只通过已有 CLI 覆盖布局和输出目录，完整命令为：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python scripts/sid/build_rqkmeans.py \
  --config configs/sid/rqkmeans_e4_bge_m3_1024x3.yaml \
  --output-dir outputs/sid/rqkmeans/e4_bge_m3/gpu_greedy_4096x1024x256_s500k_i20_r3 \
  --codebook-sizes 4096,1024,256 \
  --no-progress
```

量化器仍为 Faiss GPU sequential residual K-Means，`iterations=20`、`nredo=3`、`max_beam_size=1`、GPU temporary memory 512 MiB。使用开发服务器单卡 RTX A6000 48GB 和 `/ofs/map_search/hudan/envs/poi-gr`；全量输入以 mmap/8,192 行分块处理，RAM 中只保留固定 50 万训练样本。任务所有日志、临时目录和产物均位于项目 `outputs/`，未使用系统 `/tmp`。宿主任务从 22:14:46 到 22:22:25，退出码为 0；核心构建记录从 22:16:35 到 22:22:14，共 338.63 秒。

三层样本训练均无死码：

| 层 | 码本 | 样本使用数 | 样本残差 MSE/L2 | 训练时间 |
|---|---:|---:|---:|---:|
| S1 | 4,096 | 4,096 | 0.286139 | 60.25 秒 |
| S2 | 1,024 | 1,024 | 0.226778 | 44.51 秒 |
| S3 | 256 | 256 | 0.201869 | 42.22 秒 |

固定 23,372 条 Validation 上 E4/BGE 的唯一率分别为 99.5636%/99.5764%，仅相差 3 个不同 SID。该子集在高容量布局下只有约 100 个碰撞冗余，方差过大，不用于替代后续北京全量结论。

### 10.2 全量结构结果

| 输入 × 布局 | 唯一 SID | 碰撞 POI | 碰撞冗余 | P99 / 最大桶 | L1/L2/L3 使用数 | L1/L2/L3 熵 |
|---|---:|---:|---:|---:|---:|---:|
| BGE × `1024³` | 76.0725% | 36.1501% | 23.9275% | 6 / 467 | 1024/1024/1024 | 0.9751/0.9378/0.9202 |
| E4 × `1024³` | 76.1573% | 35.7477% | 23.8427% | 6 / 288 | 1024/1024/1024 | 0.9706/0.9445/0.9273 |
| BGE × `4096×1024×256` | 81.8938% | 29.1512% | 18.1062% | 4 / **96** | 4096/1024/256 | 0.9747/0.9298/0.9080 |
| **E4 × `4096×1024×256`** | **82.7331%** | **27.9655%** | **17.2669%** | **4 / 125** | **4096/1024/256** | **0.9712/0.9373/0.9128** |

同布局 E4 相对 BGE 增加 19,614 个不同 SID，唯一率提高 0.8392 个百分点；碰撞 POI 减少 27,711 条、比例降低 1.1857 个百分点。全量重构 MSE/维由 BGE 的 $1.82466\times10^{-4}$ 增至 E4 的 $2.01403\times10^{-4}$，余弦由 0.901359 降至 0.890361，再次确认“更少碰撞”不等于“更低重构误差”。

在 E4 内部，把容量从 `1024³` 前移为 `4096×1024×256`，增加 153,688 个不同 SID，唯一率提高 6.5758 个百分点；碰撞 POI 减少 181,883 条、比例降低 7.7822 个百分点，最大桶从 288 降至 125。该布局迁移不是微小波动，而是稳定的大幅结构收益。

### 10.3 容量分配与 Embedding 的交互

原始 BGE 的容量前移收益为：

$$
81.8938\%-76.0725\%=5.8213\ \mathrm{pp}
$$

E4 的容量前移收益为：

$$
82.7331\%-76.1573\%=6.5758\ \mathrm{pp}
$$

因此在唯一率的加性尺度上，Embedding 与容量前移的描述性交互为：

$$
\Delta_{\mathrm{interaction}}=6.5758-5.8213=+0.7544\ \mathrm{pp}
$$

这与 EXP-20260808-06 的“E4 与量化器算法收益重叠”不矛盾：直接 RQ-KMeans 已解决对称布局中的码本失活，但 E4 引入的 Query 意图区分方向仍需要更大的 S1 容量承载。也就是说，E4 与“换成 RQ-KMeans”不是正交模块，E4 与“把更多 bit 分配给 S1”却存在正互补。

### 10.4 前缀类别语义与收益边界

| 输入 × 布局 | Depth-1 micro | Depth-2 micro | Depth-3 micro | Depth-1 macro | Depth-2 macro | Depth-3 macro |
|---|---:|---:|---:|---:|---:|---:|
| BGE × `1024³` | 67.0196% | 76.8735% | 96.0949% | 64.4349% | 81.6343% | 98.1632% |
| E4 × `1024³` | 61.7668% | 74.1673% | 95.4859% | 57.7363% | 79.8975% | 97.9337% |
| BGE × `4096×1024×256` | 68.8522% | 83.3590% | 97.1629% | 65.4931% | 88.6955% | 98.6111% |
| E4 × `4096×1024×256` | 63.6370% | 81.3855% | 96.8838% | 58.9060% | 87.7277% | 98.5114% |

容量前移在 E4 上同时把 depth-1/2/3 micro-purity 提高 1.8702/7.2183/1.3979 个百分点，说明它不只是靠随机展开减少碰撞，也恢复了更清晰的中层类别结构。但同布局下 E4 仍比 BGE 低 5.2151/1.9734/0.2791 个百分点，静态类别目标与 Query 增强目标的张力依旧存在。

E4 的最大桶为 125，差于同布局 BGE 的 96；因此不能写成 E4 全面改善碰撞分布。更准确的结论是：E4 改善平均唯一率和碰撞覆盖，`4096` S1 又大幅压缩其长尾，但最极端热点桶仍需单独分析或在下游使用碰撞后缀。

### 10.5 产物与结论

| 产物 | 路径 / SHA256 |
|---|---|
| 完整目录 | `outputs/sid/rqkmeans/e4_bge_m3/gpu_greedy_4096x1024x256_s500k_i20_r3/` |
| Resolved config signature | `53e070aec4182031b23c57313ea8202c2015ecd04e806c87a8fcd9659e825c4c` |
| 全量 SID | `sid_codes.npy` / `8defb5671861753de80b3670272d0d38265ebe5c8ef2d4268438a4095cb3169f` |
| 全量 metrics | `metrics.json` / `2665ad4ff16d46dded658c0de0bc61fc361498a8d1c27bebff938e1ffb575036` |

结论：E4 的非对称迁移验证通过，E4+`4096×1024×256` 冻结为当前“纯 SID 结构最优”，但尚不能直接冻结为下游生成最优。它相对 `1024³` 增加 3,072 个 S1 token，会提高第一步分类和词表成本；下一阶段必须用 SFT Teacher-Forcing、完整 identifier HR/NDCG、合法率和时延验证结构收益是否能转化为生成收益。

## 11. EXP-20260808-08：E4 `2048×1024×512` 与 SFT 候选冻结

### 11.1 目标与执行协议

本实验补齐 E4 在固定 30-bit 总路径容量下的工程折中布局，回答是否值得用较小的 S1 词表替代纯结构最优的 `4096×1024×256`。输入固定为 E4/α=0.30/β=0.85 全量 POI 向量，训练协议与另外两档完全一致：三层顺序残差 K-Means、每层 500,000 条固定样本、20 次迭代、3 次重启、GPU `float32`，且不加入类别监督、balance penalty 或唯一化后缀。

正式任务在开发机 RTX A6000 48 GB 上完成，核心构建耗时 324.10 秒，退出码为 0，未发生 OOM。命令为：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python scripts/sid/build_rqkmeans.py \
  --config configs/sid/rqkmeans_e4_bge_m3_1024x3.yaml \
  --output-dir outputs/sid/rqkmeans/e4_bge_m3/gpu_greedy_2048x1024x512_s500k_i20_r3 \
  --codebook-sizes 2048,1024,512 \
  --no-progress
```

### 11.2 三种 E4 30-bit 布局结果

| E4 布局 | 唯一 SID | 碰撞 POI | Excess collision | P99 桶 | 最大桶 | D1/D2/D3 micro-purity | 基础 SID Token | 最大碰撞后缀 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1024×1024×1024` | 76.1573% | 35.7477% | 23.8427% | 6 | 288 | 61.7668% / 74.1673% / 95.4859% | 3,072 | 288 |
| `2048×1024×512` | 79.8731% | 31.4021% | 20.1269% | 5 | 143 | 63.1348% / 78.1489% / 96.4413% | 3,584 | 143 |
| `4096×1024×256` | 82.7331% | 27.9655% | 17.2669% | 4 | 125 | 63.6370% / 81.3855% / 96.8838% | 5,376 | 125 |

`2048×1024×512` 相对 `1024³`：

- 唯一率提高 3.7158 个百分点，碰撞 POI 降低 4.3456 个百分点；
- 唯一 SID 数增加 86,846，碰撞 POI 减少 101,564；
- 最大桶从 288 降至 143，几乎减半；
- 只增加 512 个基础 SID token。

`4096×1024×256` 相对 `2048×1024×512` 仍继续改善唯一率 2.8599 个百分点、减少碰撞 POI 3.4366 个百分点，但需要再增加 1,792 个基础 token。按“每新增 1,000 个基础 token 带来的唯一率增益”计算，`1024→2048` 为 7.2575 pp，`2048→4096` 只有 1.5960 pp，后者的边际效率约为前者的 22%。因此 `2048` 是明确的 Pareto 拐点，而 `4096` 是容量上界，不应把两者混成同一种“最优”。

同布局的原始 BGE `2048×1024×512` 唯一率为 78.9451%、碰撞 POI 为 32.7653%、最大桶为 163。E4 在相同量化布局上把唯一率提高 0.9280 个百分点、碰撞 POI 降低 1.3632 个百分点、最大桶降至 143，但 D1/D2/D3 micro-purity 分别低 5.7648/2.1104/0.3174 个百分点。这组结果适合隔离 Embedding 创新的下游贡献，不能只用静态类别纯度提前否决。

### 11.3 SFT 标签难度分析

使用 7,586,410 条 Train query-target 订单对各 S1 前缀加权，三种 E4 布局的目标分布如下：

| 布局 | $H(S_1)$ | 有效 S1 分支数 $2^{H(S_1)}$ | $H(S_2\mid S_1)$ | $H(S_3\mid S_{1:2})$ | $H(S_{1:3})$ | Train 使用的 S1 |
|---|---:|---:|---:|---:|---:|---:|
| `1024³` | 8.5439 bit | 373.2 | 5.5170 bit | 1.1841 bit | 15.2450 bit | 1,023 / 1,024 |
| `2048×1024×512` | 9.4141 bit | 682.2 | 4.9869 bit | 0.9485 bit | 15.3495 bit | 2,044 / 2,048 |
| `4096×1024×256` | 10.2215 bit | 1,193.9 | 4.4830 bit | 0.7387 bit | 15.4432 bit | 4,074 / 4,096 |

三个布局的完整路径熵只从 15.2450 增至 15.4432 bit，主要变化是把预测难度从后续 token 前移到 S1。这一点对生成模型很关键：既有 TIGER/GNPR Teacher-Forcing 已证明第一 token 是主要瓶颈，纯 SID 唯一率无法代替 SFT 验证。

固定 10,000 条 Validation 中没有完全未见的 S1，但 Train 支持不超过 1,000 单的目标数量分别为 225、498、1,103；`4096` 相比 `2048` 又把低支持 S1 目标翻倍，同时有效 S1 分支数增加约 75%。所以 `4096` 的额外结构收益很可能伴随更难的第一步分类，必须作为容量上界实测，而不是直接指定为最终模型。

### 11.4 确定的新增 SFT 候选

| 优先级 | Embedding | Quantizer / 布局 | 实验角色 | 决策 |
|---|---|---|---|---|
| P0 | E4 | RQ-KMeans `2048×1024×512` | 提出方法主模型 | 首先训练；当前结构—生成复杂度 Pareto 候选 |
| P0 | E4 | RQ-KMeans `4096×1024×256` | 容量上界 | 必须训练；检验额外 2.8599 pp 唯一率能否抵消 S1 难度和词表成本 |
| P1 | 原始 BGE | RQ-KMeans `2048×1024×512` | 同布局 Embedding 消融 | 在两组 P0 后训练；隔离 E4 对生成检索的真实增益 |
| 复用 | 原始 BGE | TIGER RQ-VAE `1024³` | 已有强基线 | 直接复用 epoch 3，不重复训练 |

E4 `1024³` 不进入新增正式 SFT：它相对 E4-2048 少 512 个基础 token，却损失 3.7158 个百分点唯一率、增加 101,564 个碰撞 POI，且最大桶翻倍；在当前结果上已被 E4-2048 支配。若论文审稿阶段需要完整 `2×3` 下游矩阵，可把 BGE `1024³` RQ-KMeans 作为 P2 补充，但它不是当前最小充分实验集合。

三组新增 SFT 必须共用相同基础模型、Train/Validation 划分、训练轮数、碰撞唯一化规则、Tokenizer 构建规则、Beam 和 Trie 解码协议。第一轮统一沿用 TIGER 的条件 collision token，只比较 SID/Embedding，不混入 GID 或局部图后缀，避免把两项创新混为一个变量。至少报告：S1 Top-1/Top-5、逐层条件 Teacher-Forcing、完整 identifier HR@1/3/5/10、NDCG@10、合法率、词表规模、训练吞吐和在线解码时延。

### 11.5 产物

| 产物 | 路径 / SHA256 |
|---|---|
| 完整目录 | `outputs/sid/rqkmeans/e4_bge_m3/gpu_greedy_2048x1024x512_s500k_i20_r3/` |
| Resolved config signature | `403d7bfff36c9ea8bab04aea9f2e1b7db94c9b771f0be5b89eb58eede4b13288` |
| 全量 SID | `sid_codes.npy` / `0618d75054293bc982f2fd51b08fe58b3e1a1fd5f335991888494ccf8bb3e34f` |
| 全量 metrics | `metrics.json` / `14c9bbbfbbb6808f6c154c3c698e74a0f48d2d83d5ff6ba5a6502c985c08fcb3` |

## 12. 下一步

1. 按同一 TIGER collision-token 协议为三组新增候选构建全局唯一 identifier、Tokenizer 与训练数据；先做映射确定性和目标无截断验证。
2. 先训练 E4-2048 主模型和 E4-4096 容量上界，用逐层 Teacher-Forcing 尽早判断 S1 难度；两者完成后再训练 BGE-2048 同布局消融。
3. 最终候选由完整 identifier HR/NDCG、合法率、词表开销和时延共同确定。只在胜出布局上追加碰撞桶局部图/GID/Dedup 创新，不在本轮量化器主实验中混入第二变量。
