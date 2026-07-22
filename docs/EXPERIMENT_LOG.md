# 实验进展

本文件只记录需要长期留存的正式实验。临时 smoke、性能比较和调试产物不单独保留。

## EXP-20260717-01：北京 POI 全量向量构建

- 状态：已完成
- 日期：2026-07-17
- 编码开始时间：2026-07-17 21:25:26（Asia/Shanghai）
- 完成时间：2026-07-17 22:51:42（Asia/Shanghai）
- 阶段：POI 文本与 Embedding
- 数据：`beijing_poi_clean_20260715_json`
- 输入行数：2,337,178
- 输入指纹：`d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071`
- 模型：Qwen3-Embedding-0.6B
- 输入字段：`text`
- ID 字段：`poi_id`
- 输出目录：`outputs/embeddings/beijing_poi_qwen3_embedding_0.6b/`

### 正式配置

| 参数 | 取值 |
|---|---|
| Device | CUDA |
| Batch size | 64 |
| Encode buffer size | 8192 |
| Max sequence length | 512 |
| Model dtype | BF16 |
| Attention | SDPA |
| Padding side | Left |
| Embedding dimension | 1024 |
| Normalize embeddings | Yes |
| Output dtype | float16 |
| Resume | Yes |

运行命令：

```bash
python scripts/build_poi_embeddings.py \
  --config configs/embedding_qwen3_0.6b.yaml
```

### 正式结果

| 指标 | 结果 |
|---|---:|
| 输入分片 | 16 |
| 输入/ID/向量行数 | 2,337,178 |
| 唯一 POI ID | 2,337,178 |
| 向量 shape | `[2337178, 1024]` |
| 向量 dtype | float16 |
| 数据扫描耗时 | 49.05 秒 |
| 模型加载耗时 | 7.93 秒 |
| 编码耗时 | 5,176.54 秒 |
| 总流水线耗时 | 约 1 小时 27 分 14 秒 |
| 编码吞吐 | 451.49 条/秒 |
| 峰值 allocated / reserved 显存 | 5.47 / 6.27 GiB |
| `embeddings.npy` 大小 | 4,786,540,672 字节 |
| 非有限向量值 | 0 |
| 抽样 L2 范数 | 最小 0.9980，均值 1.0005，最大 1.0038 |

`manifest.json` 和 `progress.json` 均为 `completed`。NPY 文件大小与头部声明一致，可通过 mmap 读取；ID 行数、唯一性和向量行数完全一致。全量 POI 文本向量构建完成，可以进入下一阶段的 RQ-VAE SID 设计与实现。

## EXP-20260718-01：北京 POI 4B 全量向量构建

- 状态：已完成
- 日期：2026-07-18
- 编码开始时间：2026-07-18 00:26:39（Asia/Shanghai）
- 完成时间：2026-07-18 06:05:48（Asia/Shanghai）
- 阶段：POI 文本与 Embedding
- 数据：`beijing_poi_clean_20260715_json`
- 输入行数：2,337,178
- 输入指纹：`d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071`
- 模型：Qwen3-Embedding-4B
- 输入字段：`text`
- ID 字段：`poi_id`
- 输出目录：`outputs/embeddings/beijing_poi_qwen3_embedding_4b/`

### 正式配置

| 参数 | 取值 |
|---|---|
| Device | CUDA |
| Batch size | 64 |
| Encode buffer size | 8192 |
| Max sequence length | 512 |
| Model dtype | BF16 |
| Attention | PyTorch SDPA |
| Padding side | Left |
| Embedding dimension | 2560 |
| Normalize embeddings | Yes |
| Output dtype | float16 |
| Resume | Yes |

运行命令：

```bash
python scripts/build_poi_embeddings.py \
  --config configs/embedding_qwen3_4b.yaml
```

### 正式结果

| 指标 | 结果 |
|---|---:|
| 输入分片 | 16 |
| 输入/ID/向量行数 | 2,337,178 |
| 唯一 POI ID | 2,337,178 |
| 向量 shape | `[2337178, 2560]` |
| 向量 dtype | float16 |
| 数据扫描耗时 | 117.87 秒 |
| 模型加载耗时 | 515.71 秒 |
| 编码耗时 | 20,348.28 秒 |
| 总流水线耗时 | 约 5 小时 49 分 42 秒 |
| 编码吞吐 | 114.86 条/秒 |
| 峰值 allocated / reserved 显存 | 14.49 / 16.26 GiB |
| `embeddings.npy` 大小 | 11,966,351,488 字节 |
| 抽样非有限向量值 | 0 |
| 抽样 L2 范数 | 最小 0.9966，均值 1.0013，最大 1.0039 |

`manifest.json` 和 `progress.json` 均为 `completed`。NPY 文件头声明与实际文件大小一致，可通过 mmap 读取；ID 行数、唯一性和向量行数完全一致。对首尾及均匀分布的 4,606 条向量抽样检查未发现非有限值，归一化范数符合 float16 精度预期。4B 全量 POI 文本向量构建完成。

## EXP-20260719-01：北京 POI MiniOneRec 风格 RQ-VAE SID 构建

- 状态：已完成，待用户验收
- 日期：2026-07-19
- 开始/结束时间：2026-07-19 23:23:38—23:47:24（Asia/Shanghai）
- 阶段：RQ-VAE Semantic ID
- 目标：使用 Qwen3-Embedding-0.6B 的全量北京 POI 向量训练三层残差量化模型，导出所有 POI 的 RQ SID，并评估重建、码本使用和碰撞。
- 假设：参考 MiniOneRec 的 MLP RQ-VAE 与末层 Sinkhorn 重分配，可以兼顾语义重建和 SID 唯一性。

### 数据与代码

- 数据版本：`beijing_poi_clean_20260715_json`
- Embedding：Qwen3-Embedding-0.6B，shape `[2337178, 1024]`，float16
- Embedding 指纹：`d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071`
- 训练范围：每个 epoch 使用全部 2,337,178 条向量；本实验没有单独划分验证集
- Git：以 `48d322d` 为基线，工作树包含尚未提交的 MiniOneRec 风格 RQ-VAE 实现与配置
- 任务签名：`ec6b68764e21e7a307a01cc73d4e3b0d9a70f21a31cae8d49edd7b569ed02ab2`
- 代码：`src/poi_gr/rqvae.py`
- 入口：`scripts/train_rqvae.py`
- 配置：`configs/rqvae_qwen3_embedding_0.6b.yaml`
- 环境：Python 3.12.11、PyTorch 2.9.1、scikit-learn 1.7.1、NVIDIA RTX A6000、BF16

### 方案与正式配置

模型先用 MLP 将 1024 维向量编码为 32 维 latent，再依次量化三层残差，最后通过对称 MLP 解码器重建原始向量。Codebook 是可学习参数，通过反向传播更新，不使用 EMA 或 dead-code 重置。

| 参数 | 取值 |
|---|---|
| Input / hidden / latent dims | 1024 / `[768, 512, 256, 128, 64]` / 32 |
| RQ levels / codebook size | 3 / 每层 256 |
| Codebook 初始化 | 第一批 4,096 条向量分别初始化三层 K-Means，约占全量 0.175% |
| K-Means | `max_iter=100`、`n_init=1`、每层使用 `20260719 + level` 作为种子 |
| 训练分配 | 最近邻；训练阶段各层 Sinkhorn epsilon 均为 0 |
| Loss | `reconstruction_mse + quantization_loss` |
| Quantization loss | `codebook_loss + 0.25 × commitment_loss`，三层取均值 |
| Batch / eval batch size | 4096 / 8192 |
| Epochs | 20 |
| Optimizer | AdamW，learning rate `1e-3`，weight decay `0` |
| Learning-rate schedule | 1 epoch 线性 warmup，之后保持 `1e-3` |
| Gradient clip norm | 1.0 |
| Block shuffle / workers | 65,536 / 4 |
| Checkpoint metric | 每轮全量最近邻 SID 的重复 assignment 比例，越低越好 |
| 导出碰撞处理 | 对碰撞组的末层残差做 Sinkhorn 重分配，epsilon `0.003`、50 次迭代、最多 20 轮 |
| Seed / resume | `20260719` / Yes |

运行命令：

```bash
python -u scripts/train_rqvae.py \
  --config configs/rqvae_qwen3_embedding_0.6b.yaml
```

### 产物与结果

- 产物目录：`outputs/rqvae/beijing_poi_qwen3_embedding_0.6b/`
- 运行状态：`completed`
- 总耗时：1,425.52 秒（约 23 分 46 秒）
- CUDA 峰值显存：allocated 0.301 GiB，reserved 0.436 GiB
- 最低训练总损失：第 2 轮，`0.000483961`
- 最低原始 SID 碰撞率：第 20 轮，`0.490216834`
- 最终选择：`checkpoint_best_collision.pt`，第 20 轮

第 20 轮 checkpoint 的全量重建结果：

| 指标 | 结果 |
|---|---:|
| SID shape / dtype | `[2337178, 3]` / `uint16` |
| 重建 MSE | 0.000343832 |
| 重建 L2 | 0.352084 |
| 重建余弦相似度 | 0.803871 |
| Quantization loss | 0.000418286 |

原始最近邻 SID 与最终 Sinkhorn 重分配结果：

| 指标 | 原始最近邻 SID | 最终导出 SID |
|---|---:|---:|
| 唯一 SID 数 / 比例 | 1,191,454 / 50.98% | 1,647,044 / 70.47% |
| 重复 assignment 数 / 比例 | 1,145,724 / 49.02% | 690,134 / 29.53% |
| 碰撞 POI 数 / 比例 | 1,575,192 / 67.40% | 933,668 / 39.95% |
| 碰撞组数量 | 429,468 | 243,534 |
| 碰撞组 P95 / 最大大小 | 9 / 491 | 10 / 140 |

码本使用情况：

| 层级 | 原始使用 code 数 | 原始 perplexity | 最终使用 code 数 | 最终 perplexity |
|---|---:|---:|---:|---:|
| Level 1 | 130 / 256 | 102.61 | 130 / 256 | 102.61 |
| Level 2 | 256 / 256 | 243.98 | 256 / 256 | 243.98 |
| Level 3 | 256 / 256 | 250.66 | 256 / 256 | 162.84 |

从第 2 轮到第 20 轮，训练总损失由 `0.000483961` 上升到 `0.000761983`，但重建 MSE 由 `0.000426925` 降至 `0.000344513`，余弦相似度由 `0.749090` 升至 `0.803429`，原始 SID 唯一率由 `45.57%` 升至 `50.98%`。总损失上升主要来自 quantization/commitment 项增长，不能单独用它判断 SID 质量。因此本实验按全量原始 SID 碰撞率选择第 20 轮，而不是按总损失选择第 2 轮。

`manifest.json` 和 `metrics.json` 状态与上述配置、指标一致；`training_history.jsonl` 共 20 轮；`sids.npy` 可通过 mmap 读取，行数与 `poi_ids.jsonl` 的 2,337,178 行一致。重建指标基于原始最近邻量化结果，Sinkhorn 只在导出时重分配最后一层 token，不应把重分配后的 SID 唯一性解释为重建质量提升。

### 结论与下一步

本实验已跑通全量 MiniOneRec 风格 RQ-VAE 训练、checkpoint 恢复、逐轮碰撞评估、SID 导出和末层碰撞处理。Sinkhorn 将唯一 SID 比例从 50.98% 提升到 70.47%，最大碰撞组从 491 降到 140，但第一层仅使用 130 个 code，且最终仍有 39.95% 的 POI 位于碰撞组中。当前结果适合作为 RQ-VAE 基线，尚不能作为最终唯一 POI 编码。

下一步由用户决定：结合 Geohash GID 缓解空间碰撞，或先增加 K-Means 初始化样本、训练轮数及 RQ-KMeans 等对照。用户确认前不进入 GID、SFT 或后续训练阶段。

## EXP-20260722-01：0.6B 无 Instruction 向量召回基线

- 状态：已完成
- 日期：2026-07-22
- 开始/结束时间：2026-07-22 11:31:56—11:36:47（Asia/Shanghai）
- 阶段：POI Embedding 相关性评测 E1
- 目标：验证 Qwen3-Embedding-0.6B 在不添加 Query Instruction 时，对北京全量 POI 候选库的精确向量召回能力。
- 假设：复用 POI 端编码器、last-token pooling 和归一化配置，原始 Query 与 POI 向量的内积可作为后续模型及 Instruction 对照实验的基线。

### 数据与代码

- 评测数据：固定 10,000 条北京检索订单，日期范围 2026-07-01—2026-07-14
- 评测数据 SHA256：`52b2bc62349dffaaf31444e03d012e75a22b5bedb4fea6327b8c7fe4f90282f2`
- 唯一订单：10,000；唯一目标 POI：7,601
- POI 候选库：2,337,178 条，输入指纹 `d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071`
- POI Embedding：Qwen3-Embedding-0.6B，shape `[2337178, 1024]`，float16
- Git：`48d322d`，工作树包含尚未提交的向量评测实现、配置及其他既有改动
- 代码：`scripts/evaluate_embedding_retrieval.py`
- 配置：`configs/embedding_retrieval_eval.yaml`
- 环境：Python 3.10.20、PyTorch 2.9.1+cu128、Faiss GPU 1.8.0、NVIDIA RTX A6000

Gate 0 全部通过：评测行数和唯一订单正确，Query/目标 POI 均非空；POI 向量和 ID 均为 2,337,178 行，ID 全部唯一且与排序后的原始分片逐行一致；10,000 条目标均存在于候选库；全量向量无 NaN/Inf，2,048 条抽样 L2 范数为 0.998047/1.000506/1.003797（最小/均值/最大）。

### 正式配置

| 参数 | 取值 |
|---|---|
| Query 输入 | 原始 `query`，无前缀或 Instruction |
| Query batch / buffer | 256 / 2,048 |
| Query max sequence length | 128 |
| Model dtype / output dtype | BF16 / float16 |
| Padding / pooling | Left / last-token |
| Normalize embeddings | Yes |
| Faiss index | GPU `IndexFlatIP`，精确检索 |
| Faiss storage / input dtype | float32 / float32 |
| Add / query batch | 32,768 / 64 |
| Top-K | 20 |

运行命令：

```bash
python -u scripts/evaluate_embedding_retrieval.py \
  --config configs/embedding_retrieval_eval.yaml \
  --model qwen3_0.6b \
  --instruction none
```

### 正式结果

| 指标 | 结果 |
|---|---:|
| Hit@1 | 12.03% |
| Hit@3 | 19.09% |
| Hit@5 | 23.13% |
| Hit@10 | 28.25% |
| Hit@20 | 34.51% |
| MRR@10 | 16.6907% |
| Top-20 命中 / 未命中 | 3,451 / 6,549 |

Query 长度分桶：

| 字符数 | 样本数 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| 1—2 | 2,927 | 0.92% | 3.38% | 5.06% | 1.95% |
| 3—5 | 4,552 | 8.88% | 21.75% | 28.12% | 14.21% |
| 6—10 | 1,796 | 27.56% | 46.94% | 54.62% | 35.88% |
| >10 | 725 | 38.21% | 52.55% | 57.38% | 44.26% |

### 性能与产物

| 项目 | 结果 |
|---|---:|
| Gate 0 | 105.80 秒 |
| 模型加载（内部计时 / 实际 wall time） | 43.00 / 115.10 秒 |
| Query 编码 | 9.20 秒，1,087.43 条/秒 |
| GPU 索引构建 | 51.62 秒 |
| 10,000 条检索 | 7.22 秒 |
| 总耗时 | 291.59 秒 |
| Query 编码峰值 allocated / reserved | 3.48 / 4.32 GiB |
| Faiss 观测显存增量峰值 | 8.92 GiB |

- 产物目录：`outputs/evaluation/embedding_retrieval/qwen3_0.6b_no_instruction/`
- Query 向量：`outputs/evaluation/embedding_retrieval/query_embeddings/qwen3_0.6b_no_instruction.npy`
- Query shape/dtype：`[10000, 1024]` / float16
- Query 抽样前的全量范数范围：0.998047—1.003810，均值 1.000798
- 结果：`metrics.json`、`retrieval_results.npz`、`run_manifest.json`、`run.log`

独立核验通过：Query 行映射与评测 JSONL 完全一致；Top-K shape 为 `[10000, 20]`；目标排名只包含 `-1` 或 1—20；整体指标重算一致；Query 向量、映射和召回结果 SHA256 均与 manifest 一致。

### 结论与下一步

E1 已形成可复现的无 Instruction 基线。召回效果随 Query 长度明显上升，1—2 字 Query 的 Hit@10 仅 5.06%，而 6—10 字和 10 字以上分别达到 54.62% 和 57.38%；这说明原始短 Query 的语义信息不足是当前基线的主要弱项，但仅凭本实验不能区分歧义、热度、地理约束缺失或订单标签噪声的贡献。

下一步由用户核验本基线后，再选择运行 4B 对照或 `poi_en` Query Instruction 对照。本实验不评估 RQ-VAE，也不进入后续生成式检索阶段。

## EXP-20260722-02：4B 无 Instruction 向量召回对照

- 状态：已完成
- 日期：2026-07-22
- 正式运行时间：2026-07-22 12:55:36—13:02:36（Asia/Shanghai）
- 阶段：POI Embedding 相关性评测 E2
- 目标：在评测数据、输入形式和精确检索口径与 E1 完全一致时，评估 Qwen3-Embedding-4B 的无 Instruction 召回能力。
- 假设：更大参数量的 Embedding 模型可能提高原始 Query 与 POI 文本的语义匹配能力；实验只报告对照结果，不自动选择模型。

### 数据与代码

- 评测数据：与 E1 相同的 10,000 条北京检索订单，SHA256 为 `52b2bc62349dffaaf31444e03d012e75a22b5bedb4fea6327b8c7fe4f90282f2`
- E1 对齐：10,000 个 `order_id`、目标 `poi_id` 及行顺序逐行一致，行映射 SHA256 为 `86a37f3c4c0f01f6000e7eb39a5c57f9bfef2d595124e3a33263fc4eb5923610`
- POI 候选库：2,337,178 条，输入指纹 `d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071`
- POI Embedding：Qwen3-Embedding-4B，shape `[2337178, 2560]`，float16
- Git：`48d322d`，工作树包含尚未提交的向量评测实现、配置及其他既有改动
- 代码：`scripts/evaluate_embedding_retrieval.py`
- 配置：`configs/embedding_retrieval_eval.yaml`
- 环境：Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、Faiss GPU 1.8.0、NVIDIA RTX A6000

Gate 0 全部通过：评测集严格为 10,000 行且唯一订单为 10,000，Query/目标 POI 均非空；POI 向量 shape 为 `[2337178, 2560]`，ID 行数和唯一数均为 2,337,178，向量与排序后的原始 POI 分片逐行一致；10,000 条目标全部存在于候选库；全量向量无 NaN/Inf，2,048 条抽样 L2 范数为 0.996400/1.001324/1.003862（最小/均值/最大）。

### 正式配置

| 参数 | 取值 |
|---|---|
| Query 输入 | 原始 `query`，无前缀或 Instruction |
| Query batch / buffer | 64 / 1,024 |
| Query max sequence length | 128 |
| Model dtype / output dtype | BF16 / float16 |
| Padding / pooling | Left / last-token |
| Normalize embeddings | Yes |
| Query shape | `[10000, 2560]` |
| Faiss index | GPU `GpuIndexFlatIP`，精确检索 |
| Faiss storage / input dtype | float32 / float32 |
| POI add 分批 | 1,000,000 + 1,337,178 |
| Query search batch / Top-K | 64 / 20 |

运行命令：

```bash
python -u scripts/evaluate_embedding_retrieval.py \
  --config configs/embedding_retrieval_eval.yaml \
  --model qwen3_4b \
  --instruction none
```

### 正式结果

| 指标 | E1：0.6B | E2：4B | E2 - E1 |
|---|---:|---:|---:|
| Hit@1 | 12.03% | 9.72% | -2.31 pp |
| Hit@3 | 19.09% | 15.07% | -4.02 pp |
| Hit@5 | 23.13% | 17.63% | -5.50 pp |
| Hit@10 | 28.25% | 21.18% | -7.07 pp |
| Hit@20 | 34.51% | 25.12% | -9.39 pp |
| MRR@10 | 16.6907% | 13.0759% | -3.6148 pp |

Query 长度分桶对照：

| 字符数 | 样本数 | Hit@1（E1→E2） | Hit@5（E1→E2） | Hit@10（E1→E2） | MRR@10（E1→E2） |
|---|---:|---:|---:|---:|---:|
| 1—2 | 2,927 | 0.92%→0.51% | 3.38%→1.30% | 5.06%→2.43% | 1.95%→0.89% |
| 3—5 | 4,552 | 8.88%→6.30% | 21.75%→14.17% | 28.12%→17.99% | 14.21%→9.54% |
| 6—10 | 1,796 | 27.56%→23.16% | 46.94%→41.37% | 54.62%→48.22% | 35.88%→30.99% |
| >10 | 725 | 38.21%→35.03% | 52.55%→46.48% | 57.38%→49.93% | 44.26%→40.07% |

### 性能、显存与产物

| 项目 | E1：0.6B | E2：4B |
|---|---:|---:|
| Gate 0 | 105.80 秒 | 171.06 秒 |
| 模型加载（内部 / wall time） | 43.00 / 115.10 秒 | 93.59 / 154.57 秒 |
| Query 编码 | 9.20 秒 | 18.04 秒 |
| GPU 索引构建 | 51.62 秒 | 55.18 秒 |
| 10,000 条检索 | 7.22 秒 | 16.44 秒 |
| 总耗时 | 291.59 秒 | 420.16 秒 |
| Query 峰值 allocated / reserved | 3.48 / 4.32 GiB | 8.50 / 9.08 GiB |
| Faiss 观测显存增量峰值 | 8.92 GiB | 23.81 GiB |

- 产物目录：`outputs/evaluation/embedding_retrieval/qwen3_4b_no_instruction/`
- Query 向量：`outputs/evaluation/embedding_retrieval/query_embeddings/qwen3_4b_no_instruction.npy`
- 结果：`metrics.json`、`retrieval_results.npz`、`run_manifest.json`、`run.log`
- Query 编码后模型和 tokenizer 已释放；allocated/reserved 降至 8.13/20.00 MiB，进程结束后显存为 0 MiB。

独立核验通过：Query 行映射与 E1、评测 JSONL 完全一致；Query shape/dtype、范数和 SHA256 正确；Top-K shape 为 `[10000, 20]`，分数均为有限 float32；重新流式读取 2,337,178 个 POI ID 并构造目标行号后，10,000 条目标排名与保存结果逐行一致，整体指标重算一致。

### 执行异常与修正

首次按 32,768 条直接向 GPU 分批 add，在 98% 时因 Faiss Flat 数据末次扩容需要同时保留旧缓冲和申请新缓冲而 OOM。第二次改用 CPU Flat 暂存时，在 90% 的主存扩容阶段被系统以退出码 137 终止。两次均未产生不完整召回结果，最终日志覆盖为成功运行日志。

最终采用两批 GPU add：先加入 1,000,000 条，再加入剩余 1,337,178 条；建索引前按“最终索引 + 首批旧缓冲 + 4 GiB 余量”执行 35.83 GiB 显存门禁。该修正不改变 float32 存储、`GpuIndexFlatIP`、精确内积或 Top-20 口径。

### 结论与下一步

在完全相同的无 Instruction 评测口径下，4B 在整体指标和四个 Query 长度分桶上均低于 0.6B，且编码、检索耗时及显存占用更高。这只是当前 POI 文本构建方式与原始 Query 输入下的对照结果，不据此自动选择最终模型，也不外推到添加 Query Instruction 后的表现。

下一步由用户核验 E1/E2 对照并决定模型或是否运行 `poi_en` Instruction；本实验完成后停止，不评估 RQ-VAE，也不进入后续生成式检索阶段。
