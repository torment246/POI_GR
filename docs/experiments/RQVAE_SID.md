# RQ-VAE 与 SID 实验

本文件记录从 POI Embedding 到离散 Semantic ID 的正式实验，包括 RQ-VAE、后续 RQ-KMeans 对照、碰撞分析、GID 与完整 SID。SFT 和生成式检索实验不记录在本文件中。

## 当前结论

- Qwen3-Embedding-0.6B 的三层 MiniOneRec 风格 RQ-VAE 全量链路已经跑通。
- 原始最近邻 SID 唯一率为 50.98%，末层 Sinkhorn 重分配后为 70.47%。
- 第一层只使用 130/256 个 code，最终仍有 39.95% 的 POI 位于碰撞组中。
- 当前结果作为 RQ-VAE 基线，不作为最终唯一 POI 编码。
- 下一步需要在“结合 Geohash GID”和“先做初始化、训练轮数、RQ-KMeans 对照”之间确认顺序。

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
