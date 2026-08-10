# GNPR-SID 复现实验

本文件集中记录 GNPR-SID 的两条北京适配路径：先按论文多视图思路构造行为稀疏特征并审计其局限，再构造覆盖全量 POI 的 `BGE-M3 文本 + 类别 + PlusCode6` content-geo 输入，使用 GNPR Diversity Loss 训练三层 RQ-VAE。

## 当前结论

- 8192 用户哈希行为版已完成 703,306 条交互 POI 的三容量训练，但最大碰撞桶由用户哈希别名主导，只保留为消融和失败审计，不冻结为下游标识。
- 全量 content-geo 输入覆盖 2,337,178 条 POI；特征权重固定为 `1/0.25/0.25`，避免类别和区域稀疏块主导重构。
- GNPR 发布代码实际启用的 utilization diversity loss 已接入，`λ=0.25`、内部 scale `0.05`，20 epoch 适配为第 7 轮起启用；未配置该项的 V1/TIGER/GenPOI 行为保持不变。
- 256/512/1024 三容量四卡训练和全量导出均已完成；最终冻结 `512×3 / epoch 20 + conditional Dedup Token`，2,337,178 条 identifier 全局唯一。
- Qwen3-0.6B 四卡 A100 全参数 SFT 已完成 3 epoch；在与 TIGER 相同的固定 10,000 条 Validation 上直接无约束 Beam=10 生成，epoch 3 最优，HR@1/HR@10/NDCG@10 为 49.66%/83.07%/67.2390%，Valid ID Rate 为 78.808%。

## 实验记录
## EXP-20260805-01 GNPR-SID 三容量 RQ-VAE 全量对比

### 目标、数据与代码状态

- 目标与假设：在北京行为 POI 上复现 GNPR-SID 的多视图稀疏特征与三层残差量化，对比 `256×3`、`512×3`、`1024×3`；容量增加应降低完整 SID 碰撞，同时保持各层码本有效使用。
- 数据版本：`outputs/embeddings/gnpr_sid/history10_pluscode6_top10_hash8192_behavior_v1/`，包含 703,306 条 `interaction_count > 0` 的 POI。逻辑输入为 9384 维，由类别 402、区域 766、小时 24 和用户哈希 8192 四块稀疏特征组成；数据以 32 个 Parquet 分片保存，不展开稠密矩阵。
- 目录覆盖：当前 SID 覆盖北京全量 2,337,178 条 POI 的 30.0921%；其余 1,633,872 条无交互 POI 未进入码本训练和本次指标计算。
- 代码状态：基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 GNPR-SID 流式数据集、RQ-VAE、训练恢复、全量 SID 导出和评估实现的未提交工作树。
- 环境：`poi-gr`，Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、PyArrow 19.0.1，单卡 RTX A6000 48GB。

### 配置与命令

- 模型为 `9384→512→256→128→64` 编码器、GNPR 发布实现顺序的 `64→512→256→128→9384` 解码器和三层残差码本；参数量随容量分别约为 6.43M、6.48M、6.58M。
- 重建使用正负位置等权的 balanced MSE；量化权重 1.0、commitment beta 0.25、diversity 权重 0.25、发布实现内部 scale 0.05。diversity 使用 straight-through 的硬利用率梯度并只约束编码器，避免 9384 维北京输入下发布版 softmax 近似导致码本坍塌。
- K-Means 使用 Faiss GPU、固定 100,000 条样本和 20 iterations；训练 batch 512、AdamW、学习率 `3e-4`、weight decay `1e-4`、无 dropout、seed 2024。验证集按 POI ID 稳定哈希取 1%，仅在指定 checkpoint 做验证和全量导出。
- 256 容量评估 epoch 10/20；512 和 1024 仅评估 epoch 20。256 任务曾在 epoch 5 后为取消逐轮评估而受控停止，并从 checkpoint 恢复，正式结果未受影响；512 完成后串行启动 1024，两个任务及串行链退出码均为 0。

~~~bash
python scripts/gnpr/train_rqvae.py \
  --config configs/sid/rqvae_gnpr_sid.yaml \
  --experiment GNPR-SID-256x3

python scripts/gnpr/train_rqvae.py \
  --config configs/sid/rqvae_gnpr_sid.yaml \
  --experiment GNPR-SID-512x3

python scripts/gnpr/train_rqvae.py \
  --config configs/sid/rqvae_gnpr_sid.yaml \
  --experiment GNPR-SID-1024x3
~~~

### 全量 SID 结果

| 容量 / Epoch | Val 重构余弦 | Distinct SID / 唯一率 | excess / 碰撞 POI | P99 / 最大桶 | 全量利用率 L1/L2/L3 |
|---|---:|---:|---:|---:|---:|
| `256×3 / 10` | 10.3964% | 602,057 / 85.6038% | 14.3962% / 21.8147% | 5 / 27 | 97.66% / 100% / 100% |
| `256×3 / 20` | 10.9153% | 606,798 / 86.2780% | 13.7220% / 20.7231% | 5 / 28 | 97.66% / 100% / 100% |
| `512×3 / 20` | 11.2333% | 646,368 / 91.9042% | 8.0958% / 12.1724% | 3 / 21 | 93.36% / 100% / 100% |
| `1024×3 / 20` | 11.3730% | 662,774 / 94.2369% | 5.7631% / 8.8185% | 3 / 17 | 91.11% / 100% / 100% |

| 容量 / Epoch | 类别 Micro P1/P2/P3 | 区域 Micro P1/P2/P3 |
|---|---:|---:|
| `256×3 / 10` | 20.97% / 26.31% / 88.99% | 3.83% / 14.36% / 86.62% |
| `256×3 / 20` | 20.99% / 26.35% / 89.62% | 3.90% / 14.61% / 87.27% |
| `512×3 / 20` | 21.14% / 40.21% / 94.23% | 3.93% / 32.40% / 92.76% |
| `1024×3 / 20` | 21.09% / 66.54% / 96.13% | 4.15% / 62.24% / 94.91% |

### 重构、层级与碰撞审计

- 重构余弦按原 RQ-VAE 的定义，在固定 7,060 条验证 POI 上逐行计算 9384 维重建向量与原始 binary multi-hot 输入的 cosine 后取均值。每条输入平均只激活 11.31 维，中位数 8；同维度常数重建的余弦基准为 3.2902%。因此 `1024×3` 的 11.3730% 明显学习到信号，但绝对重建质量仍弱，且不能与稠密、L2 归一化文本 Embedding 的 80% 余弦直接比较。
- 为避免 Prefix3 Micro Purity 被 singleton 误读，补充同前缀 POI 对的一致率。Prefix1/2/3 的类别一致率为 7.57%/13.16%/23.81%，区域一致率为 1.57%/2.72%/6.38%；随机 POI 对基准分别为 6.83%/1.18%。完整 SID 中 91.18% 的 POI 是 singleton，原表 96.13% 的 Prefix3 类别 Micro Purity 主要由 singleton 拉高。
- 使用 seed 2024 在随机、共享 Prefix1、共享 Prefix2、完整 SID 碰撞四组各抽 10,000 对 POI。输入 multi-hot 余弦依次为 9.34%/12.10%/19.65%/33.75%；小时 Jaccard 为 12.46%/17.97%/11.46%/6.66%，用户哈希 Jaccard 为 0.03%/0.43%/41.96%/98.66%。层级确实从粗到细缩小输入特征范围，但第二、三层主要细化用户哈希行为，而不是 POI 文本语义、类别或地理。
- 最大桶 SID 为 `[400,787,66]`，包含 17 个 POI，跨 9 个类别、14 个区域，最大地理跨度 51.5 km。代表项包括海淀的“金隅凤栖家园南区-2号楼”、西城的“阜成门外大街1号院1号楼”、通州的“大稿新村14号楼-6单元”和朝阳的“cj株式会社”，不构成合理语义簇。
- 最大桶的 17 条记录都只有一个用户哈希且均为桶 5896，但回查原始 `passenger_id` 后是 17 个不同用户。Top 10 最大碰撞桶共 151 个 POI，其中 135 个只有一次交互；桶内 1,069 个 POI 对全部共享同一压缩哈希，实际共享原始用户的只有 7 对，占 0.65%。当前最大碰撞主要是将 3,822,927 个用户压到 8192 桶后产生的哈希别名，不是真实共同用户偏好。

### 产物、结论与下一步

- 正式产物位于 `outputs/sid/gnpr_sid/hash8192_behavior_v1/GNPR-SID-{256,512,1024}x3/`；补充余弦位于上级目录的 `reconstruction_validation_metrics.json`。三组 SID 均为 `[703306,3]`、`int32`，各层 code 范围和行数核验通过。
- 256 的 epoch 20 相比 epoch 10 唯一率提高 0.6741 个百分点。`1024×3 / epoch 20` 相比 512 唯一率提高 2.3327 个百分点、碰撞 POI 比例降低 3.3539 个百分点、最大桶从 21 降到 17，是当前输入方案内的数值最优 checkpoint。
- 当前结果可以保留为“8192 用户哈希适配版”消融结果，也能验证 GNPR 代码链路；但不应直接作为论文 GNPR-SID 的最终可信 baseline，更不应先追加 collision token 掩盖输入哈希造成的伪相似。下一最小步骤应先确定能保留用户身份相似性的有界编码，再重做一个最小容量对照。无交互 POI 仍需在全目录评测中按 miss 处理并单独报告目录覆盖率。
## EXP-20260805-02 GNPR content-geo 全量 SID（已中止）

### 目标、数据与代码状态

- 目标与假设：按 GenPOI 地图搜索场景对 GNPR-SID 做 content-geo 适配，以全量 BGE-M3 文本向量显式融合 `category_code` 和 POI PlusCode6；移除缺失严重的访问时间和不可扩展的用户身份块，使冷 POI 也能进入统一 SID 目录。首个正式容量只运行 `256×3 / 20 epoch`，验证训练适配稳定后再决定 512/1024。
- 数据版本：`outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/`，严格按 BGE-M3 行序覆盖 2,337,178 条 POI。三个独立单位范数块为 BGE-M3 1024 维、类别 402 维 one-hot 和 PlusCode6 766 维 one-hot，拼接后除以 `sqrt(3)`，逻辑输入共 2192 维；不设置 UNK，无效静态索引直接阻断。
- 输入核验：BGE、静态特征和紧凑索引均为 2,337,178 行；类别范围 `0–401`、区域范围 `0–765`，无无效索引。训练按 batch 从 BGE memmap 和两个 `int32` 索引构造融合输入，不写全量稠密矩阵。
- 代码状态：基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 content-geo 输入、可选流式 Vanilla RQ-VAE 数据源和导出兼容性的未提交工作树。
- 环境：`poi-gr`，Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128，单卡 NVIDIA RTX A6000 48GB。

### 配置与命令

- 模型与 TIGER 同构：`2192→512→256` 编码器、对称解码器、三层残差码本；重建为 element-mean MSE，codebook/commitment 权重为 1.0/0.25。
- K-Means 使用 Faiss GPU、固定 500,000 条训练 POI、20 iterations、seed 42；训练 batch 4096、流式 block 8192、学习率 `3e-4`、weight decay 0、验证比例 1%、固定 checkpoint 为 epoch 20。
- K-Means 三层均使用全部 256 个码，初始化样本均方距离依次为 0.00124449/0.00073305/0.00049922，初始化总耗时 65.45 秒。

~~~bash
python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_gnpr_content_geo.yaml \
  --experiment GNPR-ContentGeo-BGE-M3-256x3
~~~

### 当前指标、产物与结论

- 状态：在 epoch 2 完成后受控中止，未生成 epoch 20 checkpoint，也未启动 512/1024。任务进程已退出，诊断产物保留在 `outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/GNPR-ContentGeo-BGE-M3-256x3/`，日志位于同级 `.jobs/256.log`。
- epoch 1：Validation 重构余弦为 70.7347%，三层使用码数为 55/113/137；固定 23,372 条验证子集的 SID distinct ratio 为 18.5692%，最大桶为 653。
- epoch 2：Validation 重构余弦升至 73.6392%，但使用码数退化为 55/64/10，第三层归一化熵仅 0.0121；SID distinct ratio 降至 5.1215%，最大桶增至 2,003。重构改善与标识区分能力相反，判定为训练坍塌。
- 原因审计：三个单位范数块等权融合，使仅 402/766 种的类别和区域共同占输入 2/3 平方能量；同时 `element_mean` MSE 从 1024 扩到 2192 维后，重构项相对量化项缩至 TIGER 的 46.72%。K-Means 三层初始化均使用全部码，排除输入缺失和初始化死码作为主因。
- 下一步：将文本/类别/区域相对权重改为 `1/0.25/0.25`，并以 `2192/1024` 恢复重构损失尺度；先做全量数据、256×3、2 epoch 的稳定性闸门。
## EXP-20260805-03 GNPR content-geo 加权适配诊断（已完成）

### 目标、数据与代码状态

- 目标与假设：修正 EXP-20260805-02 的特征能量和损失尺度失配；若坍塌由等权稀疏块捷径引起，则第二轮完整 SID 唯一率不应再断崖下降，第三层利用率与熵应保持有效。
- 数据版本：沿用 `outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/` 的 2,337,178 条全量 POI、固定 BGE 行序和无 UNK 的 402 类别/766 PlusCode6 索引；训练仍按 batch 构造 2192 维输入，不写全量稠密矩阵。
- 代码状态：基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用未提交工作树；新增运行时特征块权重和独立重构损失权重，默认值保持旧 RQ-VAE/TIGER/GenPOI 行为不变。
- 环境：`poi-gr`，单卡 NVIDIA RTX A6000 48GB；启动前显存占用 0 MiB、GPU 利用率 0%。

### 配置、命令与验收

- 文本/类别/区域相对 L2 权重为 `1.0/0.25/0.25`，整体归一化后平方能量占比为 88.89%/5.56%/5.56%；重构损失权重为 `2.140625 = 2192/1024`。其余模型、K-Means、划分、batch、优化器和 seed 与 EXP-20260805-02 相同。
- 本次使用全量 POI 和 500,000 条 K-Means 样本，只把最大训练轮数与 checkpoint 改为 epoch 2；不属于缩小数据或码本的 smoke。

~~~bash
python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_gnpr_content_geo.yaml \
  --experiment GNPR-ContentGeo-BGE-M3-256x3 \
  --output-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/diagnostics/GNPR-ContentGeo-BGE-M3-256x3-weighted-v2-e2 \
  --max-epochs 2 \
  --checkpoint-epochs 2 \
  --no-resume
~~~

### 结果、产物与结论

| Epoch | Val cosine | Val 码字数 L1/L2/L3 | Val 熵 L1/L2/L3 | Distinct SID / ratio | P99 / Max |
|---|---:|---:|---:|---:|---:|
| 1 | 77.1433% | 37/104/88 | 0.6236/0.8143/0.7913 | 18,323 / 78.3972% | 5 / 21 |
| 2 | 79.0529% | 41/162/126 | 0.6454/0.9009/0.8648 | 20,034 / 85.7180% | 4 / 18 |

- 状态为 `completed`，固定 epoch 2 checkpoint、恢复 checkpoint 和两行连续训练指标均通过一致性检查；训练阶段耗时 365.10 秒。K-Means 三层均使用全部 256 个码，均方距离为 0.00114443/0.00093629/0.00080845。
- 与旧等权配置相比，epoch 1 唯一率由 18.5692% 提升到 78.3972%，最大桶由 653 降到 21；epoch 2 没有复现唯一率跌至 5.1215%、第三层仅使用 10 个码的坍塌，反而提升到 85.7180%、126 个码，稳定性闸门通过。
- 这组指标来自固定 23,372 条 Validation 监控子集，不是 2,337,178 条全量 SID 导出指标。产物位于 `outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/diagnostics/GNPR-ContentGeo-BGE-M3-256x3-weighted-v2-e2/`。
- 结论：特征能量与损失尺度失配是上一实验坍塌的主要原因，当前加权方案可以进入 20 epoch 正式验证。下一步应只延长同一 256 配置，完成后导出全量 SID；在 256 通过最终闸门前仍不启动 512/1024。
## EXP-20260805-04 GNPR content-geo 加权版 256 正式训练（已中止）

### 目标、数据与代码状态

- 目标与假设：沿用 EXP-20260805-03 已通过两轮稳定性闸门的加权方案，完成全量北京 POI 的 `256×3 / 20 epoch` 正式训练；假设后续轮次保持有效的残差码本组合，不再出现旧等权配置的第二轮坍塌。
- 数据版本：`outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/`，严格按 BGE-M3 行序覆盖 2,337,178 条 POI；类别 402、PlusCode6 区域 766，无 UNK 和无效索引。
- 代码状态：基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 GNPR content-geo 加权融合和重构损失倍率的未提交工作树；相关 12 个合成测试、compileall 与 diff check 已通过。
- 环境：`poi-gr`，单卡 NVIDIA RTX A6000 48GB；启动前显存占用 0 MiB、GPU 利用率 0%。

### 配置、命令、产物与验收

- 文本/类别/区域相对权重为 `1.0/0.25/0.25`，整体归一化后平方能量占比为 88.89%/5.56%/5.56%；重构损失权重为 `2.140625`。模型为 `2192→512→256`、三层 256 残差码本；Faiss GPU K-Means 样本 500,000、20 iterations，训练 batch 4096、学习率 `3e-4`、seed 42、固定 checkpoint epoch 20。
- 诊断任务的 `max_epochs=2` 属于配置签名，不能直接恢复到 20 轮；正式任务在新目录使用相同 seed 和协议重新初始化，避免篡改 checkpoint 签名。

~~~bash
python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_gnpr_content_geo.yaml \
  --experiment GNPR-ContentGeo-BGE-M3-256x3 \
  --output-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/GNPR-ContentGeo-BGE-M3-256x3-weighted-v2
~~~

- 任务在完成 epoch 5 后按用户要求停止，以切换到 4×6000D 训练平台；训练进程和监督进程均已退出，未生成 epoch 20 固定 checkpoint。中止任务的输出目录及 `.jobs/256_weighted_v2.*` 随后按用户要求清理，约 32MB，已不可恢复；EXP-20260805-03 的两轮诊断产物继续保留。
- epoch 5 的固定 Validation 监控子集唯一率为 91.3700%，三层使用码数为 77/237/204、归一化熵为 0.7233/0.9727/0.9482，P99/最大桶为 3/14，未出现坍塌。这些是删除前从五行连续训练指标核验的实际结果，不代表完成的 20 epoch 实验。
- 此后发现当时共享 RQ-VAE 训练目标尚未接入 GNPR Diversity Loss；已停止平台上的无 Diversity Loss 任务并清空其 `4x6000d/` 输出，不把中途指标登记为正式结果。加入损失后的正式三容量任务另记为 `EXP-20260805-05`。

## EXP-20260805-05 GNPR content-geo 三容量 Diversity Loss 正式训练

### 目标与假设

- 目标：在同一套全量 content-geo 输入和加权重构配置上，为 `256×3/512×3/1024×3` 加入 GNPR 发布代码实际生效的码本利用率 Diversity Loss，并行训练 20 epoch。
- 假设：从约三分之一训练进度开始加入利用率约束，可以改善各层码本覆盖和 SID 碰撞，同时保留前 6 轮已建立的重构结构。
- 状态：2026-08-05 15:00（Asia/Shanghai）启动，三组均训练至 epoch 20、生成固定 checkpoint 并以退出码 0 结束；全量 SID 尚未导出，当前监控子集指标不能替代全量结果。

### 数据、代码、配置与环境

- 数据仍为 `outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/`，覆盖并严格对齐 2,337,178 条 POI；逻辑输入为 BGE-M3 1024 维、类别 402 维和 PlusCode6 766 维。
- 文本/类别/区域相对权重为 `1.0/0.25/0.25`，重构损失权重为 `2.140625`；三层码本容量是本实验唯一的三组结构变量。
- 总目标为既有重构与量化目标加 `0.25 × Ldiv`；`Ldiv` 使用发布代码实际启用的 hard-utilization straight-through 梯度、内部 scale `0.05`、温度 `0.5`，只向编码器回传。论文文字中的 compactness 项在发布实现中被注释，本实验未自行恢复。
- 前 6 epoch 不计算 Diversity Loss，第 7 epoch 起真正计算并加入总目标；未配置该项的 V1、TIGER 和 GenPOI 配置默认权重为 0，不改变旧实验行为或签名。
- 平台为 4 张 NVIDIA RTX 6000D；256/512/1024 分别绑定 GPU 0/1/2，GPU 3 保留余量。Python 3.10.20、PyTorch 2.9.1+cu128，Adam 学习率 `3e-4`、batch 4096、seed 42。

~~~bash
bash launchers/run_train_gnpr_content_geo_4x6000d_20epoch.sh
~~~

### Epoch 20 监控结果与产物

| 容量 | Val cosine | Val diversity loss | Val 码字数 L1/L2/L3 | Val 熵 L1/L2/L3 | Monitor SID 唯一率 | P99 / Max |
|---|---:|---:|---|---|---:|---|
| `256×3` | 76.4418% | 0.180498 | 184/232/237 | 0.9149/0.9590/0.9530 | 97.8479% | 2 / 5 |
| `512×3` | 79.5766% | 0.330300 | 448/461/437 | 0.9642/0.9670/0.9529 | **99.5550%** | 1 / 3 |
| `1024×3` | 74.0899% | 3.841584 | 240/116/130 | 0.7087/0.4091/0.3932 | 42.9018% | 14 / 51 |

- 256/512 在第 7 轮启用 Diversity Loss 后保持有效码本使用，epoch 20 的监控 SID 唯一率继续高于启用前第 6 轮的 89.7527%/94.1340%。
- 1024 在 epoch 7—16 一度稳定，epoch 16 监控 SID 唯一率为 99.2042%；从 epoch 17 起总 loss 和 Diversity Loss 突增，epoch 18 唯一率降至 20.5331%，epoch 20 仍只有 42.9018%，判定为后期训练坍塌。不能只按码本容量更大选择 1024。
- 三组 `train_metrics.jsonl` 均为连续 20 行，`stop_reason=max_epochs`，训练耗时为 827.51/769.72/828.82 秒；`checkpoint_epoch_20.pt` SHA256 分别为 `2e337bfb87af9b43d82ea0431996fd8233e5166f5d9e023f18f4c6d8b40807f5`、`9349fef24499ad4110c7d405c30fc90d32de92f504dd6c1dde0e00b0210b5d18`、`aeed65bab44c5598d7148c1b777c256f42d0d186559c3f3a1444a088ffaf8d2d`。
- 输出位于 `outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-{256,512,1024}x3/`，日志位于同级 `.jobs/{256,512,1024}.log`。
- 结论：训练阶段暂以 512×3 最稳定，但上述唯一率只来自固定 23,372 条 Validation 监控子集。三容量全量导出和最终选择另记为 `EXP-20260805-06`。

## EXP-20260805-06 GNPR content-geo 三容量全量 SID 评估

### 目标、数据与代码状态

- 目标：从 EXP-20260805-05 的三个 epoch 20 固定 checkpoint 导出全部 2,337,178 条北京 POI 的三层 SID，按与 TIGER 完全相同的评估器比较重构、码本利用率、完整 SID 唯一率、碰撞桶和逐层前缀类别纯度，并冻结 GNPR 下游候选容量。
- 假设：固定 23,372 条 Validation 监控子集只适合判断训练稳定性，不能直接估计 233 万目录的真实碰撞；若 1024 后期坍塌已经污染固定 checkpoint，则全量评估应出现大规模码本失活和碰撞，而不是容量带来的自然提升。
- 数据版本：沿用 `outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/`，BGE-M3、402 类别和 766 PlusCode6 索引与 2,337,178 个唯一 POI ID 严格同序；三组 SID shape 均为 `[2337178,3]`、dtype 为 `int32`。
- 代码状态：基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 GNPR content-geo、Diversity Loss 和统一 SID 评估器的未提交工作树；本实验没有修改训练参数或 checkpoint。
- 环境：Python 3.10.20；服务器真实环境使用一张空闲 NVIDIA RTX A6000 48GB 顺序完成 GPU 编码，碰撞、前缀和 POI 元数据指标在 CPU 侧流式/内存映射计算，避免三组并行导致 OOM。

### 配置与命令

- 三组均人工选择 `checkpoint_epoch_20.pt`；导出 batch 为 8192。评估指标和 TIGER 的 `EXP-20260730-01` 均来自 `scripts/sid/export_rqvae.py` 与 `src/poi_gr/sid/evaluation.py`，没有为 GNPR 修改唯一率或碰撞定义。

~~~bash
python scripts/sid/export_rqvae.py \
  --checkpoint checkpoint_epoch_20.pt \
  --run-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-<CAPACITY>x3 \
  --output-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-<CAPACITY>x3/evaluations/epoch_20 \
  --device cuda \
  --batch-size 8192
~~~

### Epoch 20 重构与全量碰撞指标

Val 重构和码字数来自固定 23,372 条 Validation；完整 SID、碰撞、全量利用率和熵来自 2,337,178 条全量 POI。

| 容量 | Val total / recon / cosine | Val 码字数 L1/L2/L3 | Distinct SID / ratio | excess / colliding POI | P99 / Max | 全量利用率 L1/L2/L3 | 全量熵 L1/L2/L3 |
|---|---|---|---|---|---|---|---|
| `256×3` | 0.050457 / 0.000189 / 76.44% | 184/232/237 | 996,517 / 42.6376% | 57.3624% / 78.1474% | 12 / 191 | 73.83%/93.36%/94.92% | 0.916/0.960/0.954 |
| `512×3` | 0.083977 / 0.000167 / **79.58%** | 448/461/437 | **1,818,251 / 77.7969%** | **22.2031% / 35.5242%** | **5 / 223** | 88.28%/92.38%/90.82% | 0.966/0.968/0.954 |
| `1024×3` | 0.962084 / 0.000206 / 74.09% | 240/116/130 | 73,183 / 3.1313% | 96.8687% / 98.8941% | 535 / 4,757 | 44.24%/36.13%/47.95% | 0.709/0.409/0.394 |

- 256 的监控唯一率 97.8479% 在全量目录上降为 42.6376%；512 从 99.5550% 降为 77.7969%。这不是导出错误，而是监控子集只有全量的 1%，其中大量全量碰撞桶只抽到一个成员，因此监控唯一率系统性偏高。
- 512 在三组中同时取得最高 Val reconstruction cosine、最高完整 SID 唯一率、最低碰撞 POI 比例和有效的三层码本熵，是本轮唯一满足重构与离散质量平衡的容量。
- 1024 只有 73,183 个不同 SID；最大桶包含 4,757 个 POI，热点桶混入地铁站、楼栋、住宅区和门牌等不同类别，说明这不是正常的同义/近邻聚类，而是 checkpoint 坍塌后的非语义聚集。

### 前缀类别纯度

表中为各深度的 `prefix bucket count / category micro purity / P99 / Max`；类别覆盖率均为 100%。完整深度纯度会受到大量 singleton 桶抬升，需和深度 1、2 一起解释。

| 容量 | Depth 1 | Depth 2 | Depth 3 |
|---|---|---|---|
| `256×3` | 189 / 35.34% / 27,238 / 29,162 | 27,501 / 41.39% / 407 / 1,447 | 996,517 / 75.27% / 12 / 191 |
| `512×3` | 452 / 44.63% / 9,359 / 9,757 | 153,916 / 62.02% / 86 / 505 | 1,818,251 / 96.69% / 5 / 223 |
| `1024×3` | 453 / 48.48% / 39,839 / 67,526 | 11,052 / 50.76% / 3,786 / 14,161 | 73,183 / 54.06% / 535 / 4,757 |

- 512 的前两层类别纯度仍低于 TIGER 最佳 1024 的 69.28%/77.27%，说明当前 GNPR content-geo 的区分能力主要在第三层释放，粗到细的类别层级不如 TIGER 清晰；Depth 3 的 96.69% 高纯度部分来自 64.48% POI 已成为 singleton，不能单独作为语义层级更好的证据。

### 1024 后期坍塌分析

- 坍塌有明确时间证据：epoch 16 的监控唯一率仍为 99.2042%，Val 三层使用 834/859/692 个码；epoch 17 的 Val diversity loss 从 0.6663 升至 1.9233、第三层骤降到 261 个码；epoch 18 进一步变为 211/121/99 个码、监控唯一率 20.5331%。epoch 20 只是部分恢复，无法修复已形成的集中映射。
- 当前 hard-utilization loss 在每个 batch 内要求均匀使用码本。batch 4096 下，256/512/1024 的每码期望样本数分别只有 16/8/4；1024 的硬计数最稀疏、方差最大，assignment 边界对编码器小幅变化最敏感。
- Diversity Loss 在第 7 轮一次性启用。1024 的 train total loss 从 epoch 6 的 0.000568 跳到 epoch 7 的 2.628861，其中内部 diversity loss 为 10.0614、乘外部权重后贡献约 2.515；相同冲击在 256/512 上较小。该目标的前向值使用硬最近邻计数，而梯度使用 soft straight-through 且只作用于编码器，码本在 diversity 分支中 detach。高容量下编码器分布与码本更新一旦失配，硬分配会跨越大量边界，形成“利用率失衡→diversity 梯度增大→编码器继续漂移”的正反馈。
- epoch 18 的 train used codes 仍显示 933/920/894，是因为训练指标聚合了整轮中持续变化的模型；同一轮结束后的固定 Val 只有 211/121/99。全量 epoch 20 快照的利用率和熵与 Val 一致，因此不能用 train 聚合利用率否认坍塌。
- 现有 epoch 汇总没有逐 step 梯度范数，且未保存 epoch 16 checkpoint，因此无法把触发点归因到某一个 batch；可以确定的是“当前 batch/权重/突然启用策略下的容量相关优化不稳定”，不能据此断言 GNPR 方法或 1024 容量本身必然坍塌。

### 与 TIGER 的同口径比较

以下均比较追加 collision token 之前的三层 base SID；TIGER 追加第四层后会全局唯一，GNPR 若进入 SFT 也需要独立追加确定性 collision token。

| 容量 | GNPR / TIGER 唯一率 | GNPR / TIGER colliding POI | GNPR / TIGER P99 | GNPR / TIGER Max |
|---|---|---|---|---|
| `256×3` | 42.64% / 48.09% | 78.15% / 68.13% | 12 / 15 | 191 / 598 |
| `512×3` | **77.80% / 61.72%** | **35.52% / 52.76%** | **5 / 10** | **223 / 389** |
| `1024×3` | 3.13% / 71.68% | 98.89% / 40.79% | 535 / 7 | 4,757 / 306 |

- GNPR 最佳 512 相比 TIGER 同容量唯一率高 16.08 个百分点；即使与 TIGER 已选中的 1024 比，仍高 6.12 个百分点，碰撞 POI 低 5.27 个百分点，P99/最大桶为 5/223 对 7/306。
- TIGER-1024 的 Val cosine 为 88.52%，高于 GNPR-512 的 79.58%，但前者重构 1024 维归一化 BGE，后者重构加权的 2192 维文本/类别/区域输入且损失目标不同，数值不能作严格横向优劣结论；在各自方法内部，TIGER 选 1024、GNPR 选 512 是一致的容量选择方式。

### 产物、结论与下一步

- 三组正式评估目录均为各实验下的 `evaluations/epoch_20/`，包含 `sid_codes.npy`、`sid_manifest.json`、`metrics.json` 和 `collision_cases.jsonl`；GPU 导出耗时分别为 65.20/65.97/176.28 秒。
- 三组 `sid_codes.npy` SHA256 分别为 `157a129e585ec699eb85cb9a5d9de01509eaa656069d0a757c1eec52175d1945`、`4d47272cea870bdccec8f711097956ff728bc156fb05718c4f4f51f46e0fcff6`、`92f558765a3e688b83a087d21c66c90823f083e9069e1570adf693d64c1000c6`；`metrics.json` SHA256 分别为 `c967de5fe731ed4602c4a79e2d40181808403e8600659b460bbcd1c5495961ee`、`477f2a6d8799b2421f2daa696e439c1e84802174826ba08d9f282caff8b674ba`、`65c90dd21e4e34fe22593d7f311fab84e63ea0abe7b224ff397c654d53a6b70b`。
- 最终选择 `512×3 / epoch 20` 作为 GNPR-SID 下游候选；256 作为低容量对照保留，1024 明确标记为失败 checkpoint，不进入训练数据。下一最小步骤是为 GNPR-512 追加确定性 collision token 并核验全局唯一映射，不在本实验提前构建 SFT 数据。

## EXP-20260805-07 GNPR 1024×3 epoch 16 可复现性与全量评估

### 目标、数据与实验边界

- 目标：验证 EXP-20260805-05 原训练日志中尚未坍塌的 `1024×3 / epoch 16` 是否应替代 `512×3 / epoch 20`。由于原正式任务只保存 epoch 20，无法从日志恢复原 epoch 16 权重，本实验使用相同数据、seed、损失、batch 和优化器从头复跑到 epoch 16，再做 2,337,178 条全量 SID 评估。
- 假设：若 1024 的坍塌只发生在原任务 epoch 17 以后，复跑 epoch 16 应稳定复现原日志的码本利用率，并在全量唯一率或前缀纯度上明显优于 512。
- 数据仍为 `outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/` 的全量 BGE-M3、402 类别和 766 PlusCode6 索引，POI 数量和行序与前述三容量实验完全一致。
- 本实验只能回答“同配置重新训练到16轮是否可复现并可用”，不能冒充缺失的原6000D epoch 16 checkpoint。后续曾按用户要求从该checkpoint受控延长17–20轮，但用户随后取消20轮分析，本实验及最终选择只使用epoch 16结果。

### 配置、命令与环境

- 配置保持 `1024×3`、batch 4096、学习率 `3e-4`、Diversity Loss 从 epoch 7 启用、外部权重 0.25、内部 scale 0.05、温度 0.5；只把最大轮数和固定checkpoint设置为16。
- 环境为Python 3.10.20和服务器真实环境的一张NVIDIA RTX A6000 48GB；三层Faiss GPU K-Means均使用500,000条样本和20次迭代。训练耗时1,917.18秒。

~~~bash
python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_gnpr_content_geo.yaml \
  --experiment GNPR-ContentGeo-BGE-M3-1024x3 \
  --output-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/epoch16_rerun_a6000/GNPR-ContentGeo-BGE-M3-1024x3 \
  --max-epochs 16 \
  --checkpoint-epochs 16 \
  --no-resume \
  --no-progress

python scripts/sid/export_rqvae.py \
  --checkpoint checkpoint_epoch_16.pt \
  --run-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/epoch16_rerun_a6000/GNPR-ContentGeo-BGE-M3-1024x3 \
  --output-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/epoch16_rerun_a6000/GNPR-ContentGeo-BGE-M3-1024x3/evaluations/epoch_16 \
  --device cuda \
  --batch-size 8192
~~~

### 复跑轨迹核验

| 节点 | 原6000D日志 | A6000复跑 | 结论 |
|---|---|---|---|
| epoch 1 Val cosine | 78.1607% | 78.1787% | 早期重构高度一致 |
| epoch 6 Val cosine / monitor唯一率 | 82.9420% / 95.8283% | 82.8616% / 95.7684% | Diversity启用前高度一致 |
| epoch 7 Val diversity / monitor唯一率 | 0.7049 / 98.8533% | 0.7155 / 98.5581% | 损失切换后仍接近 |
| epoch 16 Val cosine | 83.3875% | 80.1882% | 后期开始分化 |
| epoch 16 Val码本 L1/L2/L3 | 834/859/692 | 844/726/264 | 复跑第三层已明显失活 |
| epoch 16 monitor唯一率 | 99.2042% | 98.3870% | 1%子集仍掩盖全量碰撞 |

- 复跑到epoch 10仍与原轨迹接近，但epoch 16第三层只使用264个码、熵为0.6143；原日志同轮为692个码、熵0.9025。说明当前1024配置对GPU数值差异和优化轨迹敏感，不能把原日志的高监控唯一率当作可稳定复现的checkpoint质量。

### 2,337,178条全量SID指标

| 指标 | GNPR 1024@16复跑 | GNPR 512@20 | TIGER 1024@20 |
|---|---:|---:|---:|
| Val reconstruction cosine | 80.1882% | 79.5766% | 88.52% |
| Distinct SID | 1,476,624 | **1,818,251** | 1,675,209 |
| 完整SID唯一率 | 63.1798% | **77.7969%** | 71.6766% |
| singleton POI比例 | 48.0476% | **64.4758%** | 59.2106% |
| colliding POI比例 | 51.9524% | **35.5242%** | 40.7894% |
| 碰撞桶P99 / Max | 9 / 229 | **5 / 223** | 7 / 306 |
| 全量码本使用 L1/L2/L3 | 886/792/464 | 452/473/465 | 561/1024/1024 |
| 全量熵 L1/L2/L3 | 0.944/0.885/0.618 | **0.966/0.968/0.954** | 0.858/0.969/0.973 |
| 类别micro purity D1/D2/D3 | **71.45%/85.17%/97.19%** | 44.63%/62.02%/96.69% | 69.28%/77.27%/96.28% |

- 1024@16的Val cosine只比512@20高0.61个百分点，但全量唯一率低14.62个百分点、碰撞POI高16.43个百分点，第三层利用率45.31%且熵只有0.618；当前可用checkpoint没有形成比512更均衡的离散标识。
- 1024@16的优势是Depth 1/2类别纯度明显高于512，甚至略高于TIGER-1024，说明较大码本在坍塌前能形成更清晰的类别层级；但其第三层已处于失活边缘，且跨GPU复跑不稳定。对后续唯一item identifier和可复现实验而言，这个语义优势不足以抵消碰撞和稳定性问题。
- 全量`sid_codes.npy`、manifest、统一指标和Top 20碰撞桶位于上述`evaluations/epoch_16/`；SID SHA256为`c3dd1e5a0607882843bcd72ed04071dd5fa8c429dcbdfc09357b71a7b78de27a`，指标SHA256为`1a2d3975fa3099a3859419b47a4b8d1728d074e3ddbe284efcd24e8f6bab1d0e`。
- 结论：不使用1024@16替换512@20。`512×3 / epoch 20`继续作为当前GNPR下游候选；1024@16保留为“前缀类别纯度较高、但全量碰撞和可复现性未通过”的消融结果。

## EXP-20260805-08 GNPR 512×3 论文 Dedup Token 唯一标识

### 目标与假设

- 目标：将已选定的 `512×3 / epoch 20` 三层 base SID 转换为可供 GNPR 下游生成训练使用的唯一 POI 标识。
- 假设：按照论文 §4.1.2 和作者 V1 数据，仅对三层 SID 发生碰撞的 POI 追加 `<d_n>`，可在保留单例三 Token 表示的同时得到全量唯一标识；同桶编号按 `poi_id` 字典序固定后应可确定复跑。
- 与 TIGER 的区别：TIGER 当前实现为所有 POI 固定追加第四个 `C` Token；GNPR 论文和作者样例中，单例仍为 `<a_i><b_j><c_k>`，只有碰撞 SID 才变为 `<a_i><b_j><c_k><d_n>`。

### 数据、代码状态与配置

- 输入为 `GNPR-ContentGeo-BGE-M3-512x3/evaluations/epoch_20/sid_manifest.json` 对齐的 2,337,178 条三层 SID；SID SHA256 为 `4d47272cea870bdccec8f711097956ff728bc156fb05718c4f4f51f46e0fcff6`，码本容量为 `512×3`。
- 本实验运行于未提交工作树；工作树包含用户既有修改和本任务新增的 GNPR identifier 模块、CLI 与合成测试，未创建 commit。
- 碰撞桶内按 `poi_id` 字典序从 0 连续编号；单例内部 NPY 第四列使用 `-1` 哨兵，序列化和 SFT 时省略第四 Token；碰撞行序列化为 `D0-D222`。
- 环境为服务器 CPU、Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1；Parquet 按 100,000 行分块、Zstd level 3 流式写入，没有加载订单数据或使用 GPU。

### 命令

~~~bash
python scripts/gnpr/build_identifiers.py \
  --sid-manifest outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-512x3/evaluations/epoch_20/sid_manifest.json \
  --output-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-512x3/gnpr_ids/epoch_20 \
  --chunk-rows 100000
~~~

### 核心指标与核验

| 指标 | 结果 |
|---|---:|
| POI / base SID distinct | 2,337,178 / 1,818,251 |
| base SID 唯一率 | 77.7969% |
| 碰撞桶 / 碰撞 POI | 311,336 / 830,263 |
| 保持三 Token 的单例 POI | 1,506,915 |
| Dedup Token 容量 | 223（`D0-D222`） |
| 最终 identifier distinct / 唯一率 | 2,337,178 / 100% |
| 第二次全量构建耗时 | 56.93 秒 |
| 两次全量映射 SHA256 | `3cee68db4f1360b1103452e0e9d1c5e49edf5f11bf70e38b514c1037911abde2`（一致） |

- 合成测试覆盖单例不追加、碰撞桶连续编号、三层 SID 可逆、全局唯一和重复构建哈希一致，共 3 项通过。
- 全量复跑后再次检查 `gnpr_ids.npy` 为 `[2337178,4]` 内部表示、Parquet 为 2,337,178 行，manifest 声明 POI ID 和序列化标识均唯一；全量两次映射哈希完全一致。

### 产物、结论与下一步

- 正式目录为 `outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-512x3/gnpr_ids/epoch_20/`，包含 `dedup_codes.npy`、`gnpr_ids.npy`、`poi_gnpr_id_mapping.parquet`、`gnpr_id_manifest.json` 和 `metrics.json`。
- 结论：`512×3 / epoch 20 + conditional Dedup Token` 通过全量唯一、可逆和确定性核验，冻结为 GNPR-SID 下游标识；256 和 1024 不进入 GNPR SFT 数据。
- 下一步单独定义 GNPR SFT 数据契约。论文下游训练除普通 next-POI 生成外，明确使用每条历史访问时间与目标时间，并在附录给出多段裁剪及 20% 历史位置填空增强；作者公开 V1 SFT 样例和 V2 转换代码只展示标准 next-POI 样本，未包含额外生成损失、负采样或地理约束解码。

## EXP-20260805-09 GNPR 地图检索 SFT 全量数据与训练入口

### 目标与假设

- 目标：使用冻结的 `512×3 + conditional Dedup` identifier，将 003 的目标订单和最近 10 条历史转换为可直接用于 Qwen3-0.6B SFT 的 Train/Valid/Test Messages，并准备与 TIGER 同口径的三轮训练入口。
- 假设：在相同目标订单、时间切分、历史长度、当前 Query/GID 输入和全局 batch 下，仅替换 POI identifier，可以公平观察 GNPR SID 对下游生成检索的影响。
- 本版为地图检索适配，不输入访问时间、目标时间和用户 ID，也不启用论文附录的多段裁剪或 20% 历史位置填空增强；因此不是 GNPR 原始推荐任务的严格复刻。

### 数据、代码状态与配置

- 输入为 003 的 32 个 JSONL 分片，共 8,790,513 条目标订单；历史窗口为 2026-04-01—06-30，最多保留最近 10 条，目标按 2026-07-01—12/07-13/07-14 切分。
- identifier 映射覆盖 2,337,178 条 POI，映射 SHA256 为 `3cee68db4f1360b1103452e0e9d1c5e49edf5f11bf70e38b514c1037911abde2`；三层单例省略 `d` Token，碰撞项使用 `d0-d222`。
- 代码运行于未提交工作树，包含 GNPR 数据构建器、扩词表复用入口、数据注册、正式/Smoke 配置和四卡训练脚本；未创建 commit。
- 环境为 `poi-gr`、Python 3.10.20；全量数据构建在服务器 CPU 上逐行读取订单，identifier NPY 使用 mmap，输出采用临时目录完成后原子替换。

### 命令

~~~bash
python scripts/gnpr/build_sft_data.py \
  --orders-dir data/beijing_order_clean_20260701_20260714_history10_20260401_20260630_json \
  --gnpr-id-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-512x3/gnpr_ids/epoch_20 \
  --output-dir data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1 \
  --train-start 2026-07-01 --train-end 2026-07-12 \
  --valid-date 2026-07-13 --test-date 2026-07-14 \
  --history-start 2026-04-01 --history-end 2026-06-30 \
  --max-history-events 10 --geohash-length 6

python scripts/pid/prepare_vocab.py \
  --model-dir models/Qwen3-0.6B \
  --tokens data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/special_tokens.json \
  --output-dir models/Qwen3-0.6B-GNPR-Vocab-v1 \
  --expected-token-count 1805 \
  --mapping-filename poi_token_mapping.json \
  --schema-version gnpr-vocab-v1

python scripts/sft/validate_tokenization.py \
  --model-dir models/Qwen3-0.6B-GNPR-Vocab-v1 \
  --train-file data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/train.jsonl \
  --valid-file data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/valid.jsonl \
  --output-dir data/sft/tokenized/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1 \
  --dataset-dir configs/sft \
  --cutoff-len 512 \
  --batch-size 4096 \
  --workers 4 \
  --preprocessing-batch-size 1000 \
  --train-dataset gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_train \
  --valid-dataset gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_valid \
  --mapping-filename poi_token_mapping.json \
  --smoke-train-rows 10000 \
  --smoke-valid-rows 2000
~~~

### 核心结果与产物

| 指标 | 结果 |
|---|---:|
| 扫描 / 保留订单 | 8,790,513 / 8,790,513 |
| Train / Valid / Test | 7,586,410 / 597,421 / 606,682 |
| 空 Query | 0 |
| 历史覆盖率 / 平均长度 | 72.0157% / 4.9153 |
| 空历史样本 | 2,459,967 |
| 新增 Token / 扩展后词表 | 1,805 / 153,474 |
| Packed Train / Validation Cache | 2,758,797 / 201,947 |
| Smoke Packed Train / Validation | 3,661 / 670 |
| 超过 512 Token / Assistant 目标截断 | 1,706 / 0 |
| 数据 Manifest SHA256 | `f818a689099d4e40519a27254acd3fddadafb1ff4cd2abe4a84ceb57e1d85ce4` |

- 正式数据位于 `data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/`；Train/Valid/Test SHA256 分别为 `07d070618c5b3e0a113a44d9d085ea932a10126361023e0e7a0807c7a80f23da`、`b685c9b4e6f76ce05aa346a838a06d5089bdf2f23c9fc1fdc95a753221110ef8`、`4af8b5cc116dd1b4e5ba8a469d52365686269b7f0e81c9bed49f3e65f7604a38`。
- 32 个输入分片均完整扫描；抽样核验 Messages 只包含 `user/assistant`，目标符合 `<a_i><b_j><c_k>[<d_n>]`，Prompt 不含时间、用户 ID 或 passenger 字段。
- 扩词表模型位于 `models/Qwen3-0.6B-GNPR-Vocab-v1/`；四卡训练入口为 `launchers/run_train_gnpr_sft_4x6000d_3epoch.sh` 和 `launchers/run_train_gnpr_sft_4a100_3epoch.sh`，两者全局 batch 均为 `16×8×4=512`，按 epoch 保存且最多保留 3 个完整 checkpoint。
- 正式 Tokenized Cache 位于 `data/sft/tokenized/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/`，约 30GB；数据构建与 Cache 后台任务退出码均为 0。Cache Manifest SHA256 为 `896f2fc596d75f23bd43c8146f3f69e09a2b38ded3854fc4c1bb40145a96e93f`，长度统计 SHA256 为 `ca31031b6f63d405c15fc4a5b87fcf6d2313fda248adc5af1692d5e6eddd5d8b`。
- 总 Token 长度 P50/P90/P95/P99/P99.9/Max 为 145/295/304/336/445/944；Train/Valid 中分别有 1,580/126 条超过 512，但 Assistant 目标截断数均为 0。使用 `datasets.load_from_disk` 重载后，Train/Validation 行数、列结构和首/中/末样本固定长度均通过核验；原始 Train 前 10,000 条事件结构也无异常。
- 本实验只完成数据、词表、Cache 和训练入口；尚未运行 GPU smoke、正式 SFT 或生成式评测，不记录 checkpoint、Loss 或召回指标。下一步先运行 20-step GPU smoke，通过后再启动一个四卡三轮任务。

## EXP-20260807-01 GNPR 三轮 SFT 与固定 10,000 条 Validation 无约束评测

### 目标、假设与数据版本

- 目标：完成 GNPR 地图检索适配版的三轮 Qwen3-0.6B 全参数 SFT，并在与 TIGER 完全相同的固定 10,000 条 Validation 业务键上比较 epoch 1/2/3 checkpoint。
- 假设：在数据、历史长度、全局 batch、生成 Beam 和评测样本均固定时，训练轮次增加应降低 Validation Loss 并提高 HR/NDCG；论文及作者发布实现没有提供 Trie 或地理约束解码，因此本实验直接生成条件长度 GNPR identifier，不用约束补齐非法候选。
- SFT 数据为 `data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/`，Train/Valid/Test 为 7,586,410/597,421/606,682；训练使用同名 packed Cache，Train/Validation 为 2,758,797/201,947。固定评测集从该 Validation 按 V1/TIGER 参考样本的 `order_id + searchid` 精确对齐，业务键 SHA256 为 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`，目标 POI 错配数为 0。
- identifier 固定为 `512×3 / epoch 20 + conditional d0-d222`，映射覆盖 2,337,178 条 POI，SHA256 为 `3cee68db4f1360b1103452e0e9d1c5e49edf5f11bf70e38b514c1037911abde2`；Tokenizer 词表为 153,474，三个 checkpoint 的 `tokenizer.json` SHA256 均为 `3556999bc37575b8886d6c2061db837678171cbb6002a69dc62aa5fb21b201f0`。

### 代码状态、配置与环境

- 仓库基线提交为 `e4515195097dcb10aac0f3281ee7ddc0e482886c`；训练和评测运行于包含目录整理、GNPR 评测适配器及用户既有修改的未提交工作树，未创建 commit。
- 训练配置为 `configs/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1.yaml`：Qwen3-0.6B、全参数 BF16、cutoff 512、每卡 batch 16、梯度累积 8、四卡全局 batch 512、学习率 `5e-5`、3 epoch、按 epoch 保存。
- 训练使用四卡 A100，完成 3 epoch、16,167 step，`train_loss=0.5693`、运行时间 17:30:03.58。Trainer 已完整保存最终模型和三个 checkpoint；平台随后执行旧提交的包装脚本时在第 87 行遇到不匹配双引号并以退出码 2 标记任务失败，该错误发生在 Trainer 完成、权重和指标落盘之后，不影响 checkpoint。三个 checkpoint 均经文件、epoch、Tokenizer 和实际 CUDA 生成核验。
- 正式评测使用本地 RTX A6000 48GB、Python 3.10.20、PyTorch 2.9.1+cu128、Transformers 4.52.4；batch 32、Beam=10、返回 10 个候选、`max_new_tokens=7`。三段单例和四段碰撞 identifier 均按严格结构解析；不在冻结映射中的 identifier 保留原 Beam 排名并计为 miss，不使用 Trie、地理剪枝或候选补位。

### 命令

~~~bash
python scripts/gnpr/evaluate_retrieval.py \
  --valid-file data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoints \
    outputs/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/checkpoint-5389 \
    outputs/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/checkpoint-10778 \
    outputs/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/checkpoint-16167 \
  --expected-checkpoint-steps 5389 10778 16167 \
  --expected-checkpoint-epochs 1 2 3 \
  --tokenizer models/Qwen3-0.6B-GNPR-Vocab-v1 \
  --identifier-dir outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-512x3/gnpr_ids/epoch_20 \
  --output-dir outputs/eval/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3 \
  --num-beams 10 \
  --per-device-eval-batch-size 32 \
  --chunk-size 1000 \
  --cutoff-len 512 \
  --skip-data-hash
~~~

- 正式推理前已使用同一命令的 `--preflight-only` 完整核验源 Validation SHA256、固定子集、identifier、Tokenizer、三个 checkpoint 和前 100 条 Prompt；正式命令的 `--skip-data-hash` 只避免重复扫描 725MB 源文件，不跳过固定子集 SHA256 或 checkpoint 权重 SHA256。

### 核心指标

| Epoch / checkpoint | Validation Loss | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Valid ID Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / `checkpoint-5389` | 0.547218 | 40.73% | 63.42% | 68.88% | 73.42% | 40.73% | 54.1633% | 56.4180% | 57.9210% | 74.799% |
| 2 / `checkpoint-10778` | 0.421038 | 47.71% | 70.90% | 77.30% | 81.52% | 47.71% | 61.4745% | 64.1178% | 65.5149% | 79.342% |
| 3 / `checkpoint-16167` | 0.406466 | **49.66%** | **72.61%** | **78.62%** | **83.07%** | **49.66%** | **63.2718%** | **65.7607%** | **67.2390%** | 78.808% |

- 三轮均完成 10,000 条、100,000 个原始 Beam 候选；实际 batch 均为 32，未触发 OOM 回退。纯推理耗时为 1204.74/1190.01/1189.53 秒，峰值显存约 22.78GiB。
- epoch 1/2/3 的非法候选率为 25.201%/20.658%/21.192%；主要错误均为生成的合法结构 identifier 不在冻结目录映射中，少量错误来自位置 Token、EOS、结构或长度。epoch 3 的 Valid ID Rate 略低于 epoch 2，但所有 HR/NDCG 均继续提升，因此按预设的 NDCG@10、HR@10、HR@1、Validation Loss 顺序选择 epoch 3。
- 汇总 JSON/CSV SHA256 分别为 `e0e62cdacb4a59765b739e790d0b718adde979285c422b7c5e4eee6a44f9272c` 和 `82189cac59d7bec2445445b566b0b8fbf1eac402506c6f06ef76d7ec2d709425`；正式评测进程退出码为 0。

### 产物、结论与下一步

- 训练目录为 `outputs/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/`；评测目录为 `outputs/eval/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/`，包含固定子集及 manifest、预检结果、三个正式 run 的原子进度和结果、错误 Case，以及 `valid_checkpoint_results.json/csv`。
- 结论：GNPR 地图检索适配版的三轮 SFT 与固定子集无约束评测闭环完成，冻结 `checkpoint-16167` 为 Validation 最优 checkpoint。其 HR@1/HR@10/NDCG@10 比同口径 TIGER epoch 3 低 2.21/4.09/3.1800 个百分点，但 Valid ID Rate 高 4.703 个百分点；该结果用于完整方法链路比较，不能把差异单独归因于 SID 构建。
- 当前不运行完整 Validation 或 Test。GNPR 复现闭环完成后，下一最小步骤回到 Query-Augmented Relational SID 创新线，固定既有 10,000 条 Query 的 BGE-M3 embedding 和行映射。

## EXP-20260807-03 TIGER/GNPR SID 逐层 Teacher-Forcing 诊断

### 目标、假设与数据版本

- 目标：用一次前向即可得到的 gold-prefix teacher-forcing 指标，定位 GNPR epoch 3 相对 TIGER epoch 3 的生成差距首先出现在哪一层，区分 SID 前缀组织、整体 SFT 和无约束解码三类原因；本实验不重训、不改变 SID，也不运行 Beam。
- 判据：若训练资源或整体 SFT 是主因，GNPR 各层应普遍更差；若 SID 层级组织是主因，差距应集中在特定前缀层且给定正确前缀后的后层可恢复；若无约束解码是主因，teacher-forcing 完整 identifier 差距应明显小于自由生成 HR@1 差距。
- 两种方法均复用各自 EXP-20260804-01、EXP-20260807-01 已冻结的同一组 10,000 条 Validation 业务键。两份方法数据的 `sample_id` 顺序 SHA256 均为 `a81c7d91f82a3e6df89002b8eb0c320625dc601e33518e1d36e869c4487c25c1`；TIGER/GNPR JSONL SHA256 分别为 `b06f4bd5cd62e12a3157e86394d1a82ce752872c34fe4b81e7c6fbe8f5282fa1`、`8a2a39c7419686e901f0e5328e6f39e09f887f136d72be7118d2e12e92381052`，差异只包含方法目标 identifier 等方法字段。
- checkpoint 分别为 TIGER `checkpoint-16713` 和 GNPR `checkpoint-16167`，均为 epoch 3；模型权重 SHA256 分别为 `d6b228a936a2f6824708b2246618494a2a4cd578e686aeced2d73d09d194307e`、`a5fcac4d4ab46404d7eadcf55639948efc5c559a70cea2802a792eeb51323a10`。

### 代码状态、配置与环境

- 仓库基线提交为 `e4515195097dcb10aac0f3281ee7ddc0e482886c`；运行使用包含目录整理、`src/poi_gr/sft/teacher_forcing.py`、`scripts/sft/diagnose_sid_teacher_forcing.py` 和合成测试的未提交工作树，未创建 commit。
- 每条样本使用与 SFT 完全一致的 Prompt 和 gold target。模型一次前向后只抽取目标位置 logits，统计每层 Top-1、Top-10、NLL，以及前三层累计全位置正确率和完整 identifier 全位置正确率；后层均条件于正确 gold 前缀，因此不受前层自由生成错误传播、Beam、Trie 或非法 ID 映射影响。
- TIGER 目标固定为三层 SID 加 collision token；GNPR 单例为三层 SID，2,926 条碰撞样本额外包含 dedup token。结构起止 token 与 EOS 同时核验，但不纳入 semantic prefix。batch 32、cutoff 512，两次正式运行均未触发 OOM 回退。
- 环境为本地 NVIDIA RTX A6000 48GB、Python 3.10.20、PyTorch 2.9.1+cu128、Transformers 4.52.4；TIGER/GNPR 纯前向耗时分别为 119.11/109.70 秒，峰值显存约 6.24/6.09GiB。

### 命令

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/sft/diagnose_sid_teacher_forcing.py \
  --method tiger \
  --data-file outputs/eval/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/validation_subset_10000.jsonl \
  --checkpoint outputs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-16713 \
  --tokenizer models/Qwen3-0.6B-TIGER-Vocab-v1 \
  --output-dir outputs/eval/sid_teacher_forcing_fixed10k_v1/tiger_e3 \
  --expected-epoch 3 --batch-size 32 --checkpoint-rows 1000

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/sft/diagnose_sid_teacher_forcing.py \
  --method gnpr \
  --data-file outputs/eval/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/validation_subset_10000.jsonl \
  --checkpoint outputs/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/checkpoint-16167 \
  --tokenizer models/Qwen3-0.6B-GNPR-Vocab-v1 \
  --output-dir outputs/eval/sid_teacher_forcing_fixed10k_v1/gnpr_e3 \
  --expected-epoch 3 --batch-size 32 --checkpoint-rows 1000
~~~

### 核心指标

| Gold-prefix 目标位置 | TIGER Top-1 | GNPR Top-1 | GNPR-TIGER | TIGER Top-10 | GNPR Top-10 | GNPR-TIGER | TIGER NLL | GNPR NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SID 第 1 层 | 75.06% | 53.69% | **-21.37pp** | 96.29% | 89.14% | **-7.15pp** | 0.9027 | 1.7925 |
| SID 第 2 层 | 81.04% | 83.43% | +2.39pp | 96.55% | 94.01% | -2.54pp | 0.7386 | 0.7773 |
| SID 第 3 层 | 85.70% | 92.02% | +6.32pp | 96.85% | 96.66% | -0.19pp | 0.5986 | 0.4108 |

| Gold-prefix 全位置同时正确 | TIGER Top-1 | GNPR Top-1 | GNPR-TIGER | TIGER Top-10 | GNPR Top-10 | GNPR-TIGER |
|---|---:|---:|---:|---:|---:|---:|
| 前 1 层 | 75.06% | 53.69% | -21.37pp | 96.29% | 89.14% | -7.15pp |
| 前 2 层 | 61.75% | 49.31% | -12.44pp | 93.79% | 87.15% | -6.64pp |
| 前 3 层 | 53.98% | 48.50% | -5.48pp | 92.15% | 86.64% | -5.51pp |
| 完整 identifier | 50.76% | 48.40% | **-2.36pp** | 92.04% | 86.63% | **-5.41pp** |

- GNPR 第一层在候选数更少（512 对 TIGER 的 1024）的情况下，Top-1 仍低 21.37 个百分点、NLL 高 0.8898；但给定正确第一层后，第二、三层 Top-1 分别高 2.39、6.32 个百分点。差距不是各层普遍退化，而是预测难度被集中到最先生成、无法由后缀纠正的根前缀。
- GNPR 单例/碰撞组的完整 identifier Top-1 全位置正确率为 50.35%/43.68%，说明 conditional dedup 仍带来次要难度；但结构 token 基本为 100%，主要瓶颈仍在所有 10,000 条样本共有的第一层。
- teacher-forcing 完整 identifier 的方法间 Top-1 差距为 2.36 个百分点，与无约束 Beam 自由生成 HR@1 差距 2.21 个百分点只差 0.15 个百分点。该指标不是 Beam 排名指标，数值不要求完全相等，但差距高度接近，未显示解码过程额外放大 GNPR 劣势。

### 产物、结论与下一步

- 正式结果位于 `outputs/eval/sid_teacher_forcing_fixed10k_v1/{tiger_e3,gnpr_e3}/result.json`，结果 SHA256 分别为 `2424c7b7a1bdbcb1bfc9ca24b55485e43770a2c99067daaf968de6cc84a022a6`、`02efc13335b42a3b6e0bde87197a523cbedaae6078a4eeb6338d447872df88f9`；两次进程退出码均为 0。2 条样本 GPU smoke 先行通过，正式结果各覆盖 10,000 条。
- 结论：当前 GNPR 相对 TIGER 的劣势主要来自 SID 第一层不够容易由 Query/历史预测，而不是 A100 与 6000D 的训练卡差异、整体 SFT 未学会或无约束解码。该定位与既有静态审计一致：GNPR-512 的第一、二层类别纯度 44.63%/62.02%，低于 TIGER-1024 的 69.28%/77.27%。GNPR 的 base SID 唯一率更高和后层更易预测，不能抵消根前缀弱语义对自回归检索造成的损失。
- 本实验能定位层级，但不能单独区分根因来自 content-geo 特征融合、512 容量选择还是 Diversity Loss。后续 Query-Augmented Relational SID 实验应把第一层 Query 可预测性、前缀语义纯度与最终唯一率并列作为选择标准；当前不因此重训既有 GNPR，也不提前启动 RQ-KMeans。
