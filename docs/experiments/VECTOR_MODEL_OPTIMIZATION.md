# 向量模型优化与生成式检索下游实验

本文件记录“替换或优化 POI/Query 向量模型后，沿固定 TIGER 下游链路验证”的方法与正式实验。共享模型接入和纯向量召回仍记录在 [V1 实验](V1.md)，Train Query 聚合的 E1—E4 仍记录在 [Embedding 优化实验](EMBEDDING_OPTIMIZATION.md)；本文件只负责把候选向量继续接到统一 RQ-VAE、碰撞去重和最终 SFT，避免把连续向量、静态 SID 与生成式指标混成同一结论。

## 1. 研究目标与决策口径

当前问题不是单独追求更高的向量 Hit@K 或 SID 唯一率，而是判断地图检索训练得到的向量空间能否让生成模型更容易产生正确 POI identifier。实验按以下三级证据推进：

1. 固定 SFT Validation 10,000 条上的全量精确向量召回，用于筛掉明显弱于 BGE-M3 的编码器；
2. 固定 TIGER RQ-VAE `1024×1024×1024` 的全量 SID 唯一率、碰撞、码本利用率和前缀类别纯度，用于筛掉量化后退化的编码器；
3. 完全配对的三轮 SFT 与固定 10,000 条 Beam=10 生成式评测，作为最终是否提升的唯一主结论。

前两级只决定是否值得投入一次昂贵 SFT。即使 SID 唯一率更高，也不能宣称生成式检索已经超过 TIGER。

## 2. 冻结的公平比较协议

### 2.1 只替换向量输入

- POI 候选库固定为北京 2,337,178 条，POI 行顺序和 `poi_ids.jsonl` 与 TIGER 完全一致；
- 当前线上 MMBERT 仍读取冻结的名称、地址、别名 `text`，不额外加入历史 Query、拼音或线上服务的其他字段；
- Query 精确召回使用各模型自己的编码器，但都是原始 Query、无 Instruction、L2 normalize 和全量 `IndexFlatIP`；
- E4 是 BGE-M3 上叠加 Train-only Query 类别残差的 1024 维向量，不是独立原始编码器，因此与 MMBERT 的比较用于方法选型，不能解释成纯模型架构对照。

### 2.2 固定 TIGER 量化器与去重

三组输入统一使用：

- RQ-VAE：`hidden_dim=512`、`latent_dim=256`、三层 `1024³`；
- KMeans 初始化：seed 42、固定 500,000 行、20 iterations、Faiss GPU；
- 训练：20 epoch、batch 4096、Adam、学习率 `3e-4`、固定 1% Validation、不早停；
- SID：epoch 20 checkpoint 全量导出三层 `[S1,S2,S3]`；
- 去重：同一三层桶内按 `poi_id` 字典序分配从 0 开始的 `C`，单例也追加 `C0`，形成固定四层唯一 identifier。

线上模型输出维度是训练得到的 128，因此只把 RQ-VAE `input_dim` 从 1024 改为 128；隐藏层、潜空间、码本、训练划分和优化协议均不变。该设计比较相同下游容量下的向量空间，不把“改编码器”和“改量化器”同时引入。

### 2.3 后续 SFT 冻结项

若候选通过静态门槛，后续数据必须从 TIGER SFT 做逐行 identifier-only remap，保持 Train/Valid/Test 的 `7,586,410/597,421/606,682`、sample key、Query、请求 GID、历史窗口和样本顺序不变。Tokenizer 只新增本候选实际使用的 `S1/S2/S3/C` 原子 Token；训练仍使用 Qwen3-0.6B、4×6000D、3 epoch 和固定 10,000 条 Validation。最终主表至少报告无约束 Beam=10 的 HR@1/3/5/10、MRR@10、NDCG@10、Valid ID Rate，并补充三层 Bucket 与 gold-prefix 诊断。

## 3. 当前候选与对照

| 输入 | 维度 | 是否使用 Train Query 聚合 | 量化器 | 四层规则 |
|---|---:|---|---|---|
| TIGER BGE-M3 | 1024 | 否 | TIGER RQ-VAE `1024³` | `[S1,S2,S3,C]` |
| E4 BGE-M3 | 1024 | 是，`α=0.30, β=0.85` | 同上 | 同上 |
| MMBERT Recall | 128 | 否；模型训练本身使用 Query–POI 正负样本 | 同上，仅 `input_dim=128` | 同上 |

固定 10k 连续向量精确召回：

| 输入 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TIGER BGE-M3 | 11.99% | 20.69% | 25.25% | 32.39% | 40.25% | 17.7627% | 21.2113% |
| MMBERT Recall 128 | 17.26% | 27.90% | 33.21% | 41.43% | 51.00% | 24.1966% | 28.2718% |
| E4 BGE-M3 | **20.02%** | **34.92%** | **43.76%** | **56.37%** | **66.78%** | **30.2279%** | **36.3976%** |

MMBERT 是当前最强的无显式 Train Query 聚合原始编码器候选；E4 的连续向量结果仍更高，但既有 `2×2` 已证明这类增益不一定转化为最终 SFT。

## 4. 正式实验记录

### EXP-20260820-04：MMBERT Recall 128 → TIGER RQ-VAE → collision token

#### 目标与状态

- 日期：2026-08-20；状态：已完成。
- 目标：在完全固定 TIGER 量化和去重协议下，判断 `EXP-20260820-03` 的线上 MMBERT 128 维向量是否仍保留静态结构收益，并决定是否值得进入一次配对 SFT。
- 预注册判断：若全量三层 SID 没有坍塌，且唯一率、碰撞 POI 和码本利用率整体优于 TIGER BGE-M3，则进入 SFT；最终是否提升仍只看后续生成式指标。

#### 数据、配置与环境

- 输入向量：`outputs/embeddings/beijing_poi_mmbert_recall_128/embeddings.npy`，shape `[2337178,128]`、float16、L2 normalize；向量 SHA256 为 `041c4e2b071e5f9a2c45442b8a6bba2e3b1e4edd34873305119d208dd54a8ffe`。
- POI ID：2,337,178 行、全局唯一，SHA256 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`。
- 配置：`configs/sid/rqvae_tiger_mmbert_recall_128_1024x3.yaml`，SHA256 `42c4aa18fa499416faee05b2fcc69ac175a75595a5b8f4a6738a43c6030389f3`。
- 固定划分：KMeans 500,000 行索引 SHA256 `0f7f22d7a7543965d581bcc9098a833533dbbdbb701b867f54c76d0bf66d6eb7`；Validation 23,372 行索引 SHA256 `409e476abf82cd609447c6c9a6f437d96ecd897a40c5e610a33a3bc996041249`，与 BGE/E4 相同。
- 环境：`poi-gr`，Python 3.10.20、PyTorch 2.9.1+cu128、NumPy 1.26.4、单卡 NVIDIA RTX A6000。

正式核心命令：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_tiger_mmbert_recall_128_1024x3.yaml \
  --experiment TIGER-MMBERT-RECALL-128-1024x3 \
  --no-progress

/ofs/map_search/hudan/envs/poi-gr/bin/python scripts/sid/export_rqvae.py \
  --run-dir outputs/sid/tiger/mmbert_recall_128/TIGER-MMBERT-RECALL-128-1024x3 \
  --checkpoint checkpoint_epoch_20.pt \
  --output-dir outputs/sid/tiger/mmbert_recall_128/TIGER-MMBERT-RECALL-128-1024x3/evaluations/epoch_20 \
  --device cuda --batch-size 4096

/ofs/map_search/hudan/envs/poi-gr/bin/python scripts/tiger/build_identifiers.py \
  --sid-manifest outputs/sid/tiger/mmbert_recall_128/TIGER-MMBERT-RECALL-128-1024x3/evaluations/epoch_20/sid_manifest.json \
  --output-dir outputs/sid/tiger/mmbert_recall_128/TIGER-MMBERT-RECALL-128-1024x3/tiger_ids/epoch_20
```

训练在进入正式全量前完成 4,096 行、`16³`、1 epoch 的训练→导出→去重 smoke，四层 ID 唯一率 100%；smoke 只用于贯通，不进入正式指标。正式训练完成 20 epoch，训练耗时 124.89 秒，epoch 20 Validation reconstruction cosine 为 0.760705。不同输入维度的逐维 MSE 和重构 cosine 不作为编码器优劣主指标。

#### 全量 SID 结果

| 输入 + 同一 RQ-VAE | 三层唯一率 | 碰撞 POI | Excess collision | 最大桶 | S1/S2/S3 利用率 |
|---|---:|---:|---:|---:|---:|
| TIGER BGE-M3 | 71.6766% | 40.7894% | 28.3234% | 306 | 54.7852% / 100% / 100% |
| E4 BGE-M3 | 74.3539% | 37.4194% | 25.6461% | 334 | 73.3398% / 100% / 100% |
| MMBERT Recall 128 | **83.3988%** | **25.5070%** | **16.6012%** | **154** | **100% / 100% / 100%** |

MMBERT 相对 TIGER BGE-M3 的三层唯一率提高 `11.7223pp`，碰撞 POI 降低 `15.2824pp`，excess collision 降低 `11.7223pp`；相对 E4 分别改善 `9.0449pp`、`11.9124pp` 和 `9.0449pp`。三层完整 SID 数为 1,949,179，碰撞桶 208,144 个，碰撞 POI 596,143 个。

码本熵与类别前缀：

| 输入 | S1/S2/S3 normalized entropy | depth-1/2/3 category micro-purity |
|---|---:|---:|
| TIGER BGE-M3 | 0.85776 / 0.96925 / 0.97290 | 69.2751% / 77.2651% / 96.2830% |
| E4 BGE-M3 | 0.90611 / 0.96927 / 0.97138 | 64.1303% / 74.8755% / 95.9466% |
| MMBERT Recall 128 | **0.97128 / 0.98330 / 0.97976** | 52.1307% / 70.0336% / **97.0495%** |

MMBERT 明显提高码本均衡性与最终三层类别纯度，但 depth-1/2 类别纯度相对 TIGER 分别下降 `17.1444pp/7.2315pp`。这说明线上监督空间形成了更均衡、细粒度的分区，却不再主要按现有 402 类组织粗层；它可能更贴近 Query–POI 相关性，也可能增加生成模型首层预测难度，必须由 SFT 和逐层诊断判定。

#### collision token 与产物

- 在三层桶内按 `poi_id` 字典序追加 `C0–C153`，所需碰撞 Token 从 TIGER 的 306 个降到 154 个；最终 `[S1,S2,S3,C]` shape 为 `[2337178,4]`，2,337,178 条均唯一。
- SID 目录：`outputs/sid/tiger/mmbert_recall_128/TIGER-MMBERT-RECALL-128-1024x3/`；checkpoint SHA256 `f2a8b178f28faefa854db3e88c52e0582f2e2d768452b1cf5d932edc7b39707b`。
- 三层 `sid_codes.npy` SHA256 `895ec26b162cdab2f4cda36083136e401e49ddea4373146d02596c9fa0b376a1`；SID metrics SHA256 `bab7c45ba92719d0aed4a50caebe0978d6a5cbce3977e7082d5e45bbdecce2a2`。
- 四层 `tiger_ids.npy` / mapping / collision codes SHA256 分别为 `4d4cd1109e496d5f220b5b656dd867b943a50377f314e9cc582bf29f4c1ffe7e` / `755dd3c914136991432e6adcb0dbb726c48f6a3d08d9485ced3d946c82c3da3e` / `2b7b927a30af7eb5986266b5f6cdcf7baf418f191b1b1cda49a444d070d93c9b`；identifier metrics SHA256 `3878f38ad09b4a3eb955b6ddc99f2f8d07997961ee167ccd1b0ac6563cb5bdaa`。
- 运行控制：`outputs/run_control/EXP-20260820-04_mmbert_rqvae_sid/`；完整流水线日志 SHA256 `11d6934c9052adae1d67d836f5ba8be9f591b068b3942e3104e68014483df693`，最终退出码为 0。

#### 结论与下一步

本候选通过静态 SID 门槛，而且改善幅度大于 E4+RQ-VAE：更高的唯一率、全层 100% 码本利用率、更少的碰撞 POI、更小的最大桶和更小的碰撞词表同时成立。因此值得投入下一次配对 SFT。

当前不能写成“超过 TIGER”：前两层类别纯度明显下降，且既有 E4/RQ-KMeans 已证明静态唯一率与连续向量收益可能在完整四层生成时反转。下一实验只构建 MMBERT 四层 identifier 的配对 SFT 数据和四卡三轮入口；训练后与 TIGER epoch 3 在同一固定 10k 上比较，若 HR@1/HR@10/NDCG@10 没有整体超过 TIGER，则把本结果定义为“量化结构改善但不可生成”，继续优化向量模型时必须加入逐层可预测性约束，而不是继续追求唯一率。

### EXP-20260820-05：MMBERT 四层 identifier 配对 SFT 输入冻结

#### 目标与状态

- 日期：2026-08-20—21；状态：已完成训练输入冻结；4×A100 首次正式启动因 CUDA OOM 在 step 1 后失败，待按修订显存配置重跑。
- 目标：只把 TIGER 的四层 identifier 替换为 `EXP-20260820-04` 的 MMBERT `[S1,S2,S3,C]`，保持样本、Query、请求 GID、历史窗口、时间划分和训练超参数不变，为一次昂贵但可归因的四卡三轮 SFT 做准备；平台可使用 4×RTX PRO 6000D 或 4×A100，但两者不得混用同一输出目录。
- 结论口径：本实验只证明数据、Tokenizer、Cache 和训练入口完整可运行，不产生生成式检索提升结论；是否超过 TIGER 仍只看后续固定 10,000 条 Validation、Beam=10 指标。

#### 配对 Messages

- identifier 输入固定为 `outputs/sid/tiger/mmbert_recall_128/TIGER-MMBERT-RECALL-128-1024x3/tiger_ids/epoch_20/`，mapping SHA256 为 `755dd3c914136991432e6adcb0dbb726c48f6a3d08d9485ced3d946c82c3da3e`，容量为 `[1024,1024,1024,154]`。
- 正式数据位于 `data/sft/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1/`。Train/Valid/Test 精确为 `7,586,410/597,421/606,682`，共 8,790,513 条；历史覆盖率 72.0157%，平均历史长度 4.9153，空历史保留。
- 三个 split 完整扫描 32 个原始分片，输入文件清单、逐文件行数与哈希、处理规则、Prompt 协议、时间切分、用户桶规则和 `stats.json` 均与原 TIGER 数据一致。Train/Valid/Test 各取同序前 100 条逐字段核验，`sample_id/order_id/searchid/target_poi_id/Query/GID/HISTORY` 全部相同，唯一变化是历史和目标 identifier。
- Messages manifest SHA256 为 `ee112d40298eb8f179ddf06e11056eea2ba21e4c9b423d91699eba02c7ab259f`；Train/Valid/Test SHA256 分别为 `33c1d0dd6ed4d259510b4ac23ef6f095d157dfce7ce2585a00b434a3520fc1eb`、`23fbc8ed750d0360d46f0e330b509b2c1c9ff6875543898be8b4d32418c64e1f`、`39da2cdc0f03571f057f1b6eaf55ce53c1d5353a8ef11444eef25c66accde960`。

#### 扩词表与 cutoff 1024 Cache

- `special_tokens.json` 共 5,274 项普通原子 Token：16 个结构 Token、32 个 Geohash 字符 Token、2,000 个用户桶 Token、三层各 1,024 个 SID Token 和 154 个 collision Token；SHA256 为 `6b849d8dc3fc953d87faeb56c37184c4a2a7aacf1a9d8d2e24652c612dbf072a`。
- 扩词表模型位于 `models/Qwen3-0.6B-TIGER-MMBERT-Recall-128-1024x3-Vocab-v1/`。Tokenizer 从 151,669 扩到 156,943，5,274 项均通过单 Token 编码、唯一 ID、普通 Token 和 round-trip 校验；mapping SHA256 为 `451a0c1c5b0eaf8a5f2a2f773cc4bbd7d532ed659123d3543f444544e35d04d8`，扩展 Tokenizer SHA256 为 `213b29e466b20837ae03a3df46978b42eb6b560985bbc54054f26c6fa7f97a39`。
- 新的历史 SFT 统一采用完整 Source+Target `cutoff_len=1024`。Train 7,586,410 条和 Valid 597,421 条全量预检的输入最大 945、目标恒为 8、完整序列最大 953；超过 1024 的样本数和目标截断数均为 0，因此没有删除历史或裁剪 CURRENT/Target。
- 正式 packed cache 位于 `data/sft/tokenized/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1/`，约 28 GiB；packed Train/Validation 为 `1,296,883/96,949`，10,000/2,000 原始行 smoke 为 `1,711/323`。Cache manifest / length stats SHA256 为 `39e9f0e0e6ae2cf525d1804654ebb4b2da64d32800622e6756c41f05b8536761` / `182ce597994dcda43ba01a262daa200b2642fad46e6e0a9bd33368f6291a93f1`。Test 只保留 JSONL，没有进入预检或 Cache。

#### 4×6000D / 4×A100 训练入口与验证

- 训练配置为 `configs/sft/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1.yaml`；6000D/A100 启动器分别为 `launchers/run_train_tiger_mmbert_recall_128_1024x3_sft_4x6000d_3epoch.sh` 和 `launchers/run_train_tiger_mmbert_recall_128_1024x3_sft_4a100_3epoch.sh`。
- 协议固定为 Qwen3-0.6B 全参数 BF16、`qwen3_nothink`、packing、3 epoch、每轮保存和验证、保留 3 个 checkpoint，不启用 gradient checkpointing。首次 A100 启动使用每卡 batch 16、gradient accumulation 8；OOM 后将训练与验证单卡 batch 降为 8、gradient accumulation 增为 16，4 卡 global batch 仍为 512，模型、数据、学习率和优化器均不变。
- 两个启动器分别只接受恰好 4 张 RTX PRO 6000D 或恰好 4 张 A100；都使用 `master_port=0` 避免固定端口冲突。短 TMPDIR 分别为 `outputs/tmp/tm128d` 和 `outputs/tmp/tm128a`，并沿用已验证的平台环境、OFS 挂载和 `poi-gr` 激活协议。输出目录分别以 `_gpu4_6000d_e3` 和 `_gpu4_a100_e3` 结尾，防止平台间覆盖。
- 两个入口的初始 dry-run 均已解析模型、数据注册、Cache、1024 cutoff、global batch 512 和自动端口，未启动 torchrun；正式 cache 重新加载得到 Train/Validation `1,296,883/96,949`，首/中/末 packed 样本均为 1,024 长且含有效 labels。显存参数修订后再次 dry-run 得到 `8 × 16 × 4 = 512`，两个启动器的 `bash -n` 与 `git diff --check` 均通过。
- 当前配置 / 6000D launcher / A100 launcher / 修订后 dry-run resolved config SHA256 分别为 `acb5e4460e990b47ca1d9679ec2dcc5442e749f513f3a82d4a4809cb7d0ad8ba`、`a79a9298059a7e68e7fcf9584e02ba96b5991c66efdab4bd62dc0a02a5b25b5c`、`5cd50b5296696f1c0d572ca1a80bbde2d3a8dbed1c372f55c6ee18e1d9a2be3e`、`7ca17dd93bf1328f23d4cd208d34dee8f36183e70c74082e56f70d5c90a26443`。
- 数据与 Cache 的后台运行控制位于 `outputs/run_control/EXP-20260820-05_mmbert_sft_prep/`，两个正式阶段退出码均为 0；Messages/Cache 日志 SHA256 为 `099aec87f9dcdb24817a13c6d7ca37a3190e383ee38ad0757cfd92322058acd1` / `11dd1b426a35ac4490a06c9363a1fb81496a38f0eec0214933edb847c5119d6c`。

#### A100 首次正式启动失败与修订

- 命令：`bash launchers/run_train_tiger_mmbert_recall_128_1024x3_sft_4a100_3epoch.sh`；环境为单机 4×A100 80GB、全参数 BF16、packing、cutoff 1024。
- 任务完成 `1/7599` step 后失败；唯一 Trainer 记录为 loss 20.6087、已见 524,288 token。各卡当时约使用 71.41 GiB、仅余 7.72 GiB，下一次 DDP forward 还需申请 9.58 GiB，四个 rank 均报 `torch.OutOfMemoryError`。未生成任何 checkpoint 或模型权重，因此该次输出不可评测，也不能写入生成指标。
- 失败产物位于 `outputs/sft/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1_gpu4_a100_e3/`，保留 `train_console.log`、`trainer_log.jsonl`、TensorBoard event 和当次 `resolved_config.json` 作为诊断证据。socket、NCCL abort 与 `PYTORCH_CUDA_ALLOC_CONF` 提示均为告警或 OOM 后继现象，不是根因。
- 修订只降低 micro-batch 并等比例增加梯度累积，保持 global batch 512；若重跑仍 OOM，再单独评估启用 gradient checkpointing，不在本次提前改变计算图。

#### 结论与下一步

MMBERT 四层 SID 的数据与 cache 已冻结，首次 4×A100 正式训练暴露的是 micro-batch 显存问题，不是数据或分布式初始化问题。下一步使用修订后的 4×6000D 或 4×A100 launcher 从头重跑；训练完成后先验证三个 epoch checkpoint 完整性，再按固定普通 Validation 10k 做三轮无约束 Beam=10，并对 epoch 3 补充合法路径和逐层 gold-prefix 诊断。当前仍不得写成“MMBERT 超过 TIGER”。

## 5. 后续向量模型优化规则

- 每个新 checkpoint 先复用固定 10k 精确向量评测，再复用本文件的 TIGER RQ-VAE 配置；不因输入维度变化调整码本容量。
- 只有连续向量和全量 SID 均通过门槛才投入 SFT，避免为每个 checkpoint 重建约 879 万条数据和三轮训练。
- 首轮只比较 checkpoint；POI 文本字段扩展、向量训练目标、RQ-VAE 结构和 SID 容量分别做后续单变量消融。
- 若 SFT 失败，优先诊断 S1/S2/S3 gold-prefix、三层 Bucket HR 和热点碰撞请求占比，再决定是否训练“层级可预测性”辅助目标；不从静态纯度或唯一率直接猜原因。
- 任何最终结论均以生成式 SFT 指标为主，向量 Hit@K、SID 唯一率和码本利用率只作为机制解释。

### `EXP-20260822-01`：MMBERT 四层 identifier 三轮全参数 SFT

#### 目标、数据与代码状态

- 目标与假设：在 `EXP-20260820-05` 已冻结的同行同序输入上，从头完成修订显存参数后的三轮训练，检验 MMBERT Recall 128 带来的连续向量和静态 SID 收益是否值得进入生成评测。本实验只完成训练，不提前根据 Validation Loss 宣称超过 TIGER；
- 数据与模型：复用 `data/sft/tokenized/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1/`，packed Train/Validation 为 `1,296,883/96,949`；Tokenizer 为 `models/Qwen3-0.6B-TIGER-MMBERT-Recall-128-1024x3-Vocab-v1/`，四层 identifier 容量为 `[1024,1024,1024,154]`。Train+Valid 已在输入冻结阶段确认 cutoff 1024 下超长和目标截断均为 0；
- 代码状态：运行与核验对应 Git HEAD `7dd52b3437a9db41834aef638e446010d0df3610` 的非干净工作树；训练 launcher SHA256 为 `5cd50b5296696f1c0d572ca1a80bbde2d3a8dbed1c372f55c6ee18e1d9a2be3e`；
- 配置：Qwen3-0.6B 全参数 BF16、packing、`cutoff_len=1024`、3 epoch、学习率 `5e-5`、4×A100、单卡 batch 8、梯度累积 16，global batch 保持 512；未启用 gradient checkpointing。resolved config SHA256 为 `9146171422521e78d00d1675b13317a803714ed3449d4a53dacc01ed702a852b`；
- 正式命令：

```bash
bash launchers/run_train_tiger_mmbert_recall_128_1024x3_sft_4a100_3epoch.sh
```

#### 训练结果、产物与结论

| Epoch | Checkpoint | Validation Loss |
|---:|---:|---:|
| 1 | `checkpoint-2533` | 0.480174 |
| 2 | `checkpoint-5066` | 0.367225 |
| 3 | `checkpoint-7599` | 0.352294 |

- 状态：三轮完整结束，global step 为 7,599；训练 loss 为 0.596612，累计处理 3,984,064,512 个输入 Token，运行 67,739.83 秒（约 18 小时 49 分），最终模型和三个完整 checkpoint 均已保存。先前 step 1 OOM 失败没有 checkpoint，本次按修订 micro-batch 从头运行，不是断点续训；
- 产物：训练目录为 `outputs/sft/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1_gpu4_a100_e3/`；`all_results.json` SHA256 为 `adb1bb7704f0a45a7ef214c09db43fe4ff7248b9c2c9c9a303dc77c4df8a59f6`，epoch-3 模型 SHA256 为 `e927ebdfc6d385755b87c1015f98cf3e422e7e13dc5b4a4bd58b593612a7d24d`，训练日志 SHA256 为 `147f03eb47f644a958d0c3479600279ce03e8be58bfbfd3bbca231c84331d58d`；
- 核验：epoch 3 已通过真实评测 preflight，2,337,178 条四层 mapping、Tokenizer 容量、checkpoint epoch/step/Validation Loss 均匹配，固定随机 Validation 10k 的目标 POI mismatch 为 0；
- 结论与下一步：训练收敛且三轮 Validation Loss 单调下降，现已具备正式评测条件，但尚无 HR/NDCG。下一步按固定随机 10k 做 epoch 1/2/3 无约束 Beam=10 和 epoch 3 合法路径诊断，并在四个泛化 10k 上只评 epoch 3 无约束结果；只有完整 SID 指标整体超过 TIGER 才冻结 MMBERT 作为新编码器。

### `EXP-20260822-03`：MMBERT 四层 identifier 固定随机与四类泛化评测

#### 目标、数据、代码与协议

- 目标与假设：检验 MMBERT Recall 128 在连续向量精确召回、三层 SID 唯一率和碰撞规模上的静态收益，能否转化为 Query→完整四层 identifier 的生成召回；同时判断收益是否集中在 Train 未见目标，而不是常规已见目标；
- 数据版本：固定随机集精确复用 `outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl` 的 10,000 个 `order_id + searchid`，参考 SHA256 为 `a2e0366d3d8f582e53a5293b687dc08f85b2960d60c7fb44d2fd02168063a944`，目标 POI mismatch 为 0。泛化集复用 `generalization_validation_suite_10k_v1` 的四个互斥 Validation 子集，suite manifest SHA256 为 `d274413e82dd175f28e0510146c2d1a46213f6f1b553d19094a1efa6ac601539`，每组 10,000 条；
- checkpoint：固定随机无约束评测 `checkpoint-2533/5066/7599`，对应 epoch 1/2/3；合法路径与四类泛化只评 epoch 3 `checkpoint-7599`，模型 SHA256 为 `e927ebdfc6d385755b87c1015f98cf3e422e7e13dc5b4a4bd58b593612a7d24d`；
- 代码状态：运行对应 Git HEAD `7dd52b3437a9db41834aef638e446010d0df3610` 的非干净工作树；四卡 launcher SHA256 为 `b18f3cd7bf0b6d097b749649f6d0c109d7ba705ae97804871e92fbda8ac926f4`，TIGER 协议评测器 SHA256 为 `6cd1f0d32676c4e9aa3d73bb391c5e3cd66038b179110a0edb57d4be2663299e`；
- 配置与环境：4×RTX 6000D/RTX PRO 6000，MMBERT 固定集和泛化集各占一条单卡 lane；Qwen3-0.6B、Beam=10、返回 10 个候选、batch 32、chunk 1,000、`cutoff_len=1024`。无约束主口径保留非法候选原排名并计 miss；合法路径结果只作诊断；
- 正式命令：

```bash
bash launchers/run_evaluate_mmbert_ghr_aligned_sft_4x6000d.sh
```

#### 固定随机 Validation 10k

所有指标均为百分比。

| Epoch | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Valid ID Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.17 | 64.96 | 70.68 | 75.64 | 42.17 | 55.6677 | 58.0300 | 59.6684 | 60.758 |
| 2 | 48.68 | 72.14 | 78.44 | 82.80 | 48.68 | 62.5991 | 65.2064 | 66.6475 | 68.539 |
| 3 | **49.89** | **73.56** | **79.79** | **83.84** | **49.89** | **63.9770** | **66.5457** | **67.8842** | **68.836** |
| epoch 3 合法路径 | **50.13** | **73.91** | **80.26** | **84.76** | **50.13** | **64.2628** | **66.8841** | **68.3631** | **100.000** |

- epoch 3 相对 epoch 2 的 HR@1/HR@10/NDCG@10 继续提高 `1.21/1.04/1.2367pp`，三轮尚未出现指标反转；但无约束 epoch 3 相对 TIGER 低 `1.98/3.32/2.5348pp`，合法路径下相对 TIGER 同约束仍低 `2.04/3.23/2.5862pp`；
- 合法路径把 MMBERT 自身 HR@1/HR@10/NDCG@10 提高 `0.24/0.92/0.4789pp`，说明非法组合造成部分损失，但无法解释与 TIGER 的主体差距；无约束 epoch 3 的 100,000 个 Beam 候选中有 31,085 个完整 identifier 不在目录，其他结构/EOS 错误只有 79 个，主要仍是合法位置 Token 的跨层错误组合；
- MMBERT 的连续向量 Hit@10 比 BGE-M3 高 9.04pp，三层唯一率高 11.7223pp、碰撞 POI 少 15.2824pp，但完整生成三项主指标全部回落，正式否定“连续召回或静态唯一率提升会自然转化为生成召回提升”的假设。

#### 四类泛化 Validation 10k

| 子集 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Valid ID Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 已见 Query / 未见 Pair | 16.62 | 35.83 | 43.41 | 51.85 | 16.62 | 27.7150 | 30.8463 | 33.6101 | 67.651 |
| 新 Query / 已见目标 | 45.58 | 60.76 | 65.49 | 69.45 | 45.58 | 54.5618 | 56.5112 | 57.8077 | 46.685 |
| 长尾目标（Train 1—5） | 20.32 | 29.27 | 33.38 | 38.68 | 20.32 | 25.5439 | 27.2277 | 28.9425 | 37.801 |
| 冷目标（Train 0） | **7.93** | **12.38** | **14.99** | **18.39** | **7.93** | **10.4915** | **11.5687** | **12.6800** | 31.944 |
| **四组宏平均** | **22.6125** | **34.5600** | **39.3175** | **44.5925** | **22.6125** | **29.5781** | **31.5385** | **33.2601** | **46.0203** |

- 宏平均相对 TIGER 的 HR@1/HR@10/NDCG@10 低 `2.3800/6.7775/4.4348pp`，相对当前最接近 TIGER 的 BGE+RQ-KMeans 仍低 `2.0375/5.8700/3.7983pp`，因此 MMBERT 不替换 TIGER/BGE 作为当前完整生成 SID；
- 唯一局部正向出现在冷目标：相对 TIGER 的 HR@1/HR@10/NDCG@10 高 `0.85/0.61/0.8673pp`，相对 BGE+RQ-KMeans 的 HR@10/NDCG@10 也高 `0.08/0.4953pp`，但仍低 Centered GenPOI 的 20.81%/13.1261%。这支持线上 Query–POI 监督空间对无 Train 目标有迁移价值，同时表明其粗层组织和完整序列可生成性不适合直接替换 TIGER；小于 1pp 的单 seed 差异只作为方向信号。

#### 状态、产物、结论与下一步

- 状态核验：共享四卡任务生成 `16/16 completed` 成功汇总；本方法 8 个 checkpoint×数据集×解码单元均为 10,000 条，业务键与四个参考集分别一致，目标 POI mismatch 全为 0，实际 batch 均为 32，日志无 Traceback、OOM 或 child failure；
- 产物：固定无约束、合法路径和四类泛化结果分别位于 `outputs/eval/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1_gpu4_a100_e3/`、`outputs/eval/legal_path_fixed10k_v1/tiger_mmbert_recall_128_e3/` 和 `outputs/eval/generalization_validation_suite_10k_v1/mmbert_ghr_aligned_sft_epoch3_unconstrained_cutoff1024_v1/<subset>/mmbert_e3/`。两个固定结果 SHA256 为 `5d24e54444766957ee14a0089b507530adfe92cd2c903e8f17fea8cc034004c6`/`3540880db9f4f1f27341cfc166ebb43396a416e4659fc7969cf3e0f837f15946`；共享汇总 SHA256 为 `8c0abd3911998a5b8e6276211a6054ad11da61f463fa948eed7f856ac64f2ca0`；
- 结论：MMBERT 的冷目标信号值得保留为后续表征训练线索，但当前“MMBERT 向量→冻结 TIGER RQ-VAE→collision token”的完整方法没有通过随机集或泛化宏平均门槛，不再投入相同配置的更多 seed/epoch；
- 下一步：先对 epoch 3 补固定 10k 的 S1/S2/S3 gold-prefix、三层 Bucket 和条件 `C` 准确率，定位损失是在粗层可预测性还是后缀；若粗层已明显落后，则后续应在向量/SID 联合训练中显式加入层级 Query 可预测性，而不是继续追求静态唯一率。

### EXP-20260826-01：MMBERT epoch 3 逐层、Bucket 与 C 后缀归因诊断

#### 目标、假设与冻结输入

- 日期：2026-08-26；状态：已完成。目标是在不重新训练、不改变 MMBERT identifier 和不引入合法路径约束的前提下，对 `checkpoint-7599` 补齐固定随机 Validation 10k 的 gold-prefix 逐层准确率、无约束 Beam=10 三层 Bucket 覆盖和条件 `C` 准确率，判断 `EXP-20260822-03` 的端到端回落来自前三层 `Query→SID`，还是来自碰撞后缀；
- 预注册判断沿用本文件既定口径：若 MMBERT 三层唯一 Bucket 仍明显低于 TIGER，则把 MMBERT 保留为 scorer/teacher 候选，不再把其当前 RQ-VAE SID 作为生成器目标；只有 Bucket 强而 `C` 弱时，才优先考虑前缀保持的局部后缀或桶内排序；
- 评测数据由完整 MMBERT Validation 597,421 条按冻结 TIGER 参考集的 `order_id + searchid` 精确同序选择。参考 SHA256 为 `a2e0366d3d8f582e53a5293b687dc08f85b2960d60c7fb44d2fd02168063a944`，MMBERT 对齐子集 SHA256 为 `033a02071fa90c7196b7eefa2bfb9ae0a6be932ff27de67537f368199984b5d3`，业务键 SHA256 为 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`，目标 POI mismatch 为 0；该子集与 `EXP-20260822-03` 的固定 MMBERT 子集逐字节一致；
- checkpoint 固定为 epoch 3 `checkpoint-7599`，模型 SHA256 为 `e927ebdfc6d385755b87c1015f98cf3e422e7e13dc5b4a4bd58b593612a7d24d`。Tokenizer mapping SHA256 为 `451a0c1c5b0eaf8a5f2a2f773cc4bbd7d532ed659123d3543f444544e35d04d8`；2,337,178 条冻结 identifier 容量为 `[1024,1024,1024,154]`，mapping SHA256 为 `755dd3c914136991432e6adcb0dbb726c48f6a3d08d9485ced3d946c82c3da3e`；
- 运行对应 Git HEAD `54802e6674e283722ecee00fb862530df30cef9a` 的非干净工作树；正式运行前唯一已有改动是用户保留的 `third_party/LLaMA-Factory` 子模块状态，本实验未修改评测源码。Teacher-Forcing 脚本、TIGER 评测器和方法代码 SHA256 分别为 `52709176849c33febee5fde06d98f38e732801da568d53c285282bde90f881d0`、`6cd1f0d32676c4e9aa3d73bb391c5e3cd66038b179110a0edb57d4be2663299e` 和 `eb42c361ae6573b545c561e545b8e5af90644b05586c08a1af9200fe608ecf8f`。

#### 配置、命令与环境

- Teacher-Forcing 只计算 Assistant 目标位置，所有位置都给定正确 Prompt 和前序目标 Token；`S1/S2/S3` 的逐位置准确率分别表示 gold 前缀下的条件准确率，三级累计值要求前三层同时正确。`C` 条件准确率给定正确 `S1/S2/S3`；其中 `C>0` 组按 MMBERT 自身目标 code 划分，不等于所有目标桶大小大于 1 的请求，也不与 TIGER 的 2,356 条 `C>0` 构成同样本对照；
- 生成诊断保持论文复现主协议：无约束 Beam=10、返回 10 个候选、非法完整 ID 保留原 rank 并计 miss、无地理裁剪。`raw-slot Bucket` 保留原 Beam 槽位和重复桶；`unique Bucket` 移除不可在冻结目录展开的前三层桶，再按首次出现顺序去重；
- 环境为 `/ofs/map_search/hudan/envs/poi-gr`、Python 3.10、PyTorch 2.9.1+cu128、单卡 NVIDIA RTX A6000 48GB，短 `TMPDIR=outputs/tmp/mmdiag`。两项正式任务实际 batch 均为 32；Teacher-Forcing 推理 119.15 秒、峰值显存 7,137,369,600 bytes，Beam 生成推理 1,298.12 秒、峰值显存 26,476,898,816 bytes；
- 正式命令如下：

```bash
TMPDIR=outputs/tmp/mmdiag CUDA_VISIBLE_DEVICES=0 \
/ofs/map_search/hudan/envs/poi-gr/bin/python scripts/sft/diagnose_sid_teacher_forcing.py \
  --method tiger \
  --data-file outputs/eval/mmbert_layer_diagnostics_fixed10k_v1/bucket_generation/validation_subset_10000.jsonl \
  --checkpoint outputs/sft/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1_gpu4_a100_e3/checkpoint-7599 \
  --tokenizer models/Qwen3-0.6B-TIGER-MMBERT-Recall-128-1024x3-Vocab-v1 \
  --token-mapping-filename poi_token_mapping.json \
  --token-capacities 1024 1024 1024 154 \
  --output-dir outputs/eval/mmbert_layer_diagnostics_fixed10k_v1/teacher_forcing \
  --expected-epoch 3 --cutoff-len 1024 --batch-size 32 --checkpoint-rows 1000

TMPDIR=outputs/tmp/mmdiag CUDA_VISIBLE_DEVICES=0 \
/ofs/map_search/hudan/envs/poi-gr/bin/python scripts/tiger/evaluate_retrieval.py \
  --valid-file data/sft/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoints outputs/sft/tiger_mmbert_recall_128_1024x3_history10_query_gid_v1_gpu4_a100_e3/checkpoint-7599 \
  --expected-checkpoint-steps 7599 --expected-checkpoint-epochs 3 \
  --tokenizer models/Qwen3-0.6B-TIGER-MMBERT-Recall-128-1024x3-Vocab-v1 \
  --token-mapping-filename poi_token_mapping.json \
  --identifier-dir outputs/sid/tiger/mmbert_recall_128/TIGER-MMBERT-RECALL-128-1024x3/tiger_ids/epoch_20 \
  --output-dir outputs/eval/mmbert_layer_diagnostics_fixed10k_v1/bucket_generation \
  --num-beams 10 --per-device-eval-batch-size 32 --chunk-size 1000 \
  --cutoff-len 1024 --skip-data-hash --final-checkpoint-only --bucket-diagnostics
```

- 正式运行前，100 条 Prompt/identifier preflight 和 10 条两类 GPU smoke 均通过，相关单元测试 `tests/sft/test_teacher_forcing.py tests/tiger/test_eval.py` 为 `16 passed`。第一次 Teacher-Forcing smoke 误传不受支持的 `--batch-size 10`，在 argparse 阶段退出码 2，模型未加载且没有生成指标；改为 batch 8 后 10 条 smoke 完成。该失败只属于启动参数校验，不混入正式结果。

#### Gold-prefix 逐层结果

| 指标 | MMBERT | TIGER 参考 | MMBERT − TIGER |
|---|---:|---:|---:|
| `S1` 条件 / 累计 Top-1 | 69.06% | 75.06% | -6.00pp |
| `S2` 条件 Top-1 | 78.16% | 81.04% | -2.88pp |
| `S1+S2` 累计 Top-1 | 54.57% | 61.75% | -7.18pp |
| `S3` 条件 Top-1 | **88.31%** | 85.70% | +2.61pp |
| `S1+S2+S3` 累计 Top-1 | 50.17% | 53.98% | -3.81pp |
| 完整 `[S1,S2,S3,C]` Top-1 | 48.83% | 50.76% | -1.93pp |

- MMBERT 的前三层累计 Top-10 为 87.80%，完整四层 Top-10 仍为 87.80%；逐位置 Top-10 为 S1/S2/S3/C `92.09/94.42/96.20/100.00%`。结构 Token 的 `target_open/target_close/EOS` Top-1 均为 100%，不存在闭合混淆；
- 以 4,883 条完整四层 Top-1 正确样本反推 5,117 条失败：3,094 条在 S1 首次失败，1,449 条在正确 S1 后首次失败于 S2，440 条首次失败于 S3，只有 134 条在前三层全对后失败于 C。也就是说，完整 ID 的 Teacher-Forcing 失败有 88.78% 已在前两层发生，只有 2.62% 可归因于最后的 C；
- 全部 10,000 条的条件 C Top-1 为 97.43%。MMBERT 自身 `C>0` 组有 1,003 条，条件 C Top-1/Top-10 为 83.2502%/100%；`C=0` 组有 8,997 条，对应 99.0108%/100%。作为方向参考，TIGER 自身 `C>0` 组为 2,356 条、条件 C Top-1 为 81.0272%；两者 mapping、样本集合和碰撞率不同，不能把 +2.22pp 写成严格配对提升；
- MMBERT 深层局部条件预测并不弱：S3 和 C 均高于 TIGER 方向值，差距集中在 S1/S2。这与静态诊断中 MMBERT depth-1/2 类别 micro-purity 只有 52.13%/70.03%、明显低于 TIGER 的 69.28%/77.27% 相互印证。

#### 自由生成 Bucket 与完整 POI

| 排名口径 | HR@1 | HR@3 | HR@5 | HR@10 | 相对精确 POI 增量 |
|---|---:|---:|---:|---:|---:|
| 精确 `[S1,S2,S3,C]→POI` | 49.80% | 73.66% | 79.71% | 83.83% | — |
| 三层 Bucket，保留原 Beam 槽位 | 51.11% | 74.39% | 80.30% | 84.19% | +1.31/+0.73/+0.59/+0.36pp |
| 三层唯一可展开 Bucket | **52.05%** | **75.66%** | **81.24%** | **84.19%** | **+2.25/+2.00/+1.53/+0.36pp** |

| 唯一可展开 Bucket | MMBERT | TIGER | MMBERT − TIGER |
|---|---:|---:|---:|
| HR@1 | 52.05% | 55.53% | -3.48pp |
| HR@3 | 75.66% | 79.64% | -3.98pp |
| HR@5 | 81.24% | 85.10% | -3.86pp |
| HR@10 | 84.19% | 88.06% | -3.87pp |

- 10,000 条中有 1,617 条精确 Top-10 miss，其中 1,581 条连目标三层 Bucket 都没有进入 Beam；因此 97.77% 的精确 miss 已经发生在前三层覆盖阶段，当前 Beam 内即使理想消除 C/桶内排序错误也只新增 36 条命中、HR@10 上界只增加 0.36pp。Top-1 的理想 Bucket 空间也只有 225 条、即 2.25pp；
- 同次正式生成的 NDCG@1/3/5/10 为 `49.8000/63.9925/66.4894/67.8504%`，Valid ID Rate 为 68.894%；
- MMBERT 固定 10k 的目标三层桶中 7,980 条为单例，1,179/647/143/50/1 条分别落在大小 `2/3—5/6—10/11—50/51+`，平均桶大小 1.4567、最大 60；TIGER 对应单例只有 61.46%、平均桶大小 2.5339。MMBERT 确实显著减少请求侧碰撞，但更低的 Bucket HR 证明“更唯一”没有转化为“更可生成”；
- 100,000 个 Beam 候选中，30.222% 的前三层前缀无法在目录展开，4.155% 是已出现桶的重复候选，每条请求平均保留 6.5623 个唯一可展开桶；MMBERT 的不可展开率比 TIGER 23.870% 高 6.352pp，而唯一桶均值与 TIGER 6.5462 基本相同。完整 ID Valid Rate 为 68.894%，主要非法项仍是 31,024 个 `identifier_not_in_corpus`；
- 本次 A6000 复跑的精确 HR@1/3/5/10 为 `49.80/73.66/79.71/83.83%`，相对 `EXP-20260822-03` 的 6000D 历史值变化 `-0.09/+0.10/-0.08/-0.01pp`，NDCG@10 变化 -0.0338pp。checkpoint、数据和评测代码哈希均一致，差异小于 0.10pp，记录为跨硬件无约束 Beam 的数值轨迹扰动；本实验的 Bucket 增量统一以同一次 A6000 结果为分母，不混用历史精确指标。

#### 产物核验、结论与下一步

- 正式根目录为 `outputs/eval/mmbert_layer_diagnostics_fixed10k_v1/`。Teacher-Forcing `result.json` / `progress.json` SHA256 为 `4c9914d0330ba023c927c9aa6cbed38846dd37c662396a60ef1db71a9d204b70` / `3b796be291b5084467645ba7805c951c874ccd52964805d015f1941cb17fc5a4`；Bucket 汇总 / 正式 run `result.json` SHA256 为 `1b45b4c302dbe8d29773476bff8e6fcf0fbbc6029a4634466f7f744c2233e9df` / `9fc3c3979cb151881d337eb742f77115c499ab2f2a76e97a03fabc4b997e9303`；
- 正式 `candidate_trace.jsonl` 为 10,000 行、每行 10 个候选、约 36 MiB，SHA256 为 `273366469aa0fe746bf48a894a3a0ee997e4bdf0ce9ad1f09bf326da87b3a8c9`，manifest SHA256 为 `847a4dd803c22e8571322bf4e6a080273a60a90569ee0edb929a15ee78436319`。独立从 `tiger_ids.npy` 和 Token mapping 逐行复算确认：目标 code、目标/候选桶大小、原始 Token 前缀、exact/raw/unique rank、唯一桶列表和三组 rank histogram 均为 0 mismatch；10 个 1,000 行分片各自行数与哈希通过，顺序拼接 SHA256 与总轨迹完全一致；
- Teacher-Forcing / Bucket 日志 SHA256 为 `0c48cfe909d222c5dedc81e28a733152f1357d722a9c8746301dcfba518bc68c` / `2b962d5666701b21e615c44cfcfab466534643e66944baf9a368494ec3b8bced`，均无 Traceback、OOM、失败或 batch 回退。全部模型、轨迹和日志继续保持 Git 忽略；
- 核心结论：MMBERT 的 128 维监督向量提高了连续召回、码本均衡性、三层唯一率和冷目标局部泛化，但当前 `MMBERT→RQ-VAE` 的粗层离散组织不具备更好的 Query 可预测性。失败根因不是 C 容量或桶内消歧，而是 S1/S2 的层级对齐和自由生成目标桶覆盖；继续减碰撞、扩 C、增加 epoch/seed 或沿当前 SID 直接重训都缺乏证据；
- 可验证的下一假设是：MMBERT 的价值更适合作为冻结 TIGER 候选的连续 scorer/teacher，而不是直接替换生成 identifier。下一最小门禁建议复用现有 TIGER/MMBERT 候选轨迹和冻结向量，先离线测量 MMBERT 对 TIGER 候选的重排增量、候选互补覆盖及冷目标分桶；只有该无训练门禁为正，才讨论 prefix-preserving distillation/辅助排序目标。该建议待用户确认，本实验不启动新 SFT，也不批准继续扩搜静态码本。

## EXP-20260904-03：活跃闭集 MMBERT Recall 128 与 TIGER `512³` SID

### 目标、数据与严格对照

- 日期：2026-09-04—05；状态：已完成。目标是在新的 716,245 POI 活跃闭集上，用线上 MMBERT Recall checkpoint 重新编码目录，并保持 `EXP-20260904-02` 的三层 `512×512×512` TIGER RQ-VAE、全目录 KMeans 初始化、20 epoch 和稳定 collision token 规则不变，形成与 active-BGE 可直接比较的静态 SID 候选；
- POI 主表固定为 `data/beijing_poi_active_order14d_history10_20260715_json/`，POI ID SHA256 为 `8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。MMBERT 使用线上 checkpoint `/ofs/map_search/xiaolu/multilingual_search/recall/mmBERT-emb/output_checkpoint/output_4gpus_1pos_30neg_add_mlp_v2_plus/checkpoint-00612000`，仍只读取冻结的名称、地址和别名文本，执行 attention-mask mean、训练好的 `Linear(768,128)` 和 L2 normalize，不读取 query 或订单标签；
- 编码产物为 `outputs/embeddings/beijing_poi_active_order14d_history10_mmbert_recall_128/`，shape `[716245,128]`、float16，全部 finite，L2 norm min/mean/max 为 `0.999858/1.000000/1.000154`。manifest / NPY SHA256 为 `424b8b61315e2ed1f0368b99bd1091773080f72f13ff94bd499b0ca528d10b56` / `adb7584d23a748af54c579d924a1a7b365ab8d935a6dda9db25f9cb8c78ceb28`；编码耗时 2,979.66 秒，吞吐 240.38 POI/s；
- SID 配置为 `configs/sid/rqvae_tiger_mmbert_recall_128_active_716k_512x3.yaml`：输入/隐藏/latent 为 `128/512/256`，三层 `512³`，三级 FAISS GPU KMeans 均覆盖 716,245/716,245 行、20 iterations，seed 42、batch 4096、20 epoch。全流程入口为 `launchers/run_build_tiger_active716k_mmbert_recall_128_512x3.sh`，7 个阶段均以退出码 0 完成。

### 训练与静态 SID 结果

- 三级 KMeans 初始化后每层均使用 512/512 个码、无死码，初始化 MSE 为 `0.0385193/0.0310101/0.0258807`；正式训练段耗时 57.85 秒，epoch 20 Train/Validation reconstruction cosine 为 `0.716160/0.714953`。全量导出的三级码字利用均为 512/512，normalized entropy 为 `0.978291/0.986728/0.987152`；
- 三层 SID 共 648,917 个不同组合，唯一率 `90.5999%`；碰撞 excess 67,328（`9.4001%`），116,159 个 POI 位于碰撞桶（`16.2178%`），共 48,831 个碰撞桶。桶大小 P50/P90/P95/P99/Max 为 `1/1/2/3/30`；
- 按原 TIGER 规则在三层桶内按 `poi_id` 字典序稳定分配第四层，只需 `C0–C29` 共 30 个 collision token。最终 `[S1,S2,S3,C]` 为 `[716245,4]` int32，716,245 个标识全部唯一；独立复检确认前三列逐行等于 `sid_codes.npy`、第四列逐行等于 `collision_codes.npy`，各层范围为 `[0,511]/[0,511]/[0,511]/[0,29]`。

| 同一 active 目录与 `512³` 协议 | active-BGE | active-MMBERT | MMBERT − BGE |
|---|---:|---:|---:|
| 三层 SID 唯一率 | 77.0310% | **90.5999%** | **+13.5689pp** |
| collision excess ratio | 22.9690% | **9.4001%** | **-13.5689pp** |
| 碰撞 POI ratio | 33.7781% | **16.2178%** | **-17.5603pp** |
| 最大三层桶 | 152 | **30** | **-122** |
| 所需 collision token | 152 | **30** | **-122** |
| 三级码字利用 | 229/506/491 | **512/512/512** | 全层满利用 |

### 产物、核验与结论

- 正式 run 位于 `outputs/sid/tiger/active_716k_mmbert_recall_128/TIGER-ACTIVE716K-MMBERT-RECALL-128-512x3-FULLINIT/`；epoch-20 checkpoint / resolved config SHA256 为 `239d7d468a191e838494fa3ac549171c900913fe52ca3309137a838c88eb62b1` / `762b2dcc744f7ce557ce6af47b226b6eb8fbf9c9d5ed2c75898a0b81b6b3cb01`；
- SID manifest / NPY SHA256 为 `44e7b9fb4b927066b378894124037484a552f98dc8c32f5437750ebc31fbff6d` / `d53e5e800b577714d924eafe01d997c35d8166702a151f4faa6a0c4378c4a82e`；identifier manifest / NPY / mapping SHA256 为 `f8330dfff2a419fb533806cd26bc1c55a7999245d24147486470f38f9de1a954` / `978f458fdf2f7f2e73b4908d32cca736f8e264952151251854b2584abfdc4d1a` / `fbc2aa910018bb8e7a95748a503afa874d4f4540e0331f0ab34a5fa9f0c6277e`。正式 pipeline 日志 SHA256 为 `b87361c446b6c6c411aa13f17a1bef75e4c54b7b74b7bad00d9ab429922f3114`；
- 相关 Embedding subset/RQ-VAE 回归测试、Python 编译和 launcher shell 语法检查均通过。两次未完成的慢速全库 subset 尝试已完整保留在 `outputs/quarantine/EXP-20260904-03_active_mmbert_sid/`，不混入正式结果；正式向量由可断点恢复的 active 目录直接编码生成；
- 本实验只证明 MMBERT 在同一活跃目录、同一码本容量下显著改善静态量化均衡和碰撞规模，不能写成生成式检索超过 TIGER。旧 233 万目录上的 MMBERT 已出现“静态 SID 更优、SFT 反而更差”，因此下一步若投入训练资源，必须与 active-BGE 使用完全相同的样本、词表构造、global batch、三轮训练和六项评测协议做配对 SFT，最终只以固定 10k 和四类泛化 HR/NDCG 决策。

## EXP-20260906-01：活跃闭集 MMBERT 配对 SFT 输入与六项评测入口

### 严格配对数据与 Token 门禁

- 日期：2026-09-05—06；状态：正式训练入口已就绪，尚未产生 checkpoint。基于 `EXP-20260904-03` 的 `[S1,S2,S3,C]` 唯一标识重建 history10 数据，Messages 位于 `data/sft/tiger_active716k_mmbert_recall_128_512x3_history10_query_gid_v1/`，Train/Valid/Test 为 `7,586,410/597,421/606,682`，总计 8,790,513 条；manifest / special-token SHA256 为 `e8bdb183f2cabcbd58cae5a96c770526cce8b2bee26f72b9b01934fc00b9abc7` / `97c22e3f8fb5c03cba951680ac13bb543ea86df467b6b2158bf3994e8de1a51e`；
- 与 `EXP-20260905-01` 的 active-BGE Messages 做了 8,790,513 条同行同序全量审计：`sample_id/user_token/order_id/searchid/target_poi_id/history_length/split` 完全相同，Prompt 和 Target 在把 identifier 归一化后完全相同，只有历史与目标 identifier 字段变化。审计日志位于 `outputs/run_control/tiger_active716k_mmbert_sft_data/paired_audit.log`，最终状态为 `only_identifier_fields_differ=true`；
- 扩词模型位于 `models/Qwen3-0.6B-TIGER-Active716K-MMBERT-Recall-128-512x3-Vocab-v1/`：在 Qwen3-0.6B 的 151,669 词表上新增 3,614 个普通原子 Token，扩至 155,283。Token mapping / tokenizer JSON SHA256 为 `fb97ffc3210759c8820f90e9fb7638d0d943f568263dd23fe02c4131c73192d0` / `7c374e49a90a341102b6f7fa5d5d218c02a17f7c49db770fe1aed54f7e097930`，全部新增 Token 通过原子编码和往返解码门禁；
- Train+Valid 共 8,183,831 条完成 `cutoff_len=1024` 全量预检：输入/完整序列最大为 `945/953`，目标长度固定 8，超过 1024、requested/effective cutoff 下的 Assistant 目标截断均为 0。正式 packed cache 位于 `data/sft/tokenized/tiger_active716k_mmbert_recall_128_512x3_history10_query_gid_v1/`，Train/Validation 为 `1,296,883/96,949`，10k/2k smoke 为 `1,711/323`；cache manifest / length stats SHA256 为 `b8d6022a52e3806ce52d4311d8719fa672ad459705b3a1e751f7e1d49bd5514d` / `d1b80cc56460aa36b96231b7a081002a9c5f88f482da31154b436c7ace3a2014`；
- 第一次 cache 尝试在沙箱的 16 进程格式转换入口因 `multiprocess.Manager` 无法创建 AF_UNIX socket 而 `PermissionError` 退出，未发布正式目录；宿主环境重试复用已完成的全量长度预检，最终退出码 0 并原子发布 28G cache。该调试失败不混入正式训练结果，也不是数据、长度或 Tokenizer 错误。

### 训练与评测入口

- 正式配置为 `configs/sft/tiger_active716k_mmbert_recall_128_512x3_history10_query_gid_v1.yaml`：Qwen3-0.6B 全参数 SFT、4×RTX PRO 6000D、单卡 batch 8、累积 16、global batch 512、BF16、seed 42、三轮。按 1,296,883 packed Train rows 计算为每轮 2,533 step，epoch 3 固定 `checkpoint-7599`；
- 一键入口为 `launchers/run_train_tiger_active716k_mmbert_recall_128_512x3_4x6000d_3epoch.sh`。Shell 语法和全链路 dry-run 均以退出码 0 完成：训练后只评 epoch 3，同时运行固定同业务键 Validation 10k 的无约束 Beam=10、全目录合法路径约束，以及已见 Query/未见 Query–POI、新 Query/已见目标、长尾目标、冷目标四个冻结泛化 10k，共六项；
- 配置 / launcher SHA256 为 `32c6b8f55c214f1db7845bdd1ad7fcd126b58ba8e05c8423b0bb031fdc419cec` / `5b738bd149885e1aa0108946980e3d938ef81668e9175ea7027824042481e669`。相关 TIGER 数据和训练入口回归测试为 `35 passed`。本实验目前只确认“可以安全提交正式训练”，没有 Loss、HR、NDCG 或相对 active-BGE 的生成式结论；最终仅按六项配对 SFT 结果决策 MMBERT 是否保留。

## EXP-20260908-02：活跃闭集 MMBERT 配对三轮 SFT 与评测中止

- 日期：2026-09-07—08；状态：三轮 SFT 完成，生成评测尚未启动。数据、Tokenizer、SID 和配置沿用 `EXP-20260906-01`；代码基线为 `54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树有未提交改动；
- 4×RTX PRO 6000D、BF16、单卡 batch 8、累积 16、global batch 512，共完成 7,599 step。epoch 1/2/3 Validation Loss 为 `0.476800/0.363435/0.348109`，最终 checkpoint 为 `outputs/sft/tiger_active716k_mmbert_recall_128_512x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-7599`；模型、优化器、调度器、四卡 RNG 状态和 Trainer state 均完整；
- 平台 failed 的原因与 active-BGE 相同：训练成功后未写出 `epoch_checkpoints.json`，后置索引检查退出；索引现已按三个标准 checkpoint 补建。该故障不影响 checkpoint，但使平台六项生成评测没有启动；
- 本服务器的配对补评 runner 按 TIGER→MMBERT 顺序执行，用户在 TIGER 首个泛化单元期间要求停止，因此 runner 从未进入 MMBERT。当前没有 active-MMBERT 的 HR/NDCG/合法率，不能根据较高的 Validation Loss 或静态 SID 唯一率判断它是否优于 active-BGE；
- 训练产物位于 `outputs/sft/tiger_active716k_mmbert_recall_128_512x3_history10_query_gid_v1_gpu4_6000d_e3/`，运行中止记录位于 `outputs/run_control/active716k_pair_eval/`。若继续，直接在训练平台从现有 epoch-3 checkpoint 运行固定 10k 双解码和四类泛化，不在本服务器恢复。

### 2026-09-09：A100 六项评测闭环

- 复用同一 `checkpoint-7599` 在 4×A100 平台完成固定 10k 无约束/合法路径与四个无约束泛化 10k。六项均为 `completed`、各 10,000 条，Test 未读取；与 active-BGE 对应单元的业务键顺序 SHA256 一致，target mismatch 为 0。active-BGE 的四个泛化单元也在本次 A100 任务中完成，构成严格同卡型配对；但其两项固定 10k 已存在而被 launcher 跳过，所以固定集的小幅差异需保留 A6000/A100 硬件口径提示；

| Validation 切片 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | Valid ID Rate |
|---|---:|---:|---:|---:|---:|---:|
| 固定 10k，无约束 | 49.46% | 73.27% | 79.61% | 84.04% | 67.6860% | 70.243% |
| 固定 10k，合法路径 | 49.77% | 73.79% | 79.95% | 84.57% | 68.0924% | 100% |
| 已见 Query / 未见 Query–POI | 15.72% | 33.83% | 42.05% | 50.63% | 32.4171% | 69.679% |
| 新 Query / 已见目标 | 44.28% | 59.82% | 64.58% | 68.54% | 56.7105% | 48.779% |
| 长尾目标（Train 频次 1—5） | 20.14% | 29.63% | 34.14% | 38.84% | 29.0055% | 40.648% |
| 冷目标（Train 频次 0） | 8.66% | 14.06% | 16.72% | 19.92% | 13.9358% | 33.731% |
| 四类宏平均 | 22.2000% | 34.3350% | 39.3725% | 44.4825% | 33.0172% | 48.2093% |

- 固定 10k 无约束相对 active-BGE 的 HR@1/HR@10/NDCG@10 为 `-0.32/-0.47/-0.3636pp`；合法路径下为 `-0.02/-0.51/-0.1710pp`，因此 MMBERT 没有超过固定主评测。四类泛化宏平均则为 `+2.0750/+2.2175/+2.3183pp`，主要由长尾目标的 `+4.40/+2.26/+3.7958pp` 和冷目标的 `+4.21/+10.44/+7.3830pp` 贡献；已见 Query/未见 Pair 三项反而下降 `1.16/2.82/1.9627pp`，不是全面提升；
- MMBERT 四类宏平均 Valid ID Rate 比 active-BGE 低 `11.1823pp`，冷目标低 `15.458pp`，但长尾/冷启动召回仍显著更高。这说明其 SID 含有更强的稀疏目标语义泛化信号，同时 identifier 路径更难被 Qwen 合法生成；静态三层唯一率和满码本利用率不能直接等价为下游可生成性；
- 相对旧 233 万目录 MMBERT，active 版固定无约束 HR@1/HR@10/NDCG@10 为 `-0.43/+0.20/-0.1982pp`，四类宏平均为 `-0.4125/-0.1100/-0.2429pp`，整体基本稳定。对照 active-BGE 的明显回落，当前最有价值的信号不是“active 缩库提高主指标”，而是“MMBERT 表征对目录/SID 重建及长尾冷目标更稳健”；它仍不足以整体替代 TIGER，可作为后续可生成前缀结构中的语义教师或辅助信号；
- 完整六项汇总为 `outputs/eval/tiger_active716k_mmbert_recall_128_512x3_history10_query_gid_v1_gpu4_6000d_e3/epoch3_evaluation_summary.json`，SHA256 为 `cd682d42fee5d933110fd4e7a43687c41e05c7c566e6655999592143ed518911`。
