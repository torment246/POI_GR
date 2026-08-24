# GenPOI 复现实验

本文件集中记录 GenPOI 在北京地图 POI 检索场景中的完整复现链路：BGE-M3 + GeoPE32 → 三层 RQ-VAE SID → Geohash6/Dedup 唯一 PID → 最近 10 条历史序列 SFT → SSP 地理深度预测与 TCG Trie 约束评测。

## 当前结论

- GeoPE32 三容量对比选择 `1024×3 / epoch 20`，全量 SID 唯一率 73.1125%；Geohash6 + Dedup 后 2,337,178 条 PID 全局唯一。
- 历史序列数据、3,626 个原子 Token、正式 packed Cache和四卡 RTX PRO 6000D 三轮训练均已完成。
- 固定 10,000 条 Validation 的 TCG+SSP、Beam=10 评测已完成旧版和 Centered 两组 epoch 1/2/3 checkpoint；Centered epoch 3 的 HR@1/HR@10/NDCG@10 为 51.98%/87.83%/70.7430%，为当前 GenPOI 最优。
- 三轮生成结构和 PID 合法率均为 100%；结果仅代表固定 Validation 子集，尚未运行完整 Validation 或 Test。

## 实验记录
## EXP-20260802-01 GenPOI GeoPE32 三容量 SID 全量对比

### 目标与假设

- 目标：在北京 2,337,178 条 POI 的 BGE-M3 向量上加入 GenPOI GeoPE，再使用三层 RQ-VAE 对比 `256×3`、`512×3`、`1024×3` 的完整 SID 质量。
- 假设：GeoPE 会按空间位置重排原始语义邻域，因此不要求第一层类别纯度或码本利用率高于直接 BGE-M3；重点核验三层完整 SID 的唯一率、碰撞分布和 Prefix3 类别纯度是否随容量稳定改善。
- 状态：32 锚点 GeoPE 全量向量、三组 20 epoch 训练和三组 2,337,178 条全量 SID 导出均已完成。

### 数据、代码与环境

- 源向量：`outputs/embeddings/beijing_poi_bge_m3/`，shape `[2337178,1024]`，dtype `float16`；GeoPE 产物为 `outputs/embeddings/beijing_poi_bge_m3_genpoi_geope/`，shape 与 dtype 不变。
- GeoPE：从北京本地等距投影坐标抽样 200,000 条，使用 MiniBatch K-Means 拟合 32 个锚点；每个锚点对应 32 维向量分段，按锚点到 POI 的方位角对连续二维分量旋转。旋转前后最大范数差为 `1.1921e-7`。
- GeoPE fingerprint 为 `7d3451e284f67c4bf343eb74f96d5a8b41ec874050f0b83836c430243617487d`；POI ID 2,337,178 行且唯一，SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`。
- 固定划分：train 2,313,806，validation/monitor 23,372；KMeans 初始化固定抽样 500,000 行，seed 42。三容量复用相同行号哈希。
- 代码状态：基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 GeoPE、RQ-VAE 稳定性适配和相关测试的未提交工作树。
- 环境：`poi-gr`，Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、NVIDIA RTX A6000。

### 配置与命令

- 模型统一为 `1024 -> 512 -> 256 -> 128 -> 32` Encoder、三层 residual quantization 和对称 Decoder；重构输入按 L2 归一化，平方误差使用 `vector_sum`。
- Adam learning rate `5e-4`、batch size 4096、20 epoch、codebook/commitment 权重 `1.0/0.25`；Faiss GPU K-Means 使用 500,000 条样本和 20 iterations，仅保存 epoch 20。
- 256/512/1024 训练耗时分别为 762.56/754.82/918.37 秒；全量导出耗时分别为 177.32/767.58/969.52 秒，均正常退出。

~~~bash
python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_genpoi_bge_m3_geope.yaml \
  --experiment GenPOI-BGE-M3-GeoPE-<CAPACITY>x3 \
  --no-resume \
  --no-progress

python scripts/sid/export_rqvae.py \
  --run-dir outputs/sid/genpoi/bge_m3_geope/GenPOI-BGE-M3-GeoPE-<CAPACITY>x3 \
  --checkpoint checkpoint_epoch_20.pt \
  --output-dir outputs/sid/genpoi/bge_m3_geope/GenPOI-BGE-M3-GeoPE-<CAPACITY>x3/evaluations/epoch_20 \
  --device cuda \
  --batch-size 4096
~~~

### Epoch 20 全量结果

| 容量 | Val recon / cosine | Val 码字数 L1/L2/L3 | Distinct SID / ratio | excess / colliding POI | P99 / Max | 全量利用率 L1/L2/L3 |
|---|---|---|---|---|---|---|
| `256×3` | 0.350838 / 82.47% | 66/256/254 | 967,245 / 41.3852% | 58.6148% / 77.5753% | 15 / 455 | 25.78%/100%/99.22% |
| `512×3` | 0.345740 / 82.72% | 62/512/493 | 1,355,532 / 57.9987% | 42.0013% / 59.8515% | 10 / 365 | 12.11%/100%/96.48% |
| `1024×3` | 0.340739 / 82.97% | 59/1024/973 | 1,708,769 / 73.1125% | 26.8875% / 40.7493% | 6 / 232 | 5.76%/100%/95.31% |

### 前缀类别纯度与 TIGER 对照

| 容量 | Prefix1 Macro/Micro | Prefix2 Macro/Micro | Prefix3 Macro/Micro |
|---|---:|---:|---:|
| `256×3` | 35.04% / 33.20% | 54.67% / 57.47% | 87.85% / 81.20% |
| `512×3` | 31.78% / 30.81% | 54.95% / 59.25% | 93.60% / 88.97% |
| `1024×3` | 32.36% / 30.46% | 58.82% / 61.91% | 97.17% / 94.72% |

- GeoPE `1024×3` 相比直接 BGE-M3 的 TIGER `1024×3`，完整 SID 唯一率由 71.6766% 提升到 73.1125%，P99/最大桶由 `7/306` 降到 `6/232`；碰撞 POI 比例基本持平，为 40.7493% 对 40.7894%。
- GeoPE 的第一层只使用 59 个粗前缀，明显少于 TIGER 的 561 个，且 Prefix1 类别 Micro Purity 为 30.46%，低于 TIGER 的 69.28%。这是逐 POI 地理旋转和 32 维潜变量共同改变粗层聚类的结果，不能把完整 SID 的差异仅归因于 GeoPE。
- 三组初始化样本上的三层码本均 100% 使用；训练期间重构损失持续下降，第二层在 1024 容量下全量使用，第三层使用 976/1024，完整 SID 指标随容量单调改善，未发现输入错位、数值异常或全层塌缩。

### 产物、结论与下一步

- 正式产物位于 `outputs/sid/genpoi/bge_m3_geope/GenPOI-BGE-M3-GeoPE-{256,512,1024}x3/`；每组包含 epoch 20 checkpoint、训练指标和 `evaluations/epoch_20/` 下的 SID、manifest、metrics 与碰撞 Case。
- 本实验选择 `1024×3 / epoch 20` 作为 GenPOI 后续 PID 候选。唯一率只衡量标识冲突，不直接等价于地图检索效果；GeoPE 的收益仍需在后续 GenPOI GID/Dedup PID 和固定 Query/历史输入的生成评测中验证。
- 本实验不修改 TIGER 已有 SID、collision token、SFT 数据或正在运行的生成模型训练任务。
## EXP-20260802-02 GenPOI Geohash6 GID 与唯一 PID

### 目标、数据与代码状态

- 目标与假设：基于已选定的 GenPOI GeoPE `1024×3 / epoch 20` SID，按照 `[G1..G6,S1..S3,optional Dedup]` 构建地图 POI 标识；Geohash6 先拆分跨区域同 SID，剩余局部碰撞再用确定性 Dedup 保证严格一一映射。
- 数据版本：SID 输入为 `outputs/sid/genpoi/bge_m3_geope/GenPOI-BGE-M3-GeoPE-1024x3/evaluations/epoch_20/`，shape `[2337178,3]`、dtype `int32`、码本 `[1024,1024,1024]`；POI ID 为 2,337,178 行且唯一，SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`。
- 代码状态：基线提交为 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 GenPOI GeoPE/SID 和 PID 兼容性修正的未提交工作树。原 PID 构建器硬编码旧 Qwen 实验名和旧 PID 指标，初次运行均在写产物前以 exit code 2 结束；已改为校验三层 1024 结构、epoch 20、哈希、行映射以及当前 PID-001 产物的逐字段重算指标。
- 环境：`poi-gr`，Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1；base PID 和 Dedup PID 构建分别耗时 136.84 秒和 97.13 秒。

### 配置与命令

- GID 使用标准 Geohash6，经纬度顺序为 longitude-first，不做坐标系转换；base PID 固定为 `[G1..G6,S1..S3]`。
- 单例 base PID 不追加有效 Token，在固定宽度数组第十列写 `-1` sentinel；碰撞桶内按 `poi_id` 字典序分配连续 `D0..D(bucket_size-1)`。预留 Dedup 容量为 512，实际最大桶为 216。

~~~bash
python scripts/pid/build_geohash.py \
  --sid-manifest outputs/sid/genpoi/bge_m3_geope/GenPOI-BGE-M3-GeoPE-1024x3/evaluations/epoch_20/sid_manifest.json \
  --poi-data data/beijing_poi_clean_20260715_json \
  --geohash-length 6 \
  --order gid_sid \
  --output-dir outputs/pid/genpoi/GenPOI-BGE-M3-GeoPE-1024x3-e20-G6

python scripts/pid/build_dedup.py \
  --pid-manifest outputs/pid/genpoi/GenPOI-BGE-M3-GeoPE-1024x3-e20-G6/pid_manifest.json \
  --dedup-capacity 512 \
  --output-dir outputs/pid/genpoi/GenPOI-BGE-M3-GeoPE-1024x3-e20-G6-Dedup
~~~

### 全量结果与核验

| 标识 | Distinct / ratio | 碰撞 POI / ratio | P99 / 最大桶 |
|---|---:|---:|---:|
| GeoPE 三层 SID | 1,708,769 / 73.1125% | 952,383 / 40.7493% | 6 / 232 |
| Geohash6 + SID base PID | 2,004,443 / 85.7634% | 519,541 / 22.2294% | 4 / 216 |
| Final PID | 2,337,178 / 100% | 0 / 0% | 1 / 1 |

- Geohash6 使 distinct ratio 提升 12.6509 个百分点，拆开 432,842 条原 SID 碰撞 POI；剩余 186,806 个碰撞桶涉及 519,541 条 POI。
- 1,817,637 条单例保持九段 PID；519,541 条碰撞 POI 使用 Dedup，最大 Code 为 `D215`。`final_pid_codes.npy` shape 为 `[2337178,10]`，删除第十列可逐行恢复 base PID，最终全量唯一性校验通过。
- 映射表 `poi_pid_mapping.parquet` 为 2,337,178 行、24 个 row group，SHA256 为 `619b8afa106e3b08508589e59f95dfd2df56e233da6b152d7730157da1faa1d1`；Final PID manifest SHA256 为 `c1a30d1d50929d99b1008c6d524d28ee2431c75b4fd9345bfb4a633be3e54686`。

### 产物、结论与下一步

- base PID 位于 `outputs/pid/genpoi/GenPOI-BGE-M3-GeoPE-1024x3-e20-G6/`；唯一 PID 与 POI 映射位于同名 `-Dedup/` 目录，均由 Git 忽略。
- GenPOI identifier 阶段完成，可作为 `scripts/genpoi/build_sft_data.py` 的正式 PID 输入；本实验未生成 Train/Validation/Test，也未修改 TIGER 产物或训练任务。
- 下一最小步骤是生成 GenPOI 正式时间切分数据，并在训练前完成字段计数、PID 命中率、历史因果性和 Tokenizer 长度预检。
## EXP-20260803-01（GenPOI-SFT-PREP-001）历史序列 SFT 训练前闭环

### 目标与假设

- 使用选定的 BGE-M3 GeoPE32 `1024×3 / epoch 20` SID、Geohash6 和确定性 Dedup PID，构建地图检索适配的 GenPOI 历史序列 SFT 数据，并准备 Qwen3-0.6B 单卡三轮训练入口。
- 假设按历史位置、历史 Query、历史交互 POI、当前位置和当前 Query 的顺序线性化最近 10 条历史，可以在不引入原始 passenger ID 和事件时间文本的前提下验证 GenPOI 方法差异。
- 本实验完成全量 Train/Valid/Test 数据、扩词表、Train/Valid 长度预检、正式 Tokenized Cache、单卡 dry-run 和实际 GPU smoke；Test 未进入缓存，也未启动正式训练。

### 数据、标识与处理规则

- 输入为 `data/beijing_order_clean_20260701_20260714_history10_20260401_20260630_json/` 的 32 个分片；目标订单为 2026-07-01 至 07-14，历史窗口为 2026-04-01 至 06-30，历史按事件时间正序排列并截取最近 10 条。
- 全量保留 8,790,513 条目标订单，Train/Valid/Test 严格按 12 天/1 天/1 天切分为 7,586,410/597,421/606,682；空 Query 为 0，PID 匹配率为 100%。历史覆盖率为 72.0157%，平均历史长度为 4.9153，2,459,967 条空历史样本保留。
- PID 映射为 `GenPOI-BGE-M3-GeoPE-1024x3-e20-G6-Dedup`，覆盖 2,337,178 条 POI；映射 SHA256 为 `619b8afa106e3b08508589e59f95dfd2df56e233da6b152d7730157da1faa1d1`，Manifest SHA256 为 `c1a30d1d50929d99b1008c6d524d28ee2431c75b4fd9345bfb4a633be3e54686`。
- 每条历史依次写入 `<USER_GID>`、原始 `<QUERY>` 和 `<POI_PID>`，当前请求只写入 `<USER_GID>` 与原始 `<QUERY>`；Assistant 目标为目标 POI 的九层单例 PID 或十层 Dedup PID。事件时间仅用于因果和窗口校验，不进入 Prompt。
- 正式数据 Manifest SHA256 为 `89df085db5e1f11ccb5fbf2adaf5536bde9176102f25d7dcf26b7ee9158fcbd0`；Train/Valid/Test SHA256 分别为 `2ac285983953ed9ecd4b5d29e81e5556a7f7fd1e5ee98c5623612efa03557770`、`06b6e133b45e0ece5534bccaeeeddd4577aaaf32387fde46372b4f8a428827c8`、`8438c44cc4fd4a5cdc95420fae3de8eee3db2179493e3a202195f21ab3473390`。

### 词表、长度预检、GPU Smoke 与正式缓存

- 特殊 Token 共 3,626 个：三层各 1,024 个 SID Token、512 个 Dedup Token、32 个 Geohash 字符 Token 和 10 个结构 Token。Qwen3-0.6B Tokenizer 从 151,669 扩展至 155,295，全部新增 Token 均作为普通原子 Token；扩展后 Tokenizer SHA256 为 `63cb4e325becc8e1adb5dbe0d82d507215a908e3a15d2e5bb620ef7b68f30470`，映射文件 SHA256 为 `b4869d26c2f7840a8239e27eb443730db89ef5016029c7623f17f75d5e826ae0`。

| 长度 | P50 | P90 | P95 | P99 | P99.9 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Input | 145 | 307 | 315 | 347 | 456 | 947 |
| Target | 11 | 12 | 12 | 12 | — | 12 |
| Total | 156 | 318 | 326 | 358 | 467 | 958 |

- Train+Valid 共 3,001 条总序列超过 512 Token，占 0.036670%；Assistant 目标在 `cutoff_len=512` 下截断数为 0，因此正式配置固定为 512。
- 正式 packed Cache 为 Train 2,926,615 行、Validation 214,227 行，均为固定 512 Token；`datasets.load_from_disk` 已核对 split、行数、标准列以及首/中/末样本的定长和非空 Assistant label。Cache Manifest SHA256 为 `8deaa6fc4c4c34c10a3a3140a8861a1c01e6e823aa8da2d7622789d547552fe1`。
- 固定前 10,000 条 Train 和前 2,000 条 Valid 源样本分别打包为 3,890/715 条序列；在 RTX A6000 上完成 20/20 个全参数 BF16 更新 step。首末训练 Loss 为 19.0392/6.6339，step 10/20 的 Eval Loss 为 6.862313/6.383718，总 Train Loss 为 8.523848；Loss 和 Gradient Norm 均为有限值，无 OOM、NaN 或异常退出。
- `checkpoint-10` 与 `checkpoint-20` 的模型、优化器、调度器、随机状态和 Trainer 状态均已核对完整。Smoke 运行耗时 169.38 秒，处理 10,240 个输入 Token；验证完成后已清理 16GB Smoke 权重，仅保留可复现配置和 48MB 固定 Smoke Cache。
- 首次数据运行在第一行发现历史时间为带时区 ISO 格式后立即失败，未写出正式产物；修正为统一的北京时间解析后全量构建成功。首次缓存尝试因受限沙箱禁止 multiprocessing 本地 socket，在格式转换开始时失败；同一命令在允许本机多进程通信的环境中成功完成，失败尝试不作为正式结果。

### 配置、命令与环境

~~~bash
python scripts/genpoi/build_sft_data.py \
  --orders-dir data/beijing_order_clean_20260701_20260714_history10_20260401_20260630_json \
  --pid-mapping outputs/pid/genpoi/GenPOI-BGE-M3-GeoPE-1024x3-e20-G6-Dedup/poi_pid_mapping.parquet \
  --pid-manifest outputs/pid/genpoi/GenPOI-BGE-M3-GeoPE-1024x3-e20-G6-Dedup/final_pid_manifest.json \
  --output-dir data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1 \
  --train-start 2026-07-01 \
  --train-end 2026-07-12 \
  --valid-date 2026-07-13 \
  --test-date 2026-07-14 \
  --history-start 2026-04-01 \
  --history-end 2026-06-30 \
  --sid-codebook-size 1024 \
  --geohash-length 6 \
  --max-history-events 10

python scripts/pid/prepare_vocab.py \
  --model-dir models/Qwen3-0.6B \
  --tokens data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/special_tokens.json \
  --output-dir models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --expected-token-count 3626 \
  --mapping-filename poi_token_mapping.json \
  --schema-version genpoi-vocab-v1

python scripts/sft/validate_tokenization.py \
  --model-dir models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --train-file data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/train.jsonl \
  --valid-file data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/valid.jsonl \
  --output-dir data/sft/tokenized/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1 \
  --dataset-dir configs/sft \
  --cutoff-len 512 \
  --batch-size 4096 \
  --workers 4 \
  --preprocessing-batch-size 1000 \
  --train-dataset genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_train \
  --valid-dataset genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_valid \
  --mapping-filename poi_token_mapping.json \
  --smoke-train-rows 10000 \
  --smoke-valid-rows 2000

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  configs/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_smoke.yaml

bash launchers/run_train_genpoi_sft_4x6000d_3epoch.sh
~~~

- 单卡正式配置为全参数 BF16、packing、`cutoff_len=512`、关闭 gradient checkpointing；单卡 Batch 16、梯度累积 32、全局 Batch 512，训练 3 epoch。
- 四卡 RTX PRO 6000D 入口保持相同数据、模型、精度和学习率；单卡 Batch 16、梯度累积 8，`16 × 8 × 4 = 512`，因此和单卡入口保持相同全局 Batch。启动前会检查当前进程恰好可见 4 张 GPU 并输出显卡型号与显存。
- 保存和完整 Validation 均按 epoch 执行，`save_total_limit=3`；训练成功后 `epoch_checkpoints.json` 会显式记录 epoch 1/2/3 对应的标准完整 checkpoint。
- 单卡 dry-run 已通过，解析结果为 `16 × 32 × 1 = 512`；四卡 RTX PRO 6000D dry-run 也已通过，解析结果为 `16 × 8 × 4 = 512`。两者均为 3 epoch、`save_strategy=epoch`、`eval_strategy=epoch`；单卡临时解析配置 SHA256 为 `4e9b1a7953ff17bf7ff492d4b35ba46ae9f6c7c2d44a2c29652329fd3ede4ddb`，临时 dry-run 目录均在验证后清理。
- 环境为 Python 3.10.20、LLaMA-Factory 0.9.4、Transformers 4.52.4、PyTorch 2.9.1+cu128；GPU smoke 使用 NVIDIA RTX A6000 48GB。代码基线提交为 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 GenPOI 数据、词表和训练入口实现的未提交工作树。

### 产物、结论与下一步

- 全量数据位于 `data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/`，扩词表模型位于 `models/Qwen3-0.6B-GenPOI-Vocab-v1/`，正式缓存位于 `data/sft/tokenized/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/`。
- 正式配置为 `configs/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1.yaml`；保留的正式平台入口为四卡 RTX PRO 6000D 的 `launchers/run_train_genpoi_sft_4x6000d_3epoch.sh`。数据、唯一 PID、原子 Token、Assistant-only label、缓存加载、实际 GPU 计算和训练参数均已形成可运行闭环。
- 当前 Loss 仅来自 20-step smoke，不能作为正式模型效果；尚无 GenPOI 三轮正式训练指标或 epoch checkpoint。GPU smoke 已通过，下一步可启动三轮正式训练，完成后先核验 epoch 1/2/3 checkpoint，再实现 GenPOI 对应的约束生成评测。
## EXP-20260804-02（GENPOI-SFT-EVAL-001）TCG+SSP 固定子集评测

### 目标与假设

- 目标：在与主任务和 TIGER 相同的固定 10,000 条 Validation 业务键集合上，对 GenPOI epoch 1/2/3 执行 TCG Trie 约束生成，并使用 SSP 预测的地理前缀深度限制候选。
- 状态：三个 checkpoint 均已完成 10,000 条评测，外层任务退出码为 0；此前 epoch 1 在 6,000 条处因同机 memory-cgroup OOM 风险人工停止，随后使用相同签名从原子分片恢复完成。

### 数据、约束与已完成准备

- Validation 来源为 `data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/valid.jsonl`，共 597,421 条，SHA256 为 `06b6e133b45e0ece5534bccaeeeddd4577aaaf32387fde46372b4f8a428827c8`。
- 固定子集通过相同 `order_id + searchid` 业务键与主任务 10,000 条对齐，目标 POI 不一致数为 0；GenPOI 子集 SHA256 为 `b4564ce9267821a896ceae683e964d27c6d492b6607bb7449349244bfe27402c`。
- TCG Trie 基于 2,337,178 条全局唯一 GenPOI PID 构建，manifest SHA256 为 `d50d16aebb0b180a8c58d215580388a2e4bd927bdcfee8637be73d8570d41130`；加载 Trie 增加约 110MB 主存。
- SSP 使用 `gamma=2`，10,000 条预测已全部完成且未读取目标字段；预测前缀长度 1/2/3 的数量为 3,777/6,207/16，预测文件 SHA256 为 `798e7b262cc41081569cb97d2b53f2ddd822099eb0577fcc315645340dfca50d`。
- 三个 checkpoint 为 `checkpoint-5717/11434/17151`，对应 epoch 1/2/3，完整 Validation Loss 为 `0.31379724/0.23585363/0.22581725`。

### 实际运行配置与恢复过程

~~~bash
python scripts/sft/evaluate_retrieval.py \
  --mode valid-checkpoints \
  --valid-file data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoints \
    outputs/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-5717 \
    outputs/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-11434 \
    outputs/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-17151 \
  --expected-checkpoint-steps 5717 11434 17151 \
  --expected-checkpoint-epochs 1 2 3 \
  --tokenizer models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --trie-dir outputs/eval/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/trie \
  --ssp-predictions-dir outputs/eval/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/ssp_predictions \
  --output-dir outputs/eval/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3 \
  --num-beams 10 \
  --top-k 10 \
  --per-device-eval-batch-size 128 \
  --chunk-size 1000 \
  --cutoff-len 512
~~~

- batch 128 和 64 在未提交当前分片时发生 CUDA OOM，自动回退至 batch 32 后稳定；这两次显存探测与后续系统 memory-cgroup OOM 是不同事件。
- epoch 1 曾提交到 `6,000/10,000`，恢复前 `next_line=6000`、`chunk_count=6`；恢复时复用相同 checkpoint、Tokenizer、Trie、SSP 预测、固定子集和 evaluator 签名，没有从第 0 条重跑。
- 三组最终都由 10 个原子分片覆盖 10,000 条样本，实际 batch size 为 32；结构合法率、PID 有效率均为 100%，候选重复率为 0%。

### 核心指标

| Checkpoint | Epoch | Valid Loss | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | Valid PID Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-5717` | 1 | 0.313797 | 0.4206 | 0.6653 | 0.7320 | 0.7910 | 0.612685 | 1.0 |
| `checkpoint-11434` | 2 | 0.235854 | 0.4904 | 0.7373 | 0.8014 | 0.8567 | 0.681685 | 1.0 |
| `checkpoint-17151` | 3 | 0.225817 | **0.5085** | **0.7557** | **0.8208** | **0.8713** | **0.698751** | 1.0 |

- epoch 3 在固定子集的 HR@1/3/5/10 和 NDCG@10 上均为三轮最优；其 GID6/SID3/Final PID top-1 exact match 为 0.7392/0.5176/0.5085。
- epoch 3 吞吐为 5.18 samples/s，平均单样本耗时 192.92 ms，峰值显存为 24,324,041,728 bytes；SSP 预测前缀深度不读取目标字段。

### 产物、结论与限制

- 正式产物位于 `outputs/eval/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/`，包含固定子集、SSP 预测、Trie、三个完整 run、错误 Case 和汇总 JSON/CSV；外层 `.evaluation.exit=0`。
- 汇总 JSON/CSV SHA256 分别为 `a6665e82a5300bb16ca7ef9f5daf80a03f9ade04872cfc9b3d08a956741d0019` 和 `464923e5479d201b5b1b998aaadec7e9a1836191beb772664cd9fb13e0d10720`。
- 该实验只证明固定 10,000 条 Validation、Beam=10、当前 SSP `gamma=2` 下 epoch 3 最优；未执行完整 Validation、Test、其他 Beam 或 SSP 参数对比。

## EXP-20260805-10 GenPOI Centered GeoPE32 SID 修正版与唯一 PID

### 目标与假设

- 目标：修正旧 GeoPE32 SID 第一层码本利用率过低的问题，并在新 `1024×3 / epoch 20` SID 上沿用既有 GenPOI `Geohash6 → 条件 Dedup` 流程，生成全量一一对应的 POI PID。
- 假设：BGE-M3 全局公共方向经逐 POI GeoPE 旋转后会干扰粗层聚类；先做全量均值中心化和 L2 归一化，再使用 TIGER 稳定的 256 维 RQ-VAE 骨架，可恢复三层码本利用率。该修改不保证同时提高完整 SID 唯一率，因此继续使用 Geohash6 和局部 Dedup 保证最终标识唯一。
- 状态：256/512/1024 三组 20 epoch 训练、全量 SID 导出、新 1024 Geohash6 base PID 和最终 Dedup PID均已完成；所有正式任务退出码为 0。

### 数据、代码与环境

- 原始 BGE-M3 输入仍为 2,337,178 条北京 POI、1024 维；新 GeoPE 输入位于 `outputs/embeddings/beijing_poi_bge_m3_genpoi_geope32_centered/`，预处理为 `global_mean_center_l2`，32 个 GeoPE 锚点和二维分量旋转方法不变。
- POI ID SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`，新 SID 与 `data/beijing_poi_clean_20260715_json/` 的 16 个分片逐行核验一致。
- RQ-VAE Encoder/Decoder 为 `1024→512→256→512→1024`，latent 256，三层码本，重构使用 `element_mean` 且不重复归一化；codebook/commitment 权重为 `1.0/0.25`，学习率 `3e-4`，batch 4096，20 epoch，seed 42，500,000 条固定样本执行 20 次 Faiss GPU K-Means 初始化。Diversity Loss 未启用。
- 训练在 A100 上执行；PID 构建环境为 Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1。代码基线提交为 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含当前 GenPOI/PID 实现的未提交工作树。

### 配置与命令

~~~bash
bash launchers/run_train_genpoi_geope32_centered_a100_20epoch.sh

python scripts/pid/build_geohash.py \
  --sid-manifest outputs/sid/genpoi/bge_m3_geope32_centered_tiger_rqvae/a100_single/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3/evaluations/epoch_20/sid_manifest.json \
  --poi-data data/beijing_poi_clean_20260715_json \
  --geohash-length 6 \
  --order gid_sid \
  --output-dir outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6

python scripts/pid/build_dedup.py \
  --pid-manifest outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6/pid_manifest.json \
  --dedup-capacity 512 \
  --output-dir outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6-Dedup
~~~

- base PID顺序固定为 `[G1..G6,S1..S3]`；单例最终为九段，第十列保存 `-1` sentinel；仅对base PID碰撞POI追加第十段 `D` Token。

### 三容量 SID 指标

| 容量 | Val 重构 Cosine | 全量利用率 L1/L2/L3 | Distinct SID / ratio | 碰撞 POI / ratio | P99 / Max | Prefix Micro Purity L1/L2/L3 |
|---|---:|---:|---:|---:|---:|---:|
| `256×3` | 56.9772% | 100%/100%/100% | 879,581 / 37.6343% | 1,804,574 / 77.2117% | 23 / 790 | 55.0577%/65.0139%/82.9197% |
| `512×3` | 60.4367% | 100%/100%/100% | 1,267,483 / 54.2313% | 1,421,278 / 60.8117% | 12 / 585 | 59.3435%/68.7714%/89.6668% |
| `1024×3` | 63.5604% | 100%/100%/100% | 1,536,997 / 65.7629% | 1,113,076 / 47.6248% | 8 / 383 | 62.3882%/73.2099%/93.6951% |

- 新三容量的三层全量码本利用率均为 100%，1024 三层归一化熵为 96.61%/97.57%/97.53%，旧版第一层只使用 59/1024 个码的低利用率问题已经消失。
- 新 1024 的完整 SID 唯一率低于旧版的 73.1125%，重构 Cosine 也从旧输入空间的 82.97% 降到 63.56%；这反映中心化输入和重构损失尺度改变后，码本健康与细粒度重构之间的取舍，不能把本版本记为对旧 baseline 的全面提升。

### Geohash6 与最终唯一 PID

| 标识 | Distinct / ratio | 碰撞 POI / ratio | P99 / 最大桶 |
|---|---:|---:|---:|
| 三层 Semantic SID | 1,536,997 / 65.7629% | 1,113,076 / 47.6248% | 8 / 383 |
| `GID6 + SID` base PID | 1,862,523 / 79.6911% | 690,673 / 29.5516% | 5 / 281 |
| Final PID | 2,337,178 / 100% | 0 / 0% | 1 / 1 |

- Geohash6 使 422,403 条原 SID 碰撞 POI变为唯一，解决比例为 37.9492%；仍有216,018个碰撞base PID桶、690,673条POI需要Dedup。
- 1,646,505条单例保留九段PID；690,673条碰撞POI使用 `D0–D280`，未超过预留容量512。删除最终数组第十列可逐行恢复base PID。
- 独立全量验收重新加载 `[2337178,9]` base数组、`[2337178,10]` final数组和2,337,178行Parquet，重新计算Final PID distinct数为2,337,178；Parquet映射SHA256为 `2c43454de25c1e6ec2d0a8d2eebea22cca7b21520d919df37ee3a1fd62cafaea`。

### 产物、结论与下一步

- 新SID产物位于 `outputs/sid/genpoi/bge_m3_geope32_centered_tiger_rqvae/a100_single/`；1024 base PID位于 `outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6/`，最终唯一PID位于其 `-Dedup/` 目录。
- 本实验保留旧GenPOI SID/PID、SFT、checkpoint和评测产物，不覆盖旧 baseline。新1024是可用于后续对照的稳定码本版本；若继续训练，必须基于新映射重新生成SFT数据和扩词表，不能复用旧PID缓存。
- 下一步只在用户确认后沿用既有 GenPOI 数据构建代码，把PID映射输入替换为本实验的新 `poi_pid_mapping.parquet`，随后重新生成词表、tokenized cache和SFT训练配置。

## EXP-20260806-01 GenPOI Centered GeoPE32 全量 SFT 数据与缓存

### 目标与假设

- 目标：只替换为 `EXP-20260805-10` 冻结的新 PID 映射，复用既有 GenPOI 历史序列输入和时间切分，生成可直接用于三轮四卡 SFT 的全量 Messages 数据、packed Cache 和训练入口。
- 假设：新旧 PID 都使用六层 GID、三层 SID 和可选一层 Dedup，Token 集合不变，因此在完整 Token 清单与 Tokenizer 指纹一致时，可以复用 `models/Qwen3-0.6B-GenPOI-Vocab-v1/`，但旧 Messages 和旧 Tokenized Cache 不可复用。
- 状态：8,790,513 条 Messages、正式 packed Cache、固定 smoke Cache、6000D/A100 四卡三轮入口和 dry-run 均已完成；尚未启动本版本正式 SFT。

### 数据版本、代码与环境

- 订单输入为 `data/beijing_order_clean_20260701_20260714_history10_20260401_20260630_json/`；PID 输入为 `outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6-Dedup/poi_pid_mapping.parquet`，映射 SHA256 为 `2c43454de25c1e6ec2d0a8d2eebea22cca7b21520d919df37ee3a1fd62cafaea`。
- PID 序列保持 `[G1..G6,S1..S3,optional D]`；历史事件为 `GID → Query → POI PID`，当前请求为 `GID → Query`，Assistant 目标为目标 POI PID。历史窗口为 2026-04-01—06-30，最多保留最近 10 条且必须早于目标订单。
- Train/Valid/Test 严格按 2026-07-01—12、07-13、07-14 切分；空历史保留，空 Query 才过滤，本次空 Query 为 0。
- 运行环境为 Python 3.10.20、datasets 3.6.0、Transformers 4.52.4、LLaMA-Factory 0.9.4；缓存构建使用 CPU 和 OrangeFS，没有启动 GPU。代码基线提交为 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含当前 GenPOI/SFT 实现的未提交工作树。

### 配置与命令

~~~bash
python scripts/genpoi/build_sft_data.py \
  --orders-dir data/beijing_order_clean_20260701_20260714_history10_20260401_20260630_json \
  --pid-mapping outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6-Dedup/poi_pid_mapping.parquet \
  --pid-manifest outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6-Dedup/final_pid_manifest.json \
  --output-dir data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2 \
  --train-start 2026-07-01 \
  --train-end 2026-07-12 \
  --valid-date 2026-07-13 \
  --test-date 2026-07-14 \
  --history-start 2026-04-01 \
  --history-end 2026-06-30 \
  --max-history-events 10 \
  --geohash-length 6 \
  --sid-codebook-size 1024

python scripts/sft/validate_tokenization.py \
  --model-dir models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --train-file data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/train.jsonl \
  --valid-file data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/valid.jsonl \
  --output-dir data/sft/tokenized/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2 \
  --dataset-dir configs/sft \
  --cutoff-len 512 \
  --batch-size 4096 \
  --workers 4 \
  --preprocessing-batch-size 1000 \
  --train-dataset genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_train \
  --valid-dataset genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_valid \
  --mapping-filename poi_token_mapping.json \
  --smoke-train-rows 10000 \
  --smoke-valid-rows 2000
~~~

- 正式训练配置为全参数 BF16、3 epoch、四卡、单卡 batch 16、梯度累积 8、全局 batch 512、`cutoff_len=512`、packing、只训练 Assistant、每个 epoch 保存和评估，最多保留三个 checkpoint。

### 数据与缓存指标

| 产物 | Train | Validation | Test |
|---|---:|---:|---:|
| Messages JSONL | 7,586,410 | 597,421 | 606,682 |
| Packed Cache | 2,935,702 | 214,860 | 不缓存 |
| Smoke Packed Cache | 3,903 | 718 | 不缓存 |

- 历史覆盖率为 72.0157%，平均历史长度为 4.9153；Messages manifest SHA256 为 `93644f70b407ae47ab6e32aee228aabcb5626b284de8b73901aac0f53e38394f`。
- 新版 Token 清单仍为 3,626 个 Token，`special_tokens.json` SHA256 为 `cbe3977a1ac52ccbb69cb066b8926db53fd1dbd9b3c58992b32b816076d50404`，完整 Tokenizer 指纹为 `63cb4e325becc8e1adb5dbe0d82d507215a908e3a15d2e5bb620ef7b68f30470`，与旧 GenPOI 词表契约一致。
- 输入 Token 长度 P50/P90/P95/P99/P99.9/Max 为 145/308/316/348/457/949；总长度为 156/319/327/359/468/960。3,142 条输入超过 512，但 Assistant 目标截断数为 0。
- 正式 Cache manifest SHA256 为 `ca124c48b08e1ffb2fc8db84e052f9902ae3c32c9084e58383f442dc552b32a3`。重新加载 Cache 后对 Train/Validation 首、中、末样本核验：四个序列字段长度均为 512、监督标签非空，多模态占位字段为 null。

### 产物、结论与下一步

- Messages 数据位于 `data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/`，约 11GB；正式 Cache 位于同名 `data/sft/tokenized/` 目录，约 32GB。旧 GenPOI 数据、缓存、checkpoint 和评测均未覆盖。
- 6000D、A6000 和 A100 入口分别为 `launchers/run_train_genpoi_centered_sft_4x6000d_3epoch.sh`、`launchers/run_train_genpoi_centered_sft_4xa6000_3epoch.sh` 与 `launchers/run_train_genpoi_centered_sft_4a100_3epoch.sh`；三者 `bash -n`、可执行权限、全局 batch 校验及 dry-run 均通过。
- 结论：本版本已达到正式四卡 SFT 启动条件；同一数据、Tokenizer、模板、cutoff 和 packing 配置可直接复用该 Cache，不因 GPU 类型、epoch、batch 或学习率变化重新构建。
- 下一步：按实际申请到的 GPU 类型只启动一个四卡入口，核验训练日志读取的是本版本 Cache，并在 epoch 1/2/3 分别形成完整 checkpoint 后再开始固定 10,000 条 Validation 评测。

## EXP-20260807-04 GenPOI Centered 三轮 SFT 与固定 10,000 条 TCG+SSP 评测

### 目标、假设与数据版本

- 目标：完成 Centered GeoPE32 + TIGER-RQVAE 新 PID 的三轮 Qwen3-0.6B 全参数 SFT，并严格复用旧版 GenPOI 的固定 10,000 条 Validation、Query-only SSP `gamma=2` 和 TCG Trie 约束 Beam=10 协议，对比 epoch 1/2/3 及旧版最佳结果。
- 假设：Centered 版本修复第一层码本利用率后，若新 SID 更适合生成模型学习，则在相同业务键、GID、Query、历史长度、Tokenizer 和约束解码下，SID3/Final PID 及 HR/NDCG 应提高；SSP 输入和预测不应因 SID/PID 变化而改变。
- SFT 数据为 `data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/`，Train/Valid/Test 为 7,586,410/597,421/606,682，packed Train/Validation 为 2,935,702/214,860。PID 映射 SHA256 为 `2c43454de25c1e6ec2d0a8d2eebea22cca7b21520d919df37ee3a1fd62cafaea`，Tokenizer JSON SHA256 为 `dbc99c7765ca0753a38a6035f3fd7853adc486875ccceca11319f5e70164f85e`。
- 固定子集按 V1 参考集的 `order_id + searchid` 精确对齐，业务键 SHA256 为 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`，目标 POI 错配为 0；Centered 子集 SHA256 为 `59384c3ac10a25faf2d29459fb8362861dbb548f3d34d4cfe1aeb6697445bbdb`。

### 代码状态、训练与环境

- 仓库基线提交为 `e4515195097dcb10aac0f3281ee7ddc0e482886c`；训练和评测使用包含既有目录整理、Centered 数据入口及本次评测源码指纹路径修正的未提交工作树，未创建 commit。
- 训练解析配置为四卡全参数 BF16、packing、cutoff 512、每卡 batch 16、梯度累积 8、全局 batch 512、学习率 `5e-5`、cosine scheduler、warmup 0.03、3 epoch、每个 epoch 保存和评估。四卡 A100 共完成 17,202 step，`train_loss=0.360075`，运行时间 20:41:11.29。
- 平台外层在 Trainer 完成后显示 failed，但本地训练日志明确记录 `Training completed`、最终模型保存、训练指标、Loss 图和四卡 NCCL 正常销毁，没有 Trainer traceback。`checkpoint-5734/11468/17202` 分别为 epoch 1/2/3，均包含约 2.398GB `model.safetensors`、优化器、调度器、四卡随机状态、Trainer 状态及一致 Tokenizer；当前 A100 入口 `bash -n` 退出码为 0，因此三个 checkpoint 可直接使用，无需重训。
- 评测在本地 NVIDIA RTX A6000 48GB、Python 3.10.20、PyTorch 2.9.1+cu128、Transformers 4.52.4 上执行。目录整理后，评测器记录源码指纹的路径从旧 `generative_eval.py/pid_trie.py` 同步为 `sft/evaluation.py` 与 `pid/trie.py`，21 项相关合成测试通过。

### 约束准备与命令

- TCG Trie 由新 Centered Final PID 构建，包含 2,337,178 个叶子、6,424,459 个节点，加载内存 121,488,768 bytes，manifest SHA256 为 `040f970fcce9547b63c0eadacc7d4d7f66f98c015de14fc8583de896115ac68b`。
- SSP 沿用旧版冻结 BGE-M3 proximity head，`gamma=2`，输入字段仅为当前 Query，`target_fields_used=[]`。Centered 与旧版 proximity Train/固定 Valid Parquet 的 SHA256 分别完全一致；重新生成的 10,000 条预测中预填深度 1/2/3 的数量为 3,777/6,207/16，预测 Parquet SHA256 仍为 `798e7b262cc41081569cb97d2b53f2ddd822099eb0577fcc315645340dfca50d`。

~~~bash
bash launchers/run_train_genpoi_centered_sft_4a100_3epoch.sh

python scripts/pid/build_trie.py \
  --pid-mapping outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6-Dedup/poi_pid_mapping.parquet \
  --pid-manifest outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6-Dedup/final_pid_manifest.json \
  --tokenizer models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --output-dir outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/trie

python scripts/genpoi/build_proximity_data.py \
  --train-file data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/train.jsonl \
  --valid-file outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/validation_subset_10000.jsonl \
  --output-dir data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/proximity_bge_m3_v1

python scripts/genpoi/predict_proximity.py \
  --data-dir data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/proximity_bge_m3_v1 \
  --head-dir outputs/genpoi/proximity_head_bge_m3_v1 \
  --output-dir outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/ssp_predictions

python scripts/sft/evaluate_retrieval.py \
  --mode valid-checkpoints \
  --valid-file data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoints \
    outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/checkpoint-5734 \
    outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/checkpoint-11468 \
    outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/checkpoint-17202 \
  --expected-checkpoint-steps 5734 11468 17202 \
  --expected-checkpoint-epochs 1 2 3 \
  --tokenizer models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --trie-dir outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/trie \
  --ssp-predictions-dir outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/ssp_predictions \
  --output-dir outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3 \
  --num-beams 10 --top-k 10 --per-device-eval-batch-size 32 \
  --chunk-size 1000 --cutoff-len 512 --skip-data-hash
~~~

- 正式命令前已完成全量源 Validation SHA256、固定子集、Prompt、三个 checkpoint、Tokenizer、Trie 和 SSP 契约核验，并依次执行 2 条 TCG-only 与 2 条 SSP+TCG GPU smoke；正式命令的 `--skip-data-hash` 只跳过已完成的 597,421 条源文件重复哈希，不跳过固定子集、checkpoint、Trie 或 SSP 指纹。

### 核心指标

| Checkpoint | Epoch | Valid Loss | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Valid PID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-5734` | 1 | 0.298704 | 43.56% | 67.84% | 74.89% | 80.49% | 43.56% | 57.9127% | 60.8250% | 62.6556% | 100% |
| `checkpoint-11468` | 2 | 0.225725 | 50.04% | 74.54% | 81.12% | 86.89% | 50.04% | 64.5773% | 67.3003% | 69.2006% | 100% |
| `checkpoint-17202` | 3 | 0.217134 | **51.98%** | **76.31%** | **82.77%** | **87.83%** | **51.98%** | **66.4127%** | **69.0840%** | **70.7430%** | 100% |

| Epoch | 相对旧版 HR@1 | 相对旧版 HR@3 | 相对旧版 HR@5 | 相对旧版 HR@10 | 相对旧版 NDCG@10 |
|---|---:|---:|---:|---:|---:|
| 1 | +1.50pp | +1.31pp | +1.69pp | +1.39pp | +1.3871pp |
| 2 | +1.00pp | +0.81pp | +0.98pp | +1.22pp | +1.0321pp |
| 3 | **+1.13pp** | +0.74pp | +0.69pp | **+0.70pp** | **+0.8679pp** |

- epoch 1/2/3 的 GID6/SID3/base PID9/Final PID top-1 exact match 分别为 67.74/47.19/45.92/43.56%、72.91/53.26/52.18/50.04%、74.00/55.21/54.11/51.98%。相较旧版 epoch 3，GID6 只提高 0.08pp，而 SID3 提高 3.45pp，说明主要收益来自新语义 SID，而非完全相同的 SSP 地理预测。
- 三轮均返回每条 10 个无重复候选，结构合法率和 PID 有效率均为 100%，没有 OOM 回退或恢复事件。纯推理吞吐为 5.42/5.27/5.33 samples/s，峰值显存约 24.32GB。

### 产物、结论与下一步

- 训练目录为 `outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/`；正式评测目录为同名 `outputs/eval/` 路径，包含固定子集、Centered Trie、SSP 预测、三轮原子 run、错误 Case 和汇总 JSON/CSV。汇总 JSON/CSV SHA256 分别为 `72003faa9efc8b9ccebc5b4ff5bb514d6866eda1277cf79a676811aa887f3ae2`、`3cf958b7799e327442430cffb9be3eb49e7024a520fc4f3ff918034848b6e1d2`，正式进程退出码为 0。
- 结论：冻结 `checkpoint-17202` 为 Centered GenPOI 当前 Validation 最优 checkpoint。Centered 版本在三轮各自同 epoch 上均优于旧版，epoch 3 的 HR@1/HR@10/NDCG@10 为 51.98%/87.83%/70.7430%，并刷新当前四条复现方法中的 HR@1 和 NDCG@10 最优值；训练平台的外层 failed 不影响可用结果，无需重训。
- 本实验仍只覆盖固定 10,000 条 Validation、Beam=10 和 SSP `gamma=2`，不替代完整 Validation/Test，也不证明 Centered 输入的所有设计都优于旧版。复现闭环到此完成，下一步回到 Query-Augmented Relational SID 创新线，不自动启动其他 GenPOI 训练或 Test 评测。

## EXP-20260808-03 Centered GenPOI 固定 10,000 条逐层 Teacher-Forcing 诊断

### 目标、假设与数据版本

- 目标：将 Centered GenPOI epoch 3 加入既有 TIGER/GNPR 固定 10,000 条逐层诊断，区分前置 GID、三层 semantic SID 和可选 Dedup 对完整 PID 学习难度的贡献；本实验不重训、不运行 Beam、Trie 或 SSP。
- 假设：若 Centered GenPOI 的主要优势来自更易预测的三层 SID，则在正确 GID6 作为 gold prefix 时，S1/S2/S3 的逐位置及累计正确率应不低于 TIGER/GNPR；若完整 PID 仍受地理路径限制，GID 越细的后层及 GID6 累计正确率会成为主要损失来源。
- 数据为 Centered 正式评测的 `validation_subset_10000.jsonl`，SHA256 为 `59384c3ac10a25faf2d29459fb8362861dbb548f3d34d4cfe1aeb6697445bbdb`。TIGER、GNPR 与 GenPOI 的 10,000 个 `sample_id` 顺序 SHA256 均为 `a81c7d91f82a3e6df89002b8eb0c320625dc601e33518e1d36e869c4487c25c1`，业务样本严格同序。
- checkpoint 为 `checkpoint-17202`、epoch 3，模型 SHA256 为 `1d0cdf6b35dfdfbd7f5e3f7b556524e614a095ed0288cff8e300cb3a2ddfe348`；Tokenizer JSON SHA256 为 `dbc99c7765ca0753a38a6035f3fd7853adc486875ccceca11319f5e70164f85e`。

### 评价口径、代码与环境

- 每条样本使用与 SFT 完全一致的 user Prompt 和 Assistant gold PID。模型一次前向后，在每个目标位置使用前一位置 logits，对完整词表统计 gold Token 的 Top-1、Top-10 和 NLL；后层始终看到正确 gold prefix，因此不包含前层自由生成错误传播。
- Centered GenPOI 的单例目标为 `G1..G6,S1..S3,EOS`，Dedup 目标额外包含 `D`。`G1..G6` 单独记录逐位置及累计上下文准确率；`S1..S3` 作为 semantic SID，分别位于目标索引 6/7/8。三层 SID 指标均条件于正确 GID6，适合判断语义码本在当前完整方法中的可学习性，但不能直接解释为不带地理前缀时的纯 SID 能力。
- Top-10 表示每个 gold-prefix 位置的真实 Token 是否进入完整词表 Top-10；“完整 identifier Top-10 all”要求所有标识位置都分别进入 Top-10，不等价于 Beam=10 召回率。Wrapper/EOS 只进入完整序列指标，不进入 identifier；GenPOI Assistant 本身没有 TIGER/GNPR 的 `<TARGET_POI>` wrapper。
- 代码基线提交为 `e4515195097dcb10aac0f3281ee7ddc0e482886c`，运行使用增加 GenPOI 目标契约、GID 累计指标和合成测试的未提交工作树。5 项专项测试、2 条真实 GPU smoke 均通过。
- 环境为本地 NVIDIA RTX A6000 48GB、Python 3.10.20、PyTorch 2.9.1+cu128、Transformers 4.52.4；batch 32，纯前向 119.31 秒，83.81 samples/s，峰值显存 6,806,810,112 bytes，未触发 OOM 回退。

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/sft/diagnose_sid_teacher_forcing.py \
  --method genpoi \
  --data-file outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/validation_subset_10000.jsonl \
  --checkpoint outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/checkpoint-17202 \
  --tokenizer models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --output-dir outputs/eval/sid_teacher_forcing_fixed10k_v1/genpoi_centered_e3 \
  --expected-epoch 3 --batch-size 32 --checkpoint-rows 1000
~~~

### 核心指标

| GenPOI gold-prefix 位置 | Top-1 | Top-10 | NLL |
|---|---:|---:|---:|
| GID 第 1 层 | 99.99% | 100.00% | 0.0003 |
| GID 第 2 层 | 100.00% | 100.00% | 0.0000 |
| GID 第 3 层 | 99.69% | 100.00% | 0.0118 |
| GID 第 4 层 | 91.90% | 99.96% | 0.2325 |
| GID 第 5 层 | 88.85% | 99.52% | 0.3398 |
| GID 第 6 层 | 86.03% | 98.88% | 0.4498 |
| SID 第 1 层（给定正确 GID6） | **83.97%** | 98.29% | 0.5323 |
| SID 第 2 层（给定正确 GID6+S1） | **88.67%** | 97.89% | 0.4413 |
| SID 第 3 层（给定正确 GID6+S1+S2） | **89.27%** | 97.57% | 0.4376 |
| Dedup（2,915 条） | 86.00% | 99.79% | 0.3987 |

| Gold-prefix 累计全位置正确 | TIGER | GNPR | Centered GenPOI |
|---|---:|---:|---:|
| SID 前 1 层 Top-1 | 75.06% | 53.69% | **83.97%** |
| SID 前 2 层 Top-1 | 61.75% | 49.31% | **76.03%** |
| SID 前 3 层 Top-1 | 53.98% | 48.50% | **69.90%** |
| 完整 identifier Top-1 | **50.76%** | 48.40% | 50.00% |
| 完整 identifier Top-10 all | 92.04% | 86.63% | **94.47%** |

- GenPOI GID1—GID6 的累计 Top-1 依次为 99.99/99.99/99.68/91.66/82.86/73.12%，精细地理层是完整路径的第一处主要损失。该 GID6 累计结果与正式 SSP+TCG 评测的 GID6 top-1 74.00% 接近，两个独立评价口径相互支持。
- 在 gold GID6 条件下，GenPOI S1/S2/S3 单层 Top-1 为 83.97/88.67/89.27%，三层累计为 69.90%，均高于 TIGER/GNPR 的对应累计结果。其 S1/S2 NLL 为三者最低，S3 NLL 0.4376 略高于 GNPR 的 0.4108。
- GenPOI 完整 PID Top-1 all 为 50.00%，略低于 TIGER 50.76%、高于 GNPR 48.40%；说明强三层 SID 并未无损转化为完整序列优势，GID6 路径和可选 Dedup 消耗了部分收益。GenPOI 在正式 SSP+TCG Beam 评测中 HR@1 达到 51.98%，约束解码又补回了这部分路径合法性和搜索收益。
- 2,915 条 Dedup 与 7,085 条单例的完整 identifier Top-1 分别为 49.57%/50.18%，当前固定集上 Dedup 只带来次要差距；这不改变随机 Dedup 缺乏实体语义和增量稳定性的长期问题。

### 产物、结论与下一步

- 正式结果位于 `outputs/eval/sid_teacher_forcing_fixed10k_v1/genpoi_centered_e3/result.json`，SHA256 为 `287590c6e99ba43640f9f3634220e92143f513888fa6736e9e0ae0a729effd9c`；进程状态为 `completed`、退出码 0。
- 结论：Centered GenPOI 在正确地理前缀下拥有三者最易学习的 semantic SID，GNPR 的根 SID 弱预测问题未在 GenPOI 上复现；但 GenPOI 把主要难度前移到了细粒度 GID4—GID6，完整 identifier 与 TIGER 仍处于同一水平。后续 RQ-KMeans/RQ-VAE 同输入选择应把“给定统一上下文后的 SID1/2/3 可预测性”与完整 GID+SID 路径同时报告，不能只依据 SID 唯一率或约束 Beam 最终 HR。
- 本实验仍只覆盖固定 10,000 条、epoch 3 和 gold-prefix 诊断，不包含自由生成错误传播，也不能隔离 GenPOI 高 SID 准确率中由 gold GID 提供的条件信息。若后续需要严格比较纯 SID，本实验建议的最小补充是统一目标结构或训练一个不在 Assistant 前置 GID 的对照，而不是直接把当前 83.97% 与 TIGER 75.06%解释为码本绝对提升。

## EXP-20260811-01 复杂地理 Query Validation 10k 三方法专项评测

### 目标、假设与数据版本

- 目标：从完整 597,421 条 Validation 中冻结新的 10,000 条复杂地理 Query 专项集，使用各方法已冻结的 epoch-3 checkpoint 和原论文复现解码协议，比较 TIGER、GNPR-SID 与 Centered GenPOI，检验 GenPOI 的地理表示和 SSP+TCG 是否在更相关的 Query 分布上体现优势；本实验不重训、不读取 Test。
- 假设：若 Centered GenPOI 的地理模块有效，则在显式包含附近、距离、方位、路口/对面、内外/入口或地铁出口关系的 Query 上，其 HR@3/5/10 或 NDCG@10 相对 TIGER 的优势应比原随机固定 10k 更明显；若只提高合法率而不能改善排序，则 TIGER 仍会在主要召回指标上领先。
- Query 取自 `data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/valid.jsonl` 的 `<CURRENT><QUERY>`，先做 NFKC、转小写和去空白，再要求长度不低于 6 且至少命中一类显式地理关系规则。完整 Validation 中有 15,029 条候选，按 `SHA256(order_id\0searchid\0target_poi_id)` 从小到大稳定无放回抽取 10,000 条；筛选不使用任何模型输出，也不使用用户—目标 GID 公共前缀标签。
- 专项集 JSONL SHA256 为 `3ab77534d7ab4e289dd34fd0c10cb709f501143a3d0d782787873020be5c6a0f`，manifest SHA256 为 `1ff12b1cd3e3c65819e941c74c5554744fdcd4aeaf833296efef86b2c9a4f710`。TIGER、GNPR-SID 和 GenPOI 按 `order_id + searchid + target_poi_id` 精确对齐，业务键顺序 SHA256 均为 `4f83210dddb5087bbac3122b3b9646a2a933517b9b4311bb57dc3cc6a1a15f07`，目标 POI 错配为 0。
- 选中样本命中方位/交通出口/距离/路口或对面/内外入口/附近规则的次数为 6,260/3,106/670/422/157/100；规则可重叠。用户—目标 GID 公共前缀长度 2/3/4/5/6 的样本数为 115/4,043/3,958/1,785/99，该分布只用于审计，不参与抽样。

### 代码状态、配置与环境

- 运行时 Git HEAD 为 `7dd52b3437a9db41834aef638e446010d0df3610`，工作树非干净。本实验新增 `src/poi_gr/sft/geo_query_subset.py`、`scripts/sft/build_geo_query_subset.py` 和对应合成测试；TIGER/GNPR 评测器增加显式 `--final-checkpoint-only`，默认三 checkpoint 协议不变。目录整理后 TIGER 的方法源码指纹路径同步为 `src/poi_gr/methods/tiger/eval.py`。
- 服务器 Python/Transformers 冷启动会触发 Auto 类遍历完整模型注册表；评测共享加载器在核验三份配置均固定为 Qwen2 fast tokenizer 和 `Qwen3ForCausalLM` 后改为直接具体类加载。该变化不修改 tokenizer 文件、Prompt 模板、模型权重或生成参数。
- 三者均为 epoch 3、Beam=10、返回 10 个候选、BF16、batch 32、chunk 1,000、cutoff 512。TIGER/GNPR 保持无约束生成并将非法 ID 留在原 Beam 排名；GenPOI 保持冻结 Centered Trie、Query-only 7 类 proximity head、`gamma=2` SSP 和 TCG 约束。
- GenPOI 新 SSP 预测 10,000 条全部完成，`target_fields_used=[]`；预测 λ=3/4 为 6,223/3,777 条，对应预填 GID 1/2 层，预测 Parquet SHA256 为 `f51cc07647aead12893a31eb8e6524a3b6d114a08113f73efa1d960292ba066c`。首次误调用北京序分类头入口时被 `schema_version` 在加载权重前拒绝，随后按 Centered 原协议改用 `predict_proximity.py`，未改动分类头。
- 环境为 NVIDIA RTX A6000 48GB、Python 3.10.20、PyTorch 2.9.1+cu128、Transformers 4.52.4。三次正式推理退出码均为 0，未发生 OOM、batch 回退或断点恢复；TIGER/GNPR/GenPOI 吞吐为 6.96/6.65/4.80 samples/s，峰值 PyTorch allocated 显存约 24.26/24.46/24.37GB。

~~~bash
python scripts/sft/build_geo_query_subset.py \
  --valid-file data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/valid.jsonl \
  --output-dir outputs/eval/geo_query_complex_validation_10k_v1 \
  --subset-size 10000 --min-query-length 6

python scripts/genpoi/predict_proximity.py \
  --data-dir outputs/eval/geo_query_complex_validation_10k_v1/genpoi_centered_e3/proximity_eval_data \
  --head-dir outputs/genpoi/proximity_head_bge_m3_v1 \
  --output-dir outputs/eval/geo_query_complex_validation_10k_v1/genpoi_centered_e3/ssp_predictions \
  --encode-batch-size 256 --encode-buffer-size 8192

python scripts/tiger/evaluate_retrieval.py ... \
  --checkpoints outputs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-16713 \
  --expected-checkpoint-steps 16713 --expected-checkpoint-epochs 3 \
  --final-checkpoint-only --num-beams 10 --skip-data-hash

python scripts/gnpr/evaluate_retrieval.py ... \
  --checkpoints outputs/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/checkpoint-16167 \
  --expected-checkpoint-steps 16167 --expected-checkpoint-epochs 3 \
  --final-checkpoint-only --num-beams 10 --skip-data-hash

python scripts/sft/evaluate_retrieval.py --mode valid-checkpoints ... \
  --checkpoints outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/checkpoint-17202 \
  --expected-checkpoint-steps 17202 --expected-checkpoint-epochs 3 \
  --num-beams 10 --top-k 10 --skip-data-hash
~~~

### 核心指标

| 方法 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Valid ID/PID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TIGER epoch 3 | **64.52%** | 79.79% | 83.86% | 87.17% | **64.52%** | **73.6083%** | 75.2954% | 76.3827% | 64.126% |
| GNPR-SID epoch 3 | 61.26% | 75.87% | 79.44% | 82.34% | 61.26% | 69.9319% | 71.4098% | 72.3661% | 73.414% |
| Centered GenPOI epoch 3 | 63.63% | **80.36%** | **84.80%** | **88.97%** | 63.63% | 73.5806% | **75.4148%** | **76.7836%** | **100%** |

| Centered GenPOI 相对 TIGER | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| 原随机固定 Validation 10k | +0.11pp | +0.31pp | +0.51pp | +0.67pp | +0.3240pp |
| 复杂地理 Query Validation 10k | **-0.89pp** | **+0.57pp** | **+0.94pp** | **+1.80pp** | **+0.4009pp** |

- 三种方法在专项集上的 HR@1 均比原随机固定 10k 高 11.60—12.65pp，说明显式地理关系和更长文本同时提高了目标辨识度；该专项集是“地理相关”而非“更难”集合，不能把共同增益归因于 GenPOI。
- Centered GenPOI 的主要正证据出现在候选覆盖：相对 TIGER 的 HR@10 优势由 0.67pp 扩大到 1.80pp，HR@3/5 优势也扩大到 0.57/0.94pp，NDCG@10 保持第一；但 HR@1 反而低 0.89pp，NDCG@3 低约 0.03pp。当前证据支持 SSP+TCG 改善地理 Query 的 Top-K 覆盖和后段排序，不支持其已经解决 Top-1 精排。
- GenPOI GID6/SID3/base PID9/Final PID top-1 exact match 为 87.14/69.19/67.82/63.63%，完整结构及 PID 合法率均为 100%；相比原随机集的 GID6 74.00%，地理 Query 的更强位置线索确实被模型利用，但最终 Top-1 仍受 SID 和 Dedup 排序影响。
- GNPR 在专项集上的 HR@10 比原随机集下降 0.73pp，尽管 HR@1 上升 11.60pp；其无效候选率由原结果的约 21.19% 上升至 26.586%，仍明显弱于 TIGER 与 GenPOI，未显示 content-geo SID 对此类 Query 的额外收益。

### 产物、结论与下一步

- 全部产物位于 `outputs/eval/geo_query_complex_validation_10k_v1/`。TIGER/GNPR/GenPOI 汇总 JSON SHA256 分别为 `53320cff75bbf952f1b09efe93571c6ca9892003d628be08552b95ab6505e9b8`、`d11d307a64097a1f63e5bbb47a18a5fffad5c8e99eff8c261e0ac3fe1284a02e`、`2a05c19bdba7c3642538aa2677161d4bb57aec32262518fa37faf48170f6ea3b`；三组 run 均覆盖 10,000 条且状态为 `completed`。
- 结论：本实验给出“有限支持”。Centered GenPOI 在复杂地理 Query 上成为 HR@3/5/10、NDCG@5/10 最优完整方法，Top-10 相对 TIGER 的优势明显扩大；但 TIGER 仍是 HR@1 和 NDCG@1/3 最优，说明 GenPOI 的地理约束更像扩大正确候选覆盖，尚未转化为稳定 Top-1 优势。
- 本结果仍是北京同城 Validation，不能证明跨城泛化；规则会将部分含“东门”“A口”的 POI 名称视为地理约束，且三种完整方法的标识结构与解码约束不同。若要隔离地理约束本身，下一最小补充应在同一专项集只对 Centered GenPOI 做 TCG-only 与 SSP+TCG 配对，而不是继续扩充三方法矩阵。

## EXP-20260817-03 Centered GenPOI epoch 3 四类泛化 Validation 10k

### 目标、数据与代码状态

- 目标与假设：在四类冻结泛化集上检验 Centered GenPOI 的显式 GID PID、TCG 与 Query-only SSP 是否主要改善 Top-K 覆盖，尤其是长尾和冷目标；假设其相对 TIGER 的优势会集中在 HR@3/5/10，而非所有集合的 HR@1。
- 数据版本与 `EXP-20260817-01/02` 完全相同，四组各 10,000 条且三方法逐组业务键一致。SSP 仅从当前 Query 预测，冻结 head SHA256 为 `231b923f...f1e93`，没有使用目标字段。
- 代码状态：基线提交 `7dd52b3437a9db41834aef638e446010d0df3610`，运行使用未提交工作树。前三组在训练平台完成；cold-target 的旧 512 运行在 Prompt 校验阶段失败、未生成正式 run，随后使用当前 1024 安全协议在本机补跑。

### 配置、失败原因、修复与环境

- checkpoint 固定为 epoch 3 `checkpoint-17202`；TCG+SSP、`gamma=2`、Beam=10、Top-K=10、batch 32、chunk 1,000。前三组沿用历史 `cutoff_len=512`；cold-target 显式使用 1024，其他模型、Trie、SSP 和生成参数不变。
- cold-target 共 7/10,000 条超过 512，最大 566；第 63 条使旧 Source-prefix 截断删除 Assistant generation marker，前置校验报 `Assistant generation prompt 位置不正确`。增大到 1024 后 100 条本机 A6000 smoke 先行通过，随后正式 10,000/10,000 完成，结构与 PID 合法率均为 100%；正式结果 SHA256 为 `ccb6f200e1c9d350af81456d1f4fa8f6fe4fe2d35eaf57839dad69774e17c2cb`。
- `seen_query_unseen_pair` 与 `unseen_query_seen_target` 在训练平台单卡 RTX 6000D 完成，`long_tail_target` 同样在 6000D 完成；cold-target 在本机 RTX A6000 完成，峰值显存约 25.05GiB、生成耗时 1,941.69 秒、5.15 samples/s。

~~~bash
CUDA_VISIBLE_DEVICES=0 TMPDIR=/ofs/map_search/hudan/poi_genret/outputs/tmp/gcold \
python scripts/sft/evaluate_retrieval.py \
  --mode valid-checkpoints \
  --valid-file data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/valid.jsonl \
  --reference-validation-subset outputs/eval/generalization_validation_suite_10k_v1/cold_target_10000.jsonl \
  --checkpoints outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/checkpoint-17202 \
  --tokenizer models/Qwen3-0.6B-GenPOI-Vocab-v1 \
  --trie-dir outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/trie \
  --ssp-predictions-dir outputs/eval/generalization_validation_suite_10k_v1/paper_baselines/cold_target/genpoi_centered_e3/ssp_predictions \
  --output-dir outputs/eval/generalization_validation_suite_10k_v1/paper_baselines/cold_target/genpoi_centered_e3 \
  --num-beams 10 --top-k 10 --per-device-eval-batch-size 32 --chunk-size 1000 --cutoff-len 1024 --skip-data-hash
~~~

### 核心结果、审计与结论

| Validation 专项集 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 已见 Query / 未见配对 | **18.93%** | **41.87%** | **52.41%** | **63.89%** | **18.93%** | **32.2553%** | **36.6035%** | **40.3454%** |
| 未见 Query / 已见目标 | 49.07% | **66.49%** | **71.72%** | **76.74%** | 49.07% | 59.4048% | **61.5631%** | **63.2055%** |
| 长尾目标（Train 1—5） | 25.73% | **39.84%** | **46.79%** | **55.11%** | 25.73% | **33.9490%** | **36.8203%** | **39.5168%** |
| 冷目标（Train 0） | **7.28%** | **12.01%** | **15.39%** | **20.81%** | **7.28%** | **9.9946%** | **11.3793%** | **13.1261%** |

- 四组宏平均 HR@1/HR@10/NDCG@10 为 25.2525%/54.1375%/39.0484%，比 TIGER 高 0.26/2.7675/1.3535pp。GenPOI 在四组 HR@10/NDCG@10 均为第一；HR@1 只在已见 Query 新配对和冷目标第一，在新 Query 已见目标与长尾目标分别低 TIGER 0.52/0.28pp，符合“改善候选覆盖多于 Top-1 精排”的假设。
- 真实 Token 审计显示四组超 512 行数为 `0/11/6/7`，最大长度为 `501/644/605/566`。cold-target 已按 1024 完整补跑；另外两组历史 512 结果最多受 0.11/0.06pp 扰动，最坏界仍不足以翻转其与 TIGER 的 HR@1 排序或 Top-K 结论。
- 正式产物位于 `outputs/eval/generalization_validation_suite_10k_v1/paper_baselines/<subset>/genpoi_centered_e3/`。本结果支持显式地理结构与约束生成改善泛化 Top-K，但北京同城数据仍不能证明跨城泛化；严格发表表应统一把所有受影响单元重评为 1024。
