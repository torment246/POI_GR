# TIGER 复现实验

本文件集中记录 TIGER 在北京地图 POI 检索场景中的完整复现链路：BGE-M3 向量 → 三层 RQ-VAE SID → collision token 唯一 identifier → 最近 10 条历史序列 SFT → 无约束 Beam Search 评测。

## 当前结论

- 三容量 SID 对比选择 `1024×3 / epoch 20`，全量 SID 唯一率 71.6766%；追加 `C0–C305` 后 2,337,178 条四层 identifier 全局唯一。
- 历史序列数据、原子 Token、正式 packed Cache 和四卡 RTX PRO 6000D 三轮训练均已完成。
- 固定 10,000 条 Validation、Beam=10 下，epoch 3 的 HR@1/HR@10/NDCG@10 为 51.87%/87.16%/70.42%，为三个 checkpoint 最优。
- TIGER 评测不使用 Trie；epoch 3 的 Valid ID Rate 为 74.105%，非法 identifier 保留原 Beam 排名并按 miss 处理。

## 实验记录
## EXP-20260730-01 BGE-M3 TIGER SID 三容量全量对比

### 目标与假设

- 目标：以既有 BGE-M3 北京 POI 全量向量作为直接输入，复用已验证的 Vanilla RQ-VAE 结构和数据划分，在 20 epoch 下对比 `256×3`、`512×3`、`1024×3` 三种 TIGER Semantic ID 容量。
- 假设：BGE-M3 在固定订单向量召回上优于 Qwen3-Embedding-0.6B，其向量分布也可能提高 RQ-VAE 码本利用率和完整 SID 唯一率；输入编码器变化后学习率需要先做稳定性检查，不能直接沿用旧值。
- 状态：三组正式训练和 2,337,178 条 POI 的全量 SID 导出均已完成；最初沿用 `lr=1e-3` 的 256×3 运行发生码本塌缩，已明确标为失败诊断产物，未纳入最终对比。

### 数据、代码与环境

- Embedding：BGE-M3，shape `[2337178,1024]`，dtype `float16`，已归一化；输入 POI 数据 fingerprint 为 `d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071`，Embedding manifest signature 为 `2ee5349438f7d668d96f9dfc8d1e217142972e526860d417b327aafaeadbeaa4`。
- POI ID：2,337,178 行且无重复，SHA256 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`；全量向量 finite 校验通过。
- 固定划分：train 2,313,806，validation/monitor 23,372；validation 行号 hash `409e476abf82cd609447c6c9a6f437d96ecd897a40c5e610a33a3bc996041249`。
- KMeans 初始化：三组复用同一 500,000 行样本，行号 hash `0f7f22d7a7543965d581bcc9098a833533dbbdbb701b867f54c76d0bf66d6eb7`。
- 代码状态：运行时基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，使用包含 BGE-M3、TIGER 方法配置和学习率核验的未提交工作树。
- 环境：`poi-gr`，Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、NVIDIA RTX A6000。

### 配置选择、正式命令与耗时

- 模型结构保持 `1024 -> 512 -> 256` Encoder、三层 residual quantization、`256 -> 512 -> 1024` Decoder；latent dimension 256。
- 正式训练统一使用 Adam、learning rate `3e-4`、weight decay `0`、batch size 4096、20 epoch；loss 为 reconstruction + `1.0 × codebook` + `0.25 × commitment`。
- 逐层 Faiss GPU KMeans 使用固定 500,000 行样本和 20 iterations；seed 42，关闭 early stopping，仅保存 epoch 20。
- `lr=1e-3` 的 256×3 全量失败运行在 epoch 20 仅产生 330 个不同 SID，唯一率 0.0141%，验证集三层仅使用 `55/3/2` 个码字，最大桶 29,472。失败产物保留在 `outputs/sid/tiger/bge_m3/TIGER-BGE-M3-256x3-failed-lr1e-3/`。
- 固定前 200,000 行的非正式稳定性探测中，512×3 的 `lr=1e-4` 和 `5e-5` 最终验证码字数分别为 `13/319/259`、`1/242/264`；`lr=3e-4` 在 256×3、512×3、1024×3 下分别为 `66/121/75`、`83/223/153`、`93/409/220`，因此采用 `3e-4` 进入全量实验。
- 三组全量训练耗时分别为 708.31、714.51、715.24 秒，均以 `stop_reason=max_epochs` 正常结束。

~~~bash
python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_tiger_bge_m3.yaml \
  --experiment TIGER-BGE-M3-<CAPACITY>x3 \
  --no-resume \
  --no-progress

python scripts/sid/export_rqvae.py \
  --run-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-<CAPACITY>x3 \
  --checkpoint checkpoint_epoch_20.pt \
  --output-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-<CAPACITY>x3/evaluations/epoch_20 \
  --device cuda \
  --batch-size <65536-or-262144>
~~~

### Epoch 20 训练与全量 SID 指标

Val 码字数和 reconstruction 指标来自固定 23,372 条 validation；其余指标来自 2,337,178 条全量 SID。

| 容量 | Val total / recon / cosine | Val 码字数 L1/L2/L3 | Distinct SID / ratio | excess / colliding POI | P99 / Max | 全量利用率 L1/L2/L3 | 全量 entropy L1/L2/L3 |
|---|---|---|---|---|---|---|---|
| `256×3` | 0.000939 / 0.000249 / 86.26% | 248/256/256 | 1,124,008 / 48.0925% | 51.9075% / 68.1294% | 15 / 598 | 96.88%/100%/100% | 0.964/0.975/0.979 |
| `512×3` | 0.000944 / 0.000230 / 87.41% | 366/512/512 | 1,442,404 / 61.7156% | 38.2844% / 52.7641% | 10 / 389 | 72.07%/100%/100% | 0.905/0.967/0.973 |
| `1024×3` | 0.000903 / 0.000211 / 88.52% | 529/1018/1024 | 1,675,209 / 71.6766% | 28.3234% / 40.7894% | 7 / 306 | 54.79%/100%/100% | 0.858/0.969/0.973 |

### 与 Qwen3-Embedding-0.6B 旧基线的同容量对照

两组均采用相同 POI、RQ-VAE 结构、固定划分、KMeans 样本、epoch 20 和全量评估脚本；Embedding 编码器和学习率不同，因此本表用于选择候选 SID，不把差异归因于单一因素。

| 容量 | Qwen distinct ratio | BGE-M3 distinct ratio | 变化 | Qwen / BGE colliding POI | Qwen / BGE Max |
|---|---:|---:|---:|---:|---:|
| `256×3` | 26.9583% | 48.0925% | +21.1342 pp | 88.8080% / 68.1294% | 634 / 598 |
| `512×3` | 56.2184% | 61.7156% | +5.4972 pp | 60.8875% / 52.7641% | 710 / 389 |
| `1024×3` | 68.5345% | 71.6766% | +3.1420 pp | 46.2398% / 40.7894% | 322 / 306 |

### 产物、结论与下一步

- 正式产物位于 `outputs/sid/tiger/bge_m3/TIGER-BGE-M3-{256,512,1024}x3/`；每组包含 epoch 20 checkpoint、训练指标和 `evaluations/epoch_20/` 下 shape `[2337178,3]` 的 `sid_codes.npy`、manifest、碰撞指标与 Top 20 Case。
- 三种容量均未出现正式训练塌缩；唯一率随容量单调提升，且 BGE-M3 在三种同容量对照中均高于既有 Qwen3-Embedding-0.6B 结果。此前“唯一率突然变高”来自第一层全量码本利用率显著提高，并非数据量或评估口径变化。
- `1024×3 / epoch 20` 在本次 BGE-M3 对比中最好，完整 SID 唯一率 71.6766%，碰撞 POI 比例 40.7894%，可作为 TIGER 下一阶段候选。
- 本实验不自动替换当前 Qwen 主链路的 GID、Final PID 与 SFT 数据，也未追加 TIGER 论文的碰撞 Token。下一步需先人工确认是否选择 BGE-M3 `1024×3`，确认后再单独构建 TIGER collision token 和对应训练数据。
## EXP-20260730-02 TIGER collision token 与唯一 item identifier

### 目标与论文规则

- 目标：在已确认的 BGE-M3 `1024×3 / epoch 20` 三层 Semantic ID 后追加 TIGER 论文的第四个 collision token，构建固定长度、全局唯一且可逆的 POI item identifier。
- 论文规则：同一三元 SID 内的 item 使用从 0 开始的第四个 code 区分；即使三元 SID 没有碰撞，单例 item 也固定追加 `C0`。collision code 可在不同 SID 桶之间复用，完整四元组必须唯一。
- 论文未规定同桶 item 的稳定排序。本实现按 `poi_id` 字典序分配 `C0…C(n-1)`，使输入行顺序变化不影响 POI 到 collision code 的映射。
- 本实验只完成 identifier，不构建用户历史序列、训练数据、Tokenizer、Trie 或生成模型。

### 数据、代码、配置与环境

- 输入 SID：`TIGER-BGE-M3-1024x3 / epoch 20`，shape `[2337178,3]`，dtype `int32`，三层容量均为 1024；SID 文件 SHA256 为 `1e20b73380edc58e8455997cab8c794bc8103ad85ee8536c3ade2e632466575e`。
- SID manifest SHA256 为 `1195ca8c17fd0f637a88634759b47e38b0f164ad858ac3ee985dfd14fc8136d8`；POI ID 为 2,337,178 行且唯一，SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`。
- 代码状态：基线提交 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 BGE-M3 TIGER SID、独立 collision token 构建器和合成测试的未提交工作树。
- 方法配置：`configs/methods/baselines/tiger.yaml`；identifier 顺序固定为 `[S1,S2,S3,C]`，分配规则为同 SID 桶内按 `poi_id` 字典序从 0 连续编号。
- 环境：`poi-gr`，Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1；最终一次全量构建耗时 52.41 秒。

~~~bash
python scripts/tiger/build_identifiers.py \
  --sid-manifest outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/evaluations/epoch_20/sid_manifest.json \
  --output-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/tiger_ids/epoch_20 \
  --chunk-rows 100000
~~~

### 全量结果与确定性核验

| 指标 | 结果 |
|---|---:|
| POI 数 / 原三元 SID 数 | 2,337,178 / 1,675,209 |
| 原 SID 唯一率 | 71.6766% |
| 原 SID 碰撞桶 / 碰撞 POI | 291,352 / 953,321 |
| 原 SID collision excess | 661,969（28.3234%） |
| 单例 POI | 1,383,857 |
| 最大原 SID 桶 | 306 |
| collision token 数量 / 编号范围 | 306 / `C0–C305` |
| 使用 `C0` / 非零 collision code 的 POI | 1,675,209 / 661,969 |
| 四层 TIGER identifier distinct / unique ratio | 2,337,178 / 100% |

- `tiger_ids.npy` shape 为 `[2337178,4]`，dtype 为 `int32`；前三列与原始 `sid_codes.npy` 逐行完全一致，第四列与 `collision_codes.npy` 完全一致。
- POI 映射表中 `poi_id` 与 `tiger_id_key` 均为 2,337,178 个不同值。最大桶 SID `[465,543,609]` 的 306 条 POI 按 `poi_id` 排序后严格对应 `C0–C305`。
- 相同输入完整构建两次，`poi_tiger_id_mapping.parquet` SHA256 均为 `7998acbfc5dcd7222370bc258ef52fa2cf79ae22cd43d47758b473a6aa2250a8`，映射确定性通过。

### 产物、结论与下一步

- 产物目录为 `outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/tiger_ids/epoch_20/`，包含 `collision_codes.npy`、`tiger_ids.npy`、`poi_tiger_id_mapping.parquet`、`tiger_id_manifest.json` 和 `metrics.json`。
- `tiger_ids.npy` SHA256 为 `7aefe79a34299ff33c20c3960766d0b712b294981643ead495f7364cf78c061b`；collision 数组 SHA256 为 `312ada28834bb10baffbf6b4fd895ec83fe3754ae5f790a991c19374e24bcd8d`。
- BGE-M3 1024×3 的 TIGER item identifier 阶段完成。下一最小步骤是定义用户历史 POI 到四层 TIGER identifier 的严格时间序列数据契约；本实验不提前实现该阶段。
## EXP-20260730-03（TIGER-SFT-PREP-001）历史序列 SFT 训练前闭环

### 目标与假设

- 按 TIGER 地图检索适配格式构建全量历史序列 SFT 数据，扩展四层 item identifier、用户和结构 Token，并准备 Qwen3-0.6B 的四卡三轮正式训练入口。
- 假设 BGE-M3 `1024×3 / epoch 20` 的三层 SID 加固定 collision token 可作为全局唯一生成目标；当前 Query/GID 与最近 10 条历史 Query/GID/POI identifier 可在 `cutoff_len=512` 下完成 Assistant-only SFT。
- 本实验只读取 Train/Valid，完成词表、长度预检、实际 GPU smoke、正式 Tokenized Cache 和四卡 dry-run；未启动三轮正式训练，未读取 Test。

### 数据、模型与词表

- TIGER SFT 数据共 8,790,513 条，Train/Valid/Test 为 7,586,410/597,421/606,682；数据 Manifest SHA256 为 `581a4dd3c285a83f863c4503a603328fbc0e5c13e22a7abbef4fe4a860e896cf`。
- Train/Valid SHA256 分别为 `c812e0d2fa40cb4295cf66d41d015c0c0ee0e86a4764ca943e8a34b44c98489a` 和 `20af3356c487952bf8a58753979b3c23c452d99d627bff858066f213d8dd8dfc`。历史覆盖率为 72.0157%，平均历史长度为 4.9153，空历史保留；2000 个用户哈希 Token 不暴露原始 passenger ID。
- Prompt 使用当前请求 `<CURRENT_REQUEST><GID_*>原始 Query`，历史按 `<HISTORY><GID_*>原始 Query<S1_*><S2_*><S3_*><C_*>` 排列；Assistant 目标固定为 `<TARGET_POI><S1_*><S2_*><S3_*><C_*></TARGET_POI>`，没有额外 `T` 前缀。
- 特殊 Token 共 5,426 个；Qwen3-0.6B Tokenizer 从 151,669 扩展至 157,095，全部新增 Token 均为普通原子 Token且往返一致。扩词表映射 SHA256 为 `c504e0b92e81c15ea5206ff9638d274f4706b91b0d1a1b9d8f9cd7adc6535a9d`。

### 长度预检、GPU Smoke 与正式缓存

| 长度 | P50 | P90 | P95 | P99 | P99.9 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Input | 144 | 299 | 307 | 340 | 448 | 945 |
| Target | 8 | 8 | 8 | 8 | — | 8 |
| Total | 152 | 307 | 315 | 348 | 456 | 953 |

- Train+Valid 共 2,302 条超过 512 Token，占 0.028129%；Assistant 目标在 `cutoff_len=512` 下截断数为 0，因此正式配置固定为 512。
- 实际单卡 GPU smoke 使用 3/3 条 Train/Valid 源样本和 2/1 条 packed 样本，完成 20 step；局部 Loss 从 21.5322 降至 7.1661，step 10/20 的 Eval Loss 为 8.012436/7.405159。Loss、Gradient Norm 和 checkpoint 状态均为有限值，无 NaN、OOM 或异常退出。
- 正式 packed Cache 为 Train 2,851,853 行、Validation 208,680 行，均为固定 512 Token；`datasets.load_from_disk` 已完整加载并核对 split、列、行数和样本长度。Cache Manifest SHA256 为 `541f2966a4e7656faf6dcac8a7f5520af93f6d68ac78e7f0441dd68a50667e56`。
- 首次全量缓存任务在 Train 分词 100% 后、汇总 Dataset 时被系统强制终止，未产生成功退出码或正式 Manifest；已生成的 16 个 Train packed Arrow 分片被完整保留。恢复阶段先有一次错误的 513 长度断言失败和一次 16 并发扫描被平台回收，最终以 4 并发完整扫描 16/16 Train 分片、补算 16/16 Valid 分片并原子生成正式缓存，恢复任务 `exit=0`。这些失败均未被记录为成功运行。

### 配置、命令与环境

~~~bash
python scripts/sft/recover_tokenized_cache.py \
  --interrupted-build-dir data/sft/tokenized/.tiger_bge_m3_1024x3_history10_query_gid_v1.building-1474229 \
  --output-dir data/sft/tokenized/tiger_bge_m3_1024x3_history10_query_gid_v1 \
  --model-dir models/Qwen3-0.6B-TIGER-Vocab-v1 \
  --preflight-path data/sft/tokenized/.tiger_bge_m3_1024x3_history10_query_gid_v1.preflight.json \
  --smoke-cache-dir data/sft/tokenized/tiger_bge_m3_1024x3_history10_query_gid_v1_smoke \
  --train-dataset tiger_bge_m3_1024x3_history10_query_gid_v1_train \
  --valid-dataset tiger_bge_m3_1024x3_history10_query_gid_v1_valid \
  --workers 4 \
  --batch-size 1000
~~~

- 四卡正式配置为全参数 BF16、packing、`cutoff_len=512`、关闭 gradient checkpointing；单卡 Batch 16、梯度累积 8、四卡全局 Batch 512，训练 3 epoch。
- 保存与完整 Validation 均按 epoch 执行，`save_total_limit=3`；成功训练后标准 `checkpoint-<step>` 将由 `epoch_checkpoints.json` 显式对应 epoch 1/2/3。
- 4 卡 dry-run 已通过，解析结果为 `16 × 8 × 4 = 512`、3 epoch、`save_strategy=epoch`、`eval_strategy=epoch`；验证时正式输出目录不存在且没有同名训练进程。
- 环境为 Python 3.10.20、LLaMA-Factory 0.9.4、Transformers 4.52.4、PyTorch 2.9.1+cu128；实际 smoke GPU 为 NVIDIA RTX A6000 48 GB。代码基线提交为 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含 TIGER SFT 实现的未提交工作树。

### 产物、结论与下一步

- 全量数据位于 `data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1/`，扩词表模型位于 `models/Qwen3-0.6B-TIGER-Vocab-v1/`，正式缓存位于 `data/sft/tokenized/tiger_bge_m3_1024x3_history10_query_gid_v1/`。
- 正式配置为 `configs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1.yaml`。训练前数据、词表、Assistant-only label、实际 GPU 计算、缓存加载和四卡参数均已形成可运行闭环。
- 当前结论仅为“正式训练输入已就绪”；尚无 TIGER 三轮训练 Loss、Validation 指标、checkpoint 或生成式召回结果。下一步可在四卡 A100 平台运行入口脚本，完成后先核验 epoch 1/2/3 checkpoint，再单独实现 TIGER Trie 约束评测。
## EXP-20260804-01（TIGER-SFT-EVAL-001）三轮正式训练与固定 10,000 条 Validation 评测

### 目标与假设

- 目标：核验 TIGER 地图检索适配版三轮正式训练产物，并在与主任务完全相同的固定 10,000 条 Validation 业务键集合上评测 epoch 1/2/3。
- 评测遵循本项目对 TIGER 的复现口径：四层 `[S1,S2,S3,C]` identifier 使用无约束 Beam Search 生成，再查询冻结的 identifier→POI 映射；不使用 Trie、TCG 或地理剪枝。非法 identifier 保留原 Beam 排名并计为 miss，避免删除非法候选后人为抬高指标。

### 数据、训练产物与代码状态

- Validation 来源为 2026-07-13 的 `data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1/valid.jsonl`，共 597,421 条，SHA256 为 `20af3356c487952bf8a58753979b3c23c452d99d627bff858066f213d8dd8dfc`。
- 固定子集通过 `order_id + searchid` 与主任务固定 10,000 条逐条对齐，业务键 SHA256 为 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`，目标 POI 不一致数为 0；TIGER 子集 SHA256 为 `b06f4bd5cd62e12a3157e86394d1a82ce752872c34fe4b81e7c6fbe8f5282fa1`。
- 三个完整 checkpoint 为 `checkpoint-5571/11142/16713`，分别对应 epoch 1/2/3；完整 Validation Loss 为 `0.38476661/0.30576935/0.29646355`。
- identifier 映射覆盖 2,337,178 条 POI，映射 SHA256 为 `7998acbfc5dcd7222370bc258ef52fa2cf79ae22cd43d47758b473a6aa2250a8`；Tokenizer 词表大小为 157,095，映射 SHA256 为 `c504e0b92e81c15ea5206ff9638d274f4706b91b0d1a1b9d8f9cd7adc6535a9d`。
- 代码基线提交为 `cd64b415f5e591fadafdfa18770fce3e8ee3a7e0`，运行使用包含独立 TIGER evaluator 和固定业务键子集对齐逻辑的未提交工作树。

### 配置、命令与环境

~~~bash
python scripts/tiger/evaluate_retrieval.py \
  --valid-file data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoints \
    outputs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-5571 \
    outputs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-11142 \
    outputs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-16713 \
  --expected-checkpoint-steps 5571 11142 16713 \
  --expected-checkpoint-epochs 1 2 3 \
  --tokenizer models/Qwen3-0.6B-TIGER-Vocab-v1 \
  --identifier-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/tiger_ids/epoch_20 \
  --output-dir outputs/eval/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3 \
  --num-beams 10 \
  --per-device-eval-batch-size 32 \
  --chunk-size 1000 \
  --cutoff-len 512
~~~

- 生成参数为确定性 Beam Search，`num_beams=num_return_sequences=10`、`max_new_tokens=7`、`cutoff_len=512`；每 1,000 条原子保存一次进度。
- 环境为 Python 3.10.20、Transformers 4.52.4、PyTorch 2.9.1+cu128，评测使用单张 48GB GPU；实际 batch size 为 32，峰值显存约 24.26GB。

### 核心指标

| Checkpoint | Epoch | Valid Loss | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | Valid ID Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-5571` | 1 | 0.384767 | 0.4478 | 0.6831 | 0.7498 | 0.8022 | 0.632821 | 0.69350 |
| `checkpoint-11142` | 2 | 0.305769 | 0.5014 | 0.7477 | 0.8098 | 0.8611 | 0.690533 | 0.74428 |
| `checkpoint-16713` | 3 | 0.296464 | **0.5187** | **0.7600** | **0.8226** | **0.8716** | **0.704190** | 0.74105 |

- 三个 checkpoint 均完成 10,000 条评测并生成 `result.json`，总任务退出码为 0。epoch 3 在 HR@1/3/5/10 和 NDCG@10 上均最优，因此固定子集口径下选择 `checkpoint-16713`。
- Valid ID Rate 在 epoch 2 达到最高 74.428%，epoch 3 略降至 74.105%，说明召回提升并非来自合法率同步上升；无约束生成仍约有四分之一候选无法映射到冻结 identifier 表，这是 TIGER 与约束解码方法比较时必须保留的差异。

### 产物、结论与下一步

- 正式产物位于 `outputs/eval/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/`；汇总 JSON/CSV SHA256 分别为 `9a6eee4796a76cf9ff0af5435ef3495adec5d7447124443a08634d811509718d` 和 `a0c63c8169693db8c9053f61b8a0bb05efb206a6c86a3a3215c0ee1d0787d35c`。
- 该实验只证明固定 10,000 条 Validation、Beam=10 下 epoch 3 最优；未执行完整 Validation、Test 或其他 Beam 规模。

## EXP-20260817-01 TIGER epoch 3 四类泛化 Validation 10k

### 目标、数据与代码状态

- 目标与假设：在固定普通/复杂地理 10k 之外，检验论文复现 TIGER 对已见 Query 新配对、新 Query 已见目标、Train 频次 1—5 长尾目标和 Train 零频冷目标的泛化能力；假设目标是否在 Train 出现是最主要的难度分界。
- 数据版本：`outputs/eval/generalization_validation_suite_10k_v1/` 的四个互斥 Validation 子集，每组 10,000 条；参考 JSONL SHA256 依次为 `abb7ef59...01a3be`、`d74c3b4d...406e56`、`fe5290bb...4ee18`、`884b2ea0...52a`。三方法逐组业务键 SHA256 完全一致。
- 代码状态：基线提交 `7dd52b3437a9db41834aef638e446010d0df3610`，运行使用未提交工作树；每个 `result.json` 固定记录模型、Tokenizer、评估器与数据签名。

### 配置、命令与环境

- checkpoint 固定为 epoch 3 `checkpoint-16713`；无约束 Beam=10、返回 10 个候选、batch 32、chunk 1,000、历史实验 `cutoff_len=512`，不使用 Trie 或地理剪枝。
- 四个单元在训练平台单卡 RTX 6000D 上并行完成，每个结果均为 `completed`、10,000 行。

~~~bash
python scripts/tiger/evaluate_retrieval.py \
  --valid-file data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/generalization_validation_suite_10k_v1/<subset>_10000.jsonl \
  --checkpoints outputs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-16713 \
  --tokenizer models/Qwen3-0.6B-TIGER-Vocab-v1 \
  --identifier-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/tiger_ids/epoch_20 \
  --output-dir outputs/eval/generalization_validation_suite_10k_v1/paper_baselines/<subset>/tiger_e3 \
  --num-beams 10 --per-device-eval-batch-size 32 --chunk-size 1000 --cutoff-len 512
~~~

### 核心结果、审计与结论

| Validation 专项集 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 已见 Query / 未见配对 | 17.29% | 39.66% | 49.72% | 60.07% | 17.29% | 30.3067% | 34.4579% | 37.8330% |
| 未见 Query / 已见目标 | 49.59% | 66.32% | 71.08% | 75.30% | 49.59% | 59.4685% | 61.4309% | 62.8133% |
| 长尾目标（Train 1—5） | 26.01% | 38.67% | 44.49% | 52.33% | 26.01% | 33.3835% | 35.7853% | 38.3206% |
| 冷目标（Train 0） | 7.08% | 11.29% | 13.74% | 17.78% | 7.08% | 9.5045% | 10.5062% | 11.8127% |

- 四组宏平均 HR@1/HR@10/NDCG@10 为 24.9925%/51.3700%/37.6949%。新 Query 但目标已见明显好于目标未见，冷目标 HR@10 只有 17.78%，说明生成模型主要受目标训练覆盖约束，而不只是 Query 文本是否重复。
- 事后真实 Token 审计发现四组超过旧 512 的行数为 `0/8/5/7`，最大总长度为 `489/635/593/556`；这些行沿用旧框架 Source-prefix 截断，单组指标理论最大扰动不超过 0.08pp。结果保留为历史 512 口径，不据此声称逐样本 Prompt 全部完整。
- 正式产物位于 `outputs/eval/generalization_validation_suite_10k_v1/paper_baselines/<subset>/tiger_e3/`。如需发表级严格表，应统一以 1024 重评受影响单元；当前排序边距大于相应最大扰动，本轮不重复运行。

## EXP-20260819-01 TIGER epoch 3 三层基础 Bucket 召回诊断

### 目标与假设

- 目标：在不重新训练、不改变 TIGER 论文复现解码协议的前提下，保存固定普通 Validation 10k 的逐样本 Beam 候选，并测量去掉第四层 collision code 后的三层基础 Bucket-HR，判断当前误差主要来自前三层 `Query→SID`，还是来自桶内 `C` 后缀。
- 假设：若三层 Bucket-HR 显著高于精确 POI HR，则现有 Beam 已覆盖目标语义桶，后续应优先做桶展开与条件重排；若增量很小，则仅替换 `C` 或训练桶内排序器无法解决主要 Top-K 覆盖误差。
- 本实验只诊断 TIGER epoch 3；不运行 E4/BGE RQ-KMeans、不加入 E4/地理/历史重排、不读取 Test。

### 数据、代码状态与评测口径

- 数据继续使用 `EXP-20260804-01` 的固定普通 Validation 10k：来源文件 SHA256 为 `20af3356c487952bf8a58753979b3c23c452d99d627bff858066f213d8dd8dfc`，对齐后子集 SHA256 为 `b06f4bd5cd62e12a3157e86394d1a82ce752872c34fe4b81e7c6fbe8f5282fa1`，业务键 SHA256 为 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`，目标 POI mismatch 为 0。
- checkpoint 固定为 epoch 3 `checkpoint-16713`；identifier 仍为冻结的 `[S1,S2,S3,C]` 映射，2,337,178 行、三层容量 `1024³`，映射 SHA256 为 `7998acbfc5dcd7222370bc258ef52fa2cf79ae22cd43d47758b473a6aa2250a8`。
- 精确 POI 主指标完全保持原论文复现口径：无约束 Beam=10，非法完整 ID 保留原槽位并计 miss。新增诊断定义为：
  - `raw-slot Bucket-HR`：以 `[S1,S2,S3]` 为桶，保留全部原始 Beam 槽位和重复桶；只要候选能解析出目录中存在的基础桶，即使完整 `C` 非法也可命中桶。
  - `unique Bucket-HR`：先移除无法在目录中展开的基础桶，再按首次出现顺序对基础桶去重；该口径对应“生成若干唯一可展开桶”的候选上界。
- 代码基线提交为 `7dd52b3437a9db41834aef638e446010d0df3610`，运行使用包含可选 Bucket 诊断、独立前三层前缀解析、原子候选轨迹保存和逐行一致性校验的未提交工作树；评测器与 TIGER 方法代码 SHA256 分别为 `6cd1f0d32676c4e9aa3d73bb391c5e3cd66038b179110a0edb57d4be2663299e` 和 `eb42c361ae6573b545c561e545b8e5af90644b05586c08a1af9200fe608ecf8f`。

### 配置、命令与环境

~~~bash
TMPDIR=outputs/tmp/tbkt python scripts/tiger/evaluate_retrieval.py \
  --valid-file data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoints outputs/sft/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-16713 \
  --expected-checkpoint-steps 16713 \
  --expected-checkpoint-epochs 3 \
  --tokenizer models/Qwen3-0.6B-TIGER-Vocab-v1 \
  --identifier-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/tiger_ids/epoch_20 \
  --output-dir outputs/eval/tiger_bucket_diagnostics_fixed10k_v3/tiger_e3 \
  --num-beams 10 \
  --per-device-eval-batch-size 32 \
  --chunk-size 1000 \
  --cutoff-len 512 \
  --skip-data-hash \
  --final-checkpoint-only \
  --bucket-diagnostics
~~~

- 正式运行前完成 10 条单卡 GPU smoke；正式运行使用本机 NVIDIA RTX A6000 48GB，实际 batch size 32，耗时 1,251.78 秒、吞吐 7.989 条/秒、峰值显存 24,256,799,232 bytes，进程退出码为 0。

### 核心结果

| 排名口径 | HR@1 | HR@3 | HR@5 | HR@10 | 相对精确 POI HR 增量 |
|---|---:|---:|---:|---:|---:|
| 精确 `[S1,S2,S3,C]→POI` | 51.87% | 76.00% | 82.26% | 87.16% | — |
| 三层 Bucket，保留原 Beam 槽位 | 54.89% | 78.03% | 83.65% | 88.06% | +3.02/+2.03/+1.39/+0.90pp |
| 三层唯一可展开 Bucket | **55.53%** | **79.64%** | **85.10%** | **88.06%** | **+3.66/+3.64/+2.84/+0.90pp** |

| 目标三层 Bucket 大小 | 样本数 | 占比 |
|---|---:|---:|
| 1 | 6,146 | 61.46% |
| 2 | 1,643 | 16.43% |
| 3—5 | 1,396 | 13.96% |
| 6—10 | 522 | 5.22% |
| 11—50 | 252 | 2.52% |
| 51+ | 41 | 0.41% |

- 目标桶平均大小为 2.5339，最大为 172；38.54% 的请求目标位于碰撞桶，与此前请求加权碰撞审计一致。
- 三层前缀只从原始序列中的 `<TARGET_POI>, S1, S2, S3` 独立解析；即使后续 `C`、闭合符或 EOS 非法/缺失，只要该前缀存在且可在目录展开，仍计入 Bucket 候选。100,000 个 Beam 候选中，23.870% 的前三层前缀无法在目录展开，10.668% 是此前已出现基础桶的重复候选；每条请求平均保留 6.5462 个唯一可展开桶。
- 精确 HR/NDCG、Valid ID Rate 74.105% 以及四类 invalid error 计数均与 `EXP-20260804-01` 的历史结果逐字段完全一致，说明新增轨迹与 Bucket 统计没有改变原解码和主指标。

### 产物核验、结论与下一步

- 正式目录为 `outputs/eval/tiger_bucket_diagnostics_fixed10k_v3/tiger_e3/runs/valid_checkpoint-16713_beam10_bucketdiag_subset10000/`。`candidate_trace.jsonl` 使用 `tiger-candidate-trace-v3`，共 10,000 行、每行 10 个候选、约 36 MiB，SHA256 为 `efbbaa404774afaae9329565e7b59dea10a20b97562cae860039f8d6b28b5fd0`；manifest SHA256 为 `fdb9a9ca818e76e794180a5f213b8cf613b8b3c023fdf54c9c5ada89a20a8da7`。
- 独立逐行复算确认：从原始 token 重建三层前缀及目录桶大小后，行号连续、候选 rank 为 1—10，存储的前缀、桶大小和 exact/raw/unique rank 均为 0 条不一致；10 个分片各 1,000 行且哈希全部通过，按序拼接的 SHA256 与总轨迹一致，复算三组 HR 与 `result.json` 完全一致。实现收敛期间的 v1 漏计后缀非法候选，v2 虽然指标正确但轨迹中的桶大小不自洽；二者仅保留为调试产物，正式结论只采用通过逐行复算的 v3。
- 关键结论：Beam=10 下有 1,284 条精确 POI miss，其中 1,194 条连目标三层桶也未进入候选；删除 `C` 并在当前 Beam 内做理想桶展开最多只新增 90 条 Top-10 命中，即 HR@10 理论增量仅 0.90pp。当前 Top-10 主瓶颈因此是前三层目标 Bucket 覆盖，而不是 collision code。
- Top-1 仍存在最多 3.66pp 的理想 Bucket 排名空间，说明局部重排可能改善首位排序，但不能单靠它大幅提高候选覆盖。下一最小实验应使用同一口径测量 BGE+RQ-KMeans 与 E4 候选；只有它们的 Bucket-HR 明显高于 TIGER，才继续 E4/地理/历史桶内重排，否则转入受限 QD-RQ Top-32 弱权重门禁。本实验不据 TIGER 单点结果提前启动新 SFT。

## EXP-20260830-02 TIGER SID + Geohash6 组合键与局部 Dedup

### 目标与假设

- 目标：冻结论文复现采用的 `TIGER-BGE-M3-1024×3 / epoch 20` 三层 SID，为 2,337,178 条目标 POI 编码标准 Geohash6，并以九元组合键 `GID6+SID3` 重新分桶；只在该组合键仍碰撞时追加局部 `D`，为后续 `GID+SID+D` 与 `SID+GID+D` 自回归顺序消融提供同一套唯一 POI 映射。
- 假设：Geohash6 能拆开一部分跨区域语义碰撞；交换 GID6 与 SID3 只是九元组列置换，静态碰撞与 Dedup 分配必须完全相同，后续效果差异只能由自回归顺序和相应历史表示引起。
- 边界：本实验只构建并核验静态映射，不训练 Qwen、不生成评测指标，也不把旧 TIGER `C` 直接沿用为新组合键的去重码。

### 数据、代码、配置与环境

- 输入 SID 为 `outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/evaluations/epoch_20/`，SID manifest SHA256 为 `1195ca8c17fd0f637a88634759b47e38b0f164ad858ac3ee985dfd14fc8136d8`；三层 shape `[2337178,3]`、容量均为 1024，SID-only 指标逐字段重算一致。
- POI 主表为 16 个北京全量分片；2,337,178 个 `poi_id` 与 SID 行序逐行一致，POI ID SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`。目标 GID 使用 POI 主表 `lng/lat` 的标准 Geohash6，经度先编码，不做坐标转换；不使用请求位置 `disp_lng/disp_lat`。
- 代码基线提交 `54802e6674e283722ecee00fb862530df30cef9a`，运行使用含既有 PID 构建器的未提交工作树；`geohash.py`/`dedup.py` SHA256 为 `f279d4ee3eb431f57cdbca2746e9e8ced6052039de5b56aada5f3ba57bc2dd3d` / `c06b128e60a2778fa1256bcb65d1473f45ad2379c6ce46d6e72fe6ca534042d1`。
- 环境为 Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1；GID+SID 构建耗时 105.19 秒，Dedup 构建耗时 117.58 秒，均正常退出。

~~~bash
python scripts/pid/build_geohash.py \
  --sid-manifest outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/evaluations/epoch_20/sid_manifest.json \
  --geohash-length 6 \
  --order gid_sid \
  --output-dir outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6

python scripts/pid/build_dedup.py \
  --pid-manifest outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6/pid_manifest.json \
  --output-dir outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6-Dedup \
  --dedup-capacity 512
~~~

### 核心指标

| 标识空间 | Distinct / ratio | Colliding POI / ratio | 碰撞桶 | P50 / P90 / P95 / P99 / Max |
|---|---:|---:|---:|---:|
| TIGER SID3 | 1,675,209 / 71.6766% | 953,321 / 40.7894% | 291,352 | 1 / 2 / 3 / 7 / 306 |
| GID6+SID3 | 1,934,335 / 82.7637% | 595,173 / 25.4655% | 192,330 | 1 / 1 / 2 / 5 / 277 |
| GID6+SID3+[D] | 2,337,178 / 100% | 0 / 0% | 0 | 1 / 1 / 1 / 1 / 1 |

- Geohash6 使 358,148 个原 SID 碰撞 POI 变为唯一，SID 碰撞 POI 解决率为 37.5685%；原 291,352 个碰撞 SID 桶中 120,386 个完全解决。
- 1,742,005 个 POI 的九元组合键已唯一，不输出 D；595,173 个 POI 位于 192,330 个残余桶，桶内按 `poi_id` 字典序分配连续局部 D。最大桶 277，实际最大 `D_276`；预留 512 个 D Token 足够。
- Final PID 数组和 Parquet 已直接验证全局唯一；删除可选 D 后可恢复原九元组合键。由于 `SID3+GID6` 是相同九列的双射置换，它与 `GID6+SID3` 共享完全相同的桶、D 和 POI 映射，不重复保存第二份 233.7 万行静态表。

### 产物、结论与下一步

- base PID manifest 位于 `outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6/pid_manifest.json`，SHA256 为 `b061b4fce816858db2618783d6f5da538b847a0a231812fb950929dfe0320277`。
- Final PID manifest 与映射位于 `outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6-Dedup/`；manifest/mapping SHA256 为 `a9bbf767d303b51d551d32ca91297c2d26dc02a2220a689c6df2282fdd8d9e5c` / `385cc6353db2afa62de3ff403c7fddcb4563bb1bf4491f5c2c8f72b3aba8d3ac`。
- 静态映射门禁通过。下一步只允许在冻结的同一 TIGER history10 样本上成对线性化两种顺序，共享词表初始化、训练超参和评测子集；D 始终位于序列最后，避免额外位置混淆。

## EXP-20260830-03 两种 PID 顺序的成对全量 SFT 数据

### 目标与假设

- 目标：在 `EXP-20260830-02` 冻结的同一份 `TIGER SID3 + POI Geohash6 + 局部 D` 唯一映射上，一次扫描同时生成 `GID6+SID3+[D]` 与 `SID3+GID6+[D]` 两版 history10 SFT 数据，供后续只比较自回归位置顺序的三轮训练。
- 假设：两版使用相同样本、样本顺序、请求 GID、目标 POI、历史 POI、D 分配、Token 集合及模型初始权重；D 对单例省略、对残余碰撞样本追加且始终位于最后。训练可见内容的唯一变量是历史和目标 POI 内 GID6/SID3 的前后顺序。
- 边界：本实验只构建数据并完成轻量契约测试，不启动正式 SFT，不把旧 TIGER `C` 复制到新 PID，也不改当前请求位置的 `<USER_GID>`。

### 数据、代码、配置与环境

- 源数据固定为 `data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1/`，manifest SHA256 为 `581a4dd3c285a83f863c4503a603328fbc0e5c13e22a7abbef4fe4a860e896cf`；Train/Valid/Test 为 7,586,410/597,421/606,682。
- Final PID mapping/manifest SHA256 为 `385cc6353db2afa62de3ff403c7fddcb4563bb1bf4491f5c2c8f72b3aba8d3ac` / `a9bbf767d303b51d551d32ca91297c2d26dc02a2220a689c6df2282fdd8d9e5c`。构建器逐行反查源 TIGER 四层 ID，核验其前三层与 Final PID 的 SID3 完全一致后才允许替换历史和目标标识。
- 代码基线提交 `54802e6674e283722ecee00fb862530df30cef9a`，实际运行使用未提交工作树；`pid_order_data.py` / CLI SHA256 为 `86db75e3759127f261699acd36322b20f1d0043433dc5fe6610b4d6a1ed894de` / `8fd83c077fad135b592bb45f48b2c939f4a004c31c8113f36c93c4661b94c54a`。环境为开发服务器 Python 3.10.20。
- 两版共享 5,632 个新增普通 Token：16 个结构 Token、32 个 GID Token、2,000 个用户 Token、3×1,024 个 SID Token 和 512 个 D Token；不含旧 `C` 和 `<D_-1>`。history wrapper 统一改为 `<POI_PID>...</POI_PID>`。

~~~bash
python scripts/tiger/build_pid_order_sft_data.py \
  --source-sft-dir data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1 \
  --tiger-id-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/tiger_ids/epoch_20 \
  --pid-mapping outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6-Dedup/poi_pid_mapping.parquet \
  --pid-manifest outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6-Dedup/final_pid_manifest.json \
  --gid-sid-output-dir data/sft/tiger_bge_m3_1024x3_gid6_sid_dedup_history10_query_gid_v1 \
  --sid-gid-output-dir data/sft/tiger_bge_m3_1024x3_sid_gid6_dedup_history10_query_gid_v1
~~~

### 核心结果

| 项目 | `GID6+SID3+[D]` | `SID3+GID6+[D]` |
|---|---:|---:|
| Train / Valid / Test | 7,586,410 / 597,421 / 606,682 | 7,586,410 / 597,421 / 606,682 |
| history POI 出现次数 | 43,208,167 | 43,208,167 |
| 需要 D / 不需要 D 的目标样本 | 2,247,612 / 6,542,901 | 2,247,612 / 6,542,901 |
| Train SHA256 | `4a960288...293b43` | `c83fbb4b...63292` |
| Valid SHA256 | `e000f9fb...74769` | `e4432d22...54df4` |
| Test SHA256 | `8cfb11bc...6b18b` | `411b9093...faaa` |
| Manifest SHA256 | `361aa038...7143df` | `16ce77a0...3b24b` |

- 两个数据目录大小逐切分完全相同：Train 11,632,515,604 bytes、Valid 880,656,518 bytes、Test 905,912,883 bytes；共享 `special_tokens.json` 逐字节一致，SHA256 为 `88a1015661de3d2be47be839f661a03f7fd75cdad8d772c87a6afed17743e7ca`。
- 构建过程对每行成对检查 `sample_id/order_id/searchid/target_poi_id/history_length/split/requires_dedup`；源 Train/Valid/Test 行数及 SHA256 均与冻结 manifest 相同。两版分别写临时目录，全部完成并生成哈希后才原子切换为正式目录。
- `python -m unittest tests.tiger.test_pid_order_data -v` 的 4 项合成测试全部通过，覆盖顺序唯一变量、D 可选且恒在尾部、重复构建确定性和错误 TIGER SID 来源拒绝；真实全量 mapping 的三行 smoke 也通过。

### 产物、结论与下一步

- 两版数据分别位于 `data/sft/tiger_bge_m3_1024x3_gid6_sid_dedup_history10_query_gid_v1/` 与 `data/sft/tiger_bge_m3_1024x3_sid_gid6_dedup_history10_query_gid_v1/`；均包含 Train/Valid/Test、共享 Token 表、统计和自描述 manifest，保持 Git 忽略。
- 成对数据门禁通过。下一步必须从同一份扩词后 Qwen3-0.6B 权重启动，统一 `cutoff_len=1024`、global batch 512、seed 42、3 epoch 和无约束/约束评测协议；正式训练前仍需分别完成 Train/Valid 全量 Token 长度门禁与 packed cache。

## EXP-20260831-01 两种 PID 顺序的全量 Token 门禁与四卡 SFT 入口

### 目标与假设

- 目标：为 `EXP-20260830-03` 的两版成对 Messages 构建同源扩词模型、完成 Train/Valid 全量 Token 长度扫描和 LLaMA-Factory packed cache，并冻结两份可在 4×RTX PRO 6000D 上分别运行三轮的全参数 SFT 入口。
- 假设：共享完全相同的初始模型、普通原子 Token 集合和训练超参后，`GID6→SID3→[D]` 与 `SID3→GID6→[D]` 的后续差异可归因于 POI PID 内 GID/SID 的自回归位置顺序；D 对单例省略、对残余碰撞样本追加且始终位于最后。
- 边界：本实验只完成训练输入与启动门禁，不启动正式 SFT，不读取或缓存 Test，也不产生召回指标。

### 数据、代码状态与共享初始化

- 两版 Train/Valid 仍为 7,586,410/597,421 行。GID-first 的 Train/Valid SHA256 为 `4a9602886f946e0b0bd8e2736058648df7803cc80ef08dc3bf9c204314293b43` / `e000f9fb5d94dd357bf73e299ed0856196fac589e7e9183006861d1e69674769`；SID-first 为 `c83fbb4b45db214551c24ac32dbd8738cbc5698923f41b488853869cce463292` / `e4432d225c13765d5ec68ea21eb7ac7aa95588786aafe74230e872397e554df4`。
- 两版共用 `models/Qwen3-0.6B-TIGER-PID-Vocab-v1/`。原始词表 151,669，新增 5,632 个普通原子 Token 后为 157,301；Token 来源 SHA256 为 `88a1015661de3d2be47be839f661a03f7fd75cdad8d772c87a6afed17743e7ca`，扩展 Tokenizer 指纹为 `e837b0d23047e85a624a89e10e2c9f0fd7069634d2bdda96195dc7965ea419c4`。扩词映射和初始化权重 SHA256 分别为 `c847965d65b7d35ddb7e8fa9b760124bd0c0bb1d5aa05d85b5d393e7e016e786` / `bded12fe8196dc23e162242972c98f89fbb2be92cd07a3dc75d1d349c4c03f52`。
- 代码基线提交为 `54802e6674e283722ecee00fb862530df30cef9a`，运行使用未提交工作树；`pid_order_data.py` / 数据 CLI 当前 SHA256 为 `86db75e3759127f261699acd36322b20f1d0043433dc5fe6610b4d6a1ed894de` / `3b0c00ea93291b3dbe6aedf889cab518bbbde938399c849311f04a2272426962`。两份训练 YAML SHA256 为 `89ab3f1edd7f3c1dfdb2e7c7e08af2cb7aa63c092e2ca38516c7bce876863bf9` / `588f22ef69ed9925be5c5b66b69347de570ebbf826926255f4d5ada34508115b`。

### 配置、命令与环境

- 两版统一采用 Qwen3-0.6B 全参数 BF16、`cutoff_len=1024`、packing、`train_on_prompt=false`、3 epoch、单卡 Train/Valid batch 8、梯度累积 16、global batch 512、学习率 `5e-5`、cosine、warmup ratio 0.03、seed/data seed 42；只允许数据注册名、Cache/输出路径、实验名和独立 master port 不同。
- 训练入口要求恰好可见 4 张、每张显存至少 39 GiB 的 GPU；4×RTX PRO 6000D 满足门禁，4×A100 40GB/80GB 也可使用。两个任务应作为独立四卡平台任务启动，不能同时争用同一组四卡。
- 环境为 Python 3.10.20、LLaMA-Factory 0.9.4、Transformers 4.52.4。受限沙箱内首次构建在 Hugging Face Datasets 多进程 Manager 创建本地 socket 时失败，未产生正式 Cache；按工作流切到允许本地 IPC 的相同服务器环境后，两版全量构建均以退出码 0 完成并原子发布。

~~~bash
python scripts/sft/validate_tokenization.py \
  --model-dir models/Qwen3-0.6B-TIGER-PID-Vocab-v1 \
  --train-file data/sft/<stem>/train.jsonl \
  --valid-file data/sft/<stem>/valid.jsonl \
  --output-dir data/sft/tokenized/<stem> \
  --train-dataset <stem>_train \
  --valid-dataset <stem>_valid \
  --cutoff-len 1024 \
  --workers 8 \
  --mapping-filename poi_token_mapping.json

DRY_RUN=1 bash launchers/run_train_tiger_pid_gid6_sid_4gpu_3epoch.sh
DRY_RUN=1 bash launchers/run_train_tiger_pid_sid_gid6_4gpu_3epoch.sh
~~~

### 全量门禁结果

| 项目 | `GID6+SID3+[D]` | `SID3+GID6+[D]` |
|---|---:|---:|
| 原始 Train / Valid | 7,586,410 / 597,421 | 7,586,410 / 597,421 |
| packed Train / Validation | 1,530,367 / 114,303 | 1,530,367 / 114,303 |
| smoke 原始 Train / Valid | 10,000 / 2,000 | 10,000 / 2,000 |
| smoke packed Train / Validation | 2,015 / 381 | 2,015 / 381 |
| Input Token 最大值 | 978 | 978 |
| Target Token P50 / P90 / Max | 13 / 14 / 14 | 13 / 14 / 14 |
| 总 Token P50 / P90 / P99 / Max | 178 / 365 / 405 / 991 | 178 / 365 / 405 / 991 |
| 超过 1,024 / Assistant 目标截断 | 0 / 0 | 0 / 0 |
| Cache 大小 | 35,445,940,080 bytes | 35,445,940,080 bytes |
| Cache Manifest SHA256 | `b74c327d...ccb1f3` | `632fa4f5...07d88` |

- Train+Valid 共 8,183,831 行；4,816,937 行超过旧的 128 Token，占 58.8592%，因此本实验必须保持 `cutoff_len=1024`，不能回退旧的短上下文配置。
- 两版长度分布、packed 行数、smoke 行数和 Cache 字节数完全一致；Cache Manifest 中模型路径、Tokenizer 指纹、Train/Valid 文件哈希、packing、模板和 cutoff 均与当前输入逐项一致，且 split 只包含 Train/Valid。
- 两份 launcher 的 `bash -n`、可执行位、模型/Cache 契约检查和 `DRY_RUN=1` 均通过；dry-run 生成的 resolved config 只在实验名、数据/Cache/输出路径和 master port 等预期字段上不同，未启动 `torchrun`。为保证本机可复现，dry-run 会跳过平台 OFS 挂载和 GPU 门禁；正式运行仍保留原平台挂载流程并执行四卡显存检查。

### 产物、结论与下一步

- 正式 Cache 位于 `data/sft/tokenized/tiger_bge_m3_1024x3_{gid6_sid,sid_gid6}_dedup_history10_query_gid_v1/`；两份 Cache 各约 34 GiB，包含 packed Train/Validation、固定 smoke、长度统计和输入指纹，不含 Test。
- 训练配置为 `configs/sft/tiger_bge_m3_1024x3_{gid6_sid,sid_gid6}_dedup_history10_query_gid_v1.yaml`；平台入口为 `launchers/run_train_tiger_pid_{gid6_sid,sid_gid6}_4gpu_3epoch.sh`。launcher 属于本地平台文件，继续由 Git 忽略。
- 训练前门禁已通过，但截至本实验记录时没有 checkpoint、Validation Loss 或召回结果。下一步是在两个独立的 4×6000D 平台任务中分别执行 launcher；两边三轮都完成后，再使用同一固定 Validation 10k、相同 Beam/约束协议比较 exact PID 与去掉 D 后的 Bucket 召回。

## EXP-20260901-01 两种 PID 顺序 epoch-3 固定 10k 双解码评测准备

### 目标、训练产物与评测合同

- 目标：只比较 `EXP-20260831-01` 两个三轮 SFT 的 epoch-3 checkpoint，在同一固定 10,000 条 Validation、Beam=10 下同时给出无约束与全目录合法路径约束结果。四项评测并行分配到 4 张 RTX PRO 6000D，每项独占一张卡。
- GID-first 使用 `outputs/sft/tiger_bge_m3_1024x3_gid6_sid_dedup_history10_query_gid_v1_gpu4_e3/checkpoint-8967`，epoch 3 Validation Loss 为 `0.20180898904800415`，模型 SHA256 为 `c05c9eca00655f7672d517a5aecb13d42858530b42dc694cbe84f2f428a40b50`。
- SID-first 使用 `outputs/sft/tiger_bge_m3_1024x3_sid_gid6_dedup_history10_query_gid_v1_gpu4_e3/checkpoint-8967`，epoch 3 Validation Loss 为 `0.2122611254453659`，模型 SHA256 为 `2bb46c2afe601bdd9c2abe415482a3700633ea0e32050c2d818eb534d4bede6e`。
- 固定子集继续复用 TIGER 合法路径实验的 10,000 个 `order_id + searchid` 业务键，业务键 SHA256 为 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`。两版逐条对齐后目标 POI 不一致数均为 0；GID-first/SID-first 子集 SHA256 分别为 `186f8792102bff8630f688a5da7949c96b43ac48729025509953da1f62fa5fc6` / `8a61d6b0ee4881fe34b3d33568060c29feae26bfeabafc630e7a1bd32e2b96c7`。
- 无约束模式保留非法候选原始 Beam 槽位并按 miss 处理；约束模式只允许冻结全目录中存在的完整 PID。两种 PID 均允许单例省略 `D`、碰撞项追加 `D`，因此目标是带 wrapper 的 9/10 个内部 Token 变长路径，不能复用只接受固定四层 TIGER ID 的旧 evaluator。

### 实现、Trie 与四卡入口

- `scripts/tiger/evaluate_pid_order_retrieval.py` 新增两种顺序的统一 evaluator，严格核验 checkpoint step/epoch、Tokenizer、mapping、固定业务键与目标 POI，并支持分块原子进度、断点续跑和 CUDA OOM 时逐级降低 batch。
- `src/poi_gr/pid/trie.py` 与 `scripts/pid/build_trie.py` 支持 `gid_sid` / `sid_gid` 两种列序。两棵 Trie 均覆盖 2,337,178 个唯一叶子；GID-first/SID-first 的加载内存分别为 125,992,320 / 265,037,568 bytes，manifest SHA256 分别为 `9e33c2e9f028ad2fa51cffff36a055589a9f9f7da0e63f553a30f973423067a7` / `58a53b172e0b502d4898e86eb148e39f9c1141dd44d74a47fd02795d40097925`。
- 四卡入口为 `launchers/run_evaluate_tiger_pid_order_epoch3_fixed10k_4x6000d.sh`。GPU 0/1 分别运行 GID-first 无约束/约束，GPU 2/3 分别运行 SID-first 无约束/约束；每项使用 batch 64、chunk 500、`cutoff_len=1024`，最终汇总到 `outputs/eval/pid_order_fixed10k_v1/epoch3_comparison_4x6000d.json`。

~~~bash
cd /ofs/map_search/hudan/poi_genret
bash launchers/run_evaluate_tiger_pid_order_epoch3_fixed10k_4x6000d.sh
~~~

### 门禁状态与边界

- 两版固定子集预检、两棵全库 Trie、四项各 32 条的 GPU smoke 均通过；两个约束 smoke 的 Valid ID Rate 均为 100%。32 条结果只用于验证结构、mapping 和解码链路，不作为方法结论或正式指标。
- 单卡正式 GID-first 无约束任务在用户决定改用四卡平台后已停止，停止时 `next_line=0`、`sample_count=0`，没有提交任何正式样本或指标；零进度状态可由四卡入口安全覆盖。
- launcher 的 `bash -n` 和 `--dry-run` 已通过，相关 trie/evaluator 单元测试通过。正式四项 10k 尚未运行，因此本节不提前报告或推断 GID-first 与 SID-first 的优劣；平台任务完成后再把汇总指标补入本节。
- 首次 4×6000D 平台运行中，两项约束评测完整完成 10,000 条；GID-first/SID-first 的 HR@1/HR@10/NDCG@10 分别为 `50.31%/86.51%/69.2316%` 与 `49.60%/85.55%/68.3797%`，两者均为 100,000/100,000 个合法候选。两个无约束进程在推理前因同一顺序共享固定名 `.preflight.json.tmp` 发生并发竞争而退出，因此平台总任务为 failed，未生成四项汇总；该失败不污染两份约束结果。
- 修复后 JSON 原子写入为每个进程创建唯一临时文件，正式 batch 由发生过 OOM fallback 的 64 固定为 32；两个真实 evaluator 进程并发写同一 GID-first preflight 均以退出码 0 完成，最终 JSON 合法且无临时文件残留。launcher dry-run 已确认自动跳过两份完整约束结果，只补跑 GPU 0 的 GID-first 无约束与 GPU 2 的 SID-first 无约束；正式补跑待平台重新启动。

## EXP-20260904-01 活跃闭集 TIGER `512³` 的 500k 初始化对照

### 目标、数据与协议

- 目标：把 TIGER 的 POI 目录由北京全量 2,337,178 条切换为两周全部目标与保留历史涉及的 716,245 条活跃闭集，统一使用三层 `512×512×512` RQ-VAE，先验证缩库后的原始 TIGER 训练链路。
- 活跃目录 manifest SHA256 为 `dc13c3f137c57ba715f129ff2ccbbd8909d910da3cebc4a9fdbce2172f4d144a`，POI ID SHA256 为 `8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。同一 `models/bge-m3` 以 BF16、batch 64、长度 512、L2 normalize 重新编码为 `[716245,1024]` float16；Embedding manifest / NPY SHA256 为 `5fb22053cd5076d7d8ece031e9ea2b4ba024bf41d5f37f26fc0ffd6b08860855` / `4e12642ff52abce3c8700f69157582adfe448ab524184317458e3337da8b1b5a`。全量 finite 与 ID 对齐通过，随机 128 行相对旧全库向量的平均/最小余弦为 `0.99999547/0.99991328`。
- RQ-VAE 保持 TIGER 的 `1024→512→256` encoder、三层 residual quantization、逆向 decoder、seed 42、batch 4096、Validation 1%、AdamW `3e-4`、20 epoch；本对照按旧默认仅从 709,082 个 Train POI 中无放回采 500,000 条进行逐层 FAISS-GPU KMeans 初始化。
- 代码基线提交为 `54802e6674e283722ecee00fb862530df30cef9a`，运行使用未提交工作树；环境为 Python 3.10.20、PyTorch 2.9.1+cu128、RTX A6000。正式命令为 `python scripts/sid/train_rqvae.py --config configs/sid/rqvae_tiger_bge_m3_active_716k_512x3.yaml --experiment TIGER-ACTIVE716K-BGE-M3-512x3`。

### 结果、产物与决策

- 任务退出码为 0，20 epoch 全部完成，训练段耗时 249.64 秒；三级 KMeans 初始化均使用 512/512 个码且无死码，初始化 MSE 为 `0.00212247/0.00169620/0.00143591`。
- Epoch 20 Train/Validation reconstruction cosine 为 `0.844451/0.845738`；Validation 三级码字利用为 `199/501/483`，7,163 条监控样本的三层 SID 唯一率为 `98.9809%`、碰撞 POI 为 `1.9266%`、最大桶为 4。这里只是 Validation monitor，尚未导出 716,245 条全量 SID，不能当作最终静态指标。
- Checkpoint / resolved config SHA256 为 `cb460a102d0442976b022afbc9c41b3ac086d9f1e6931e86ec4cec316ed94bc0` / `c05da15eeb923a85e97f7da1536f4538016b84c8f456853ca22b38bdfe3d9b52`；产物在 `outputs/sid/tiger/active_716k_bge_m3/TIGER-ACTIVE716K-BGE-M3-512x3/`，日志在 `outputs/run_control/tiger_active716k_512x3/train.log`。
- 用户确认 SID 构建不涉及 query/订单监督，要求 KMeans 使用全部 716,245 条目录向量。该对照完整保留但不再导出 identifier；下一版显式采用 `sample_scope=all`，Validation 仅用于观察曲线，不从 KMeans 初始化池中扣除。

## EXP-20260904-02 活跃闭集 TIGER `512³` 全目录初始化与唯一标识

### 目标、数据与泄露边界

- 目标：在 716,245 条活跃闭集 POI 上按原始 TIGER 流程训练三层 `512×512×512` RQ-VAE，但 KMeans 初始化直接覆盖全部目录向量，不从初始化池扣除 1% reconstruction monitor；完成全量 SID 导出，并追加确定性碰撞 Token 得到可供后续 SFT 使用的四层唯一标识。
- 输入继续使用 `data/beijing_poi_active_order14d_history10_20260715_json/` 和 `outputs/embeddings/beijing_poi_active_order14d_history10_bge_m3/`；目录/POI ID/Embedding manifest/Embedding NPY SHA256 分别为 `dc13c3f137c57ba715f129ff2ccbbd8909d910da3cebc4a9fdbce2172f4d144a`、`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`、`5fb22053cd5076d7d8ece031e9ea2b4ba024bf41d5f37f26fc0ffd6b08860855`、`4e12642ff52abce3c8700f69157582adfe448ab524184317458e3337da8b1b5a`。
- 此阶段只读取 POI 名称、地址和别名编码得到的 BGE 向量，不读取 query、订单标签、Train/Valid/Test 归属或召回答案，因此全目录 KMeans 不构成下游 SFT/召回评测的信息泄露。需要单独标明的是：7,163 条 1% Validation 也参与 KMeans 初始化，所以该 reconstruction Validation 只用于同目录训练曲线监控，不能宣称为对初始化严格未见的泛化集。

### 实现、配置与运行

- `initialization.sample_scope=all` 显式定义初始化池为全部 716,245 行；当 `sample_size` 覆盖整个池时，按目录行序完整使用，实际初始化索引为 `0…716244`，SHA256 为 `3ff99cf4dda7e338504b61aad1ac9e2732b570db975c1393d209df587ed0c349`。默认 `train` 行为保持向后兼容，已有实验配置签名不变。
- 其余配置保持 TIGER：BGE-M3 1024 维输入，`1024→512→256` encoder，三层 residual quantization，逆向 decoder，seed 42、batch 4096、AdamW `3e-4`、20 epoch、Validation 1%。正式配置 SHA256 为 `ceb6340ee90a96057c4f077187b21f4f78b443688063c76497eb568f9e915a79`；环境为 Python 3.10.20、PyTorch 2.9.1+cu128、RTX A6000。
- 新增的全目录/默认 Train 初始化范围测试连同现有 RQ-VAE 测试共 14 项通过；8,192 行合成 smoke 验证全池与 Validation 存在交集、三层初始化无死码并能完成训练。正式训练、全量导出和 identifier 构建均以退出码 0 完成。

~~~bash
python scripts/sid/train_rqvae.py \
  --config configs/sid/rqvae_tiger_bge_m3_active_716k_512x3.yaml \
  --experiment TIGER-ACTIVE716K-BGE-M3-512x3-FULLINIT

python scripts/sid/export_rqvae.py \
  --checkpoint outputs/sid/tiger/active_716k_bge_m3/TIGER-ACTIVE716K-BGE-M3-512x3-FULLINIT/checkpoint_epoch_20.pt \
  --run-dir outputs/sid/tiger/active_716k_bge_m3/TIGER-ACTIVE716K-BGE-M3-512x3-FULLINIT \
  --output-dir outputs/sid/tiger/active_716k_bge_m3/TIGER-ACTIVE716K-BGE-M3-512x3-FULLINIT/evaluations/epoch_20 \
  --device cuda --batch-size 4096

python scripts/tiger/build_identifiers.py \
  --sid-manifest outputs/sid/tiger/active_716k_bge_m3/TIGER-ACTIVE716K-BGE-M3-512x3-FULLINIT/evaluations/epoch_20/sid_manifest.json \
  --output-dir outputs/sid/tiger/active_716k_bge_m3/TIGER-ACTIVE716K-BGE-M3-512x3-FULLINIT/tiger_ids/epoch_20
~~~

### 训练、SID 与标识结果

- 三级 KMeans 均使用 716,245/716,245 条向量、512/512 个码且无死码，初始化 MSE 为 `0.00212182/0.00169604/0.00143159`；初始化耗时 52.45 秒，20 epoch 总训练段耗时 250.31 秒。Epoch 20 Train/Validation reconstruction cosine 为 `0.844073/0.845426`，Validation monitor SID 唯一率为 `99.1065%`。
- 相比 `EXP-20260904-01` 的 500k Train 初始化，三级初始化 MSE 分别下降约 0.0307%/0.0095%/0.3007%，Validation monitor SID 唯一率提高 0.1256pp；变化很小，说明 500k 已足够稳定，但本版更符合“目录全量构建”的定义。由于 monitor 参与初始化，reconstruction cosine 不用于两版优劣的泛化结论。
- 全量三层 SID 共 551,731 个不同组合，唯一率 `77.0310%`；碰撞 excess 164,514（`22.9690%`），241,934 个 POI 位于碰撞桶（`33.7781%`），77,420 个碰撞桶，桶大小 P50/P90/P95/P99/Max 为 `1/2/3/6/152`。三级最终码字利用为 `229/512`、`506/512`、`491/512`，利用率 `44.73%/98.83%/95.90%`。
- 第四层 `C` 在每个三层 SID 桶内按 `poi_id` 字典序从 0 稳定编号，跨桶复用；需要 152 个 Token，最大 `C=151`。最终 `[S1,S2,S3,C]` 为 `[716245,4]` int32，716,245 个标识全部唯一，前三列逐行等于导出的 SID，第四列逐行等于 collision code。

### 产物、结论与下一步

- 正式 run 位于 `outputs/sid/tiger/active_716k_bge_m3/TIGER-ACTIVE716K-BGE-M3-512x3-FULLINIT/`；epoch-20 checkpoint / resolved config SHA256 为 `5f01b5b23463b0645dfca1fc2665f104357feccec8bd3a25c96937f6f4f9a2f9` / `250e226f1ae7e48f80808726468e719b3f73ac7e5d19f5696d4634f95f27dc89`。
- 全量 SID manifest / NPY SHA256 为 `ba6738dd72d9eb6ab91470586809930d7a0cfa14a99a81b9c94e22add8caf52e` / `c6072dbeb622eae94f28ee28b0065693840680305c8d70a0d501172ec9d61c6d`；TIGER identifier manifest / NPY / mapping SHA256 为 `43833a58f2e0ac579f2c589d0da6fa562e4105b1f52896d8644f07f565fa06b8` / `70f13cc5e760fbd90de8ca73fd9263bc04f25bc123fdec2ce432bdacbe8f9880` / `33d4e820f0d818d5a1d2d3b0290ef30f9a3254999b651e1f8f6293b48819e5b8`。
- 本版作为 716,245 活跃闭集的新 TIGER SID 基线冻结；后续应基于该四层映射重建与原 TIGER 完全同协议的 SFT Train/Valid/Test，再用固定 Validation 10k 做无约束与合法路径约束评测。不能把缩库后的 SID 静态唯一率直接与旧 2,337,178 目录的数值作方法提升结论。

## EXP-20260905-01：活跃闭集 TIGER 三轮 SFT 输入与六项评测入口

### 数据、Token 门禁与训练配置

- 日期：2026-09-05；状态：正式训练入口已就绪，尚未产生 checkpoint。基于 `EXP-20260904-02` 的 `[S1,S2,S3,C]` 唯一标识重建原 TIGER history10 协议数据，Train/Valid/Test 仍为 `7,586,410/597,421/606,682`，不按训练目标再次裁剪；全部目标和历史 POI 均由 716,245 活跃目录覆盖。Messages 位于 `data/sft/tiger_active716k_bge_m3_512x3_history10_query_gid_v1/`，manifest SHA256 为 `80d39eb00269baffe0b5c72aa9ecbcaf6d832b3f52e18329390adceabaf088a1`；
- 扩词模型位于 `models/Qwen3-0.6B-TIGER-Active716K-512x3-Vocab-v1/`，Token mapping / tokenizer JSON SHA256 为 `5f45209ffcf45fb9691c2d1ad369bc2fde2c206a356407c50baa2afa01ae6cdb` / `78081f50b24803c490bf53d20d95adf346aec662dd7dc6258dd1c05c05a925ff`；
- Train+Valid 共 8,183,831 条完成 `cutoff_len=1024` 全量长度门禁：完整序列最大 953、目标长度固定 8、超长和 Assistant 目标截断均为 0。packed cache 位于 `data/sft/tokenized/tiger_active716k_bge_m3_512x3_history10_query_gid_v1/`，Train/Validation 为 `1,296,883/96,949`；cache manifest SHA256 为 `e62c82201e8080ac0bb84277ea7d8761f8568ab50b3958af035b555b1a8d29e7`，缓存构建与 supervisor 退出码均为 0；
- 正式配置为 `configs/sft/tiger_active716k_bge_m3_512x3_history10_query_gid_v1.yaml`：Qwen3-0.6B 全参数 SFT、4×RTX PRO 6000D、单卡 batch 8、累积 16、global batch 512、BF16、seed 42、三轮；按 1,296,883 packed rows 计算为每轮 2,533 step，epoch 3 固定 `checkpoint-7599`。

### 一键入口与当前结论

~~~bash
cd /ofs/map_search/hudan/poi_genret/launchers
bash run_train_tiger_active716k_512x3_4x6000d_3epoch.sh
~~~

- launcher 已完成全链路 dry-run：训练结束后会先核验三个 epoch checkpoint，再只评 epoch 3；固定同业务键 Validation 10k 同时执行无约束 Beam=10 和全目录合法路径约束，并继续执行四个冻结泛化 10k（已见 Query/未见 Query–POI、新 Query/已见目标、长尾目标、冷目标），最终汇总六项结果；
- 当前只确认“可安全启动正式训练”，没有记录任何臆测的 Loss、HR 或 NDCG。若输出目录已经出现 checkpoint，launcher 的恢复/拒绝覆盖逻辑应按实际状态执行，不得把新任务隐式混入旧 checkpoint。

## EXP-20260908-01：活跃闭集 TIGER 三轮 SFT 与中止的六项评测

### 训练、故障与恢复

- 日期：2026-09-07—08；状态：三轮 SFT 完成，固定 10k 双解码完成，四个泛化评测按用户要求中止。数据、Tokenizer、SID 和训练配置沿用 `EXP-20260905-01`；代码基线为 `54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树有未提交改动；
- 4×RTX PRO 6000D、BF16、单卡 batch 8、累积 16、global batch 512，共完成 7,599 step。epoch 1/2/3 Validation Loss 为 `0.440864/0.347435/0.335524`，最终 checkpoint 为 `outputs/sft/tiger_active716k_bge_m3_512x3_history10_query_gid_v1_gpu4_6000d_e3/checkpoint-7599`，模型 SHA256 为 `008f0011686fc2bb4e71be9cf746fd41d17c187e1c7b3282ebbc11f6cae24c35`；
- 平台任务在训练成功后显示 failed，并非模型或 checkpoint 失败：LLaMA-Factory 多卡 launcher 以 `sys.exit(0)` 结束，使外层 `scripts/sft/train.py` 未执行 `epoch_checkpoints.json` 写入；后置检查因此退出。三个 epoch checkpoint 均完整，索引已由相同校验函数补建；
- 2026-09-08 在单张 RTX A6000 上按原六项协议顺序补评。32 条 smoke 退出码为 0、峰值显存 16,567,435,776 bytes；固定 10k 无约束和合法路径约束均完成。运行到首个泛化单元、尚未提交任何正式候选分块时，用户要求不再占用本服务器，精确 runner 进程组以 SIGTERM 停止，记录为退出码 143；MMBERT 和其余泛化单元未由该 runner 启动。

### 已完成的固定 10k 结果

| 解码 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | Valid ID Rate |
|---|---:|---:|---:|---:|---:|---:|
| 无约束 Beam=10 | 49.78% | 73.58% | 79.79% | 84.51% | 68.0496% | 74.769% |
| 全目录合法路径约束 | 49.79% | 73.72% | 80.07% | 85.08% | 68.2633% | 100% |

- 相对旧 233 万目录 TIGER 的固定 10k 无约束 `51.87%/87.16%/70.4190%`，活跃闭集的 HR@1/HR@10/NDCG@10 分别低 `2.09/2.65/2.3694pp`。本次同时改变目录、码本容量、SID 和训练目标分布，差值不能归因为单一“缩库”因素；
- 合法路径约束相对自身无约束仅提高 HR@1/HR@10/NDCG@10 `0.01/0.57/0.2137pp`，说明当前主要差距不是非法组合。四类泛化和 active-MMBERT 配对生成指标尚未完成，不能形成最终方法排名；
- 正式结果位于 `outputs/eval/tiger_active716k_bge_m3_512x3_history10_query_gid_v1_gpu4_6000d_e3/`；中止状态和日志位于 `outputs/run_control/active716k_pair_eval/`。后续若继续，只允许在训练平台复用现有 `checkpoint-7599` 补跑缺失单元，不重训、不在本服务器恢复。

### 2026-09-09：A100 泛化补评闭环

- 复用同一 `checkpoint-7599` 在训练平台完成此前缺失的四个无约束泛化 10k；四项均为 `completed`、各 10,000 条，Test 未读取。四个集合与 active-MMBERT 使用完全相同的业务键顺序，target mismatch 为 0；本次 A100 launcher 自动跳过已有的两项固定 10k，因此固定结果仍来自 2026-09-08 的单卡 A6000，四项泛化是严格同卡型 A100 配对结果；

| Validation 切片 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | Valid ID Rate |
|---|---:|---:|---:|---:|---:|---:|
| 已见 Query / 未见 Query–POI | 16.88% | 35.65% | 44.55% | 53.45% | 34.3798% | 74.517% |
| 新 Query / 已见目标 | 43.43% | 59.47% | 64.82% | 69.55% | 56.6535% | 59.479% |
| 长尾目标（Train 频次 1—5） | 15.74% | 24.85% | 29.72% | 36.58% | 25.2097% | 54.381% |
| 冷目标（Train 频次 0） | 4.45% | 6.06% | 7.16% | 9.48% | 6.5527% | 49.189% |
| 四类宏平均 | 20.1250% | 31.5075% | 36.5625% | 42.2650% | 30.6989% | 59.3915% |

- 相对旧 233 万目录 TIGER 的四类宏平均 `24.9925%/51.3700%/37.6949%`（HR@1/HR@10/NDCG@10），active 版本分别下降 `4.8675/9.1050/6.9960pp`。结合固定 10k 的同步下降，缩小到 716,245 条活跃目录并未自动改善生成检索；本次还同时把三层容量改为 `512³` 并重训 SID/SFT，因此只能将其记录为整个 active-BGE 配方的负结果，不能把下降单独归因于缩库；
- 完整六项汇总为 `outputs/eval/tiger_active716k_bge_m3_512x3_history10_query_gid_v1_gpu4_6000d_e3/epoch3_evaluation_summary.json`，SHA256 为 `7200b9dc02d4f4f98218da0f4dd4921bfb6d5841d6fbfc4f7d6cb673bab2e235`。
