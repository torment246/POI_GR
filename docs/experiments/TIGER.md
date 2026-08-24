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
