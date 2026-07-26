# SFT 数据与训练实验

本文件统一记录生成式 POI 检索的 SFT 数据、smoke test、正式训练和约束生成评测。当前已完成订单主任务数据构建、Qwen3-0.6B 词表扩展、两 epoch 全参数 SFT，以及固定 10,000 条 Validation、Beam=10 的四 checkpoint Trie 约束评测；尚未执行完整 Validation、Beam 对比或 Test。

## EXP-20260723-04（SFT-DATA-001）订单主任务 SFT 数据

### 目标与假设

- 主任务定义为“原始 Query + 用户位置 Geohash6 → 目标 POI 唯一 Final PID”。假设严格时间切分、唯一 PID 标签和不改变原始 Query 的首版数据，可作为后续 SFT smoke test 的可复现输入基线。
- 本实验只构建和验证 Messages JSONL，不包含 POI-PID 辅助对齐、负样本、采样、用户历史、时间特征、tokenizer 修改或模型训练。

### 数据、代码与环境

- 订单输入实际使用 `data/beijing_order_clean_20260701_20260714_json/` 的 32 个 JSONL 分片。任务最初给出的 `beijing_poi_clean_20260715_json` 是 POI 主表目录，已由用户校正，不用于本实验。
- 真实订单字段包含预计的 `order_id/searchid/query/disp_lng/disp_lat/create_time/source_dt/poi_id` 等字段，并额外包含订单日期、展示区域和 POI 快照日期字段；额外字段不进入 SFT。
- 唯一 PID 映射为 `BJ-RQVAE-1024x3-e20-G6-Dedup`，映射 SHA256 为 `050a2a90223e7f521434360d1608ce60a39326392752c36c65dfe7458beeacf0`。
- 代码基线提交为 `4cf01c252aa49d50b98c443205397ef1204f89b8`，运行使用包含 SID、RQ-VAE、PID 与 SFT-DATA-001 实现的未提交工作树。
- 环境为 Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1。

### Messages 格式与处理规则

User 内容固定为：

~~~text
<QUERY>{原始 query}</QUERY>
<USER_GID><G_x1><G_x2><G_x3><G_x4><G_x5><G_x6></USER_GID>
~~~

Assistant 内容固定为九层单例 PID 或十层 Dedup PID：

~~~text
<G_g1><G_g2><G_g3><G_g4><G_g5><G_g6><S1_x><S2_x><S3_x>[<D_x>]
~~~

- 唯一过滤规则为 `query is null` 或 `query.strip()` 为空；本次全量数据实际没有空 Query。
- 非空 Query 保留原文；不删除重复订单，不做 Query 长度过滤、纠错、标准化或采样。
- 用户位置使用原始 `disp_lng/disp_lat` 按标准 Geohash6 编码，不做坐标转换。
- 单例 PID 不输出 Dedup Token，`-1` 仅是固定宽度映射的终止哨兵；特殊 Token 表不包含 `<D_-1>`。
- 时间切分严格使用 `create_time`：2026-07-01 至 07-12 为 train，07-13 为 valid，07-14 为 test；`source_dt` 仅逐行核验日期一致性。

### 配置与命令

~~~bash
python scripts/build_sft_main_data.py \
  --orders-dir data/beijing_order_clean_20260701_20260714_json \
  --pid-mapping outputs/pid/BJ-RQVAE-1024x3-e20-G6-Dedup/poi_pid_mapping.parquet \
  --pid-manifest outputs/pid/BJ-RQVAE-1024x3-e20-G6-Dedup/final_pid_manifest.json \
  --output-dir data/sft/beijing_order_main_v1 \
  --train-start 2026-07-01 \
  --train-end 2026-07-12 \
  --valid-date 2026-07-13 \
  --test-date 2026-07-14 \
  --geohash-length 6
~~~

### 核心结果

| 指标 | 结果 |
|---|---:|
| 原始订单 / 保留样本 | 8,790,513 / 8,790,513 |
| 空 Query | 0 |
| Train / Valid / Test | 7,586,410 / 597,421 / 606,682 |
| 唯一 Query | 1,574,244 |
| 唯一目标 POI / 全量覆盖率 | 520,333 / 22.2633% |
| PID 匹配率 | 100% |
| 需要 Dedup 的样本 / 比例 | 2,508,120 / 28.5321% |

### 产物、验证与结论

- 输出目录为 `data/sft/beijing_order_main_v1/`，严格只包含 `train.jsonl`、`valid.jsonl`、`test.jsonl`、`special_tokens.json`、`manifest.json` 和 `stats.json`。
- Manifest SHA256 为 `f254e6f2ad8c0591732887163a1211a46e9718f4071674886c12b26360334ce9`。
- 全量预检未发现非法用户坐标、未匹配 POI、PID 不完整、时间解析失败、日期越界或 `source_dt` 不一致；三份 JSONL 已逐行回读核验 Messages、用户 GID、目标 PID 和行数守恒。
- 相同输入完整复跑后六个输出文件 SHA256 全部一致，数据顺序、sample ID 与序列化结果确定。
- 该数据可作为后续 SFT smoke test 的候选输入，但本实验不自动进入训练、tokenizer 修改、Trie 或约束解码。

## EXP-20260724-01（SFT-001）Qwen3-0.6B 订单主任务全参数 SFT

### 目标与假设

- 使用 Qwen3-0.6B 全参数微调学习“原始 Query + 用户 Geohash6 → 唯一 Final PID”，只对 Assistant PID 序列计算自回归交叉熵。
- 假设扩展后的 3,620 个 POI Token 能作为原子 Token 稳定训练，且两轮训练可建立首个 teacher-forcing Validation Loss 基线。
- 本实验只训练和评估 Train/Valid，不读取 Test，不执行生成、Trie、HR@K、NDCG@K 或 POI-PID 辅助对齐。

### 数据、模型与词表

- 原始 Train/Valid 为 7,586,410/597,421 条，SHA256 分别为 `4ed3f3849e0beb10df500f013bc08b4663b0dca5df92dde9c8b3cb4a7ec9ffe4` 和 `14dabc842ba4623f19b2d9e21cafa026917d333db78016956cc4bcfc506ba187`。
- 训练复用 packing 后的 Tokenized Cache：Train 2,344,413 行、Validation 185,277 行；缓存 Manifest SHA256 为 `960ea4a03e4802370e801bb37d1883479b207623b21d3d7db5194cc777c47743`。缓存只包含 `train` 和 `validation` 两个 split。
- 基础模型为 `models/Qwen3-0.6B/`，模型 SHA256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`，配置 SHA256 为 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`。
- 扩词表模型为 `models/Qwen3-0.6B-POI-Vocab-v1/`。Tokenizer 由 151,669 扩展至 155,289，新增 3,620 个普通原子 Token；模型词表同步为 155,289，输入 Embedding 与 LM Head 保持绑定。
- 扩展后 Tokenizer SHA256 为 `e7147bfb2084c48f620405b38575f1495e1cf825d945fb7ef04190810c7173ca`；`poi_token_mapping.json` SHA256 为 `665aaf9fed32a1e7606051078c77f16b2f7c45e1ea139c8641249c7ec0f1f424`。

### 序列预检与 Smoke

| 长度 | P50 | P90 | P95 | P99 | P99.9 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Input | 22 | 25 | 28 | 37 | 54 | 776 |
| Target | 11 | 12 | 12 | 12 | — | 12 |
| Total | 33 | 37 | 40 | 48 | 65 | 787 |

- Train+Valid 共 237 条总序列超过 128 Token，占 0.002896%；目标 PID 在 `cutoff_len=128` 下截断数为 0，因此正式训练保持 128。
- 固定前 10,000 条 Train 和前 2,000 条 Valid 的 20-update-step smoke 已通过 Token、label mask、finite loss/gradient、checkpoint 保存恢复和连续 step 检查。Smoke 权重及输出按用户要求在正式训练前清理，不作为正式实验产物保留。

### 配置、命令与环境

~~~bash
bash run_train_sft_single_a100_2epoch.sh
~~~

- 单卡 A100，`per_device_train_batch_size=64`、`gradient_accumulation_steps=8`，全局有效 Batch Size 为 512；`packing=true`、`cutoff_len=128`、BF16、全参数训练、关闭 gradient checkpointing。
- AdamW，学习率 `5e-5`，weight decay `0.01`，cosine scheduler，warmup ratio `0.03`，max grad norm `1.0`，seed/data seed 均为 42。
- 共 2 epoch、9,158 个 update step；每 0.5 epoch 完整验证并保存可恢复 checkpoint，不启用 early stopping 或 best-checkpoint 自动选择。
- 环境为 Python 3.10.20、LLaMA-Factory 0.9.4、Transformers 4.52.4、PyTorch 2.9.1+cu128、CUDA runtime 12.8。代码基线提交为 `4cf01c252aa49d50b98c443205397ef1204f89b8`，运行使用包含 SFT-001 实现和平台入口的未提交工作树。
- Resolved Config SHA256 为 `b2fa61b5f32f7f6164e6e57e8943e45c0b40f8f3944ba13be3aa8721d00cd159`。

### Checkpoint 与核心指标

下表的 Train Loss 是 checkpoint 前最近一次 `logging_steps=20` 的局部窗口值，并列出实际日志 step；Validation Loss 来自完整 185,277 行 packed Validation 集。

| Checkpoint | 实际 epoch | 局部 Train Loss（日志 step） | Validation Loss | Validation 相对前一节点改善 |
|---|---:|---:|---:|---:|
| `checkpoint-2290/` | 0.500109 | 0.3886（2280） | 0.396341 | — |
| `checkpoint-4580/` | 1.000218 | 0.2628（4580） | 0.281600 | 28.95% |
| `checkpoint-6870/` | 1.500328 | 0.1854（6860） | 0.239919 | 14.80% |
| `checkpoint-9158/` | 2.000000 | 0.1740（9140） | 0.227494 | 5.18% |

- 四个 checkpoint 均包含 `model.safetensors`、optimizer、scheduler、trainer state 和 RNG state，可用于标准恢复。
- 全程平均 Train Loss 为 0.458500；该值包含训练初期高损失阶段，不等价于最终局部 Train Loss。
- 训练耗时 10:48:59.95，训练吞吐为 120.412 packed samples/s、0.235 update steps/s、15,412.699 tokens/s；最后一次完整 Validation 耗时 0:08:46.22。
- Trainer 报告训练阶段 GPU peak delta 为 37,825 MiB；结合训练前、分配增量和峰值增量估算进程绝对峰值约 43.66 GiB。
- 日志未出现 Traceback、OOM、RuntimeError、NaN、Inf 或异常终止；训练后期局部 grad norm 稳定在约 0.8～0.9。

### 产物、结论与下一步

- 正式产物位于 `outputs/sft/qwen3_0.6b_main_v1_a100_e2/`，包括四个完整 checkpoint、resolved config、trainer state、Train/Eval 结果、TensorBoard event、终端日志和 loss 曲线。
- Validation Loss 从 0.5 epoch 的 0.396341 连续下降至 2.0 epoch 的 0.227494，总降幅 42.60%，在 teacher-forcing 口径下尚未出现验证损失反弹。
- 边际收益持续递减，1.5→2.0 epoch 仅改善 5.18%；局部 Train/Validation Loss 间距在第二轮扩大。因此当前只能判定 2.0 epoch 的 teacher-forcing loss 最低，不能据此宣称其正式检索效果最佳。
- 截至 SFT-001 完成时，阶段状态为“训练完成，等待约束解码评估”。后续 `EXP-20260725-01（SFT-EVAL-001）` 已在固定口径下比较四个 checkpoint 的合法 PID 生成率和 HR@K/NDCG@K；正式推理 checkpoint 是否冻结仍以该实验的审核结论为准。

## EXP-20260725-01（SFT-EVAL-001）Final PID Trie 约束生成式检索评测

### 目标与假设

- 构建覆盖全部 2,337,178 条 Final PID 的紧凑整数 Trie，并验证单例九层 PID、Dedup 十层 PID 和 EOS 的合法分支约束。
- 使用相同的固定 Validation 子集和 Beam=10 比较 0.5、1.0、1.5、2.0 epoch 四个 checkpoint。假设约束解码能够保证候选 PID 合法，并用相同数据顺序判断训练轮数带来的相对变化。
- 原计划执行完整 597,421 条 Validation、Beam 探索和冻结后的 Test；因完整评测耗时较长，按用户确认将本轮缩小为固定 10,000 条 Validation 子集、仅 Beam=10。未执行完整 Validation、Beam 对比、配置冻结或 Test 读取。

### 数据、代码与环境

- Validation 来源为 2026-07-13 的 `data/sft/beijing_order_main_v1/valid.jsonl`，共 597,421 条，SHA256 为 `14dabc842ba4623f19b2d9e21cafa026917d333db78016956cc4bcfc506ba187`。
- 固定子集按 `sample_id` 字典序选择最小的 10,000 条，再按原始源行号恢复顺序；子集 SHA256 为 `a2e0366d3d8f582e53a5293b687dc08f85b2960d60c7fb44d2fd02168063a944`，子集 Manifest SHA256 为 `4c23ec4797ed0fed023ef5e49d7714c3dd1d406fae4179976e6a04ac8ea7b4a7`。相同输入重复调用会复核并复用同一子集。
- 四组评测使用 `outputs/sft/qwen3_0.6b_main_v1_a100_e2/` 下的 `checkpoint-2290/4580/6870/9158`，Tokenizer 为 `models/Qwen3-0.6B-POI-Vocab-v1/`。
- 代码基线提交为 `4cf01c252aa49d50b98c443205397ef1204f89b8`，运行使用包含 Trie 和生成评测实现的未提交工作树。
- 环境为 Python 3.10.20、LLaMA-Factory 0.9.4、Transformers 4.52.4、PyTorch 2.9.1+cu128，GPU 为单卡 NVIDIA RTX A6000 48 GB。

### Trie、配置与命令

- Trie 使用 CSR 风格的 `child_offsets/child_token_ids/child_node_ids` 和终止节点数组，不为每个节点创建 Python dict。共 6,863,886 个节点、6,863,885 条边和 2,337,178 个唯一叶子；文件大小 128,520,240 bytes，加载数组内存 128,519,600 bytes，核心构建耗时 9.14 秒。
- Prompt 使用与训练一致的 LLaMA-Factory `qwen3_nothink` 模板；正式运行前固定校验 100 条，确认不包含目标 PID 或 `<think>`，generation prompt、EOS、PAD、attention mask 和左 padding 均正确。
- 解码固定为 Trie 开启、BF16、`num_beams=10`、`num_return_sequences=10`、`max_new_tokens=11`、`length_penalty=1.0`、`do_sample=false`、`early_stopping=true` 和归一化 sequence score 排序。实际 batch size 为 128，分片大小为 1,000。

~~~bash
python scripts/build_pid_trie.py \
  --pid-mapping outputs/pid/BJ-RQVAE-1024x3-e20-G6-Dedup/poi_pid_mapping.parquet \
  --pid-manifest outputs/pid/BJ-RQVAE-1024x3-e20-G6-Dedup/final_pid_manifest.json \
  --tokenizer models/Qwen3-0.6B-POI-Vocab-v1 \
  --output-dir outputs/eval/qwen3_0.6b_main_v1_a100_e2/trie

python scripts/evaluate_sft_retrieval.py \
  --mode valid-checkpoints \
  --valid-file data/sft/beijing_order_main_v1/valid.jsonl \
  --checkpoints \
    outputs/sft/qwen3_0.6b_main_v1_a100_e2/checkpoint-2290 \
    outputs/sft/qwen3_0.6b_main_v1_a100_e2/checkpoint-4580 \
    outputs/sft/qwen3_0.6b_main_v1_a100_e2/checkpoint-6870 \
    outputs/sft/qwen3_0.6b_main_v1_a100_e2/checkpoint-9158 \
  --tokenizer models/Qwen3-0.6B-POI-Vocab-v1 \
  --trie-dir outputs/eval/qwen3_0.6b_main_v1_a100_e2/trie \
  --num-beams 10 \
  --top-k 10 \
  --length-penalty 1.0 \
  --per-device-eval-batch-size 128 \
  --chunk-size 1000 \
  --validation-subset-size 10000 \
  --output-dir outputs/eval/qwen3_0.6b_main_v1_a100_e2
~~~

### 固定 10,000 条 Validation checkpoint 对比

| Epoch | Checkpoint | Beam | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | `checkpoint-2290` | 10 | 0.3444 | 0.5743 | 0.6472 | 0.7103 | 0.3444 | 0.4815 | 0.5116 | 0.5322 |
| 1.0 | `checkpoint-4580` | 10 | 0.4119 | 0.6611 | 0.7368 | 0.7975 | 0.4119 | 0.5599 | 0.5912 | 0.6111 |
| 1.5 | `checkpoint-6870` | 10 | 0.4457 | 0.6990 | 0.7750 | 0.8376 | 0.4457 | 0.5961 | 0.6274 | 0.6479 |
| 2.0 | `checkpoint-9158` | 10 | 0.4577 | 0.7141 | 0.7878 | 0.8483 | 0.4577 | 0.6096 | 0.6402 | 0.6599 |

### 2.0 epoch 分层诊断与性能

| 诊断项 | 结果 |
|---|---:|
| 合法结构 / 合法 PID 比例 | 100% / 100% |
| GID6 / SID3 / Base PID9 / Final PID Exact Match | 0.7069 / 0.4944 / 0.4853 / 0.4577 |
| Singleton 样本数 / HR@1 / HR@10 / NDCG@10 | 7,072 / 0.4641 / 0.8231 / 0.6476 |
| Dedup 样本数 / HR@1 / HR@10 / NDCG@10 | 2,928 / 0.4423 / 0.9092 / 0.6897 |
| Base PID9 正确条件下 Dedup Token 准确率 | 0.8243（1,295 / 1,571） |
| 返回候选数均值 / 重复候选比例 | 10.0 / 0% |
| 吞吐 / 平均单样本耗时 | 13.73 samples/s / 72.81 ms |
| 峰值显存 | 27.05 GiB |

四组运行共耗时 2,918.39 秒，每组均由 10 个完整分片覆盖 10,000 条样本；中断恢复签名包含数据、checkpoint、Tokenizer、Trie、生成配置和评测代码哈希，配置变化不会复用旧进度。最佳 checkpoint 的五类错误 Case 各保留 100 条，仅存于 Git 忽略的评测产物中。

### 产物、结论与下一步

- 输出目录为 `outputs/eval/qwen3_0.6b_main_v1_a100_e2/`；保留全量 Final PID Trie、固定子集及 Manifest、Prompt 校验、四组分片进度、JSON/CSV 汇总和 500 条固定错误 Case。
- `checkpoint-9158` 在该固定子集上的 NDCG@10、HR@1 和 HR@10 均为四组最高，且 0.5→2.0 epoch 的各项主指标单调提升；因此它是本轮固定子集口径下的排序最优 checkpoint。
- 该结论不能替代完整 Validation，也未比较其他 Beam。当前没有生成 `selected_config.json`，没有读取或评测 2026-07-14 Test，不能将上述指标写成正式 Test 结果。
- 下一步需由用户先核验本轮结果，再单独决定是否冻结 `checkpoint-9158 + Beam=10`、扩大 Validation、执行 Test 或训练 3 epoch；本任务不自动进入这些步骤。
