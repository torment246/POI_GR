# BeamRisk-SFT：Beam 风险对齐监督微调

## 1. 文档状态

- 方案版本：`v1.3`
- 记录日期：2026-09-03
- 当前状态：正式 Epoch 1 已完成并产出有效 `checkpoint-5571`，Validation loss 为 `0.3851958513`；epoch-1 四卡挖掘分片已复用并完成正式收口，100,000 条风险 pair 已通过独立全量校验，当前可直接续训 Epoch 2，无需重跑 Epoch 1 或重复约一小时的 Beam 生成
- 方法范围：只改生成模型的 SFT 目标与训练流程，不改 POI 向量、RQ-VAE、SID、Tokenizer、Prompt、数据划分和解码评测协议
- 第一对照：现有 TIGER SFT
- 正式训练原则：从与 TIGER 相同的初始模型重新训练，不加载任何已经完成 SFT 的 TIGER checkpoint

本文档是 BeamRisk-SFT 的方法主文档。`beamrisk_sft/` 是这条创新线的独立项目根目录：后续新增的配置、代码、测试、挖掘产物、训练 checkpoint、评测结果和实验记录全部放在本目录内，避免把这条 SFT 创新线混入现有 SID、Embedding、碰撞实验或仓库公共目录。TIGER 已有模型、数据和评测产物只作为只读基线输入引用，不在本目录中复制或改写。

## 2. 方法命名与一句话定义

方法名称统一为：

> **BeamRisk-SFT：Beam-Risk Aligned Supervised Fine-Tuning**

中文名称统一为：

> **Beam 风险对齐监督微调**

一句话定义：

> BeamRisk-SFT 在标准生成式检索 SFT 中显式建模目标路径的真实 Beam 风险：由当前模型自行产生 Beam=10 轨迹，定位目标第一次被剪枝的深度，并分别对“进不了 Top-10”的首剪枝边界和“进入 Top-10 但排不到第一”的最终排序边界进行对齐训练。

方法名称只体现 SFT 优化，不把“冻结 SID”写入名称。SID 不变是公平实验的控制变量，不是方法卖点。

## 3. 研究问题

当前生成模型使用标准 Teacher-Forcing 交叉熵：在正确历史和正确目标前缀下，逐位置最大化 gold Token 概率。实际检索则使用 Beam Search：目标前缀必须在每一层都留在 Beam 中，一旦某层被剪掉，后续 Token 即使容易预测也无法挽回。

BeamRisk-SFT 重点回答：

1. 在 SID、数据、Prompt 和模型规模完全不变时，显式优化 Beam 中的目标前缀生存风险，能否超过标准 TIGER SFT；
2. 标准 token-level CE 与最终 HR/NDCG 的错位，是否是当前多组 SID 创新无法兑现静态收益的主要原因；
3. 能否用少量、真实、模型自挖掘的边界负例替代随机 Item 负例，在可控训练成本下同时改善 HR@10 与 HR@1/NDCG；
4. 训练过程中的 prefix 指标、Beam 生存率和最终检索指标之间能否形成可解释的因果链路。

## 4. 现有证据与方法动机

### 4.1 主要失败已经发生在三层目标 Bucket 覆盖

TIGER 固定 10,000 条 Validation、无约束 Beam=10 的 epoch-3 结果为：

| 指标 | 数值 |
|---|---:|
| HR@1 | 51.87% |
| HR@3 | 76.00% |
| HR@5 | 82.26% |
| HR@10 | 87.16% |
| NDCG@10 | 70.4190% |
| Valid ID Rate | 74.105% |
| 三层唯一 Bucket HR@10 | 88.06% |

在 1,284 条精确 POI Top-10 miss 中，1,194 条连正确的三层 `[S1,S2,S3]` Bucket 都没有进入候选，占全部 miss 的 92.99%。即使删除碰撞 Token 并在现有 Beam 内理想展开 Bucket，最多只新增 90 条命中，HR@10 上限只增加 0.90pp。详见 [TIGER Bucket 诊断](../docs/experiments/TIGER.md#exp-20260811-01-tiger-三层-bucket-诊断与-collision-上界)。

因此第一瓶颈是目标语义前缀没有在 Beam 中存活，而不是继续改变碰撞后缀。

### 4.2 Teacher-Forcing 局部正确不等于自由生成可召回

TIGER epoch 3 在 gold prefix 条件下：

| 位置 | Top-1 | Top-10 |
|---|---:|---:|
| S1 | 75.06% | 96.29% |
| S2 | 81.04% | 96.55% |
| S3 | 85.70% | 96.85% |
| S1—S3 累计 | 53.98% | 92.15% |

但自由生成的三层唯一 Bucket HR@10 只有 88.06%，比 gold-prefix 三级累计 Top-10 低 4.09pp。Teacher-Forcing 结果见 [固定 10k 逐层结果](../outputs/eval/sid_teacher_forcing_fixed10k_v1/tiger_e3/result.json)。

GHR EXP-09 的 gold-prefix 三级累计 Top-1 为 54.07%，略高于 TIGER 的 53.98%，自由生成 Bucket HR@10 却只有 85.43%，比 TIGER 低 2.63pp。这进一步证明局部分类准确率不能保证模型自身前缀下的 Beam 覆盖。详见 [GHR 逐层归因](../docs/experiments/GHR_SID.md#815-exp-20260823-01三组-ghr-epoch-3-逐层bucket-与后缀归因诊断)。

### 4.3 普通 eval loss 不是最终检索风险

标准 eval loss 同时平均：

- 真正决定检索路径的 S1、S2、S3、C；
- 几乎确定的 `<TARGET_POI>`；
- 几乎确定的闭合符和 EOS；
- 大量单例 Bucket 中容易预测的 C0。

它没有显式区分以下两种情况：

1. gold Token 概率提高了，但目标前缀仍在 Beam 第 11 名以后；
2. gold Token 概率提高了，并且刚好跨过 Beam 第 10 名边界。

两者可能得到相近的 CE 改善，但只有第二种能提高 HR@10。因此 BeamRisk-SFT 不以更低的总 loss 作为成功结论，而直接优化和报告 Beam 风险。

### 4.4 合法路径不是当前主矛盾

合法 Trie 约束将 TIGER epoch-3 HR@10 从 87.16% 提高到 87.99%，增量为 0.83pp。约束有工程价值，但不足以解释剩余 12.84pp 的 Top-10 miss。因此第一版不把 Masked Softmax 或强制合法生成作为主创新，只把合法路径用于诊断和候选属性审计。

## 5. 方法边界与冻结项

### 5.1 必须保持不变

BeamRisk-SFT 第一版严格复用 TIGER 的以下产物和协议：

- 初始模型：`models/Qwen3-0.6B-TIGER-Vocab-v1`；
- 该目录仅是 Qwen3-0.6B 加 TIGER Token 的 SFT 前初始化，不是已训练 TIGER checkpoint；
- SID：固定四个代码 Token `[S1,S2,S3,C]`；
- SID 映射：TIGER BGE-M3 RQ-VAE `1024×3 / epoch 20` 与原碰撞码；
- Tokenizer、Token ID、扩词表初始化；
- Train/Validation/Test 原始样本和业务键；
- history=10、Query、请求 GID 和消息模板；
- Assistant 目标序列与固定长度；
- `cutoff_len=512` 和现有 history-safe 截断协议；
- 主 CE 的 packed cache；
- 模型规模、全参数训练、BF16；
- 主评测的无约束 Beam=10、固定 Validation 10k；
- 合法路径约束只作为并列诊断，不替换无约束主指标。

### 5.2 明确禁止

第一版不允许：

- 从 TIGER `checkpoint-5571`、`checkpoint-11142` 或 `checkpoint-16713` 初始化；
- 先完整训练标准 TIGER 三轮，再追加一个对齐阶段；
- 修改 RQ-VAE、码本容量、SID 顺序、C 分配或目标长度；
- 增加 GID、关系实体或新的 target Token；
- 使用 E4、MMBERT 或其他 Query/POI 向量教师；
- 用 Validation 或固定 10k 挖掘训练负例；
- 为第一版引入 DPO、PPO、GRPO 或单独 reward model；
- 同时改变数据采样、模型结构、推理打分和 SFT loss，导致归因不清；
- 只因 combined training loss 更低就宣称方法有效。

## 6. BeamRisk-SFT 核心设计

### 6.1 基本记号

给定输入 `x`，目标 POI 的四 Token SID 为：

```text
y = [S1, S2, S3, C]
```

长度为 `t` 的目标前缀记为 `g_t = y[1:t]`。模型对前缀的累计分数为：

```text
Score_theta(g_t | x)
  = sum_{j=1..t} log P_theta(y_j | x, y_<j)
```

训练和挖掘的 Beam 宽度固定为 `K=10`，与主评测一致。

### 6.2 真实 Beam 轨迹

候选不由随机 Item、向量近邻或静态 Trie 兄弟产生，而由当前训练模型对 Train 请求执行真实无约束 Beam=10 得到。轨迹必须保存每一个 SID 深度的：

- 当前存活 Beam 前缀及累计分数；
- gold 前缀是否仍在 Beam；
- gold 前缀当前 rank；
- 第 10 名边界前缀及分数；
- gold 与第 10 名的 margin；
- 前缀是否符合位置 Token 类型；
- 前缀能否在全量 TIGER 目录中展开；
- 完整候选能否映射到 POI；
- 最终 gold rank。

候选选择属于 stop-gradient 的离散步骤。一个 epoch 内使用上一个 epoch checkpoint 挖掘并冻结的候选，下一轮再刷新，避免在每个训练 batch 内运行昂贵 Beam Search。

### 6.3 首剪枝生存损失

如果 gold 在深度 `t*` 第一次掉出 Beam，则：

- 正前缀：`g_t*`；
- 负前缀：该深度实际存活的第 10 名边界前缀 `b_t*`；
- `t*` 是因果上的首次失败点，后续前缀不再重复构造损失。

损失定义为：

```text
L_survive
  = softplus((Score_theta(b_t* | x)
              - Score_theta(g_t* | x)) / tau)
```

第一版固定：

- `tau = 1.0`；
- `margin = 0`；
- 不为 S1/S2/S3 手工指定位置权重；
- 每条请求最多一个首剪枝负例。

这个目标只要求 gold 越过当前 Beam=10 生存边界，不要求 gold 压过所有随机负例。它直接对应 HR@10，并减少过度压制其他可能相关 POI 的风险。

### 6.4 最终排序损失

如果 gold 已进入最终 Top-10，但 rank 为 2—10，则它不存在生存问题，改为修复最终排序：

- 正路径：完整 gold SID；
- 负路径：当前 Beam 中排在第一的错误完整 SID；
- 两条路径都按当前模型重新计算累计分数。

损失定义为：

```text
L_rank
  = softplus((Score_theta(y_negative | x)
              - Score_theta(y_gold | x)) / tau)
```

它主要服务 HR@1 和 NDCG@10。若 gold 已经 rank1，不增加排序损失，只保留标准 CE。

### 6.5 状态互斥的 Survive-then-Rank 策略

| 当前模型状态 | 边界样本 | 启用损失 | 目标 |
|---|---|---|---|
| gold 未进入 Top-10 | 首次失败 gold 前缀 vs 当层第 10 名 | `L_survive` | 先让目标活下来 |
| gold rank 2—10 | 完整 gold SID vs 当前 rank1 SID | `L_rank` | 再把目标排上去 |
| gold rank1 | 无 | 仅标准 CE | 防止过度优化已正确请求 |

两种风险损失对单条样本互斥，避免同一请求从多个前缀重复产生高度相关梯度。

### 6.6 严格目录等价保护

最终排序负例若能映射到 POI，需要执行严格等价检查。只有在目标与负例的规范化名称、地址、别名集合和精确坐标签名全部一致时，才视为当前输入难以辨别的目录重复，并跳过 `L_rank`。

约束如下：

- 只允许严格相等，不允许模糊文本相似阈值；
- 只过滤完整 POI 排序负例，不过滤中间前缀边界；
- 被过滤样本仍参加标准 CE；
- 单独报告过滤数量和请求占比；
- 第一版不继续为这些完全相同 POI设计细关系或新 SID。

该规则是训练噪声保护，不是新的碰撞表示方法。

### 6.7 总训练目标

主 CE 数据流保持不变，风险数据流只提供辅助梯度：

```text
L_total = L_CE + lambda_risk * L_BeamRisk
```

其中：

```text
L_BeamRisk =
  L_survive,  if state == miss@10
  L_rank,     if state == hit@10_not@1
  0,          if state == hit@1
```

第一版主配置固定：

```text
lambda_risk = 0.2
tau         = 1.0
margin      = 0
beam_size   = 10
```

不在第一轮同时搜索多个 `lambda`。只有主实验产生明确正向信号后，才允许做局部敏感性验证。

## 7. 从头训练的完整流程

BeamRisk-SFT 是一个从初始模型开始的三轮 SFT 流程，不是 TIGER SFT 之后的 post-training。

```text
相同扩词表初始模型
        |
        v
Epoch 1：标准 CE 建立可用生成策略
        |
        v
Train-only Beam=10 自挖掘 #1
        |
        v
Epoch 2：CE + 首剪枝/最终排序风险损失
        |
        v
Train-only Beam=10 自挖掘 #2
        |
        v
Epoch 3：CE + 刷新后的风险损失
        |
        v
固定 10k 无约束主评测 + 合法路径诊断
```

### 7.1 初始化

训练必须：

- 从 `models/Qwen3-0.6B-TIGER-Vocab-v1` 加载；
- 新建 optimizer、scheduler 和 RNG 状态；
- 使用 `seed=42`、`data_seed=42`；
- 总计划仍为三个 epoch；
- 不读取任何历史 SFT checkpoint 的模型或优化器状态。

### 7.2 Epoch 1：策略建立阶段

第一个 epoch 使用与 TIGER 完全相同的标准 CE。原因不是复用 TIGER，而是随机初始化的 SID Token 尚未形成有意义的 Beam 排序，此时自挖掘的边界负例大多是结构噪声。

Epoch 1 是 BeamRisk-SFT 自身训练过程的一部分，必须在新输出目录中由相同初始模型重新产生。它不能从现有 TIGER epoch-1 checkpoint 复制。

工程门禁：在相同初始权重、数据顺序和配置下，BeamRisk-SFT epoch 1 的 CE loss、checkpoint step 和标准评测应与 TIGER epoch 1 一致或只存在可解释的浮点误差。若出现系统性差异，不进入负例挖掘。

### 7.3 自挖掘 #1

使用 BeamRisk-SFT 自己的 epoch-1 checkpoint，只读取 Train：

1. 按业务键 SHA256 顺序构造最多 500,000 条确定性候选池；
2. 在候选池上执行与主评测一致的无约束 Beam=10；
3. 逐层记录 gold prefix、边界 prefix、分数和首次失败深度；
4. 对 hit@10-not@1 请求记录完整 gold 和 rank1 路径；
5. 对 hit@1 请求只统计，不输出风险 pair；
6. 目标输出约 100,000 个有效 pair；
7. 若扫描 500,000 条后 pair 少于 50,000，先停止正式 Epoch 2，重新评估风险池规模和训练集可挖掘性，不静默扩大数据。

固定候选池只由 Train 业务键决定，不按 Validation 表现选择。

### 7.4 Epoch 2：第一次风险对齐

Epoch 2 从同一 BeamRisk-SFT epoch-1 checkpoint 连续恢复模型、optimizer 和三轮 scheduler：

- 主数据流继续完整遍历原 packed Train，计算标准 CE；
- 辅助数据流读取自挖掘 #1 的 pair；
- pair 使用当前模型重新打分，不能把 epoch-1 参考分数当作训练常数；
- 候选身份在本 epoch 内冻结，不对 Beam 选择反向传播；
- 每条请求只计算一种风险损失；
- 辅助梯度不得替换或删除主 CE batch。

### 7.5 自挖掘 #2

Epoch 2 完成后，在完全相同的 Train 候选池上使用新的 epoch-2 checkpoint 重新生成轨迹：

- 已修复请求可能从 miss@10 变为 hit@10 或 hit@1；
- 新暴露请求可以进入风险集；
- 所有边界分数和负路径全部刷新；
- 保存与自挖掘 #1 的状态迁移矩阵；
- 不把两轮旧负例简单拼接。

### 7.6 Epoch 3：刷新后的风险对齐

Epoch 3 使用自挖掘 #2 的 pair，训练规则与 Epoch 2 相同。最终 checkpoint 是从相同基座连续完成三轮 BeamRisk-SFT 的结果，不追加第四轮，也不加载 TIGER 模型。

## 8. 数据流与训练开销设计

### 8.1 双数据流而不是重建全部 packed cache

主 CE 继续复用：

```text
data/sft/tokenized/tiger_bge_m3_1024x3_history10_query_gid_v1
```

风险 pair 单独保存原 Prompt、正路径、负路径和轨迹元数据。这样可以：

- 保持 TIGER 主 CE 数据与 packing 完全不变；
- 避免为少量 hard pair 重建 758 万行训练数据；
- 不让 auxiliary pair 改变主数据顺序；
- 独立核验每个风险样本的来源与分数。

### 8.2 辅助 batch 频率

第一版规划：

| 项目 | 主 CE 流 | BeamRisk 辅助流 |
|---|---:|---:|
| global batch | 512 | 128 个 pair |
| 启用 epoch | 1、2、3 | 2、3 |
| 频率 | 每个 optimizer step | 每 4 个 optimizer step 追加一次 |
| 是否替代主 batch | — | 否 |
| 负路径数/请求 | — | 1 |

辅助流每次包含正、负两条待评分路径，因此按序列数估算，每四个主 step 额外处理约 256 条序列，相对 4×512 条主序列约增加 12.5% 的前向/反向样本量。实际耗时需通过 smoke 实测，不能只按样本数推断。

### 8.3 基线训练超参数

除 BeamRisk 辅助流外，第一版沿用 TIGER：

```yaml
finetuning_type: full
num_train_epochs: 3.0
per_device_train_batch_size: 16
gradient_accumulation_steps: 8
global_batch_size: 512
learning_rate: 5.0e-5
optim: adamw_torch
weight_decay: 0.01
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0
bf16: true
cutoff_len: 512
packing: true
seed: 42
data_seed: 42
```

风险损失从 Epoch 2 开始，不重置学习率，不新建第二个 optimizer。Epoch 1—3 属于一个连续 scheduler。

## 9. 风险 pair 产物契约

每轮挖掘至少产生：

```text
beamrisk_sft/outputs/mining/epoch_1/
├── risk_pairs.jsonl
├── dataset/                         # Trainer 直接读取的 Arrow 数据
├── mining_metrics.json
├── catalog_path_diagnostics.json    # 结构合法但目录不可展开的路径审计
├── validation_report.json           # 全量 schema/泄漏/词表/首批次校验
├── state_transition_input.json
├── manifest.json
└── parts/

beamrisk_sft/outputs/mining/epoch_2/
├── risk_pairs.jsonl
├── dataset/
├── mining_metrics.json
├── catalog_path_diagnostics.json
├── validation_report.json
├── state_transition_from_epoch_1.json
├── manifest.json
└── parts/
```

`risk_pairs.jsonl` 每行至少包含：

```text
schema_version
sample_key
order_id
searchid
target_poi_id
risk_type                  # first_prune / final_rank
first_prune_depth          # 1/2/3/4 or null
gold_rank_at_depth
boundary_rank
reference_checkpoint
reference_positive_score
reference_negative_score
reference_margin
positive_token_ids
negative_token_ids
negative_structure_valid
negative_catalog_expandable
negative_poi_id
strict_duplicate_filtered
```

产物必须满足：

- 所有业务键来自 Train；
- 与 Validation 固定 10k 业务键交集为 0；
- 每个请求最多一个有效 pair；
- 正负前缀在首剪枝任务中长度相同；
- 完整排序路径可按 Token 逐项复算；
- 行数、分片、输入 checkpoint、数据文件和代码 SHA256 写入 manifest；
- 使用独立校验器复算状态分类和 margin；
- 不把 reference score 当成训练时 current score。

## 10. 工程实现

`beamrisk_sft/` 作为独立项目根目录。除复用的 TIGER 模型、Tokenized Train/Validation 和基线评测结果外，本方法新增的所有文件都只能写入这个目录。当前实现如下：

```text
beamrisk_sft/
├── README.md                         # 方法主文档与总进展
├── configs/
│   └── main_v1.yaml                  # 正式训练的唯一主配置
├── run_train_4x6000d.sh              # 4×6000D 完整流程入口
├── scripts/
│   ├── preflight.py                  # 训练前冻结协议与输入校验
│   ├── recover_stages.py             # 完整 checkpoint 的阶段记录恢复
│   ├── prepare_candidates.py         # Train-only 500k 候选池
│   ├── mine.py                       # 四卡真实 Beam 轨迹与 pair 构建
│   ├── finalize_mining.py            # 复用完整分片的 CPU-only 收口入口
│   ├── validate_risk_pairs.py        # 独立复算
│   ├── smoke_trainer.py              # 真实 Trainer/优化器/DDP 契约冒烟
│   ├── smoke_formal_risk_batch.py    # 正式首批风险数据的全词表反向冒烟
│   ├── train_stage.py                # 单 epoch 边界训练入口
│   └── summarize_final_eval.py       # 双口径结果校验与统一汇总
├── src/
│   └── beamrisk_sft/
│       ├── candidate_pool.py
│       ├── checkpoints.py            # 四卡状态完整性、链路审计与恢复
│       ├── catalog.py
│       ├── config.py
│       ├── tracing.py
│       ├── mining.py
│       ├── loss.py
│       ├── scoring.py
│       ├── risk_store.py
│       ├── smoke.py                  # 极小 Qwen3 双流训练与跨 rank 同步检查
│       ├── trainer.py
│       ├── evaluation.py
│       ├── validation.py
│       ├── workflow.py
│       └── schema.py
├── tests/
│   ├── test_tracing.py
│   ├── test_loss_and_scoring.py
│   ├── test_risk_store.py
│   ├── test_candidate_pool.py
│   ├── test_checkpoints.py
│   ├── test_catalog.py
│   ├── test_mining_finalization.py
│   ├── test_mining_states.py
│   ├── test_evaluation.py
│   ├── test_schema.py
│   ├── test_trainer.py
│   └── test_config.py
└── outputs/
    ├── mining/                       # 两轮 Beam 风险挖掘产物
    ├── sft/                          # checkpoint、日志与训练状态
    ├── eval/                         # 固定 10k 和泛化评测结果
    ├── logs/                         # 每个编排阶段的控制台日志
    ├── run_state/                    # 最终 checkpoint 与退出状态
    ├── smoke/                        # 小规模测试产物
    └── cache/                        # BeamRisk 专属临时缓存
```

路径规则：

- 新代码不写入仓库公共 `scripts/`、`src/` 或 `third_party/`；
- 新配置不写入公共 `configs/sft/`；
- 新 checkpoint、日志、挖掘文件和评测结果不写入公共 `outputs/`；
- 现有 `models/Qwen3-0.6B-TIGER-Vocab-v1`、TIGER tokenized cache 和固定 10k 只读引用；
- 所有启动脚本先定位 `poi_genret` 仓库根目录，再把 `beamrisk_sft/src` 加入 `PYTHONPATH`；
- 输出目录必须由配置显式解析到 `poi_genret/beamrisk_sft/outputs/`，禁止依赖启动时所在目录；
- 若 LLaMA-Factory 缺少扩展点，通过 `beamrisk_sft/src/beamrisk_sft/` 内的继承或包装实现，不直接修改第三方代码。

### 10.1 必须通过的最小测试

1. 人工构造 Beam：gold 在 S1/S2/S3/C 分别首次掉出时，定位深度正确；
2. gold 从未进入 Beam 时，不错误标记后续多个失败；
3. hit@10-not@1 与 hit@1 状态互斥；
4. 边界候选确实为第 10 名，Tie 使用稳定顺序；
5. 正负前缀同长度，累计 log-prob 与手工值一致；
6. `softplus(s_neg-s_pos)` 的梯度提高正路径、降低负路径；
7. risk weight 为 0 时，与标准 CE 数值一致；
8. 严格重复只过滤最终排序 pair；
9. Validation 业务键进入 mining 时立即报错；
10. 分布式四卡合并后的 pair 行序、去重和哈希稳定；
11. 四卡 checkpoint 必须包含 `rng_state_0.pth`—`rng_state_3.pth`，缺少任一 rank 立即失败；单卡才使用 `rng_state.pth`；
12. checkpoint 已完整但阶段 marker 未提交时，恢复器只能补 marker/manifest，不得重训或改写模型、optimizer、scheduler、RNG。
13. 真实 `CustomSeq2SeqTrainer.train()` 必须完成 CE、梯度累积、定时 BeamRisk backward 和 optimizer update；四进程 DDP 结束后各 rank 参数必须完全同步。
14. 位置 Token 合法但不存在于 TIGER 全量目录的自由生成路径必须保留为负例，不能被误判为 mapping 数据损坏；只有可映射路径参与严格重复 POI 过滤。
15. 每次正式 mining 校验后，必须用下一 epoch 将实际读取的首个四卡 global risk batch、正式 157,095 词表和变长路径完成一次真实前向/反向；CUDA 平台同时覆盖 BF16。

## 11. 正式实验设计

### 11.1 主实验只跑一组

第一轮不做大规模超参网格，只比较：

| 方法 | 初始化 | SID/数据 | 三轮训练 | 风险损失 |
|---|---|---|---|---|
| TIGER SFT | 相同扩词表初始模型 | TIGER 原样 | 标准 CE | 无 |
| BeamRisk-SFT | 相同扩词表初始模型 | TIGER 原样 | Epoch 1 CE；Epoch 2/3 CE+BeamRisk | `lambda=0.2` |

现有 TIGER epoch 1/2/3 结果作为已完成基线；BeamRisk-SFT 必须重新从初始模型跑满自己的三个 epoch。

### 11.2 训练前门禁

正式四卡训练前必须依次完成：

1. 轨迹器 32 条 CPU/单卡合成测试；
2. 1,000 条真实 Train Beam 风险 smoke；
3. 1,000 条 pair 独立复算；
4. 20-step 双数据流过拟合与梯度方向检查；
5. 四卡 20-step DDP smoke；
6. Epoch-1 CE-only 小规模对照，确认 risk 关闭时与 TIGER 训练一致；
7. 输出目录、端口、断点恢复和 miner 退出码检查；
8. 正式风险池与固定 10k 业务键零交集检查。

### 11.3 Checkpoint 与评测

保存 BeamRisk-SFT epoch 1/2/3。评测顺序：

1. 三个 epoch 的标准 Validation CE loss；
2. epoch 3 的固定 10k 无约束 Beam=10 主评测，并保存 Bucket 诊断；
3. epoch 3 的固定 10k 合法路径约束 Beam=10 并列诊断；
4. 自动校验两种模式使用同一个 checkpoint 和同一批 10,000 条业务键；
5. 自动汇总三轮 CE loss、两种生成式检索指标、约束增量和相对 TIGER 差值；
6. 主门禁通过后再运行 gold-prefix、完整 Beam 生存分析与四类泛化 10k；
7. 普通固定 10k 未通过时，不直接投入完整 Validation。

固定 10k 使用现有 TIGER 实验冻结的 Validation 业务键，不读取正式 Test。无约束和合法路径约束在训练结束后分别占用 GPU0、GPU1 并行执行，GPU2、GPU3 此时空闲；不修改已有 TIGER evaluator，只把其只读调用结果写入：

```text
beamrisk_sft/outputs/eval/main_v1/
├── unconstrained/
├── legal_path_constrained/
├── final_eval_summary.json
└── final_eval_summary.md
```

## 12. 指标与预注册门禁

### 12.1 主指标

无约束固定 10k 是第一主口径：

- HR@1；
- HR@3；
- HR@5；
- HR@10；
- NDCG@10；
- Valid ID Rate。

合法路径约束是诊断口径，不能替代无约束结果。

### 12.2 BeamRisk 专项指标

必须新增：

- S1/S2/S3/C 的 gold prefix Survival@10；
- 首剪枝深度分布；
- 每层 gold 到第 10 名的 margin 分布；
- miss@10、hit@10-not@1、hit@1 请求比例；
- Epoch 1→2→3 状态迁移矩阵；
- 风险 pair 中非法路径比例；
- 严格重复负例过滤比例；
- `L_survive`、`L_rank` 和标准 CE 分开记录；
- Bucket HR@1/3/5/10；
- Bucket→完整 POI 的各 K 损失。

### 12.3 第一门禁：超过 TIGER SFT

BeamRisk-SFT epoch 3 至少需要：

1. 无约束 HR@1 高于 51.87%；
2. 无约束 HR@10 高于 87.16%；
3. 无约束 NDCG@10 高于 70.4190%；
4. 三层唯一 Bucket HR@10 不低于 88.06%；
5. HR@3/5 不出现有意义回退；
6. paired bootstrap 或逐请求配对检验支持提升，不把几条样本波动直接称为有效；
7. 逐层 Survival 改善与最终指标方向一致。

只有某一个单项越过 TIGER，不算主方法成立。

### 12.4 第二门禁：更强论文基线

如果要声明通用地图生成式检索提升，还需与当前固定 10k 的完整论文方法包络比较。Centered GenPOI 当前为：

- HR@1 51.98%；
- HR@10 87.83%；
- NDCG@10 70.7430%。

第一阶段先验证 SFT 优化是否稳定超过同 SID、同数据、同无约束协议的 TIGER；通过后再判断能否超过完整论文方法包络。

## 13. 结果解释规则

| 结果现象 | 解释 | 后续动作 |
|---|---|---|
| Survival 与 Bucket HR 提升，完整 HR@1 不升 | 生存目标有效，最终排序目标不足 | 只检查 `L_rank` 和状态比例，不改 SID |
| HR@1 提升，HR@10 下降 | 排序过强，压制候选多样性或首剪枝样本不足 | 降低 final-rank 占比或恢复自然风险比例 |
| Teacher-Forcing 提升，Survival 不升 | 仍停留在局部 token 优化，Beam 对齐实现或候选已失效 | 检查 current-score 重算和候选刷新 |
| Valid ID 提升但 HR 不升 | 只学会合法格式，没有改善目标排序 | 不算方法提升 |
| combined loss 更低但 HR/NDCG 不升 | loss 标度变化或辅助任务稀释 | 不算方法提升 |
| Epoch 2 短暂提升、Epoch 3 回落 | 负例陈旧、风险过拟合或 CE/风险失衡 | 检查状态迁移和刷新结果，不直接增加 epoch |
| 仅碰撞桶改善 | 方法主要学习 C/局部排序 | 单独声明专项结果，不能称整体提升 |
| 普通 10k 提升、四类泛化下降 | 可能过拟合头部请求或自挖掘池 | 不进入完整 Validation，先查风险池分布 |

## 14. 与相关工作的关系及创新边界

### 14.1 论文启发

- [RIPOR](https://arxiv.org/abs/2311.09134)说明完整 ID 排序正确不足以保证每个前缀在 Beam 中存活，并提出 prefix-oriented ranking；BeamRisk-SFT 借鉴“前缀存活是独立目标”的判断，但不重建相关性 SID，也不使用外部相关性教师。
- [GLEN](https://aclanthology.org/2023.emnlp-main.477/)使用 prefix-aware dynamic negative sampling；BeamRisk-SFT 借鉴“负例必须接近真实前缀竞争”的思想，但不按前缀规则随机取 Item，而直接取当前 Beam 的第 10 名边界。
- [Sequence-to-Sequence Learning as Beam-Search Optimization](https://aclanthology.org/D16-1137/)指出 Teacher-Forcing 与 Beam 推理存在 exposure bias 和目标错位；BeamRisk-SFT 将这一思想具体化到固定长度 POI SID 的检索指标。
- [MSL](https://arxiv.org/abs/2504.04178)通过 Masked Softmax 排除不存在的 Item Token；其启发是语言模型全词表 CE 与推荐候选空间并不一致，但现有 TIGER 合法约束增益有限，所以 BeamRisk-SFT 第一版不把合法掩码作为主方法。
- [APAO](https://arxiv.org/abs/2603.02730)是最接近的工作，采用 prefix pointwise/pairwise loss 和 adaptive worst-prefix weighting。BeamRisk-SFT 不把“普通前缀加权”当作创新，而使用真实 Beam=10 轨迹、首个因果失败点和实际第 K 名边界，并区分 survive 与 final rank 两种检索状态。
- [PRO](https://arxiv.org/abs/2606.09241)从 prefix retention 分析 RQ 索引、码本和解码的错位；BeamRisk-SFT 只处理生成模型 SFT，不修改 tokenizer、SID 和推理打分。
- [Enhancing Generative Retrieval with Reinforcement Learning from Relevance Feedback](https://aclanthology.org/2023.emnlp-main.768/)使用 SFT、reward model 和强化排序三阶段。地图点击标签存在单正例与潜在多相关 POI，第一版不承担 reward model 噪声和额外采样成本，采用可直接验证的成对边界损失。

### 14.2 可以主张的创新

若实验成立，方法创新应凝练为：

> 提出面向生成式检索的 Beam 风险对齐 SFT。该方法利用当前生成器的真实 Beam 轨迹识别目标路径的首剪枝边界，并按请求状态执行“先存活、再排序”的互斥风险优化，使训练目标直接对应 Beam@K 生存和最终检索排序，同时不增加线上推理模块。

地图 POI 场景的专项设计包括：

- 230 万级目录上的真实路径可展开审计；
- 请求加权而不是目录等权的风险样本；
- 对完全相同目录实体的严格负例保护；
- 同时报告三层 Bucket 生存和精确 POI 排序；
- 无约束主协议与合法路径诊断分离。

### 14.3 不能主张的创新

以下表述不能单独作为创新结论：

- “首次给前面 SID Token 加更大权重”；
- “首次使用 prefix loss”；
- “首次使用 hard negative”；
- “首次让模型生成合法 SID”；
- “loss 比 TIGER 更低”；
- “SID 固定不变”；
- “训练后再用 DPO/Pairwise 微调”。

## 15. 风险与控制

### 15.1 训练集自挖掘可能过易

模型在 Train 上的命中率可能明显高于 Validation，导致可用 pair 不足。通过 500,000 条扫描上限和 50,000 条最低门禁先验证，不从 Validation 借样本。

### 15.2 离线负例会陈旧

一个 epoch 内候选冻结，但 Epoch 2 后强制刷新；第一版不跨两轮累计旧 pair。若 epoch 内状态变化过快，后续再研究半 epoch 刷新，不在第一版提前增加复杂度。

### 15.3 单点击标签包含潜在假负例

只取一个 Beam 边界而不是大量随机负例；只要求跨过边界而不是大 margin；对严格目录重复进行过滤；保留 CE；同时观察 HR@10 是否因过度压制而下降。

### 15.4 额外计算可能被误认为单纯算力收益

主实验先与 TIGER 比较方法结果并完整报告训练时间、Token 数和辅助前向量。如果主实验通过，后续补充计算量匹配的 CE-only 控制，确认收益不是简单增加训练计算。

### 15.5 自定义 Trainer 破坏 TIGER 主链路

BeamRisk 使用独立配置、输出目录和 Trainer；risk weight 设为 0 时必须复现标准 CE；不覆盖现有 TIGER cache、checkpoint 或评测产物。

## 16. 首轮实验之后才允许的消融

只有 BeamRisk-SFT 主实验通过第一门禁，才按价值排序运行：

1. 去掉最终排序损失，只保留首剪枝生存损失；
2. 去掉首剪枝，只保留完整路径排序；
3. 用随机负例替代真实 Beam 边界；
4. 不刷新 Epoch 2 风险对；
5. 去掉严格目录等价保护；
6. 计算量匹配的 CE-only 延长训练；
7. `lambda_risk` 在主值附近做最小三点敏感性验证；
8. 第二随机种子与完整 Validation。

第一轮不得同时跑完这些消融，避免在主假设未成立前消耗多组四卡训练。

## 17. 当前决策与运行顺序

已经冻结的方案决策：

1. 方法名为 BeamRisk-SFT；
2. 只改变 SFT 训练目标和训练编排；
3. SID、碰撞码、Tokenizer、Prompt、数据和评测协议复用 TIGER；
4. 从相同扩词表初始模型重新训练三轮；
5. 不加载任何 TIGER SFT checkpoint；
6. Epoch 1 为方法内部 CE 策略建立阶段；
7. Epoch 1/2 后分别用当前模型在 Train 上自挖掘；
8. Epoch 2/3 使用 CE + BeamRisk；
9. 每条请求只使用一个真实 Beam 边界负例；
10. 主评测仍为固定 10k 无约束 Beam=10；
11. `poi_genret/beamrisk_sft/` 是唯一新增写入根目录，代码、配置、测试和全部运行产物均不得散落到仓库其他目录。

正式入口按以下顺序执行，并对每一步失败立即停止：

1. 严格预检输入、TIGER 冻结配置和既有输出；
2. 用极小 Qwen3 在四 rank NCCL 下执行真实 Trainer 契约测试；
3. 一次性构造 Train-only SHA 最小 500k 候选池；
4. 从扩词表初始模型训练 Epoch 1；
5. 四卡挖掘 epoch-1 风险 pair 并独立校验；
6. 连续恢复 optimizer、scheduler 与 RNG，训练 Epoch 2；
7. 四卡刷新 epoch-2 风险 pair 并独立校验；
8. 连续恢复训练 Epoch 3；
9. 使用现有 TIGER 评测协议对固定 10k 执行无约束主评测和合法路径诊断。

平台统一使用下面这个无参数入口：

```bash
bash beamrisk_sft/run_train_4x6000d.sh
```

Launcher 会根据输出状态自动选择模式：没有 checkpoint 时从初始模型开始；检测到任意 checkpoint 或 epoch marker 时，自动切换为严格恢复模式。因此平台任务中断后仍可直接重提同一条无参数命令，不会误入从头训练。日志会明确打印 `resume_mode=fresh` 或 `resume_mode=auto`。

也可以显式指定恢复，行为与自动恢复相同：

```bash
bash beamrisk_sft/run_train_4x6000d.sh --resume
```

恢复模式首先在任何 GPU 训练或挖掘之前执行以下 CPU 门禁：

1. 核验 Transformers 当前版本的分布式 RNG 保存与读取命名契约；
2. 核验 checkpoint 名称、epoch、global step、三轮总 step 和 epoch-end eval loss；
3. 核验模型 safetensors header、optimizer/scheduler/training args 的 PyTorch ZIP64 容器；
4. 四卡模式严格要求四份 `rng_state_<rank>.pth`，不再错误寻找单卡文件；
5. 核验 checkpoint 必须形成连续的 Epoch 1→2→3 链；
6. 若 checkpoint 已完整而进程恰好在阶段 marker 提交前退出，只原子补写 manifest 和 marker，再跳过已完成 epoch；
7. 若 checkpoint 本身不完整，恢复模式直接报错，绝不把半截状态当作可续训点。

当前 `checkpoint-5571` 已通过上述深度审计：模型包含 310 个 tensor，模型文件 `2,405,366,248` bytes，optimizer `4,810,925,951` bytes，四份 rank RNG 状态齐全，`epoch=1.0`、`global_step=5571`、`max_steps=16713`。恢复记录位于 `outputs/sft/main_v1/stages/epoch_1.json`；再次启动会输出“Epoch 1 已完成，跳过”，随后核验并复用 epoch-1 mining 产物，再进入 Epoch 2。

当前 epoch-1 挖掘也已经完成并可恢复复用。四个 rank 分片合计有 120,000 条原始 pair，扫描了 236,432 条 Train 请求；确定性截取后正式数据为 100,000 条，其中 `first_prune=30,172`、`final_rank=69,828`。正式 JSONL SHA256 为 `ef2e25c4c1af35feed6de00706530431ab5900372c45323c0168001719e42986`。独立校验确认：与固定 Validation 10k 交集为 0、最大 Token ID `157075 < 157095`、最大 `prompt+path=511 <= 512`，下一阶段第一个风险事件为 optimizer step 5572，四卡 global risk batch 为 128。恢复启动检测到四个 rank 的 manifest/risk/state 分片齐全后，会改走 CPU-only `finalize_mining.py --resume` 校验并收口，不再启动四卡 Beam 生成；只有分片尚未全部产生时才进入分布式 miner。

本次正式挖掘暴露了一个需要明确区分的路径状态：Beam 可以生成位置类型均正确的 `[S1,S2,S3,C]`，但该组合不一定真实存在于 TIGER POI 目录。原实现错误地要求所有这类路径都能映射到 POI，因而在挖掘已经结束后报错。120,000 条原始 pair 中，共有 83,647 条结构合法的最终排序负例，其中 4,188 条（5.0068%，4,081 个唯一路径）目录不可展开；这不是 mapping 缺失，而是无约束生成本来就需要抑制的错误路径。修复后的规则为：

- 结构合法且目录可展开：映射到具体 POI，并执行严格目录等价过滤；
- 结构合法但目录不可展开：保留为有效 `final_rank` 负例，记录 `negative_catalog_expandable=false` 和 `negative_poi_id=null`；
- Token 位置结构也不合法：同样保留为无约束生成负例，并独立计数；
- 目录 mapping 查询默认仍保持严格模式，仅挖掘收口这个语义明确的调用点允许返回可映射子集。

最终 100,000 条训练 pair 中包含 3,537 条目录不可展开路径和 35 条结构非法路径；它们均已通过 schema、词表范围、长度和真实首批次读取校验。详细数据记录在 `outputs/mining/epoch_1/catalog_path_diagnostics.json` 与 `outputs/mining/epoch_1/manifest.json`。

Launcher 使用 `torch.distributed.run --standalone` 自动选择 rendezvous 端口，不再使用容易与平台其他任务冲突的固定 `master_port`。

## 18. 进展记录

| 日期 | 状态 | 记录 |
|---|---|---|
| 2026-09-02 | 方案已记录 | 方法命名为 BeamRisk-SFT；否决 TIGER epoch3 后补训；冻结从相同初始模型重新训练、Epoch 1/2 自挖掘、Epoch 2/3 风险对齐的主流程；尚未写代码或运行实验。 |
| 2026-09-02 | 目录边界已修正 | 独立项目根目录调整为 `poi_genret/beamrisk_sft/`；后续新增代码、配置、测试、挖掘产物、checkpoint、日志和评测结果全部收口在该目录，仓库已有 TIGER 资产仅只读引用。 |
| 2026-09-02 | v1.1 工程实现完成 | 已实现精确 live-Beam hook、首剪枝/最终排序互斥分类、Train-only 候选池、严格目录重复保护、当前模型路径重打分、双数据流 Trainer、连续三轮 scheduler、两轮刷新挖掘、独立校验与 4×6000D launcher。CPU 预检、真实 Tokenizer 布局、真实 PyArrow/TIGER 映射、极小 Qwen3 live-Beam 接口、静态检查和 21 项单测已通过；当前节点没有可见 GPU，因此尚未进行 CUDA smoke 或正式训练。 |
| 2026-09-02 | 最终双口径评测接入 | 四卡入口在 epoch 3 后自动并行运行固定 Validation 10k 的无约束与合法路径约束 Beam=10；严格核验 checkpoint、业务键和子集 SHA 一致，并输出训练 loss、检索指标、约束增量及相对 TIGER 差值的 JSON/Markdown 汇总。 |
| 2026-09-03 | Epoch 1 完成、续跑修复 | 四卡正式 Epoch 1 完成：`checkpoint-5571`、train loss `0.8616279203`、Validation loss `0.3851958513`。任务随后因旧检查器只接受单卡 `rng_state.pth` 而误报失败，实际四卡正确产物为 `rng_state_0.pth`—`rng_state_3.pth`。已按 Transformers 4.52.4 的真实保存/加载逻辑修复，并增加保存后 DDP barrier、四卡状态契约测试、深度 checkpoint 预检、连续链审计及 marker 自动恢复；现有 checkpoint 已验证并恢复，可用 `--resume` 跳过 Epoch 1。静态检查、Shell 语法检查和全部 25 项单测通过。 |
| 2026-09-03 | 无参数自动续跑 | 首次重提仍使用平台原有无参数命令，严格预检因未收到 `--resume` 在 GPU 工作前主动终止，checkpoint 未受影响。Launcher 已消除这一人工参数依赖：检测到 checkpoint/epoch marker 后自动追加恢复语义、深度校验并跳过已完成阶段；同一条无参数平台命令即可安全重提。 |
| 2026-09-03 | epoch-1 挖掘收口修复 | 四卡已完整生成的 120,000 条 pair 在收口时遇到 4,081 个“位置 Token 合法、全路径不在目录”的自由生成组合。修复了“结构合法必然可映射”的错误假设：不可展开路径作为真实无约束负例保留，严格重复过滤只作用于可映射 POI。复用原四卡分片在 CPU 完成正式收口，得到并验证 100,000 条风险 pair，不需要重新挖掘。 |
| 2026-09-03 | Stage-2 Trainer 防迟发门禁 | 新增真实 Qwen3 `BeamRiskTrainer.train()` 测试，覆盖 CE、与正式配置一致的 `gradient_accumulation_steps=8`、第 4 optimizer step 风险注入、backward 和参数更新；单进程测试通过。随后在四进程 CPU DDP 下验证每 rank 恰好 1 个风险事件、2 个本地 pair，训练后跨 rank 参数最大差异为 0。平台 launcher 现会在任何长训练/挖掘前自动运行同一套四 rank CUDA/NCCL BF16 smoke，失败即提前退出；目录不可展开负例的完整收口回归测试也已加入，当前共 30 项单测。 |
| 2026-09-03 | 正式风险首批次已反向验收 | 按 Epoch 2 的真实调度读取 optimizer step 5572 的首个全局 128 pair（四个 rank 各 32、micro batch 8），用 157,095 正式词表完成四进程 Qwen3 前向、BeamRisk backward 和 optimizer step；四 rank 均为有限 loss，最长实际序列 379，更新后参数最大差异为 0。平台在每轮 mining 后会自动以 CUDA/NCCL BF16 重跑同一门禁，再进入下一 epoch。 |
| 2026-09-04 | v1 完成但未通过 TIGER 门禁 | 三轮训练及固定 10k 双口径评测完成；无约束 HR@1/HR@10/NDCG@10 分别为 50.61%/86.53%/69.511%，低于 TIGER 51.87%/87.16%/70.419%。诊断确认主要问题为风险注入步持续梯度裁剪、softplus 无边界停止、91% `final_rank` 跨三层语义桶，以及关系型假负例。当前冻结 v1、不修改代码，完整分析见 `BEAMRISK_V1_RESULT_ANALYSIS.md`。 |
