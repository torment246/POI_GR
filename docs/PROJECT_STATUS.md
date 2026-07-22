# 项目状态

更新时间：2026-07-22。

## 当前阶段

仓库处于真实 POI 数据生成式检索项目的重新启动阶段。

- 旧数据、旧代码、旧配置和旧实验记录已经清除。
- 本地已准备清洗后的北京 POI 数据，位于 Git 忽略的 `data/`。
- Query—目标 POI 训练样本、切分清单和字段契约尚未完整交付。
- 本地已准备 Qwen3 生成模型和 Embedding 候选模型，位于 Git 忽略的 `models/`。
- 已建立通用 POI 文本向量流水线和 RQ-VAE SID 训练评估流水线、配置、轻量测试和依赖清单。
- 已建立 Query 编码、Gate 0 对齐检查、Faiss GPU 精确召回和分桶指标评估流水线。
- 第一版技术路线以根目录 `方案.md` 为准。

## 已确定原则

- 生成模型首版使用 Qwen3-0.6B，采用非思考模式。
- 先完成数据契约和最小验证，再开发特征、编码、SFT 与约束解码。
- 每次只交付一个可独立核验的最小步骤，用户确认后再继续。
- 所有正式实验只记录在 `docs/EXPERIMENT_LOG.md`。
- 真实数据、模型和实验产物不进入 Git。
- Qwen3-Embedding-0.6B 全量配置采用 batch size 64、编码缓冲区 8192；2,337,178 条北京 POI 全量向量已完成并通过校验。
- Qwen3-Embedding-4B 全量配置采用 batch size 64、编码缓冲区 8192 和 PyTorch SDPA；2,337,178 条北京 POI 全量向量已完成并通过校验。
- Qwen3-Embedding-0.6B 无 Instruction 的 10,000 条订单召回基线已完成；使用 GPU `IndexFlatIP` 对 2,337,178 个 POI 做精确 Top-20 检索，Hit@1/10/20 为 12.03%/28.25%/34.51%。
- Qwen3-Embedding-4B 无 Instruction 对照已按相同数据和检索口径完成，Hit@1/10/20 为 9.72%/21.18%/25.12%；本轮不自动选择模型。

## 开工前仍需确认

1. POI 数据字段、类型、空值含义和稳定主键。
2. Query 样本字段、目标 POI 标签来源和负样本定义。
3. 训练、验证、测试切分以及长尾、新 POI、no-result 专项集合。
4. 数据版本标识和本地可复现校验方式。

## 已完成的最小步骤

- 按排序后的 Spark JSON 分片流式读取 `poi_id` 和 `text`。
- 构建与向量行严格同序的 `poi_ids.jsonl`。
- 编码结果顺序写入标准 NPY，完成后可通过 mmap 读取，避免 OrangeFS 上直接写 memmap 的性能问题。
- 输出数据指纹、模型参数、环境版本和断点进度。
- 输出编码吞吐和 CUDA 峰值 allocated/reserved 显存。
- 模型路径和输出维度由配置决定，切换 4B 不需要修改核心代码。
- 使用合成数据验证顺序、重复 ID 拒绝、输出 shape/dtype 和断点续写。
- 使用有限缓冲区扩大长度排序范围，同时保持 POI ID 与向量行顺序不变。
- 使用 mmap 读取 0.6B Embedding，避免把全量向量加载到内存。
- 实现三层 EMA 残差码本、dead-code 重置、可恢复 checkpoint 和确定性训练/验证切分。
- 实现全量 SID 导出、重建误差、余弦相似度、码本使用率、perplexity、唯一性和碰撞评估。
- 262,144 条真实向量的 5 epoch smoke 已通过，未再出现码本全部塌缩；该结果不作为正式实验结论。
- 完成 2,337,178 条 0.6B Embedding 的首轮全量 RQ-VAE 训练和 SID 导出；训练、恢复、评估和产物校验链路均已跑通。
- checkpoint 选择指标已从包含 commitment loss 的总损失改为验证集重建 MSE；复用第 20 轮 checkpoint 完成全量 SID 重新导出，没有重新训练。
- 修正后的全量重建余弦为 0.831692，唯一 SID 比例为 52.15%，碰撞 POI 比例为 66.15%；三层码本均使用全部 256 个 code。
- 完成 10,000 条固定评测订单的 Gate 0：订单、目标 POI、全量向量、POI ID、原始分片顺序和指纹全部通过检查。
- 完成 E1：复用 POI 编码器配置生成原始 Query 向量，通过 Faiss GPU float32 `IndexFlatIP` 输出 Top-20、目标排名、整体指标和 Query 长度分桶指标。
- 完成 E2：保持 E1 的评测数据、顺序、无 Instruction 输入和指标逻辑不变，使用 4B 向量完成 GPU float32 `GpuIndexFlatIP` 精确 Top-20 召回及逐行独立核验。

## 下一最小步骤

由用户核验 E1/E2 对照：当前相同无 Instruction 口径下 0.6B 的整体和各长度分桶指标均高于 4B，但暂不自动选择最终模型。确认后再决定是否运行 Query Instruction 对照。RQ-VAE 方面仍需确认当前 SID 是否结合后续 GID 使用，或者先增加碰撞约束及对照实验。

在用户确认前，不进入 GID、SFT 或后续训练流程，也不自动运行新的 Embedding 对照实验。
