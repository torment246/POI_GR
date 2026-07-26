# 项目状态

更新时间：2026-07-26。

## 当前阶段

北京城市第一版生成式 POI 检索链路已完成从 POI Embedding、Semantic SID、Final PID、SFT 到固定子集 Trie 约束评测的最小闭环。

- Qwen3-Embedding-0.6B/4B 已完成 2,337,178 条北京 POI 全量编码和固定 10,000 条订单精确召回对照；当前下游 SID 使用 0.6B Embedding。
- 三种 RQ-VAE 码本容量均训练 50 epoch，并完成 epoch 20/30/40/50 共 12 组全量 SID 评估；当前语义 SID 基线为 `1024×3 / epoch 20`。
- Geohash6 GID 将 PID 唯一率提升至 84.89%；确定性 Dedup Code 使 2,337,178 条 Final PID 全局唯一。
- 订单主任务按时间切分为 7,586,410/597,421/606,682 条 Train/Valid/Test，PID 标签匹配率为 100%。
- Qwen3-0.6B 已完成两轮全参数 SFT；固定 10,000 条 Validation、Beam=10 的约束生成评测中，2.0 epoch 的 HR@1/HR@10/NDCG@10 为 45.77%/84.83%/65.99%。

## 已确定原则

- 第一版技术路线以根目录 `方案.md` 为唯一基线。
- 生成模型使用 Qwen3-0.6B 非思考模式，训练目标为 Query + 用户 Geohash6 到唯一 Final PID。
- Trie 使用全部有效 Final PID 限制生成前缀，保证生成候选能够映射到真实 POI。
- 实验索引维护在 `docs/EXPERIMENT_LOG.md`，Embedding、SID/PID、SFT 详情分别归档到对应阶段文档。
- 真实数据、模型、Embedding、checkpoint、日志和实验大产物不进入 Git。

## 已完成的最小闭环

1. POI 文本流式编码及 POI ID 行级对齐。
2. Query 编码、Gate 0 对齐检查和 Faiss GPU 精确召回。
3. RQ-VAE 训练、SID 导出、碰撞指标、前缀类别纯度和热点桶分析。
4. Geohash6 GID、九层 base PID 及确定性 Dedup Final PID。
5. SFT Messages 数据、时间切分、特殊 Token 词表和 Tokenized Cache 预检。
6. Qwen3-0.6B 两轮全参数 SFT 和四个 checkpoint 完整 Validation Loss。
7. 紧凑 Final PID Trie、约束 Beam Search 和固定 10,000 条 Validation 生成式召回评测。

## 当前限制

- 生成式评测只覆盖固定 10,000 条 Validation 和 Beam=10，尚未运行完整 Validation、Beam 规模对比或 Test。
- 当前只证明 `checkpoint-9158` 在固定子集口径下最优，尚未生成正式冻结配置。
- RQ-KMeans、增量 Dedup 分配、no-result 增量分析和 GRPO/DPO 尚未开始。
- Prefix3 类别纯度受到大量单例桶影响，不能脱离桶大小和 Micro Purity 解释为完整语义层级质量。
- 当前 GID 固定为 Geohash6，边界效应和跨区域检索仍需专项验证。

## 下一最小步骤

先审核固定子集评测结果，并冻结正式 checkpoint、Beam 和评测口径；再单独决定是否运行完整 Validation、Beam 对比和 Test。上述基线确认后，再评估 no-result 增量、RQ-KMeans、增量更新或偏好优化。
