# RQ-VAE 与 SID 实验

本文件记录从 POI Embedding 到离散 Semantic ID 的正式实验，包括 RQ-VAE、RQ-KMeans、碰撞分析、GID 与完整 SID。SFT 和生成式检索实验不记录在本文件中。

## 当前状态

- SID 阶段尚未形成当前有效的实现或实验基线。
- 旧 RQ-VAE 实现、配置、测试和本地产物已停止复用，后续不得从旧结果推断新实验结论。
- 已完成的 0.6B/4B POI Embedding 保留，可作为重新设计 SID 实验时的候选输入。
- 新实验必须先明确最小目标、数据范围、初始化方式、训练与选择指标，再实现代码和运行 smoke test。
- 只有实际运行的正式实验才在本文件追加 `EXP-YYYYMMDD-NN` 记录，并同步更新实验总索引。

## 重新开始前需要确认

1. 首轮使用 0.6B 还是 4B Embedding，以及选择依据。
2. 先建立 RQ-VAE 基线，还是同时设计 RQ-KMeans 对照。
3. K-Means 初始化样本量、采样方式和随机种子。
4. RQ 层数、每层码本大小、latent 维度和损失定义。
5. checkpoint 选择指标，以及重建质量、码本使用率和 SID 碰撞之间的优先级。
6. 是否在本阶段加入 Geohash GID，还是先只评估纯语义 SID。
