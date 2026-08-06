# 项目状态

更新时间：2026-08-06。

## 当前阶段

北京地图 POI 生成式检索已完成 V1 Baseline、TIGER 和 GenPOI 的固定 10,000 条 Validation 最小闭环；GNPR-SID 已完成加入 Diversity Loss 后的三容量训练、2,337,178 条全量 SID 评估和 512×3 唯一标识构建。实验详情已按方法归档，不再拆成 Embedding、SID 和 SFT 三类文档。

| 方法 | 当前进展 | 固定子集结果 | 状态 |
|---|---|---|---|
| V1 Baseline | Qwen3-Embedding-0.6B → RQ-VAE → Geohash6/Dedup → 2 epoch SFT → Final PID Trie | epoch 2：HR@1/HR@10/NDCG@10 = 45.77%/84.83%/65.99% | 已完成 |
| TIGER | BGE-M3 → RQ-VAE → collision token → 历史 10 条 SFT → 无约束 Beam | epoch 3：51.87%/87.16%/70.42%，Valid ID Rate 74.105% | 已完成 |
| GenPOI | BGE-M3 + GeoPE32 → RQ-VAE → Geohash6/Dedup → 历史 10 条 SFT → SSP+TCG | 旧版 epoch 3：50.85%/87.13%/69.8751%，Valid PID Rate 100% | 旧版闭环完成；Centered 修正版训练准备完成 |
| GNPR-SID | BGE-M3 + 类别 + PlusCode6 → Diversity Loss RQ-VAE → 碰撞 SID Dedup → 历史 10 条 SFT | 512×3 base SID 唯一率 77.7969%，追加 `d0-d222` 后标识唯一率 100% | 全量数据、扩词表和 Cache 完成；GPU smoke/正式 SFT 待运行 |

上述生成指标均使用同一组按 `order_id + searchid` 对齐的 10,000 条 Validation 业务键和 Beam=10，但三种方法的约束机制不同：V1 使用完整 Final PID Trie，TIGER 按论文复现口径无约束生成并将非法 ID 计为 miss，GenPOI 使用 SSP 预测地理前缀深度和 TCG Trie。

## 共享数据与已确定输入

- 北京 POI 目录包含 2,337,178 条在线可检索 POI，POI ID 全局唯一；POI 文本使用名称、地址和别名。
- 订单与历史序列数据共 8,790,513 条目标订单，Train/Valid/Test 严格按 2026-07-01—12、07-13、07-14 切分为 7,586,410/597,421/606,682。
- 用户历史来自 2026-04-01—06-30，最多保留最近 10 条；历史覆盖率 72.0157%，平均长度 4.9153，空历史样本保留。
- Qwen3-Embedding-0.6B、4B 和 BGE-M3 已完成全量 POI 编码与相同 10,000 条订单精确召回；BGE-M3 Hit@10/20 为 32.78%/40.81%，为当前共享文本编码器首选。
- 真实数据、模型、Embedding、checkpoint、日志和大体积实验产物均保留在 Git 之外。

## 方法进展

### V1 Baseline

- Qwen `1024×3 / epoch 20` 的三层 SID 全量唯一率为 68.5345%；追加 Geohash6 后为 84.8916%，确定性 Dedup 后 2,337,178 条 Final PID 全局唯一。
- Qwen3-0.6B 完成两轮全参数 SFT。固定 10,000 条 Validation 中，2.0 epoch checkpoint 为四个节点最优，生成结构和 Final PID 合法率均为 100%。
- 尚未运行完整 Validation、其他 Beam 或 Test；RQ-KMeans、no-result 增量和 GRPO/DPO 尚未开始。

### TIGER

- BGE-M3 三容量 20 epoch 对比选择 `1024×3`，全量 SID 唯一率为 71.6766%；追加 `C0–C305` 后四层 item identifier 全局唯一。
- 历史序列 SFT、5,426 个扩展 Token、正式 packed Cache 和四卡 RTX PRO 6000D 三轮训练均完成。
- epoch 3 在固定子集检索指标最好，但约 25.90% 的 Beam 候选无法映射到冻结 identifier 表；该差异按论文复现口径保留，不使用 Trie 后处理掩盖。

### GenPOI

- 32 个北京 K-Means 地理锚点的 GeoPE 已覆盖全量 POI；三容量对比选择 `1024×3`，全量 SID 唯一率 73.1125%，Geohash6 + Dedup 后最终 PID 全局唯一。
- 历史序列 SFT、3,626 个扩展 Token、正式 packed Cache、四卡三轮训练、SSP 预测和 TCG Trie 评测均完成。
- epoch 3 固定子集指标最好，三轮结构合法率和 PID 有效率均为 100%。GeoPE 第一层只使用 59 个粗前缀，是否具有有效地理重排仍需通过专项消融判断。
- Centered GeoPE32 + TIGER-RQVAE 修正版已解决第一层码本低利用率，冻结的新 1024 PID 全局唯一；对应 8,790,513 条 Messages 和 2,935,702/214,860 条 packed Train/Validation Cache 已完成，复用既有 3,626 Token 模型，6000D/A6000/A100 四卡三轮入口可直接启动。该版本尚未训练和评测，不覆盖上面的旧版闭环结论。

### GNPR-SID

- 8192 用户哈希行为版覆盖 703,306 条交互 POI；`1024×3` 在该输入内唯一率为 94.2369%，但最大碰撞主要来自不同用户映射到相同哈希桶，未冻结为正式下游标识。
- 当前 content-geo 输入为全量 BGE-M3 文本、402 类别和 766 PlusCode6 区域，采用 `1.0/0.25/0.25` 特征权重和 `2.140625` 重构损失权重，按 batch 流式构造 2192 维输入。
- GNPR 发布代码实际启用的 utilization Diversity Loss 已加入共享 RQ-VAE：外部权重 0.25、内部 scale 0.05，第 7 epoch 起启用；论文文字中的 compactness 项在发布实现中被注释，本轮未自行恢复。
- `256×3/512×3/1024×3` 已完成 20 epoch 和全量 SID 导出，全量唯一率为 42.6376%/77.7969%/3.1313%，碰撞 POI 比例为 78.1474%/35.5242%/98.8941%。最终选择 512×3；1024 从 epoch 17 起坍塌，已由全量最大桶 4,757 和低码本熵确认，不进入下游候选。
- 由于原任务未保存1024 epoch 16权重，已在A6000上按相同配置复跑至16轮并完成全量评估：唯一率63.1798%、碰撞POI 51.9524%，低于512；其前两层类别纯度更高，但第三层只使用464/1024个码且跨GPU轨迹未稳定复现，因此不替换512。
- GNPR-512 的 base SID 唯一率比 TIGER 已选 1024 高 6.12 个百分点，但 GNPR 前两层类别纯度低于 TIGER，说明其优势主要是减少最终碰撞，不代表粗粒度语义层级更清晰。
- 已按论文和作者 V1 数据格式，仅对 830,263 条碰撞桶 POI 追加 `d` Token；1,506,915 条单例保持三 Token。最终 2,337,178 条 GNPR identifier 全局唯一，完整复跑映射 SHA256 均为 `3cee68db4f1360b1103452e0e9d1c5e49edf5f11bf70e38b514c1037911abde2`。
- 地图检索 SFT 第一版已确认不输入时间和用户 ID，与 TIGER 保持历史 `Query/GID/POI identifier`、当前 `Query/GID` 的同口径输入；`event_time/create_time` 仅用于窗口、因果和切分校验。论文附录的多段裁剪与 20% 历史位置填空增强本版不启用，以保持等样本量比较。
- GNPR SFT 构建器使用 `gnpr_ids.npy` mmap 和数字 POI ID 紧凑索引，已完整扫描 32 个 003 分片并原子写出 8,790,513 条 Messages；Train/Valid/Test 为 7,586,410/597,421/606,682，空 Query 为 0，历史覆盖率 72.0157%、平均长度 4.9153，与 TIGER 的切分和样本量一致。
- GNPR 专属词表已在 Qwen3-0.6B 上增加 1,805 个普通原子 Token，词表从 151,669 扩展至 153,474；正式 packed Train/Validation Cache 为 2,758,797/201,947 条，固定 smoke Cache 为 3,661/670 条，Assistant 目标截断数为 0。数据、缓存重载和四卡 6000D/A100 三轮入口均已核验，尚未运行 GPU smoke 或正式 SFT。

## 当前限制

- 三个已完成方法只在固定 10,000 条 Validation 和 Beam=10 上对比，不能替代完整 Validation 或 Test。
- V1、TIGER 和 GenPOI 的输入格式、唯一标识和约束解码不同，当前结果用于比较完整方法链路，不能把差异归因于单一模块。
- GNPR content-geo 是面向地图全目录覆盖的适配版，不等同于保留完整时间和协同用户特征的原论文严格输入。
- 共享机器曾发生 memory-cgroup OOM；全量数据处理必须继续使用 mmap、分片流式聚合或 YARN，避免同时运行高内存 Spark 和 GPU 评测。
- Geohash6 的边界效应、跨区域召回、长尾/新增 POI 和 no-result 增量仍未形成正式专项实验。

## 下一最小步骤

启动一个与已申请 GPU 类型对应的 Centered GenPOI 四卡三轮入口；确认日志读取新版 Tokenized Cache，并在 epoch 1/2/3 形成完整 checkpoint。GNPR 保持为独立任务，下一步先运行 20-step GPU smoke，核验通过后再启动正式三轮 SFT。
