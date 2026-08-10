# 项目状态

更新时间：2026-08-08。

## 当前阶段

北京地图 POI 生成式检索已完成 V1、TIGER、GNPR-SID 以及旧版与 Centered GenPOI 的固定 10,000 条 Validation 最小闭环。Query-Augmented Relational SID 创新线已冻结 E4/α=0.30/β=0.85，并完成 RQ-VAE/RQ-KMeans 对称 `1024³` `2×2` 与 BGE/E4 三档 30-bit 容量分配验证；E4-4096 是当前纯 SID 结构最优，E4-2048 是结构—生成复杂度 Pareto 主候选。完整方法实验按方法归档，后续 Query 增强向量实验单独记录。

| 方法 | 当前进展 | 固定子集结果 | 状态 |
|---|---|---|---|
| V1 | Qwen3-Embedding-0.6B → RQ-VAE → Geohash6/Dedup → 2 epoch SFT → Final PID Trie | epoch 2：HR@1/HR@10/NDCG@10 = 45.77%/84.83%/65.99% | 已完成 |
| TIGER | BGE-M3 → RQ-VAE → collision token → 历史 10 条 SFT → 无约束 Beam | epoch 3：51.87%/87.16%/70.42%，Valid ID Rate 74.105% | 已完成 |
| GenPOI | BGE-M3 + Centered GeoPE32 → TIGER-RQVAE → Geohash6/Dedup → 历史 10 条 SFT → SSP+TCG | Centered epoch 3：51.98%/87.83%/70.7430%，Valid PID Rate 100% | 旧版与 Centered 修正版闭环均完成 |
| GNPR-SID | BGE-M3 + 类别 + PlusCode6 → Diversity Loss RQ-VAE → 碰撞 SID Dedup → 历史 10 条 SFT → 无约束 Beam | epoch 3：HR@1/HR@10/NDCG@10 = 49.66%/83.07%/67.2390%，Valid ID Rate 78.808% | 已完成 |

上述生成指标均使用同一组按 `order_id + searchid` 对齐的 10,000 条 Validation 业务键和 Beam=10，但四种方法的约束机制不同：V1 使用完整 Final PID Trie，TIGER 和 GNPR 按论文复现口径无约束生成并将非法 ID 计为 miss，GenPOI 使用 SSP 预测地理前缀深度和 TCG Trie。

## 共享数据与已确定输入

- 北京 POI 目录包含 2,337,178 条在线可检索 POI，POI ID 全局唯一；POI 文本使用名称、地址和别名。
- 订单与历史序列数据共 8,790,513 条目标订单，Train/Valid/Test 严格按 2026-07-01—12、07-13、07-14 切分为 7,586,410/597,421/606,682。
- 用户历史来自 2026-04-01—06-30，最多保留最近 10 条；历史覆盖率 72.0157%，平均长度 4.9153，空历史样本保留。
- Qwen3-Embedding-0.6B、4B 和 BGE-M3 已在无 Train 订单泄漏的固定 SFT Validation 10,000 条上完成全量精确召回；BGE-M3 Hit@10/20 为 32.39%/40.25%，为当前共享文本编码器首选。
- 真实数据、模型、Embedding、checkpoint、日志和大体积实验产物均保留在 Git 之外。

## 方法进展

### RQ-KMeans SID

- 原始 BGE 与冻结 E4 在对称 `1024³` 下的 RQ-VAE/RQ-KMeans `2×2` 已完成；E4+RQ-KMeans 的唯一率为 76.1573%，但 E4 与量化器算法的全局结构收益存在明显重叠。
- 原始 BGE 与 E4 的 `1024³`、`2048×1024×512`、`4096×1024×256` 三种 30-bit 布局均已完成；容量前移到 S1 可同时减少碰撞并改善中深层类别纯度。
- E4-2048 的唯一率/碰撞 POI/最大桶为 79.8731%/31.4021%/143，仅比 `1024³` 多 512 个基础 token 即取得 3.7158 个百分点唯一率增益，冻结为 SFT 主模型；E4-4096 以 82.7331%/27.9655%/125 作为容量上界，BGE-2048 作为同布局 Embedding 消融。

### Query-Augmented Relational SID

- 已复用 V1 SFT 固定 10,000 条 Validation 业务键，生成独立的 Embedding 评测数据 `data/eval/embedding_retrieval/sft_validation_10k_v1/eval.jsonl`；输出只保留 `sample_id/order_id/searchid/query/poi_id/split`。
- 已完整扫描 7,586,410 条 Train 和 597,421 条 Validation，复合业务键、`order_id`、`searchid` 的 Train/Eval 重叠数均为 0；评测数据 SHA256 为 `5557769526bd1a41a86adac5efb26eb2bf878fed8b2e2dad9f58afe93663e713`。
- 已从固定 Train 流式提取原始 `Query → target_poi_id`，按 BLAKE2b-64 原始 Query 哈希构建 256 个 ZSTD Parquet 分片；7,586,410 行守恒、逐行哈希错位为 0。严格 Train-only 口径包含 1,408,778 个唯一原始 Query、2,063,175 个唯一 Query–POI 对，覆盖 491,213 个目标 POI，占全量目录 21.0174%。
- 已逐分片生成连续 Query ID、`count(q,i)`、`df(q)` 和 POI 覆盖统计；全量聚合耗时 174.63 秒、峰值 RSS 221.80 MiB，四项行数/计数守恒均通过，完整 SHA256 复核可复用。固定评测中 7,215/7,520 个唯一目标 POI、9,693/10,000 条样本有 Train Query 覆盖。
- 三种原始 POI 向量已在单卡 A100 上按相同协议重评测；BGE-M3 的 Hit@1/10/20、MRR@10 和独立补算 NDCG@10 为 11.99%/32.39%/40.25%/17.7627%/21.2113%，仍为三者最优。
- 已冻结 10,000 条 BGE Query 向量，SHA256 为 `98e2032b86b76cf42991dc1d1f4967ff456661fb32fbae185652163c85bf70e0`；行映射 SHA256 为 `d512aa206468ea672b473ce38c353eca703c64289ac890d9a935cd3c9b2f855f`。
- 已完成 1,408,778 条唯一 Train Query 的 BGE-M3 正式编码：行号严格等于连续 `query_id`，shape 为 `[1408778, 1024]`、dtype 为 float16、SHA256 为 `e300e248a274a0febbcfb94b0b457f76916814564316029e258fd2bfe692ad67`，独立 `--validate-only` 复检通过。任务在开发机 RTX A6000 上用 BF16、batch 256、buffer 8,192 和 max length 128 完成，编码耗时 316.65 秒，峰值 allocated/reserved 显存约 1.76/2.45 GiB。
- 已为 491,213 个有 Train Query 的 POI 构建 E1 等权均值与 E2 `log1p(count) × IDF` 加权聚合，两个 491,213×1,024 float16 产物通过完整哈希、有限值、POI 行和计数守恒复检；全量聚合耗时 321.73 秒、峰值 RSS 6.72 GiB。
- `EXP-20260807-05` 在开发机 A6000 上完成 12 组 2,337,178 POI float32 `GpuIndexFlatIP` 精确召回。E1/α=0.30 的 Hit@10/MRR@10/NDCG@10 为 48.27%/26.0006%/31.2401%；E2/α=0.30 为 51.18%/27.7598%/33.2917%，相对 E0 提高 18.79/9.9971/12.0804 个百分点，成为当前连续向量最优候选。
- E2 对 Train 未见相同 Query 文本的 1,601 条样本仍将 Hit@10/NDCG@10 从 55.59%/41.24% 提高到 63.21%/49.98%，收益不只来自精确 Query 复现。分桶同时表明 `>20` 个不同 Query 的头部 POI 在 α=0.20 达峰，由此提出并在下一项正式实验检验异质性感知内容收缩。
- `EXP-20260807-06` 已完成六组 Kish 有效 Query 数驱动的逐 POI 动态 α 精确召回。组内最优 Hit@10/NDCG@10 仅为 41.13%/26.5432%，相对 E2/0.30 下降 10.05/6.7485 个百分点；成对排名为 454 胜、6,009 平、3,537 负。结论是分桶局部最优不能逐候选拼接为单一全库索引，E3 不进入 SID 输入候选。
- `EXP-20260808-01` 已完成统一 α=0.30 下四组类别公共 Query 方向残差。β=0.75 的 Hit@1/3/5/10/20、MRR@10、NDCG@10 为 19.95%/34.94%/43.41%/55.82%/66.71%/30.0583%/36.1412%，相对 E2 的 Hit@10/NDCG@10 提高 4.64/2.8495 个百分点，成对排名为 2,760 胜、6,453 平、787 负。最优 β 位于预设上界，只能视为冻结候选内最优。
- `EXP-20260808-02` 已完成 β=0.85/1.00 的边界补充。0.85 的 Hit@1/3/5/10/20、MRR@10、NDCG@10 为 20.02%/34.92%/43.76%/56.37%/66.78%/30.2279%/36.3976%，相对 0.75 的成对排名为 932 胜、8,556 平、512 负；1.00 的 NDCG@10 回落至 36.2709%，最终冻结 α=0.30、β=0.85。

### V1

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
- Centered GeoPE32 + TIGER-RQVAE 修正版已解决第一层码本低利用率，冻结的新 1024 PID 全局唯一；四卡 A100 三轮 SFT 和相同 Query-only SSP `gamma=2`、TCG Trie 的固定 10,000 条评测均已完成。epoch 3 的 HR@1/HR@10/NDCG@10 为 51.98%/87.83%/70.7430%，较旧版分别提高 1.13/0.70/0.8679 个百分点，结构和 PID 合法率均为 100%。
- Centered epoch 3 的固定 10,000 条 gold-prefix 诊断已完成：正确 GID6 条件下 S1/S2/S3 Top-1 为 83.97/88.67/89.27%，三层累计 69.90%，高于 TIGER/GNPR；但 GID6 累计 Top-1 只有 73.12%，完整 PID Top-1 all 为 50.00%。当前证据表明 GenPOI 语义 SID 易学，主要难度前移到细粒度地理路径。

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
- GNPR 专属词表已在 Qwen3-0.6B 上增加 1,805 个普通原子 Token，词表从 151,669 扩展至 153,474；正式 packed Train/Validation Cache 为 2,758,797/201,947 条，固定 smoke Cache 为 3,661/670 条，Assistant 目标截断数为 0。
- 四卡 A100 全参数 SFT 已完成 3 epoch，Validation Loss 为 0.547218/0.421038/0.406466；训练后包装脚本的后置引号语法错误不影响已完整保存的三个 checkpoint。固定 10,000 条无约束 Beam=10 评测在本地 A6000 上完成，epoch 3 的 HR@1/HR@10/NDCG@10 为 49.66%/83.07%/67.2390%，Valid ID Rate 为 78.808%，冻结为当前 Validation 最优 checkpoint。
- 固定 10,000 条逐层 teacher-forcing 诊断显示，GNPR 第一层 SID Top-1 为 53.69%，比 TIGER 低 21.37 个百分点；给定正确前缀后，GNPR 第二、三层反而高 2.39/6.32 个百分点。完整 identifier 的方法间差距为 2.36 个百分点，与自由生成 HR@1 的 2.21 个百分点差距接近，当前证据把主因定位为 GNPR 根前缀 Query 可预测性不足，而非训练 GPU 或无约束解码。

## 当前限制

- 四个已完成方法只在固定 10,000 条 Validation 和 Beam=10 上对比，不能替代完整 Validation 或 Test。
- V1、TIGER 和 GenPOI 的输入格式、唯一标识和约束解码不同，当前结果用于比较完整方法链路，不能把差异归因于单一模块。
- GNPR content-geo 是面向地图全目录覆盖的适配版，不等同于保留完整时间和协同用户特征的原论文严格输入。
- 共享机器曾发生 memory-cgroup OOM；全量数据处理必须继续使用 mmap、分片流式聚合或 YARN，避免同时运行高内存 Spark 和 GPU 评测。
- Geohash6 的边界效应、跨区域召回、长尾/新增 POI 和 no-result 增量仍未形成正式专项实验。

## 下一最小步骤

代码目录整理与 V1、TIGER、GNPR、旧版及 Centered GenPOI 的固定子集复现闭环均已完成。创新线 E4 已冻结连续向量并完成 RQ-VAE/RQ-KMeans 与全部 30-bit 布局验证。下一最小步骤是为 E4-2048、E4-4096、BGE-2048 按统一 TIGER collision-token 规则构建全局唯一 identifier，再进行同协议 SFT 与逐层 Teacher-Forcing 对照。
