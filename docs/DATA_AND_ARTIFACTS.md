# 数据与产物管理

## Git 保存内容

- 新流水线的源码、配置和测试。
- 合成测试样例或经过明确批准的匿名样例。
- README、方案和长期维护文档。

## Git 禁止保存内容

- 真实 POI、Query、行为标签和其他业务字段。
- 下载的模型目录和模型权重。
- Embedding、NPY/NPZ、Parquet、checkpoint 和二进制模型产物。
- 训练输出、日志、自动报告、图表和临时分析表。
- 凭证、内部服务地址和敏感绝对路径。

## 当前本地产物

| 路径 | 用途 | Git 规则 |
|---|---|---|
| `data/` | 内部源数据和后续生成数据 | 全部忽略 |
| `models/` | 本地 Qwen3 生成模型和 Embedding 模型 | 全部忽略 |
| `outputs/embeddings/` | 全量 Embedding、ID 映射和运行元信息 | 全部忽略 |
| `outputs/sid/rqvae/` | RQ-VAE checkpoint、SID 编码、指标和碰撞 Case | 全部忽略 |
| `outputs/sid/genpoi/` | GenPOI GeoPE RQ-VAE checkpoint、三容量 SID、指标和碰撞 Case | 全部忽略 |
| `outputs/sid/gnpr_sid/` | GNPR-SID 行为版和全量 content-geo 版的 RQ-VAE checkpoint、SID、Dedup 唯一标识与指标 | 全部忽略 |
| `outputs/pid/` | Geohash GID、base PID、Dedup 映射和 Final PID | 全部忽略 |
| `outputs/evaluation/` | Query 向量、行映射、召回结果、指标和运行元信息 | 全部忽略 |
| `data/sft/` | SFT Messages、特殊 Token 表和 Tokenized Cache | 全部忽略 |
| `data/eval/embedding_retrieval/` | 固定 Embedding 评测集及其数据指纹 Manifest | 全部忽略 |
| `outputs/sft/` | SFT checkpoint、训练日志、TensorBoard 和 Loss 结果 | 全部忽略 |
| `outputs/eval/` | Final PID Trie、固定评测子集、生成候选和检索指标 | 全部忽略 |
| `outputs/experiments/` | smoke test 和其他实验产物 | 全部忽略 |

只有新流水线真正需要时才创建额外产物目录，不提前创建空目录。

## 数据契约要求

任何特征生成或训练开始前，必须明确并校验：

- POI 和 Query 的字段名、类型、含义与隐私级别；
- 稳定、非空且唯一的 POI 标识；
- Query 到目标 POI 标签的来源和有效性；
- 空值、异常坐标、重复记录和失效 POI 的处理规则；
- 确定性去重、排序和数据切分规则；
- POI 元数据、特征、编码和模型目标之间的行对齐关系。

不得把已删除实验中的字段假设直接用于真实数据。

## 版本与可复现性

- 数据版本使用稳定标识，不在文档中记录真实样本内容。
- 正式实验必须记录数据版本、配置、代码提交、命令、环境和产物位置。
- 可重新生成的特征和训练产物由版本化代码与配置重建。
- 不可替代的源数据快照和重要 checkpoint 保存在批准的内部存储中。

当前向量产物约定：

- `embeddings.npy`：顺序写入的二维标准 NPY，完成后支持 mmap，行顺序与 `poi_ids.jsonl` 一致；
- `poi_ids.jsonl`：每行一个 JSON 字符串形式的 POI ID；
- `manifest.json`：输入指纹、模型参数、shape、dtype、环境和耗时；
- `progress.json`：已安全写入的断点位置和任务状态；恢复时以该位置截断未确认的尾部数据。
- GenPOI GeoPE 正式目录为 `outputs/embeddings/beijing_poi_bge_m3_genpoi_geope/`，保存旋转后的 `embeddings.npy`、同序 `poi_ids.jsonl`、地理锚点和 manifest；当前北京版本使用 32 个锚点，每个锚点对应 32 维向量分段。

SID 与 PID 产物约定：

- RQ-VAE 训练目录保存固定 epoch checkpoint、训练状态、运行配置和指标；固定 checkpoint 的全量评估目录保存 `sid_codes.npy`、`sid_manifest.json`、`metrics.json` 和 `collision_cases.jsonl`；
- GNPR-SID 的 004 原始审计特征位于 `outputs/embeddings/gnpr_sid/history10_pluscode6_top10_v1/`；北京适配版正式 SID 输入位于 `outputs/embeddings/gnpr_sid/history10_pluscode6_top10_hash8192_behavior_v1/`，只保留 703,306 条 `interaction_count > 0` 的 POI，并将 Top10 `passenger_id` 通过固定 BLAKE2b 哈希压缩到 8192 桶。类别、区域、小时和用户哈希组成逻辑 9384 维稀疏输入，32 个 Parquet 分片只保存各特征块的激活索引，不展开稠密矩阵；
- GNPR-SID 正式 RQ-VAE 产物位于 `outputs/sid/gnpr_sid/hash8192_behavior_v1/GNPR-SID-{256,512,1024}x3/`，只覆盖上述 703,306 条行为 POI；`1024×3 / epoch 20` 在当前输入内数值最优，完整 SID 唯一率为 94.2369%、最大碰撞桶为 17，但用户哈希碰撞审计未通过，尚未冻结。固定 7,060 条验证 POI 的重构余弦保存在上级目录 `reconstruction_validation_metrics.json`；
- GNPR-SID 当前 content-geo 正式输入位于 `outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/`，按既有 BGE-M3 行序保存 2,337,178 条 `category_indices.npy` 和 `region_indices.npy`；类别与 PlusCode6 词表分别为 402/766，不设置 UNK，最终逻辑输入为 `1024+402+766=2192` 维。训练只按 batch 读取 BGE memmap 并构造两个 one-hot 块，不保存全量稠密融合矩阵；
- GNPR-SID content-geo 三容量正式目录为 `outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-{256,512,1024}x3/`；运行使用 `1.0/0.25/0.25` 特征块权重、`2.140625` 重构权重和第 7 epoch 起启用的 GNPR utilization Diversity Loss。各目录的 `evaluations/epoch_20/` 已保存 2,337,178 条全量 `sid_codes.npy`、manifest、统一指标和 Top 20 碰撞桶；全量唯一率为 42.6376%/77.7969%/3.1313%，选择 512×3，1024 标记为后期坍塌失败产物；
- GNPR 1024 epoch 16的A6000独立复跑位于`outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/epoch16_rerun_a6000/GNPR-ContentGeo-BGE-M3-1024x3/`，`evaluations/epoch_16/`保存全量SID和统一指标；唯一率63.1798%、碰撞POI 51.9524%，仅作为可复现性诊断和前缀纯度消融，不作为下游冻结标识；
- GNPR 下游冻结标识位于 `outputs/sid/gnpr_sid/bge_m3_category_pluscode6_full_v1/4x6000d/GNPR-ContentGeo-BGE-M3-512x3/gnpr_ids/epoch_20/`。`gnpr_ids.npy` 使用四列内部表示，单例第四列为 `-1` 且序列化时省略，只有碰撞 SID 追加 `D0-D222`；`poi_gnpr_id_mapping.parquet` 保存 2,337,178 条 POI 的可逆映射，POI ID 和序列化 GNPR identifier 均全局唯一；
- TIGER identifier 目录保存固定四层 `[S1,S2,S3,C]` 的 `tiger_ids.npy`、`collision_codes.npy`、`poi_tiger_id_mapping.parquet`、manifest 和 metrics；单例固定使用 `C0`，同一三元 SID 桶内按 `poi_id` 字典序分配 collision code，POI ID 与完整 identifier 均须全局唯一；
- Geohash PID 目录保存 `gid_codes.npy`、`pid_codes.npy`、`pid_manifest.json`、碰撞指标和残余碰撞 Case；
- Dedup PID 目录保存 `final_pid_codes.npy`、`poi_pid_mapping.parquet` 和 `final_pid_manifest.json`，其中 POI ID 与 Final PID 均须全局唯一；
- GenPOI `1024×3 / epoch 20` 的 base/final PID 分别位于 `outputs/pid/genpoi/GenPOI-BGE-M3-GeoPE-1024x3-e20-G6/` 和同名 `-Dedup/` 目录；最终映射覆盖 2,337,178 条 POI，单例为九层 `[G1..G6,S1..S3]`，碰撞项追加第十层 Dedup Token；
- Centered GeoPE32 + TIGER-RQVAE 修正版的 base/final PID 分别位于 `outputs/pid/genpoi/GenPOI-CenteredGeoPE32-TIGER-RQVAE-1024x3-e20-G6/` 和同名 `-Dedup/` 目录；最终仍使用 `[G1..G6,S1..S3,optional D]`，覆盖 2,337,178 条 POI 且全局唯一；
- Final PID Trie 使用紧凑整数数组和 manifest 保存，构建输入必须与 Final PID 映射、Tokenizer 及 Token 清单一致；
- 上述数组、映射、checkpoint、Case 和 manifest 均位于 `outputs/`，由 Git 忽略。

Embedding 召回评测产物约定：

- Query-Augmented Relational SID 创新线冻结的评测集位于 `data/eval/embedding_retrieval/sft_validation_10k_v1/`。`eval.jsonl` 按既有 V1 SFT 固定 10,000 条 Validation 顺序保存最小字段契约，SHA256 为 `5557769526bd1a41a86adac5efb26eb2bf878fed8b2e2dad9f58afe93663e713`；`manifest.json` 固定来源、时间切分、业务键指纹和 Train/Eval 零重叠核验。真实 Query 数据继续由 Git 忽略；源码与版本化配置可重新核验该产物；
- 训练正向 Query 分片位于 `outputs/embeddings/query_augmented_bge_m3/train_query_shards_v1/`。输入为固定 `data/sft/beijing_order_main_v1/train.jsonl` 的 7,586,410 条 `Query → target_poi_id`，源 SHA256 为 `4ed3f3849e0beb10df500f013bc08b4663b0dca5df92dde9c8b3cb4a7ec9ffe4`；产物按原始 Query 的 BLAKE2b-64 哈希分为 256 个 ZSTD Parquet，总计约 82 MiB，分片行数为 19,734–100,193。独立逐行复算确认哈希错位为 0；Train-only 唯一 Query、唯一 Query–POI 对和目标 POI 数分别为 1,408,778、2,063,175 和 491,213；
- Train-only 聚合统计位于 `outputs/embeddings/query_augmented_bge_m3/train_query_stats_v1/`，包含各 256 个分片的连续唯一 Query 目录、Query–POI 订单计数和 POI 覆盖统计，总大小 56,586,021 bytes。Query 订单数与 POI 订单数均守恒为 7,586,410，Query `poi_df` 与 POI 唯一 Query 数之和均为 2,063,175；完整 `--validate-only` 文件 SHA256 复核已通过；
- Train-only BGE-M3 Query 向量位于 `outputs/embeddings/query_augmented_bge_m3/train_query_embeddings_v1/embeddings.npy`，行号严格等于连续 `query_id`。正式产物 shape 为 `[1408778, 1024]`、dtype 为 `float16`、SHA256 为 `e300e248a274a0febbcfb94b0b457f76916814564316029e258fd2bfe692ad67`；全量有限值、范数抽样和独立 `--validate-only` 校验均通过。该产物在开发机 RTX A6000 上生成，模型 BF16、batch 256、buffer 8,192、`max_seq_length=128`，峰值 allocated/reserved 显存约 1.76/2.45 GiB；
- E1/E2 稀疏 Query 聚合位于 `outputs/embeddings/query_augmented_bge_m3/query_poi_aggregates_v1/`，覆盖 491,213 个有 Train Query 的 POI。`covered_poi_rows.npy` 固定其在 2,337,178 条 POI 目录中的行号；`e1_query_mean.npy` 与 `e2_query_weighted.npy` 均为 `[491213, 1024]` float16，SHA256 分别为 `d8c3fb59d96e048e2d2f78f5324969df6b76383d6b5611df7bd2c6b312e34e42` 和 `a580591e5aeaaa5f2df78c6f26d6fd68200178a1ca79c1fcec0eb78b5fbb367a`。同目录还保存 Train 订单数、不同 Query 数和 E2 权重和；manifest SHA256 为 `57b21716b3c171f5b5641e1abb90d4e06217c227abb7402e2525d818d60547bd`；
- 无泄漏三模型 E0 重评测位于 `outputs/evaluation/sft_validation_10k_v1/`，与旧 `outputs/evaluation/embedding_retrieval/` 同级。三组 Query 行映射 SHA256 均为 `d512aa206468ea672b473ce38c353eca703c64289ac890d9a935cd3c9b2f855f`；冻结的 BGE-M3 Query 向量 SHA256 为 `98e2032b86b76cf42991dc1d1f4967ff456661fb32fbae185652163c85bf70e0`，E0 召回结果 SHA256 为 `8052a40b64b6b875c61a596388a764644042bb701936a5394eda4d9ff4eb3a9c`；
- E1/E2 正式精确召回位于 `outputs/evaluation/sft_validation_10k_v1/query_augmented_bge_m3_v1/`。目录按 `e1/e2 × alpha_0p05...0p50` 保存 12 组 `metrics.json`、Top-20 `retrieval_results.npz` 和运行 manifest，总计约 16 MiB；全量融合 POI 向量在索引构建时按行分块确定性生成，不重复落盘。正式 `summary.json` SHA256 为 `ca90786281d6dde5bd239140cd291e0eff5ff6190874f00c54bf6ad43bf913af`，精确 Query 重复与成对 bootstrap 诊断 `analysis.json` SHA256 为 `6206b1295be2194fce6c37ee593690e07c3dabe158c5a878f4e1e231e705c638`，当前最优 E2/α=0.30 的召回 NPZ SHA256 为 `1b7b8b81e8b2c8807c22298b2de8eab33fa155ffad8b982a451b08c7d81fa964`；
- E3 Train-only 异质性特征位于 `outputs/embeddings/query_augmented_bge_m3/query_heterogeneity_v1/`。目录按 E1/E2 的 491,213 个覆盖 POI 行序保存 E2 权重平方和、Kish 有效 Query 数与内容—Query 余弦；manifest SHA256 为 `980c9e5b24cc86f4e5b2a9fae3946f287511d393bb3984c529cdc05e8073c9d0`，三个数组均通过 shape、dtype、有限值、数学边界与完整文件哈希校验；
- E3 六组逐 POI 动态融合正式召回位于 `outputs/evaluation/sft_validation_10k_v1/query_adaptive_bge_m3_v1/`。每组保存指标、Top-20、目标 rank、实际 $\alpha_i$ 和运行 manifest，不保存全量融合向量。正式 `summary.json` SHA256 为 `563b7acd680ddb7d737f2f4d83f2e4544da6264f8f1b24f627d7b984c087c40b`；组内最优 `kish_t20_g1` 召回 NPZ SHA256 为 `a7b073d76904831265c4235a0d350be53643bb76aa5bcbcb041b1dc6c947e2d1`，但其总体指标低于 E2/α=0.30，不进入冻结候选；
- E4 Train-only 类别统计位于 `outputs/embeddings/query_augmented_bge_m3/query_category_residual_v1/`，仅保存 402 类支持计数和 `[402,1024]` float32 原始类别 Query 中心，不保存四份全量残差向量。manifest SHA256 为 `59004d8dd2dc74c059450eb858e2bbb63ffd94c88b53dc525c886007448f2d7f`，类别中心 SHA256 为 `42b2f9800f39aec6ed7fb94019f4351d959c6ba8660b25fea04dd5bdaf047491`；290 个支持度不低于 32 的类别覆盖 489,968 个有 Query POI，1,245 个稀有类别 POI 保持 E2 聚合；
- E4 四组统一 α 类别残差精确召回位于 `outputs/evaluation/sft_validation_10k_v1/query_category_residual_bge_m3_v1/`，检索时按分块在线计算 $\operatorname{Normalize}(\bar q_i-\beta\mu_{cat(i)})$。正式 `summary.json` SHA256 为 `b76d20fbab642412ec8d1b962f6a288058f785f0338e9328b95e12f95f7a044b`；冻结候选内最优 β=0.75 的召回 NPZ SHA256 为 `d9040ded85caad1a72d8f03e9e30cd4567d3e85f7323fdb09527b486d8bfd9b6`，Hit@10/NDCG@10 为 55.82%/36.1412%；
- E4 边界统计位于 `outputs/embeddings/query_augmented_bge_m3/query_category_residual_boundary_v1/`，候选预注册为 β=0.85/1.00，类别中心与主 E4 逐字节一致；manifest SHA256 为 `ae65ec6bd8fda28228822a080fb65a131d933d68acc0ae6aae903c46f8b303d8`；
- E4 边界精确召回位于 `outputs/evaluation/sft_validation_10k_v1/query_category_residual_boundary_bge_m3_v1/`。`summary.json` SHA256 为 `b4ad04be5b39e11d44d45813c32c54c5e97e05bd07432cdc990a5e4591a59d31`；β=0.85 的最终召回 NPZ SHA256 为 `d54662d1cdc0c8932b21a1040e2f524ef875c6e4ceb4ff6aaac3546a48a80d3d`，Hit@10/NDCG@10 为 56.37%/36.3976%；β=1.00 已出现回落，最终连续向量参数冻结为 α=0.30、β=0.85；
- `query_embeddings/*.npy`：按固定评测 JSONL 行顺序保存的 Query 向量；
- `query_embeddings/*_rows.jsonl`：Query 行号、订单标识和目标 POI 的对齐映射；
- `metrics.json`：整体召回指标和 Query 长度分桶指标；
- `retrieval_results.npz`：Top-K POI 行号、相似度和目标排名；
- `run_manifest.json`：Gate 0、模型配置、Faiss 设备、耗时、显存和产物指纹。

SFT 与约束生成评测产物约定：

- SFT 数据目录保存 Train/Valid/Test Messages JSONL、`special_tokens.json`、`manifest.json` 和 `stats.json`；
- 扩词表模型及 Tokenizer 位于 `models/`；正式 Tokenized Cache 位于 `data/sft/tokenized/<data_version>/`，只包含 packed `train`、`validation`、长度统计和输入指纹 Manifest，不读取或缓存 Test；
- TIGER SFT 的四层目标 Token、用户 Token、结构 Token 和数据版本必须与扩词表映射、Cache Manifest 和训练配置一致；中断恢复只能复用完成全文件扫描且固定长度通过的 Arrow 分片，并使用原子目录替换生成正式缓存；
- GenPOI 历史序列 SFT 正式数据位于 `data/sft/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/`，扩词表模型位于 `models/Qwen3-0.6B-GenPOI-Vocab-v1/`，正式 packed Cache 位于 `data/sft/tokenized/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1/`；数据使用 GeoPE 1024×3 + Geohash6 + 可选 Dedup 的唯一 PID，Test 只保留 JSONL，不进入 Tokenized Cache；
- Centered GenPOI 修正版 SFT 数据位于 `data/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2/`，Train/Valid/Test 为 7,586,410/597,421/606,682；Token 集合与旧版一致，复用 `models/Qwen3-0.6B-GenPOI-Vocab-v1/`。正式 packed Cache 位于同名 `data/sft/tokenized/` 目录，Train/Validation 为 2,935,702/214,860，约 32GB，Test 不进入缓存；
- Centered GenPOI 正式训练位于 `outputs/sft/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/`；四卡 A100 三轮共 17,202 step，`checkpoint-5734/11468/17202` 分别对应 epoch 1/2/3，三个 checkpoint 均包含模型、优化器、调度器、Trainer 状态和四卡随机状态；
- GNPR 历史序列 SFT 正式数据位于 `data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/`，Train/Valid/Test 为 7,586,410/597,421/606,682；历史使用 Query/GID/条件 Dedup identifier，当前请求使用 Query/GID，Prompt 不含时间或用户 ID。扩词表模型位于 `models/Qwen3-0.6B-GNPR-Vocab-v1/`，新增 1,805 个普通原子 Token；正式 Cache 位于同名 `data/sft/tokenized/` 目录，packed Train/Validation 为 2,758,797/201,947 条、固定 smoke Cache 为 3,661/670 条，约 30GB，Assistant 目标截断数为 0。Cache Manifest SHA256 为 `896f2fc596d75f23bd43c8146f3f69e09a2b38ded3854fc4c1bb40145a96e93f`，Test 未进入预处理；
- 北京适配 SSP 的独立校准数据位于上述 SFT 数据目录的 `proximity_bge_m3_bj_ordinal_v1/`：Train 复用论文版 7,586,410 条 Parquet，Calibration 使用 2026-07-13 完整 Validation 中排除固定 10,000 条生成评测样本后的 587,421 条记录；manifest 必须保持 `calibration_overlap_count=0`，固定评测样本不得参与分类头训练、选择或阈值校准；
- 正式训练目录保存完整 checkpoint、resolved config、Trainer 状态、训练与验证指标、TensorBoard 和运行日志；
- 生成评测目录保存 Final PID Trie、固定子集及其 manifest、分片进度、汇总指标和错误 Case；
- TIGER 固定 10,000 条正式评测位于 `outputs/eval/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/`，包含三个 epoch 的独立 run、完整汇总 JSON/CSV 和退出码；
- GenPOI TCG+SSP 评测位于 `outputs/eval/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/`；三个 epoch 的 10,000 条固定 Validation run、完整汇总 JSON/CSV 和退出码均已生成。epoch 1 曾从 `next_line=6000` 的原子分片恢复，最终三组 `result.json` 均为 `completed`，外层退出码为 0；
- Centered GenPOI TCG+SSP 评测位于 `outputs/eval/genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_gpu4_a100_e3/`；目录包含固定 10,000 条子集、新 PID Trie、与旧版相同的 Query-only SSP 预测、epoch 1/2/3 完整 run 和汇总 JSON/CSV，三组结构及 PID 合法率均为 100%，正式评测进程退出码为 0；
- 三方法逐层 Teacher-Forcing 诊断统一位于 `outputs/eval/sid_teacher_forcing_fixed10k_v1/`；`tiger_e3/gnpr_e3/genpoi_centered_e3` 均使用相同顺序的 10,000 个 `sample_id`。GenPOI 目录额外记录 GID1—GID6 累计、三层 SID、可选 Dedup 和完整 PID gold-prefix 指标，正式 `result.json` SHA256 为 `287590c6e99ba43640f9f3634220e92143f513888fa6736e9e0ae0a729effd9c`；
- GNPR 无约束生成评测位于 `outputs/eval/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3/`；固定子集按 TIGER 业务键顺序对齐，三个 epoch 的正式 run、错误 Case 和汇总 JSON/CSV 均已生成，三组 `result.json` 各含 10,000 条且状态为 `completed`，外层退出码为 0；
- Test 结果只有在 checkpoint、Beam 和评测配置冻结后才能生成并记录。
