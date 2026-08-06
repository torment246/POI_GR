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
- GNPR 历史序列 SFT 正式数据位于 `data/sft/gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1/`，Train/Valid/Test 为 7,586,410/597,421/606,682；历史使用 Query/GID/条件 Dedup identifier，当前请求使用 Query/GID，Prompt 不含时间或用户 ID。扩词表模型位于 `models/Qwen3-0.6B-GNPR-Vocab-v1/`，新增 1,805 个普通原子 Token；正式 Cache 位于同名 `data/sft/tokenized/` 目录，packed Train/Validation 为 2,758,797/201,947 条、固定 smoke Cache 为 3,661/670 条，约 30GB，Assistant 目标截断数为 0。Cache Manifest SHA256 为 `896f2fc596d75f23bd43c8146f3f69e09a2b38ded3854fc4c1bb40145a96e93f`，Test 未进入预处理；
- 北京适配 SSP 的独立校准数据位于上述 SFT 数据目录的 `proximity_bge_m3_bj_ordinal_v1/`：Train 复用论文版 7,586,410 条 Parquet，Calibration 使用 2026-07-13 完整 Validation 中排除固定 10,000 条生成评测样本后的 587,421 条记录；manifest 必须保持 `calibration_overlap_count=0`，固定评测样本不得参与分类头训练、选择或阈值校准；
- 正式训练目录保存完整 checkpoint、resolved config、Trainer 状态、训练与验证指标、TensorBoard 和运行日志；
- 生成评测目录保存 Final PID Trie、固定子集及其 manifest、分片进度、汇总指标和错误 Case；
- TIGER 固定 10,000 条正式评测位于 `outputs/eval/tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3/`，包含三个 epoch 的独立 run、完整汇总 JSON/CSV 和退出码；
- GenPOI TCG+SSP 评测位于 `outputs/eval/genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3/`；三个 epoch 的 10,000 条固定 Validation run、完整汇总 JSON/CSV 和退出码均已生成。epoch 1 曾从 `next_line=6000` 的原子分片恢复，最终三组 `result.json` 均为 `completed`，外层退出码为 0；
- Test 结果只有在 checkpoint、Beam 和评测配置冻结后才能生成并记录。
