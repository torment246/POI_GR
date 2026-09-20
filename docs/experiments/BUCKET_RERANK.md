# Bucket 内 POI 重排方法与实验

## 当前结论

`EXP-20260828-01/02` 已分别完成 TIGER 和 E4 容量后移 `512×1024×2048` 的固定普通 Validation 10,000 条、Beam=10、无训练 Bucket→POI 解析。两轮均复用已有三层 SID 候选轨迹，不改变 SID 或 Qwen，不读取 Test；第二轮把 POI 连续语义信号替换为生成该 SID 的冻结 E4 向量。

两轮结论均为明确负结果：八个候选排序变体都未通过“HR@1 与 NDCG@10 高于原完整 ID，同时 HR@10 不下降”的门禁。TIGER 最优简单桶内热度为 `50.64%/83.52%/67.1616%`，低于原 `C` token 的 `51.87%/87.16%/70.4190%`；E4 最优简单桶内热度为 `48.46%/81.19%/64.8585%`，同样低于原 `C` 的 `50.58%/87.07%/69.7840%`。因此：

- 当前 TIGER 的 `C` 不是可以直接删除的随机去重编号，它已经从请求频次中学到很强的桶内先验；
- 不得将整个候选池按 BGE、词法或热度全局重排，也不得把一个较大早期 Bucket 全部塞到后续 Bucket 之前；
- 本轮只冻结 Bucket 展开、对齐、排序评测和终态校验基础设施，不冻结任何 resolver；
- 该负结果不否定 Cluster SID 或桶内解析，只否定“无训练的单一启发式可以直接替换 `C`”这一假设；E4 更高的 Bucket HR@1 并未改变这一判断；
- 下一版若继续，第一优先不是训练新 ranker，而是先做 `C`-preserving candidate completion：完整保留原 Beam 中合法 `C` 候选及顺序，只用 Bucket 展开结果填补非法、重复或不足的 POI 槽位。该合同应保证已有命中不被启发式重排破坏，再判断是否还需要训练 residual ranker。

## 方法定位与边界

Cluster SID 将多个相近实体映射到共享 SID 后，必须存在 SID→Item 的解析环节；这一问题分别对应 [CQ-SID](https://arxiv.org/html/2605.14434v1) 的 Cluster SID 候选集合和 [Snapchat SID](https://arxiv.org/html/2604.03949v1) 的桶内 Item 映射实践。地理信号后续可参考 [OneLoc](https://arxiv.org/html/2508.14646v1)，但本轮没有引入地理特征，也不是上述论文的复现。

本方法线的长期目标是：

```text
Query → 生成若干 [S1,S2,S3] Bucket → 展开合法 POI 候选
      → 融合生成、语义、词法、地理和质量信号 → 最终 POI Top-10
```

第一轮刻意收缩为最弱、最容易证伪的无训练基线：

- 生成器固定为 TIGER epoch 3 无约束 Beam=10；
- SID、Qwen、`C` token、候选 Beam 和固定评测集均不改变；
- 不读取 Test，不训练排序器，不用 Validation 调参；
- POI 文本只用于确定性词法分数，BGE 使用已经冻结的 Query/POI 归一化向量；
- Train 信号仅为 `log1p(train_order_count)` 热度；
- 原 TIGER `[S1,S2,S3,C]` 排名作为主基线，精确 `poi_id` 为主指标，严格重复 POI canonical 指标作为补充。

## 第一版数据流

1. 按 `sample_id + order_id + searchid + target poi_id` 将 TIGER 候选轨迹、固定 Embedding 评测 JSONL 和 Query BGE 行映射逐行严格对齐。
2. 对每条请求按首次 Beam rank 去重合法三层 `[S1,S2,S3]`，从冻结 TIGER identifier 目录展开桶内所有 POI 行。
3. 对候选计算三类独立信号：
   - `popularity = log1p(Train 订单数)`，Train 未见 POI 为 0；
   - `lexical`：NFKC、转小写、只保留字母数字；名称/别名优先，地址乘 0.8、品类乘 0.6；
   - `semantic`：冻结且 L2 归一化的 Query/POI 连续向量 float32 点积；EXP-01 为 BGE-M3 Query×BGE-M3 POI，EXP-02 为 BGE-M3 Query×E4 POI。
4. 比较两种排序合同：
   - `bucket_then_*`：保持生成 Bucket 顺序，仅在每个 Bucket 内排序，再按桶顺序拼接；
   - `global_*`：忽略 Bucket 顺序，在全部展开候选上全局排序；RRF 固定 `k=60`，没有在 Validation 上搜索参数。
5. 输出每种方法的最终 POI Top-10、精确/canonical rank、成对胜负、碰撞/单例切片和失败案例。

## EXP-20260828-01：TIGER 固定 10k 无训练 Bucket 解析基线

### 目标与假设

目标是回答一个先于训练排序器的问题：TIGER 三层 Bucket 的召回增量，能否通过现有无训练信号转化为最终 POI Top-10 增益。

预注册式门禁为：候选方案必须同时提高精确 POI `HR@1` 与 `NDCG@10`，并且 `HR@10` 不低于原 TIGER `C` token。检验三个假设：

- H1：展开 Beam 内三层 Bucket 后存在可利用的目标覆盖空间；
- H2：热度、词法或 BGE 至少有一种可以直接替换 `C` 完成桶内排序；
- H3：跨 Bucket 全局融合可以弥补目标 Bucket 位次较后的情况。

### 数据版本与输入指纹

| 输入 | 版本或 SHA256 |
|---|---|
| TIGER epoch 3 Beam=10 候选轨迹 | `efbbaa404774afaae9329565e7b59dea10a20b97562cae860039f8d6b28b5fd0` |
| 候选轨迹源 `result.json` | `c22f0a1a1f39e3e05207573c67fea2f6fd3c3b75668b62efb3a52e0553fe1a0e` |
| 固定 Validation 10k Embedding 评测 JSONL | `5557769526bd1a41a86adac5efb26eb2bf878fed8b2e2dad9f58afe93663e713` |
| Query BGE 行映射 | `d512aa206468ea672b473ce38c353eca703c64289ac890d9a935cd3c9b2f855f` |
| Query BGE-M3 向量 | `98e2032b86b76cf42991dc1d1f4967ff456661fb32fbae185652163c85bf70e0` |
| POI BGE-M3 manifest | `3fabca469d0a6e9920b8d6523c69959db1f733c6564f5b6cb1b3db21ca7a969a` |
| TIGER identifier manifest | `259f0d23dd4e8e48395f344fceb6ba2610243a178c463cf52717af4e72ae7ecb` |
| Train-only Query–POI 聚合 manifest | `57b21716b3c171f5b5641e1abb90d4e06217c227abb7402e2525d818d60547bd` |

三份请求侧输入均为 10,000 行，顺序错位和目标 POI 错位均为 0。POI mapping、BGE 行序和目录 POI ID 完全一致。未读取 Test。

### 代码状态

- 基础提交：`54802e6674e283722ecee00fb862530df30cef9a`；运行时工作树为 dirty，包含正在进行的 TIGER-Joint 工作和本方法新增文件。
- 核心实现：`src/poi_gr/methods/bucket_rerank/evaluation.py`，SHA256 `a8ad4e48a36fe8c8b7dad670dbd45df4b38fde764bccc9bbee23441a99a391ee`。
- 命令入口：`scripts/bucket_rerank/evaluate_tiger.py`，SHA256 `7ceb75ecdda90240a5fa2f14c173709c887370478cdf19db69bd6daeccf2d4a3`。
- 合成测试：`tests/bucket_rerank/test_evaluation.py`，SHA256 `ac5399b2fc4ab29c9433c6ea1af8b031c93168e9d4eb0c11783d0ae57028b7de`。

### 配置、命令与环境

- `top_k=10`，`RRF k=60`，不训练、不使用地理、不读取 Test；
- 设备：CPU，冻结 NPY 使用 mmap；
- Python `3.10.20`，NumPy `1.26.4`；
- 正式运行耗时 `555.33` 秒，峰值 RSS `3657.75 MiB`。

正式命令：

```bash
TMPDIR=outputs/tmp/br10k \
PYTHONPATH=src \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/bucket_rerank/evaluate_tiger.py \
  --output-dir outputs/bucket_rerank/EXP-20260828-01_tiger_fixed10k_v1 \
  --examples-per-kind 10
```

终态复检：

```bash
PYTHONPATH=src /ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/bucket_rerank/evaluate_tiger.py \
  --output-dir outputs/bucket_rerank/EXP-20260828-01_tiger_fixed10k_v1 \
  --validate-only
```

### 候选池

| 指标 | 结果 |
|---|---:|
| 请求数 | 10,000 |
| 目标 POI 在展开池中的比例 | 88.06% |
| 每请求候选数 mean / p50 / p90 / p95 / p99 / max | 14.5595 / 12 / 28 / 36 / 65 / 205 |
| 无可展开候选请求 | 46 |
| 目标属于碰撞/单例 Bucket 的请求 | 3,854 / 6,146 |
| 本轮实际加载的目录 POI 行 | 62,946 |

H1 成立但空间有限：唯一 Bucket HR@1/HR@10 为 `55.53%/88.06%`，原精确 POI HR@1/HR@10 为 `51.87%/87.16%`，因此固定 TIGER 上理论增量分别只有 366/90 条请求。

### 主指标

以下均为精确 `poi_id` 百分比，粗体仅标记原方法基线，不表示本轮产生了新胜者。

| 排序 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| **原 TIGER `C` token** | **51.87** | **76.00** | **82.26** | **87.16** | **70.4190** |
| Bucket 内 Train 热度 | 50.64 | 70.21 | 77.21 | 83.52 | 67.1616 |
| Bucket 内词法 | 50.12 | 69.25 | 76.82 | 83.49 | 66.6733 |
| Bucket 内 BGE | 46.58 | 67.80 | 75.98 | 83.13 | 64.7151 |
| 全局 Train 热度 | 29.42 | 62.19 | 75.68 | 86.03 | 57.5323 |
| 全局词法 | 35.83 | 57.34 | 68.31 | 82.11 | 57.3330 |
| 全局 BGE | 27.82 | 47.98 | 59.98 | 77.70 | 50.1214 |
| 全局词法+BGE RRF | 36.98 | 59.36 | 70.30 | 81.93 | 58.2385 |
| 全局热度+词法+BGE RRF | 40.22 | 66.85 | 76.56 | 85.35 | 62.7533 |

原 TIGER 的严格 canonical HR@1/HR@10 为 `52.50%/87.16%`；各重排方法的 canonical 与 exact 指标相同，没有通过重复 POI 口径改变结论。

### 配对与条件诊断

相对原 `C` token，三个保留 Bucket 顺序的方案发生了以下精确命中翻转：

| 方案 | HR@1 胜/负/净命中 | HR@10 胜/负/净命中 |
|---|---:|---:|
| Train 热度 | 123 / 246 / -123 | 61 / 425 / -364 |
| 词法 | 172 / 347 / -175 | 65 / 432 / -367 |
| BGE | 179 / 708 / -529 | 64 / 467 / -403 |

在全部 3,854 条目标碰撞 Bucket 请求上，原 `C`、热度、词法和 BGE 的 HR@1 分别为 `48.39%/43.88%/42.53%/33.34%`；在 6,146 条目标单例 Bucket 请求上，原 `C` 为 `54.05%`，三种 Bucket 方案均为 `54.88%`。简单展开确实在单例桶回收 51 个 Top-1，但碰撞桶上的损失更大。

进一步只看“目标 Bucket 已由 Qwen 排在第一位”的 5,553 条请求，其中 2,180 条属于碰撞桶：

| 排序 | 5,553 条 HR@1 | 2,180 条碰撞 Bucket HR@1 |
|---|---:|---:|
| 原 TIGER `C` | 93.4090% | 85.5505% |
| Train 热度 | 91.1939% | 77.5688% |
| 词法 | 90.2575% | 75.1835% |
| BGE | 83.8826% | 58.9450% |

这一条件切片说明，主要失败不是目标桶位置错误，而是无训练信号覆盖不了 `C` 已学到的请求条件桶内偏好。另一方面，简单按桶拼接会让较大的早期桶占满 Top-10，把后续正确桶挤出；全局排序又会丢弃 Qwen 已经很强的 Bucket 顺序。因此 H2、H3 均被拒绝。

### 产物与复检

正式产物位于 `outputs/bucket_rerank/EXP-20260828-01_tiger_fixed10k_v1/`：

- `metrics.json`：完整指标和切片，SHA256 `3f52a22a85c5f22fa591381ecdceb2db80a0c85aaf128052443701145af422ea`；
- `rankings.npz`：各方案 10,000 条 rank 与 Top-10 行号，SHA256 `99056aeea055a11afdc3687bb0b7cd8616ed10c492bc629db698e00cb95f0ef4`；
- `comparison_cases.jsonl`：成对胜负样例，SHA256 `f715da66cf334e95a5492d3e6c99dfde4962fccfc119507edc7a8d6e45d70f7a`；
- `manifest.json`：输入、配置、环境、耗时和输出指纹；
- `_SUCCESS`：终态标记。

`--validate-only` 已复核行数、数组 shape 和 SHA256，状态为 `passed`。4 个合成单元测试、相关目录 `compileall`、CLI `--help` 和 32 条真实 smoke 均通过；smoke 不计为正式实验。

### 结论与下一步

- 本轮实验状态为“已完成，负结果”，没有候选进入后续冻结或 Test。
- TIGER 固定候选池在 HR@10 上只比原完整 ID 多 0.90pp，单靠排序无法解决 1,194 条“目标 Bucket 未进入 Beam”的请求。
- 如果继续 TIGER 桶内解析，下一最小实验应训练一个以原 `C` 排名/完整序列概率为强先验的保守 residual ranker，并预先定义只有高置信度才允许改写 `C` 的门控；训练负样本只来自同 Bucket，固定 10k 仍仅用于一次终态评测。
- 另一条更有辨识力的候选是先把同一解析协议迁移到 E4 容量后移 `512×1024×2048`：它的 Bucket HR@1 为 59.69%，且已有证据显示 `C>0` 条件 C Top-1 只有 70.85%，比 TIGER 更可能存在可兑现的桶内空间。迁移前需先固定其完整候选轨迹与同口径基线，不能直接引用 TIGER 的本轮数值。
- 地理、POI 质量和训练式 listwise/pairwise 排序均未在本轮实现；是否进入下一版等待单独确认。

## EXP-20260828-02：E4 容量后移固定 10k 无训练 Bucket 解析

### 目标与假设

本实验把 EXP-01 的同一 Bucket 展开、排序和最终 POI Top-10 合同迁移到 E4 RQ-KMeans 容量后移 `512×1024×2048`。除候选轨迹、identifier 和 POI 连续向量外，请求、固定 10k、Train 热度、词法规则、Beam 与指标均不改变。

沿用 EXP-01 的主门禁：候选方案必须同时提高精确 POI `HR@1` 与 `NDCG@10`，且 `HR@10` 不低于该 E4 生成器原始完整 `[S1,S2,S3,C]`。检验三个假设：

- H1：E4 容量后移的唯一 Bucket HR@1/HR@10 `59.69%/88.45%` 相对完整 ID `50.58%/87.07%` 留有可利用空间；
- H2：生成该 SID 的 E4 POI 向量、词法或 Train 热度至少有一种能够直接替代 `C`；
- H3：由于既有 Teacher-Forcing 诊断显示 E4 的碰撞后缀弱于 TIGER，同一无训练解析合同在 E4 上应比 EXP-01 更容易兑现。

### 数据版本与输入指纹

| 输入 | 版本或 SHA256 |
|---|---|
| E4 后移 epoch 3 Beam=10 候选轨迹 | `68556a152e6cf076f46a635b311fc0b76bef1a4d2d64a0ea400e71649a08c943` |
| 候选轨迹 manifest | `4c4ea82bf215441254d944a9d37a588cd47896310b7a09a0b4d44df5b59863c7` |
| 候选轨迹源 `result.json` | `d406e37e61d644009579195106268eed3615c3154c50f3ac04da55f2c291d8c2` |
| 固定 Validation 10k Embedding 评测 JSONL | `5557769526bd1a41a86adac5efb26eb2bf878fed8b2e2dad9f58afe93663e713` |
| Query BGE 行映射 | `d512aa206468ea672b473ce38c353eca703c64289ac890d9a935cd3c9b2f855f` |
| Query BGE-M3 向量 | `98e2032b86b76cf42991dc1d1f4967ff456661fb32fbae185652163c85bf70e0` |
| E4 `α=0.30, β=0.85` 全量 POI 向量 manifest | `f2d599441177c46588e1cb18aa55627dbf76b3e4743f20196ad3de7e7b20c664` |
| E4 后移 identifier manifest | `bfd958d835918ad66b7eb3bb6b3c72ca5e45e64197d06cbae00e904a9647b241` |
| E4 后移 POI→ID mapping | `627526a4c5cdfee6d22c6f8d23a53e673011c05b343d7ff41f098e7f9a1ba0bd` |
| Train-only Query–POI 聚合 manifest | `57b21716b3c171f5b5641e1abb90d4e06217c227abb7402e2525d818d60547bd` |

候选轨迹、请求、Query 向量和目标 POI 逐行严格对齐为 10,000 条；identifier 与 E4 POI 向量均为 2,337,178 行。未读取 Test，E4 未在线重算。

### 代码状态

- 基础提交：`54802e6674e283722ecee00fb862530df30cef9a`；运行时工作树为 dirty，包含本方法兼容 E4 的未提交修改以及其他正在进行的工作。
- 核心实现：`src/poi_gr/methods/bucket_rerank/evaluation.py`，SHA256 `0f07e774de8ff3c33eccdf0aacff78bb99498534b94ad454251fd6a82deb9bec`。
- 命令入口：`scripts/bucket_rerank/evaluate_tiger.py`，SHA256 `e8456fadb539287ac68bee07cbe8b8ce29bfc34dab401dba6314799f51b84cc3`。
- 合成测试：`tests/bucket_rerank/test_evaluation.py`，SHA256 `9e6a8dd8866321af1f027045037bf9f54ffce2511909881344fa4af43e38b85c`。
- 本轮只把 `bge` 排序信号泛化命名为 `semantic`，并新增目标 Bucket 位次切片；旧 EXP-01 的变体名与终态产物仍可向后兼容校验。

### 配置、命令与环境

- `top_k=10`，`RRF k=60`，不训练、不使用地理、不读取 Test；
- Query 向量为冻结 BGE-M3，POI 向量为冻结 E4 `α=0.30, β=0.85`；
- 设备：CPU，冻结 NPY 使用 mmap；
- Python `3.10.20`，NumPy `1.26.4`；
- 正式运行耗时 `558.27` 秒，峰值 RSS `2150.20 MiB`。

正式命令：

```bash
TMPDIR="$PWD/outputs/tmp/bre4" \
python scripts/bucket_rerank/evaluate_tiger.py \
  --output-dir outputs/bucket_rerank/EXP-20260828-02_e4_512x1024x2048_fixed10k_v1 \
  --candidate-trace outputs/eval/bucket_diagnostics_fixed10k_v1/rqkmeans_e4_512x1024x2048_e3/runs/valid_checkpoint-16710_beam10_bucketdiag_subset10000/candidate_trace.jsonl \
  --candidate-trace-manifest outputs/eval/bucket_diagnostics_fixed10k_v1/rqkmeans_e4_512x1024x2048_e3/runs/valid_checkpoint-16710_beam10_bucketdiag_subset10000/candidate_trace_manifest.json \
  --source-result outputs/eval/bucket_diagnostics_fixed10k_v1/rqkmeans_e4_512x1024x2048_e3/runs/valid_checkpoint-16710_beam10_bucketdiag_subset10000/result.json \
  --poi-embeddings outputs/embeddings/query_augmented_bge_m3/e4_alpha0p30_beta0p85_full_v1/embeddings.npy \
  --poi-embedding-manifest outputs/embeddings/query_augmented_bge_m3/e4_alpha0p30_beta0p85_full_v1/manifest.json \
  --identifier-dir outputs/pid/rqkmeans/e4_bge_m3/512x1024x2048_tiger_collision_v1 \
  --generator-label 'RQ-KMeans E4 512x1024x2048 epoch 3 unconstrained Beam=10 trace' \
  --semantic-label 'E4 alpha=0.30 beta=0.85' \
  --examples-per-kind 10
```

终态复检：

```bash
python scripts/bucket_rerank/evaluate_tiger.py \
  --output-dir outputs/bucket_rerank/EXP-20260828-02_e4_512x1024x2048_fixed10k_v1 \
  --validate-only
```

### 候选池

| 指标 | 结果 |
|---|---:|
| 请求数 | 10,000 |
| 目标 POI 在展开池中的比例 | 88.45% |
| 目标 Bucket 排第一的请求 | 5,969（59.69%） |
| 每请求候选数 mean / p50 / p90 / p95 / p99 / max | 19.2978 / 13 / 42 / 58 / 114 / 485 |
| 无可展开候选请求 | 92 |
| 目标属于碰撞/单例 Bucket 的请求 | 5,277 / 4,723 |
| 目标 Bucket 排第一且碰撞的请求 | 3,360 |
| 本轮实际加载的目录 POI 行 | 64,514 |

H1 的“候选空间存在”成立：相对轨迹本地复现的完整 ID，Bucket HR@1 理论多 911 条、HR@10 理论多 138 条。但该数字只是目标位于某个 Bucket 的上界，不代表任意桶内排序能够找到目标。

### 主指标

以下均为精确 `poi_id` 百分比；粗体仅标记原方法基线。

| 排序 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| **原 E4 完整 `C` token** | **50.58** | **75.56** | **82.07** | **87.07** | **69.7840** |
| Bucket 内 Train 热度 | 48.46 | 67.87 | 74.46 | 81.19 | 64.8585 |
| Bucket 内词法 | 46.33 | 65.67 | 73.15 | 80.54 | 63.1583 |
| Bucket 内 E4 | 42.90 | 63.91 | 72.62 | 80.72 | 61.3887 |
| 全局 Train 热度 | 29.88 | 60.77 | 74.08 | 84.52 | 56.8608 |
| 全局词法 | 34.76 | 55.55 | 65.45 | 77.89 | 54.9352 |
| 全局 E4 | 25.67 | 48.34 | 61.54 | 78.70 | 49.7603 |
| 全局词法+E4 RRF | 34.96 | 58.20 | 69.13 | 81.29 | 56.8863 |
| 全局热度+词法+E4 RRF | 39.21 | 65.62 | 75.35 | 84.66 | 61.7214 |

八个变体均未过门禁。严格 canonical 口径只把原完整 ID HR@1 从 `50.58%` 提高到 `51.32%`；所有展开排序的 canonical 与 exact 相同，因此重复 POI 口径不改变结论。

### 配对与条件诊断

最优简单方案 Bucket 内热度相对原完整 ID：

| 指标 | 胜 / 负 / 净命中 | 差值 |
|---|---:|---:|
| HR@1 | 248 / 460 / -212 | -2.12pp |
| HR@10 | 81 / 669 / -588 | -5.88pp |

只看“目标 Bucket 已排第一”的请求：

| 排序 | 全部 5,969 条 HR@1 | 其中 3,360 条碰撞 Bucket HR@1 |
|---|---:|---:|
| 原完整 `C` | 84.7378% | 74.6429% |
| Train 热度 | 81.1861% | 66.5774% |
| 词法 | 77.6177% | 60.2381% |
| E4 | 71.8713% | 50.0298% |

这组切片直接拒绝 H2、H3：即使目标 Bucket 已由生成器排到第一，原 `C` 仍显著优于 E4、词法和热度。既有 `C>0` Teacher-Forcing 条件 Top-1 `70.85%` 与这里不是同一条件集合；前者固定 gold 前缀，后者要求自由生成目标 Bucket 第一，不能用前者推断后者一定可被无训练信号替换。

目标为单例 Bucket 的 4,723 条请求中，Bucket-preserving 三方案 HR@1 都为 `55.2403%`，略高于原完整 ID `53.9911%`；但碰撞 Bucket 上的损失远大于这部分回收。全局方案虽然最多把 HR@10 保留到 `84.66%`，却破坏了 Qwen Bucket 顺序并大幅降低 HR@1。因此更大的 Bucket 上界真实存在，但“展开后重排所有候选”不是可行的兑现方式。

### 产物与复检

正式产物位于 `outputs/bucket_rerank/EXP-20260828-02_e4_512x1024x2048_fixed10k_v1/`：

- `metrics.json`：完整指标和四类条件切片，SHA256 `6eff3de993e69c676d129efd4add6256078edafb904e2532534d96a7961946fb`；
- `rankings.npz`：各方案 10,000 条 rank、Top-10、目标桶位次与候选池统计，SHA256 `59caa2cec2207d3c647345e8018ec328919dabbf609749017d40b4941821e5fe`；
- `comparison_cases.jsonl`：成对胜负样例，SHA256 `854a723104fc71646b4ac454d56ec6a790b95dd8f6db1f712138083a80f773f5`；
- `manifest.json`：输入、配置、环境、耗时和输出指纹；
- `_SUCCESS`：终态标记。

`--validate-only` 已复核 10,000 行、数组 shape 和三个输出 SHA256，状态为 `passed`。5 个合成单元测试、相关目录 `compileall`、CLI `--help`、旧 EXP-01 向后兼容复检和 32 条 E4 真实 smoke 均通过；smoke 不计为正式实验。

### 结论与下一步

- 本轮状态为“已完成，负结果”；没有候选冻结，也没有读取 Test。
- H1 仅在候选上界意义上成立，H2/H3 被拒绝。E4 的连续向量适合构造 SID，不等于其 Query 点积天然适合区分同 SID 碰撞实体。
- TIGER 和 E4 两轮共同证明：当前 `C` 是请求条件下的强判别器，不是可以在推理期删除的任意编号；单看 Bucket HR 会高估可兑现收益。
- 下一最小实验建议改为 `C`-preserving candidate completion：原 Beam 中合法、去重后的完整 ID POI 及相对顺序保持不变，只把展开候选用于填补非法、重复或不足的 Top-10 槽位。它首先验证能否在不牺牲既有 HR 的前提下回收 Bucket-only 目标；通过后才有理由训练同 Bucket residual ranker，并把地理作为可解释的增量特征。
