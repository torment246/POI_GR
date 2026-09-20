# TIGER-Joint 联合训练方案与实验进展

本文件统一记录 TIGER-Joint 的第一版方案、实现闸门和正式实验。该方法只回答一个问题：在不引入新 Embedding、独立 Query 编码器或新标识符结构的前提下，从头联合训练 TIGER 三层 RQ-VAE SID 与 Qwen 生成器，是否能比现有“先冻结 SID、再训练生成器”的 TIGER 获得更可生成的三级语义桶，并改善冷目标覆盖。

## 当前状态

- 更新时间：2026-08-30。
- 当前阶段：`J0—J4` 第一版闭环与一次终态合法路径诊断均已完成。`EXP-20260827-04` 在 4×RTX 6000D 上从 fresh KMeans RQ-VAE 和 vanilla Qwen fresh 扩词起点完成三轮、44,454 个同步 optimizer step，并保存三个完整 epoch checkpoint；`EXP-20260829-01` 完成 epoch 3 无约束 Bucket 评测，`EXP-20260830-01` 在同一 checkpoint、SID、固定 Validation 10k 和 Beam=10 上只增加全目录三级合法路径约束。
- 正式结论：epoch 3 的全目录三级 SID 唯一率为 81.9255%。无约束唯一 Bucket HR@1/3/5/10 为 22.84/32.90/34.77/34.97%，候选可展开率只有 18.410%；合法路径约束把 100,000 个候选的可展开率和每请求 10 个唯一合法桶覆盖均提升到 100%，HR@1/3/5/10 变为 22.99/35.40/40.00/44.64%，相对无约束只提高 0.15/2.50/5.23/9.67pp，仍远低于 TIGER。非法组合路径是失败因素，但不是充分解释；正确终态 Bucket 的概率排序仍然较弱。
- 当前边界：`EXP-20260830-01` 是用户在第一版停止后明确要求的解码诊断，不改变预注册主结果，也不重新开启方法线。本结果拒绝的是“每批使用当前 RQ-VAE hard code，同时更新 RQ-VAE 与 Qwen”的第一版实现，不等价于证明所有联合训练不可行。只评测了 epoch 3，不能声称 epoch 3 优于 epoch 1/2；不补 `C`、Query encoder、额外 epoch、Test 或泛化评测，也不再用当前联合 checkpoint 作为主线候选。

## 1. 核心研究问题与可验证假设

已有复现和诊断给出的证据是一致的：

- TIGER epoch 3 的完整 HR@1/HR@10/NDCG@10 为 51.87%/87.16%/70.4190%，三级唯一可展开 Bucket HR@1/3/5/10 为 55.53%/79.64%/85.10%/88.06%。
- MMBERT、GHR 和多组 Embedding × Quantizer 候选虽然改善过连续召回、静态唯一率或局部后缀，最终失败主要发生在 S1/S2 和自由生成的三级目标桶覆盖，而不是碰撞后缀容量不足。
- 冷目标的 TIGER HR@1/HR@10/NDCG@10 只有 7.08%/17.78%/11.8127%；全量 2,337,178 个 POI 中只有 491,213 个具有 Train Query 聚合，说明不能只靠行为正样本学习目录表示。

因此第一版只检验以下假设：

> 使用当前生成器的请求表示约束目标 POI 的 RQ-VAE 连续量化表示，同时保留全目录等权 RQ 重建目标，可以让 SID 的内容结构和 Query 可预测性在同一次训练中共同优化；这种优化应首先体现在三级 Bucket 生成覆盖，而不是静态 SID 唯一率。

对应的可证伪条件是：若联合模型在同一固定 Validation 10k 上的三级 Bucket HR@1/3/5/10 不能达到 TIGER 的 55.53%/79.64%/85.10%/88.06%，则第一版联合目标没有解决当前核心瓶颈，不进入碰撞后缀或更复杂架构扩展。

## 2. 第一版冻结范围

| 部分 | 第一版选择 | 明确不做 |
|---|---|---|
| POI 表征 | 冻结 BGE-M3 全目录 1,024 维向量 | 不换 MMBERT，不融合 Train Query 向量 |
| SID 模型 | 从头构造 `1024→512→256` RQ-VAE，三层码本均为 1,024；方法内重新生成固定切分和 KMeans 样本索引 | 不读取 TIGER resolved config、epoch 20 checkpoint 或任何旧 SID mapping |
| 生成模型 | 直接加载 `models/Qwen3-0.6B`，在内存中新建三级 SID、用户、GID 和结构 Token 行 | 不加载 `Qwen3-0.6B-TIGER-Vocab-v1`、`checkpoint-16713` 或旧 Token mapping |
| 输入 | 复用 TIGER 的用户 Token、最近 10 条历史 Query/GID/POI 和当前 Query/GID 语义 | 不新增独立 Query encoder，不加入 UniSearch 其他模块 |
| 输出 | 固定生成三层 `[S1,S2,S3]` | 第一版不生成 `C`；动态 `C` 需要全目录重映射，不能按 batch 正确定义 |
| 优化 | 行为批与全目录 POI 批在每个 optimizer step 联合更新 | 不按 epoch 先更新 SID、再更新 Qwen，不交替冻结 |
| 时长 | 正式候选固定训练 3 个行为 epoch | 不用额外 epoch 搜索替代公平对照 |

`S1/S2/S3` 仍是三级码本的 Token 命名空间，不代表复用任何旧 POI→SID 分配。新词表直接从 vanilla Qwen 扩展，共新增 5,120 个普通原子 Token：16 个结构 Token、32 个 Geohash 字符 Token、2,000 个用户 Token 和 `3×1,024` 个 SID Token；没有 `C`/`D` Token，也没有 `<POI_TIGER_ID>` wrapper。新模型扩词后词表为 156,789，而旧 TIGER 扩词表为 157,095，二者不共用扩词产物。

## 3. 动态数据契约

旧 TIGER JSONL 已把固定 SID 直接写入 Prompt，不能用于联合训练，也不能再作为反查 POI 行的中间适配层。新数据必须从原始行为记录的 `poi_id` 出发，只用 BGE 自带 `poi_ids.jsonl` 建立一次 POI→行号映射；缓存保存无固定 SID 的 token 模板和 BGE 目录行号，运行时由“当前 RQ-VAE 参数”产生代码并回填。

每条行为样本至少包含：

| 字段 | 类型/形状 | 含义 |
|---|---|---|
| `schema_version` | string | 固定为 `tiger-joint-sid-free-data-v1`；旧 TIGER schema 直接拒绝 |
| `sample_id` | string | 由原始分片相对路径和行号稳定哈希得到的样本键 |
| `split` | enum | `train`、`valid` 或 `test`，正式准备只读取 Train/Valid |
| `input_ids_template` | `int64[L]` | 历史和目标 SID 槽位使用占位 Token 的完整模板 |
| `labels_template` | `int64[L]` | 非监督位置为 `-100`；静态目标结构 Token 可保留标签 |
| `attention_mask` | `bool/int64[L]` | 动态 padding 后的有效位置 |
| `query_state_position` | `int64` | 当前请求 `</CURRENT>` 的位置，用于抽取 Qwen 请求表示 |
| `history_poi_rows` | `int64[H]` | 最近 `H≤10` 条历史 POI 在冻结 BGE mmap 中的行号 |
| `history_sid_positions` | `int64[H,3]` | 每条历史的 S1/S2/S3 回填位置 |
| `target_poi_row` | `int64` | 目标 POI 在冻结 BGE mmap 中的行号 |
| `target_sid_positions` | `int64[3]` | Assistant 目标 S1/S2/S3 的输入与 label 回填位置 |

持久化 JSONL 在 Token 化前保存 `messages`、`history_poi_rows` 和 `target_poi_row`；`input_ids_template` 等位置字段由同一 tokenizer 契约在预检/缓存阶段生成。主入口显式拒绝 `target_tiger_id_key`、旧 TIGER schema、`<POI_TIGER_ID>`、`<C_*>` 和 `test.jsonl`。`<POI_SID><S1_0><S2_0><S3_0></POI_SID>` 只是未赋值槽位，batch 内实际 Token 每一步都来自当前 fresh RQ-VAE。

运行时 collator 再按上述行号从只读 mmap 批量 gather `history_embeddings`、`target_embeddings` 和独立均匀目录采样得到的 `catalog_embeddings`。缓存、POI ID 列表和 BGE manifest 必须绑定行数与 SHA256，禁止用字符串 POI ID 在训练热路径中反复查表。

正式数据固定 `cutoff_len=1024`、不 packing、动态 padding。进入正式缓存或训练前必须完整扫描 Train/Valid，要求超长样本为 0、Assistant 目标截断为 0、所有 SID 槽位有效且互不重叠。Test 在方案冻结和训练期间保持不可见。

## 4. 联合目标与梯度路径

设生成交叉熵、请求—POI 对齐损失和全目录 RQ-VAE 损失分别为 $L_{gen}$、$L_{align}$、$L_{rq}$：

$$
L = L_{gen} + \lambda_{align}L_{align} + \lambda_{rq}L_{rq}.
$$

- `L_gen`：只对 Assistant 目标序列计算 causal LM loss。历史和目标三层 Token 均由当前 RQ-VAE 的 hard argmin 代码动态回填。
- `L_align`：取当前请求 `</CURRENT>` 的 Qwen 最后一层 hidden，经一个 `hidden_size→256` 线性投影，与目标 POI 的 straight-through 量化表示做 in-batch InfoNCE。相同 `target_poi_row` 的重复样本全部视为正例，避免把同一 POI 当作假负例。
- `L_rq`：独立均匀目录批上的原 TIGER reconstruction + codebook + commitment loss，覆盖有行为和无行为 POI。

梯度边界如下：

| 损失 | Qwen | Query 投影 | RQ encoder | RQ decoder | codebook |
|---|---:|---:|---:|---:|---:|
| `L_gen` | 是 | 否 | 否 | 否 | 否 |
| `L_align` | 是 | 是 | 是（straight-through） | 否 | 否 |
| `L_rq` | 否 | 否 | 是 | 是 | 是 |

hard argmin 和离散 Token ID 本身不可导，因此只把 `L_gen` 与 RQ-VAE 相加并不构成真实联合训练；`L_align` 是第一版必需的最小连续桥。正式训练前只允许用固定的梯度尺度预检确定两个权重，不能根据 Validation HR 反复调权重。

## 5. 一个 optimizer step 的数据流

1. 清空 Qwen、Query 投影和 RQ-VAE 梯度。
2. 从行为流取一个 microbatch，并从全目录均匀采样对应的 POI microbatch。
3. 用当前 RQ-VAE 为历史与目标 POI 计算三层 hard codes，回填 Prompt 与 Assistant labels。
4. Qwen 前向一次，计算 `L_gen`，并在 `</CURRENT>` 位置取得请求 hidden 计算 `L_align`。
5. 对目录 POI 批执行 RQ-VAE 前向，计算 `L_rq`。
6. 对加权总损失反向；若使用梯度累积，同一组未更新参数完成所有 microbatch 后才 step。
7. 在累积边界同步 step Qwen 与 RQ-VAE optimizer/scheduler。DDP 下 RQ-VAE 必须与 Qwen 一样同步梯度，不能每卡维护不同 SID。

训练热路径不导出 2,337,178 行 SID mapping。只有 epoch/checkpoint 边界冻结当前参数后，才流式导出全目录三层代码并构建 Validation Bucket/Trie；因此一次 Validation 对应一个明确的 RQ-VAE 与 Qwen 快照。

## 6. 冷启动 POI 协议

- 训练时：全目录均匀 RQ 流覆盖所有 POI，包括 Train 目标频次为 0 的 POI；只有真实行为样本进入 `L_align`，不为冷 POI 伪造 Query 正例。
- 离线评测：复用冻结的 `cold_target_10000`，首先报告三级 Bucket HR@1/3/5/10；同时报告目标桶是否可由当前全目录映射展开，避免把桶内候选误当成精确 POI 命中。
- 上线增量：训练结束后冻结 RQ-VAE。新 POI 只需计算 BGE 向量、编码三层 SID 并加入 Bucket/Trie，无需立即重训 Qwen 或在线更新 RQ-VAE。
- 能力边界：纯生成器无法保证为从未出现过的新 SID 路径分配足够概率；同桶展开只能让冷 POI 进入候选，最终精排仍需使用现有检索链路。第一版只验证联合训练是否改善桶覆盖，不宣称解决完整冷启动排序。

## 7. 评测与停止门禁

主对照只使用现有 TIGER 结果，不引入其他方法：

| 口径 | TIGER epoch 3 基线 | TIGER-Joint 第一版用途 |
|---|---:|---|
| 三级唯一 Bucket HR@1/3/5/10 | 55.53/79.64/85.10/88.06% | 直接主指标 |
| 三级 gold-prefix 累计 Top-1 | 53.98% | 逐层归因 |
| 完整四层 HR@1/3/5/10 | 51.87/76.00/82.26/87.16% | 仅作上下文，不直接比较；联合第一版没有 `C` |
| 完整四层 NDCG@10 | 70.4190% | 仅作上下文 |
| 冷目标完整 HR@1/HR@10/NDCG@10 | 7.08/17.78/11.8127% | 仅作上下文；联合第一版主报冷目标 Bucket 指标 |

固定随机 Validation 10k 是首要门禁；通过后才运行四类泛化集。若三级 Bucket HR 未达到 TIGER，不增加 `C`、不接新 encoder、不扩码本也不追加训练轮次。若 Bucket 指标通过，再单独定义无泄漏的桶内展开/排序协议，之后才能报告可与完整 POI HR/NDCG 直接比较的指标。

## 8. 实施阶段与进展

| 阶段 | 最小产出 | 验收标准 | 状态 |
|---|---|---|---|
| `J0` 联合核心 | 独立源码目录、动态 SID batch 契约、三项 loss、Qwen last-hidden 适配与合成单步更新 | loss 全有限；动态代码来自当前 RQ；Qwen、投影、RQ encoder/codebook 均更新 | 已完成 |
| `J1` 真实数据 | 原始行为+BGE 行序的独立构建脚本、mmap gather、Train/Valid 全量 1024 预检 | 无旧 SID 依赖；行序/哈希对齐；超长与目标截断均为 0；Test JSONL 不产出/不读取 | 已完成：`EXP-20260827-01` 全量 8,183,831 行通过 |
| `J2` GPU smoke | 小规模真实样本过拟合与多卡单步 | loss 与生成同时过拟合、目标 SID 不塌缩、无 OOM/NaN、跨卡 SID 参数一致 | 已完成：`EXP-20260827-03` fresh KMeans + 100-step；两进程合成 DDP 参数同步及后续四卡正式运行均通过 |
| `J3` 正式联合训练 | 从头 RQ-VAE + 扩词表 Qwen，3 个行为 epoch，按 epoch 保存 | 三个完整联合 checkpoint 和可恢复状态 | 已完成：`EXP-20260827-04`，44,454 step、三个 epoch checkpoint、退出码 0 |
| `J4` 导出与评测 | epoch 3 全目录 SID 导出、固定 10k Bucket 评测与逐行复核 | 与冻结 TIGER 业务键完全一致，指标可复现 | 已完成但效果门禁失败：`EXP-20260829-01`；停止后按用户要求补充 `EXP-20260830-01` 合法路径诊断，结论不变 |

## 9. 实验记录

`J0` 合成测试和有界真实前缀/单步属于 smoke，不单独分配实验编号。`EXP-20260826-02` 已保留为旧适配链路的历史兼容诊断；严格 SID-free 的全量工程门禁和第一次正式联合更新分别为 `EXP-20260827-01/02`。

### 2026-08-26：J0 动态 SID 与联合单步 smoke

- 代码状态：在未提交工作树中新建 `src/poi_gr/methods/tiger_joint/` 和 `tests/tiger_joint/`；未修改既有 `src/poi_gr/methods/tiger/` baseline，也未修改共享 RQ-VAE 实现。
- 合成契约：3 条行为、2 条历史、4 条目录 POI，Embedding 维度 6；fresh RQ-VAE 为 `6→5→4`、三层 `4×4×4`，tiny causal generator hidden size 为 8。该尺寸只验证数据流和梯度，不代表正式配置。
- 核验结果：历史与目标代码逐项等于同一时刻 RQ-VAE `encode_codes` 的直接结果；三层 Token 正确回填到历史输入、目标输入和目标 labels；生成、对齐、RQ 与总 loss 全部有限；一次 backward/step 后 generator embedding、Query 投影、RQ encoder 和第一层 codebook 均发生更新；重复目标 POI 的多正例 InfoNCE 测试通过；非法 query/SID 槽位重叠被拒绝。
- 回归结果：3 个 TIGER-Joint 合成测试和 1 个共享 RQ-VAE forward/backward 测试共 4/4 通过，`compileall` 退出码为 0；未读取真实北京数据、模型 checkpoint、GPU 或 Test split。

~~~bash
python -m pytest -q \
  tests/tiger_joint \
  tests/sid/test_rqvae.py::RQVAETest::test_forward_three_level_residual_quantization_and_backward

python -m compileall -q \
  src/poi_gr/methods/tiger_joint \
  tests/tiger_joint
~~~

- 当前结论：hard code 动态回填、连续梯度桥和目录 RQ 锚点已经形成最小可运行闭环，但这不是模型效果实验。下一步进入 `J1`：在独立 `scripts/tiger_joint/` 中构建不含固定 SID 的真实模板。允许用有界真实前缀验证服务器单步连通性；完整 Train/Valid 的 1,024 长度与目标零截断预检仍是任何正式缓存或训练的前置闸门。

### 2026-08-26：旧 SID 适配前缀与单卡联合单步 smoke（历史诊断）

> 边界修订：本节结果按当时实现真实记录，但数据先读取旧 TIGER `[S1,S2,S3,C]` 再反查 POI 行，且生成侧使用旧 TIGER 扩词表。因此它只证明联合核心在服务器上可运行，不再证明当前严格 SID-free 的 `J1/J2` 数据隔离。

- 目标与假设：验证旧 TIGER 四层 ID 只用于离线恢复冻结 BGE 行号，运行时模板已彻底移除旧 `C` 与旧 SID 值；再验证真实 Qwen、Query 投影和从头随机初始化 RQ-VAE 能在服务器上完成同一个联合 optimizer step，且三条损失路径均产生非零梯度。
- 数据版本：只读取 `tiger_bge_m3_1024x3_history10_query_gid_v1` 的 Train 前 32 行和 Valid 前 2 行，数据 `build_fingerprint=aa29346ff8420b54a27764d93ee62dc8e3043125f59d73b0546a8ff04c57d691`；目录为 2,337,178×1,024 的冻结 BGE-M3 mmap，manifest signature 为 `2ee5349438f7d668d96f9dfc8d1e217142972e526860d417b327aafaeadbeaa4`；旧唯一四层 ID manifest SHA256 为 `259f0d23dd4e8e48395f344fceb6ba2610243a178c463cf52717af4e72ae7ecb`。Test 样本文件未读取。
- 代码状态：未提交工作树；新增 `preparation.py`、真实服务器单步入口和对应合成测试。旧四层 ID 通过紧凑排序键映射到 POI 行后，历史与目标统一替换为三个动态槽位；collator 从只读 mmap gather 历史、目标和独立目录批。既有 TIGER baseline 与共享 RQ-VAE 实现未修改。
- 模型与配置：`Qwen3-0.6B-TIGER-Vocab-v1` BF16、词表 157,095、hidden 1,024；fresh FP32 RQ-VAE 为 `1024→512→256` 和 `1024³`；Query 投影为 `1024→256`。行为 batch 为 2，目录 batch 为 64，`cutoff_len=1024`，`λ_align=0.1`、`λ_rq=1.0`、温度 0.07；Qwen/投影学习率为 `5e-5`，RQ 学习率为 `3e-4`，seed 为 42。该 smoke 使用随机 RQ 初始化，正式训练仍须采用冻结的全量 KMeans 初始化协议。
- 真实模板预检：32 条 Train 完整序列长度为 39～299，30 个不同目标在 fresh RQ 下形成 8 个不同三级 SID；2 条 Valid 长度为 39～40。动态模板不含旧 `C`。为避免 batch=1 时 InfoNCE 恒为 0，smoke 从该固定前缀中稳定选择前 2 条“目标行不同且初始三级 SID 不同”的样本；这只是梯度连通性测试，不是训练采样策略。
- 核心结果：一次双 optimizer step 成功，耗时 11.8123 秒；`L_total/L_gen/L_align/L_rq` 分别为 20.963072/20.894487/0.643409/0.004244。裁剪前 Qwen、Query 投影、RQ encoder、RQ decoder、RQ codebook 梯度范数分别为 619.4013、0.464722、0.626502、0.002613、0.004288，均为有限正数；RQ-VAE 总梯度范数为 0.626522。目标 batch 在 S1/S2/S3 分别使用 2/2/1 个码，完整三级 SID 为 2 个。
- 环境与资源：`Python 3.10.20`、`torch 2.9.1+cu128`、`numpy 1.26.4`、NVIDIA RTX A6000；单步前已分配显存 1,222,063,104 bytes，峰值 allocated/reserved 为 6,090,321,920/6,203,375,616 bytes。运行结束后 GPU 回到 0 MiB 占用。
- 产物：`outputs/tiger_joint/smoke/server_single_step_v1/run_state.json`；只保存配置、loss、梯度和资源指标，没有 checkpoint、样本内容或 SID mapping。

~~~bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TMPDIR=outputs/tmp/tjs1 PYTHONPATH=src \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/smoke_joint_step.py \
  --output-dir outputs/tiger_joint/smoke/server_single_step_v1 \
  --train-scan-rows 32 \
  --valid-scan-rows 2 \
  --behavior-batch-size 2 \
  --catalog-batch-size 64
~~~

- 结论：第一版数据流、显存规模和连续梯度桥在真实服务器环境中可运行；但单步不支持“loss 已下降”或“联合训练有效”的结论。正式训练入口前仍有两个闸门：完整扫描 Train/Valid 并确认超过 1,024 与目标截断均为 0；冻结与 TIGER 对照一致的 500,000 POI、三层顺序 KMeans 初始化及其可恢复状态。完成这两个闸门后再实现三轮训练、epoch checkpoint 和 DDP 同步。

### EXP-20260826-02：旧 SID 适配 Train/Valid 全量预检（历史兼容诊断）

> 状态修订：实验实际运行结果与产物完整保留，但它读取了旧 TIGER JSONL、旧 identifier mapping、旧 resolved config 和旧 TIGER 扩词表。用户确认“只复用 BGE 向量、SID 从头训练”后，本实验不再是严格 SID-free `J1` 的正式闸门，不能据此启动训练。

- 目标与假设：在编写正式训练器前，证明现有 TIGER 行为数据可以无损转换为“旧四层 ID 只恢复 POI 行、当前 RQ-VAE 动态生成三级 SID”的联合模板；完整 `qwen3_nothink` Source+Target 必须全部落在 1,024 Token 内，且 Assistant 目标不能截断。同时复算原 TIGER 的固定 Validation/KMeans 抽样，保证联合训练只从头训练参数而不改变公平对照的目录切分与初始化协议。
- 数据版本：Train/Valid 来自 `data/sft/tiger_bge_m3_1024x3_history10_query_gid_v1/`，`build_fingerprint=aa29346ff8420b54a27764d93ee62dc8e3043125f59d73b0546a8ff04c57d691`，数据 manifest SHA256 为 `581a4dd3c285a83f863c4503a603328fbc0e5c13e22a7abbef4fe4a860e896cf`。Train 为 7,586,410 行、SHA256 `c812e0d2fa40cb4295cf66d41d015c0c0ee0e86a4764ca943e8a34b44c98489a`；Valid 为 597,421 行、SHA256 `20af3356c487952bf8a58753979b3c23c452d99d627bff858066f213d8dd8dfc`。扫描器只构造这两个路径，Test 样本未读取。
- 冻结输入：POI 向量为 `[2337178,1024]` 的 BGE-M3 mmap，manifest signature `2ee5349438f7d668d96f9dfc8d1e217142972e526860d417b327aafaeadbeaa4`；旧唯一四层 ID manifest SHA256 为 `259f0d23dd4e8e48395f344fceb6ba2610243a178c463cf52717af4e72ae7ecb`。生成基座为 `Qwen3-0.6B-TIGER-Vocab-v1`，model config SHA256 `578e77c4d311cd1574065ed43944afc28b67854cdf384f32430bd4acf9ae9973`，`tiger_token_mapping.json` SHA256 `c504e0b92e81c15ea5206ff9638d274f4706b91b0d1a1b9d8f9cd7adc6535a9d`。
- 代码与工作树：Git revision 为 `54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树未提交；本方法新增批量四层 ID 查表、动态模板逐槽验证、长度 histogram、全量流式入口和合成测试。`run_state.json` 保存了完整 `git status --short`，既有 TIGER baseline、共享 RQ-VAE 与用户其他未提交修改均未覆盖。
- 配置：`cutoff_len=1024`、不 packing、Tokenizer batch 4,096、每 250,000 行记录进度；目标固定为 `<TARGET_POI><S1_0><S2_0><S3_0></TARGET_POI>`，完整 Target 恒为 7 Token。每条历史和目标的旧 `[S1,S2,S3,C]` 先批量映射到冻结 POI 行，字符串与 Token 两层校验均确认只保留三个动态占位槽，旧 `C` 使用数为 0。
- 初始化契约：从 `TIGER-BGE-M3-1024x3/resolved_config.json` 复算 seed 42 的 23,372 行固定 Validation 和 500,000 行 Train-only KMeans 样本；Validation/KMeans 行索引 SHA256 分别精确复现为 `409e476abf82cd609447c6c9a6f437d96ecd897a40c5e610a33a3bc996041249` / `0f7f22d7a7543965d581bcc9098a833533dbbdbb701b867f54c76d0bf66d6eb7`。正式训练必须按 `seed→fresh 1024→512→256 RQ-VAE→Qwen/Query 投影` 顺序构造，三级码本均为 1,024；KMeans 固定 `faiss_gpu`、逐层 residual、20 次迭代、batch 8,192、`max_points_per_centroid=2048`。本次只冻结并验证协议，没有加载旧 RQ-VAE checkpoint，也没有实际执行 KMeans 聚类。
- 环境：开发服务器 CPU 流式任务，128 个逻辑 CPU；`/ofs/map_search/hudan/envs/poi-gr`，Python 3.10.20、NumPy 1.26.4，启用 fast tokenizer 并行。扫描期间只保留单个 JSONL/Tokenizer batch、紧凑 ID 排序索引和只读 Embedding mmap；未占用 GPU。

~~~bash
env \
  TMPDIR=/ofs/map_search/hudan/poi_genret/outputs/tmp/tjp1 \
  TOKENIZERS_PARALLELISM=true \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  /ofs/map_search/hudan/poi_genret/scripts/tiger_joint/preflight_dynamic_data.py \
  --output-dir /ofs/map_search/hudan/poi_genret/outputs/tiger_joint/preflight/j1_full_v1 \
  --batch-size 4096 \
  --progress-every 250000
~~~

- 核心结果：Train+Valid 共 8,183,831 行全部通过，映射 48,522,625 个历史/目标 identifier，验证 145,567,875 个动态三级槽位。完整序列长度 min/P50/P90/P95/P99/P99.9/max 为 39/147/296/304/337/445/946，Source 最大 939，Target 恒为 7；超过 1,024 为 0，Target 截断为 0，旧碰撞 Token 使用为 0。Train/Valid 实际行数与 SHA256 均等于 manifest；总扫描 2,296.61 秒，含启动校验共 2,362.50 秒，进程退出码为 0。
- 产物：正式状态为 `outputs/tiger_joint/preflight/j1_full_v1/run_state.json`，SHA256 `47e5151e4755a9996e0a46beb90a9420dd747adc377f3eba928aa8e82f25dbc0`；日志和退出码位于 `outputs/run_control/tiger_joint_j1_full_v1/`，日志 SHA256 为 `950df90867f1e9b255a7bc3aad9d1145018de0a0060e12645ae0df9b301b9fc0`。产物只含聚合统计、输入指纹和协议，不保存真实样本文本、动态缓存、SID mapping 或 checkpoint。
- 当时结论：旧适配数据在 1,024 Token 下无需历史裁剪，且适配链路可复现原切分/抽样。当前结论：这些长度分布仍可作工程参考，但旧数据输入和旧初始化文件依赖违反严格隔离要求，`J1` 正式闸门重新打开。
- 下一步：从原始行为和 BGE 行序全量重建 Train/Valid，再用 vanilla Qwen fresh 词表重新执行全量预检；在新实验通过前不实现正式三轮训练入口。

### 2026-08-27：严格 SID-free 真实数据与单卡联合单步 smoke

- 目标：证明新主路径不读取旧 TIGER SID 仍能从真实行为得到 BGE 行号、生成动态三级槽位，并用 vanilla Qwen fresh 扩词表与 fresh RQ-VAE 完成一次联合更新。本节是有界 smoke，不分配实验编号，不作为全量 `J1` 门槛或效果实验。
- 数据：从原始订单目录直接扫描 453 行，使用 BGE `poi_ids.jsonl` 将原始历史/目标 `poi_id` 映射到行号，保留 Train/Valid 各 32 条；构建指纹为 `9f2121d896d70535a1271fa8696f36de21feaad1a4dd29077d2a2ac5797cf23d`。扫描中遇到的 28 条 Test 日期记录在解析 `source_dt` 后、行为格式化前跳过，未产出 `test.jsonl`，训练入口也禁止读取 `test.jsonl`。
- 隔离契约：manifest 明确记录旧 TIGER JSONL、SID mapping、RQ-VAE checkpoint、Qwen-TIGER checkpoint 均未加载；BGE shape 为 `[2337178,1024]`，manifest signature 为 `2ee5349438f7d668d96f9dfc8d1e217142972e526860d417b327aafaeadbeaa4`，`poi_ids.jsonl` SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`。数据、预检和 smoke 均对这三个值做一致性校验。
- 词表与模型：直接从 `models/Qwen3-0.6B` 扩展 5,120 个 fresh Token，最终词表为 156,789；没有旧 `C`/`D` Token 或 `<POI_TIGER_ID>`。RQ-VAE 随机新建为 `1024→512→256`、`1024³`，未加载任何 checkpoint。该 smoke 的随机初始化只用于梯度连通性，正式训练仍需执行本方法自己生成的全目录 KMeans 初始化。
- Token 预检：先对真实 Train/Valid 各 2 条运行 prefix 预检，4 条完整序列长度为 39～299，目标恒为 7 Token，超过 1,024 和目标截断均为 0；该结果只覆盖前缀，不替代全量门槛。
- GPU 结果：在 NVIDIA RTX A6000 上从 32 条 Train 候选中得到 30 个不同目标行和 8 个 fresh 初始三级 SID，稳定选取 2 条不同目标/不同 SID 样本完成一个 optimizer step。`L_total/L_gen/L_align/L_rq` 为 `20.883993/20.815416/0.643324/0.004244`；裁剪前 Qwen、Query 投影、RQ encoder/decoder/codebook 梯度范数为 `613.994896/0.482921/0.644863/0.002613/0.004288`，均为有限正数。单步耗时 9.7940 秒，峰值 allocated/reserved 为 6,092,873,728/6,203,375,616 bytes，未保存 checkpoint。
- 失败尝试：受限沙箱中的第一次入口因 CUDA 不可见而中止；宿主 GPU 上只给 2 条 Train 候选的第二次入口因两条样本被 fresh RQ-VAE 量化为同一 SID 而按预设 smoke 规则拒绝。扩大候选到 32 条后通过；两次失败均不是训练实验，也没有指标结论。
- 产物：数据在 `outputs/tiger_joint/data/sid_free_smoke32_v1/`，prefix 预检在 `outputs/tiger_joint/preflight/sid_free_smoke_v1/run_state.json`，成功单步在 `outputs/tiger_joint/smoke/sid_free_single_step_v3/run_state.json`。产物不保存样本文本副本之外的旧 SID、全目录 SID mapping 或 checkpoint。
- 结论与下一步：严格隔离的数据、Tokenizer、BGE mmap gather 和联合梯度链路可运行；尚不能说明 loss 会下降或检索有效。下一步仅全量构建 SID-free Train/Valid 并重跑正式 1,024 Token 门槛，完成后再决定是否进入小样本过拟合。

### EXP-20260827-01：严格 SID-free Train/Valid 全量构建与 Token 预检

- 状态：已完成。全量 SID-free 数据构建与同一冻结产物上的正式 Token 预检均退出 0，`formal_gate_passed=true`；本实验未启动 KMeans、训练或评测。
- 目标与假设：从原始订单 `poi_id` 和冻结 BGE `poi_ids.jsonl` 行序独立重建 Train/Valid，证明主路径不读取旧 TIGER JSONL、旧 SID mapping、旧 RQ-VAE checkpoint 或旧 Qwen-TIGER checkpoint；随后用 vanilla `Qwen3-0.6B` 加 5,120 个 fresh Token 完整扫描，验证所有 Source+Target 不超过 1,024 Token 且 Assistant 目标截断为 0。假设原始订单重建的有效 Train/Valid 行数与公平对照口径闭合，但只以本次实际产物为准。
- 数据版本：原始订单目录为 `data/beijing_order_clean_20260701_20260714_history10_20260401_20260630_json/`；Train 为 2026-07-01—12，Valid 为 2026-07-13，2026-07-14 Test 在解析 `source_dt` 后、行为格式化前跳过且不写 `test.jsonl`。历史窗口固定为 2026-04-01—06-30、最多 10 条。POI 目录只复用 `outputs/embeddings/beijing_poi_bge_m3/` 的 2,337,178×1,024 BGE-M3 mmap 与 `poi_ids.jsonl` 行序；已知 embedding manifest signature 为 `2ee5349438f7d668d96f9dfc8d1e217142972e526860d417b327aafaeadbeaa4`，POI ID 文件 SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`，最终仍以 full manifest 复核值为准。
- 代码与工作树：Git revision 为 `54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树未提交；`src/poi_gr/methods/tiger_joint/` 和 `scripts/tiger_joint/` 为本方法独立实现。既有 MMBERT 文档修改和 `third_party/LLaMA-Factory` 子模块脏状态属于并行遗留状态，本实验不修改、不清理；运行状态与最终文档保存实际 `git status --short`。
- 构建配置：数据 schema 为 `tiger-joint-sid-free-data-v1`，动态历史/目标只保存 BGE 行号与 `<S1_0><S2_0><S3_0>` 槽位；三级容量为 `1024³`，用户哈希 Token 为 2,000 个，Geohash 长度为 6。输出采用同目录临时目录完成后原子替换，全量逐分片流式读取，BGE 只读 mmap，构建阶段不加载 Qwen 或 RQ-VAE。
- 预检配置：直接加载 `models/Qwen3-0.6B` tokenizer 并追加本次 `special_tokens.json`，`cutoff_len=1024`、不 packing、batch 4,096、每 250,000 行报告进度；扫描器必须看到 `scan_mode=full` 和四项旧产物加载标志全为 `false`，且拒绝任何 `test.jsonl`、旧 `C/D` Token 或 `<POI_TIGER_ID>`。
- 环境：开发服务器 CPU 流式任务，项目环境 `/ofs/map_search/hudan/envs/poi-gr`；日志、PID 和退出码写入 `outputs/run_control/tiger_joint_exp_20260827_01/`。运行前确认无同名进程，工作盘可用空间约 22 TiB；本实验不占用 GPU。

~~~bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/build_sid_free_data.py \
  --output-dir outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1

TMPDIR=outputs/tmp/tjf1 TOKENIZERS_PARALLELISM=true \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/preflight_dynamic_data.py \
  --data-dir outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1 \
  --output-dir outputs/tiger_joint/preflight/EXP-20260827-01_sid_free_full_v1 \
  --batch-size 4096 \
  --progress-every 250000
~~~

- 构建结果：32/32 个原始分片完整扫描，共 8,790,513 行；Train/Valid 为 7,586,410/597,421，与公平对照行数一致，Test 日期 606,682 行在行为格式化前跳过，空 Query 为 0，保留历史事件 40,338,794 条。Train/Valid SHA256 为 `eee9781abb6eb90251ce1b106240b746df18496b2256f75a3edc3a59bd557370` / `1d59b7d0c97216f614c8ee4d4eb565adc6d8df6fa424399eca353dc279235c40`，构建指纹为 `515fd5d179105e66e997d921eb48e4ab7e1aae51002f7cbb853fb9c11902439c`；全部输入分片的逐文件 SHA256 保存于 manifest。构建从 11:13:34 到 12:09:43，共 56 分 09 秒，退出码 0；最终目录没有 `test.jsonl`，四项旧产物加载标志均为 `false`。
- 运行说明：启动监督时曾误触发第二个同命令临时构建；发现两个独立 `.building-*` 目录后立即停止后启动者并删除其约 395 MiB 未完成临时副本，只保留 11:13:33 启动的宿主进程 `3461112`。两者从未写入正式目录，最终 manifest、输出哈希和退出码全部来自保留的单一原子构建；重复启动曾覆盖进度日志开头，因此输入完整性只以 manifest 的 32 个分片哈希和后续预检复算为准。
- 预检结果：Train/Valid 的 7,586,410/597,421 行及 SHA256 均在 Token 扫描中重新计算并与数据 manifest 精确一致，共验证 8,183,831 行、40,338,794 个历史事件、48,522,625 个历史/目标 POI 行引用和 145,567,875 个动态三级槽位。完整序列 min/P50/P90/P95/P99/P99.9/max 为 `39/147/296/304/337/445/946`，Source 最大 939，Target 恒为 7；超过 1,024、目标截断和旧 collision Token 使用均为 0。Train/Valid 扫描分别耗时 1,584.75/145.92 秒，纯扫描共 1,731.57 秒；程序内部总耗时 1,894.87 秒，宿主控制从 12:51:59 到 13:24:52，退出码 0。
- 初始化与隔离复核：预检只读取 Train/Valid，`test_samples_read=false`，`old_sid_artifacts_loaded=[]`、`legacy_artifacts_loaded=[]`。vanilla Qwen model config SHA256 为 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`；fresh 初始化契约仍由方法常量和 BGE 行数生成，seed 42 的 23,372 行 Validation 与 500,000 行 KMeans 抽样 SHA256 分别为 `409e476abf82cd609447c6c9a6f437d96ecd897a40c5e610a33a3bc996041249` / `0f7f22d7a7543965d581bcc9098a833533dbbdbb701b867f54c76d0bf66d6eb7`。这些只冻结下一阶段协议，本实验没有运行 FAISS KMeans，也没有创建 RQ-VAE/Qwen checkpoint。
- 产物：数据冻结于 `outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1/`，manifest SHA256 为 `473c20cdf5b179ae9e28d4d2cad047c7dbe03323db381fb4ee41c74f71028766`；正式预检状态为 `outputs/tiger_joint/preflight/EXP-20260827-01_sid_free_full_v1/run_state.json`，SHA256 为 `b6189094fbe454971aa6d152cd23e4729dc3633dd332c0bdd6718208fd88271b`。日志、PID、时间和退出码位于 `outputs/run_control/tiger_joint_exp_20260827_01/`，预检日志 SHA256 为 `d151f9593484a5ce5f5520c8129cb08ddc13393bc7e001c80b4eb3ee6ee0f34b`；产物不含 Test 样本、SID mapping、KMeans 中间量或 checkpoint。
- 结论与下一步：严格 SID-free `J1` 正式门禁关闭：原始行为与冻结 BGE 行序可以无损重建公平对照的 Train/Valid，vanilla Qwen fresh 词表下不需要裁剪历史或目标，旧 SID/Qwen-TIGER/RQ-VAE/Test 依赖均为 0。该结论只证明数据和 Token 契约可进入下一工程阶段，不证明联合训练有效。下一最小步是 `J2` 有界小样本过拟合，先验证 loss 可下降和动态 SID 随参数更新仍保持批内一致；本实验不实现或启动三轮正式训练。

### EXP-20260827-02：fresh SID 与 Qwen 固定小样本联合过拟合

- 状态：已完成，进程退出码 0，但 `j2_bounded_overfit_gate_passed=false`。本实验只执行 30 个 optimizer step，未保存 checkpoint，不属于正式三轮训练。
- 目标与假设：固定两个不同目标 POI、两个初始不同 fresh SID 的 Train 样本和一个 64 POI 目录批，每个 batch 同时更新 Qwen、Query 投影与随机新建 RQ-VAE。假设联合路径不仅能让生成 loss 明显下降，还能在 RQ-VAE 更新导致 SID 变化时持续使用当步新码监督 Qwen，并保持两个不同目标的三级 SID 不塌缩。若 loss 下降但目标 SID 合并，则判定过拟合门禁失败，不能把更容易记忆的退化目标误当成联合训练有效。
- 数据版本：只读取 `EXP-20260827-01_sid_free_full_v1/train.jsonl` 前 64 行用于确定固定 batch；Valid/Test 样本均不读取。数据 manifest/build fingerprint 为 `473c20cdf5b179ae9e28d4d2cad047c7dbe03323db381fb4ee41c74f71028766` / `515fd5d179105e66e997d921eb48e4ab7e1aae51002f7cbb853fb9c11902439c`，并强制绑定 `formal_gate_passed=true` 的预检状态 `b6189094fbe454971aa6d152cd23e4729dc3633dd332c0bdd6718208fd88271b`。POI 向量仍只用冻结 BGE-M3 `[2337178,1024]` mmap 与其行序。
- 代码与工作树：Git revision 为 `54802e6674e283722ecee00fb862530df30cef9a`，运行前工作树未提交；本实验新增独立入口 `scripts/tiger_joint/overfit_joint.py`，并在联合 loss 输出中增加教师强制 Token/整段准确率。既有 MMBERT 文档修改和 `third_party/LLaMA-Factory` 子模块脏状态不属于本实验，不修改、不清理。
- 配置：seed 42；从前 64 条 Train 候选按“目标行与初始 fresh SID 均不同”稳定选 2 条，固定复用 30 step；固定均匀目录批 64；RQ-VAE 为随机新建 `1024→512→256`、`1024³`，不运行 KMeans、不加载 checkpoint；Qwen 从 `models/Qwen3-0.6B` 新扩至 156,789 词，BF16 全参数更新。`lr_qwen=5e-5`、`lr_rq=3e-4`、`λ_align=0.1`、`λ_rq=1.0`、InfoNCE temperature 0.07，各参数组裁剪阈值 1.0；每一步各执行一次 Qwen+投影 optimizer step 和一次 RQ-VAE optimizer step。
- 验收门槛：30/30 step 完成且无 NaN/Inf；每一步 Qwen、Query 投影、RQ-VAE、RQ encoder/decoder/codebook 六组梯度均为有限正数；每次观察的 RQ codes、输入和 labels 完全一致；最终 generation loss 相对初始至少下降 50%，教师强制 Token 准确率至少 80%，两个样本自由生成的五 Token `<TARGET_POI>+S1/S2/S3+</TARGET_POI>` 精确率与合法率均为 100%，并且初始两个不同完整 SID 在最终仍为两个不同完整 SID。任一项不满足即 `j2_bounded_overfit_gate_passed=false`。
- 环境：单卡 NVIDIA RTX A6000，项目环境 `/ofs/map_search/hudan/envs/poi-gr`，短路径 `TMPDIR=outputs/tmp/tjo1`；正式日志、PID、起止时间与退出码写入 `outputs/run_control/tiger_joint_exp_20260827_02/`。启动前必须再次确认没有训练进程且 GPU 空闲。

~~~bash
CUDA_VISIBLE_DEVICES=0 \
TMPDIR=outputs/tmp/tjo1 \
TOKENIZERS_PARALLELISM=false \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/overfit_joint.py \
  --data-dir outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1 \
  --preflight-state outputs/tiger_joint/preflight/EXP-20260827-01_sid_free_full_v1/run_state.json \
  --output-dir outputs/tiger_joint/overfit/EXP-20260827-02_sid_free_bounded_v1 \
  --optimizer-steps 30 \
  --train-scan-rows 64 \
  --behavior-batch-size 2 \
  --catalog-batch-size 64
~~~

- 启动前 smoke：同一输入与超参数的 3-step 临时 smoke 已退出 0，generation loss 从 `20.815416` 降至 `10.110707`，峰值 allocated/reserved 为 `6,109,950,976/7,971,274,752` bytes；两个初始不同 SID 在第 3 步暂时合并为一个，因此该现象已加入正式门禁。smoke 状态 SHA256 为 `925b1d02bedcbbec35eaeeb46f5f0b5b56444d7426d86031e77882ce497f6a25`，不作为正式结果，也不保存 checkpoint。
- 核心指标：30/30 step 完成，所有观察点的当前 RQ codes 与动态 input/labels 完全一致；Qwen、Query 投影、RQ-VAE、RQ encoder/decoder/codebook 六组梯度每一步均为有限正数。初始→最终 `L_total/L_gen/L_align/L_rq` 为 `20.883993→4.635510 / 20.815416→4.564740 / 0.643324→0.693147 / 0.004244→0.001456`，generation loss 下降 `78.0704%`，通过 50% 单项阈值。教师强制 Token 准确率仅从 `14.2857%` 到 `57.1429%`，整段准确率始终为 0；最终两个自由生成均在 2 Token 后结束，五 Token 动态目标精确率与合法率均为 `0/2=0%`。
- SID 与梯度诊断：两个固定目标的初始完整 SID 数为 2；第 3 步其中一条在 S1/S2 同时变化，完整 SID 数降为 1，此后到第 30 步均未恢复。`L_align` 从第 3 步起基本固定为 `log(2)=0.693147`；Query 投影梯度范数范围为 `5.90e-9～1.4795`，虽保持数值正值但在塌缩后接近零。Qwen/RQ encoder/RQ decoder/codebook 梯度范数范围分别为 `17.5306～613.9949 / 0.06831～2.03029 / 0.000419～0.002613 / 0.003293～0.004951`。fresh Qwen 新增 embedding 行、Query 投影和 RQ-VAE 的初始/最终 SHA256 均不同，确认三部分实际更新。
- 运行与产物：宿主控制时间为 2026-08-27 14:50:30—14:54:16，程序内训练与评估耗时 17.85 秒；峰值 allocated/reserved 为 `6,109,950,976/7,971,274,752` bytes。唯一实验产物为 `outputs/tiger_joint/overfit/EXP-20260827-02_sid_free_bounded_v1/run_state.json`，SHA256 `3b07cafc20d621ffdda930c9e64b498011869aa098e72223b4cea9decbd1507f`；日志 SHA256 为 `37dd15ed6e3c6cccb23f21bda54595d293b886096e2fa72829bd3a365829001a`，PID、时间与退出码位于 `outputs/run_control/tiger_joint_exp_20260827_02/`。产物只有聚合 trace 与哈希，不保存真实样本文本、原始生成 Token、SID mapping、Valid/Test 数据或 checkpoint。
- 结论与下一步：本实验通过了“同一步动态 SID 监督正确、联合三部分持续更新、单卡显存可承受”三个工程条件，但否定了随机初始化下“generation loss 下降即可视为过拟合成功”。联合 RQ 参数在第 3 步更新后观察到目标码塌缩，随后 InfoNCE 退化且 Qwen 未学会完整动态目标，因而不能进入多卡或三轮正式训练。该结果不直接否定包含全目录方法自有 KMeans 初始化和真实多样行为流的完整第一版；下一最小闭环只能实现冻结协议中的方法自有 KMeans 初始化，并原样复跑本实验，先证明两个目标不塌缩且自由生成通过，再讨论多卡同步和训练器。

### EXP-20260827-03：方法自有 KMeans 初始化与 J2 联合学习闭环

- 状态：已完成。fresh RQ-VAE KMeans 初始化退出码 0、`formal_initialization_passed=true`；相同固定 batch 的 30-step 进程退出码 0 但原时长门禁仍为 false；不改变模型、数据和 loss 的 100-step 有界诊断退出码 0，`j2_bounded_overfit_gate_passed=true`。三段均未读取 Valid/Test 样本，未保存训练 checkpoint 或全量 SID mapping。
- 目标与假设：检验 `EXP-20260827-02` 的失败究竟是联合框架不可学，还是随机码本在极小 batch 上早期越过 Voronoi 边界导致动态监督退化。假设方法自有顺序 KMeans 能先稳定并分散三层 codebook；若目标 SID 不再塌缩但 30 step 仍只学会静态结构 Token，则延长同一固定过拟合轨迹可以区分“框架不能学”与“fresh SID Token 所需步数超过原门禁时长”。
- 数据版本：KMeans 只读取冻结 BGE-M3 `[2,337,178,1,024]` mmap，manifest signature/POI 行序 SHA256 为 `2ee5349438f7d668d96f9dfc8d1e217142972e526860d417b327aafaeadbeaa4` / `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`；固定 1% 目录 Validation 为 23,372 行，余下 2,313,806 行中无放回抽取 500,000 行，索引 SHA256 为 `0f7f22d7a7543965d581bcc9098a833533dbbdbb701b867f54c76d0bf66d6eb7`。J2 仍只 Tokenize `EXP-20260827-01` Train 前 64 行，再固定选择相同的两个不同目标与 64 条目录；数据 manifest/build fingerprint 和正式预检 SHA256 分别为 `473c20cdf5b179ae9e28d4d2cad047c7dbe03323db381fb4ee41c74f71028766` / `515fd5d179105e66e997d921eb48e4ab7e1aae51002f7cbb853fb9c11902439c` / `b6189094fbe454971aa6d152cd23e4729dc3633dd332c0bdd6718208fd88271b`。
- 代码与工作树：Git revision `54802e6674e283722ecee00fb862530df30cef9a`，工作树未提交。新增 `initialization.py` 与 `initialize_rqvae.py`，初始化产物严格绑定 BGE 行序和 J1 预检哈希；共享 KMeans helper 的层数迭代改为读取实际 RQ-VAE `model.codebook_sizes`，使方法级最小配置不必伪造共享训练配置。旧 SID、旧 RQ-VAE 和 Qwen-TIGER 产物加载列表均为空；既有 MMBERT 文档修改与 LLaMA-Factory 子模块脏状态未清理。
- 配置：fresh RQ-VAE 仍为随机新建 encoder/decoder 的 `1024→512→256` 与 `1024³`，只用 fresh encoder 编码固定 500,000 行后执行三级 residual `faiss_gpu` KMeans，seed 42、每层 20 iteration、batch 8,192、max points/centroid 2,048。过拟合继续使用 vanilla Qwen BF16 fresh 扩词、行为 batch 2、目录 batch 64、`lr_qwen=5e-5`、`lr_rq=3e-4`、`λ_align=0.1`、`λ_rq=1.0`、temperature 0.07 和各组 clip 1.0；100-step 诊断仅把 `optimizer_steps` 从 30 改为 100。
- 命令：正式初始化与两段 J2 均在项目环境、单卡和短 `TMPDIR` 下运行；下列命令省略了外层 run-control 的 PID、日志、时间和退出码包装。

~~~bash
CUDA_VISIBLE_DEVICES=0 TMPDIR=outputs/tmp/tjk1 \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/initialize_rqvae.py \
  --preflight-state outputs/tiger_joint/preflight/EXP-20260827-01_sid_free_full_v1/run_state.json \
  --output-dir outputs/tiger_joint/initialization/EXP-20260827-03_fresh_kmeans_v2

CUDA_VISIBLE_DEVICES=0 TMPDIR=outputs/tmp/tjo1 \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/overfit_joint.py \
  --data-dir outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1 \
  --preflight-state outputs/tiger_joint/preflight/EXP-20260827-01_sid_free_full_v1/run_state.json \
  --rq-initialization-dir outputs/tiger_joint/initialization/EXP-20260827-03_fresh_kmeans_v2 \
  --output-dir outputs/tiger_joint/overfit/EXP-20260827-03_kmeans_bounded_v1 \
  --optimizer-steps 30 --train-scan-rows 64 \
  --behavior-batch-size 2 --catalog-batch-size 64

CUDA_VISIBLE_DEVICES=0 TMPDIR=outputs/tmp/tjo1 \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/overfit_joint.py \
  --data-dir outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1 \
  --preflight-state outputs/tiger_joint/preflight/EXP-20260827-01_sid_free_full_v1/run_state.json \
  --rq-initialization-dir outputs/tiger_joint/initialization/EXP-20260827-03_fresh_kmeans_v2 \
  --output-dir outputs/tiger_joint/smoke/kmeans_overfit_100step_v1 \
  --optimizer-steps 100 --train-scan-rows 64 \
  --behavior-batch-size 2 --catalog-batch-size 64
~~~

- 初始化结果：首次写入 `...fresh_kmeans_v1` 的运行在完成 500,000 行 latent 编码后，因方法配置缺少共享 helper 假定的 `codebook_sizes` 属性退出 2；只留下 `status=failed` manifest 和样本索引，没有 checkpoint，随后修正接口并从头运行 `v2`。正式 `v2` 的三层 KMeans 分别使用 `1024/1024/1024` 个码、dead code 均为 0，初始化样本 MSE 为 `0.00200944/0.00156306/0.00129009`；初始化模型、checkpoint 和 manifest SHA256 分别为 `822c1b6bdf5c7833c5c85674517bee03894ab990939f38d2759215e3300aa137` / `41f29dd7cd9c46893cf2a20e18283900876db53adb163e89f351e1bf642a768a` / `2806be09e4ccf96524f187a1d34b22add8dc0c44628394c117d6e9e2b20367c2`。
- 30-step 结果：两个初始不同目标 SID 在最终仍为 2 个，未重现随机初始化的 2→1 塌缩；generation loss `21.526163→4.993332`，下降 `76.8034%`，alignment/RQ loss 最终为 `0.00013745/0.00096215`。但教师强制总 Token 准确率仍为 `57.1429%`，自由精确/合法率仍为 0/2；这 57.14% 对应 7 个监督 Token 中 4 个静态结构 Token 已学会，而 3 个 SID Token 尚未学会，因此原 30-step 门禁按实标记失败。
- 100-step 结果：相同初始化和 batch 下，SID Token 准确率在 step 54 首次高于 0、step 70 达 66.67% 且总 Token 达 85.71%，step 88 起 SID/静态/总 Token 和整段准确率均为 100%。最终 generation/align/RQ loss 为 `0.043157/0/0.00050664`，generation 相对初始下降 `99.7995%`；两个目标完整 SID 始终保持 distinct=2，最终自由动态目标精确率与当前 SID 合法率均为 `2/2=100%`。六组梯度每一步均为有限正数、每个观察点的动态 SID/input/label 一致，全部 J2 检查通过。
- 环境与资源：Python 3.10.20、torch 2.9.1+cu128、CUDA 12.8、NumPy 1.26.4、单卡 NVIDIA RTX A6000。正式 KMeans 程序内耗时约 105.15 秒，峰值 allocated/reserved `85,074,944/134,217,728` bytes；30/100-step 训练评估分别耗时 16.63/40.54 秒，峰值 allocated/reserved 约 `6,110,119,936/7,969,177,600` bytes。宿主 run-control 覆盖 15:59:20—16:30:25，所有重试后正式段退出 0。
- 产物：正式初始化位于 `outputs/tiger_joint/initialization/EXP-20260827-03_fresh_kmeans_v2/`；30/100-step 状态位于 `outputs/tiger_joint/overfit/EXP-20260827-03_kmeans_bounded_v1/run_state.json` 和 `outputs/tiger_joint/smoke/kmeans_overfit_100step_v1/run_state.json`，SHA256 为 `76a20ce0cacb3c4ebd360f148f28ee4583532c3f9a171b28ec052edb6520121c` / `91d2cdcb80e84f45e3d719cba5817603fd7ad0c49f619bed782f5c7629dd8acf`。日志、PID、起止时间和退出码位于 `outputs/run_control/tiger_joint_exp_20260827_03/`；KMeans retry、30-step 和 100-step 日志 SHA256 分别为 `8ef8e853bb84be3fe95b1d8f3da4c72babec0be4ddc7a8923cd7529bf0a07173` / `025284ea518fe13bfb8054fd9e2b07fb79eda85749ad2e715e10fac204d01` / `ed9f985ac59ba04efe3646cab3f801fbcd7c84c0732695c6693dc3d377453c2a`。
- 结论与下一步：当前联合框架不是“完全按照 UniSearch”才可学习；真正阻断 `EXP-02` 的是随机码本造成的早期目标退化，而 KMeans 后原 30-step 失败主要是 fresh SID Token 行学习时长不足。`L_gen+0.1L_align+L_rq`、每个 batch 同步更新 Qwen/投影/RQ-VAE 的最小设计已通过单卡可学习性门禁。下一步不改方法，只在真实多样 Train 流中加入梯度累积、独立目录采样和固定全目录 probe，先做有界训练确认 codebook 利用率、SID churn、显存和吞吐，再决定完整三轮的资源布局。

### 2026-08-27：J3 流式训练入口与四卡 DDP 契约核验

本节是代码与 smoke 核验，不是新的正式实验，不分配 `EXP` 编号，也不登记三轮训练指标。

- 真实多样流 smoke：`scripts/tiger_joint/train_joint.py` 已只读取严格 SID-free `train.jsonl`，使用 buffer shuffle、动态 padding、当前 RQ-VAE 动态回填三级 SID，以及独立全目录 POI 采样。单卡行为 batch 8、累积 1、目录 batch 64 的 2-step 运行退出 0；两个 step 各含 8 个不同目标行和 8 个不同三级 SID，generation loss 从 `21.300` 降至 `15.200`，各参数组梯度有限。固定 4,096 行 probe 的完整 SID 数从 4,093 变为 4,091，三层最终使用码数约为 954/844/766；峰值 allocated/reserved 为 `8,331,326,976/11,815,354,368` bytes。产物位于 `outputs/tiger_joint/smoke/diverse_train_2step_v1/run_state.json`，不含 checkpoint。
- DDP 数据与参数契约：全量 7,586,410 行按 JSONL byte offset 精确切为 `1,896,602/1,896,603/1,896,602/1,896,603` 行，各 rank 都是 237,076 个 microbatch，不重复也不丢行。Qwen、Query 投影和 RQ-VAE 被同一个 DDP forward 边界持有，非累积边界使用 `no_sync()`，optimizer 边界三部分一起同步和 step。两进程 CPU/Gloo 合成 smoke 退出 0，生成器、投影、RQ encoder 与 codebook 梯度均为有限正数，step 后两 rank 全部参数逐元素一致；该结果验证同步结构，不替代四卡 NCCL 实测。
- 精确累积口径：行为目标 `L_gen+0.1L_align` 按当前累积组的实际全局行为行数取均值，目录 `L_rq` 按实际目录采样行数独立取均值。每轮最后不足 16 个 microbatch 的累积组因此不会用同一个补偿系数混合两类样本。InfoNCE 负例仍限定在每 rank 当前行为 microbatch 8 条内，DDP 只平均四卡梯度；第一版没有实现跨 rank gather，不能把它写成 32 条全局 InfoNCE batch。
- 四卡配置：`configs/tiger_joint/v1_4gpu.yaml` 固定每 rank 行为 batch 8、累积 16、目录 batch 64，等效全局行为 batch 512、完整 optimizer step 目录行 4,096；每轮 14,818 个 optimizer step，三轮共 44,454 step。Qwen/投影峰值学习率为 `5e-5`，RQ 峰值学习率为 `3e-4`，二者均先经过 3%（1,333 step）线性 warmup，再按同一归一化 cosine 曲线衰减至 0；`λ_align=0.1`、`λ_rq=1.0` 和 loss 定义不变。RQ warmup 用于降低 fresh 动态 SID 在 Qwen Token 行尚未学稳时的早期漂移，不代表降低 `L_rq` 权重。
- 恢复契约：每个完整 epoch 保存 Qwen、fresh tokenizer、RQ-VAE、Query 投影、两个 optimizer、两个 scheduler 和每个 DDP rank 各自的 Python/NumPy/CPU/CUDA RNG state；恢复时校验配置签名、world size、epoch 边界与全部文件 SHA256，并分别恢复 Qwen/RQ 的 scheduler step，只允许写入新的输出目录。这样可以继续同一条三轮学习率轨迹，同时拒绝旧 SID/Qwen-TIGER checkpoint 和不完整 step checkpoint。
- 平台入口：本地忽略文件 `launchers/run_train_tiger_joint_4x6000d_3epoch.sh` 固定申请/校验 4 张 RTX PRO 6000D，并调用 `torch.distributed.run --nproc_per_node=4`。脚本 `bash -n`、配置/输入路径门禁、可执行权限和 44,454-step 计算均已通过；开发服务器当前无法连接 NVIDIA driver，因此未冒充执行四卡 NCCL 或平台 dry-run。平台中先运行下列 dry-run，再使用同一命令去掉参数启动；正式启动后才创建 `EXP-20260827-04` 记录。

~~~bash
bash launchers/run_train_tiger_joint_4x6000d_3epoch.sh --dry-run
bash launchers/run_train_tiger_joint_4x6000d_3epoch.sh
~~~

- 当前结论：四卡资源在已有单卡显存证据下有充足余量，当前剩余风险不是容量，而是训练平台上的 NCCL、实际吞吐与长轨迹 SID churn。下一步只执行平台 dry-run 和最短 NCCL smoke；二者通过后原样启动三轮，不再回到本地长训，也不在训练前改 loss 或加入新方法。

### EXP-20260827-04：四卡三轮 SID-free 联合训练

- 状态：已完成。训练平台进程退出码为 0，三轮共 44,454 个 optimizer step，三个 epoch 边界 checkpoint 均完整保存；训练期间只读取 Train，不读取 Validation/Test，也未加载旧 SID、旧 RQ-VAE 或 Qwen-TIGER checkpoint。本实验只验收联合训练产物，不将训练 probe 当作检索效果。
- 目标与假设：在不改变 `EXP-20260827-03` 已通过的三项 loss、fresh KMeans 初始化和动态 SID 数据流的前提下，用四卡完整跑过三个行为 epoch。假设全目录 RQ 重建锚点能够在长轨迹中维持可用码本，Qwen 能跟随每个 batch 的当前 hard SID 监督，最终 checkpoint 可在冻结全目录映射上获得不低于 TIGER 的三级 Bucket 覆盖。
- 数据版本：Train 为 `outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1/train.jsonl` 的 7,586,410 行，data manifest/build fingerprint 为 `473c20cdf5b179ae9e28d4d2cad047c7dbe03323db381fb4ee41c74f71028766` / `515fd5d179105e66e997d921eb48e4ab7e1aae51002f7cbb853fb9c11902439c`；正式预检 SHA256 为 `b6189094fbe454971aa6d152cd23e4729dc3633dd332c0bdd6718208fd88271b`。冻结 BGE manifest signature/POI 行序 SHA256 为 `2ee5349438f7d668d96f9dfc8d1e217142972e526860d417b327aafaeadbeaa4` / `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`；fresh KMeans 初始化 manifest SHA256 为 `2806be09e4ccf96524f187a1d34b22add8dc0c44628394c117d6e9e2b20367c2`。四个 rank 的无重叠行数为 1,896,602/1,896,603/1,896,602/1,896,603，总和与 Train 完全一致。
- 代码与工作树：Git revision 为 `54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树未提交；完整 `git status --short` 已写入 `run_state.json`。训练使用独立 `src/poi_gr/methods/tiger_joint/`、`scripts/tiger_joint/train_joint.py`、`configs/tiger_joint/v1_4gpu.yaml` 和平台 launcher，既有 TIGER baseline 未被改写。
- 配置与更新流：vanilla Qwen3-0.6B 由 151,936 扩至 156,789 词表并以 BF16 全参数训练；fresh RQ-VAE 为 `1024→512→256`、三级 `1024³`。每 rank 行为 batch 8、梯度累积 16、目录 batch 64，等效全局行为 batch 512、每个 optimizer step 目录样本 4,096。每个累积边界同时更新 Qwen、Query 投影和 RQ-VAE，loss 固定为 `L_gen + 0.1×L_align + 1.0×L_rq`；Qwen/RQ 峰值学习率为 `5e-5/3e-4`，共同经过 1,333 step 线性 warmup 后 cosine 衰减至 0，clip norm 为 1.0。配置签名为 `825bcfcb0da3e5676861229b02f29c01920726a06c3e3067bcaec8cceb944383`。

~~~bash
bash launchers/run_train_tiger_joint_4x6000d_3epoch.sh
~~~

- 环境与资源：4×NVIDIA RTX 6000D，Python 3.10.20、torch 2.9.1+cu128、CUDA 12.8、NumPy 1.26.4。训练从北京时间 2026-08-27 20:08:09 运行至 2026-08-29 09:01:42，程序内耗时 132,477.67 秒（36.7994 小时）；各卡峰值 allocated 为 15.94/18.00/18.00/20.88 GB。三轮实际消费 22,759,230 条行为样本，每轮 14,818 step。
- 训练终态：固定 4,096 行 probe 的完整 SID 数从 4,093 保持为 4,093；S1/S2/S3 使用码数从 962/848/776 变为 414/938/927，对应利用率从 93.95/82.81/75.78% 变为 40.43/91.60/90.53%。最终相邻 probe churn 为 0 只说明训练末段映射稳定，不能证明 Qwen 在此前动态标签变化中保持对齐。RQ-VAE 模块 SHA256 从 `822c1b6bdf5c7833c5c85674517bee03894ab990939f38d2759215e3300aa137` 变为 `22c83345274a4df0ba5fab25e136ab8b4c1b938e44e3ee9977dfbb0046c097c9`，确认 RQ-VAE 实际更新。
- 产物：训练目录为 `outputs/tiger_joint/train/EXP-20260827-04_joint_v1_4x6000d_e3/`，`checkpoint-step-14818/29636/44454` 分别对应 epoch 1/2/3；每个目录包含 Qwen/tokenizer、RQ-VAE、Query 投影、两个 optimizer/scheduler 和逐 rank RNG 的 `joint_state.pt`。epoch 3 的 model/joint state SHA256 为 `f8eecbdb065b023e773b0b4ac67942d1fc0aa422516fddd38aa17f7cd6bb638c` / `01130bcbd0bf6ed2456bc06d4241b409959e2f9c2f9f31f9a6c062d2127ec406`；训练 `run_state.json` SHA256 为 `97d78ced63166d3fd5a25d09649f4c07d186b0bdd134d612ace57ce96581fffd`。日志、起止时间和退出码位于 `outputs/run_control/tiger_joint_exp_20260827_04_4x6000d/`。
- 结论与下一步：三轮训练工程门禁通过，证明四卡同步、完整数据消费和 checkpoint 恢复状态可用；probe 同时显示 S1 码本显著收缩。该实验本身不能回答联合训练是否有效，下一步只冻结 epoch 3，导出全目录当前 SID 并在固定 Validation 10k 上评测三级 Bucket；若低于 TIGER 即按预注册门禁停止，不追加训练轮次或复杂模块。

### EXP-20260829-01：epoch 3 全目录 SID 与固定 Validation 10k Bucket 评测

- 状态：已完成，正式评测与独立复核均退出 0；预注册的四个 Bucket HR 截止点全部失败。本轮只评测 epoch 3，不读取 Test，也不评测冷目标或四类泛化集。
- 目标与假设：冻结 `checkpoint-step-44454` 的 Qwen、RQ-VAE 和 tokenizer，先为全目录 2,337,178 个 POI 导出唯一的 epoch-3 三级 SID 映射，再在与 TIGER 完全相同的固定 Validation 10k 上执行无约束 Beam=10。假设若联合训练真正改善 Query 可预测的语义桶，则去除不可展开路径并按首次 Beam rank 去重后的 Bucket HR@1/3/5/10 至少达到 TIGER 的 55.53/79.64/85.10/88.06%。
- 数据版本与对齐：固定参考集为 `outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl`，SHA256 `a2e0366d3d8f582e53a5293b687dc08f85b2960d60c7fb44d2fd02168063a944`；从严格 SID-free Valid `1d59b7d0c97216f614c8ee4d4eb565adc6d8df6fa424399eca353dc279235c40` 按同一 `order_id + searchid` 对齐后的文件 SHA256 为 `5bfe2dac8d52ed336cac0ce1fb63c65afe3496805acc3146b953cbf0cc28a28b`，10,000 个业务键 SHA256 为 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`。历史 POI 的 SID 也由同一个 epoch-3 全目录映射回填；Prompt 长度 min/mean/max 为 32/150.355/538，无截断。
- 代码与工作树：Git revision 为 `54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树未提交。新增 `src/poi_gr/methods/tiger_joint/evaluation.py` 统一全目录索引、动态 Prompt、候选解析和 Bucket 指标；新增 `scripts/tiger_joint/evaluate_bucket_retrieval.py` 执行哈希门禁、SID 导出、固定 10k 生成和分片 trace；`scripts/tiger_joint/verify_bucket_retrieval.py` 不调用聚合结果，逐行重解析 100,000 个原始候选并独立复算指标。
- 评测协议：只使用 epoch 3、三级 `[S1,S2,S3]`、无碰撞 Token、无 Trie/合法路径约束的 Beam=10；报告两种口径。`raw slot` 保留非法、不可展开和重复候选占用的原 Beam 名次；主口径 `unique expandable` 删除最终全目录中不存在的路径，并按首次 Beam rank 对可展开 Bucket 去重。语法合法只要求能够解析三级 Token，能否展开必须再查冻结 epoch-3 目录索引。

~~~bash
TMPDIR=outputs/tmp/tje3 \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/evaluate_bucket_retrieval.py \
  --valid-file outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoint outputs/tiger_joint/train/EXP-20260827-04_joint_v1_4x6000d_e3/checkpoint-step-44454 \
  --embedding-dir outputs/embeddings/beijing_poi_bge_m3 \
  --output-dir outputs/eval/tiger_joint/EXP-20260829-01_epoch3_bucket_fixed10k \
  --reuse-sid-artifact-dir outputs/eval/tiger_joint/smoke_epoch3_bucket_2_v2 \
  --expected-step 44454 --expected-epoch 3 --num-beams 10 \
  --per-device-eval-batch-size 16 --sid-batch-size 8192 \
  --chunk-size 500 --cutoff-len 1024

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/verify_bucket_retrieval.py \
  --eval-dir outputs/eval/tiger_joint/EXP-20260829-01_epoch3_bucket_fixed10k
~~~

- 环境与资源：NVIDIA RTX A6000，Python 3.10.20、torch 2.9.1+cu128、CUDA 12.8、NumPy 1.26.4；生成 batch 为 16，10,000 条生成耗时 1,072.14 秒，峰值 CUDA allocated 为 13,609,360,896 bytes。全目录 SID 使用完成端到端 smoke 时已从完全相同 checkpoint/BGE 行序导出的确定性产物；正式入口重新校验 checkpoint、RQ-VAE、BGE、行序和 SID 全量哈希后复制到隔离实验目录，未复用任何旧 TIGER SID。
- 全目录 SID 构建：输出 `int32[2,337,178,3]`，S1/S2/S3 分别使用 425/1,024、1,024/1,024、1,024/1,024 个码，利用率 41.5039/100/100%；归一化熵为 0.8536/0.9743/0.9731，Kish 有效码数为 333.63/736.84/734.88。三级完整 Bucket 为 1,914,745 个，唯一率 81.9255%；碰撞 excess 422,433（18.0745%），604,971 个 POI 位于碰撞桶，桶大小 P50/P90/P95/P99/max 为 1/1/2/5/669。S1 只有 425 个前缀桶，S1 桶大小 P50/P99/max 为 5,222/14,717/20,514；固定 10k 目标桶平均/最大大小为 2.0863/171，其中 7,472 条目标落在单例桶。
- 生成与 Bucket 指标：100,000 个 Beam 候选的三级 Token 语法合法率为 99.894%，完整 wrapper 合法率为 99.884%，但可在 epoch-3 全目录中展开的候选率只有 18.410%；不可展开率为 81.484%，可展开重复 Bucket 率为 0.007%。每条请求唯一可展开 Bucket 数 min/mean/max 为 0/1.8403/10。

| 口径 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---:|---:|---:|---:|
| 原始 Beam 槽位 Bucket | 19.48% | 29.03% | 31.75% | 34.97% |
| 唯一可展开 Bucket | 22.84% | 32.90% | 34.77% | 34.97% |
| TIGER epoch 3 基线 | 55.53% | 79.64% | 85.10% | 88.06% |
| 联合 epoch 3 − TIGER | -32.69pp | -46.74pp | -50.33pp | -53.09pp |

- 独立复核与产物：正式目录为 `outputs/eval/tiger_joint/EXP-20260829-01_epoch3_bucket_fixed10k/`，包含全目录 `sid_codes.npy`、SID manifest/metrics、对齐子集、20 个各 500 行的候选 trace 分片、结果和复核状态。SID/result/verification SHA256 分别为 `ff408b33c06cd7f30ee13eaf9723a3db60b0102dd2b1bdef02315e612992d75b` / `c2c76f509ad34fa28487ed51332aa58fc8167938174f94182fd36b2b25498823` / `cd12c23dc228c6dd3cd51d26d923590cd38ed29773adb00d9f7be1903242b4cd`。独立脚本检查 10,000 行、100,000 候选和 20 个分片的原始 Token 重解析、Beam rank、目标行→SID、目录桶大小、去重顺序、分片哈希和全部 Bucket 指标，七项检查均为 true。日志、PID、时间与退出码位于 `outputs/run_control/tiger_joint_exp_20260829_01_epoch3_bucket/`。
- 结论：静态唯一性不是本轮最主要失败点；三级 SID 仍有 81.93% 唯一率，S2/S3 也全部使用，但 S1 从 fresh 初始化和训练 probe 的高利用状态收缩到全目录 425 个码。更直接的生成失败是 Qwen 大量输出“各层 Token 均合法、组合路径却不属于最终目录”的 SID：平均只剩 1.84 个可展开 Bucket，导致 HR@10 只有 34.97%。现有证据与训练期动态离散标签漂移一致——Qwen 在三轮中学习过不同 SID 分配，训练末段 churn 归零不能追溯校准早先监督——但本实验不能把它证明为唯一因果因素。
- 决策与下一步：第一版 `L_gen + 0.1L_align + L_rq`、逐 batch 同步更新 hard SID 的联合假设被拒绝。按预注册门禁不增加 `C`、Trie、Query encoder、额外 epoch、Test 或泛化评测；也不能因为只评测 epoch 3 就声称它优于 epoch 1/2。后续研究回到冻结 SID 的两阶段主线；若未来重新研究联合训练，必须先把离散标签稳定机制（如 momentum teacher、阶段性 hard mapping 刷新或显式合法路径监督）作为新的可证伪方法单独立项，而不是继续调当前 checkpoint。

### EXP-20260830-01：epoch 3 全目录合法路径约束 Bucket 诊断

- 状态：已完成。正式生成与独立复核均退出 0；本轮是在 `EXP-20260829-01` 已按门禁停止后，由用户明确要求补充的一次终态解码诊断，不修改模型、不重训、不读取 Test，也不把约束结果替换为第一版主协议结果。
- 目标与假设：固定 `checkpoint-step-44454`、epoch-3 全目录 SID、Prompt、固定 Validation 10k、Beam=10、batch=16 和 5 个生成 Token，仅在每个生成位置施加目录 Prefix 约束。假设若无约束实验的主要损失来自 81.484% 不可展开组合，则强制所有 Beam 沿真实 `[S1,S2,S3]` 路径生成后，Bucket HR 应大幅接近 TIGER；若合法率恢复但 HR 仍低，则还存在目标 Bucket 概率排序不足。
- 数据版本与冻结指纹：Valid 文件、固定参考集、对齐子集和 10,000 个业务键 SHA256 分别为 `1d59b7d0c97216f614c8ee4d4eb565adc6d8df6fa424399eca353dc279235c40`、`a2e0366d3d8f582e53a5293b687dc08f85b2960d60c7fb44d2fd02168063a944`、`5bfe2dac8d52ed336cac0ce1fb63c65afe3496805acc3146b953cbf0cc28a28b` 和 `28636f76b43586c9583bdbccf145194908fdbbff81cfa5ffb2dd383d332b9d50`，与无约束实验完全一致。复用的 `int32[2,337,178,3]` SID 全量 SHA256 为 `ff408b33c06cd7f30ee13eaf9723a3db60b0102dd2b1bdef02315e612992d75b`；checkpoint model/joint state SHA256 为 `f8eecbdb065b023e773b0b4ac67942d1fc0aa422516fddd38aa17f7cd6bb638c` / `01130bcbd0bf6ed2456bc06d4241b409959e2f9c2f9f31f9a6c062d2127ec406`。
- 代码与工作树：Git revision 为 `54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树未提交并保留既有并行修改。`JointLegalPathConstraint` 直接在压缩排序 Bucket key 上按 mixed-radix 范围查找合法下一层 code，并映射回动态 SID Token；评测入口新增显式 `--legal-path-constraint`，默认仍保持原无约束协议。32 条真实 smoke 的 320 个候选全部结构合法、可展开且每请求有 10 个唯一桶；该 smoke 只作为运行门禁，不单独编号。
- 约束协议：生成序列固定为 `<TARGET_POI> → S1 → S2 → S3 → </TARGET_POI>`。根节点只允许全目录实际使用的 S1；后续只允许当前合法前缀下真实存在的 S2/S3；完整三级路径必须命中 epoch-3 Bucket 后才允许闭合。约束发生在 Beam logits 上，不是生成后过滤。由于没有 `C`，本轮只报告 Bucket HR，不报告精确 POI HR/NDCG。

~~~bash
TMPDIR=/ofs/map_search/hudan/poi_genret/outputs/tmp/tjc3 \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=0 \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/evaluate_bucket_retrieval.py \
  --valid-file outputs/tiger_joint/data/EXP-20260827-01_sid_free_full_v1/valid.jsonl \
  --reference-validation-subset outputs/eval/qwen3_0.6b_main_v1_a100_e2/validation_subset_10000.jsonl \
  --checkpoint outputs/tiger_joint/train/EXP-20260827-04_joint_v1_4x6000d_e3/checkpoint-step-44454 \
  --embedding-dir outputs/embeddings/beijing_poi_bge_m3 \
  --output-dir outputs/eval/tiger_joint/EXP-20260830-01_epoch3_constrained_bucket_fixed10k \
  --reuse-sid-artifact-dir outputs/eval/tiger_joint/EXP-20260829-01_epoch3_bucket_fixed10k \
  --expected-step 44454 --expected-epoch 3 --num-beams 10 \
  --per-device-eval-batch-size 16 --sid-batch-size 8192 \
  --chunk-size 500 --cutoff-len 1024 --legal-path-constraint

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/tiger_joint/verify_bucket_retrieval.py \
  --eval-dir outputs/eval/tiger_joint/EXP-20260830-01_epoch3_constrained_bucket_fixed10k
~~~

- 环境与资源：NVIDIA RTX A6000，Python 3.10.20、torch 2.9.1+cu128、CUDA 12.8、NumPy 1.26.4；正式进程从北京时间 2026-08-30 14:29:14 运行至 14:51:32，退出码 0。纯生成耗时 1,169.08 秒，较无约束 1,072.14 秒增加 96.94 秒（约 9.04%）；峰值 CUDA allocated 为 13,609,360,896 bytes，合法前缀缓存最终包含 47,238 项。
- 核心结果：100,000 个候选的三级前缀合法率、完整 wrapper 合法率和目录可展开率均为 100%，不可展开率、重复可展开 Bucket 率均为 0；每条请求恰好得到 10 个唯一可展开 Bucket。raw-slot 与 unique Bucket 排名因此完全相同。

| 口径 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---:|---:|---:|---:|
| 联合 epoch 3 无约束唯一 Bucket | 22.84% | 32.90% | 34.77% | 34.97% |
| 联合 epoch 3 合法路径约束 | 22.99% | 35.40% | 40.00% | 44.64% |
| 约束 − 无约束 | +0.15pp | +2.50pp | +5.23pp | +9.67pp |
| TIGER epoch 3 无约束 Bucket 参考 | 55.53% | 79.64% | 85.10% | 88.06% |
| 约束联合 − TIGER 参考 | -32.54pp | -44.24pp | -45.10pp | -43.42pp |

- 对比边界：前两行是同一联合 checkpoint、同一 SID 和同一业务键，仅解码约束不同，可以直接归因于合法路径约束；TIGER 行仍是无约束解码，只作数值背景，不是约束协议完全匹配的比较。即便把约束视为对联合模型更有利的诊断，HR@10 仍只到 44.64%。
- 独立复核与产物：正式目录为 `outputs/eval/tiger_joint/EXP-20260830-01_epoch3_constrained_bucket_fixed10k/`，含 20 个各 500 行的候选 trace 分片。`result.json`、`verification.json`、候选 manifest SHA256 分别为 `9308b75704326c2f557c4345ed3d3abc45225edbfab604acd56fbf787f296eb7`、`10b59e122a112fb9a5ca48ced4640d7b6d8f1d38224223f21b6090c269330d26`、`c70a976a26174f2e7fad1d6a55b59627658acd8a3ae345f88929c8cadbb6f7e9`。独立脚本逐行验证 10,000 个目标、100,000 个原始 Token 候选、Beam rank、目录桶大小、去重顺序、分片哈希、约束模式和全部指标；运行日志 SHA256 为 `51d612165f3d8600b276759b8e802986c945cccc446e31d568d13cffe7d24254`，控制文件位于 `outputs/run_control/tiger_joint_exp_20260830_01_epoch3_constrained/`。
- 结论与下一步：合法路径约束完全修复了“生成合法层级 Token、却组合成目录外路径”的工程问题，也在较深 Top-K 上回收部分召回；但 Top-1 仅提高 0.15pp、Top-10 仍低 TIGER 参考 43.42pp，说明最终目录不一致不是唯一原因，Qwen 对正确终态 Bucket 的相对概率仍未学好。第一版联合训练的负结论保持不变，不继续评测 epoch 1/2、Test、泛化集或增加 `C`；若未来重新立项，必须先改变 hard-label 稳定机制，而不是只给当前 checkpoint 加 Trie。
