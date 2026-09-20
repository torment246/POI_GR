# QG-PRQK

QG-PRQK 是面向北京 active POI 闭集的三层 Semantic ID 方法。2026-09-19 已登记用户确认的 **v3.0：软粒度监督的 POI 搜索分布与共享编码双视图量化**。新方案尚未实现；当前可运行代码、命令和已完成 A0/A4 实验仍对应 v2.1-CAT。

| 版本 | 状态 | 阅读入口 |
|---|---|---|
| v3.0 | 方案已确认，未实现、未训练；下一步为数据契约与合成验证 | [主规范：差异、公式与实现步骤](docs/methods/QG_PRQK_METHOD_SPEC.md) |
| v2.1-CAT | 现有源码与冻结实验；A0/A4 全量 Test 已验收 | [旧规范归档](docs/methods/QG_PRQK_V2_1_CAT_SPEC.md)、[实验记录](docs/experiments/QG_PRQK.md) |

v3.0 数据流为：全部合法 Train Query–POI 关系 → 软粒度与可靠性 → 固定 Query 参照上的 POI 分层搜索分布 → 同一 POI assignment 下的内容/搜索双质心 → 同协议 Qwen SFT。首轮保持现有类别和地理算法，先做 Query 开/关配对；真实距离冲突项随后独立验证。完整字段、模块复用和验收顺序只在主规范维护，不存在可直接启动 v3.0 的 CLI/YAML。

以下介绍 **v2.1-CAT 当前实现**。它通过 Query–POI 图、类别软约束和局部地理残差构建 `S1 → S2 → S3`，使用：

- active POI：716,245 条；
- POI 表示：冻结 BGE-M3，1024 维；
- Query view：D1/D2 使用 Raw BGE，D3 使用最终 Query Adapter；
- 量化器：Projection-Residual RQ-KMeans；
- 码本容量：`512 × 512 × 512`；
- Final ID：基础 ID 冲突时才在末尾追加确定性 Dedup token；
- SFT 后端：Qwen3-0.6B + LLaMA-Factory，全参数训练；
- 数据边界：Query 监督和 SID 静态评测只读取 Train，不读取业务 Validation/Test。

两种进入 SFT 的目标为：

| 变体 | S3 parent | SFT 目标 |
|---|---|---|
| `a4_gid_parent` | `(GID6, S1, S2)` | `GID6 + S1 + S2 + S3 + [D]` |
| `a4_nogid` | `(S1, S2)` | `S1 + S2 + S3 + [D]` |

`[D]` 只在完整基础 ID 冲突时出现，并且永远放在最后。

2026-09-14 两个四卡 A100 SFT 及一台双卡 6000D 的十项 Validation 评测均已完成、退出码 0。GID-parent / NoGID 的 epoch-3 checkpoint 为 `8949 / 7404`；固定 10k 的 HR@1/HR@10/NDCG@10 为 `50.31/85.19/68.7252%` / `50.19/84.67/68.2651%`，四类泛化宏平均为 `22.6750/46.6900/34.2854%` / `21.8275/46.3325/33.5531%`。GID 整体略高，NoGID 冷目标更好。完整结果与来源见 [正式实验记录](docs/experiments/QG_PRQK.md#exp-20260914-01双分支-epoch-3-固定-10k-与四类泛化评测)，已完成的训练和 Validation 不需要重新运行。用户随后确认补充最后一天全量 Test；两版各 606,682 条正式推理现已完成，结果已核验，见下方“最后一天全量 Test 双卡评测”。

2026-09-18 已核实 **A0-GID（POI-only PRQ-KMeans）三轮 SFT 与五集双解码评测完成**：末轮 `checkpoint-8955`，Validation loss 为 `0.198839`。A0→A4 的约束固定 10k HR@10 为 87.06→87.34%，四类泛化宏平均为 53.7250→53.9075%。补充单卡五组各 100 条配对 Qwen 诊断已完成，正确前缀下 S1/S2/S3 同时 Top-1 正确率均为 43.4%；尚未出现新增组件的稳定整段预测收益。完整结果、范围与解释见 [A0/A4 诊断验收](docs/experiments/QG_PRQK.md#exp-20260918-01a0a4-qwen-配对逐层诊断验收)。

2026-09-19 已验收 A0-GID 全量 Test 双解码，两种模式各 606,682 条。约束 HR@1/HR@10/NDCG@10 为 `51.1357/87.2192/69.9686%`；A4−A0 仅 `+0.0125/−0.0157/+0.0183pp`，新增组件组合尚无清晰净收益。见 [EXP-20260918-02](docs/experiments/QG_PRQK.md#exp-20260918-02a0-gid-全量-test-双解码验收)。v3.0 的改进仍需重新 SFT 验证。

## v2.1-CAT 已实现的数据流

```text
Train orders
  └─> Query–POI statistics
       └─> D0/D1/D2/D3 query supervision
            ├─> D3 BGE -> Query Adapter selection -> adapted D3 view
            └─> D1/D2 Raw BGE
                    │
active POI BGE ──> POI-only base codebook
                    │
                    └─> Query graph + category soft constraints
                            └─> relational S1/S2 codebooks
                                    ├─> GID-parent local S3
                                    └─> no-GID local S3
                                            │
                                            └─> static SID evaluation
                                                    └─> unique Final IDs
                                                            └─> SFT data/vocab/cache
                                                                    └─> LLaMA-Factory SFT
```

### Query 监督含义

| 深度 | 含义 | Query view |
|---|---|---|
| D0 | 上下文型或无法稳定落到类别/POI | 不进入当前 SID 图监督 |
| D1 | Coarse-category Query | Raw BGE |
| D2 | Fine-category Query | Raw BGE |
| D3 | Exact-core Query | Final adapted BGE |

类别特征只作为 S1/S2 assignment 的软代价，不拼进 BGE，也不硬设 `S1 = category_id`。S3 不再使用 D1/D2 类别监督，只使用 D3 residual、固定 parent 内的局部地理信息和困难实体图。

## 源码结构

```text
qg_prqk/
├── configs/                  # 冻结实验协议与 SFT 配置
├── docs/                     # 方法规范、执行历史和实验记录
├── launchers/                # 训练平台脚本，本地忽略且不提交
├── outputs/                  # 数据、模型、日志和历史源码快照，不提交
├── scripts/
│   ├── qg_prqk.py            # 方法主链路的统一 CLI 入口
│   ├── visualize_sid_collisions.py  # 碰撞图薄入口，保留已冻结 V2 的 CLI 哈希
│   ├── analyze_sid_migration.py     # A0→A4 成员迁移薄入口，不改已有 CLI 哈希
│   └── a0_gid_sft.py                # A0-GID 平台准备/四卡训练薄入口
├── src/qg_prqk/
│   ├── adapters/             # Query Adapter 模型、检索、Gate 与全量选择
│   ├── commands/             # 统一 CLI 的参数解析与命令适配
│   ├── data/                 # 数据契约、Query 统计/监督/Embedding/图
│   ├── sft/                  # SFT 数据、词表、Cache 与 LLaMA-Factory 接入
│   ├── sid/                  # 基础、关系化、局部码本及评测/Final ID
│   ├── artifacts.py          # 原子写入、哈希和 manifest 公共逻辑
│   ├── cli.py                # 唯一命令路由
│   └── config.py             # 历史 v1.1 1024×3 兼容合同，非当前 SID 配置
└── tests/                    # 与上述领域同名的轻量测试
```

源码不从 `src/poi_gr` 导入项目实现。复用代码已经复制到本目录并在 `configs/code_provenance.yaml` 登记来源和哈希；PyTorch、FAISS、Transformers 与 LLaMA-Factory 仍作为第三方依赖，不重复复制源码。

### 当前 512×3 与历史 1024×3 配置

| 用途 | Python 入口 | YAML |
|---|---|---|
| 当前 SID 基础配置，供 A0/A4 构建配置继承 | `sid/pipeline_config.py::load_downstream_config`，`CURRENT_SID_CODEBOOK_SIZES=(512,512,512)` | `configs/qg_prqk_v2_1_category_active_512x3.yaml` |
| 历史 v1.1 数据准备与兼容校验 | `config.py::load_legacy_config`，`LEGACY_V1_CODEBOOK_SIZES=(1024,1024,1024)` | `configs/qg_prqk_1024x3.yaml` |

`config.py::load_config` 保留为历史加载器的兼容别名，不是自动选择当前方案的通用入口。历史上游配置、文件名和签名不修改；当前下游解析后明确覆盖码本为 512×3。BGE 的 **1024 维**和 SFT 的 **1024 cutoff**不是码本大小，不应替换成 512。

只读核对当前容量（不启动训练、不读取全量业务数据）：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python qg_prqk/scripts/qg_prqk.py inspect-sid-config \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml
```

预期 `codebook_sizes=[512,512,512]`、`poi_embedding_dim=1024`、`downstream_started=false`。

## 统一 CLI

查看所有命令：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py --help
```

查看某个命令的参数：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py build-query-graph --help
```

也可以显式设置源码路径后使用模块入口：

```bash
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -m qg_prqk --help
```

公共命令按职责分为：

- 数据与监督：`preflight`、`build-active-poi`、`build-query-stats`、`build-query-supervision`、`cache-query-embeddings`、`build-query-graph`；
- Query Adapter：`smoke-query-adapter`、`train-query-adapter`、`gate-query-adapter`、`select-query-adapter`；
- Semantic ID：`build-base-codebook`、`build-relational-codebook`、`build-local-codebook`、`build-local-codebook-nogid`、`evaluate-sid`、`build-final-identifiers`、`visualize-sid`；
- SFT：`build-sft-data`、`prepare-sft-vocab`、`prepare-sft-cache`、`train-sft`、`evaluate-sft`、`evaluate-sft-test`；
- 历史诊断：统一使用 `diagnose-*` 命令，不参与 canonical checkpoint 更新。

过去每个 `scripts/*.py` 再调用一个 `*_cli.py` 的双层入口已经移除。现在 `scripts/qg_prqk.py` 只负责把本地 `src` 加入导入路径，所有子命令由 `qg_prqk.cli` 路由；参数解析位于 `commands/`，算法实现位于对应领域包，二者职责不混合。

## SFT 训练入口（已完成，保留用于代码阅读）

以下为训练阶段原命令。现在两个固定输出目录已经非空，训练保护会拒绝重复运行（包括训练 dry-run）；当前应使用后面的 `evaluate-sft`。不要删除或覆盖训练输出：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py train-sft \
  --variant a4_gid_parent \
  --dry-run

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py train-sft \
  --variant a4_nogid \
  --dry-run
```

训练平台使用两份固定四卡 launcher：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a4_gid_parent_sft_4gpu_3epoch.sh
bash qg_prqk/launchers/run_train_qg_prqk_a4_nogid_sft_4gpu_3epoch.sh
```

launcher 最终调用统一的 `train-sft` 命令；实际模型训练、DDP、优化器、scheduler 和 checkpoint 保存由本地固定版本 LLaMA-Factory 完成。QG-PRQK 代码负责数据、词表、Cache、配置和启动前门禁。

## A0-GID：平台准备与四卡训练

在分配到四卡资源的训练平台节点上运行：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a0_gid_sft_4gpu_3epoch.sh
```

该脚本不要求 GPU 名称必须包含 A100 或 6000D；要求恰好四张可见 CUDA GPU、支持 BF16、每卡至少 28 GiB 可用显存。显存门禁是启动前检查，不保证任意型号性能或峰值显存均相同；不自行降低 batch 或修改算法。默认使用 `CUDA_VISIBLE_DEVICES=0,1,2,3`，平台已设置该变量时保留它。训练配置为 [a0_gid_history10_v1.yaml](configs/sft/a0_gid_history10_v1.yaml)。

```text
平台单进程 CPU：冻结 A0 SID + 相同 GID6 → 可选末位 Dedup → A0-GID 全目录
            → 逐行替换 A4 Train/Valid 的历史和目标 ID
            → 全量 1024 零截断检查 → LLaMA-Factory packed Cache
平台四卡 GPU：通过全部准备门禁 → Qwen3-0.6B 三轮 SFT
```

- 最终目标是 `GID6 + S1 + S2 + S3 + [D]`；A0 SID、BGE 均不重训，GID 使用与 A4 同一份确定性编码。Dedup 按新的完整 A0-GID 桶重新分配，不复用 A4 Dedup。
- 输入为同一 716,245 POI、7,586,410 Train 与 597,421 Valid；从冻结 A4-GID Messages 只替换历史/目标 identifier 及对应 identifier 元数据，Query、请求 GID、历史事件、顺序与业务主键保持不变。准备阶段不打开、不生成 Test；后续 A0 Test 需单独授权准备。
- 复用 A4 的同一份扩词初始模型，不从 A4 SFT checkpoint 续训、不重新随机扩词。历史 P8 静态指标显示 A0 的 `GID6+SID` 最大桶为 102，已有共同词表含 159 个 Dedup token；平台仍按实际新 mapping 再核对，缺词即停止。
- 训练后端仍是 LLaMA-Factory；与 A4 保持相同模型、seed=42、3 epoch、cutoff=1024、packing、BF16、global batch=512（4×8×16）和 LR=5e-5。identifier 长度和 packing 可能改变实际 step 数，不强行截到 A4 的 8949 step。
- 完整数据和缓存第一次在**训练节点 CPU/内存/磁盘**上生成，不是四个 GPU worker 各自构造；GPU 在准备期会等待。共享 OFS 的读写瓶颈不会仅因更换节点而消失，不保证这一阶段由 GPU 加速。
- 已完成阶段按来源/代码/配置合同、成功标记和文件大小/mtime 快速检查后复用；首次构建保存完整 SHA256，消息转换/Token 检查均核对全量文件哈希。正常重启不重复扫描全量数据或重算缓存。检测到变化会停止，不静默覆盖；失败的消息临时目录留存，未完成阶段需重做，**不是逐行断点续跑**。训练输出非空也会停止，不隐式覆盖或续训 checkpoint。
- 超过 1024、目标截断、词表不覆盖或来源不匹配时停止，不能靠截断/换词表继续训练。旧 A4 数据、模型、CLI、评测计划和图形源码均未修改。

仅核对入口，不构造、不训练（开发机可运行）：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a0_gid_sft_4gpu_3epoch.sh --dry-run
```

预期是 `status=planned_not_prepared`，它不等于全量缓存已经通过门禁。只在平台准备、不立即训练：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a0_gid_sft_4gpu_3epoch.sh --prepare-only
```

| 产物 | 路径（相对 `qg_prqk/`） |
|---|---|
| A0-GID 全目录 | `outputs/final_identifiers/a0_gid_order_a_v1/` |
| Train/Valid 与独立 dataset_info | `outputs/sft_data/a0_gid_order_a_history10_v1/` |
| packed Cache | `outputs/sft_tokenized/a0_gid_history10_v1/` |
| 三轮训练输出 | `outputs/sft/a0_gid_history10_v1_gpu4_e3/` |
| 分阶段日志/退出码/长度门禁/缓存凭证 | `outputs/run_control/sft_a0_gid_history10_v1_gpu4_e3/` |

平台主日志会先显示 `[1/3]`、`[2/3]`、`[3/3]`，数据阶段每 100,000 行报告进度。出现 `A0_GID_PREPARATION_COMPLETED` 后才会启动四卡 SFT。流水线配置为 `configs/sft/a0_gid_pipeline_v1.yaml`；源码入口为 `scripts/a0_gid_sft.py`，数据与调度职责分别在 `sft/a0_gid_data.py` 和 `sft/a0_gid_pipeline.py`，复用已有去重、格式化、长度检查、缓存和训练后端。

## PRQ-KMeans 对照：单卡 A100 约束与无约束评测

在一张 A100 平台卡上运行：

```bash
bash qg_prqk/launchers/run_evaluate_prqk_a0_gid_epoch3_dual_decode_1xa100.sh
```

输入固定为 `a0_gid_history10_v1_gpu4_e3/checkpoint-8955`、对应 Train/Valid 准备产物和 `a0_gid_order_a_v1` 全目录标识。标识是 `GID6 + S1/S2/S3 + [D]`，POI-only SID 及末位去重均不改变。复用 QG-HRQ 的固定 10k、已见 Query/未见 Query–POI、新 Query/已见目标、长尾目标、冷目标五份冻结业务主键；不重新抽样，不读取 Test。

脚本在平台上自动完成 CPU 缓存与全目录核验，先运行每项 2 条 GPU smoke，再运行 10 项正式评测。同一张 A100 先由无约束 worker 顺序完成五组，再由全目录约束 worker 顺序完成五组；每个模式只加载一次模型。两者都沿用 Beam=10、cutoff=1024、max_new_tokens=13、length_penalty=1、renormalize_logits=true；无约束非法和重复候选仍占原排名。平台主日志实时转发当前模式进度，500 条一个原子断点。

- `--dry-run`：只检查小文件、来源 manifest 和 checkpoint 完成状态，不扫描全量、不加载 GPU。
- `--prepare-only`：准备并核验五组缓存、权重哈希、完整标识与样例 token，不生成候选。
- `--smoke-limit 8`：两种模式五组各 8 条，写独立 smoke 目录。
- 同一命令重启复用匹配的缓存和已完成 chunk；来源、参数或代码变化则拒绝混合结果。

输出根目录：`outputs/eval/a0_gid_epoch3_dual_decode_v1/`。`plan.json` 绑定来源；`data/` 是共享五组缓存；`{unconstrained,constrained}/results/<subset>/result.json` 为十项结果；`results/summary.json` 和 `summary.csv` 汇总 HR/NDCG@1/3/5/10、MRR@10、合法 ID 率与四类泛化宏平均（不含固定 10k，数值为 0–1）。日志在 `outputs/run_control/eval_a0_gid_dual_decode_v1/`。原双卡 6000D 启动器保留为兼容入口，但当前推荐使用上述单卡 A100 入口；二者共享同一评测协议和结果目录，不应同时启动。本次交付只运行本地静态预检与合成测试，完整 CPU/GPU 门禁在平台启动时执行。

实现：`src/qg_prqk/sft/a0_evaluation.py` 处理 A0 数据格式适配、调度与断点，直接复用既有 `evaluation.py` 的模板/生成/计分和 `constrained_decoding.py` 的前缀约束。没有修改旧 A4 评测代码或结果。

## 两个 SFT 的双卡联合评测

新增约束解码入口见下方“全目录约束解码双卡评测”；本节原命令保留为无约束协议，不覆盖其已完成结果。

在一台双卡 6000D 节点上只启动这一份脚本。两个分支各五个集合，共十项评测交错进入同一队列；任意时刻最多两个单卡 worker，空闲 GPU 自动领取下一项：

```bash
bash qg_prqk/launchers/run_evaluate_qg_prqk_a4_dual_epoch3_fixed10k_generalization_2x6000d.sh
```

每个分支固定评测 epoch 3 的五个集合：随机 10k、已见 Query 未见配对、未见 Query 已见目标、长尾目标、冷目标；后四组也各 10k。配置位于 `configs/sft/evaluation_epoch3_fixed10k_generalization_v1.yaml`，固定无约束 Beam=10、cutoff=1024、batch=32；OOM 时只降低 batch，不改变样本或候选协议。完整目标 ID 的长短不同，因此 GID-parent / NoGID 的最大生成预算分别为 13 / 7，均包含包装、可选末位 D 和 EOS。

加 `--dry-run` 可同时核验两个分支的五组对齐缓存、checkpoint、完整 POI 映射和每组前 8 条格式化样例，不启动 GPU 推理；直接复用已准备的数据，不因四卡改双卡重扫原始 Validation。加 `--smoke-limit 8` 则在双卡上先评测每个分支每组 8 条，结果写到独立 `smoke/`。不加参数才是两个分支共十项完整评测。CLI 对应 `evaluate-sft --variant both`；如需单独补跑，也可指定 `a4_gid_parent` 或 `a4_nogid`，仍共用两张 GPU。

两个旧四卡评测入口已从 `launchers/` 清理，恢复备份在 `outputs/run_control/eval_launcher_archive_4x6000d/`（相对 `qg_prqk/`，不会被新脚本调用）。四卡 SFT 训练脚本保留不变。

输出统一在 `qg_prqk/outputs/eval/sft_epoch3_fixed10k_generalization_v1/<variant>/`：

- `data/`：五个方法专属子集和来源 manifest；按原 `(order_id, searchid)` 顺序对齐，不重新抽样，不读取 Test。
- `plan_2x6000d.json`：双卡计划、模型权重 SHA256、checkpoint 元数据、代码/数据来源与预检结果；首版四卡 `plan.json` 只读保留，不覆盖。
- `results/<subset>/progress.json`、`result.json`：每 500 条原子保存断点；同一命令重启自动续跑，输入/协议变化则拒绝混用结果。
- `results/summary.json`、`summary.csv`：五组 HR@1/3/5/10、NDCG、MRR@10、合法 ID 比例，以及四类泛化宏平均（不混入随机 10k）。

总日志/PID/退出码在 `outputs/run_control/eval_a4_dual_epoch3_2x6000d/`，各项日志在其 `<variant>/` 子目录（相对 `qg_prqk/`）。非法或重复候选保留原 beam 排名，不去重前移、不做地理剪枝、不做桶展开重排；最终指标是完整 Final ID 对应的精确 POI 命中。单卡 batch/显存门禁不变；只是总并发从四个改为两个。

## 全目录约束解码双卡评测

只改变 epoch-3 的解码方式，比较 GID-parent 与 NoGID 在既有固定随机 10k、已见 Query/未见 Query–POI、新 Query/已见目标、长尾目标、冷目标五组上的结果。两版共十项、每项 10,000 条；不重新抽样、不读取 Test、不重训或选模。

在同一台双卡 6000D 平台启动：

```bash
bash qg_prqk/launchers/run_evaluate_qg_prqk_a4_dual_epoch3_constrained_fixed10k_generalization_2x6000d.sh
```

- 数据直接复用原无约束评测的十份 JSONL，按冻结 plan SHA256 绑定原 checkpoint、样本、词表、Final ID 和其他生成参数。CPU 预检校验全目录、十份 JSONL 哈希、业务键唯一/顺序/互斥、两版目标和 CURRENT 一致；每组先检查 8 条完整 Token，剩余记录在各生成 chunk 内检查，不截断、不丢行。
- 合法路径覆盖两版各 716,245 个 POI 的完整 `<TARGET_POI> + Final ID + </TARGET_POI> + EOS`，包含可选末位 D；用排序终态路径与有界前缀缓存实现 Trie 等价查询，不保存大量逐节点 Python 对象。候选不按 Query、当前 GID、热度或 Train 覆盖剪枝，冷目标仍可生成。
- 保持无约束评测的 Beam=10、batch=32、cutoff=1024、length_penalty=1、early_stopping=true、renormalize_logits=true，GID/NoGID 最大生成长度仍为 13/7。添加前缀 mask 后，保留的 `renormalize_logits=true` 会对合法后继重新归一化，因此此实验衡量的是约束搜索整体效果，不是给旧结果简单过滤非法 ID。
- GPU0/GPU1 各负责一个分支，先自动运行两版固定集各 2 条 GPU smoke；合法且完整才进入十项正式评测。每 500 条进度同时写 worker 日志与平台主日志。约束下仍出现非法候选则停止；OOM 沿用旧逻辑降低 batch 并重试尚未入账 chunk，不修改训练或评分协议。
- 默认保留候选原排名和重复项，不做桶展开、后处理重排或去掉非法项再前移。输出 Recall@1/3/5/10、NDCG、MRR@10、合法 ID 率；四类泛化宏平均不包含固定 10k。完整汇总同时给出每版每组相对旧无约束结果的差值；smoke 不与全量无约束结果作差。
- `--dry-run` 仅做 CPU 预检；`--smoke-limit 8` 单独跑十组各 8 条。重新使用原命令可复用相同来源的完成结果与 chunk 断点；不同配置、来源或源码拒绝混用，不覆盖旧无约束产物。

输出根目录为 `qg_prqk/outputs/eval/sft_epoch3_constrained_fixed10k_generalization_v1/`：各分支 `plan.json`，以及 `results/<subset>/{progress,result}.json`；两版十组合并结果与无约束差值为根目录 `summary.json`，GPU smoke 单独位于 `smoke2/`。日志和退出码在 `qg_prqk/outputs/run_control/eval_a4_dual_epoch3_constrained_2x6000d/`，其中每个 worker 的退出码为 `full/<variant>.console.exit`。

代码入口 `scripts/evaluate_sft_constrained.py` → `sft/constrained_evaluation.py`，前缀约束在 `sft/constrained_decoding.py`；旧 `evaluation.py` 的模板、生成循环、精确 POI 解析与评分直接复用且不修改。已通过真实 CPU 预检、31 项相关合成测试（含小型随机 Qwen 的真实 Beam=10、两种长度和左 padding）、Ruff 与 shell 语法检查。当前只连接到开发机单卡 A6000，不能代替双卡 6000D 平台；正式 GPU 评测未启动，尚无约束解码业务指标。

## 最后一天全量 Test 双卡评测

用户已确认下一项评测为 2026-07-14 全部 606,682 条 Test，不再抽取 10k。使用同一台双卡 6000D，GPU 0 跑 GID-parent `checkpoint-8949`，GPU 1 跑 NoGID `checkpoint-7404`：

```bash
bash qg_prqk/launchers/run_evaluate_qg_prqk_a4_dual_epoch3_full_test_2x6000d.sh
```

加 `--dry-run`（或 `--preflight-only`）执行 CPU 预检，不挂载或使用 GPU；加 `--smoke-limit 8` 则每版只生成前 8 条，结果写入独立 smoke 目录。CLI 为 `evaluate-sft-test`，配置为 `configs/sft/evaluation_epoch3_full_test_20260714_v1.yaml`。

- 输入直接读取两版 `outputs/sft_data/*/test.jsonl`。通过冻结 manifest 确认 Test 日期（Messages 本身没有日期列），全量核验两文件 SHA256、606,682 行、唯一且同序的业务键、目标 POI、当前 Query/GID 和历史长度；不复制原 JSONL。worker 仅保存约 4.63 MiB 的行偏移表，逐 chunk 加载 Messages。
- 模型 SHA256、词表、Final ID、BF16、cutoff=1024、batch=32、Beam=10 和生成参数继承已完成的 Validation 协议；完整 ID 精确匹配，末位 D 必须正确，非法/重复候选不前移，不启用 Trie，不选择 checkpoint。没有调用或修改 SID/SFT 训练链路。
- CPU 预检检查每版前 100 条的精确 Token/目标映射，**不等于完成所有 Test 的 Token 长度扫描**；其余行在实际生成前逐 chunk 检查。超长或标签不一致立即报错，不截断、不丢行，也不根据 Test 改数据规则。
- 每 500 条保存 `progress.json`，最后不足 500 条仍完整计入；相同命令重启从已完成 chunk 续跑，完成的分支跳过。OOM 只减小 batch，重试未计入的 chunk；来源变化拒绝续跑。
- worker 的每块进度、速度和剩余推理时间同时写入独立日志与平台主日志；父进程每 30 秒补充存活提示，初始化期间不把零样本进度误报为失败。

输出全部位于 `qg_prqk/outputs/eval/sft_epoch3_full_test_20260714_v1/`：

- `test_data_manifest.json`、`<variant>/plan.json`：全量输入与模型/源码指纹；
- `<variant>/results/{progress,result}.json`：两版各自的断点和最终指标；
- `summary.json`、`summary.csv`：两版全量 HR@1/3/5/10、NDCG@3/5/10、MRR@10、合法 ID 率及 GID−NoGID 差值，不混入 Validation 或计算五组宏平均；
- `<variant>/smoke/<N>/` 和 `smoke/<N>/`：独立 smoke 结果，不冒充全量。

总日志、PID、退出码在 `outputs/run_control/eval_a4_dual_epoch3_full_test_2x6000d/`。原 Validation 入口、模型、源码指纹、plan 和结果保持不变；本次交付脚本，不自动运行正式 Test。

2026-09-14 交接时完成 CPU 预检；用户随后已跑完正式 Test，两版各 606,682 条、退出码均为 0。GID/NoGID 的 HR@1/HR@10/NDCG@10 分别为 `50.9631/85.5120/69.1641%` / `50.1919/85.1705/68.5076%`，合法 ID 率为 `64.8803%/69.6188%`。来源签名、计数与排名直方图复算已通过，结果见该输出目录的 `summary.json`；不需要重跑。本轮约束解码只使用前述五组 Validation，不复用 Test 选模。

## 三版 SID 的类别与区域可视化

参考 GenPOI 图 4/5 的展示方式，另存一版共同五类 t-SNE 与具体前缀分布，不覆盖旧版热门 code/residual 图。

```bash
export TMPDIR="$PWD/qg_prqk/outputs/tmp/sv2"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib:${LD_LIBRARY_PATH:-}
/ofs/map_search/hudan/envs/poi-gr/bin/python qg_prqk/scripts/qg_prqk.py visualize-sid-categories \
  --config qg_prqk/configs/qg_prqk_sid_category_region_v2.yaml --validate-only
```

- 输入：冻结的三版 716,245 行 SID、512×3 POI 码本与 POI metadata；不读取业务 Validation/Test，也不重算 BGE。
- t-SNE：全库数量最多的同五个真实粗类别，各取固定 200 个 POI。三版使用相同样本、颜色、联合 PCA50+t-SNE 坐标；输入为三层选中 POI 码向量分别归一化后等权串联，不把整数 SID、类别或经纬度当作投影特征。
- 前缀：每类固定抽样的第一个 POI 为锚点，展示它在每版的 `[S1,*,*] → [S1,S2,*] → [S1,S2,S3]`。类别与 Geohash5 网格条统计对应完整桶；不只保留五类，不加 GID 条件，区域不是行政区。两种 A4 的前两层应相同。
- 输出：[同类 t-SNE](outputs/figures/qg_prqk_sid_category_region_v2/sid_category_tsne.png)、[前缀类别/区域图](outputs/figures/qg_prqk_sid_category_region_v2/sid_prefix_category_region.png)，同目录 PDF 可放大查看；`sample_and_coordinates.parquet` 保存全部抽样 ID/SID/坐标，`metrics.json` 保存桶内完整计数与全库指标，`manifest.json` 保存来源和代码哈希。
- 上述命令只复核已有产物，预期返回 `status=validated`。首次生成去掉 `--validate-only`；`--dry-run` 只做输入预检。输出目录已存在时拒绝覆盖；重新生成须复制配置并使用新的 QG 输出目录。

字体资产已复制到 `outputs/inputs/fonts/NotoSansCJKsc-Regular.otf`，运行时不依赖其他方法的源码或资源目录。此图是 GenPOI 风格适配，不是原文 GeoPE embedding 的严格复现；颜色聚集或单例桶 100% 不能单独证明检索提升。实现与实验解释见 [代码说明](docs/QG_PRQK_CODE_GUIDE.md) 和 [方法记录](docs/experiments/QG_PRQK.md)。

### 碰撞案例补充：末层不再使用单例

查看 [三版碰撞前缀图](outputs/figures/qg_prqk_sid_collision_examples_v1/sid_collision_prefix_category_region.png)、[可放大 PDF](outputs/figures/qg_prqk_sid_collision_examples_v1/sid_collision_prefix_category_region.pdf) 和 [全部碰撞 POI 明细](outputs/figures/qg_prqk_sid_collision_examples_v1/collision_members.html)。

本版先筛选在三种方法中完整 SID 桶大小都为 3–20 的 POI，再从上述五类中各以 seed=42 抽取一个共同锚点，不按区域/类别纯度挑选。15 个末层案例全是碰撞桶；图中逐层统计完整桶，HTML 展示末层全部成员的 POI ID、名称、完整类别、Geohash5、经纬度。同网格不等于同地点；这些是碰撞条件案例，不能代表全库效果。

沿用上面的 `poi-gr`、TMPDIR、字体缓存和线程环境：

```bash
python qg_prqk/scripts/visualize_sid_collisions.py --validate-only
```

首次构建去掉 `--validate-only`；输入默认引用已有 V2 配置，输出默认写入上述新的碰撞目录，拒绝覆盖。不重算 t-SNE、不更新 SID；选例参数与来源哈希在该目录的 `manifest.json`，完整桶计数在 `collision_examples.json`。此入口仅薄转发到 `sid/collision_visualization.py`，不修改已冻结 V2 的 CLI/绘图源码，以保持旧图严格哈希核验可用。

### A0→A4：前缀改变与桶成员迁移

查看 [全库变化概览](outputs/figures/qg_prqk_sid_migration_v1/sid_migration_overview.png)、[GID-parent 迁移案例](outputs/figures/qg_prqk_sid_migration_v1/sid_migration_gid_examples.png)、[NoGID 迁移案例](outputs/figures/qg_prqk_sid_migration_v1/sid_migration_nogid_examples.png) 和 [保留/移出/移入的成员明细](outputs/figures/qg_prqk_sid_migration_v1/migration_examples.html)。同目录有可放大的 PDF。

全库 716,245 POI 逐层比较编码与实际同伴集合，另列完整 SID 的“原碰撞→单例”和“原单例→碰撞”。案例按**排除自身后的同类别成员比例**提高、持平、降低分组，固定 seed=42 抽取，不取最大增益；S1 使用粗类别，后两层使用完整细类别。区域 Geohash5 单独展示，不合成一个总分。两侧都是非单例时才比较语义比例，不把单例当成 100%。

沿用上面的环境，复核命令为：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python qg_prqk/scripts/analyze_sid_migration.py --validate-only
```

首次生成去掉 `--validate-only`；重新生成必须用 `--output-dir qg_prqk/outputs/figures/<新目录>`，已有目录拒绝覆盖。输入仍为冻结 V2 来源；输出 `metrics.json` 是全库统计，`examples.json` 保存案例与每组至多 5 个成员预览，`memberships.npz` 保存案例全部成员行号和 POI ID，manifest 绑定参数、来源与产物哈希。成员变化表示至少一个同伴发生变化，不等于整桶重建；码号变化也不等于同伴变化。案例只能解释局部现象，不代表总体提升或 Query 监督的独立收益。

实现分为 `sid/migration_analysis.py` 的统计/选例/发布与 `sid/migration_plots.py` 的绘图/HTML；薄入口不改已有 CLI。没有重训、重算 t-SNE、更新 SID 或读取业务 Validation/Test。

## 历史阶段编号

阶段编号只用于解释已经冻结的配置、manifest、输出目录和实验记录。当前源码请按领域目录和语义化命令阅读。

| 历史编号 | 实际职责 | 当前源码 |
|---|---|---|
| P2 | Query–POI 统计 | `data/query_statistics.py` |
| P2.5-CAT | D0/D1/D2/D3 与类别边 | `data/query_supervision.py` |
| P3A | Query Adapter Gate/全量选择 | `adapters/` |
| P4 | Query 图与分层 view | `data/query_graph*.py` |
| P5 / A0 | POI-only 基础码本 | `sid/base_*.py` |
| P6 | Query/类别关系化 S1/S2 | `sid/relational_*.py` |
| P7 / A4 | 局部 S3 | `sid/local_*.py`、`sid/nogid_*.py` |
| P8 | 静态比较与 SID 发布 | `sid/evaluation*.py` |

冻结 YAML 文件名暂时保留阶段编号，因为名称、内容哈希和互相引用已经写入历史 manifest。源码整理不伪造新配置，也不改写旧实验依据。

## 核验

轻量回归：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib:${LD_LIBRARY_PATH} \
MPLCONFIGDIR=qg_prqk/outputs/cache/matplotlib \
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  -m unittest discover -s qg_prqk/tests -v
```

语法检查：

```bash
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  -m compileall -q qg_prqk/src qg_prqk/scripts qg_prqk/tests
```

以上检查不读取业务 Validation/Test，不启动 GPU 训练，也不会覆盖既有产物。
当前开发机需要让 `poi-gr/lib` 位于 `LD_LIBRARY_PATH` 最前，以免 FAISS 误载系统旧版 `libstdc++`；正常打包环境只需保证 FAISS 与 C++ runtime 兼容。

## A0-GID 全量 Test 双卡评测

使用 active 716,245 库训练的 A0-GID epoch 3 `checkpoint-8955`，覆盖与四模型评测相同的 **2026-07-14 全量 Test 606,682 条**。按本次用户指定，启动器放在 `qg_prqk/launchers/`。在仓库根目录执行：

```bash
bash qg_prqk/launchers/run_evaluate_prqk_a0_gid_epoch3_full_test_dual_decode_2x6000d.sh --dry-run
bash qg_prqk/launchers/run_evaluate_prqk_a0_gid_epoch3_full_test_dual_decode_2x6000d.sh
```

需要两张空闲 RTX PRO 6000D，每卡至少 36 GiB 可用显存。第一张卡运行无约束，第二张卡运行完整 active 目录 Trie 约束；两种模式均使用 Beam=10、cutoff=1024、max_new_tokens=13，初始 batch=32，不增加地理剪枝。评测直接复用既有 Test 编码和候选评分；无约束非法候选保留原排名，约束模式沿用合法后继重新归一化。

A0 原 SFT 产物只有 Train/Validation，因此读取冻结 A4-GID Test，将目标和历史 ID 经 POI 行号映射到 A0 ID，保留业务键、Query、位置、历史顺序和 Test 划分。准备阶段核验完整模型哈希、两份目录及全量源文件，对所有样本验证目标和历史映射；不另存一份 Test JSONL。前 100 条做格式化 Token 检查，其余逐 chunk 使用原编码器检查。

默认先完成两种模式各 2 条 GPU smoke，通过后各跑全量。每 500 条保存断点，OOM 降 batch 后重试尚未入账的 chunk；再次执行同一命令可续跑。`--prepare-only` 只做 CPU 全量准备，`--smoke-limit N` 只生成每模式前 N 条（1—100）。准备、smoke 和正式评测都有来源指纹及独立结果，来源不一致会拒绝复用。

输出根目录为 `qg_prqk/outputs/eval/a0_gid_epoch3_full_test_20260714_dual_decode_v1/`：`plan.json` 保存来源，`{unconstrained,constrained}/results/` 保存各自结果和断点，`results/summary.json` / `summary.csv` 汇总 HR/NDCG@1/3/5/10、MRR@10、合法 ID 率及约束增量。smoke 使用对应 `smokeN/` 目录。日志、PID 和退出码在 `qg_prqk/outputs/run_control/eval_a0_gid_full_test_dual_decode_v1/`。本次只交付入口及轻量核验，正式全量推理待平台运行。

## 文档

- 方法与代码总览：`docs/QG_PRQK_CODE_GUIDE.md`；
- 新版本方法定义：[v3.0 主规范](docs/methods/QG_PRQK_METHOD_SPEC.md)；
- 当前代码对应的历史方法：[v2.1-CAT 规范归档](docs/methods/QG_PRQK_V2_1_CAT_SPEC.md)；
- 历史执行计划：`docs/methods/QG_PRQK_CODEX_EXECUTION_PLAN.md`；
- 当前实施状态：`docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md`；
- 正式实验记录：`docs/experiments/QG_PRQK.md`；
- 运行命令与历史恢复：`docs/QG_PRQK_CODEX_COMMANDS.md`。

历史复核命令若指向 `outputs/run_control/**/source_snapshot/`，必须继续使用对应冻结快照；不要用当前源码冒充旧实验源码。

## 四模型全量 Test 约束对照入口

2026-09-17 按用户确认，使用 2026-07-14 全量 606,682 条 Test。平台入口在仓库根目录：

```bash
bash launchers/run_evaluate_active_four_models_full_test_4x6000d.sh --dry-run
bash launchers/run_evaluate_active_four_models_full_test_4x6000d.sh
```

四张卡分别运行 GNPR 无约束、GenPOI Query-only SSP（γ=2）+TCG、QG-HRQ(GID) 与 QG-HRQ(NoGID) 全目录约束。四个 checkpoint 均来自 active 716,245 库的已完成 epoch 3；QG 两版直接复用原 full Test 编码和评分，仅注入既有 Final ID 前缀约束，保留 Beam=10、cutoff=1024、最大生成长度 13/7 与合法后继重新归一化，不增加 SSP 地理剪枝。

启动后先在 CPU 逐行核对四份 Test 的业务键、目标、CURRENT 和历史长度，并核验全文件与 checkpoint 哈希；四卡各 2 条 GPU smoke 全部通过后才跑全量。每 500 条保存断点，OOM 降 batch 后重试；再次执行同一命令可续跑同来源断点。`--prepare-only` 仅做 CPU 全量预检，`--smoke-limit N` 只生成 N 条评测结果（1—100），但仍准备全量输入和 GenPOI SSP 预测。

新输出独立放在根 `outputs/eval/active_four_models_full_test_20260714_v1/`：`plan.json` 绑定来源，`<method>/full/` 保存每模型结果，`<method>/smoke2/` 保存 smoke，根 `summary.json` / `summary.csv` 汇总四模型 HR/NDCG@1/3/5/10 和合法 ID 率（0—1）。日志、PID 和退出码在根 `outputs/run_control/active_four_models_full_test_20260714_v1/`。只有四项都完成且各覆盖 606,682 条才生成正式汇总。本次交付代码与本地轻量核验，正式 GPU 评测尚未运行；旧 Test 无约束结果保留。

## A0/A4 可预测性原因核验

`EXP-20260917-02` 已完成五组 50k 评测输入及 232,409 个历史事件标识回映射、全量 342,879 Query 的 S1 内容排名和冻结图分配公式复算。入口 `scripts/diagnose_a0_a4.py --output <qg_prqk/outputs 下未使用的 JSON 路径>`，结果为 `outputs/eval/a0_a4_diagnostic_v1/result_verified.json`；CPU 分块执行，不加载 Qwen、不读取 Test。A0/A4 S1 内容 Top-10 为 57.4874%/47.5105%，A4 图公式分配复现率 99.9977%，明确存在图对齐与内容可预测性差距。配对核验无错位；尚未把静态结论外推为 Qwen 逐层退化。详见[诊断实验](docs/experiments/QG_PRQK.md#exp-20260917-02a0a4-配对输入与全量-s1-内容图分配诊断)。

## A0/A4 Qwen 逐层诊断（单卡平台）

为区分“静态 Query 最近码变差”与“Qwen SID 生成变差”，使用两个已冻结 epoch-3 GID 模型和五组固定 Validation 业务键。每组按业务键哈希确定性选 100 条，共 500 条；选样不使用模型结果或标识符，两版逐行配对，不读取 Test。该数量用于定位问题，不作为新的全量排名或显著性结论。

在仓库根目录执行：

```bash
bash launchers/run_diagnose_qg_a0_a4_qwen_prefix_1gpu.sh --dry-run
bash launchers/run_diagnose_qg_a0_a4_qwen_prefix_1gpu.sh
```

单卡依次运行 A0、A4，每个模型检查两部分：

- 正确前缀评分：使用实际 Query/GID/历史 prompt，给定正确目标前缀，分别计算 S1/S2/S3、可选 D 和闭合 token 的全词表排名/NLL、合法后继条件排名/NLL、合法概率质量、候选数和 singleton 比例。S1 已给定正确 GID；这不是独立自由生成准确率。不同 target 长度采用左 padding 的正确位置编号，并只计算目标位置 logits。
- 自由生成：直接复用原 evaluator 的 Beam=10、max_new_tokens=13、cutoff=1024、renormalize_logits=true 和精确候选排序；分别运行无约束、完整目录约束。记录 Top-1/10 的 GID6、GID6+S1、GID6+S1/S2、GID6+SID3 和最终 ID 命中，以及目标前缀首次丢失的分段。这里是 Beam Search 的前缀覆盖，不是逐 token greedy；Top-10 首次丢失表示十个最终候选中不再有正确前缀。

启动器继承原四模型入口的平台初始化与认证，改为一张 6000D/RTX PRO 6000/A100/A6000，模型串行，初始 batch=8、OOM 减半重试未入账 chunk。先 CPU 核验权重完整哈希、50k 来源/配对、全目录和全部选中样本 Token，再两版五组各 2 条 GPU smoke，通过才执行每组 100 条。`--prepare-only` 仅 CPU 准备；`--smoke-limit N` 仅运行每组 N 条（1—100），同命令可恢复来源一致的断点。

输出根目录 `qg_prqk/outputs/eval/a0_a4_qwen_prefix_diagnostic_v1/`：`sample2/` 为 smoke，`sample100/` 为默认诊断，各含冻结 `plan.json`、两份样本缓存、`{a0,a4}_progress.json`、`{a0,a4}_result.json` 和 `summary.json`。结果保存业务键哈希、目标 token、逐层评分、十条候选及分数，便于配对追查；完整文本只在忽略的输入缓存中。日志/PID/退出码位于 `qg_prqk/outputs/run_control/a0_a4_qwen_prefix_diagnostic_v1/`。入口为 `scripts/diagnose_qwen_prefix.py` → `sft/prefix_diagnostic.py`，旧评测源码和结果不改。正式模型 GPU 诊断待用户在空闲平台启动，不能把随机小模型测试当成业务结果。
