# QG-PRQK 实现状态

## 当前状态（2026-09-19）

```yaml
PROJECT: QG-PRQK
METHOD_VERSION: v3.0
DESIGN_METHOD_VERSION: v3.0
IMPLEMENTED_METHOD_VERSION: v2.1-CAT
STATUS: DESIGN_APPROVED_NOT_IMPLEMENTED
CURRENT_PHASE: V3_DESIGN_REVIEW_AND_CODE_PLAN_COMPLETED
LAST_COMPLETED_PHASE: V3_FORMULA_REVIEW_AND_REUSE_MAPPING
NEXT_ACTION: 实施 V3-DATA 的全量 Query/Pair 与路由契约、行映射校验和合成测试；随后按依赖实现搜索分布和量化核心，真实标签准备不阻塞核心代码验证。
PRIMARY_CODEBOOK_SIZES: [512, 512, 512]
V3_DESIGN_REVIEW: COMPLETED_WITH_FORMULA_CORRECTIONS
V3_SYNTHETIC_MATH_REVIEW: PASSED_10_CASES_NOT_IMPLEMENTATION_TESTS
ITERATION_POLICY: VALIDATION_DRIVEN_REUSE_FIRST_REFACTOR_AFTER_METHOD_STABLE
V3_DATA_CONTRACT: NOT_IMPLEMENTED
V3_GRANULARITY_LABELS_AND_MODEL: NOT_STARTED
V3_FULL_QUERY_PROFILES: NOT_STARTED
V3_SHARED_ASSIGNMENT_QUANTIZER: NOT_IMPLEMENTED
V3_SFT_AND_EVALUATION: NOT_STARTED
V3_DISTANCE_GEO: DEFERRED_UNTIL_QUERY_COMPARISON
V2_1_A4_DUAL_SFT: COMPLETED
V2_1_A0_GID_SFT: COMPLETED_EPOCH3_STEP8955
V2_1_A0_A4_QWEN_DIAGNOSTIC: COMPLETED_PAIRED_500
V2_1_A0_GID_FULL_TEST: COMPLETED_BOTH_606682_VALIDATED
LAST_UPDATED: 2026-09-19 CST
WORKTREE_STATE: DIRTY_USER_WORKTREE_PRESERVED
```

- 用户已确认新方法方向，完成文档登记后又要求复核方案、规划复用代码并按结果迭代。[v3.0 主规范](../methods/QG_PRQK_METHOD_SPEC.md)定义全量软粒度、POI 搜索分布、共享编码双质心和后续局部距离约束；[执行计划](../methods/QG_PRQK_CODEX_EXECUTION_PLAN.md)已增加具体开发切片与结果驱动的调整规则。[v2.1-CAT 原规范](../methods/QG_PRQK_V2_1_CAT_SPEC.md)完整归档。当前 Python、YAML、产物和 checkpoint 未修改，现有命令不执行 v3.0。
- A0/A4 Qwen 逐层诊断见 [EXP-20260918-01](QG_PRQK.md#exp-20260918-01a0a4-qwen-配对逐层诊断验收)；A0 全量 Test 双解码见 [EXP-20260918-02](QG_PRQK.md#exp-20260918-02a0-gid-全量-test-双解码验收)。A4−A0 约束 HR@1/HR@10/NDCG@10 为 `+0.0125/−0.0157/+0.0183pp`，尚无清晰组件组合净收益。
- 设计复核修正了深度不确定性与质量混用、分层后截断导致深层权重饱和、零条件残差仍偏好某些 code 的问题；10 项一次性 CPU 合成数学检查通过，生产实现测试仍未开始。新粒度标签不能由旧类别集中度直接推出，真实人工标签仍需准备。
- 首轮候选比较 Query 开/关，共享类别与旧地理项，两边统一全 512 候选；旧 S3 内容 Top-32 不直接复用。后续再隔离真实距离项。粒度模型、搜索分布、量化器均需实现和验收，不能把新公式写成已运行结论。
- 本次只更新设计/执行文档并运行轻量数学检查，不分配 EXP 编号，不启动训练或全量处理。相关工作和设计全文仅在方法主规范维护，以下历史流水保留原样。

## 2026-09-15 状态快照（历史，不作为当前执行值）

```yaml
PROJECT: QG-PRQK
STATUS: HOLD_FOR_PRQK_A0_DUAL_DECODE_PLATFORM_LAUNCH
CURRENT_PHASE: PRQK_A0_EPOCH3_SINGLE_A100_DUAL_DECODE_ENTRY_READY
LAST_COMPLETED_PHASE: PRQK_A0_EPOCH3_CHECKPOINT_AND_STATIC_PREFLIGHT_VERIFIED
NEXT_ACTION: 用户在单卡 A100 平台启动 run_evaluate_prqk_a0_gid_epoch3_dual_decode_1xa100.sh；自动准备五组固定缓存、每项 2 条 smoke，然后同卡串行执行无约束和约束各五组评测。不读 Test，不重训 SID/SFT，不覆盖旧 A4 结果。
METHOD_VERSION: v2.1-CAT
PRIMARY_CODEBOOK_SIZES: [512, 512, 512]
CAPACITY_512_CONFIG: COMPLETED
CONTINUE_FULL_METHOD: USER_CONFIRMED_20260907
P2_FULL_QUERY_STATS: COMPLETED
V2_1_CATEGORY_MIGRATION: COMPLETED
CATEGORY_AUDIT: COMPLETED_BY_ACTIVE_P2_5_FULL
COARSE_CATEGORY_BINDING: category_code[0:2]
FINE_CATEGORY_BINDING: category_code
P2_5_CAT_QUERY_DEPTH: ACTIVE_COMPLETED
P3_EXACT_ADAPTER: FULL_COMPLETED
P3A_FULL_QUERY_CACHE: COMPLETED_AND_VALIDATED
P3A_FULL_INITIALIZATION: USER_CONFIRMED_FRESH_SEED42_SHARED_STATE
P3A_FULL_SELECT: COMPLETED_AND_VALIDATED
P3A_FULL_BEST_EPOCH: 2
P3A_FULL_FINAL: COMPLETED_AND_VALIDATED
P4_CAT_QUERY_GRAPH: FULL_COMPLETED_AND_VALIDATED
P5_CAT_POI_ONLY_PRQK: FULL_COMPLETED_AND_VALIDATED
P6_CAT_S1_S2: FULL_COMPLETED_AND_VALIDATED
P7_CAT_S3_GEO: FULL_COMPLETED_AND_VALIDATED
P8_CAT_STATIC_EVAL: FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED
P7_NOGID_S1S2_PARENT_S3: FULL_COMPLETED_AND_VALIDATED
P8_NOGID_S1S2_PARENT_COMPARISON: FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED
SID_VISUALIZATION_A0_A4_NOGID: FULL_COMPLETED_AND_VALIDATED
SID_CATEGORY_REGION_VISUALIZATION_V2: COMPLETED_AND_VALIDATED_READ_ONLY
SID_COLLISION_PREFIX_EXAMPLES: COMPLETED_15_COLLISION_BUCKETS_ZERO_SINGLETONS_READ_ONLY
P9_GID_PARENT_FINAL_ID: FULL_COMPLETED_AND_VALIDATED
P9_NOGID_FINAL_ID: FULL_COMPLETED_AND_VALIDATED
P9_DUAL_SFT_MESSAGES: FULL_COMPLETED_AND_VALIDATED
SFT_SHARED_VOCAB: COMPLETED_AND_VALIDATED
SFT_GID_PARENT_TOKENIZED_CACHE: FULL_COMPLETED_AND_VALIDATED
SFT_NOGID_TOKENIZED_CACHE: FULL_COMPLETED_AND_VALIDATED
SFT_GID_PARENT_TRAINING: COMPLETED_EPOCH3_STEP8949_EXIT0
SFT_NOGID_TRAINING: COMPLETED_EPOCH3_STEP7404_EXIT0
SFT_FIXED10K_AND_GENERALIZATION_EVALUATION: COMPLETED_10_OF_10_EACH_10000_EXIT0_VALIDATED
SFT_FULL_TEST_EVALUATION: COMPLETED_BOTH_606682_EXIT0_METRICS_VERIFIED
SFT_CONSTRAINED_VALIDATION: CPU_PREFLIGHT_PASSED_GPU_NOT_STARTED
RUNTIME_CONFIG_VERSION: v2.1-CAT_ACTIVE_POI_512x3_P10_DUAL_SFT_READY_V1
A0_GID_FINAL_ID: NOT_BUILT_PLATFORM_STAGE_PENDING
A0_GID_TRAIN_VALID: NOT_BUILT_PLATFORM_STAGE_PENDING
A0_GID_TOKENIZED_CACHE: NOT_BUILT_PLATFORM_STAGE_PENDING
A0_GID_SFT: NOT_STARTED
LAST_UPDATED: 2026-09-15 CST
GIT_COMMIT_AT_UPDATE: 54802e6674e283722ecee00fb862530df30cef9a
WORKTREE_STATE: DIRTY_USER_WORKTREE_PRESERVED
```

---

## 2026-09-15：双卡全目录约束解码入口就绪

- 原全量 Test 已由用户完成：GID/NoGID 各 606,682 条、退出码 0，HR@1 为 50.9631%/50.1919%，HR@10 为 85.5120%/85.1705%；原始结果不改。本次按用户要求新增独立约束解码对照，只评测原固定 10k 与四个泛化集，不运行新的 Test。
- 入口：`launchers/run_evaluate_qg_prqk_a4_dual_epoch3_constrained_fixed10k_generalization_2x6000d.sh`。两卡各一个分支，默认各 2 条 GPU smoke 通过后自动跑五组各 10k；模型、样本、Beam/长度/归一化等参数继承旧计划，仅添加全目录前缀约束，覆盖 GID/SID/末位可选 Dedup/闭合/EOS，不按 Query 或区域剪枝。
- 真实 launcher CPU 预检退出 0，两版各 716,245 条 Final ID 和十组 JSONL 的来源、数量、顺序、互斥、目标与 CURRENT 对齐通过；已保存独立 plan。31 项相关合成测试通过，包含真实小型 Qwen 的两种路径 Beam=10，以及 smoke 失败阻止正式评测、两卡日志转发、长 TMPDIR 拒绝和旧评分回归。
- 新目录：`outputs/eval/sft_epoch3_constrained_fixed10k_generalization_v1/`；分支内为逐组进度/结果，根 `summary.json` 将包含相对旧无约束的逐组差值和四类宏平均。日志在 `outputs/run_control/eval_a4_dual_epoch3_constrained_2x6000d/`。
- 当前宿主为单卡 A6000，非用户指定双卡 6000D，且没有平台提交接口；未启动正式 GPU 推理，无业务约束解码指标。本项只记录代码交接，不登记为完成的正式生成实验。A0-GID 数据/缓存/训练不由本次任务启动；原无约束、Test 和 SFT 源码、结果不覆盖。

## 2026-09-15：A0-GID 平台端准备与通用四卡训练入口

- 用户确认先补 A0-GID 对照，GID 在前；随后要求把全量数据与缓存工作放到训练平台，不占开发机。本次只完成源码、配置、合成测试和小型 manifest/文件状态预检，没有发布 A0 的全量数据或启动训练。
- 入口：`launchers/run_train_qg_prqk_a0_gid_sft_4gpu_3epoch.sh`。默认平台单进程 CPU 执行“Final ID → identifier-only Train/Valid → 1024 零截断检查 → packed Cache”，然后四卡 LLaMA-Factory；`--prepare-only` 只准备，`--dry-run` 只查看计划。
- 目标固定 `GID6+A0-SID3+[D]`，重新在完整 A0-GID 桶内分配末位 Dedup；Train/Valid 行数固定为 `7,586,410/597,421`，不读取 Test。复用 A4 同一扩词初始模型和全部训练超参数；数据和缓存目录独立，后续完成的阶段可校验复用，未完成阶段重做，不隐式覆盖已有训练。
- 无 A100/6000D 型号白名单，只检查四张可见 CUDA GPU、BF16 和最低可用显存；硬件运行与全量长度门禁仍须在平台通过，不将开发机 `planned_not_prepared` 当作已经具备 Cache。
- 所有源码、配置、输出和日志在 QG 下；既有 A4 模型/数据、评测源码与冻结图形依赖不变。完整命令、路径、边界和复现说明见 `../../README.md` 的“A0-GID：平台准备与四卡训练”。纯代码准备不伪装成新正式全量实验，正式编号待实际平台运行登记。

## 2026-09-14：同五类 t-SNE 与类别/区域前缀图完成

- 用户确认增加 A0→A4 迁移视图，已完成全库逐层编码/同伴变化、碰撞状态迁移及提高/持平/降低三组的 18 个展示案例：`outputs/figures/qg_prqk_sid_migration_v1/`。S1/S1S2 换码占比 `20.3397%/28.1351%`，同伴集合变化 `100%/85.3543%`；三位 SID 的 GID-parent/NoGID 换码 `41.9696%/49.6227%`，同伴变化仅 `24.0165%/25.2637%`，二者不能混淆。
- 排除自身、仅比较双侧非单例同批 POI 后，S1 粗类别同伴比例 `66.2648%→80.9657%`，S1/S2 细类别为 `57.0241%→65.6795%`。GID-parent/NoGID 分别有 `61,587/62,777` 个旧碰撞 POI 变单例，同时 `44,948/52,710` 个旧单例 POI 新增碰撞；保留负面案例，不用单例 100% 作为改善证据。不重训、不改 SID，不改变顶部 SFT/Test 主状态。细节见 `QG_PRQK.md` 的“A0→A4 前缀与成员迁移补充”。
- 用户指出末层单例不足以展示区域结构，已补充 `outputs/figures/qg_prqk_sid_collision_examples_v1/`：先要求三版完整 SID 桶均有 3–20 个不同 POI，再按五类固定 seed 抽共同锚点，末层 15/15 均为碰撞桶、0 单例，全部 54 条跨方法成员记录保存在 HTML/JSON。图中既有跨网格的同类别碰撞，也有同网格的跨类别碰撞，不按区域纯度挑选；原 V2 图和 t-SNE 未改。详情见 `QG_PRQK.md` 的“碰撞案例补充”。
- 按用户确认的 GenPOI 风格口径，三版共用五个真实类别、每类 200 个相同 POI；使用三层选中 POI 码向量归一化等权串联，联合 PCA50+t-SNE。每类固定一个锚点，展示三种方法、三层前缀的完整桶类别与 Geohash5 网格分布；不把网格冒充行政区。
- 全库 S1 粗类别纯度 A0/A4=`76.3230%/86.8014%`，S1/S2 细类别=`73.2765%/79.3472%`；两种 A4 前两层完全相同。三版高维 silhouette=`0.051067/0.061592/0.061600`，两种 A4 图形高度相近，不能宣称 S3 带来大幅语义重塑。单例占比和非单例纯度已同时记录，防止把 S3 小桶 100% 视为效果证据。
- 输出：`outputs/figures/qg_prqk_sid_category_region_v2/`，包含两个 PNG/PDF、完整抽样 ID/SID/坐标、45 个桶的精确计数、全库指标和来源 manifest。实际构建 150.88 秒，退出 0，独立输出/来源哈希复核通过；20 项相关合成回归、Ruff、compileall 通过。
- 代码、字体、缓存与输出均在 QG 下；未改旧图、码本、SID 或模型，不读业务 Validation/Test，也未启动评测。详细口径及局限并入 `QG_PRQK.md` 的“2026-09-14：GenPOI 风格同五类与具体前缀补充可视化”；复现入口见 `../../README.md`。这是补充解释性分析，顶部 SFT/Test 主阶段状态不变。

## 2026-09-14：最后一天全量 Test 入口就绪，CPU 预检通过

- 用户确认在同一台双卡 6000D 上评测 2026-07-14 全量 Test，两个分支各 606,682 条。入口为 `qg_prqk/launchers/run_evaluate_qg_prqk_a4_dual_epoch3_full_test_2x6000d.sh`，调用 `evaluate-sft-test`；完整输入、输出和复现说明见 `../../README.md`。
- 参考既有 active TIGER/MMBERT 全量 Test 的原文件视图和 `uint64` 行偏移读取设计；相关代码已复制到 QG，模型、模板、Final ID 解析、指标直接复用 QG 内模块，不调用根项目脚本。
- 已完成 Validation 的三个 evaluator 源码未修改。Test 单独实现 split 契约、成对全量扫描和 worker 调度；两段 Token 编码/生成循环从 QG Validation 适配复制，以维持已发布 plan 的源码指纹。不是重新设计生成算法。
- Test 配置通过 SHA256 引用已完成 Validation 配置，固定两份模型 SHA256；不读取 Validation 样本、不更新参数、不重建 SID、不用 Test 选择 checkpoint。结果写入独立 `sft_epoch3_full_test_20260714_v1`。
- CPU 预检进行全量文件/行契约检查和每版前 100 条 Token 检查，剩余 Token 合同在生成 chunk 内逐条检查；未宣称已完成全量 Test 零截断扫描。正式 GPU 推理尚未运行，没有 Test 效果指标。
- 真实 launcher `--dry-run` 退出 0，两版全量扫描均为 606,682 条，唯一且同序的业务键 SHA256 为 `ad01a84006f650e27e8fca3739b0106352a4f4109ee3382c309ccc16ff2d6a8f`；每版完整 Final ID 目录各 716,245 条，前 100 条样例最大输入 Token 为 GID 606 / NoGID 549，未截断。来源清单和两份 plan 已落盘，两个正式 `results/` 均未创建。
- 39 项相关合成回归全部通过，覆盖流式对齐、哈希/日期/行数契约、与 Validation 的生成评分一致性、OOM 重试、断点续跑、最后不足整块的数据、双卡调度与主日志转发；Ruff、compileall、`bash -n`、CLI/launcher `--help`、源码来源与隔离检查通过。旧 Validation plan 的源码哈希仍与当前文件一致。
- 本项是脚本交接，不新增正式实验编号。以下结果审核与历史准备记录均保留，不代表继续禁止用户本次明确要求的 Test 评测。

## 2026-09-14：双分支最终评测完成

- 正式记录：`QG_PRQK.md` 的 `EXP-20260914-01`。用户启动的一台双卡 6000D 任务于 13:24:44（北京时间）完成，总墙钟约 1 小时 26 分 49 秒；`full.exit=0`，十个 worker 全部 exit=0、各 10,000 条。
- GID / NoGID 固定 10k 的 HR@1/HR@10/NDCG@10 为 `50.31/85.19/68.7252%` / `50.19/84.67/68.2651%`；四类泛化宏平均为 `22.6750/46.6900/34.2854%` / `21.8275/46.3325/33.5531%`。宏平均不包含固定 10k。
- GID 数值上整体略高，但 NoGID 冷目标 HR@10 为 17.08%，高于 GID 的 14.18%；长尾基本持平。NoGID 各组合法 ID 率更高，两版非法候选超过 99.9% 是完整组合不在目录，尚未做逐层归因或约束解码。
- 已复核十份数据 SHA256、业务键同序/互斥、plan/run/source 签名、checkpoint 文件元数据、候选计数和排名直方图复算，以及两份 summary 的四类宏平均；全部通过。结果在 `outputs/eval/sft_epoch3_fixed10k_generalization_v1/<variant>/results/`，没有重跑或更改算法、数据和模型。
- 以下启动准备和早期阶段停止点均是历史记录，不代表当前尚未评测；当前停在结果审核，不自动推进其他实验。

## 2026-09-14：资源调整为一台双卡 6000D（历史准备）

- 用户确认没有四卡资源，两个 SFT 评测改为同一台双卡节点。当前唯一入口：`qg_prqk/launchers/run_evaluate_qg_prqk_a4_dual_epoch3_fixed10k_generalization_2x6000d.sh`，调用 `evaluate-sft --variant both`。
- `evaluation_suite.py` 使用共享十项队列，同时最多两个独立单卡 worker，空闲卡领取下一项；保留全套队列锁与分支锁，防止重复启动。每卡 batch=32、至少 36 GiB 空闲、Beam=10、cutoff=1024、chunk=500、两份 checkpoint、五组冻结样本、Final ID 查表和全部指标口径不变。
- 复用原 `data/`；新计划写为各分支 `plan_2x6000d.json`，原四卡 `plan.json` 和预检日志保留。没有旧正式生成结果需要迁移；后续仅相同计划可续跑，不混用来源。总日志/退出码在 `qg_prqk/outputs/run_control/eval_a4_dual_epoch3_2x6000d/`，结果继续按 `<variant>/results/` 隔离。
- 两个旧四卡评测脚本已从 `launchers/` 移除，可恢复副本在 `qg_prqk/outputs/run_control/eval_launcher_archive_4x6000d/`；两个四卡 SFT **训练**脚本、模型与数据未删除。以下四卡评测段落记录首版交接历史，不再是当前运行命令。
- 双卡交接核验通过：联合 launcher 的真实 `--dry-run` 退出 0，两个分支全部十个 10k 集合复用成功，`plan_2x6000d.json` 均已写出；正式 GPU 推理未启动。20 项评测回归、3 项代码隔离/来源检查、3 项 CLI 回归全部通过，覆盖双卡最大并发、跨分支十项唯一调度、失败停止同套件子进程、旧计划保留；Ruff、compileall、`bash -n`/`--help` 通过。生成/评分与数据对齐源码的 SHA256 与首版完全相同，平台初始化骨架也未改变。

## 2026-09-14：双 SFT 完成与首版四卡评测准备（历史）

- 已核验 `train_a100.exit=0`、最终权重、`trainer_state.json` 的 epoch=3 / global_step=max_steps、`train_results.json` 和末轮 Validation loss。GID-parent / NoGID 的 loss 为 `0.2006811798 / 0.3390350640`；完整训练记录见 `QG_PRQK.md` 的 `EXP-20260913-01`。
- 外层 `qg_prqk_training_manifest.json` 未落盘，保留现场，不向历史训练目录补写成功标记；此次评测 `plan.json` 重新记录读取到的 checkpoint SHA256、数据/ID manifest 与代码来源。实际训练成功以现有退出码和 Trainer 完整末步产物交叉确认。
- 新增 `evaluate-sft`，领域实现为 `sft/evaluation_data.py`、`sft/evaluation.py`、`sft/evaluation_suite.py`；平台入口是 `qg_prqk/launchers/run_evaluate_qg_prqk_a4_{gid_parent,nogid}_epoch3_fixed10k_generalization_4x6000d.sh`。
- 配置是 `qg_prqk/configs/sft/evaluation_epoch3_fixed10k_generalization_v1.yaml`。固定五组既有 Validation，不抽样、不读 Test、不修改 SID、不选 checkpoint。输出位于 `qg_prqk/outputs/eval/sft_epoch3_fixed10k_generalization_v1/`，全部正式生成指标仍为未运行。
- 两个 launcher 的真实 CPU `--dry-run` 均退出 0：各五组均为 10,000 行，业务主键哈希与原始冻结集一致；全库 716,245 Final ID 校验通过。每组前 8 条训练模板/目标映射预检通过（每分支 40 条），其余行在生成 chunk 中逐条检查；原始全量 Valid 已通过训练前零截断门禁。两份 `data/manifest.json` 和 `plan.json` 已准备好，没有创建正式生成结果或启动四卡任务。
- 交接核验：完整 QG 合成回归 196 项通过；最后补充四卡调度测试后，评测定向回归 16 项通过。Ruff、compileall、两个 launcher 的 `bash -n`/`--help`、代码复用来源校验均通过；平台初始化骨架与参考 launcher 完全一致，认证内容未写入文档，两个 launcher 继续 Git 忽略。本次没有四卡 GPU 实测，可先用 launcher 的 `--smoke-limit 8` 检查平台显存与推理，再去掉参数运行正式五组。
- 下面按时间保留的 P0–P10 停止点是历史记录，不再代表当前“尚未训练”的状态。

## 当前源码导航

2026-09-12 完成代码结构整理，但没有修改算法、配置内容、冻结产物或实验结论。公共入口统一为 `qg_prqk/scripts/qg_prqk.py`；实现按 `data/`、`adapters/`、`sid/`、`sft/` 分包，参数解析集中在 `commands/`。P3A/P4/P5/P6/P7/P8 仅作为冻结协议与历史记录编号保留，不再用作当前源码文件名。完整的“方法—阶段—实验—当前源码—产物”映射见 `../QG_PRQK_CODE_GUIDE.md`。旧记录中的源码路径仍指向当时真实文件；逐字节复核必须使用对应 `outputs/run_control/**/source_snapshot/`，不能用整理后的当前源码冒充历史快照。

---

## 1. 冻结决策

- 范围：北京单城、精确原始 `poi_id` 检索。
- POI 候选宇宙：后续阶段统一使用 `data/beijing_poi_active_order14d_history10_20260715_json/` 的 716,245 条 active POI。它由三种业务 split 的当前目标/history POI ID 构成，仅保证候选可检索，不提供 Query 监督；训练与选模仍严格 Train-only。
- POI embedding：从 `outputs/embeddings/beijing_poi_bge_m3/embeddings.npy` 按 active POI 的冻结 BGE 行子序逐值复制到 `qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/`，不重新编码，不使用 E4，不增加 PCA。
- Query：P2 高置信集合固定解释为 D3-Exact-Core；其余 Query 在 P2.5-CAT 依次判断 D2 Fine-Category、D1 Coarse-Category 和 D0，不复用旧 raw-Query 行号或聚合结果。
- 类别：完整 6 位 `category_code` 绑定为 402 个 `fine_category_id`，前 2 位绑定为 19 个 `coarse_category_id`；`category` 只作可读路径。类别不拼入 BGE、不与 POI embedding concat，也不硬映射为 SID。
- Query 粒度：D3 监督 S1/S2/S3，D2 Fine-Category 监督 S1/S2，D1 Coarse-Category 只监督 S1，D0 不参与 SID。
- SID：QG-PRQK v2.1-CAT 使用 POI/Query 独立 residual view、每层双质心共享 token index、Query–POI 层级图 alignment 和 projection residual；S1 加 coarse category 软代价，S2 加 conditioned fine category 软代价，S3 使用 D3 Query residual、GID6 局部 Geo 和 same-fine-category 困难边但不加类别分类代价。
- 已删除设计：不再融合 POI+Query 为单向量，不再使用 `Kish ESS -> per-POI gate -> lambda`，每 POI 最多 4 个 Query prototype 只可作为受控工程压缩而非核心方法。
- 主容量：`512 × 512 × 512`（2026-09-07 用户确认），每层 token index 为 0–511；BGE 仍为 1024 维。
- 类别软监督：S1/S2 category weight 启动值均为 `0.08`，Dirichlet smoothing 分别为 32/16；前 2 轮关闭类别项，第 3–6 轮线性 ramp。sample 只验证工程闭环，权重是否调整只根据 full 指标决定。
- S3 困难边：固定 `(GID6,S1,S2)` parent 内的 same fine category（即同 GID6，第一版不向相邻 cell 扩展），按 BGE/name/category/address/geo 复合分数筛选 Top-20；当前没有 canonical duplicate mapping，只排除 self-loop 和 P2 false-negative protected targets。
- Final ID：A4 GID-parent 为 `[G1..G6,S1,S2,S3] + optional [D]`；NoGID 为 `[S1,S2,S3] + optional [D]`。两者都只为完整 Base ID 的碰撞项追加 D，且 D 始终在最后。
- Dedup：复用现有基础 PID 桶内 `poi_id` 字典序、零基、确定性分配，不设计新策略。
- 第一版停止点：P8-CAT 的 `HOLD_FOR_REVIEW` 已履行。2026-09-11 用户随后明确批准两个 SID 分支进入 Final ID/SFT 数据准备；2026-09-12 又授权继续到训练脚本可启动为止。该授权不包含自动启动训练、读取 Test、外部基线、Order-B 或其他消融。
- 文档：QG-PRQK 方法、状态和实验详情统一保存在 `qg_prqk/` 下。
- 代码隔离：全部 QG 项目源码、脚本、配置、测试和方法级训练/评测入口位于 `qg_prqk/`；复用仓库代码时先复制并登记来源，QG 运行时不得导入 `src/poi_gr` 或调用仓库其他项目脚本。
- 外部依赖：PyTorch、FAISS、Transformers 与 LLaMA-Factory 只锁定版本，不复制完整源码；QG 自己的适配与编排代码仍放在 `qg_prqk/`。
- 输出隔离：QG 自己生成的所有产物统一位于 `qg_prqk/outputs/`；仓库根 `outputs/` 中的 BGE 等冻结资产只读复用。
- v2.1 active 下游配置：`qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml`，P4–P8 新输出 namespace 为 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/`。它继承冻结 active P3A-FULL/类别配置并锁定上游哈希，仅改容量与下游输出；已完成 P2.5、D3 cache、FULL 仍在原 `1024x3` 目录只读复用。历史配置、Gate 和中断产物全部保留，P2/P2.5/BGE/Adapter 不重跑。
- P5 试运行协议：用户于 2026-09-08 确认 `max_iter=60 + Top-k5/beta=15/5轮`；2026-09-09 又明确授权跳过新协议 100k/500k，由已验收 sample 直接构建全部 716,245 active POI。full manifest 显式记录该覆盖及 sample 哈希，默认 Gate 顺序仍保持严格；该授权仅限 P5 A0，不自动扩展到 P6/P7。
- P6/P7 迭代上限：用户于 2026-09-09 确认 hard alternating `max_iter=60`，其余权重、warm-up、Top-k5、projection residual、容量和停止阈值不变。旧 30 轮 downstream YAML 及 P4/P5 manifest 不修改；新 overlay 为 `qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml`。
- P6 规模协议：历史 canonical-v1 的 sample/medium 合同和所有 sample/诊断产物只读保留。2026-09-09 用户改为“小规模只跑通，随后直接全量，指标与参数只看全量”；direct-full-v2 绑定已完成 sample，显式跳过 medium。full 固定为全部 716,245 active POI、342,879 Query 和 912,980 条 S1/S2 边。
- P7 规模协议：2026-09-11 用户接受 P6 S2 诊断作为留存，明确授权 P7 sample smoke 通过后直接 full。P7 独立配置绑定冻结 P4/P5/P6 manifest，sample/full 分别使用 `11,459 POI + 832 D3 Query` 与 `716,245 POI + 291,590 D3 Query`，不读取业务 Validation/Test，完成后强制停在 `HOLD_FOR_P7_FULL_REVIEW`。
- P8 评测协议：2026-09-11 用户确认进入 P8；只比较内部 A0 与完整 A4。Prefix Probe 固定为同一 Query view 上的 content-only cosine nearest-code：S2 使用 correct S1 projection residual 与该 S1 下实际出现的候选码，S3 使用 correct GID6/S1/S2 residual 与该 parent 下实际出现的候选码；同时报告 autoregressive cumulative prefix。该 probe 是静态可预测性诊断，不等同于训练期 graph-coupled assignment，也不是最终 Qwen SFT。
- NoGID 受控变体：2026-09-11 用户要求在冻结 S1/S2 上重建 S3，不引入 GID，最终 SID 只有 `[S1,S2,S3]`。新 P7 以 `(S1,S2)` 为 parent，连续 Geo 只相对该 parent 中心表示，不读/不算/不输出 GID6/geohash；新 P8 将 A0、原 A4 与 NoGID A4 全部统一到 `(S1,S2)` S3 候选口径。该分支只是受控比较，不修改原 v2.1-CAT 定义。

---

## 2. REPO_MAP

| 范围 | 真实路径 | P0 结论 |
|---|---|---|
| QG 方法规范 | `qg_prqk/docs/methods/QG_PRQK_METHOD_SPEC.md` | v2.1-CAT canonical；类别层次锚定的 Query–Geo 双视图 PRQK |
| QG 执行计划 | `qg_prqk/docs/methods/QG_PRQK_CODEX_EXECUTION_PLAN.md` | M1、P2.5-CAT、P3A、P4-CAT–P8-CAT 单阶段状态机；P8 后强制 HOLD |
| v2.1-CAT 512 下游配置 | `qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml` | active 716,245 POI、512×3；严格继承完整方法参数，显式区分上游只读和下游输出 |
| 关系/局部码本 hard60 配置 | `qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml`、`qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml`、`qg_prqk/src/qg_prqk/sid/relational_config.py` | v1 历史 Gate 只读保留；v2 绑定已完成 sample，记录跳过 medium，并开放固定 full，不修改算法参数 |
| GID-parent 局部 S3 | `qg_prqk/configs/qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml`、`qg_prqk/src/qg_prqk/sid/{local_config,local_data,geo,local_quantizer}.py`、统一命令 `build-local-codebook` | 冻结关系化 S1/S2，只让 D3 进入 S3；固定 parent 内局部 Geo、same-fine-category 困难图、projection residual、manifest 与独立 validator；sample/full 均已完成 |
| SID 静态评测与发布 | `qg_prqk/configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml`、`qg_prqk/src/qg_prqk/sid/{evaluation_config,evaluation_data,evaluation}.py`、统一命令 `evaluate-sid` | 只读冻结上游，报告结构/类别/Query 对齐/Prefix Probe/GID6 局部分离/residual，并发布 A4 codebook、assignment、SID 和 bucket index；full 已完成，结论 `REVIEW_REQUIRED` |
| NoGID `(S1,S2)` parent S3 | `qg_prqk/configs/qg_prqk_p7_nogid_s1s2_parent_hard60_topk5_direct_full_v1.yaml`、`qg_prqk/src/qg_prqk/sid/nogid_{config,data,geo,quantizer}.py`、统一命令 `build-local-codebook-nogid` | 冻结 S1/S2，只以 `(S1,S2)` 为 S3 parent，不读/不算/不输出 GID；full 已完成并独立验收 |
| NoGID 三方静态比较 | `qg_prqk/configs/qg_prqk_p8_nogid_s1s2_parent_comparison_full_v1.yaml`、`qg_prqk/src/qg_prqk/sid/nogid_evaluation*.py`、统一命令 `evaluate-sid-nogid` | A0/原 A4/NoGID A4 都使用三位 SID 及 `(S1,S2)` S3 候选；full 已完成，NoGID 未优于原 A4，结论 `REVIEW_REQUIRED` |
| SID 三方并排可视化 | `qg_prqk/configs/qg_prqk_sid_visualization_a0_a4_nogid_full_v1.yaml`、`qg_prqk/src/qg_prqk/sid/visualization*.py`、统一命令 `visualize-sid` | 只读冻结 full SID/BGE/residual，输出码分布与转移、逐层 residual t-SNE、exact-prefix BGE cosine/类别一致性/桶层次以及可读语义样例；不改写 SID，不推进下游 |
| 双 Final ID | `qg_prqk/src/qg_prqk/sid/identifiers.py`、统一命令 `build-final-identifiers`、`qg_prqk/outputs/final_identifiers/` | GID-parent=`GID6+SID3+[D]`，NoGID=`SID3+[D]`；完整 Base ID 桶内按 poi_id 字典序分配末位 D，两者均覆盖且唯一映射 716,245 POI |
| 双 SFT Messages | `qg_prqk/src/qg_prqk/sft/data.py`、统一命令 `build-sft-data`、`qg_prqk/outputs/sft_data/` | 一次扫描冻结 history10 语料成对写出两个分支，只替换 history/target identifier；保留 Query、请求 GID、样本顺序和日期切分 |
| SFT 词表/Cache/训练入口 | `qg_prqk/src/qg_prqk/sft/`、统一命令 `prepare-sft-vocab` / `prepare-sft-cache` / `train-sft`、`qg_prqk/configs/sft/`、本地 `qg_prqk/launchers/run_train_qg_prqk_a4_*_sft_4gpu_3epoch.sh` | 共同 3,743 Token 普通原子词表和两个分支独立的 1024 Train/Valid 零截断 Cache 已完成；两分支共享 Qwen 初始状态与冻结超参数，平台 launcher 同时接受 4×A100/6000D 并强制短 TMPDIR/显存/输出门禁，Test 不注册；训练未启动 |
| 关系码本输入与闭包合同 | `qg_prqk/src/qg_prqk/sid/relational_data.py`、统一命令 `inspect-relational-inputs` | 只读核验 716,245 POI、342,879 Query、912,980 条 S1/S2 边及类别/NPY header；选样必须保留完整 Query 边并补齐目标 POI |
| 冻结 active P2.5 配置 | `qg_prqk/configs/qg_prqk_v2_1_category_active_1024x3.yaml` | 已完成阶段的不可变来源；文件中的旧容量不作为 P4–P8 设置 |
| SID 配置核验 | `qg_prqk/src/qg_prqk/sid/pipeline_config.py`、统一命令 `inspect-sid-config` | 只读 YAML/上游配置哈希校验；非图构建或聚类入口 |
| Query 图与 view | `qg_prqk/src/qg_prqk/data/query_graph*.py`、统一命令 `build-query-graph` | 显式 sample≤1000 / medium≤50000 / full=342879；full 严格复用冻结 medium、D3 mmap、Arrow 分块签名/聚合、原子缓存、恢复与独立 validator |
| POI-only 基础码本 | `qg_prqk/src/qg_prqk/sid/base_{config,data,quantizer}.py`、统一命令 `build-base-codebook` | 512×3 spherical/cosine、公共方向去除、Top-k 质心细化、projection residual；配置严格绑定旧诊断证据并隔离输出；确定性 Gate、恢复与独立 validator |
| 基础码本受控诊断 | `qg_prqk/src/qg_prqk/sid/base_{diagnostics,topk_diagnostics}.py`、统一命令 `diagnose-base-*` | 严格绑定冻结 100k manifest，重放 hard endpoint/Top-k 分支；不修改 canonical 配置或历史产物 |
| 关系码本归因诊断 | `qg_prqk/src/qg_prqk/sid/relational_{attribution,s2_diagnostics}.py`、统一命令 `diagnose-relational-*` | 隔离 graph、Query 质心先验、Top-k5、category 和 S2 因素；诊断产物不替代 canonical checkpoint |
| active 类别/BGE 行子集 | `qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/` | 从历史全库 BGE 行序精确选取 716,245 行，保存 source row、hash 和 manifest；供后续 active P2.5/P3/P4 使用 |
| 历史全库类别/BGE | `outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/`、`outputs/embeddings/beijing_poi_bge_m3/` | 2,337,178 行只读来源；旧 P2.5 与 50k Gate 保留，不再作为 active 下游候选库 |
| Query 流式管线 | `src/poi_gr/embedding/query_shards.py`、`query_stats.py`、`query_encoding.py`、`query_augmentation.py` | 可复用分片、统计、编码、mmap/manifest 框架；已有数据是 raw Query |
| 普通 RQ-KMeans | `src/poi_gr/sid/rqkmeans.py`、`scripts/sid/build_rqkmeans.py` | 可复用配置、采样、FAISS、分块编码和评测框架；算法不是 projection residual |
| 历史 Query-Predictable RQ | `src/poi_gr/sid/query_predictable_rqkmeans.py` | 仅复用 bounded Top-K routing 等局部工具，不继承其双视图目标或负实验配置 |
| GID | `src/poi_gr/pid/geohash.py` | 现有 longitude-first Geohash6 编码与 GID-first compose 可复用 |
| Dedup / Final PID | `src/poi_gr/pid/dedup.py` | 支持九 Token Base PID、碰撞项追加 D、固定宽度 10 列存储 |
| Trie / 生成评测 | `src/poi_gr/pid/trie.py`、`src/poi_gr/sft/evaluation.py` | 已支持 9/10 Token Final PID、回表与 Dedup 分组指标 |
| GID-first SFT 转换 | `src/poi_gr/methods/tiger/pid_order_data.py`、`scripts/tiger/build_pid_order_sft_data.py` | 同行同序替换历史和目标 PID 的实现可复用/泛化 |
| SFT 配置参考 | `configs/sft/tiger_bge_m3_1024x3_gid6_sid_dedup_history10_query_gid_v1.yaml` | cutoff 1024、3 epoch、global batch 512 的当前可运行参考 |
| QG 代码与配置 | `qg_prqk/configs/`、`qg_prqk/scripts/qg_prqk.py`、`qg_prqk/src/qg_prqk/`、`qg_prqk/tests/` | 当前源码按 data/adapters/sid/sft 分域，统一 CLI；D3 选样、Query cache、精确检索、Adapter、SID、SFT 和 validator 自包含，运行时不导入 `src/poi_gr` |
| QG 运行产物 | `qg_prqk/outputs/` | QG 唯一可写产物根目录；历史 P2/P2.5/50k Gate 与 active 新产物使用不同 namespace，均由 Git 忽略 |
| P3A-FULL Query cache | `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_cache_exact_full/` | 291,590×1,024 float16；36 个原子提交块、恢复来源、hash、manifest 和完整 NPY；正式 FULL 直接复用 |

---

## 3. BASELINE_BGE（历史来源与 active 派生）

- 配置：`configs/embedding/embedding_bge_m3.yaml`。
- 模型：`models/bge-m3`，SentenceTransformer，`prompt_name=null`，right padding，最大长度 512。
- POI 文本：冻结北京 POI 主表已有 `text` 字段，不重新拼接。
- 向量：`outputs/embeddings/beijing_poi_bge_m3/embeddings.npy`。
- 行映射：`outputs/embeddings/beijing_poi_bge_m3/poi_ids.jsonl`，POI ID SHA256 为 `b3d409ef673bc176eb3637d43de8841148377ba6b251e22ff52684f9b70e98e7`。
- shape/dtype：`[2,337,178,1024]` / `float16`。
- 归一化：`normalize_embeddings=true`；PCA/truncate_dim 为 `null`。
- 数据 fingerprint：`d97c1dfbb82b46ede4e83a8e504370bec798ff986e6494a69a5e0dd4a91ea071`。
- Manifest：`outputs/embeddings/beijing_poi_bge_m3/manifest.json`，SHA256 `3fabca469d0a6e9920b8d6523c69959db1f733c6564f5b6cb1b3db21ca7a969a`，状态 `completed`。
- active 派生：按 `data/beijing_poi_active_order14d_history10_20260715_json/poi_ids.jsonl` 在上述完整 BGE 行序中做严格递增行选择，生成 716,245×1,024 float16 子集；不重新编码文本，逐值保持一致。
- 决策：历史全库 BGE 只作不可变来源；active 子集发布后，P2.5/P3A-FULL 及后续阶段只读取 QG 内的 active BGE 资产。

---

## 4. BASELINE_RQK

- 配置：`configs/sid/rqkmeans_bge_m3_1024x3.yaml`。
- 实现：`src/poi_gr/sid/rqkmeans.py`；三层 sequential Euclidean K-Means，逐层执行 `residual -= selected_centroid`，不是 spherical K-Means，也不是 projection residual。
- 后端：FAISS GPU；500,000 条冻结样本，20 iterations，3 redo，seed 42，1% validation，分块/mmap/full encode。
- 完成产物：`outputs/sid/rqkmeans/bge_m3/screen/gpu_greedy_1024x3_s500k_i20_r3/`。
- SID Manifest：`sid_manifest.json`，SHA256 `e31800722e500a5224b13f7dd4668d3f2bb1d7355b0c03b65ff24f5d7eaf6998`；SID shape `[2,337,178,3]`，codebooks shape `[3072,1024]`。
- 静态结果：distinct SID 1,777,949（76.0725%），碰撞 POI 844,892（36.1501%），collision excess 23.9275%，最大桶 467；三层利用率均为 100%。
- 既有下游：旧 `[S1,S2,S3,C]`、cutoff 512 协议的固定 10k HR@1/HR@10/NDCG@10 为 51.83%/86.72%/70.1553%，不能直接作为 QG GID-first+Dedup 的公平端到端对照。
- 决策：P0–P8-CAT 不处理历史外部基线；P8 `HOLD_FOR_REVIEW` 后再由用户确认是否复用或独立重跑。

---

## 5. GID_PID_FORMAT

- GID 来源：北京 POI `lng/lat` 经 QG 自包含的标准 longitude-first Geohash6 实现生成；每层字符映射到 32 个 GID Token。
- A4 GID-parent Base/Final ID：`[G1,G2,G3,G4,G5,G6,S1,S2,S3] + optional [D]`；Base distinct=`672,294/716,245`，最大桶 56，实际使用 `D0..D55`，Final unique=`716,245/716,245`。
- NoGID Base/Final ID：`[S1,S2,S3] + optional [D]`；Base distinct=`613,198/716,245`，最大桶 159，实际使用 `D0..D158`，Final unique=`716,245/716,245`。
- `D` 在每个完整 Base ID 碰撞桶内按 `poi_id` 字典序从 0 分配，单例以 `-1` 存储且序列中省略 D；两种序列中 D 都固定为最后一位。
- 可复用样例产物：`outputs/pid/tiger/TIGER-BGE-M3-1024x3-e20-G6-Dedup/`；595,173 个碰撞 POI、192,330 个碰撞桶、最大桶 277、最大 D 为 276，追加 D 后 2,337,178 条 Final PID 全部唯一。
- Final PID Manifest：`final_pid_manifest.json`，SHA256 `a9bbf767d303b51d551d32ca91297c2d26dc02a2220a689c6df2282fdd8d9e5c`。
- Trie：`src/poi_gr/pid/trie.py` 校验 base order、9/10 Token 长度和 Token 范围；`src/poi_gr/sft/evaluation.py` 负责候选结构校验、Trie 回表、去重和 Exact/Dedup 指标。
- QG 约束：先冻结全部 SID，再独立生成 Base PID 和 D；不得在 SID 聚类中强制唯一，不得设计新的 Dedup 规则。

---

## 6. DATA_SPLITS

- 当前 POI catalog：`data/beijing_poi_active_order14d_history10_20260715_json/`，716,245 条；manifest SHA256=`dc13c3f137c57ba715f129ff2ccbbd8909d910da3cebc4a9fdbce2172f4d144a`，POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- active 定义：14 天 Train/Validation/Test 当前目标与 history 中出现过的 POI ID；记录仍是 2,337,178 条完整北京 POI 中的逐字节子集，保持原行序。它只定义闭集候选，不用作 Query 监督。
- active 类别事实：保留全局 402 类稳定 index，716,245 条 active POI 实际出现 397 个细类、全部 19 个粗类；缺失细类为 `142411、261311、261410、261412、271029`。不补入非 active POI，不把 397 类重编号。
- 实际 POI 字段：`address, alias, area, category, category_code, city, click_score, displayname, lat, layer, lng, poi_id, source_dt, text`。
- 当前 QG 使用：`poi_id, displayname, alias, category, category_code, address, lat, lng, city, source_dt, text`；`area/layer/click_score` 不进入主方法。`category_code` 完整值为 fine category，前 2 位为 coarse category；不使用不存在的 category_l1/l2 物理列。
- 已有类别 mapping manifest：`outputs/embeddings/gnpr_sid/bge_m3_category_pluscode6_full_v1/manifest.json`，SHA256 `f47dd7c6673221140f41a83fee7e32b33381ad0169e919bc5f79cc5cef140a0e`；声明 2,337,178 行、402 类、`invalid_category_count=0`、BGE POI 行序。
- 已有 fine index：`category_indices.npy`，shape `[2,337,178]`、dtype `int32`、取值 0–401、402 个 index 全部出现，SHA256 `5d895b0f79192fa4d82f6f158ded85d93d24e72e2e677f26475d08978c61e7a5`。
- 已有 vocab：`outputs/embeddings/gnpr_sid/history10_pluscode6_top10_v1/category_vocab.parquet/`，402 个唯一 6 位数字 `category_code`；其数据分片 SHA256 `7c0b04e98496cf0f2da21639fec589b3f74d63a9972e95204cd98e982e919419`，前 2 位共有 19 类。
- 订单/SFT 源：`data/sft/beijing_order_main_v1/manifest.json`，SHA256 `f254e6f2ad8c0591732887163a1211a46e9718f4071674886c12b26360334ce9`。
- Train：2026-07-01—2026-07-12，7,586,410 条，SHA256 `4ed3f3849e0beb10df500f013bc08b4663b0dca5df92dde9c8b3cb4a7ec9ffe4`。
- Validation：2026-07-13，597,421 条，SHA256 `14dabc842ba4623f19b2d9e21cafa026917d333db78016956cc4bcfc506ba187`。
- Test：2026-07-14，606,682 条，SHA256 `d1ac483a2ddc880b8261a2644b865dba1eef09f712930b51f46d8283faafa8b7`。
- 泄漏约束：Query 统计、Adapter、原型和 SID 只读取 Train；内部 dev/holdout 必须从 Train 再切分；正式 Validation/Test Query、订单标签与统计量不参与构建或调参。active POI 候选成员资格覆盖三种 split，只用于保证评测目标可检索，不计作监督泄漏。
- 已有 raw-Query 资产：256 个 hash shard、1,408,778 个唯一 Query、2,063,175 个 Query–POI pair、覆盖 491,213 个 POI；manifest 明确为 `query_normalization=none`。
- QG 决策：旧 Query 数量只作容量预估；P2 必须重新归一化、重新编号和重新统计，不得假设归一化后数量相同。

---

## 7. SFT_PIPELINE

- 冻结变换源：`data/sft/tiger_active716k_bge_m3_512x3_history10_query_gid_v1/`，只用旧 TIGER Final ID 反查 `poi_id`；旧 SID/C 不作为 QG 标签。源 Train/Valid/Test=`7,586,410/597,421/606,682`，合计 8,790,513 条、43,208,167 个 history occurrence。
- GID-parent Messages：`qg_prqk/outputs/sft_data/a4_gid_parent_order_a_history10_v1/`，manifest SHA256=`c05810df785952b5e4412d4e67d31324587e2e02fb940ebb005331d9ea09e114`；目标中需要 D 的订单为 1,618,136（18.4078%）。
- NoGID Messages：`qg_prqk/outputs/sft_data/a4_nogid_sid3_history10_v1/`，manifest SHA256=`b173fe1411b6043727bd138a88b93bb9456b2bff89522d1c46b75e6304d21e65`；目标中需要 D 的订单为 2,589,694（29.4601%）。
- 两套 Messages 严格逐行对齐，只允许 history/target identifier 不同；NoGID history/target 不含 GID，但请求 `<USER_GID>` 原样保留。两份 `special_tokens.json` 逐字节一致，共 3,743 个普通原子 Token，覆盖结构、GID32、用户桶2000、SID 512×3 和 `D0..D158`。
- 训练配置：`qg_prqk/configs/sft/a4_{gid_parent,nogid}_history10_v1.yaml`；共同 Qwen3-0.6B 初始模型、3 epoch、per-device batch 8、gradient accumulation 16、4 GPU/global batch 512、LR 5e-5、seed 42、cutoff 1024。
- 平台入口：本地 `qg_prqk/launchers/run_train_qg_prqk_a4_{gid_parent,nogid}_sft_4gpu_3epoch.sh` 复制既有训练平台骨架并登记 provenance；每个脚本都接受 4 张同型号 A100 或 6000D，使用分支独立短 TMPDIR 与 run-control，不改变训练超参数。平台字段按规则由 Git 忽略。
- 共同词表：`qg_prqk/outputs/models/Qwen3-0.6B-QGPRQK-A4-Vocab-v1/` 已完成；base tokenizer `151,669`，新增普通原子 Token `3,743`，最终 tokenizer `155,412`，两个分支共用 extended tokenizer SHA256=`e4013655c39ae69d1e8808ec412168c8553d86c84f9b6cc32b757cf58f55ec54`。扩展前 embedding 行逐值保持，新增行均 finite，input/output embedding 仍 tied。
- GID-parent Cache：`qg_prqk/outputs/sft_tokenized/a4_gid_parent_history10_v1/`，约 33 GiB；Train/Valid 原始行=`7,586,410/597,421`，packed 行=`1,527,130/114,056`，total token length max/p50/p90/p95/p99/p99.9=`988/178/364/372/404/513`，超 1024=`0`，target truncation=`0`，cache manifest SHA256=`5f0ac5f1ca64e3faf7ee1c20acff669071df395ff3fb50369ce5e889c7351576`。
- NoGID Cache：`qg_prqk/outputs/sft_tokenized/a4_nogid_history10_v1/`，约 28 GiB；Train/Valid 原始行=`7,586,410/597,421`，packed 行=`1,263,217/94,446`，total token length max/p50/p90/p95/p99/p99.9=`947/149/299/307/339/448`，超 1024=`0`，target truncation=`0`，cache manifest SHA256=`5ddc9d12f59df2b6f65d2931e45e9a3a39e3dab5db6fbb642c47e79d22906d59`。
- 平台入口验收：两个脚本 `bash -n`、`--help` 和完整 `--dry-run` 均退出 0；dry-run 分别绑定上述 Cache manifest 与数据 manifest `c05810df...e114` / `b173fe14...1e65`，确认 global batch 512、3 epoch 和 cutoff 1024，未挂载 OFS、未检查/占用 GPU、未创建训练输出。
- 代码验收：全目录 compileall、Ruff 均通过；设置 `PYTHONPATH=qg_prqk/src` 后完整 QG 合成回归 `178 passed`（191.92 秒）。
- SFT 门禁：Train+Valid 全量 token preflight 已满足零超长、零 target truncation；cache/training/evaluation 三处 cutoff 同为 1024。两个 Cache 均不读取或注册 Test。
- LLaMA-Factory：仓库 `third_party/LLaMA-Factory` 当前 commit `95ac3f2373b82662c1bd855c284d3379e6a763d3`，工作树为 modified；环境包 `llamafactory 0.9.4`。
- 当前边界：Final ID、全量 Messages、共同词表、两套 Tokenized Cache 和两个训练入口均已完成并验证；Qwen SFT checkpoint、训练指标与端到端评测仍未生成。当前等待用户审核并选择启动分支，不得自动训练或猜测指标。

---

## 8. REUSABLE_COMPONENTS

以下均为已审计的复制来源，不是 QG 的运行时入口。具体功能进入相应阶段时，必须复制到 `qg_prqk/`、登记 provenance，并从 QG 内部入口调用：

- 冻结北京 BGE-M3 POI embedding、POI 行映射、manifest 与 fingerprint。
- Train-only Query 的流式扫描、稳定 hash shard、Parquet 分区、mmap encoding、atomic manifest 和断点续跑框架。
- 普通 RQK 的配置读取、输入 fingerprint、冻结样本、FAISS GPU、chunk/mmap、codebook/SID artifact 和统一静态评测框架。
- 历史 QD-RQ 的 bounded Top-K routing、GPU assignment 和诊断工具；只允许按 QG 公式局部复用。
- Geohash6 编码、GID-first Base PID compose、Dedup assignment、Final PID manifest、Trie、回表与 Exact/Dedup 评测。
- 同行同序 SFT PID 替换、special token 构建、cutoff 1024 preflight/cache 和 Qwen3-0.6B 训练协议。
- 当前环境：`/ofs/map_search/hudan/envs/poi-gr/bin/python`；torch 2.9.1+cu128、pandas 2.3.3、pyarrow 19.0.1、scikit-learn 1.7.2、sentence-transformers 5.1.2、transformers 4.52.4；FAISS import 可用。
- P1 已将配置校验、正式 Train Query 解析、9/10 Token 结构校验和原子 Manifest 写入模式复制/适配到 `qg_prqk/src/qg_prqk/`；来源与修改记录在 `qg_prqk/configs/code_provenance.yaml`。

---

## 9. COMPONENT_STATUS

- P3A-FULL、512×3 配置迁移和 P4 1,000-query sample / 50,000-query medium / 342,879-query full 已完成并独立验收；旧 50k Adapter Gate 只用于同一 holdout 对比，未续训或覆盖。
- P4 full 已通过固定行数、medium 前缀哈希锁定、D3 mmap 分块与 Arrow 有界签名/聚合完成规模验收；真实 dry-run 峰值约 4.54 GiB，构建+内联 validator 外层峰值约 7.07 GiB，无 OOM/Swap。
- POI/Query 双质心共享 token、交替 graph-aligned assignment、S1/S2 类别软代价、Query centroid shrinkage、逐层 projection residual 与收敛诊断；POI-only PRQK 内部 A0 已由 P5 full 完成。
- P7 已完成 Geohash6 cell center/size 解码、北京局部米制坐标、局部 Geo feature、robust standardization、singleton zero 和 GID4/5/6 桶统计。
- P7 已完成固定 `(GID6,S1,S2)` parent 内的 same-fine-category + BGE/name/category/address/geo Top-20 困难图、局部细化和确定性诊断；第一版不扩展相邻 GID6。
- A0/A4 静态比较、分 D1/D2/D3 Prefix Probe、全量 SID/bucket artifact 编排与 P8-CAT 报告。
- `pyproj/polars/geohash2` 当前未安装；本项目已有 Geohash encoder，P7-CAT 应优先实现轻量 cell decode 和北京局部切平面，不因缺包改变主方法。

---

## 10. ASSUMPTION_CONFLICTS

- v2.1-CAT 已解决旧 Family 冲突：D2 改为完整 `category_code` 的 Fine-Category，D1 改为 `category_code[0:2]` 的 Coarse-Category；本版禁止 brand/family/chain/parent 和名称实体族启发式。
- QG 的 active `category_mapping.parquet` 已在 P2.5-CAT 全量生成并独立复核：716,245 行，全局 402 个 fine ID 中实际出现 397 个、19 个 coarse 全部出现；非法 6 位 code、BGE 行序错位和 `fine -> coarse` 冲突均为 0。旧 2,337,178 行版本只读保留，不作为下游输入。
- D2 启动阈值为 `count>=3, concentration>=0.80, normalized_entropy<=0.40`，D1 启动阈值为 `count>=3, concentration>=0.85, normalized_entropy<=0.35`；P2.5-CAT 仍须报告 Train-only 小范围敏感性，不读取 Valid/Test 调参。
- 已有 `S1=category_code` 硬类别负向实验唯一率仅 53.1864%、碰撞 POI 60.2040%、最大桶 601；v2.1-CAT 只能使用 warm-up 的软类别代价，`0.08` 必须通过 sample/medium Gate 后才可 full。
- 当前不存在 canonical duplicate/equivalent mapping；P7-CAT 不能假设该输入存在，只排除 self-loop 和 P2 false-negative protected targets。
- 延后决策：Final PID、Qwen SFT、普通 BGE-RQK 外部比较、Order-B 和其他消融全部在 P8-CAT `HOLD_FOR_REVIEW` 后询问用户。
- 用户确认的仓库级文档例外：QG-PRQK 专属方法、状态和实验详情只保存在 `qg_prqk/` 下，不复制到根 `docs/` 或其他方法文档。
- 用户确认的代码隔离要求：QG-PRQK 使用的全部仓库项目代码都在 `qg_prqk/`；既有实现只作为复制来源。第三方依赖与 LLaMA-Factory 当前按版本锁定的外部依赖处理，不复制完整源码。
- 用户确认的输出隔离要求：QG 新产物全部写入 `qg_prqk/outputs/`，不使用仓库根 `outputs/qg_prqk/`；根 `outputs/embeddings/` 等既有冻结输入仍按真实路径只读加载。
- P1 发现并修复：方法规范默认配置块的困难图阈值仍为旧值 0.55；已按用户确认口径统一为 0.60，配置加载器也固定拒绝其他值。
- active 数据修订：后续 canonical 配置和输出迁移到 `_active` 独立 namespace；旧 `qg_prqk_v2_1_category_1024x3.yaml`、旧 P2.5、50k Gate 和旧 full 局部产物均保留为历史证据。
- P3A 参数已在 Gate 冻结并实测：单卡 `faiss_gpu_flat_ip` 精确 Top-100、POI 向量 add 前与 Query search 前 float32 L2 归一化、Adapter batch 2,048；不再保留 CPU/GPU 或 batch 待定冲突。
- P3A 50,000 条 Gate 与 active FULL 是两个独立实验；FULL 完成时曾停在 `HOLD_FOR_P3A_FULL_REVIEW`，2026-09-07 用户确认继续完整方法后转为 P4 准备。2026-09-05 用户确认建模前 seed=42 的同结构 identity 新初始 state，SELECT/FINAL 逐值共用；不声称它与历史 Gate 未保存的初始随机权重逐值相同。
- 512 容量修订不采纳此前讨论的 Query-only/普通 additive RQK 简化，不删除类别、D1/D2、公共方向去除、投影残差或 Geo，不调整既定权重；旧目录中的 `1024x3` 是真实历史来源命名，不是待清理冲突。

---

## 11. V2_MIGRATION（历史记录，已被 v2.1-CAT 取代）

- 状态：SUPERSEDED（2026-09-02）；这是旧 v2 方法文档迁移的历史记录，不是当前执行口径，也不分配实验编号。
- 核心变化：P2 高置信集合改释为 D3-Exact-Core；新增 D0/D1/D2/D3 supervision depth；POI/Query 改为独立 residual view、双质心共享 token 和层级图 alignment；S3 只接收 D3 Query residual 并保留 GID6 局部 Geo/困难实体。
- 删除/降级：删除 POI+Query 单向量融合和 ESS 驱动的 per-POI gate/lambda；最多 4 个 Query prototype 降为可选工程压缩。
- 当时的字段处理：Exact 固定原始 `poi_id`；旧 v2 曾把 Family 候选审计列为待办。该待办已被 v2.1-CAT 明确删除，不能继续执行或恢复名称启发式。
- 当时的执行顺序：旧 v2 使用 P0–P12 编号。该编号已被 v2.1-CAT 的 M1、P2.5-CAT、P3A、P4-CAT–P8-CAT 与 `HOLD_FOR_REVIEW` 取代，不再作为当前状态机。
- 进度对齐：P2 full 保持 `COMPLETED`；P3 因无真实 cache/log/checkpoint 固定为 `CODE_ONLY`；旧 v1.1 配置和 P2 artifact 原样只读保留。
- 文件收敛：v2 更新包内容已合并进 `qg_prqk/` canonical 文档，临时更新目录已删除，只保留主目录文档。

### V2_1_CATEGORY_MIGRATION

- 状态：COMPLETED（2026-09-02）；这是方法与配置迁移，不是实验，不分配实验编号。
- 方法审核：认可“全量类别结构锚点 + Query 粒度 + S3 Local Geo”主线；类别仅进入 S1/S2 assignment 软代价，不加入 BGE，不与 embedding concat，不硬设 `S1=CategoryID`。
- 类别绑定：Fine=`category_code` 完整 6 位（402 类），Coarse=`category_code[0:2]`（19 类），`category` 仅作可读路径；existing mapping 覆盖 2,337,178 个 BGE 行序 POI、invalid=0。
- 粒度迁移：D1=Coarse-Category、D2=Fine-Category、D3=既有 Exact-Core、D0 不参与 SID；删除旧 D2 Family/Brand/Entity-Group 及名称启发式。
- 真实状态修正：更新包对 P3 的描述是条件性假设；仓库实际仍为 `CODE_ONLY`，所以 P2.5-CAT 后保留 P3A 最多 50,000 D3 Gate，不声称已有 checkpoint。
- 输入修正：仓库没有 canonical duplicate mapping，P7-CAT 不将其设为必需输入；只使用现有 fine category、GID、BGE/name、self-loop 与 false-negative 保护。
- 风险保护：已有硬类别负向实验表明大类根桶会坍塌；S1/S2 类别权重 `0.08` 只作启动值，前 2 轮关闭、第 3–6 轮 ramp，sample/medium Gate 后才允许 full。
- 第一版边界：阶段收敛为 M1、P2.5-CAT、P3A、P4-CAT–P8-CAT；P8 后固定 `HOLD_FOR_REVIEW`，不自动进入 Final PID/Qwen SFT/外部基线。
- 输出清理：本轮未生成类别审计或 supervision artifact；此前临时类别审计配置已删除。v2.1 更新目录在内容合并后删除，只保留 canonical 文档与配置。

---

## 12. 阶段记录

### 512×3 下游配置迁移（2026-09-07，非实验）

- 状态：COMPLETED；本步不启动 P4、全量 embedding、聚类或其他训练，不分配新实验编号。
- 用户决策：继续 v2.1-CAT 完整方法，在 active 716,245 POI 上使用 `[512,512,512]`；BGE 1024 维、数据、上游 Adapter 和算法权重不变。
- 实现：新增 `configs/qg_prqk_v2_1_category_active_512x3.yaml`、`src/qg_prqk/downstream_config.py`、只读 `scripts/inspect_downstream_config.py` 和 `tests/test_downstream_config.py`；同步 canonical 方法/计划/README/命令/续跑提示词，刷新源码文档 manifest。
- 数据流：新 YAML → 冻结 P3A-FULL → 冻结 active P3A/类别 YAML；校验来源配置哈希并继承算法参数，显式登记 P2.5、D3 cache、FULL/FINAL manifest 和 FINAL checkpoint。只有新下游写入 512 namespace；上游文件/目录不改名、不覆盖。
- 核验入口：`/ofs/map_search/hudan/envs/poi-gr/bin/python qg_prqk/scripts/inspect_downstream_config.py --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml`；预期 `config_validated`、`downstream_started=false`，`--resolved` 展开全部继承参数。配置检查不代替后续实际输入内容验收。
- 验证：9 项新配置测试及全部 81 项合成回归通过（测试执行 68.445 秒）；compileall、Ruff、CLI `--help`/摘要/`--resolved` 通过。另重算新配置引用的 4 份 manifest 与 FINAL checkpoint 哈希，5/5 匹配；原有源码与上游 YAML 相比迁移前快照未改变。新配置 SHA256=`a34b5cfd531d84fae14aa719d6546ff23da577830d48665a9529e74f46bd2ac5`，resolved signature=`b4bc8a786841faf3dcedff375f4e7390efbafec54df14f27a2c3b540e4ed42c2`。
- 限制：尚无 512 SID，不能报告碰撞率或声称容量调整提升效果；P4–P8 仍须逐阶段验收。

### M1 — v2.1-CAT 文档、配置与进度迁移

- 状态：COMPLETED
- 开始/完成时间：2026-09-02
- 修改范围：只更新 `qg_prqk/` 内 canonical 方法、执行计划、状态、README、命令/续跑提示词和文档 manifest，新增 v2.1-CAT 合同配置；未修改算法代码或历史 artifact。
- 审核结论：方法主线可执行；以 19 个 coarse category 锚定 S1、402 个 fine category 锚定 S2，类别保持软约束，D3 Query 与 Geo 只在 S3 精确消歧。
- 事实修正：P3 保持 `CODE_ONLY` 并保留 P3A；不存在的 canonical duplicate mapping 未写成主方法依赖；类别 `0.08` 是 Gate 启动值而非已确认 full 超参。
- 类别证据：existing manifest 声明 2,337,178 行、402 类、invalid=0；fine index 与 vocab hash 已写入 canonical 配置。M1 未生成 QG 类别数据产物，P2.5-CAT 仍须完整契约校验。
- 验证：v2.1 YAML、Markdown 围栏和 `MANIFEST.sha256` 通过；类别 index 实测为 `[2,337,178]`/`int32`/0–401，vocab 为 402 个唯一 6 位 code 和 19 个两位前缀；compileall、Ruff 及 43 个既有合成单测通过。
- 阻塞项：None；M1 完成时的唯一下一步是 P2.5-CAT，该阶段现已完成。

### M0 — v2 文档迁移与进度对齐

- 状态：COMPLETED
- 开始时间：2026-09-02
- 完成时间：2026-09-02
- 修改范围：只更新 `qg_prqk/` 内 canonical 方法、执行、状态、命令、提示词、README 和文档 manifest；未修改算法代码、YAML 配置或运行产物。
- 关键结论：v2 方向认可；已按真实字段、P3 实际完成度和“主方法后比较基线”的用户约束解决草案冲突。
- 测试：Markdown 围栏、路径、术语、阶段状态和 `MANIFEST.sha256` 静态校验通过；QG compileall 通过，43 个既有合成单测通过。未运行业务数据处理、embedding、ANN、真实 Adapter、KMeans 或 SFT。
- 当时的下一步：旧 v2 曾计划 P2.5A Category/Family 审计；该计划已被 v2.1-CAT 的 P2.5-CAT 粗/细类别固化完整取代，不得继续执行旧阶段。

### P0 — 仓库审计与基线锁定

- 状态：COMPLETED
- 开始时间：2026-08-31
- 完成时间：2026-08-31 15:30:29 CST
- 修改文件：QG v1.1 方法规范、执行计划、状态模板、提示词、命令文档、README、`MANIFEST.sha256`，以及本状态文档；按用户追加条件冻结 `qg_prqk/` 自包含代码边界。
- 执行命令：只读检查 `git status/rev-parse`、配置与 manifest、实际 POI/SFT schema、现有 artifact/依赖版本；未启动 GPU、embedding、KMeans、SFT 或评测任务。
- 测试：QG 文档 SHA256 Manifest、Markdown 代码围栏、过期路径/字段/方法顺序扫描；最终结果见本阶段验收。
- 关键发现：真实 BGE 输入、raw-Query 资产、普通 additive RQK、GID-first+Dedup、9/10 Token Trie 和 cutoff 1024 SFT 链路均已定位；这些仓库实现仅作为复制来源，QG 算法组件与自包含入口尚未实现。
- 阻塞项：None。

### P1 — 配置与数据契约骨架

- 状态：COMPLETED
- 开始时间：2026-08-31
- 完成时间：2026-08-31 16:01:06 CST
- 修改文件：`qg_prqk/configs/qg_prqk_1024x3.yaml`、`code_provenance.yaml`、`qg_prqk/src/qg_prqk/{config,data_contracts,artifacts,cli}.py`、package 初始化、`qg_prqk/scripts/preflight.py`、四份合成测试、README、方法规范阈值和本状态文档。
- 实现：冻结北京 BGE/目录/SFT 路径与 hash、实际 POI/SFT 字段、1024×3、困难图权重与阈值、GID6+SID3+[D]；P1 当时配置输出位于 `outputs/qg_prqk/`，P2 开始按用户新增要求迁移并固定到 `qg_prqk/outputs/`；`overwrite=false`；Artifact Manifest 模板状态为 `planned`。
- 复用：从现有 RQK、Query shard 和 Dedup 实现复制/适配配置校验、原子写入、Train Query 解析和 PID 结构合同；QG 运行时不导入 `poi_gr`，来源路径、commit、源/目标 SHA256 和修改说明已登记。
- 测试：compileall 通过；16 个合成单测通过；CLI `--help` 通过；真实 `--limit 2 --dry-run` 通过，验证 Embedding `[2337178,1024]`/float16、Train/Valid/Test 声明行数 7,586,410/597,421/606,682 和两条同行序 POI/ID/Train 样例。
- 输出：dry-run 只向标准输出生成 `qg-prqk-artifact-manifest-v1` 模板，未写入实验产物，未扫描或哈希全量 POI/POI-ID/Train。
- 阻塞项：None。

### P2 — Query–POI 统计管线

- 状态：COMPLETED
- 开始时间：2026-08-31
- 小样本完成时间：2026-08-31 16:33:00 CST
- 全量产物完成时间：2026-08-31 19:11:30 CST
- 修改文件：`qg_prqk/src/qg_prqk/{query_normalize,query_stats,query_stats_cli}.py`、`qg_prqk/scripts/build_query_stats.py`、Query 单测、P2 配置、provenance、README、方法/执行/状态文档，以及仓库 `.gitignore` 的 `qg_prqk/outputs/` 忽略项。
- 实现：只读 canonical `train.jsonl`；先做 NFKC、英文小写、常见标点统一、连续空白合并和首尾去空白，再以 QG 专用 BLAKE2b namespace 分成 256 个 Parquet shard。逐 shard 有界聚合 `n(q,p)`、`n(q)`、top1/top2、margin、entropy/normalized entropy；按冻结阈值保留 top1，计算非负 `w_qp`，并独立保存全部达到计数/占比阈值的 false-negative mask。
- 确定性：Query ID 按 shard 递增后 normalized Query 字典序分配；top1 同频时按 `poi_id` 字典序；展示 raw Query 同频时按 raw Query 字典序。旧 raw-Query ID、统计和 embedding 行号未复用。
- 泄漏保护：真实 SFT manifest 的日期切分和 Train hash 声明经过校验；逐行要求 `split=train`；manifest 明确记录 Valid/Test 未读取。全量模式额外核对 Train 行数与 SHA256；`--limit` 产物明确标记 prefix sample。
- 全量输出：`qg_prqk/outputs/qg_prqk_1024x3_v1_1/query_stats/`，含 `query_shards/`、`query_stats/`、`query_poi_pairs/`、`false_negative_mask/`、`stats_report.json`、`manifest.json` 和 `_SUCCESS`；246 MiB、1,027 个文件，manifest SHA256 为 `25238e011dc0b162d4e27082f4bc918d1a2e64aa3ac0a46e415bee4ab07cd162`。
- 全量结果：7,586,410 条 Train 订单生成 1,372,661 个 normalized unique Query、2,022,443 个 unique Query–POI pair，覆盖 491,213 个 POI；保留 291,590 个高置信 Query（21.2427%），覆盖 153,349 个保留 POI，产生 557,639 个 false-negative pair。Query 频次分布为 `n=1:894,490`、`n=2–4:328,420`、`n>=5:149,751`。
- 全量泄漏与指纹校验：`limit=null`、`is_prefix_sample=false`；扫描 3,727,174,704 bytes，行数与 Train SHA256 `4ed3f3849e0beb10df500f013bc08b4663b0dca5df92dde9c8b3cb4a7ec9ffe4` 均与冻结声明一致；`valid_and_test_read=false`。
- 全量性能：耗时 327.278 秒，峰值 RSS 397.863 MiB。随后独立执行 `--resume` 用时约 10.3 秒，256×4 个 Parquet 分片的 schema、行数和 SHA256 全部通过并成功复用。
- 小样本输出保留在 `qg_prqk/outputs/smoke/p2_query_stats_256_v1/query_stats/`；Train 前 256 行产生 192 个归一化 Query、197 个 Query–POI pair、16 个高置信 Query 和 22 个 false-negative pair，只作链路 smoke。
- 人工可读抽样：高置信简称包含“三里屯”“大悦城”；模糊 Query 包含“北京西”“北京西站”“雍和宫”；低频样例在报告中单列。过滤原因与 entropy/margin 一致。
- 测试：32 行真实 dry-run 通过且零写入；256 行 smoke 构建通过；全量产物的 `--resume` schema/行数/SHA256 复用通过；当前 QG 回归为 43 个合成单测，Ruff、compileall、代码隔离和 Manifest 均通过。
- 复用：Query hash shard、分片 writer、有界聚合、Parquet 审计和 resume 框架从根实现复制/适配；QG 运行时无 `poi_gr` import，来源与目标 SHA256 登记在 `qg_prqk/configs/code_provenance.yaml`。
- 阻塞项：None。

### P3（v1 历史）— Query embedding 与 Adapter 代码

- 状态：CODE_ONLY
- 开始时间：2026-08-31
- 修改文件：`qg_prqk/src/qg_prqk/{query_embedding,query_embedding_cli,hard_negative_mining,query_adapter,adapter_training,adapter_training_cli}.py`、`qg_prqk/scripts/{cache_query_embeddings,smoke_query_adapter,train_query_adapter}.py`、P3 配置、来源登记、合成测试、README 与本状态文档。
- 实现：复用冻结 BGE-M3 的无 instruction、right padding、max length 128 编码口径，Query embedding 以 `row_index=query_id` 写入可 resume 的 float16 NPY；POI BGE 仍只读。
- 负例：训练 batch 内动态 POI logits、目标 POI 语义近邻、`displayname/alias/category/category_code/address` lexical-metadata 与基于类目+距离的 local-geo 四源；`area/layer/click_score` 不使用，目标、重复 batch target 和 P2 合理多 POI 全部强制 mask。
- Adapter：`LayerNorm→Linear(1024,64)→GELU→Dropout(0.05)→Linear(64,1024)→residual→L2 normalize`；输出层零初始化，从 identity 开始；AdamW、weighted InfoNCE、梯度剪切、Train Query ID 稳定 5% dev 和 raw/adapted Recall@1/10/50+margin 已实现。
- 训练输入契约：无 pickle NPZ 必须显式携带 `qg-prqk-adapter-training-data-v1`，并包含 raw/positive/negative embedding、candidate/target POI row、CSR false-negative row、`w_qp` 和 Query ID。训练输入与输出都强制在 `qg_prqk/outputs/`。
- 合成验收：48 条、16 维旋转合成 Query 在 CPU 上训练 60 epoch，并启用动态 in-batch 负例；raw Recall@1=`0.000`、adapted Recall@1=`0.875`，mean hard margin 从 `-0.365645` 升至 `0.240613`，loss 从 `11.405870` 降至 `0.002337`。该结果只验证可训性，不是正式实验。
- 输出：本步未生成真实 P3 数据产物，未启动 BGE/GPU，未读取 Valid/Test；合成单测只在 `qg_prqk/outputs/` 临时目录写入并自动清理。
- 待完成：v2.1-CAT P2.5-CAT 完成后进入 P3A；先实现真实 bundle builder并运行最多 50,000 条 D3 Gate，再根据 overall 与困难子集指标及用户确认决定 Adapter full 或 identity。当前不得把本阶段写成 Gate/full。

### P2.5-CAT（历史全库）— 类别 Query 粒度标注

- 状态：COMPLETED（2026-09-02）；这是既有 Train-only 统计的全量数据后处理，不分配正式效果实验编号。
- 代码：`qg_prqk/src/qg_prqk/category_config.py`、`category_depth.py`、`category_depth_cli.py` 与 `qg_prqk/scripts/build_category_query_depth.py`；合成端到端测试位于 `qg_prqk/tests/test_category_depth.py`。
- 数据流：只读取冻结 P2 `manifest/query_stats/query_poi_pairs`、类别 index/vocab、BGE `poi_ids`/NPY header 和 POI `poi_id/category/category_code`；没有读取 P2 `query_shards/false_negative_mask`、原始 Train、Validation、Test 或 POI embedding values。
- 类别结果：mapping 2,337,178 行，402 fine / 19 coarse；缺失、非法 6 位 code、行序错位、fine→coarse 冲突均为 0，491,213 个 P2 target POI 全覆盖。
- Query 结果：1,372,661 个 Query 的 D0/D1/D2/D3 为 1,029,782（75.0209%）/25,639（1.8678%）/25,650（1.8686%）/291,590（21.2427%）；D3 数与覆盖 153,349 个 POI 均原样复用 P2。
- 图结果：保留 514,500 个 Query–POI 关系，展开成 1,204,570 条 Query–POI–layer 边；S1/S2/S3 覆盖 POI 为 210,965/188,570/153,349，概率和/边权和最大误差为 `5.7732e-15/5.5511e-15`。
- 敏感性：relaxed 的 D1/D2 为 33,127/38,527，baseline 为 25,639/25,650，strict 为 16,700/13,260；D3 始终固定 291,590，不使用 Valid/Test 选阈值。
- 权重：`support=log1p(min(query_count,20))/log1p(20)`；各层 reliability 再乘 `concentration^gamma*(1-normalized_entropy)`，所有值位于 `[0,1]`。
- Gate/验收：1,000 Query sample、50,000 Query medium、full、独立 `--validate-only` 和全套 45 个合成单测均通过；full 耗时 261.08 秒、峰值 RSS 471.57 MiB。
- 产物：`qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat/query_depth/`，manifest SHA256=`61cd7cf8e4bc783fe81bab9ce4ee87fcf4bab3274f25a036963931dbb811027a`，metrics SHA256=`c5ac962d365e384e2b7aaae7a1cc504ad9124d798f3119776a653f5a7f2025d5`。
- 边界：未训练 Adapter、未运行聚类、未开始 P3A，v1.1 P2 manifest SHA256 仍为 `25238e011dc0b162d4e27082f4bc918d1a2e64aa3ac0a46e415bee4ab07cd162`。

### active POI 数据修订与 P2.5-CAT（2026-09-04）

- 状态：COMPLETED；这是用户确认的候选库数据修订和既有 Train-only P2 后处理，不分配效果实验编号。
- active catalog：716,245 行，manifest SHA256=`dc13c3f137c57ba715f129ff2ccbbd8909d910da3cebc4a9fdbce2172f4d144a`，POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- QG active 资产：`qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/`；BGE shape=`[716245,1024]`/float16，BGE SHA256=`0f77fe64dce69e8d875228c747fcd742bb3852db20b657824e47e9d1a5b6d674`，source row SHA256=`4adc26b5f22318230fce3516d4a560b299e7dafbc8c75360d7f3b65fb7e49ca1`，asset manifest SHA256=`9baded48ef927795779743dedee459c46906f8fa1e711048777a19af99750862`；独立 validator 已逐值对齐历史全库 BGE/category 来源。
- 类别：所有 active 行均有合法类别；全局 402 类稳定 ID 中实际出现 397 类，缺少 `142411、261311、261410、261412、271029`，19 个粗类全部出现。没有补入非 active POI，也没有重编号。
- active P2.5：配置 SHA256=`4c35982287cfd4c4ad7a52160bce6e8878f391405c11110ace037c4358ae7cae`；输出 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_depth/`，manifest SHA256=`5f203b94d567433f89ff25a7755516b211544b9be2c484ac28e2243ccd7ce55f`，metrics SHA256=`df1e7bae77fe8954618b48a658b9b31d14288ba7d8776342ff10be8414d8b7c8`。
- 结果：716,245 行 category mapping；1,372,661 个 Query；1,204,570 条层级边；D0/D1/D2/D3 仍为 1,029,782/25,639/25,650/291,590，证明只替换候选 POI 行空间，没有改变 Query 算法、阈值或监督分层。
- 资源与验收：sample/medium/full 均通过；full 耗时 271.46 秒、峰值 RSS 425.02 MiB；后续兼容性修正后再次 `--validate-only` 通过。源访问证据保持原始 Train/Validation/Test 均未读取，active 候选成员资格只来自冻结 catalog。

### P3A — Exact Adapter 审计与补全

- 状态：GATE_COMPLETED（2026-09-03）；正式实验编号 `EXP-20260903-01`，只完成 50,000 条 D3 Gate，不是 full。
- 目标与假设：验证冻结 POI BGE 条件下，D3-only Query residual Adapter 能否在 Train-only internal dev 的 overall 与 raw Recall@1 未命中的困难子集上同时优于 raw BGE；若任一子集无提升则回退 identity。
- 数据版本：P2 manifest SHA256=`25238e011dc0b162d4e27082f4bc918d1a2e64aa3ac0a46e415bee4ab07cd162`；P2.5 manifest SHA256=`61cd7cf8e4bc783fe81bab9ce4ee87fcf4bab3274f25a036963931dbb811027a`；从 291,590 条 D3 按 Query ID 的 BLAKE2b 最小 hash 确定性选择 50,000 条，47,500/2,500 Train/internal-dev；未读取 Validation/Test。
- 代码与配置：新增 `p3a_config.py`、`p3a_data.py`、`p3a_ann.py`、`p3a_gate.py`、`p3a_cli.py` 和 `scripts/run_p3a_adapter_gate.py`；配置 `qg_prqk_p3a_1024x3_v1.yaml` SHA256=`f89cc3d38ed1c2f9f068db0b53244d20102bda7093f86056fb25d1ff4b839ebd`。工作树基于 commit `54802e6674e283722ecee00fb862530df30cef9a`，原有用户 dirty worktree 保留，未 commit。
- Query 与 ANN：BGE-M3 无 instruction、right padding、max length 128、BF16 推理、float16 cache；50,000×1,024。POI 2,337,178×1,024 使用 GPU `IndexFlatIP` 精确 Top-100，add/search 均显式 float32 L2 归一化，不使用近似索引。
- 负例与字段：每 Query 固定 6 个 semantic ANN、4 个 lexical-metadata、6 个 local-geo，另加动态 in-batch；总固定负例 800,000 条。实际读取 `poi_id/displayname/alias/category/category_code/address/lat/lng`，明确不读取 `area/layer/click_score`；目标、重复 batch target 和 50,183 条 P2 false-negative row 均 mask。
- 训练：`LayerNorm→1024→64→1024→residual→L2 normalize`，AdamW、LR `1e-3`、batch 2,048、BF16、3 epoch；loss=`2.320234/2.157093/2.113451`，Torch 峰值显存 4,180,077,568 bytes。
- 正式指标：overall 2,500 条上 raw→adapted Recall@1=`37.60%→45.52%`、Recall@10=`66.00%→71.24%`、Recall@50=`77.08%→80.28%`、Recall@100=`80.08%→83.12%`、mean hard margin=`-0.032193→-0.021226`；困难子集 1,560 条上 Recall@1=`0.00%→16.60%`、Recall@10=`45.51%→54.10%`、Recall@50=`63.27%→68.40%`、Recall@100=`68.08%→72.95%`、margin=`-0.078656→-0.072205`。头部 443 条和尾部 2,057 条的 Recall@10 也分别为 `77.20%→84.20%`、`63.59%→68.45%`。
- Gate 结论：overall 与困难子集均满足 Recall@10 或 margin 改善且无退化，严格 Gate 通过；50k 产物选择 `query_view=adapter`。该结论仅冻结 Gate 选择，不代表 291,590 条 full 已完成。
- 命令：`PYTHONPATH=qg_prqk/src /ofs/map_search/hudan/envs/poi-gr/bin/python qg_prqk/scripts/run_p3a_adapter_gate.py --config qg_prqk/configs/qg_prqk_p3a_1024x3_v1.yaml --gate medium`；随后使用同命令追加 `--validate-only` 独立复核。
- 环境与耗时：单卡 NVIDIA RTX A6000 48 GB；torch 2.9.1+cu128、FAISS 1.8.0、sentence-transformers 5.1.2；总耗时 1,112.43 秒。Query 编码 214.02 秒、POI exact index 101.23 秒、负例与 bundle 373.49 秒、Adapter 训练 56.37 秒。
- 产物：`qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat/query_adapter_exact/`，约 2.1 GiB，含 D3 selection、raw/adapted Query embedding、三组 Top-100、1.85 GB 训练 bundle、540,629-byte checkpoint、评测、日志、manifest 与 `_SUCCESS`；manifest SHA256=`12a29f1ff90d16f619231da18b99916da0afc7a2dfa4d138a18b9bb4bac96c2f`，独立 validator 通过。
- 样本验收：1,000 条 sample 最终链路成功，manifest SHA256=`57dde37abe65b55302dec3aa149f254056e1d453815742e5689abed3b90ec936`。此前 5 次样本开发运行分别因 Query float16 norm 容差、POI float16 norm 容差、FAISS 延迟导入、旧 lexical 路径过慢和 rank1 target 评测边界而失败/中止，均保留在 `p3a_sample/query_adapter_exact_*_20260903/`，不是正式实验结果；对应问题已由显式 float32 归一化、模块级 FAISS 导入、词法特征预计算、顺序向量 gather 和 rank1 回归测试修复。
- 当时下一步：等待用户确认 full 或 P4-CAT。该确认点已由用户选择 full，随后又确认把候选库迁移到 active POI；旧 Gate 本身仍只读保留。

### EXP-20260904-04 — active P3A-FULL 首次运行（中断）

- 编号修正：2026-09-12 文档整理时发现原记录误用了仓库已分配给 active MMBERT 实验的当日 03 号，故改为当日未占用的 04 号；运行目录、配置、哈希、时间和结论均未改变。

- 目标：按已确认的 20,000 条 Train-only holdout、SELECT 3 epoch 和固定 best epoch FINAL 协议训练 active D3 Adapter。
- 数据与配置：active 716,245 POI，D3 291,590 Query；配置 `qg_prqk/configs/qg_prqk_p3a_full_active_1024x3_v1.yaml`，SHA256=`4e5c9f4d2a7dd5069b5b2ff3497b1a834a0f00e4e494bf1617b1ba49ce582ba6`；P2/P2.5/Gate 来源哈希见冻结配置和 holdout manifest。
- 命令：`PYTHONPATH=qg_prqk/src /ofs/map_search/hudan/envs/poi-gr/bin/python qg_prqk/scripts/run_p3a_full.py --config qg_prqk/configs/qg_prqk_p3a_full_active_1024x3_v1.yaml`。
- 环境：开发机单卡 RTX A6000，poi-gr 环境，OrangeFS；基于 commit `54802e6674e283722ecee00fb862530df30cef9a` 的 dirty worktree，未 commit/push。
- 实测状态：2026-09-04 15:17 启动，18:11:51 最后记录编码完成 163,840/291,590 行，18:16:11 最后修改 NPY；2026-09-05 宿主机确认原 launcher/Python 均不存在、GPU 已释放。没有 run.exit、异常堆栈或 OOM 证据，终止原因未能确定，不能断言是连接、容器回收或 OOM。
- 已完成：冻结 20,000 holdout、271,590 SELECT 训练行、全量 D3 selection。尚未开始 SELECT/FINAL 模型更新，没有效果指标或 checkpoint。
- 归档：原产物完整保留在 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_adapter_exact_full_interrupted_20260904/`；原 PID/log 保留于 `qg_prqk/outputs/run_control/p3a_full_active/`；修复前源码、配置、测试和文档快照位于其中 `attempt01_source_snapshot/`。历史 50k Gate 保持不变。
- 恢复结论：原 NPY 预分配了完整 shape，尾部含零值和未确认块，文件大小不代表编码完成。只允许从日志确认的完整块中逐行验证后复制前缀；新增编码采用小块顺序写入、hash 与完成标记，最后顺序合并并原子发布。

### P3A-FULL Query cache 恢复（2026-09-05）

- 状态：COMPLETED_AND_VALIDATED；属于 `EXP-20260904-04` 的工程恢复步骤，未运行 Adapter 训练，不另报效果实验。
- 实现：`src/qg_prqk/p3a_query_cache.py` 复用 QG 内冻结 BGE loader、float32 L2 normalization 和 D3 数据契约；每 8,192 行使用缓冲顺序写入，fsync 后以 hash/进度提交。中断后已提交块可复用，未提交块不采纳；最后顺序合并、原子发布完整 NPY。每次最多保留一个块，无新增全量内存缓存。
- 输入：旧归档 D3 selection 的 SHA256=`15d2fa1978106ffa0809e01ad08a6a333b1613f7963f9549da8f809976577582`，与当前 Train D3 的 Query ID/文本/顺序逐行匹配；旧 partial NPY SHA256=`b05caef1140150809cbd31b207d7fa02ec8cd4a166f2891846d69c1ee58c9799`，日志确认完成前 163,840 行。
- 命令：原 FULL 命令增加 `--prepare-query-cache-only --recover-query-prefix-from qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_adapter_exact_full_interrupted_20260904`。环境额外显式设置 poi-gr `LD_LIBRARY_PATH`、QG 内短 `TMPDIR` 和离线模型加载；完整最小命令见 `qg_prqk/docs/QG_PRQK_CODEX_COMMANDS.md`。
- 结果：18:59:26 cache 发布完成，退出码 0、`_SUCCESS` 齐全。恢复 163,840 行，新编码 127,750 行，最终 291,590×1,024 float16；范数 min/mean/max=`0.99975228/0.99999975/1.00024736`。
- 性能：不含上游全量输入复核和旧文件来源哈希的 cache 构建耗时 232.20 秒；新编码的 16 个 buffer 推理合计 29.54 秒，36 块写入/提交合计 7.15 秒。该计时包括哪些步骤已明确分开，不把全部端到端耗时称为纯推理时间。
- 产物：`qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_cache_exact_full/`；`embeddings.npy` SHA256=`9a37d6074d0f20e792992f2d64c446c3f807a18c8437f3600c8ad39fccf4249d`；manifest SHA256=`a6e27c086f5ad6d13e771b03b40568062a873c2389df5c38a7e5e75027772bfc`。运行控制和源码快照在 `qg_prqk/outputs/run_control/p3a_full_active/attempt02_query_cache/`。
- 验证：独立重新从 P2.5 选取 D3，复核所有 chunk/hash/配置和完整 NPY；全量分块逐值对齐，恢复前缀 163,840 行与旧文件 bit-exact，未采纳旧文件未提交尾部。重复读取完成 cache 未加载 BGE。71 个合成测试、Ruff、compileall 通过；隔离测试只检查当前 src/scripts/tests，归档源码不作为执行入口。
- 初始化澄清：Gate 在 `p3a_gate.py::_train_adapter` 中创建 Adapter，随后才在 `query_adapter.py::fit_query_adapter` 中设置 seed，且没有保存 Gate 初始 state。故只能证明 FULL 的 SELECT/FINAL 共享同一新初始 state，不能证明它与历史 Gate 的随机 down-layer 权重逐值相同。cache 恢复轮因此未启动模型更新；2026-09-05 用户随后明确确认同结构/同 identity 初始化方式、建模前 seed=42 的新初始 state，SELECT/FINAL 逐值共用。
- 当时下一步：按已确认初始化口径执行正式 FULL；该任务已由下述 `EXP-20260905-02` 完成。其余 BGE/负例/mask/loss/optimizer/hyperparameters 保持冻结。

### EXP-20260905-02 — active P3A-FULL 共同新初始化正式训练

- 状态：COMPLETED_AND_VALIDATED；2026-09-05 21:51:51 CST 启动后台 wrapper PID 1426501 / Python PID 1426503，2026-09-06 01:54:49 发布最终 manifest，01:55 wrapper 退出码 0，总墙钟约 4 小时 3 分钟。2026-09-06 独立 `--validate-only` 退出码 0，随后确认原进程均已退出、GPU 已释放；72 个合成测试、Ruff、compileall 与差异检查通过。
- 目标与假设：在固定 active 闭集和未进入历史 Gate 的 Train-only 20k holdout 上，检验扩大 D3 训练规模后的 Query Adapter；不从历史 Gate 或 SELECT checkpoint 续训 FINAL。
- 数据：716,245 POI，291,590 D3；SELECT 271,590、holdout 20,000、FINAL 291,590。P2/P2.5/Gate 哈希沿用冻结配置，完整 D3 cache manifest SHA256=`a6e27c086f5ad6d13e771b03b40568062a873c2389df5c38a7e5e75027772bfc`，向量 SHA256=`9a37d6074d0f20e792992f2d64c446c3f807a18c8437f3600c8ad39fccf4249d`；直接复用缓存，不重新编码。
- 初始化：建模前 seed=42，同结构 identity 新初始 state，SELECT/FINAL 逐值重载同一文件；初始 checkpoint 和总 manifest 显式声明不复现历史 Gate 未保存的初始随机权重。其余模型、负例、mask、loss、AdamW 和超参数未改。
- 配置与命令：`LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib PYTHONPATH=qg_prqk/src /ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/scripts/run_p3a_full.py --config qg_prqk/configs/qg_prqk_p3a_full_active_1024x3_v1.yaml`；配置 SHA256=`4e5c9f4d2a7dd5069b5b2ff3497b1a834a0f00e4e494bf1617b1ba49ce582ba6`。固定 SELECT max_epoch=3，按 R@10/R@1/困难 R@10 依次选模，FINAL 固定 best_epoch。
- 环境与工作树：开发机单卡 NVIDIA RTX A6000、poi-gr、OrangeFS；项目内 `TMPDIR=qg_prqk/outputs/tmp/p3f`，离线模型加载；commit `54802e6674e283722ecee00fb862530df30cef9a` 的 dirty worktree，未 commit/push。
- 产物与控制：输出 `_active/query_adapter_exact_full/`；完整路径前缀为 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/`。PID/log/exit 与源代码快照位于 `qg_prqk/outputs/run_control/p3a_full_active/attempt03_full/`；原中断目录、cache 和历史 Gate 均保留。
- 中间产物：21:58 全量 Query cache 复核并复用；716,245 POI 精确索引构建耗时 87.25 秒；22:01 完成 Raw/Gate 对比，23:33 完成全量训练 bundle。每条 Query 固定 16 个负例，semantic/lexical/local-geo 共 `1,749,540/1,166,360/1,749,540` 个；D3 false-negative CSR 共 292,767 条，不能与全 P2 的 557,639 条混写。
- 验收：`--validate-only` 复核全量 bundle 各 NPY、顶层产物 hash、20k holdout 与 Gate 无交集、SELECT 不更新 holdout、best epoch、SELECT/FINAL 共享初始化及 Train-only/D1/D2 边界。另逐项重算 holdout/select/final 子 manifest 登记的 9 个文件和 3 个 epoch checkpoint 共 12 项哈希，均匹配；当前训练源码与启动时 `source_snapshot/src` 无差异。验证日志和退出码位于同一 run-control 下的 `validate_console.log`、`validate.exit`、`review_tests.log` 和 `review_tests.exit`。

同一 Train-only 20,000 Query、716,245 active POI 精确检索对比（除 drift 外均为百分数）：

| Query view | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | 困难 Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw BGE | 48.595 | 68.650 | 73.985 | 78.000 | 57.1702 | 49.3921 |
| 历史 50k Gate | 53.350 | 71.785 | 76.390 | 80.070 | 61.3229 | 54.1873 |
| FULL-SELECT epoch 2 | 53.955 | 72.460 | 77.100 | 80.700 | 61.9000 | 55.5393 |

困难子集固定为 Raw BGE Recall@1 未命中的 10,281 条，所有模型使用相同 false-negative mask。cosine drift 定义为 `1 - cosine(raw, view)`：

| Query view | mean | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Raw BGE | 3.7339e-8 | 0 | 1.1921e-7 | 2.9802e-7 | 8.3447e-7 |
| 历史 50k Gate | 0.252184 | 0.240677 | 0.352099 | 0.470531 | 0.634698 |
| FULL-SELECT epoch 2 | 0.240532 | 0.228514 | 0.322124 | 0.474640 | 0.681234 |

SELECT 三轮完整记录（同一个 holdout，召回/MRR 均为百分数）：

| epoch | Train loss | R@1 | R@5 | R@10 | R@20 | MRR@10 | 困难 R@10 | drift mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.060136 | 53.395 | 72.000 | 76.780 | 80.455 | 61.4114 | 54.9266 | 0.241341 |
| 2（选中） | 1.981812 | 53.955 | 72.460 | 77.100 | 80.700 | 61.9000 | 55.5393 | 0.240532 |
| 3 | 1.964936 | 53.775 | 72.215 | 76.915 | 80.545 | 61.7306 | 55.1989 | 0.250475 |

epoch 2 的主指标 R@10 最高，无需触发 tie-break。各轮 drift 的完整分位数及精确 Top-100 数组保留在 `select/epochs/epoch_0{1,2,3}/`，统一三视图记录为 `comparison/three_view_comparison.json`。

- FINAL：从共同初始 state 使用全部 291,590 D3 固定重训 2 epoch，不从 SELECT/Gate checkpoint 续训，也不再次选模。两轮 loss 为 `2.057661/1.980304`；全 D3 drift mean/p50/p90/p99/max 为 `0.252635/0.241031/0.338553/0.493687/0.752368`。FINAL 已包含原 holdout，因此不将 FINAL 在该集合上的检索当作独立验证结果；上表只报告 FULL-SELECT。
- 核心指纹：顶层 manifest=`63f6d82126ec80a9fb1208496c04eb1732ccd27d17ce7551f61b7f9972c92f0e`；共同 initial state=`ee6d28a22c3f1a4bfb1a17a91fffa972a100e2b1176da7abe64f237265622ece`；SELECT best checkpoint=`98bd5865bb16b3279ce93503aac484011e65b303c139403c5f10280d4cb5e286`；FINAL checkpoint=`dcbbebf9099baf2bb21411855749da51943cd4be1f0456abfc1c926905027308`；FINAL manifest=`5254243750a06cf46b60657007894f06f933219c524401e4b28ffc97fc9165cc`。
- 最终交付：`final/query_adapter_exact_final.pt`、`final/config.yaml`、`final/config_resolved.json`、`final/metrics.json`、`final/manifest.json` 和 `_SUCCESS` 均完整。顶层/holdout/select/final manifest 保留全部来源链，历史 50k Gate 与旧中断产物未覆盖。
- 耗时与限制：顶层 runtime 为 14,477.25 秒（不含进入 run_log 前的 Gate 复核），bundle 准备 5,501.88 秒。bundle 完成至 SELECT 首轮结束约 106.52 分钟，含训练前准备；第 2/3 轮从上一轮评测结束到训练结束分别为 1,493.07/78.30 秒；FINAL 两轮及 drift 共 227.53 秒。23:48/23:54 曾观察到进程 I/O 等待、GPU 0% 和持续磁盘读取，后续速度明显恢复，与缓存逐渐就绪的现象一致，但未做存储/缓存对照，不能断言唯一原因。本次保持原 mmap 读取实现，没有中止、换盘或重训。
- 结论：FULL-SELECT 相对 Raw 的 R@1/R@10/困难 R@10 提高 `5.360/3.115/6.1473` 个百分点，相对历史 Gate 提高 `0.605/0.710/1.3520` 个百分点。只有单 seed 和 Train-derived internal holdout，且新旧训练规模、候选宇宙与随机初始权重不完全相同，不能把差值完全归因于扩量，也不声称业务 Validation/Test 或 SID 已提升。
- 当时停止点：`HOLD_FOR_P3A_FULL_REVIEW`。FINAL 仅标记为 D3 Exact Query 默认 Query view；未来 P4 第一版 D1/D2 保持 Raw BGE、D3 使用 FINAL，本轮未验证或修改 D1/D2。该实验结束时 P4-CAT、RQ-KMeans、SID、Final PID、SFT 与外部基线均未启动。

### P4-CAT — 类别层级 Query 图与 Query embedding

#### 1,000-query sample（历史冻结）

- 状态：SAMPLE_COMPLETED_AND_VALIDATED（2026-09-07），停止于 `HOLD_FOR_P4_SAMPLE_REVIEW`；本次仅做工程 sample，不分配正式全量实验编号。
- 目标与数据：验证冻结 active P2.5 图到分层 Query view 的可运行链路。按 P2.5 全局 query_id 顺序取前 1,000 条 D1/D2/D3，跳过 D0；这是确定性工程前缀样本，不是随机或检索评测样本。候选库仍为全部 716,245 条 active POI。
- 代码：新增 `p4_data.py`、`p4_query_graph.py`、`p4_cli.py` 和 `scripts/build_query_graph.py`，新增两份合成测试；不修改冻结上游源码/YAML/checkpoint。输入字段与物理产物合同见方法规范 §6.5。
- 工作树：HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户改动保留；本次未 commit/push，实际运行源码、配置、测试另存不可覆盖快照，并由 sample manifest 锁定源码哈希。
- 实测图结果与 dry-run 一致：1,000 个 active Query 中 D1/D2/D3 为 95/73/832；S1/S2/S3 边为 1,495/1,115/832，共 3,442 条，覆盖 1,483 个 active POI；FN 1,124 条。最大条件概率和/最终权重和误差均为 `4.440892098500626e-16`。未读取原始 Train、业务 Validation/Test 或 POI embedding 数值。
- 测试：11 项 P4 合成测试与全部 92 项回归通过（33.493 秒）；覆盖越层/重复边、权重守恒、D3 文本错位/次要强边、FN/POI 对齐、源文件损坏、输出损坏、中断恢复、已完成 resume 不加载模型、拒绝 >1000/full、FINAL 角色/epoch/来源。首次测试未设置 poi-gr `LD_LIBRARY_PATH` 导致 FAISS 的 C++ runtime 导入失败，按既有 P3 环境补齐后通过，未改动算法或上游代码。
- 运行：单卡 RTX A6000，poi-gr Python；`LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib`，`PYTHONPATH=qg_prqk/src`，`HF_HOME=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/cache/huggingface`，`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`TOKENIZERS_PARALLELISM=false`，`OMP_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=8`，短 `TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p4s`；模型在 CUDA 上执行，无 CPU 编码降级。
- 命令：`/ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/scripts/build_query_graph.py --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml --limit 1000`，环境变量如上。BGE 只新编码 D1/D2 共 168 条，D3 832 条只复用 Raw 并应用 FINAL，不训练。
- 产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/sample_001000/`；运行控制及完整 src/scripts/configs/tests 快照位于 `qg_prqk/outputs/run_control/p4_cat_active_512/sample_1000_attempt01/`，PID/log/exit 为 `run.pid/console.log/run.exit`。上游目录保持只读。
- 完成时间：2026-09-07 12:43:31 CST，`run.exit=0`。builder 阶段耗时 `136.601` 秒，不包含此前输入准备与导入；D3 cache 校验约 10.28 秒，BGE 加载阶段（含相关导入）约 120.23 秒，四块 view 计算约数秒，不能据此线性推算全量耗时。
- view 结果：Raw 与最终 Query view 均为 `[1000,1024] float16`，按 256/256/256/232 行成对提交并顺序合并；D1/D2 Raw 与 view 逐值相同，D3 Raw 与冻结缓存逐值相同。D3 Adapted 相对 Raw 的 cosine drift 均值 `0.2588972747`、最大 `0.6666470766`；D1/D2 约 `1e-8` 的数值 drift 仅为浮点余弦计算误差，不表示向量发生改变。
- 独立验收：另起进程 `--validate-only` 与完成目录 `--resume` 均 `status=validated`、退出码 0；对应 `validate_console.log/validate.exit` 与 `resume_console.log/resume.exit`。完成目录 resume 未加载 BGE/Adapter，manifest 哈希未改变。`--help`、compileall、Ruff 也通过，运行快照与当前 src/configs 一致。
- 来源：62 个输入文件锁定于 manifest；input signature=`4e28eb590f4e9ebe7d4531694a9c23ee2880572801059cfcdcff919dfd585901`，512 配置 signature=`b4bc8a786841faf3dcedff375f4e7390efbafec54df14f27a2c3b540e4ed42c2`；sample manifest SHA256=`20dfdbd7a49159236694afbd9afee68261b67b37ec52e0326f696464315cb2c6`，`_SUCCESS` 绑定该哈希，节点/边/FN/分块/合并 NPY 的完整哈希均在 manifest 内。
- 当时边界与下一步：medium/full 未实现或运行，无 512 SID；sample 只证明图和 view 数据链路，不证明检索或生成收益。该停止点已履行，用户随后确认 medium。旧 sample 及 v1 快照、所有哈希保持不变，新 v2 通过显式固定 manifest 哈希复用，不将 sample 重标为 full。

#### 50,000-query medium（2026-09-07）

- 状态：MEDIUM_COMPLETED_AND_VALIDATED；50k 构建、独立 validator 与完成目录 resume 均通过，退出码均为 0，停止于 `HOLD_FOR_P4_MEDIUM_REVIEW`。用户在 sample 交付后确认继续，本步仅扩展 medium，不是全量图或聚类。
- 目标与合同：同序 active Query 前缀，沿用全部模型与算法参数；只扩展规模、分块写入/校验和严格 sample 复用。v2 合同见方法规范 §6.6。当前仍禁止 full。
- dry-run 已通过：D1/D2/D3=`3,808/3,730/42,462`，S1/S2/S3 边=`75,714/58,551/42,462`，总边 `176,727`，FN `54,982`，覆盖 POI `56,723`；概率和/权重和最大误差为 `3.3306690738754696e-15/4.773959005888173e-15`，173 个来源文件。未读取原始订单或业务 Validation/Test，也未写图/向量输出。
- 代码与测试：修改现有 P4 数据、构建器和 CLI，新增 `test_p4_medium.py`；18 项 P4 合成测试通过，含 >1000 节点、v1 sample 复用、来源/前缀损坏、中断恢复、完成 resume 与 50k 上限。全部 99 项回归通过（42.099 秒），Ruff、compileall 与 `--help` 通过。旧 sample 已用冻结快照再次 `--validate-only` 通过，manifest 哈希不变。
- 资源安排：单卡 A6000，BGE/Adapter batch 256，缓存块 8192；float16 D3 工作矩阵约 97.66 MiB，两份完整输出加分块约 390.63 MiB（不含图/manifest）。不增加 GPU batch，启动前宿主 GPU 空闲 46,080 MiB；图元数据在 50k 上限内处理，实际 RSS 见下方运行记录。
- 输出/控制：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/medium_050000/`；`qg_prqk/outputs/run_control/p4_cat_active_512/medium_50000_attempt01/`。日志、源码/配置/测试快照和退出码均留在后者；旧 sample 和所有上游产物不覆盖。
- 实际运行：wrapper PID `1696832`，14:09:18 CST 启动，14:13:15 发布 manifest，14:13:47 `run.exit=0`，含输入准备与内联验收共约 4 分 29 秒。builder 至 manifest 记录为 `179.511` 秒，发布前进程峰值 RSS `4,239.82 MiB`；BGE 加载阶段（含依赖导入）约 133.90 秒，模型就绪到七块全部提交约 8.11 秒。没有 OOM/重试或 CPU 编码降级；不将本次缓存条件下的耗时线性外推到 full。
- 真实图结果与上述 dry-run 一致。两份完整矩阵均为 `[50000,1024] float16`，缓存块为六块 8192 加末块 848。D1/D2 view 与 Raw 逐值相同、D3 Raw 与冻结 D3 cache 逐值相同，前 1000 行的 Raw/view 均逐值复用旧 sample；实际新编码 D1/D2 `7,370` 条，D3 不重编码。D3 cosine drift 均值 `0.2529901862`、最大 `0.7485306263`，只是表示诊断，不是召回结果。
- 来源指纹：medium manifest SHA256=`e9a5411167a2d39b685f61c5fde85ccff64a74905e3b68f26f144d45551140db`，input signature=`86815751ffb724cceac74e0e01d4e87007c3afaedb596b1024cd11c734bdcf1f`，config signature 仍为 `b4bc8a786841faf3dcedff375f4e7390efbafec54df14f27a2c3b540e4ed42c2`；173 个源文件、所有输出/缓存块哈希与旧 sample 的冻结 manifest SHA256 均写入新 manifest。旧 sample 哈希仍为 `20dfdbd7a49159236694afbd9afee68261b67b37ec52e0326f696464315cb2c6`。
- 命令：`/ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/scripts/build_query_graph.py --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml --gate medium --limit 50000 --reuse-sample qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/sample_001000/manifest.json --reuse-sample-sha256 20dfdbd7a49159236694afbd9afee68261b67b37ec52e0326f696464315cb2c6`；环境沿用 sample，仅短 TMPDIR 改为 `/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p4m`，并显式 `CUDA_VISIBLE_DEVICES=0`。构建时 `PYTHONDONTWRITEBYTECODE=1`；复现命令也见 `docs/QG_PRQK_CODEX_COMMANDS.md`。
- 工作树：HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户改动保留；未 commit/push。运行源码与 `source_snapshot/src` 一致，全部 YAML 与 sample 冻结配置一致；代码和产物不越出 QG 目录。medium 属于工程规模 Gate，不创建新的方法报告或正式全量实验编号。
- 独立验收：新进程 `--gate medium --limit 50000 --validate-only` 及原构建命令追加 `--resume` 均 `status=validated`、退出码 0。日志/退出码为同一控制目录下 `validate_console.log/validate.exit` 与 `resume_console.log/resume.exit`；完成 resume 未加载 BGE/Adapter，旧 sample 与 medium manifest 哈希均保持不变。
- 结论与限制：50k 数据流、图权重守恒、分层路由、不可变前缀复用及恢复合同通过工程验收，不代表检索收益。当时未生成 512 SID、未更新 Adapter，未读取业务 Validation/Test。该历史停止点已履行，用户随后确认 full。

#### 342,879-query full（2026-09-07）

- 状态：FULL_COMPLETED_AND_VALIDATED；真实构建、入口内联 validator、另起进程 `--validate-only` 与屏蔽 GPU 的完成态 `--resume` 均退出 0，停止于 `HOLD_FOR_P4_FULL_REVIEW`。该阶段是 P4 数据/view 工程验收，不单独分配正式方法实验编号。
- 目标与数据：在不改变 sample/medium 已验收算法的前提下，扩展到 active P2.5 全部 342,879 个 D1/D2/D3 Query。数据版本仍为 716,245 行 active POI、active P2.5 manifest SHA256=`5f203b94d567433f89ff25a7755516b211544b9be2c484ac28e2243ccd7ce55f`、P3A-FULL-FINAL；不读取原始订单或业务 Validation/Test。
- 实现合同：CLI 只接受固定 `full=342879`，不允许用任意 limit 冒充 full。构建/恢复必须显式锁定 medium manifest 及 SHA256；D3 Raw 通过已验冻结 NPY 的 mmap 按 8,192 行区间取数，全量来源签名按 Arrow RecordBatch 增量更新，图守恒指标用 Arrow 列聚合复算。D1/D2 Raw BGE、D3 FINAL Adapter、512×3、边权、FN 与其他算法参数不变。详见方法规范 §6.7。
- dry-run：D1/D2/D3=`25,639/25,650/291,590`；S1/S2/S3 边=`514,500/398,480/291,590`，总边 `1,204,570`，FN=`376,701`，覆盖边上 POI=`210,965`，来源文件 827 个。用时 2:25.03，峰值 RSS `4,759,404 KiB`，无 swap；不写 full 产物、不加载模型。
- 测试：新增 full→medium 冻结前缀、D3 mmap、固定 full limit、独立 validator 与完成 resume 的合成覆盖；19 项 P4 测试通过，全部 100 项 QG 回归通过（61.75 秒），Ruff、compileall 与 `--help` 通过。50k medium 在 full 启动前用 v2 冻结快照再次独立校验，哈希仍为 `e9a5411167a2d39b685f61c5fde85ccff64a74905e3b68f26f144d45551140db`。
- 运行与环境：poi-gr Python、单卡 RTX A6000 GPU 0，启动前空闲显存 46,080 MiB；`LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib`、`CUDA_VISIBLE_DEVICES=0`、`PYTHONDONTWRITEBYTECODE=1`、短 `TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p4f`。attempt02 于 14:53:23 CST 启动，15:01:54 发布 manifest，15:02:58 `run.exit=0`；入口总墙钟 9:34.85，builder 记录 318.597 秒，外层峰值 RSS `7,413,920 KiB`，无 OOM/Swap。BGE 自报加载 94.54 秒，OrangeFS 读盘等待是主要延迟。
- 真实结果：与 dry-run 图指标一致，条件概率和/最终边权和最大误差=`5.551115123125783e-15/5.10702591327572e-15`。严格复用 medium 前 50,000 行 Raw/view，仅新编码其余 D1/D2=`43,751`；D3 Raw 不重编码。D1/D2 Raw=view 逐值相同；D3 Adapted 相对 Raw cosine drift 均值 `0.2526353002`、最大 `0.7523645163`。这是表示路由诊断，不是检索或 SID 收益。
- 产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/full_342879/`，共 93 个顶层文件、约 2.7 GiB；42 对 8,192 行（末块 7,007）Raw/view 原子块，完整 Raw/view 均为 `[342879,1024] float16`。六项主 artifact 合计 `1,442,441,316` bytes，所有分块另在 progress/manifest 锁定。
- 来源指纹：full manifest SHA256=`f6f5a6b4a6f5ef78ff8eb1007e8b566a17fe5a911aae215185b0ed156ab6329d`，input signature=`7054920560a35bb7c4308041b938d65f6c6aa127b4fb25b0c055823abfed7178`，config signature=`b4bc8a786841faf3dcedff375f4e7390efbafec54df14f27a2c3b540e4ed42c2`，冻结 medium SHA256 如上；827 个来源和全部输出哈希均在 manifest 内。
- 运行控制：成功记录位于 `qg_prqk/outputs/run_control/p4_cat_active_512/full_342879_attempt02/`，包含完整 src/scripts/configs/tests 快照及 76 项 SHA256、console/validate/resume 日志和退出码。独立 validator 用时 4:17.94、`validate.exit=0`；屏蔽 GPU 的 resume 用时 3:51.38、`resume.exit=0`，日志无 BGE/Adapter 加载事件。首次启动将冻结 YAML 从原目录搬到 run-control 后解析了错误的相对路径，2.23 秒内在输入读取前失败，未创建 full 输出、未加载模型；完整保留于 `full_342879_attempt01_config_path_error/`，attempt02 使用同哈希的 canonical 原位 YAML 解决，未改数据或算法。
- 工作树：HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户改动全部保留；未 commit/push。本次代码、配置、测试、文档和产物均在 `qg_prqk/` 下，不修改冻结 sample/medium 或上游 P2/P2.5/P3A 产物。
- 结论与下一步：P4 已形成 P5–P7 SID 构建所需的完整 Query 图和分层 Query view，因此已具备启动 SID 构建链的数据前提；但当前尚无 512 SID、POI-only PRQK 或检索收益。按最小阶段规则先等待用户核验，确认后只进入 P5-CAT POI-only PRQK 512×3 内部 A0，不一次跨到 P6/P7。

### P5-CAT — POI-only PRQK 512×3 内部 A0

- 状态：`FULL_COMPLETED_AND_VALIDATED`。旧参数 10k/100k、两次受控诊断、新 hard60+Top-k5 10k 和用户授权的 active POI full 均保留；full 正式记录为 `EXP-20260909-01`，其 `HOLD_FOR_P5_HARD60_TOPK5_FULL_REVIEW` 停止点已由用户确认进入 P6 准备。
- 实现：自包含的 `p5_data.py/p5_prqk.py/p5_cli.py` 与 `build_poi_prqk.py` 冻结 512×3、seed=42、POI-fitted 公共方向去除、spherical/cosine assignment、projection residual、hard 双停止及 Top-5/β=15/5 轮质心细化。默认 sample/100k/500k/full 顺序仍严格；新增的显式 direct-full 开关只落实 2026-09-09 用户授权，并在 full manifest 中绑定 sample SHA、记录跳过 100k/500k 和 `P5_A0_ONLY` 范围。
- 数据：公共方向使用全部 716,245 条 active POI；sample 只拟合确定性 10,000 行，selected rows SHA256=`207f7f4be3fd57b6f0cd990467f6191a8bdadc358a865d8710315a6fd73251fa`。未读取 Query 图、Category、Geo、业务 Validation/Test 或原始 Train。
- 结果：三层 final distortion=`0.447954/0.539825/0.584774`，hard 分别在 `16/15/16` 轮收敛；每层 512 code 全部激活，Kish ESS=`431.98/409.39/436.64`，Gini=`0.2413/0.2727/0.2230`。零 residual 行均为 0，最大正交误差 `2.50e-6`。
- sample SID：10,000 行中 distinct SID=9,934（99.34%），collision excess=66，涉及碰撞 POI=128，最大桶=4，桶大小 p99=1；这是 sample 桶诊断，不外推 full 指标。
- 产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/sample_010000/`，manifest SHA256=`c9d18df7d1029c9b80442965f31943a31cf4c3ef076de2aa6b9fd5b6d437f4e2`。成功运行、独立 validator、155 文件源码快照与退出码位于 `qg_prqk/outputs/run_control/p5_cat_active_512/sample_010000_attempt02/`，构建/validator 均退出 0。
- I/O 修正：attempt01 直接向 OrangeFS `memmap.flush()`，20 MiB residual 单次约 10–11 分钟，运行 22:26 后主动中止并完整保留（exit 143）；attempt02 改为本地短路径生成后顺序回写 QG 目录，算法/数据/seed/参数不变，完整构建与内联复核墙钟 1:16.06。
- 核验：启动前全量 105 项 QG 回归通过；I/O 修正后 compileall、Ruff、6 项 P5 测试和全量 106 项 QG 回归（57.53 秒）通过；独立 `--validate-only` 重验来源、artifact 哈希、精确 assignment 和 projection residual。完整实验记录见 `qg_prqk/docs/experiments/QG_PRQK.md`。
- 100k 数据与结果：确定性选中 100,000 行，selected rows SHA256=`c1696ac1f5fbe2db990e2adddcf4f0a9e40869212f670ba290a3bfacf46be215`。三层 512 code 全部激活，Kish ESS=`433.38/427.68/435.27`，Gini=`0.2367/0.2429/0.2296`，零 residual 行均为 0；distinct SID=96,240（96.24%），collision excess=3,760，最大桶=23。
- 100k 收敛风险：三层 hard fit 均达到 max_iter=30 而 `hard_converged=false`；第 30 轮 assignment change=`0.00135/0.00229/0.00187`。Top-k 后 hard distortion 相对第 30 轮从 `0.466948/0.555470/0.596859` 上升至 `0.474177/0.559833/0.600312`，五轮逐轮恶化，未通过预期的收敛与细化 Gate。
- 100k 产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/medium_100000/`，manifest SHA256=`8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0`；run-control 与 155 文件快照位于 `qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_attempt01/`。构建/独立 validator 均退出 0，完整 106 项回归通过。
- 100k I/O：不再使用系统临时目录或 `memmap.flush()`；residual 在 QG 输出目录分块顺序写入并原子改名。四个 `[100000,1024] float16` residual 均正常发布，外层构建与内联复核墙钟 1:14.14。
- 100k 受控诊断合同：从已冻结 `medium_100000` 重验完整来源/产物哈希和 `_SUCCESS`，只在内存中以同一 seed/选样/初始化运行 `hard30_topk_off` 与 `hard60_topk_off`；两分支输出只作诊断，不是 canonical checkpoint。未读取 Query/Category/Geo/原始 Train/业务 Validation/Test。
- hard endpoint 结果：hard30-off 三层 final distortion=`0.466934/0.557356/0.597651`，仍全部未收敛；hard60-off 在 `43/39/33` 轮按原双停止收敛，final distortion=`0.466731/0.556393/0.597649`。因此 `max_iter=30` 不足，但从 30 继续到收敛的误差改善只为 `0.000202/0.000963/0.000003`。
- Top-k 与码分布权衡：原 Top-k5 相对其自身 hard30 trace 的 distortion 三层均恶化，但它的 distinct SID=`96,240`，高于 hard30-off 的 `95,786` 和 hard60-off 的 `95,722`；collision excess 分别为 `3,760/4,214/4,278`，Top-k 的三层 Kish ESS 也更高。这表明 Top-k 存在真实的误差–均衡/碰撞权衡，不能仅凭 distortion 直接删除。
- 诊断产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/diagnostics/medium_100000_hard_endpoint_v1/`，manifest SHA256=`00bb291a4f58ed37788382ae7d5abd70ee9294a0749c41c27277afe82d8f1e29`，comparison SHA256=`0e33fd14d6a99e2e2c0a2bc6462cf65aa9bb05e67434aad702d8320b2418abda`。运行/独立 validator 均退出 0；源码快照清单 SHA256=`d59c499f7cf3f999416ce32aae2c4dff83b0d98af9793a6e1e573dfa8f0952ab`。compileall、Ruff、9 项 P5 测试和全部 109 项 QG 回归通过。
- hard60+Top-k5 配对结果：三层 hard 分别在 `43/40/35` 轮收敛；从同一次 hard endpoint 分叉 Top-k5 后，distortion 分别增加 `0.007237/0.004646/0.003592`（`1.551%/0.836%/0.603%`），但 Kish ESS 从 `401.23/392.27/410.87` 提高到 `432.72/427.77/435.17`，Gini 从 `0.2840/0.2881/0.2637` 降至 `0.2377/0.2449/0.2324`。
- 完整路径权衡：hard60+Top-k5 的 distinct SID=`96,171`、collision excess=`3,829`、碰撞 POI=`6,906`、最大桶=22；相对 hard60-off 的 `95,722/4,278/7,618/23`，增加 449 个唯一 SID、减少 449 个冗余碰撞和 712 个碰撞 POI。Top-k 是可复现的均衡正则收益，不是 distortion 收益。
- 新诊断产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/diagnostics/medium_100000_hard60_topk5_v1/`，manifest SHA256=`f33bf2573c141b2ae47bf5e578be3cbe0debd0aed03f23e834a9c982814dbd90`，comparison SHA256=`f43cd344170ec20249fc87a330dac5c66d131a71b8959be3011e7cd1b74aa948`。attempt02 构建与冻结源码独立 validator 均退出 0；成功源码快照清单 SHA256=`fa5279beca13946505b16b3aac3a94079c62b4254253255b86b78dabad22ee01`。compileall、Ruff、12 项 P5 测试和全部 112 项 QG 回归通过。attempt01 的保护性失败、快照和临时目录也独立保留。
- 新 10k 结果：三层在 `16/15/16` 轮收敛，final distortion=`0.447954/0.539825/0.584774`，三层 512 code 全激活，distinct SID=`9,934/10,000`、collision excess=`66`、碰撞 POI=`128`、最大桶=4、零 residual 行均为 0。完整三层 assignment 与旧 10k 逐值相同；连续 codebook/residual 只有 GPU 浮点归约量级差异。
- 新 10k 产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0_hard60_topk5_v1/sample_010000/`，manifest SHA256=`29a918fb5d3b3727d393ad31dc69e6f95f61fcd49a3036af8d1981473784f4d7`；run/validate exit 均为 0，90 文件快照与日志位于 `qg_prqk/outputs/run_control/p5_cat_active_512_hard60_topk5/sample_010000_attempt01/`。
- full 结果：716,245 条 active POI 三层在 `57/52/57` 轮收敛，Top-k5 final distortion=`0.475771/0.561065/0.599759`，三层 512 code 全激活，Kish ESS=`444.82/431.33/427.15`，Gini=`0.2177/0.2367/0.2474`，零 residual 行均为 0。S1 cluster min/p50/p99/max=`180/1385.5/2869.89/3430`，未发生码本坍塌。
- full SID：distinct SID=`606,045/716,245`（84.6142%），collision excess=`110,200`，碰撞 POI=`177,206`，桶 p99/max=`4/208`。这是 P5 POI-only A0 静态对照，不是 P8 最终 SID。
- full 产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0_hard60_topk5_v1/full_716245/`，manifest SHA256=`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`；构建 `run.exit=0`，独立 validator 重试 `validate_retry01.exit=0`。90 文件快照、命令、日志和预检失败 attempt01 均保留在对应 run-control 目录。
- 限制与后续：S1/S3 到第 57 轮才满足双停止，证明 full 规模需要 60 轮上限，但余量只有 3 轮。P5 不重跑；用户随后已确认 P6/P7 复合目标上限为 60，当前只等待确认 P6 sample/medium 的闭包规模。

### P6-CAT — 类别锚定的 Query 双视图 S1/S2

- 状态：`FULL_COMPLETED_AND_VALIDATED`。sample、sample 六分支归因、716,245 POI / 342,879 Query full 构建、同闭包 P5 A0 对照、full S2-only 六分支诊断及对应独立 validator 均已完成；用户于 2026-09-11 接受 S2 诊断作为留存并确认继续 P7，正式 P6 参数与产物未修改。旧 30 轮 downstream YAML、canonical-v1、P4/P5/P6 配置、checkpoint、manifest 和输出均未覆盖。
- 实现：新增 P6 配置/数据/算法/CLI、双视图交替 assignment、POI/Query 双质心、`tau=32` Query 质心收缩、S1 coarse 与 S2 conditioned-fine 类别软代价、warm-up、Top-k5 后耦合 hard assignment、projection residual、manifest 和独立 validator。所有入口和输出均在 `qg_prqk/`。
- 失败记录：`EXP-20260909-02` 首次运行在 S1 被保护性终止，原因是跨 warm-up 权重比较 objective；没有 `_SUCCESS`。失败输出和日志完整保留。修正只让连续升高门禁在第 6 轮权重冻结后开始计数，不改变数据、权重或算法目标。
- sample 构建：attempt02 使用 11,459 POI、1,000 Query、2,610 条边，S1/S2 hard 分别在 12/9 轮收敛，Top-k5 均完成；POI code 512/512 全激活，零 residual 行为 0/0。P6 manifest SHA256=`10f40347f60e38289cb9b0405787433e176c0ab051b718773d9e3e2ab36a612a`，构建和独立 validator 均退出 0。
- 当前代码核验：加入 full S2-only 诊断协议测试后完整 QG 合成回归 `143 passed`；S1/S2 紧凑类别查表与旧公式逐元素一致，Ruff、compileall、`git diff --check` 和历史 sample validator 均通过。OOM 修复只改变分块输入、分块 NPY 写入和类别代价存储布局，不改变算法与参数。
- 同闭包 P5 A0 对照：S1/S2 图加权一致率由 `28.8821%/32.4789%` 提高到 `94.4393%/89.7598%`；S1 coarse purity 由 `77.7293%` 提高到 `81.2811%`，S1+S2 fine path purity 由 `96.3086%` 提高到 `96.6053%`。但 Query cosine distortion 由 `0.621731/0.587027` 上升到 `0.683140/0.654598`，两层均未改善；distinct prefix 从 `10,403` 降到 `10,387`，collision excess 增加 16，但最大桶从 12 降到 8。
- Gate 结论：图对齐、类别结构、code 利用率和最大桶均正常，没有坍塌；“Query distortion 改善”未通过，因此不批准自动 medium。对照 evaluation manifest SHA256=`93b15fcd2fe559b838080492470bbaee798982bc1837374c6bea2c8e3e718075`，独立 validator 退出 0。
- 六分支归因：canonical 复跑与正式 sample 的两层 POI/Query assignment 逐值一致。只关闭 graph 后 distortion 降至 `0.588880/0.582160`、图一致率降至 `37.5139%/38.5289%`；只将 Query 质心先验从 `tau=32` 降至 `1e-6` 后 distortion 降至 `0.262386/0.230850`、图一致率仍为 `63.7304%/60.7922%`。canonical 的 Query-weighted POI prior mass 为 `85.20%/90.85%`，占用码的 Query 支持中位数均仅为 2，证明 `tau=32` 是 1k Gate distortion 的首要来源，graph 是次要来源。
- 次要因素：关闭 Top-k5 后 distortion 改善 `0.016352/0.008239`，图一致率仅变化 `-0.304/+0.545pp`；关闭 category 后 distortion 只变化约 `-0.001013/+0.001321`，说明 category 不是 Query 退化来源。近零先验使 Query active code 达到 `492/500`，接近小样本记忆，因此只作因果诊断，不能冻结为候选参数。
- 诊断产物：`poi_query_category_prqk_s1_s2_hard60_topk5_v1/diagnostics/sample_001000q_011459p_attribution_v1/`，manifest SHA256=`cd8e9e61274fd9632b48778fcac5827dfa3dd5773a4e33fa22726772743ac441`，comparison SHA256=`0412aa78cfbf033d23ff302693d90ad033f143ef5c3a361446158e7c3e167f2a`；构建和冻结源码独立 validator 均退出 0。输出 role 为 `CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT`，不修改正式 P6/P4/P5 产物。
- 诊断结论：当前 1k Query 对固定 `tau=32` 明显偏稀疏，sample 结果不能稳定代表全量。用户已接受 sample 指标不作为方法 Gate，并明确要求保持 canonical 算法直接 full；不按近零先验分支调参。
- direct-full 合同：新增 `qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml`，配置签名=`2e7aa687648d3ba7e2737ade36d6d60f273c3dfe08efe20804476bb01dcc8fc0`；绑定 sample manifest SHA256=`10f40347f60e38289cb9b0405787433e176c0ab051b718773d9e3e2ab36a612a`，授权标识为 `USER_CONFIRMED_20260909_P6_DIRECT_FULL_AFTER_SMOKE`，显式记录跳过 medium，范围仅限 P6 S1/S2。
- full dry-run：OOM 修复后退出 0；严格装载 716,245 POI、342,879 Query、912,980 条 S1/S2 边，闭包新增 POI=0，未读取业务 Validation/Test，未写正式输出；墙钟 22.14 秒、峰值 RSS 1,302.08 MiB。修复前 dry-run 峰值约 4,979 MiB。
- full 失败现场：attempt03 已进入 S1，但在发布任何层级产物前被 SIGKILL，退出码 137；外层最大 RSS 10,106,060 KiB，同一时刻内核记录 memory-cgroup OOM。2.1 GiB partial output 已移入对应 run-control 的 `failed_partial_output/`；attempt01/02/03 均无有效 full manifest 或 `_SUCCESS`，不得用于方法指标。
- full 成功构建：attempt05 从相同 P5 A0 和同一 direct-full-v2 配置重新开始，S1/S2 hard fit 在 `39/12` 轮收敛，POI/Query 两层均 512 code 全激活，零 residual 行均为 0；构建、独立 validator、公平对照与对照独立 validator 的退出码均为 0。attempt04 是配置快照路径签名保护性失败，attempt06 被已有 `_SUCCESS` 的 overwrite guard 拒绝，均未修改成功产物。
- full P5 A0 对照：S1 Query distortion `0.624580→0.623430`（改善 0.001151），S2 `0.592466→0.620627`（退化 0.028162）；图加权一致率 `29.9614%→81.6757%`、`33.9990%→80.2796%`；S1 coarse/S1+S2 fine path purity 分别提高 `10.4784/6.0707pp`。`query_distortion_improved_both_levels=false`，因此当时不得自动进入 P7；后续用户审核诊断后已另行明确授权。
- full 前缀与产物：P5/P6 distinct S1+S2 prefix=`150,144/147,981`，P6 少 2,163；最大桶 `522→469`。P6 manifest SHA256=`326e00d62e132bffdcbaa05ac21a839b886fda5c9d00608c4ebfcb9486bb1624`；evaluation manifest SHA256=`e9cb07063b6dad09845fb5ccd22afde234bf0e977fef728b24b933c2aa93c74d`，两者均有 `_SUCCESS` 且通过冻结源码独立复核。
- full S2-only 四点分解：P5 path=`0.592465`，在冻结 P6-S1 residual 上使用 P5-S2 content-nearest=`0.569664`，再换 P6-S2 Query centroid content-nearest=`0.515784`，最终 coupled assignment=`0.620627`。三项效应为 S1 residual `-0.022802`、S2 质心适配 `-0.053880`、S2 coupled assignment `+0.104844`，严格加和为总退化 `+0.028162`；根因不是上游 residual 几何退化。
- full S2-only 六分支：graph-off 将 distortion 降至 `0.510165`，但图一致率由 `80.2796%` 降至 `28.4846%`，说明 graph 是主要代价且不能直接关闭；`tau→1e-6` 只改善 `0.002110`，full prior mass 仅 `4.8425%`；Top-k5 改善 distortion `0.011019`、purity `0.6618pp` 和 distinct prefix 1,794；category 仅增加 distortion `0.000403`，同时提高 fine path purity `1.4316pp`。因此 prior/Top-k/category 均不是 S2 退化主因。
- full S2-only 产物：`diagnostics/full_342879q_716245p_s2_attribution_v1/`，manifest SHA256=`7e17ce4369046d8c2c9453ced59d7b47a50afbf23bd010cb924cae83c32d55d0`，comparison SHA256=`0900f9d17569bbce8128e603bf9c582393919e6bed7b8d3daf8521bdb38dbe4d`；canonical Query assignment 逐值一致、POI match=`99.99972%`，独立 validator 退出 0。所有后台启动可见性与 overwrite guard 现场分别保留，正式 P6 未修改。

### P7-CAT — D3 Query + Geo-aware S3

- 状态：`FULL_COMPLETED_AND_VALIDATED`。用户已接受 P6 S2 诊断作为方法权衡记录，正式 P6 保持不变；P7 按“小样本只跑通，随后直接全量”的授权完成。其 `HOLD_FOR_P7_FULL_REVIEW` 已由用户确认履行，随后进入并完成 P8。
- 实现：新增独立 P7 配置、严格 loader、longitude-first Geohash6 与北京局部相对 Geo、固定 `(GID6,S1,S2)` parent、same-fine-category 困难图、D3-only 双质心 S3、Top-k5、bounded local refinement、projection residual、manifest 和独立 validator。类别不作为 S3 分类代价，D1/D2 不进入 S3；所有代码与产物均位于 `qg_prqk/`。
- 冻结权重：Query/content graph/Geo/hard collision=`0.20/0.20/0.10/0.05`；困难图复合分数 BGE/name/category/address/geo=`0.35/0.30/0.15/0.10/0.10`，阈值 `0.60`，每 POI Top-20；S3 初始化来自冻结 P5 A0，S1/S2 逐值冻结自正式 P6。
- sample：11,459 POI、832 条 D3 Query、376 条困难边。attempt01 完成计算后因 Arrow derived-field nullable 合同不匹配而由 validator 拒绝，失败目录原样保留；修正显式 non-null schema 后 attempt02 构建与独立 validator 均退出 0，manifest SHA256=`5a18914d4236476dc2c0f037afa2699897938416faaa099d72a873b40419ea68`。sample 只承担工程 smoke，不作为方法调参依据。
- full：716,245 POI、291,590 条 D3 Query、746,034 条有向困难边。hard alternating 在第 40 轮收敛，Top-k5 运行 5 轮、bounded local refinement 运行 3 轮；POI/Query S3 均激活 512/512 个码，零 residual 行为 0/0。
- full Query/结构指标：`Acc(S3 | GID6,S1,S2)` 未加权/加权=`82.7810%/85.2967%`；困难实体碰撞率未加权/加权=`16.6089%/17.5233%`；Query distortion 从初始 content-nearest 的 `0.718783` 降至最终 `0.673663`，POI semantic distortion=`0.603686`，Geo distortion=`0.245575`。
- full 唯一率：纯三层 SID distinct=`617,887/716,245`（86.2675%），collision excess=`98,358`，桶 p50/p90/p95/p99/max=`1/2/2/4/95`。相对 P5 A0 的 `606,045`（84.6142%）、collision excess `110,200`、max `208`，多 11,842 个唯一 SID且最大桶下降；该对照只说明路径结构变化，完整 A0/A4 评测留到 P8。
- GID 条件桶：`GID4/GID5/GID6 + SID` distinct ratio=`88.2620%/90.7114%/93.8637%`，collision excess=`84,073/66,529/43,951`，最大桶=`95/95/56`。商场/医院/车站关键词代理子集困难边碰撞率=`10.5263%/14.2012%/20.2117%`；它们只是类别路径 proxy，不冒充真实父子 POI 映射。
- 产物：`qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_geo_prqk_s3_hard60_topk5_v1/full_291590q_716245p/`，约 2.1 GiB；manifest SHA256=`dd395ab0d11321106455ad05b18503af9b7d3dda7cb7fd85e531655cece2133c`。构建外层墙钟 `5:09.05`、最大 RSS `8,168,752 KiB`、GPU peak allocated `20,755.21 MiB`，退出码 0；冻结源码独立 validator 墙钟 `17.67s`、退出码 0。
- 运行现场：成功 attempt01 的 130 文件源码快照清单 SHA256=`fe0338c9244ce3b1ed90986bf75a2212c0360764d00c48ef6069b0542ae028b3`，console SHA256=`4930f9458a8d8150ac5235d1be7c7fc41c35084588fe2d13268cbd6eb4929997`，validator console SHA256=`fd78cd3f5d44c19b6cf8c9efebd0078ccfaf68fd1a88377ad02777a93e564dbc`。detached 父 PID 一度不可见但子进程成功完成；attempt02 因 attempt01 已创建 overwrite-protected 目录而在数据加载前退出 2，未覆盖任何产物，两次现场均保留。
- 代码验收：sample 前后完整 QG 合成回归 `149 passed`，P7 定向回归 `6 passed`，Ruff、compileall、full dry-run 与 sample/full 独立 validator 均通过。成功标记时序已收紧为“先校验全部 artifact，再写 `_SUCCESS`”，不改变算法或参数。

### P8-CAT — 静态评测与完整 SID 发布候选

- 状态：`FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED`。用户确认进入 P8 后完成 sample smoke 与 full；只比较 A0 POI-only PRQK 初始化和 A4 Query+Category+Geo，不读取原始订单或业务 Validation/Test，不运行历史外部基线。实验完成时曾强制停在 `HOLD_FOR_REVIEW`，该停止点随后已履行。
- 实现：新增严格 P8 配置、冻结输入 loader、code/path/Kish ESS/Gini/entropy、Purity/NMI、每类别使用码数、分 D1/D2/D3 Query–POI 对齐、teacher-forced/free-running Prefix Probe、同 GID6 精确 pairwise 分离、A4 SID/码本/assignment/类别分布/bucket index 发布和独立 validator。`poi_sid.parquet` 保留 active `poi_row_index/poi_id` 并用 `bucket_id` 连接 bucket index；不强制 SID 唯一。
- 数据质量：完整 P2.5 Train Query 为 D0/D1/D2/D3=`1,029,782/25,639/25,650/291,590`，共 1,372,661；S1/S2/S3 覆盖 POI=`210,965/188,570/153,349`。active catalog 716,245 POI，实际 19 coarse、397 fine，`fine -> coarse` 一致；按 D0–D3 记录 concentration/normalized entropy 分布，并发布每层固定 20 条 P2.5 样例。
- 结构结果：A0/A4 distinct SID=`606,045/617,887`（A4 `+11,842`），collision excess=`110,200/98,358`，最大桶=`208/95`。coarse purity(S1)=`76.3230%/86.8014%`、NMI=`0.42863/0.52696`；fine purity(S1+S2)=`73.2765%/79.3472%`、NMI=`0.49205/0.51665`。同 GID6 pairwise local separation=`99.8895%/99.9264%`，GID6+SID distinct=`659,041/672,294`。A4 的结构、类别和局部分离 headline 均改善。
- 码本与 residual：A4 三层 POI/Query paired codebook cosine mean=`0.73681/0.87832/0.88426`，三层 code 均 512 全激活。A0/A4 retained-energy mean 分别为 S1 `0.71574/0.71867`、S2 `0.79517/0.79384`、S3 `0.82866/0.83117`，两者零 residual 行均为 0；未发现 residual 数值退化或 code collapse。
- Query 对齐交叉检查：A4 冻结训练 assignment 的 S1/S2/S3 加权 token 一致率=`81.6757%/80.2796%/85.2967%`，逐值复现 P6/P7 指标，证明 Query/POI 行映射正确。D1/D2/D3 的 S1 加权一致率=`68.6102%/74.6895%/84.2742%`；D2/D3 的 S1+S2 cumulative=`41.2623%/73.0157%`，D3 S1–S3 cumulative=`65.6254%`。
- Prefix Probe：A0/A4 weighted `Query->S1`=`29.9614%/21.0191%`，`Query+correct S1->S2`=`39.2334%/36.2871%`，`Query+correct GID6/S1/S2->S3`=`90.1722%/89.7239%`；autoregressive cumulative S1–S3=`12.7183%/7.7743%`。S3 teacher-forced parent 候选数 p50/p90/p99 A0 与 A4 都为 `1/4/19`，singleton parent 比例约 `65.12%/64.66%`，因此约 90% 的 S3 probe 必须结合候选难度解释，不能单独宣布成功。A4 free-running S3 candidate coverage=`16.8195%`，低于 A0 的 `25.3435%`。
- 结论：`REVIEW_REQUIRED`。A4 的类别结构、最终 SID 唯一性、最大桶与 GID6 局部分离均改善，但当前明确定义的 content-only Prefix Probe 在 full 的三个 teacher-forced 层级和 cumulative S1–S3 上都低于 A0。该结果与训练 assignment 的高图一致率不冲突：前者测试脱离图目标后的 Query-only 可预测性，后者测试训练图上的耦合对齐。P8 不自动调参；是否保留 A4、调整 graph/query 权重或采用其他 probe/SFT 验证，均等待用户审核。
- 产物：canonical full 位于 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/p8_static_evaluation_a0_vs_a4_v1/full_342879q_716245p/`，manifest SHA256=`c66ea6c5fac499ed7c377b9f2ae7d73c3fde333c7842f0d79e9c84b141c82ab3`；sample manifest SHA256=`71677030dea72014662aa8fd593abd58901a83b7d6db11f4246f23f577813ae4`。full 构建墙钟 `2:39.22`、最大 RSS `3,496,400 KiB`、退出 0，冻结源码独立 validator 退出 0；成功源码快照清单 SHA256=`60f2933334c0e0be4c21408e724bffa0fe0944cd6fff4d08282b1447edb94ae1`。
- 失败/重试保留：首个 sample/full 计算本身退出 0，但 GID6+SID 展示误把 packed GID 当成第四个 512 码层，理论容量字段无效；两份目录均完整保留为 `.failed_attempt01_gid_capacity_metric`，有效 distinct/碰撞/probe 数值未变。修正移除无效容量归一、增加 `bucket_id`、候选难度、分 depth 分布和显式 `REVIEW_REQUIRED` 后重跑。corrected full attempt02 的 detached launcher 曾短暂不可见，但实际持续运行并成功退出；随后直接重试由 overwrite guard 退出 2，未覆盖 canonical 产物。
- 验收：配置/sample/full dry-run、5 项 P8 定向测试、Ruff、compileall、sample/full 构建与独立 validator 均通过；阶段收尾时完整 QG 合成回归 `154 passed`（89.632 秒）。所有新代码、文档、运行日志、快照和产物均位于 `qg_prqk/`。

### NoGID `(S1,S2)` parent S3 受控变体

- 状态：`FULL_COMPLETED_AND_VALIDATED_REVIEW_REQUIRED`。冻结 P6 S1/S2，只在 `(S1,S2)` parent 内构建 S3，完整链路不读取、不计算、不输出 GID6/geohash，最终 POI/Query SID 均为三列 `[S1,S2,S3]`。
- 实现：P7 NoGID 独立配置/数据/Geo/PRQK/CLI 和构建脚本全部位于 `qg_prqk/`。沿用原 P7 的 P5 初始化、D3 Query 图、projection residual、Query/graph/Geo/hard 权重、warm-up、Top-k5 和 60 轮上限；连续 Geo 特征相对 `(S1,S2)` parent 中心表示，困难边只在 `(S1,S2,fine category)` 内构造。
- P7 full：716,245 POI、291,590 D3 Query、147,981 个 parent、2,952,789 条有向困难边；43 轮收敛，所有 512 码激活，S3 加权训练一致率=`84.8905%`。三位 SID distinct=`613,198`（85.6129%），collision excess=`103,047`，max bucket=`159`。full manifest SHA256=`a6ce4ba8c20f79073e5a41fa2e516b26b2bdeb7c2d7c56e260f4a94f46bf1d6c`，构建/冻结源码 validator 均退出 0。
- 公平 P8：A0/原 GID-parent A4/NoGID A4 都只使用 `(S1,S2)` S3 候选。三者 weighted S3 probe=`67.7142%/66.2934%/65.8531%`，cumulative S1–S3=`9.7802%/5.8439%/5.7867%`。NoGID 相对原 A4 少 4,689 个 distinct，最大桶 `95→159`，S3/cumulative probe 下降 `0.4403/0.0572pp`。P8 manifest SHA256=`b22ab413558b1e147542135fd554fdf0b7a71082bdd1842fc787aa2b16ed4ed8`，构建/独立 validator 均退出 0。
- 结论：NoGID 三位分支的工程和数据合同成立，且结构优于 A0；但未优于原 GID-parent A4，不自动替换 canonical 分支。详情见 `EXP-20260911-03`。
- 验收：P7 sample/full 构建、P8 full 比较及各自冻结源码 validator 均退出 0；完整 QG 合成回归 `162 passed`（97.438 秒），全目录 compileall、Ruff 和 `git diff --check` 均通过。

### SID 三方并排可视化

- 状态：`FULL_COMPLETED_AND_VALIDATED`。只读比较 `A0_POI_ONLY`、`A4_GID_PARENT` 与 `A4_NOGID_S1S2_PARENT`，没有重新训练、重新 assignment 或改写 SID；不读业务 Validation/Test，不启动下游。
- 实现：配置、严格来源 loader、exact-prefix pair 抽样、occupancy/transition/prefix 指标、逐层共同 residual-space PCA+t-SNE、三张图、可读 POI/类别样例、manifest 和 validator 全部位于 `qg_prqk/`。seed=42，每方法每个 exact-prefix 长度 20,000 pair，PCA-50+t-SNE 1000 轮，CPU 线程固定为 8。
- 结果：三方法 S1/S2/S3 均激活 512 码。A0/原 A4/NoGID S3 ESS=`427.15/426.87/407.12`，Gini=`0.2474/0.2468/0.2680`；NoGID S3 分布比另两者更偏斜。三方法 exact-prefix 0→3 的原始 BGE cosine 均约从 `0.565` 单调升到 `0.891–0.897`。A4 在 exact-prefix 2 上的 coarse/fine 一致率为原 A4 `89.24%/72.84%`、NoGID `89.80%/73.02%`，明显高于 A0 的 `77.52%/63.01%`。
- 产物：`qg_prqk/outputs/figures/qg_prqk_sid_a0_a4_nogid_full_v1/`，manifest SHA256=`3057f95601eff3441df5b5b2a67051a5babda489e8c8a7a9507e252f69e5b2f3`。正式构建 `160.46s`，独立 validator 退出 0；新增 6 项定向测试，完整 QG 回归 `168 passed`（86.609 秒）。详细设计、失败尝试和完整指标追加在 `EXP-20260911-03`。
- 结论：可视化支持“共享前缀具有递进内容语义、A4 类别组织强于 A0、NoGID S3 码分布更不均衡”的判断；不改变 NoGID 未优于原 A4 的既有结论。该分析完成时停在 `HOLD_FOR_NOGID_S3_REVIEW`，此后用户明确批准两个分支均进入 SFT 数据准备。

### P8-CAT 后续项

- 状态：`P10_SHARED_VOCAB_AND_DUAL_TOKENIZED_CACHE_COMPLETED_AND_VALIDATED`。
- 范围：双分支 Final ID/Messages、共享扩词表和双 Train/Valid Tokenized Cache 已完成；两个 Qwen SFT 入口可启动但训练尚未启动。历史外部基线、Order-B、其他方法消融和数据/容量消融继续延后。

---

## 13. 当前 NEXT_ACTION 详细说明

active BGE/category、active P2.5、P3A-FULL、P4–P8、NoGID 受控分支、三方公平比较和 SID 可视化均已完成并独立核验。用户随后决定不在两个 SID 中二选一，而是同时构造 A4 GID-parent 与纯三位 NoGID 的 SFT 分支。

P9 已完成两套覆盖 716,245 POI 的唯一 Final ID，以及由同一冻结源一次扫描生成的两套 8,790,513 行 history10 Messages。P10 已从同一 Qwen3-0.6B 构造共享 3,743 Token 词表，并分别对两个分支的 7,586,410 条 Train 与 597,421 条 Valid 完成全量 1024 门禁和不可覆盖 Cache；两者超长与 target truncation 都为 0，Test 未读取。两个四卡 launcher 的完整 `--dry-run` 均退出 0，训练输出目录不存在且没有训练进程。

当前状态为 `HOLD_FOR_QG_PRQK_DUAL_SFT_LAUNCH_REVIEW`。下一步只能由用户选择正式启动 GID-parent、NoGID，或串行启动两者；两个分支使用同一初始模型和算法配置、不同缓存与输出目录。当前不得自动启动训练、读取 Test、运行端到端评测、历史外部基线、Order-B 或其他消融。
