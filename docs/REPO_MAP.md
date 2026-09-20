# 仓库目录说明

当前只列出已经存在且职责稳定的路径。

```text
README.md                     项目入口和当前范围
方案.md                       第一版技术方案与实施顺序
AGENTS.md                     Codex 开发、核验和实验记录规则
skills/poi-genret-workflow/   本项目服务器 GPU、流式数据和实验执行 Skill
skills/make-poi-genret-ppt/   0727/0810 阶段汇报 PPT 样式、备注和交付核验 Skill
docs/PROJECT_STATUS.md        当前阶段、已确定事项和下一步
docs/EXPERIMENT_LOG.md        正式实验方法入口和总索引
docs/INNOVATION_DIAGNOSIS.md  现有创新结果、三篇论文基线差距归因和后续方向决策参考
docs/experiments/V1.md          第一版共享模型选型与完整链路
docs/experiments/EMBEDDING_OPTIMIZATION.md Query 增强向量的协议、消融和正式结果
docs/experiments/VECTOR_MODEL_OPTIMIZATION.md 新向量 checkpoint 接固定量化/SFT 下游的实验台账
docs/experiments/QGR_SID.md   Query/GID 引导的纯离散数值变长关系 SID 方法与实验台账
docs/experiments/GHR_SID.md   G6 后仅用最短细实体关系的 Query-free 碰撞 SID 方法与实验台账
docs/experiments/BUCKET_RERANK.md 生成 Bucket 展开、POI 解析和轻量重排实验台账
docs/experiments/TIGER.md     TIGER 标识符、训练和评测记录
docs/experiments/TIGER_JOINT.md TIGER 动态 SID 与生成器联合训练方案、闸门和实验台账
docs/experiments/GNPR_SID.md  GNPR-SID 输入、RQ-VAE 和后续实验记录
docs/experiments/GENPOI.md    GenPOI GeoPE、训练和 TCG+SSP 评测记录
docs/DATA_AND_ARTIFACTS.md    数据、模型和产物管理规则
docs/REPO_MAP.md              稳定目录职责
configs/                       可复现任务配置
configs/embedding/             Embedding 构建、Query 增强和向量召回评测配置
configs/sid/                   RQ-VAE、RQ-KMeans 和 SID 构建配置
configs/sft/                   各方法 SFT、smoke 与 LLaMA-Factory 数据注册配置
configs/tiger_joint/           TIGER-Joint 单卡诊断与四卡 DDP 联合训练配置
configs/methods/current/       当前已验证主线的方法契约
configs/methods/baselines/     GenPOI、TIGER 和 GNPR-SID baseline 契约
configs/methods/innovations/   QGR-SID 等创新方法的真实阶段契约
src/poi_gr/                    可复用 Python 实现
src/poi_gr/data/               行为目标与历史 POI 并集、活跃 POI 子库构建和原子发布
src/poi_gr/methods/            方法发现、状态和配置校验
scripts/                       命令行入口
scripts/methods.py             列出、查看并校验方法契约
scripts/data/                  活跃 POI 子库等共享数据构建命令
scripts/embedding/             POI 编码、固定评测集、召回评测和案例分析命令
scripts/sid/                   RQ-VAE 训练、checkpoint SID 导出和统一评估命令
scripts/pid/                   Geohash/Dedup PID、扩词表和 Final PID Trie 命令
scripts/sft/                   共享 SFT 历史安全裁剪、训练、Cache 校验/恢复、Validation 专项集和生成式评测命令
scripts/v1/                    V1 无历史 Query→Final PID 数据命令
scripts/tiger/                 TIGER 数据、标识和专属评测命令
scripts/tiger_joint/           TIGER-Joint 原始行为→BGE 行号构建、SID-free 预检、fresh KMeans、联合训练、全目录 SID/Bucket 评测和独立复核命令
scripts/bucket_rerank/         冻结生成 Bucket 展开和最终 POI Top-K 重排评测命令
scripts/genpoi/                GenPOI GeoPE、数据、SSP 和 proximity 命令
scripts/gnpr/                  GNPR 数据、标识、训练和无约束生成评测命令；build_aligned_sft_data.py 从 active TIGER 仅替换 SID 并逐行核验用户哈希等输入
scripts/qgr_sid/               QGR-SID 关系审计及后续局部编译命令
scripts/ghr_sid/               GHR-SID 北京全量结构/残留审计、旧变长与全局对齐固定后缀的 mapping/SFT 数据及生成评测命令
launchers/                     所有训练平台启动脚本，单层平铺并由 Git 忽略
tests/                         合成数据轻量测试
data/                         本地内部数据，Git 忽略
models/                       本地模型，Git 忽略
outputs/                       本地向量和实验产物，Git 忽略
third_party/LLaMA-Factory      固定版本的训练框架 gitlink
third_party/LLaMA-Factory-local.patch Python 3.10 兼容与末步重复评测修补
qg_prqk/                      QG-PRQK 方法的自包含源码、配置、测试、文档和本地产物
qg_prqk/scripts/qg_prqk.py    QG-PRQK 唯一仓库内 CLI 入口
qg_prqk/src/qg_prqk/data/     数据契约、Query 统计、Query 监督、Embedding 与 Query 图
qg_prqk/src/qg_prqk/adapters/ Query Adapter 模型、精确检索、Gate 与全量选择
qg_prqk/src/qg_prqk/sid/      基础/关系化/局部码本、静态评测、可视化与 Final ID
qg_prqk/src/qg_prqk/sft/      SFT 数据、共同词表、Tokenized Cache 与 LLaMA-Factory 接入
qg_prqk/src/qg_prqk/commands/ 统一 CLI 的参数解析和领域命令适配层
qg_prqk/docs/methods/        v3.0 主规范、v2.1-CAT 规范归档及阶段执行导航
qg_prqk/outputs/              QG-PRQK 唯一运行产物根目录，Git 忽略
```

`src/poi_gr/data/` 保存跨方法共享的数据准备实现；当前 `active_poi_catalog.py` 从两周全部目标与既有历史序列提取 POI 并集，再按原始分片和行序原样筛选北京主表。`src/poi_gr/embedding/` 统一保存 POI 编码流水线与固定 Embedding 评测集契约，其中 `mmbert_recall.py` 负责线上 ModernBERT encoder、attention-mask mean pooling 和训练好的 128 维 projection 适配；`src/poi_gr/sid/` 统一保存共享 RQ-VAE、checkpoint 导出和 SID 评估实现，`src/poi_gr/pid/` 统一保存 Geohash/Dedup PID 与 Final PID Trie，`src/poi_gr/sft/` 统一保存四种方法复用的数据契约、历史事件级 Token 预算、Validation 随机/地理/泛化专项集和生成式评测实现；`scripts/sft/build_history_safe_data.py` 在超过新协议 1024 Token 时只移除时间最早的完整历史事件，并禁止裁剪 CURRENT 与 Assistant 目标。V1、GNPR、TIGER、TIGER-Joint、GenPOI、QGR-SID、GHR-SID 和 Bucket 重排专属实现分别位于 `src/poi_gr/methods/v1/`、`src/poi_gr/methods/gnpr/`、`src/poi_gr/methods/tiger/`、`src/poi_gr/methods/tiger_joint/`、`src/poi_gr/methods/genpoi/`、`src/poi_gr/methods/qgr_sid/`、`src/poi_gr/methods/ghr_sid/`、`src/poi_gr/methods/bucket_rerank/`；其中 TIGER-Joint 的 `evaluation.py` 负责冻结 checkpoint 的全目录三级 SID 索引、动态历史 Prompt、无约束候选解析、终态目录合法路径约束及 raw/unique Bucket 指标。测试按相同职责归入 `tests/data/`、`tests/embedding/`、`tests/sid/`、`tests/pid/`、`tests/sft/` 和各方法目录，仓库根层只保留跨方法目录发现测试。

`configs/methods/` 用于区分当前主线、论文 baseline 和创新。方法配置只声明真实准备状态：已有完整入口为 `ready`，只能复用部分组件为 `partial`，尚未实现为 `missing`。实际实现开始后，可以按方法在 `src/poi_gr/methods/` 下增加独立模块，允许保留少量重复代码；不得把尚未实现的阶段标记为可运行。QGR-SID 因门禁未通过保持 `partial`；GHR-SID 的结构、identifier、旧变长与新固定五层 SFT 数据、训练和生成评测入口均已实现，方法契约标记为 `implemented`，实际进度以 `docs/experiments/GHR_SID.md` 为准。

QG-PRQK 使用独立的 `qg_prqk/` 边界：项目级复用实现已经复制到该目录并记录 provenance，运行时不导入 `src/poi_gr`；PyTorch、FAISS、Transformers 和 LLaMA-Factory 保持为第三方依赖。当前源码不再以 P3/P4/P5 等推进编号分文件，而按 `data / adapters / sid / sft` 领域组织；这些编号只保留在冻结配置、manifest、输出目录和历史实验记录中。公共命令统一通过 `qg_prqk/scripts/qg_prqk.py` 调用，历史实验如需逐字节复核则使用 `qg_prqk/outputs/run_control/**/source_snapshot/` 中的冻结源码。

QG 文档的版本入口为 [v3.0 主规范](../qg_prqk/docs/methods/QG_PRQK_METHOD_SPEC.md)与 [v2.1-CAT 规范归档](../qg_prqk/docs/methods/QG_PRQK_V2_1_CAT_SPEC.md)。2026-09-19 仅完成新方案登记，源码仍实现 v2.1-CAT；拟新增模块、schema 和运行产物只在主规范说明，不提前创建空目录或占位实现。实验详情继续追加到同一 `qg_prqk/docs/experiments/QG_PRQK.md`。

训练平台专属脚本统一平铺在仓库根目录的 `launchers/`，不再按方法嵌套子目录；脚本保留在本地工作区且不进入 Git，代码和实验文档使用 `launchers/<filename>` 记录实际平台入口和历史命令。新增平台任务从同方法、同资源类型的既有入口复制后修改；开发服务器上的一次性 Embedding、特征准备、分类头和评测不额外增加 `.sh` 包装。

真实数据、模型和可重新生成的实验产物继续保留在 Git 之外。

### GenPOI active 适配入口

- `scripts/genpoi/active_sft.py` / `src/poi_gr/methods/genpoi/active_sft.py`：平台单进程配对数据、扩词表、全量预检与 Cache 准备，完成后四卡 SFT；配置为 `configs/sft/genpoi_active716k_pipeline_v1.yaml`。支持 plan/prepare/hardware/train，平台 Shell 入口位于根 `launchers/`。
- `scripts/genpoi/build_aligned_sft_data.py` / `src/poi_gr/methods/genpoi/aligned_data.py`：复用 active TIGER 冻结样本，替换为 GenPOI PID 并逐行核验非标识输入。
- `scripts/genpoi/evaluate_active_suite.py`：active epoch-3 五集评测的输入预检、Trie 准备、单集命令与严格结果汇总；本地四卡平台入口为 `launchers/run_evaluate_genpoi_active716k_epoch3_4x6000d.sh`，遵循目录忽略规则。
- `scripts/genpoi/evaluate_active_retrieval.py`：串接固定 Validation 业务键对齐、既有 Query-only SSP 和原 TCG 评测器；支持仅准备与 dry-run。
- `configs/sid/rqvae_genpoi_active716k_geope32_centered_512x3.yaml` 与同方法 `configs/sft/` 配置：Centered GeoPE32、active 512×3 和 cutoff 1024 协议。

- `scripts/sft/evaluate_active_test.py`：四个 active epoch-3 模型的 2026-07-14 全量 Test 预检、逐行对齐、worker 和结果汇总；协议为 `configs/sft/active_four_models_full_test_20260714_v1.yaml`，平台入口为根 `launchers/run_evaluate_active_four_models_full_test_4x6000d.sh`。GNPR 复用无约束评测，GenPOI 复用 SSP+TCG，QG 两版调用 `qg_prqk/src/qg_prqk/sft/constrained_full_test.py`。

- `qg_prqk/scripts/diagnose_a0_a4.py`：A0/A4 配对 Messages/历史标识核验与冻结全量 S1 内容排名、图分配公式复算；只读静态输入，结果进入 QG 输出目录，不重训或改写原评测。

- `qg_prqk/scripts/diagnose_qwen_prefix.py` / `qg_prqk/src/qg_prqk/sft/prefix_diagnostic.py`：冻结 A0/A4 Validation 配对样本的正确前缀评分与原 Beam 双解码追踪；单卡平台入口为根 `launchers/run_diagnose_qg_a0_a4_qwen_prefix_1gpu.sh`。
- `qg_prqk/scripts/evaluate_a0_gid_test.py` / `qg_prqk/src/qg_prqk/sft/a0_full_test.py`：冻结 A0-GID 全量 Test 的 ID 映射、准备、双解码调度和断点汇总；复用原 Test scorer。按用户明确要求，双卡入口位于忽略目录 `qg_prqk/launchers/run_evaluate_prqk_a0_gid_epoch3_full_test_dual_decode_2x6000d.sh`，运行说明见 `qg_prqk/README.md`。
