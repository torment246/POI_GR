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
docs/experiments/TIGER.md     TIGER 标识符、训练和评测记录
docs/experiments/GNPR_SID.md  GNPR-SID 输入、RQ-VAE 和后续实验记录
docs/experiments/GENPOI.md    GenPOI GeoPE、训练和 TCG+SSP 评测记录
docs/DATA_AND_ARTIFACTS.md    数据、模型和产物管理规则
docs/REPO_MAP.md              稳定目录职责
configs/                       可复现任务配置
configs/embedding/             Embedding 构建、Query 增强和向量召回评测配置
configs/sid/                   RQ-VAE、RQ-KMeans 和 SID 构建配置
configs/sft/                   各方法 SFT、smoke 与 LLaMA-Factory 数据注册配置
configs/methods/current/       当前已验证主线的方法契约
configs/methods/baselines/     GenPOI、TIGER 和 GNPR-SID baseline 契约
configs/methods/innovations/   QGR-SID 等创新方法的真实阶段契约
src/poi_gr/                    可复用 Python 实现
src/poi_gr/methods/            方法发现、状态和配置校验
scripts/                       命令行入口
scripts/methods.py             列出、查看并校验方法契约
scripts/embedding/             POI 编码、固定评测集、召回评测和案例分析命令
scripts/sid/                   RQ-VAE 训练、checkpoint SID 导出和统一评估命令
scripts/pid/                   Geohash/Dedup PID、扩词表和 Final PID Trie 命令
scripts/sft/                   共享 SFT 历史安全裁剪、训练、Cache 校验/恢复、Validation 专项集和生成式评测命令
scripts/v1/                    V1 无历史 Query→Final PID 数据命令
scripts/tiger/                 TIGER 数据、标识和专属评测命令
scripts/genpoi/                GenPOI GeoPE、数据、SSP 和 proximity 命令
scripts/gnpr/                  GNPR 数据、标识、训练和无约束生成评测命令
scripts/qgr_sid/               QGR-SID 关系审计及后续局部编译命令
scripts/ghr_sid/               GHR-SID 北京全量结构/残留审计、旧变长与全局对齐固定后缀的 mapping/SFT 数据及生成评测命令
launchers/                     所有训练平台启动脚本，单层平铺并由 Git 忽略
tests/                         合成数据轻量测试
data/                         本地内部数据，Git 忽略
models/                       本地模型，Git 忽略
outputs/                       本地向量和实验产物，Git 忽略
third_party/LLaMA-Factory      固定版本的训练框架 gitlink
third_party/LLaMA-Factory-local.patch Python 3.10 兼容与末步重复评测修补
```

`src/poi_gr/embedding/` 统一保存 POI 编码流水线与固定 Embedding 评测集契约，其中 `mmbert_recall.py` 负责线上 ModernBERT encoder、attention-mask mean pooling 和训练好的 128 维 projection 适配；`src/poi_gr/sid/` 统一保存共享 RQ-VAE、checkpoint 导出和 SID 评估实现，`src/poi_gr/pid/` 统一保存 Geohash/Dedup PID 与 Final PID Trie，`src/poi_gr/sft/` 统一保存四种方法复用的数据契约、历史事件级 Token 预算、Validation 随机/地理/泛化专项集和生成式评测实现；`scripts/sft/build_history_safe_data.py` 在超过新协议 1024 Token 时只移除时间最早的完整历史事件，并禁止裁剪 CURRENT 与 Assistant 目标。V1、GNPR、TIGER、GenPOI、QGR-SID、GHR-SID 专属实现分别位于 `src/poi_gr/methods/v1/`、`src/poi_gr/methods/gnpr/`、`src/poi_gr/methods/tiger/`、`src/poi_gr/methods/genpoi/`、`src/poi_gr/methods/qgr_sid/`、`src/poi_gr/methods/ghr_sid/`，方法内文件名不再重复方法前缀。测试按相同职责归入 `tests/embedding/`、`tests/sid/`、`tests/pid/`、`tests/sft/` 和各方法目录，仓库根层只保留跨方法目录发现测试。

`configs/methods/` 用于区分当前主线、论文 baseline 和创新。方法配置只声明真实准备状态：已有完整入口为 `ready`，只能复用部分组件为 `partial`，尚未实现为 `missing`。实际实现开始后，可以按方法在 `src/poi_gr/methods/` 下增加独立模块，允许保留少量重复代码；不得把尚未实现的阶段标记为可运行。QGR-SID 因门禁未通过保持 `partial`；GHR-SID 的结构、identifier、旧变长与新固定五层 SFT 数据、训练和生成评测入口均已实现，方法契约标记为 `implemented`，实际进度以 `docs/experiments/GHR_SID.md` 为准。

训练平台专属脚本统一平铺在仓库根目录的 `launchers/`，不再按方法嵌套子目录；脚本保留在本地工作区且不进入 Git，代码和实验文档使用 `launchers/<filename>` 记录实际平台入口和历史命令。新增平台任务从同方法、同资源类型的既有入口复制后修改；开发服务器上的一次性 Embedding、特征准备、分类头和评测不额外增加 `.sh` 包装。

真实数据、模型和可重新生成的实验产物继续保留在 Git 之外。
