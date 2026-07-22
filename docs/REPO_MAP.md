# 仓库目录说明

当前只列出已经存在且职责稳定的路径。

```text
README.md                     项目入口和当前范围
方案.md                       第一版技术方案与实施顺序
AGENTS.md                     Codex 开发、核验和实验记录规则
docs/PROJECT_STATUS.md        当前阶段、已确定事项和下一步
docs/EXPERIMENT_LOG.md        唯一的正式实验进展记录
docs/DATA_AND_ARTIFACTS.md    数据、模型和产物管理规则
docs/REPO_MAP.md              稳定目录职责
configs/                       可复现任务配置
src/poi_gr/                    可复用 Python 实现
scripts/                       命令行入口
tests/                         合成数据轻量测试
data/                         本地内部数据，Git 忽略
models/                       本地模型，Git 忽略
outputs/                       本地向量和实验产物，Git 忽略
```

当前已创建 POI 文本向量和 RQ-VAE SID 最小闭环需要的代码与配置。后续目录仍按已核验步骤逐项增加，不恢复旧目录结构。
