你正在仓库 `/ofs/map_search/hudan/poi_genret` 中继续 QG-PRQK。先从文档恢复状态，并区分已确认的设计版本与已实现的代码版本；不要把文档中的计划当成运行结果。

开始前阅读：

1. 仓库根 `AGENTS.md`、`README.md`、`方案.md` 和 `docs/PROJECT_STATUS.md`；
2. `skills/poi-genret-workflow/SKILL.md`；
3. `qg_prqk/docs/methods/QG_PRQK_METHOD_SPEC.md`；
4. `qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md`；
5. `qg_prqk/docs/methods/QG_PRQK_CODEX_EXECUTION_PLAN.md` 的当前版本导航；
6. 与用户当前任务及 `NEXT_ACTION` 有关的现有代码、配置和测试。

版本边界（2026-09-19）：

- v3.0 方案已获用户确认，仅完成文档登记；尚无粒度模型、搜索分布、共享 assignment 量化器或下游结果。
- 当前源码、YAML 和已完成 A0/A4 实验属于 v2.1-CAT。复核它们时阅读 `qg_prqk/docs/methods/QG_PRQK_V2_1_CAT_SPEC.md`，使用相应冻结来源；旧文档中的“尚未启动”与阶段停止点不能覆盖最新状态或用户授权。
- v3.0 下一最小开发步骤是 V3-DATA：先建立真实字段契约和合成验证。方案登记不代表所有后续训练已开始，也不要求重跑已完成的 v2.1 实验。

工作规则：

- 以用户当前任务和持续有效的授权为准，每次完成一个明确、可核验的最小闭环；不要因历史 HOLD 字样重复索要已获得的授权。
- 先检查 `git status`，保留用户工作树；QG 源码、配置、测试和产物继续在 `qg_prqk/` 现有边界内。
- 复用 active POI、BGE、512³、GID/Dedup 和配对 SFT/评测协议；v3.0 使用独立 schema、配置和输出，不把旧 Query 筛选、Adapter policy 或自由 Query assignment 静默移入新版本。
- Query 统计、标签、粒度训练/校准和 SID 构建只用 Train；下游候选由 Validation 比较，Test 不调参。
- 新流程先字段契约和合成测试，再训练实现；全量/GPU 工作按实际授权及 workflow 执行。本次任务若只要求文档，就不启动数据处理或实验。
- 更新实际完成状态；只有真实实验才登记 EXP；不伪造指标，不覆盖历史产物，不自动 commit/push。

结束时说明修改、数据流、验证结果、限制和下一步，以及是否 commit/push。
