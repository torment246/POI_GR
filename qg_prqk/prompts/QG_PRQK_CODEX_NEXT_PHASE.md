继续 QG-PRQK，一次完成当前用户任务对应的一个最小闭环。

先阅读：

1. 仓库工作规则和 `skills/poi-genret-workflow/SKILL.md`；
2. `qg_prqk/docs/methods/QG_PRQK_METHOD_SPEC.md`；
3. `qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md`；
4. `qg_prqk/docs/methods/QG_PRQK_CODEX_EXECUTION_PLAN.md` 的当前版本导航；
5. 与 `CURRENT_PHASE/NEXT_ACTION` 及用户请求相关的代码、配置和测试。

执行前核对 `DESIGN_METHOD_VERSION`、`IMPLEMENTED_METHOD_VERSION` 和真实产物状态。2026-09-19 的设计版本为 v3.0，当前代码版本为 v2.1-CAT；v3.0 只登记了方案，下一开发步骤为 V3-DATA 字段契约与合成验证。不要把历史阶段文本中的待运行任务重新当作当前任务，也不要把 v3.0 计划写成完成。

- 以当前用户请求和已有效授权为准，不跳过依赖，不重复请求已获得的授权；历史 HOLD 仅描述当时阶段。
- 先检查工作树；旧代码、配置、checkpoint 和评测按原合同保留。需要复核旧算法时读 `qg_prqk/docs/methods/QG_PRQK_V2_1_CAT_SPEC.md`。
- 首轮新候选只隔离 Query 改动：全量软粒度搜索分布与共享 POI assignment，类别和旧地理算法固定。真实距离地理项单独延后比较。
- QG 实现与输出留在现有目录；复用实际模块，不提前搭建通用框架或批量占位文件。新 schema 不混用旧 Query 行号、筛选标记或产物签名。
- 粒度模型的标注、校准与验收使用 Train 内 Query 分组切分；SID 训练不读业务 Validation/Test。最终有效性由真实 Qwen SFT 和完整 POI 指标确认。
- 文档任务不启动训练或全量处理；代码任务先完成契约、合成测试和最小验证，再按授权推进。
- 只记录实际结果；文档/代码整理不新增 EXP；不自动 commit/push，不删除未知数据。

完成后更新实施状态和必要导航，核验 diff/status 及相关轻量检查，报告本步产物与下一最小步骤。
