# Repository Working Rules

## 工作定位

- 本仓库是代码开发仓库，不是自动报告生成仓库。
- 默认产出是源码、配置、测试和必要的正式文档更新。
- 优先完成可运行的代码修改，再给出必要说明。

## 开始任务前

- 阅读 `README.md`、`docs/PROJECT_STATUS.md` 和任务相关配置/实现。
- 先定位并复用现有模块，不重复实现已有功能。
- 不为单次任务随意增加顶层目录。

## 报告规则

- 默认不为每次任务创建 Markdown、审计、执行过程、迁移或时间戳报告。
- 用户未明确要求报告文件时，只在最终回复中简洁总结。
- 用户明确要求书面报告时，一次任务最多创建一个报告文件。
- 项目状态、实验结果、数据产物、目录变化分别优先更新 `docs/PROJECT_STATUS.md`、`docs/EXPERIMENT_LOG.md`、`docs/DATA_AND_ARTIFACTS.md`、`docs/REPO_MAP.md`。
- 不为同一主题创建重复 Markdown。

## 图表与生成文件

- 用户未明确要求时不生成图表，不为展示调试进度生成图片。
- 实验所需临时图表放入 `outputs/figures/`，由 Git 忽略。
- 只有被 README、论文、PPT 或正式文档引用的最终图表才能进入 tracked 目录。
- 不默认生成大量 CSV、JSON、PNG 或 HTML；可由代码重新生成的大型分析文件不得提交 Git。

## 代码开发

- 新脚本必须职责明确，提供 `--help` 和合理错误处理。
- 参数进入配置或命令行，不写死服务器绝对路径。
- 不写入 Token、密码、公司服务器地址或敏感路径。
- 滴滴真实业务数据不得加入公开 Git。
- 模型、checkpoint、Embedding、NPY、Parquet 和实验输出不得加入 Git。

## 执行与 Git

- 用户未明确要求时不运行长时间训练；可以运行 compileall、轻量测试、`--help` 和小样本 smoke test。
- 不执行 `git clean`，不删除未知数据或模型。
- 不执行 `git push`，除非用户明确要求。
- 完成前检查 `git diff`、`git status`、compileall 和相关轻量测试。
- 不创建 commit，除非用户明确要求。

## 最终回复

只需说明：修改内容、验证结果、未解决问题，以及是否 commit/push。
