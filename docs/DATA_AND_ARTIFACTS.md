# 数据与产物管理

## Git 保存内容

- 新流水线的源码、配置和测试。
- 合成测试样例或经过明确批准的匿名样例。
- README、方案和长期维护文档。

## Git 禁止保存内容

- 真实 POI、Query、行为标签和其他业务字段。
- 下载的模型目录和模型权重。
- Embedding、NPY/NPZ、Parquet、checkpoint 和二进制模型产物。
- 训练输出、日志、自动报告、图表和临时分析表。
- 凭证、内部服务地址和敏感绝对路径。

## 当前本地产物

| 路径 | 用途 | Git 规则 |
|---|---|---|
| `data/` | 内部源数据和后续生成数据 | 全部忽略 |
| `models/` | 本地 Qwen3 生成模型和 Embedding 模型 | 全部忽略 |
| `outputs/embeddings/` | 全量 Embedding、ID 映射和运行元信息 | 全部忽略 |
| `outputs/rqvae/` | 后续 SID 实验的可重建产物；需要时再创建 | 全部忽略 |
| `outputs/evaluation/` | Query 向量、行映射、召回结果、指标和运行元信息 | 全部忽略 |
| `outputs/experiments/` | smoke test 和其他实验产物 | 全部忽略 |

只有新流水线真正需要时才创建额外产物目录，不提前创建空目录。

## 数据契约要求

任何特征生成或训练开始前，必须明确并校验：

- POI 和 Query 的字段名、类型、含义与隐私级别；
- 稳定、非空且唯一的 POI 标识；
- Query 到目标 POI 标签的来源和有效性；
- 空值、异常坐标、重复记录和失效 POI 的处理规则；
- 确定性去重、排序和数据切分规则；
- POI 元数据、特征、编码和模型目标之间的行对齐关系。

不得把已删除实验中的字段假设直接用于真实数据。

## 版本与可复现性

- 数据版本使用稳定标识，不在文档中记录真实样本内容。
- 正式实验必须记录数据版本、配置、代码提交、命令、环境和产物位置。
- 可重新生成的特征和训练产物由版本化代码与配置重建。
- 不可替代的源数据快照和重要 checkpoint 保存在批准的内部存储中。

当前向量产物约定：

- `embeddings.npy`：顺序写入的二维标准 NPY，完成后支持 mmap，行顺序与 `poi_ids.jsonl` 一致；
- `poi_ids.jsonl`：每行一个 JSON 字符串形式的 POI ID；
- `manifest.json`：输入指纹、模型参数、shape、dtype、环境和耗时；
- `progress.json`：已安全写入的断点位置和任务状态；恢复时以该位置截断未确认的尾部数据。

SID 产物的文件契约尚未重新确定。新实现必须在 smoke test 前明确 checkpoint、编码映射、指标和 manifest 的格式；所有产物继续放在 `outputs/` 并由 Git 忽略。

Embedding 召回评测产物约定：

- `query_embeddings/*.npy`：按固定评测 JSONL 行顺序保存的 Query 向量；
- `query_embeddings/*_rows.jsonl`：Query 行号、订单标识和目标 POI 的对齐映射；
- `metrics.json`：整体召回指标和 Query 长度分桶指标；
- `retrieval_results.npz`：Top-K POI 行号、相似度和目标排名；
- `run_manifest.json`：Gate 0、模型配置、Faiss 设备、耗时、显存和产物指纹。
