# POI Generative Retrieval

面向地图 POI 生成式检索的代码开发仓库。

仓库已经清除旧公开数据基线、历史实验代码、配置、指标和结果。后续工作从内部真实 POI 数据重新开始，不继承旧实验实现与结论。

## 当前状态

- 本地保留清洗后的内部 POI 数据，位于 `data/`，不进入 Git。
- 本地保留 Qwen3 生成模型和 Embedding 模型，位于 `models/`，不进入 Git。
- 当前已具备 POI 向量构建和 Embedding 精确召回评测代码；SID 阶段已重置，尚未建立新的 RQ-VAE、SFT 或生成式检索实现。
- 新代码应从字段契约、数据读取和小样本测试开始逐步建立。
- 第一版技术路线以 [`方案.md`](方案.md) 为准，按最小可核验步骤逐步实施。

## 下一阶段

Qwen3-Embedding-0.6B 和 Qwen3-Embedding-4B 的北京 POI 全量向量均已完成并通过校验。两者无 Instruction 的 10,000 条订单 Faiss GPU 精确召回对照已完成：0.6B 的 Hit@1/10/20 为 12.03%/28.25%/34.51%，4B 为 9.72%/21.18%/25.12%，暂不自动选择模型。下一阶段从干净状态重新设计 SID 构建，先确认输入 Embedding、量化方案和验收指标，再实现最小 smoke test。

轻量验证：

```bash
python -m unittest discover -s tests -v
python scripts/build_poi_embeddings.py \
  --max-rows 64 \
  --output-dir outputs/experiments/embedding_smoke
```

全量命令：

```bash
python scripts/build_poi_embeddings.py \
  --config configs/embedding_qwen3_0.6b.yaml
```

流水线从配置读取模型路径、batch size、编码缓冲区、最大长度、精度和输出目录。切换 Qwen3-Embedding-4B 时只需使用新配置或命令行覆盖模型路径、batch size、编码缓冲区和输出目录，不修改核心代码。

## Git 边界

Git 只保存源码、配置、测试和必要的长期文档。真实业务数据、下载模型、Embedding、数组、Parquet、checkpoint、日志和实验输出必须留在本地或内部存储。

## 文档入口

- [第一版方案](方案.md)
- [项目状态](docs/PROJECT_STATUS.md)
- [实验进展索引](docs/EXPERIMENT_LOG.md)
- [数据与产物管理](docs/DATA_AND_ARTIFACTS.md)
- [仓库目录说明](docs/REPO_MAP.md)
- [Codex 协作规范](AGENTS.md)
