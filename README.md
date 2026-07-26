# POI Generative Retrieval

面向地图 POI 生成式检索的代码开发仓库。

仓库已经清除旧公开数据基线、历史实验代码、配置、指标和结果。后续工作从内部真实 POI 数据重新开始，不继承旧实验实现与结论。

## 当前状态

- 本地保留清洗后的内部 POI 数据，位于 `data/`，不进入 Git。
- 本地保留 Qwen3 生成模型和 Embedding 模型，位于 `models/`，不进入 Git。
- Qwen3-Embedding-0.6B/4B 已完成 2,337,178 条北京 POI 全量编码和固定 10,000 条订单精确召回对照，当前下游 SID 使用 0.6B Embedding。
- 已完成三种 RQ-VAE 码本容量和 12 个 checkpoint 的北京全量 SID 评估，当前语义 SID 基线为 `1024×3 / epoch 20`。
- 已完成 Geohash6 GID、确定性 Dedup Code 和全局唯一 Final PID 构建。
- 已完成 Qwen3-0.6B 两轮全参数 SFT，以及固定 10,000 条 Validation、Beam=10 的四 checkpoint Trie 约束评测。
- 第一版技术路线以 [`方案.md`](方案.md) 为准，按最小可核验步骤逐步实施。

当前主链路为：POI 文本经 0.6B Embedding 和 RQ-VAE 得到语义 SID，与 Geohash6 GID、Dedup Code 组合为唯一 Final PID；生成模型根据 Query 和用户 GID 生成 PID，并通过全量 Final PID Trie 约束候选合法性。

## 下一阶段

固定 10,000 条 Validation 子集上，2.0 epoch checkpoint 的 HR@1/HR@10/NDCG@10 为 45.77%/84.83%/65.99%，且生成结构和 Final PID 合法率均为 100%。该结果尚不能替代完整 Validation 或 Test。下一步先冻结正式 checkpoint、Beam 和评测口径，再决定是否运行完整 Validation、Beam 规模对比和 Test；之后再开展 no-result 增量分析、RQ-KMeans 对照或偏好优化。

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
