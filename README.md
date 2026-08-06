# POI Generative Retrieval

面向地图 POI 生成式检索的代码开发仓库。

仓库已经清除旧公开数据基线、历史实验代码、配置、指标和结果。后续工作从内部真实 POI 数据重新开始，不继承旧实验实现与结论。

## 当前状态

- 本地保留清洗后的内部 POI 数据，位于 `data/`，不进入 Git。
- 本地保留 Qwen3 生成模型和 Embedding 模型，位于 `models/`，不进入 Git。
- V1 Baseline 已完成从 POI Embedding、RQ-VAE、Geohash6/Dedup、Qwen3-0.6B SFT 到固定 10,000 条 Validation Trie 约束评测的最小闭环。
- TIGER 已完成三容量 SID、collision token、历史序列三轮 SFT 和无约束生成评测；GenPOI 旧版已完成 GeoPE32、唯一 PID、历史序列三轮 SFT 和 SSP+TCG 评测，Centered 修正版已完成新 PID、全量 Messages 和 Tokenized Cache，尚未正式训练。
- GNPR-SID 已选择 content-geo `512×3`，按论文规则构建 2,337,178 条全局唯一标识；无时间、无用户 ID 的 8,790,513 条地图检索 SFT 数据、1,805 Token 扩词表和正式 Tokenized Cache 均已完成并通过重载核验，GPU smoke 与正式三轮训练尚未运行。
- 实验文档按 V1、TIGER、GNPR-SID、GenPOI 四条方法线组织，不再按 Embedding、SID 和 SFT 阶段拆分。
- 第一版技术路线以 [`方案.md`](方案.md) 为准，按最小可核验步骤逐步实施。

当前主链路为：POI 文本经 0.6B Embedding 和 RQ-VAE 得到语义 SID，与 Geohash6 GID、Dedup Code 组合为唯一 Final PID；生成模型根据 Query 和用户 GID 生成 PID，并通过全量 Final PID Trie 约束候选合法性。

## 下一阶段

固定 10,000 条 Validation、Beam=10 下，V1/TIGER/旧版 GenPOI 当前最佳 checkpoint 的 HR@1 分别为 45.77%/51.87%/50.85%，HR@10 为 84.83%/87.16%/87.13%，NDCG@10 为 65.99%/70.42%/69.88%。这些结果尚不能替代完整 Validation 或 Test。下一步只启动一个与已申请 GPU 类型匹配的 Centered GenPOI 四卡三轮任务；GNPR 先单独完成 20-step GPU smoke，再启动正式三轮 SFT。

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

## 方法组织

当前主线和论文 baseline 的方法契约统一放在 `configs/methods/`：

```text
configs/methods/
├── current/
│   └── main_v1.yaml
└── baselines/
    ├── genpoi.yaml
    ├── gnpr_sid.yaml
    └── tiger.yaml
```

每个配置独立声明数据字段、历史长度、标识符定义，以及 `data / identifier / train / evaluate` 四阶段的准备状态和可复用命令。查看与校验：

```bash
python scripts/methods.py list
python scripts/methods.py show genpoi
python scripts/methods.py validate
```

`ready` 表示已有可运行入口，`partial` 表示只能复用部分现有组件，`missing` 表示必须为该方法单独实现。目录允许 baseline 和后续创新保留独立实现，不要求为了消除少量重复代码而共享训练或数据处理逻辑。首个创新方法开始实现时，再增加 `configs/methods/innovations/<method_id>.yaml` 和对应源码，不预先创建空框架。

## Git 边界

Git 只保存源码、配置、测试和必要的长期文档。真实业务数据、下载模型、Embedding、数组、Parquet、checkpoint、日志和实验输出必须留在本地或内部存储。

## 文档入口

- [第一版方案](方案.md)
- [项目状态](docs/PROJECT_STATUS.md)
- [实验进展索引](docs/EXPERIMENT_LOG.md)
- [V1 Baseline 实验](docs/experiments/V1_BASELINE.md)
- [TIGER 复现实验](docs/experiments/TIGER.md)
- [GNPR-SID 复现实验](docs/experiments/GNPR_SID.md)
- [GenPOI 复现实验](docs/experiments/GENPOI.md)
- [数据与产物管理](docs/DATA_AND_ARTIFACTS.md)
- [仓库目录说明](docs/REPO_MAP.md)
- [Codex 协作规范](AGENTS.md)
