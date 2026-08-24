# POI Generative Retrieval

面向地图 POI 生成式检索的代码开发仓库。

仓库已经清除旧公开数据基线、历史实验代码、配置、指标和结果。后续工作从内部真实 POI 数据重新开始，不继承旧实验实现与结论。

## 当前状态

- 本地保留清洗后的内部 POI 数据，位于 `data/`，不进入 Git。
- 本地保留 Qwen3 生成模型和 Embedding 模型，位于 `models/`，不进入 Git。
- V1 已完成从 POI Embedding、RQ-VAE、Geohash6/Dedup、Qwen3-0.6B SFT 到固定 10,000 条 Validation Trie 约束评测的最小闭环。
- TIGER 已完成三容量 SID、collision token、历史序列三轮 SFT、无约束生成评测和首项三层基础 Bucket 诊断；唯一可展开 Bucket HR@1/HR@10 为 55.53%/88.06%，相对精确 POI 只提高 3.66/0.90pp。GenPOI 旧版与 Centered 修正版均已完成唯一 PID、历史序列三轮 SFT 和 SSP+TCG 评测，Centered epoch 3 的 HR@1/HR@10/NDCG@10 为 51.98%/87.83%/70.7430%。
- GNPR-SID 已完成 content-geo `512×3` 唯一标识、三轮 Qwen3-0.6B SFT 和固定 10,000 条 Validation 无约束生成评测；epoch 3 的 HR@1/HR@10/NDCG@10 为 49.66%/83.07%/67.2390%。
- TIGER、GNPR-SID、Centered GenPOI epoch 3 已完成四类互斥泛化 Validation 10k。四组宏平均 HR@1/HR@10/NDCG@10 分别为 24.9925%/51.3700%/37.6949%、19.0725%/37.7675%/28.2409%、25.2525%/54.1375%/39.0484%；GenPOI 四组 Top-10/NDCG@10 均第一，但 Top-1 优势仅 0.26pp。
- GHR EXP-09/14、五组 Embedding × Quantizer、MMBERT 四层和 GHR 固定五层也已完成相同四类泛化评测，36/36 个创新单元均为 10,000 条无约束 Beam=10。创新方法中 BGE+RQ-KMeans `1024³` 宏平均 HR@1/HR@10/NDCG@10 最好，为 24.6500%/50.4625%/37.0584%，仍低 TIGER 0.3425/0.9075/0.6365pp；MMBERT/GHR 固定五层分别为 22.6125%/44.5925%/33.2601% 和 22.5025%/45.1850%/33.4979%。
- Query 增强 Embedding 已完成 E1—E4 全量精确召回和 E4 边界补充；E3 候选级动态 α 假设未通过，E4 类别公共 Query 方向残差在 α=0.30、β=0.85 时将固定 10,000 条的 Hit@10/NDCG@10 从 E2 的 51.18%/33.29% 提高到 56.37%/36.40%。β=1.00 已出现回落，最终连续向量冻结为 β=0.85。
- 线上 MMBERT Recall checkpoint 已按训练好的 `mean pooling + 768→128 projection` 接入共享流水线；固定 10k 精确召回 Hit@10 为 41.43%，接 TIGER RQ-VAE 后三层 SID 唯一率为 83.3988%，四层 ID 全局唯一。三轮 SFT 和正式评测已完成：固定 epoch 3 无约束 HR@1/HR@10/NDCG@10 为 49.89%/83.84%/67.8842%，整体低于 TIGER；冷目标为 7.93%/18.39%/12.6800%，三项局部高 TIGER 0.85/0.61/0.8673pp。
- RQ-KMeans 已完成对称 `1024³` 的 Embedding × Quantizer 下游 `2×2`、E4 五档 30-bit 容量分配和硬 `category_code` S1 消融。完整四层生成及四类泛化宏平均均未超过 TIGER；五组创新候选的三层 Bucket 诊断已补齐，E4+RQ-KMeans 容量后移 `512×1024×2048` 的唯一 Bucket HR@1/3/5/10 为 59.69/81.55/86.53/88.45%，相对容量前移高 2.31/1.32/1.20/0.71pp，四项配对区间均为正。补充 Teacher-Forcing 后，容量后移三级累计 Top-1 为 58.77%、四组最高，但 `C>0` 条件 C Top-1 仅 70.85%、完整 ID 为 49.89%，确认前三层收益被热点碰撞后缀抵消；后续重点是完整 SID 和碰撞后缀的可学习性，不再扩搜静态码本。
- QGR-SID M2-A 纯词法在 273,937 条 holdout 碰撞请求上把已知桶 HR@1 从 77.5642% 提高到 79.3047%；M2-B GEO/GID、M2-C Query residual 均显著负向。M2-D 可靠词法 HR@1 为 78.9996%，但只保留 full lexical 82.47% 的增量且 92.77% 请求仍 fallback，未过预注册门禁，已停止 mapping/SFT。
- GHR-SID 的 EXP-09 非 Query 最短描述与 EXP-14 桶共享强关系树均已完成 4×RTX PRO 6000D 三轮 SFT、固定 10,000 条 Validation 无约束 Beam=10 和 epoch 3 合法路径约束；共享关系树没有抵消更长目标和变长闭合成本。修订截断口径后的逐层诊断中，二者三级 Bucket HR@10 为 85.43%/84.70%，完整 HR@10 为 82.55%/79.78%；EXP-14 的 Bucket→完整额外损失最大。
- GHR-SID 新候选冻结 TIGER 三层，并以 latent residual+多尺度地理构造全局共享 `32×32` 后缀；2,337,178 条固定五层 ID 全局唯一。三轮 SFT 和正式评测已完成：固定 epoch 3 无约束 HR@1/HR@10/NDCG@10 为 50.48%/84.94%/68.6728%，四类宏平均为 22.5025%/45.1850%/33.4979%。逐层诊断显示其 Bucket→完整 HR@10 损失 1.03pp，与 TIGER 的 0.90pp 接近；主要差距是三级 Bucket HR@10 仍低 TIGER 2.09pp，而不是固定两层后缀本身。
- 实验文档按 V1、TIGER、GNPR-SID、GenPOI、QGR-SID 和 GHR-SID 方法线组织；Query 聚合与新 checkpoint 的向量实验分别维护 Embedding 优化、向量模型优化台账。
- 第一版技术路线以 [`方案.md`](方案.md) 为准，按最小可核验步骤逐步实施。

当前主链路为：POI 文本经 0.6B Embedding 和 RQ-VAE 得到语义 SID，与 Geohash6 GID、Dedup Code 组合为唯一 Final PID；生成模型根据 Query 和用户 GID 生成 PID，并通过全量 Final PID Trie 约束候选合法性。

## 下一阶段

固定 10,000 条 Validation、Beam=10 下，V1/TIGER/GNPR/Centered GenPOI/MMBERT/GHR 固定五层当前最佳 checkpoint 的 HR@1 分别为 45.77%/51.87%/49.66%/51.98%/49.89%/50.48%，HR@10 为 84.83%/87.16%/83.07%/87.83%/83.84%/84.94%，NDCG@10 为 65.99%/70.42%/67.24%/70.74%/67.8842%/68.6728%。GHR 三组逐层诊断已完成，下一步只补 MMBERT 的 gold-prefix、三级 Bucket 与碰撞后缀条件准确率；完成归因前不继续增加静态码本或启动新 SFT。

轻量验证：

```bash
python -m unittest discover -s tests -v
python scripts/embedding/build_poi.py \
  --max-rows 64 \
  --output-dir outputs/experiments/embedding_smoke
```

全量命令：

```bash
python scripts/embedding/build_poi.py \
  --config configs/embedding/embedding_qwen3_0.6b.yaml
```

流水线从配置读取模型路径、batch size、编码缓冲区、最大长度、精度和输出目录。切换 Qwen3-Embedding-4B 时只需使用新配置或命令行覆盖模型路径、batch size、编码缓冲区和输出目录，不修改核心代码。

## 方法组织

当前主线和论文 baseline 的方法契约统一放在 `configs/methods/`：

```text
configs/methods/
├── current/
│   └── main_v1.yaml
├── baselines/
│   ├── genpoi.yaml
│   ├── gnpr_sid.yaml
│   └── tiger.yaml
└── innovations/
    ├── qgr_sid.yaml
    └── ghr_sid.yaml
```

每个配置独立声明数据字段、历史长度、标识符定义，以及 `data / identifier / train / evaluate` 四阶段的准备状态和可复用命令。查看与校验：

```bash
python scripts/methods.py list
python scripts/methods.py show genpoi
python scripts/methods.py validate
```

`ready` 表示已有可运行入口，`partial` 表示只能复用部分现有组件，`missing` 表示必须为该方法单独实现。目录允许 baseline 和创新保留独立实现，不要求为了消除少量重复代码而共享训练或数据处理逻辑。QGR-SID 保留已完成的 Query/GID 代理并保持 `partial`；GHR-SID 的数据、唯一 identifier、Tokenizer/cache、训练和生成评测入口均已实现，方法契约标记为 `implemented`，旧变长与新固定五层候选的实际状态分别记录。

## Git 边界

Git 只保存源码、配置、测试和必要的长期文档。真实业务数据、下载模型、Embedding、数组、Parquet、checkpoint、日志和实验输出必须留在本地或内部存储。

## 第三方依赖

仓库固定 LLaMA-Factory `95ac3f2`。首次克隆后初始化并应用项目所需的 Python 3.10 兼容与末步重复评测修补：

```bash
git submodule update --init third_party/LLaMA-Factory
git -C third_party/LLaMA-Factory apply --unidiff-zero ../LLaMA-Factory-local.patch
```

补丁只需应用一次；`git -C third_party/LLaMA-Factory apply --unidiff-zero --check ../LLaMA-Factory-local.patch` 可在应用前检查状态。

## 文档入口

- [第一版方案](方案.md)
- [项目状态](docs/PROJECT_STATUS.md)
- [实验进展索引](docs/EXPERIMENT_LOG.md)
- [V1 实验](docs/experiments/V1.md)
- [Embedding 优化实验](docs/experiments/EMBEDDING_OPTIMIZATION.md)
- [向量模型优化与下游实验](docs/experiments/VECTOR_MODEL_OPTIMIZATION.md)
- [QGR-SID 方法与实验](docs/experiments/QGR_SID.md)
- [GHR-SID 方法与实验](docs/experiments/GHR_SID.md)
- [TIGER 复现实验](docs/experiments/TIGER.md)
- [GNPR-SID 复现实验](docs/experiments/GNPR_SID.md)
- [GenPOI 复现实验](docs/experiments/GENPOI.md)
- [数据与产物管理](docs/DATA_AND_ARTIFACTS.md)
- [仓库目录说明](docs/REPO_MAP.md)
- [Codex 协作规范](AGENTS.md)
