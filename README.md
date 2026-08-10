# POI Generative Retrieval

面向地图 POI 生成式检索的代码开发仓库。

仓库已经清除旧公开数据基线、历史实验代码、配置、指标和结果。后续工作从内部真实 POI 数据重新开始，不继承旧实验实现与结论。

## 当前状态

- 本地保留清洗后的内部 POI 数据，位于 `data/`，不进入 Git。
- 本地保留 Qwen3 生成模型和 Embedding 模型，位于 `models/`，不进入 Git。
- V1 已完成从 POI Embedding、RQ-VAE、Geohash6/Dedup、Qwen3-0.6B SFT 到固定 10,000 条 Validation Trie 约束评测的最小闭环。
- TIGER 已完成三容量 SID、collision token、历史序列三轮 SFT 和无约束生成评测；GenPOI 旧版与 Centered 修正版均已完成唯一 PID、历史序列三轮 SFT 和 SSP+TCG 评测，Centered epoch 3 的 HR@1/HR@10/NDCG@10 为 51.98%/87.83%/70.7430%。
- GNPR-SID 已完成 content-geo `512×3` 唯一标识、三轮 Qwen3-0.6B SFT 和固定 10,000 条 Validation 无约束生成评测；epoch 3 的 HR@1/HR@10/NDCG@10 为 49.66%/83.07%/67.2390%。
- Query 增强 Embedding 已完成 E1—E4 全量精确召回和 E4 边界补充；E3 候选级动态 α 假设未通过，E4 类别公共 Query 方向残差在 α=0.30、β=0.85 时将固定 10,000 条的 Hit@10/NDCG@10 从 E2 的 51.18%/33.29% 提高到 56.37%/36.40%。β=1.00 已出现回落，最终连续向量冻结为 β=0.85。
- RQ-KMeans 已完成对称 `1024³` 的 Embedding × Quantizer `2×2` 以及 BGE/E4 三档 30-bit 容量分配验证；E4+`4096×1024×256` 以 82.7331%/27.9655% 的唯一 SID/碰撞 POI 成为纯结构最优，E4+`2048×1024×512` 以 79.8731%/31.4021% 冻结为新增 SFT 主候选。
- 实验文档按 V1、TIGER、GNPR-SID、GenPOI 四条方法线组织；Query 增强向量创新单独维护 Embedding 优化实验台账。
- 第一版技术路线以 [`方案.md`](方案.md) 为准，按最小可核验步骤逐步实施。

当前主链路为：POI 文本经 0.6B Embedding 和 RQ-VAE 得到语义 SID，与 Geohash6 GID、Dedup Code 组合为唯一 Final PID；生成模型根据 Query 和用户 GID 生成 PID，并通过全量 Final PID Trie 约束候选合法性。

## 下一阶段

固定 10,000 条 Validation、Beam=10 下，V1/TIGER/GNPR/Centered GenPOI 当前最佳 checkpoint 的 HR@1 分别为 45.77%/51.87%/49.66%/51.98%，HR@10 为 84.83%/87.16%/83.07%/87.83%，NDCG@10 为 65.99%/70.42%/67.24%/70.74%。这些结果尚不能替代完整 Validation 或 Test。四条复现链路已闭环；Query-Augmented Relational SID 创新线已冻结 E4 连续向量与三组新增 SFT：E4-2048 主模型、E4-4096 容量上界、BGE-2048 同布局 Embedding 消融。下一步按统一 TIGER collision-token 协议构建 identifier 并进入同协议 SFT 对照。

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
- [TIGER 复现实验](docs/experiments/TIGER.md)
- [GNPR-SID 复现实验](docs/experiments/GNPR_SID.md)
- [GenPOI 复现实验](docs/experiments/GENPOI.md)
- [数据与产物管理](docs/DATA_AND_ARTIFACTS.md)
- [仓库目录说明](docs/REPO_MAP.md)
- [Codex 协作规范](AGENTS.md)
