# POI Generative Retrieval

面向 POI 生成式检索的代码仓库，当前重点是 POI 文本/空间特征、RQ-VAE Semantic ID、GID/PID 构造与离线质量评估。

## 当前范围

- `data/processed/mobilitybench/` 提供公开 MobilityBench baseline。
- 历史 MobilityBench 模型、中间数组和实验输出不再作为新实验输入。
- 下一阶段接入滴滴真实 POI 数据，重新定义字段、构建 Embedding、GID/SID，并从头运行 RQ-VAE 与 CAU-RQ-VAE。
- 本仓库只保存代码、配置、测试、正式文档和少量稳定指标；真实业务数据与生成产物不进入公开 Git。

## 仓库结构

```text
configs/                         训练与实验配置
src/                             可复用实现
scripts/                         数据、特征、训练与评估入口
docs/                            长期维护文档
reports/metrics/                 少量稳定历史指标
data/processed/mobilitybench/    公开 baseline
models/                          本地模型（Git 忽略）
outputs/                         实验输出（Git 忽略）
_local_legacy/                   旧本地产物（Git 忽略）
```

## 环境安装

建议使用独立 Python 环境：

```bash
python -m pip install -r requirements.txt
```

模型和数据由使用者在本地准备，脚本不得依赖写死的服务器绝对路径。

## 最小运行流程

先验证公开 baseline：

```bash
python scripts/validate_mobilitybench_data.py
```

完整 POI 侧流水线入口如下；真实数据接入后应先更新字段契约和配置，并先运行小样本 smoke test：

```bash
python scripts/01_build_poi_sid_train_data.py
python scripts/02_embed_poi_text.py --model models/Qwen3-Embedding-0.6B --device auto
python scripts/03_build_geo_features.py
python scripts/04_train_rqvae.py --config configs/rqvae_train.yaml --mode semantic
python scripts/05_export_and_eval_sid.py --mode semantic
```

上述命令会向 `data/embeddings/`、`data/geo/`、`data/rqvae/`、`data/sid/` 和 `outputs/` 写入被 Git 忽略的产物。不要将模型、checkpoint、Embedding、NPY、Parquet 或真实业务数据加入 Git。

## 文档入口

- [项目状态](docs/PROJECT_STATUS.md)
- [实验日志](docs/EXPERIMENT_LOG.md)
- [数据与产物管理](docs/DATA_AND_ARTIFACTS.md)
- [仓库目录说明](docs/REPO_MAP.md)
- [Codex 协作规范](AGENTS.md)
