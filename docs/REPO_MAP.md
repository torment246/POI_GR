# 仓库目录说明

当前只列出已经存在且职责稳定的路径。

```text
README.md                     项目入口和当前范围
方案.md                       第一版技术方案与实施顺序
AGENTS.md                     Codex 开发、核验和实验记录规则
docs/PROJECT_STATUS.md        当前阶段、已确定事项和下一步
docs/EXPERIMENT_LOG.md        正式实验方法入口和总索引
docs/experiments/V1_BASELINE.md 第一版共享模型选型与完整基线链路
docs/experiments/TIGER.md     TIGER 标识符、训练和评测记录
docs/experiments/GNPR_SID.md  GNPR-SID 输入、RQ-VAE 和后续实验记录
docs/experiments/GENPOI.md    GenPOI GeoPE、训练和 TCG+SSP 评测记录
docs/DATA_AND_ARTIFACTS.md    数据、模型和产物管理规则
docs/REPO_MAP.md              稳定目录职责
configs/                       可复现任务配置
configs/methods/current/       当前已验证主线的方法契约
configs/methods/baselines/     GenPOI、TIGER 和 GNPR-SID baseline 契约
src/poi_gr/                    可复用 Python 实现
src/poi_gr/methods/            方法发现、状态和配置校验
scripts/                       命令行入口
scripts/methods.py             列出、查看并校验方法契约
scripts/build_tiger_identifiers.py  构建 TIGER 固定四层唯一 item identifier
scripts/build_tiger_sft_data.py     构建地图检索适配版 TIGER 历史序列 SFT 数据
scripts/validate_sft_tokenization.py 预检 SFT 长度并构建可复用 Tokenized Cache
scripts/recover_sft_tokenized_cache.py 从已完成的 packed 分片恢复中断的正式 SFT Cache
scripts/build_genpoi_geope_embeddings.py  从已有 BGE-M3 向量构建 GenPOI GeoPE 向量
scripts/build_genpoi_sft_data.py     构建严格因果的 GenPOI 历史序列 SFT 数据
scripts/prepare_gnpr_content_geo_inputs.py 对齐 GNPR 全量文本、类别和 PlusCode6 紧凑输入
scripts/build_gnpr_identifiers.py      仅为 GNPR 碰撞 SID 追加论文 Dedup Token 并构建唯一映射
scripts/build_gnpr_sft_data.py         构建无时间、无用户 ID 的 GNPR 地图检索 Messages 数据
run_train_sft.py                通用 SFT 参数校验、配置解析和训练入口
run_train_sft.sh                       V1 通用 SFT 启动入口
run_train_sft_single_a100_2epoch.sh    V1 单卡 A100 两轮正式 SFT 入口
run_train_tiger_sft_1a100_3epoch.sh    TIGER 单卡 A100、三轮 SFT 入口
run_train_tiger_sft_4x6000d_3epoch.sh TIGER 四卡 RTX PRO 6000D、三轮全参数 SFT 平台入口
run_train_genpoi_sft_4x6000d_3epoch.sh GenPOI 四卡 RTX PRO 6000D、三轮全参数 SFT 平台入口
run_train_genpoi_centered_sft_4x6000d_3epoch.sh Centered GenPOI 四卡 RTX PRO 6000D、三轮 SFT 入口
run_train_genpoi_centered_sft_4xa6000_3epoch.sh Centered GenPOI 四卡 RTX A6000、三轮 SFT 入口
run_train_genpoi_centered_sft_4a100_3epoch.sh Centered GenPOI 四卡 A100、三轮 SFT 入口
run_train_gnpr_sft_4x6000d_3epoch.sh GNPR 四卡 RTX PRO 6000D、三轮 SFT 入口
run_train_gnpr_sft_4a100_3epoch.sh    GNPR 四卡 A100、三轮 SFT 入口
run_train_genpoi_geope32_centered_4x6000d_20epoch.sh Centered GenPOI 四卡 SID/RQ-VAE 构建入口
run_train_genpoi_geope32_centered_a100_20epoch.sh Centered GenPOI 单卡 SID/RQ-VAE 构建入口
run_train_gnpr_content_geo_4x6000d_20epoch.sh GNPR content-geo 三容量并行训练入口
run_train_single_pos_add_distill_mlp.sh mentor 原始训练入口，按要求保留
tests/                         合成数据轻量测试
data/                         本地内部数据，Git 忽略
models/                       本地模型，Git 忽略
outputs/                       本地向量和实验产物，Git 忽略
```

`src/poi_gr/` 当前包含 POI Embedding、RQ-VAE、SID 评估、Geohash/Dedup PID、SFT 数据、Final PID Trie 和生成式评测实现；`src/poi_gr/methods/tiger_identifier.py` 独立实现 TIGER 固定四 Token collision 规则，`src/poi_gr/methods/gnpr_identifier.py` 复用其确定性桶内编号并实现 GNPR 仅碰撞 SID 追加 Dedup Token 的论文规则，`src/poi_gr/methods/gnpr_data.py` 实现无时间、无用户 ID 的 GNPR 地图检索数据契约并以 mmap/逐行扫描控制内存，`src/poi_gr/methods/tiger_data.py` 独立实现地图检索适配版 TIGER 历史序列数据契约，`src/poi_gr/methods/genpoi_geope.py` 独立实现 GenPOI 地理锚点和分段旋转 GeoPE，`src/poi_gr/methods/gnpr_content_geo.py` 独立实现 GNPR 全量静态特征行序对齐和流式融合；`scripts/` 保存对应的命令行入口。稳定组件暂不因方法目录重命名或搬迁，避免破坏已完成实验的命令和产物契约。

`configs/methods/` 用于区分当前主线、论文 baseline 和后续创新。方法配置只声明真实准备状态：已有完整入口为 `ready`，只能复用部分组件为 `partial`，尚未实现为 `missing`。实际实现开始后，可以按方法在 `src/poi_gr/methods/` 下增加独立模块，允许保留少量重复代码；不得把尚未实现的 baseline 标记为可运行。首个创新点确定前不创建空的创新目录或占位实现。

训练平台专属 `.sh` 保留在本地工作区，不进入 Git；代码和实验文档继续保留实际使用的脚本名称、平台入口和历史命令。Embedding、特征准备、分类头和评测不再额外增加一次性 `.sh` 包装。

真实数据、模型和可重新生成的实验产物继续保留在 Git 之外。
