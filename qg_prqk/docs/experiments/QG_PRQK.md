# QG-PRQK 正式实验记录

本文档从 P5 起记录 QG-PRQK 主方法的正式 SID/SFT 实验。P0–P4、Adapter Gate/FULL 和未编号数据阶段的历史过程以
`QG_PRQK_IMPLEMENTATION_STATUS.md` 为准。现有实验的方法定义以
[v2.1-CAT 规范归档](../methods/QG_PRQK_V2_1_CAT_SPEC.md)为准；2026-09-19 确认的
[v3.0 主规范](../methods/QG_PRQK_METHOD_SPEC.md)尚未实现或运行，不新增实验编号、不改写旧指标。统一的实验—代码对应关系见
`../QG_PRQK_CODE_GUIDE.md`。

## 实验索引

| 实验编号 | 日期 | 阶段 | 状态 | 结论 |
|---|---|---|---|---|
| EXP-20260907-01 | 2026-09-07 | P5-CAT 10k sample | COMPLETED | POI-only PRQK 512×3 结构 Gate 通过；当时的 sample 停止点随后已履行 |
| EXP-20260907-02 | 2026-09-07 | P5-CAT 100k medium | REVIEW_REQUIRED | 产物与结构验收通过，但 hard fit 未在 30 轮内收敛且 Top-k refinement 恶化 hard distortion |
| EXP-20260907-03 | 2026-09-07 | P5-CAT 100k hard endpoint 诊断 | REVIEW_REQUIRED | hard60-off 可按原规则收敛；Top-k 恶化 distortion 但改善码均衡与 SID 唯一率，canonical 待用户决策 |
| EXP-20260908-03 | 2026-09-08 | P5-CAT 100k hard60+Top-k5 配对诊断 | COMPLETED | 同一 hard60 endpoint 上 Top-k5 三层均牺牲 distortion、改善码均衡；完整路径比 hard60-off 多 449 个唯一 SID，用户已确认据此先运行新 10k Gate |
| EXP-20260908-04 | 2026-09-08 | P5-CAT hard60+Top-k5 10k sample | COMPLETED | 新参数协议构建和独立校验通过；三层 16/15/16 轮收敛，离散结果与旧 10k 完全一致，停在 sample review |
| EXP-20260909-01 | 2026-09-09 | P5-CAT hard60+Top-k5 active POI full | COMPLETED | 用户明确授权跳过新协议 100k/500k；716,245 POI 三层在 57/52/57 轮收敛、512 code 全激活，独立 validator 通过 |
| EXP-20260909-02 | 2026-09-09 | P6-CAT S1/S2 sample 首次启动 | FAILED | 输入闭包通过后在 S1 被保护性停止；实现错误地跨 warm-up 权重比较 objective，未发布 manifest/_SUCCESS |
| EXP-20260909-03 | 2026-09-09 | P6-CAT S1/S2 graph-closed sample | REVIEW_REQUIRED | 构建与独立验证通过；图/类别改善、code 全激活，但同闭包 P5 A0 对照的 Query distortion 两层退化 |
| EXP-20260909-04 | 2026-09-09 | P6-CAT sample Query distortion 六分支归因 | REVIEW_REQUIRED | `tau=32` 在稀疏 Query code 支持下的强 POI 先验是主要来源，graph 次之；建议保持参数进入 50k medium 验证规模效应，待用户确认 |
| EXP-20260909-05 | 2026-09-09 | P6-CAT S1/S2 active POI direct full | REVIEW_REQUIRED | full 构建与双重独立 validator 通过；图/类别显著改善且 512 码全激活，S1 Query distortion 微降、S2 退化 0.028162，因此停在 full review |
| EXP-20260910-01 | 2026-09-10 | P6-CAT full S2-only 受控诊断 | REVIEW_REQUIRED | 上游 S1 residual 与 S2 质心适配均改善几何；S2 graph-coupled assignment 是退化主因，正式参数保持不变并停在诊断审核点 |
| EXP-20260911-01 | 2026-09-11 | P7-CAT GID-parent S3 full | COMPLETED | 716,245 POI 与 291,590 D3 Query 全量完成；三层 SID distinct 617,887、最大桶 95，512 个 S3 code 全激活 |
| EXP-20260911-02 | 2026-09-11 | P8-CAT A0/A4 静态评测 | REVIEW_REQUIRED | A4 类别结构、碰撞和局部分离改善，但 content-only Prefix Probe 三层及累计指标均低于 A0 |
| EXP-20260911-03 | 2026-09-11 | NoGID S3 与三方可视化 | REVIEW_REQUIRED | NoGID 工程合同成立但未优于 GID-parent；三方法无 code collapse，A4 类别组织强于 A0 |
| EXP-20260911-04 | 2026-09-11 | 双 Final ID 与 SFT Messages | COMPLETED | 两种 Final ID 均覆盖且唯一 716,245 POI；两套 Messages 各 8,790,513 行且逐行对齐 |
| EXP-20260912-01 | 2026-09-12 | 共享词表与双 Tokenized Cache | COMPLETED | 共享新增 3,743 Token；两个 Train/Valid Cache 均零超长、零 target truncation，SFT 尚未启动 |
| EXP-20260913-01 | 2026-09-13—14 | 双分支四卡 A100 SFT | COMPLETED | 两者完成 3 epoch、exit 0；末轮 Valid loss 0.200681/0.339035，生成检索待评测 |
| EXP-20260914-01 | 2026-09-14 | 双分支 epoch-3 固定 10k 与四类泛化 | COMPLETED | 十项各 10,000 条、全部 exit 0；GID / NoGID 固定 HR@10 为 85.19% / 84.67%，泛化宏平均为 46.69% / 46.3325%；冷目标 NoGID 更高 |

## EXP-20260907-01：P5-CAT POI-only PRQK 512×3 sample Gate

### 目标与假设

- 目标：在冻结 active POI BGE 上完成不含 Query、Category、Geo 的 10,000 POI A0 初始化，验证公共方向去除、三层 spherical/cosine K-Means、Top-k 质心细化和 projection residual 的正确性。
- 假设：固定 seed 的 512×3 sample 应完成三层收敛，三层 code 利用率不坍塌，残差不存在零行且与选中质心近似正交；本实验不检验业务检索收益。

### 数据版本

- active POI：716,245 行，北京候选库；sample 从 `numpy.random.Generator(PCG64(seed=42))` 的全库确定性排列前缀取 10,000 行，再按原行号排序。
- POI embedding：`qg_prqk/outputs/inputs/beijing_poi_active_bge_m3_v1/embeddings.npy`，shape `[716245,1024]`、dtype `float16`，SHA256=`0f77fe64dce69e8d875228c747fcd742bb3852db20b657824e47e9d1a5b6d674`。
- embedding manifest SHA256=`4c09d7e180d5a05bc0d88277ff370a98907dcd4a31a3d38eec06c8b06e50c392`；POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- sample 行号 SHA256=`207f7f4be3fd57b6f0cd990467f6191a8bdadc358a865d8710315a6fd73251fa`。
- 公共方向只在完整 716,245 POI 的逐行 L2-normalized BGE 上估计。未读取 Query 图、类别值、Geo、原始 Train、业务 Validation 或业务 Test。

### 代码与工作树

- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`；运行时工作树为 dirty，用户原有改动保持不动，未 commit/push。
- 新增自包含入口：`scripts/build_poi_prqk.py`、`src/qg_prqk/p5_{data,prqk,cli}.py`、`tests/test_p5_prqk.py`；运行时不导入 `src/poi_gr`。
- 成功 attempt02 的 155 个源码、脚本、配置和测试文件快照位于 `outputs/run_control/p5_cat_active_512/sample_010000_attempt02/source_snapshot/`；快照清单 SHA256=`dce6a889e68c141fe3f9361bce01c61b95e179f2d266f5f31f27d86ee9c74c99`。

### 配置

- canonical 配置：`configs/qg_prqk_v2_1_category_active_512x3.yaml`，resolved signature=`b4bc8a786841faf3dcedff375f4e7390efbafec54df14f27a2c3b540e4ed42c2`。
- `codebook_sizes=[512,512,512]`，`metric=cosine`，`init=kmeans++`，`seed=42`。
- `remove_global_direction=true`，`residual=projection`。
- hard fit：`min_iter=8`、`max_iter=30`、`objective_rel_tol=1e-4`、`assignment_change_tol=1e-3`、`patience=2`。
- Top-k refinement：enabled，`topk=5`、`beta=15`、`max_iter=5`；最终重新执行 hard assignment。
- fit/assignment 分别使用 float32/int32；持久化 residual 使用 float16；较小 code id 处理 hard assignment 平局，空 code 使用稳定 farthest-row 重置。

### 命令与环境

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ofs/map_search/hudan/poi_genret/qg_prqk/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/tmp/qg_prqk_p5_sample02_20260907 \
/ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/scripts/build_poi_prqk.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate sample
```

- Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、CUDA 12.8、单卡 NVIDIA RTX A6000。
- `run.exit=0`，外层墙钟 1:16.06、峰值 RSS 2,732,032 KiB；builder 记录 40.72 秒。其余时间主要是输入哈希和独立构建后复核。
- residual 先在本机短临时目录生成，再顺序复制到 QG 输出目录并在同目录原子改名；临时文件不是实验产物，全部正式输出仍在 `qg_prqk/outputs/`。

### 核心指标

| 指标 | S1 | S2 | S3 |
|---|---:|---:|---:|
| final cosine distortion | 0.447954 | 0.539825 | 0.584774 |
| hard iterations | 16 | 15 | 16 |
| hard converged | 是 | 是 | 是 |
| active codes / utilization | 512 / 100% | 512 / 100% | 512 / 100% |
| cluster size min / p50 / p99 / max | 1 / 18.5 / 41.89 / 53 | 5 / 17 / 47.89 / 63 | 7 / 17 / 43 / 53 |
| Kish ESS | 431.98 | 409.39 | 436.64 |
| Gini | 0.2413 | 0.2727 | 0.2230 |
| zero residual rows | 0 | 0 | 0 |
| residual orthogonality max abs dot | 2.50e-6 | 1.88e-7 | 2.08e-7 |
| retained energy mean | 0.6855 | 0.7758 | 0.8159 |

- 三层 SID：distinct 9,934 / 10,000（99.34%）；collision excess=66，涉及 128 个 POI，最大桶 4，桶大小 p99=1。
- Top-k refinement 每层运行 5 轮；三层最终 distortion 均低于最后一次 hard fit 记录值。
- structural Gate：三层完成、assignment 范围正确、codebook/residual 全为有限值、未使用 Query/Category/Geo，全部通过。

### 产物与复核

- 正式输出：`outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/sample_010000/`，约 91 MiB；包含三层 codebook/assignment、S0–S3 residual、合并矩阵、逐层/汇总 metrics、配置、progress、manifest 和 `_SUCCESS`。
- 正式 manifest SHA256=`c9d18df7d1029c9b80442965f31943a31cf4c3ef076de2aa6b9fd5b6d437f4e2`；所有 artifact 的 shape、dtype、bytes 和 SHA256 均登记在 manifest。
- 运行记录：`outputs/run_control/p5_cat_active_512/sample_010000_attempt02/`；console SHA256=`f7fae8329ef9c21fcc017bb8f95daf1bf48fb579b8288ce93c29be6a8a7eef12`。
- 独立 `--validate-only` 退出 0，日志 SHA256=`5d244847269e0e25c9a92dede89dd7c6471b43ce598245c4fe02a094cbbd90c3`；它重验来源、artifact 哈希、精确 assignment 和 projection residual。
- compileall 与 Ruff 通过；启动前全量 105 项 QG 回归通过。修正 I/O 后新增本地暂存 artifact 测试，P5 共 6 项测试通过；阶段结束时全量 106 项 QG 回归再次通过（57.53 秒）。

### 中止尝试

- attempt01 的算法配置和 sample 相同，但 residual 直接使用 `np.memmap.flush()` 写 OrangeFS；每个 20 MiB residual 约耗时 10–11 分钟。运行 22:26 后主动发送 SIGTERM，`run.exit=143`，当时 S1 已提交、S2 residual 尚未提交。
- 该问题是持久化实现退化，不是聚类失败。运行控制和 63 MiB 部分目录完整保留在 `sample_010000_attempt01/` 与 `sample_010000.interrupted_attempt01/`，未覆盖、未删除，也未纳入成功 Gate 指标。
- attempt02 只将 tensor artifact 改成本地生成后顺序回写；数据、选样、seed、算法、超参数和最终 QG 输出位置均未改变。

### 结论与下一步

- P5 10k sample 的结构正确性、收敛、利用率、残差和资源 Gate 通过；这里只证明 A0 工程与量化结构可用，不证明 Query 可预测性、类别结构、Geo 效果或业务检索收益。
- 本实验完成时停止在 `HOLD_FOR_P5_SAMPLE_REVIEW`；该历史停止点随后已获用户确认并完成确定性嵌套的 100k medium。sample manifest 路径及 SHA256 已被 100k 显式绑定，历史产物未覆盖。

## EXP-20260907-02：P5-CAT POI-only PRQK 512×3 100k medium

### 目标与假设

- 目标：在 sample 已通过的同一算法、active POI 与确定性嵌套选样协议下扩大到 100,000 POI，检查收敛、码本利用率、residual、碰撞和资源稳定性。
- 假设：三层 hard fit 应在冻结的 30 轮内满足 objective/assignment 双停止，Top-k refinement 不应系统性恶化最终 hard cosine distortion，且 512×3 不发生码本坍塌。

### 数据版本与来源

- active POI BGE、embedding manifest、POI ID 和公共方向口径与 `EXP-20260907-01` 完全相同；公共方向仍由全部 716,245 POI 估计。
- 100k 行来自同一 `PCG64(seed=42)` 排列的前 100,000 行，并按 active POI 原行号排序；selected rows SHA256=`c1696ac1f5fbe2db990e2adddcf4f0a9e40869212f670ba290a3bfacf46be215`，严格包含 10k sample 集合。
- 显式绑定 sample manifest SHA256=`c9d18df7d1029c9b80442965f31943a31cf4c3ef076de2aa6b9fd5b6d437f4e2`。未读取 Query 图、类别值、Geo、原始 Train、业务 Validation 或业务 Test。

### 代码、配置与工作树

- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户工作树保持不动，未 commit/push。
- 算法配置与 sample 相同：512×3、cosine、kmeans++、seed=42、公共方向去除、projection residual、hard min/max=8/30、objective tolerance=`1e-4`、assignment tolerance=`1e-3`、patience=2、Top-5/β=15/5 轮 refinement。
- 唯一工程变化是 residual NPY 按 8,192 行从 GPU 顺序写入 QG 同目录临时文件后原子改名，替代 sample attempt02 的系统临时盘暂存；不改变向量值、聚类或 assignment。
- 155 文件运行快照位于 `outputs/run_control/p5_cat_active_512/medium_100000_attempt01/source_snapshot/`，清单 SHA256=`3307168e89df08cd249cb15c3c5a6cd62e164b38efc76bd9891ac2dc1ff10b75`。

### 命令与环境

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ofs/map_search/hudan/poi_genret/qg_prqk/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m \
/ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/scripts/build_poi_prqk.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate medium100k \
  --previous-gate-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/sample_010000/manifest.json \
  --previous-gate-manifest-sha256 c9d18df7d1029c9b80442965f31943a31cf4c3ef076de2aa6b9fd5b6d437f4e2
```

- Python/PyTorch/CUDA/GPU 与 sample 相同，单卡 NVIDIA RTX A6000。`TMPDIR` 的绝对路径长度不超过 64 bytes，且全部临时/正式文件均位于 `qg_prqk/outputs/`。
- `run.exit=0`；外层墙钟 1:14.14、峰值 RSS 3,169,852 KiB，builder 记录 37.61 秒、峰值 RSS 2,646.52 MiB，无 OOM/Swap。

### 核心指标

| 指标 | S1 | S2 | S3 |
|---|---:|---:|---:|
| hard iteration 30 objective | 0.466948 | 0.555470 | 0.596859 |
| iteration 30 relative improvement | 3.39e-5 | 1.26e-4 | 6.53e-5 |
| iteration 30 assignment change | 0.00135 | 0.00229 | 0.00187 |
| hard converged | 否 | 否 | 否 |
| Top-k 后 final distortion | 0.474177 | 0.559833 | 0.600312 |
| Top-k 相对 hard-30 变化 | +0.007230 | +0.004362 | +0.003453 |
| active codes / utilization | 512 / 100% | 512 / 100% | 512 / 100% |
| cluster size min / p50 / p99 / max | 26 / 193.5 / 400.67 / 534 | 43 / 183.5 / 452.90 / 584 | 32 / 184 / 397.78 / 786 |
| Kish ESS / Gini | 433.38 / 0.2367 | 427.68 / 0.2429 | 435.27 / 0.2296 |
| zero residual rows | 0 | 0 | 0 |
| residual orthogonality max abs dot | 2.66e-7 | 2.58e-7 | 2.08e-7 |
| retained energy mean | 0.7141 | 0.7940 | 0.8291 |

- 三层 SID：distinct 96,240 / 100,000（96.24%）；collision excess=3,760，涉及 6,795 个 POI，最大桶=23，桶大小 p99=2。
- 三层均无空 code、objective 连续明显升高、非法 assignment、零 residual 或非有限值；结构 Gate 与资源 Gate 通过。
- 三层 hard fit 均在第 30 轮达到上限，未满足 patience=2 的双停止。S1/S3 objective 已低于相对阈值但 assignment change 仍高于 `1e-3`；S2 的 objective 与 assignment 两项均未达阈值。
- 固定 5 轮 Top-k refinement 的每一轮都使当前 hard distortion 继续升高；最终相对第 30 轮分别恶化约 1.55%、0.79%、0.58%。由于 hard 第 30 轮之后还执行了一次 centroid update，表中比较已经是对 Top-k 较宽松的估计，并不支持把该细化判为收益。

### 产物与复核

- 输出：`outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/medium_100000/`，约 797 MiB；manifest SHA256=`8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0`。
- 构建日志 SHA256=`42f8b302a5f064f6957948f9f494e051c88fd906596e5906fe5fbd2d4bb83e79`；独立 validator 日志 SHA256=`064dd4e4a4cadc5bc406c93e34bc9df06d67413ad89551bae930ea753dcc89fc`，构建和 validator 均退出 0。
- 独立 validator 重验 sample 依赖、输入/输出哈希、三层精确 hard assignment 和逐层 projection residual；compileall、Ruff、6 项 P5 测试及全量 106 项 QG 回归（53.41 秒）通过。

### 结论与下一步

- 100k 产物完整且可复现，I/O、资源、码本利用率、桶结构和 residual 均正常；因此不能把本轮标记为失败或损坏。
- 预注册的收敛假设和 Top-k 不恶化假设未通过，实验状态为 `REVIEW_REQUIRED`。当前冻结在 `HOLD_FOR_P5_MEDIUM100K_REVIEW`，不得直接运行 500k/full 或进入 P6。
- 下一步需要用户确认是否允许新增一个不覆盖现有产物的 100k 诊断：保留 hard endpoint 并比较 Top-k off，同时判断是否单独提高 hard max-iter。未经确认不得改变 canonical 参数。

## EXP-20260907-03：P5-CAT 100k hard endpoint 受控诊断

### 目标与假设

- 目标：在不覆盖 `EXP-20260907-02`、不修改 canonical 配置的前提下，用同一 100,000 POI 重放 `hard30+Top-k-off` 和 `hard60+Top-k-off`，区分“30 轮上限不足”与“Top-k refinement 本身的影响”。
- 假设 1：关闭 Top-k 后，hard30 endpoint 的 hard distortion 应优于原 Top-k5 最终 endpoint。
- 假设 2：仅提高诊断分支的 hard 上限至 60，三层应在不改停止阈值的 60 轮内收敛。
- 本实验只用于定位 100k Gate，不宣称形成新的正式 A0 checkpoint，也不扩展到 500k/full/P6。

### 数据版本与来源

- 参考产物固定为 `poi_prqk_a0/medium_100000/manifest.json`，SHA256=`8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0`；入口先复核 `_SUCCESS`、合同、所有 artifact 哈希和准确 assignment，再允许诊断。
- active POI BGE 与前序实验相同：716,245×1,024 float16，embedding SHA256=`0f77fe64dce69e8d875228c747fcd742bb3852db20b657824e47e9d1a5b6d674`，POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- 选样完全复用参考 100k 的确定性行号，selected rows SHA256=`c1696ac1f5fbe2db990e2adddcf4f0a9e40869212f670ba290a3bfacf46be215`。公共方向仍从全部 716,245 POI 估计。
- 只读 active POI embedding 值和参考 P5 100k；未读取 Query 图、Category、Geo、原始 Train、业务 Validation 或业务 Test。

### 代码、配置与工作树

- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`；运行时工作树为 dirty，用户修改未覆盖，未 commit/push。
- 新增 `src/qg_prqk/p5_diagnostics.py`、`src/qg_prqk/p5_diagnostics_cli.py`和 `scripts/diagnose_p5_medium100k.py`；它们只运行两个显式变体，并将 role 锁定为 `P5_A0_CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT`。
- canonical resolved signature 仍为 `b4bc8a786841faf3dcedff375f4e7390efbafec54df14f27a2c3b540e4ed42c2`，参考配置仍为 `max_iter=30` 且 Top-k5 enabled。诊断变体只在内存中分别设为 `30/off` 与 `60/off`；seed=42、kmeans++、cosine、min_iter=8、tolerance、patience、公共方向和 projection residual 不变。
- 161 个源码/脚本/配置/测试文件的运行快照位于 `outputs/run_control/p5_cat_active_512/medium_100000_hard_diag_attempt01/source_snapshot/`，其清单 SHA256=`d59c499f7cf3f999416ce32aae2c4dff83b0d98af9793a6e1e573dfa8f0952ab`。

### 命令与环境

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib /ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_hard_diag_attempt01/source_snapshot/scripts/diagnose_p5_medium100k.py --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml --reference-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/medium_100000/manifest.json --reference-manifest-sha256 8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0
```

- 环境与前序 P5 相同：Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、CUDA 12.8、单卡 NVIDIA RTX A6000。启动前 GPU 0 显存占用为 0 MiB，可用 46,080 MiB。
- 2026-09-07 23:08:22 CST 启动，23:09:44 结束，`run.exit=0`；外层墙钟 1:21.81，峰值 RSS 2,660,068 KiB，无 swap。builder 记录 45.62 秒、峰值 RSS 2,522.89 MiB。

### 核心结果

| 指标 | S1 | S2 | S3 |
|---|---:|---:|---:|
| 参考 hard30 trace distortion | 0.466948 | 0.555470 | 0.596859 |
| 参考 Top-k5 final distortion | 0.474177 | 0.559833 | 0.600312 |
| hard30-off final distortion | 0.466934 | 0.557356 | 0.597651 |
| hard60-off final distortion | 0.466731 | 0.556393 | 0.597649 |
| hard60 - hard30 | -0.000202 | -0.000963 | -0.000003 |
| hard30 收敛 | 否 | 否 | 否 |
| hard60 收敛轮次 | 43 | 39 | 33 |
| hard30 → hard60 assignment 变化率 | 1.429% | 17.430% | 43.610% |

- 原 Top-k5 相对其自身 hard30 trace 的 distortion 三层均恶化，且也三层都差于新 hard30-off endpoint；假设 1 成立。
- hard60-off 三层都在 60 轮之前按原 objective/assignment/patience 规则收敛；假设 2 成立。但 hard30 继续到收敛的误差改善很小，不能单独作为码分布更优的证据。
- SID 结果显示反向权衡：参考 Top-k5 / hard30-off / hard60-off 的 distinct SID 为 `96,240 / 95,786 / 95,722`，唯一率为 `96.240% / 95.786% / 95.722%`，collision excess 为 `3,760 / 4,214 / 4,278`。三个分支的最大桶均为 23，桶 p99 均为 2。
- 参考 Top-k5 的 S1/S2/S3 Kish ESS=`433.38/427.68/435.27`，高于 hard30-off 的 `401.98/404.65/408.12`和 hard60-off 的 `401.23/400.72/413.67`，说明 Top-k 的码均衡改善不是只由 distinct SID 一个指标造成。
- 参考 Top-k5 与 hard30-off 的整路径变化率为 82.855%，hard30-off 与 hard60-off 为 50.675%。S1 是直接配对；S2/S3 的上游 projection residual 已不同，应解释为完整管线分支比较，不是单层孤立因果估计。

### 产物与独立复核

- 输出：`outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/diagnostics/medium_100000_hard_endpoint_v1/`；两个分支分别保存 `[3,512,1024] float32` codebook、`[100000,3] int32` assignment 和完整 metrics。
- manifest SHA256=`00bb291a4f58ed37788382ae7d5abd70ee9294a0749c41c27277afe82d8f1e29`；comparison SHA256=`0e33fd14d6a99e2e2c0a2bc6462cf65aa9bb05e67434aad702d8320b2418abda`；`_SUCCESS` 绑定前者。原 10k/100k manifest 哈希未改。
- 构建日志 SHA256=`ceba881472f05bcf45b9006d7f0992764515b1b3d0e4745697b5d319fb674d04`。独立 `--validate-only` 退出 0，墙钟 50.89 秒、峰值 RSS 2,509,860 KiB、无 swap；校验日志 SHA256=`0c0a64d354080000cf7e3ab5b111ddf6b0496a586f790623f5463f5b1ba730e4`。
- validator 独立重验参考来源/产物哈希、诊断 artifact 哈希、两分支的精确 hard assignment、递归 projection residual 与比较指标。compileall、Ruff、9 项 P5 测试和全部 109 项 QG 回归均通过；阶段结束回归用时 55.56 秒（外层墙钟 1:09.23）。

### 结论与下一步

- `max_iter=30` 在该 100k 上的确不足，但提高上限只解决正式收敛问题，不自动意味最终 SID 更好。
- Top-k5 对 distortion 是负向的，对码均衡和 100k SID 唯一率却是正向的。该实验当时没有 hard60+Top-k5 分支，不能直接判定正式 P5 应采用 hard60-off 还是收敛后的 Top-k5；这一不确定性随后由 `EXP-20260908-03` 补齐。
- 本实验状态为 `REVIEW_REQUIRED`，canonical 配置和原 100k checkpoint 未改；当时停止于 `HOLD_FOR_P5_MEDIUM100K_DIAGNOSTIC_REVIEW`，该历史停止点已获用户确认并履行。后续配对诊断仍未授权 500k/full 或 P6。

## EXP-20260908-03：P5-CAT 100k hard60+Top-k5 配对诊断

### 目标与假设

- 目标：在不覆盖前三个 P5 产物、不修改 canonical 配置的前提下，补齐 `hard60+Top-k5`，直接判断 Top-k5 作用在已经按冻结规则收敛的 hard endpoint 后造成的 distortion 与码分布变化。
- 假设 1：每层 Top-k 分支必须从同一次 hard60 拟合的逐值相同 endpoint 分叉，排除再次初始化或重复 hard fit 带来的差异。
- 假设 2：Top-k5 若仍使 distortion 变差，但同时稳定改善三层码均衡和完整路径 SID 唯一率，则应把它解释为显式均衡正则，而不是误差优化步骤。
- 本实验只用于 P5 参数评审；不生成 canonical checkpoint，不运行 500k/full，也不进入 P6。

### 数据版本与来源

- 100k 行、active POI BGE、完整 716,245 POI 公共方向与 `EXP-20260907-02/03` 完全相同；selected rows SHA256=`c1696ac1f5fbe2db990e2adddcf4f0a9e40869212f670ba290a3bfacf46be215`。
- 显式绑定原 100k manifest SHA256=`8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0`，以及 hard endpoint 诊断 manifest SHA256=`00bb291a4f58ed37788382ae7d5abd70ee9294a0749c41c27277afe82d8f1e29`；正式入口在拟合前重验两者 `_SUCCESS`、合同和所需 artifact 哈希。
- embedding SHA256=`0f77fe64dce69e8d875228c747fcd742bb3852db20b657824e47e9d1a5b6d674`，embedding manifest SHA256=`4c09d7e180d5a05bc0d88277ff370a98907dcd4a31a3d38eec06c8b06e50c392`，POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- 只读 active POI embedding、原 100k 与前序诊断；未读取 Query 图、Category、Geo、原始 Train、业务 Validation 或业务 Test。

### 代码、配置与工作树

- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`；dirty 用户工作树保持不动，未 commit/push。
- 新增 `src/qg_prqk/p5_topk_diagnostic.py`、`src/qg_prqk/p5_topk_diagnostic_cli.py` 和 `scripts/diagnose_p5_medium100k_hard60_topk5.py`；输出 role 固定为 `P5_A0_CONTROLLED_HARD60_TOPK5_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT`。
- 每层只运行一次 Top-k-off hard60；保存其本层 endpoint 后，从同一 codebook/assignment 直接执行 5 轮 Top-k refinement。S2/S3 随 Top-k 分支 residual 继续向下；本层 pre-Top-k 端点仅作直接配对，不冒充完整 all-off SID 路径。
- canonical resolved signature 仍为 `b4bc8a786841faf3dcedff375f4e7390efbafec54df14f27a2c3b540e4ed42c2`，文件仍是 `max_iter=30`、Top-k5 enabled。诊断内存变体只把 `max_iter` 改为 60，其他 seed、停止阈值、公共方向、projection residual、`topk=5/beta=15/max_iter=5` 均不变。
- 成功 attempt02 的 87 个源码、脚本、配置和测试文件快照位于 `outputs/run_control/p5_cat_active_512/medium_100000_hard60_topk5_attempt02/source_snapshot/`，清单 SHA256=`fa5279beca13946505b16b3aac3a94079c62b4254253255b86b78dabad22ee01`。

### 命令与环境

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib /ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_hard60_topk5_attempt02/source_snapshot/scripts/diagnose_p5_medium100k_hard60_topk5.py --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml --reference-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/medium_100000/manifest.json --reference-manifest-sha256 8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0 --endpoint-diagnostic-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/diagnostics/medium_100000_hard_endpoint_v1/manifest.json --endpoint-diagnostic-manifest-sha256 00bb291a4f58ed37788382ae7d5abd70ee9294a0749c41c27277afe82d8f1e29
```

- 环境同前序 P5：Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、CUDA 12.8、单卡 NVIDIA RTX A6000；启动前 GPU 0 占用 0 MiB、可用 46,080 MiB。
- attempt02 于 2026-09-08 11:01:21 CST 启动，11:03:50 结束，`run.exit=0`；外层墙钟 2:29.08、峰值 RSS 2,738,996 KiB。builder 含内联 validator 记录 92.42 秒、峰值 RSS 2,597.69 MiB。

### 核心结果

| 指标 | S1 | S2 | S3 |
|---|---:|---:|---:|
| hard60 收敛轮次 | 43 | 40 | 35 |
| 配对 pre-Top-k distortion | 0.466731 | 0.555623 | 0.595352 |
| hard60+Top-k5 distortion | 0.473968 | 0.560269 | 0.598943 |
| Top-k5 distortion 增量 | +0.007237 | +0.004646 | +0.003592 |
| Top-k5 distortion 相对增幅 | +1.551% | +0.836% | +0.603% |
| assignment 变化率 | 19.442% | 16.802% | 16.247% |
| pre-Top-k Kish ESS | 401.23 | 392.27 | 410.87 |
| Top-k5 Kish ESS | 432.72 | 427.77 | 435.17 |
| pre-Top-k Gini | 0.2840 | 0.2881 | 0.2637 |
| Top-k5 Gini | 0.2377 | 0.2449 | 0.2324 |

- 三层 hard fit 都在 60 轮前按原双停止规则收敛，且保存的 pre-Top-k trace 与 Top-k 分支的 hard trace 完全相同；假设 1 成立。
- Top-k5 在同一 residual、同一 hard endpoint 上三层均提高 distortion，但也三层提高 Kish ESS、降低 Gini；这是直接配对结论，不依赖跨管线推断。
- 完整 `hard60+Top-k5` 路径 distinct SID=`96,171`（96.171%）、collision excess=`3,829`、涉及碰撞 POI=`6,906`、最大桶=22、桶 p99=2。相对完整 `hard60-off` 的 `95,722/4,278/7,618/23`，多 449 个唯一 SID、少 449 个冗余碰撞、少 712 个碰撞 POI，最大桶减少 1；假设 2 成立。
- 相对原 `hard30+Top-k5`，新路径少 69 个唯一 SID、collision excess 多 69；但 hard 已正式收敛。两者 S2/S3 的上游 residual 不同，不把逐层 objective 差异解释为单层因果效果。

### 产物、复核与失败尝试

- 正式输出：`outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/diagnostics/medium_100000_hard60_topk5_v1/`，约 15 MiB；配对 pre-Top-k 与 Top-k5 分别保存 `[3,512,1024] float32` codebook、`[100000,3] int32` assignment 和完整 metrics。
- manifest SHA256=`f33bf2573c141b2ae47bf5e578be3cbe0debd0aed03f23e834a9c982814dbd90`；comparison SHA256=`f43cd344170ec20249fc87a330dac5c66d131a71b8959be3011e7cd1b74aa948`；`_SUCCESS` 绑定 manifest。原 10k、100k 与 hard endpoint 诊断 manifest 哈希均未改变。
- 构建日志 SHA256=`7b1a8631357c33f13968327edacbb6c197205974e8ffd7555520c4fed673802c`。冻结 attempt02 源码独立 `--validate-only` 退出 0，墙钟 1:05.89、峰值 RSS 2,502,656 KiB；日志 SHA256=`6367a71ab874aac017d3458eeb5885f27f7f4791452fa025205bcdda41666764`。
- validator 重验两个上游 manifest/所需 artifact、全部新 artifact 哈希、逐层两个端点的精确 hard assignment、Top-k 路径递归 projection residual、分布与比较指标，以及禁止的数据读取标志。
- compileall、Ruff 与新 CLI `--help` 通过；12 项 P5 测试和全部 112 项 QG 回归通过，全量回归外层墙钟 1:56.06、峰值 RSS 870,100 KiB。
- attempt01 在 49.46 秒后以 `exit=2` 停止：最初实现为验证配对而独立重复执行两次 hard fit，GPU 浮点归约使两次 trace 未逐位一致，触发保护断言。它没有发布正式输出；日志 SHA256=`b014651037c6cf48e97c089d7d74b817b44fe6f4d5c14744d32c31bd5bc45364`，87 文件快照清单 SHA256=`6db7cbb960a833ef41de48f48bfc722446dd1a1c2f306b976b8e0a020cadf1bc`，4 KiB 原子临时目录已连同日志移入 `medium_100000_hard60_topk5_attempt01/failed_atomic_output/` 保留。attempt02 改为 hard 只拟合一次后原位分叉，数据、算法参数和正式目标不变。

### 结论与下一步

- Top-k5 不是 distortion 优化步骤，而是在该 100k 上稳定牺牲约 0.60%–1.55% 局部量化误差、换取更均衡的码使用和更少 SID 碰撞。`max_iter=60` 则解决了三层 hard fit 的正式收敛问题。
- 基于当前方法目标不仅包含重构误差，还需要可用、均衡且低碰撞的层级 token，建议 P5 改为 `max_iter=60` 并保留 `Top-k5/beta=15/5轮`。用户于 2026-09-08 已确认先采用该组合试运行；诊断实验本身的配置和 checkpoint 仍保持不可变。
- 参数审核已完成，实验状态更新为 `COMPLETED`。新组合必须从独立 namespace 的 10k Gate 重新开始；10k 结果未产生前不得运行 100k/500k/full、不进入 P6，也不启动其他下游阶段。

## EXP-20260908-04：P5-CAT hard60+Top-k5 10k sample Gate

### 目标与假设

- 目标：采用用户确认的 `max_iter=60 + Top-k5/beta=15/5轮` 组合，在不覆盖旧 30 轮配置和产物的前提下重新运行 10,000 POI sample，验证配置证据链、输出隔离、三层收敛、残差和 SID 结构。
- 假设：新协议除 hard 最大轮数外与旧 Gate 相同；10k 应在 60 轮前收敛、512 个 code 全部激活、无零 residual，且不会出现离散 assignment 或碰撞退化。本 Gate 不证明 100k/full 效果，也不读取 Query、类别或 Geo。

### 数据版本与配置

- active POI 候选库共 716,245 行；确定性 sample 为 PCG64 seed=42 排列的同一 10,000 行有序前缀，selected rows SHA256=`207f7f4be3fd57b6f0cd990467f6191a8bdadc358a865d8710315a6fd73251fa`。
- POI embedding shape=`[716245,1024]`、float16，SHA256=`0f77fe64dce69e8d875228c747fcd742bb3852db20b657824e47e9d1a5b6d674`；embedding manifest SHA256=`4c09d7e180d5a05bc0d88277ff370a98907dcd4a31a3d38eec06c8b06e50c392`；POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- 配置 `configs/qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml` SHA256=`5dd8a3eb9f017460e4ff8da9c1aadb5f008313114228085c187af4a6998fae71`，resolved signature=`1398e5032f0b9137ff5bc048c7da4af18113c7d0e66f1650c2f0d09e52962b20`。它严格绑定旧 100k、hard endpoint 和 hard60+Top-k5 三份证据 manifest，只把 P5 hard `max_iter` 从 30 改为 60，并保留 512×3、cosine、kmeans++、seed=42、全库公共方向去除、projection residual、min_iter=8、双停止、Top-k5/β15/5轮。

### 代码、工作树与命令

- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`；dirty 用户工作树保持不动，未 commit/push。正式运行使用 90 文件冻结源码快照 `outputs/run_control/p5_cat_active_512_hard60_topk5/sample_010000_attempt01/source_snapshot/`，其 `MANIFEST.sha256` 文件 SHA256=`973f5802a16388f420b1801fb7f4aa7feabc9c1c31dfb238722153689bf18240`。
- 编号校正：仓库根正式索引已占用 `EXP-20260908-01/02`，因此前一 QG 配对诊断从重复的 `EXP-20260908-01` 更正为 `EXP-20260908-03`，本实验使用 `EXP-20260908-04`；只修正文档编号，不修改任何实验产物或哈希。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib /ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/outputs/run_control/p5_cat_active_512_hard60_topk5/sample_010000_attempt01/source_snapshot/scripts/build_poi_prqk.py --config qg_prqk/configs/qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml --gate sample
```

- 同一冻结入口随后追加 `--validate-only` 作独立校验。环境为 Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、CUDA 12.8、单卡 NVIDIA RTX A6000；项目内短 `TMPDIR` 保持不变。
- 外部 GPU 任务释放后，等待器连续三次观测到 0 MiB/0% 才于 15:34:49 CST 启动。外层构建于 15:35:56 退出 0，独立 validator 于 15:36:19 退出 0；manifest 内核心拟合/发布 runtime=`27.78s`、峰值 RSS=`2630.11 MiB`。

### 核心指标

| 指标 | S1 | S2 | S3 |
|---|---:|---:|---:|
| hard 收敛轮次 | 16 | 15 | 16 |
| final distortion | 0.447954 | 0.539825 | 0.584774 |
| active codes / utilization | 512 / 100% | 512 / 100% | 512 / 100% |
| Kish ESS | 431.98 | 409.39 | 436.64 |
| Gini | 0.2413 | 0.2727 | 0.2230 |
| residual zero rows | 0 | 0 | 0 |

- 10,000 行产生 distinct SID=`9,934`（99.34%），collision excess=`66`，涉及碰撞 POI=`128`，最大桶=`4`，桶大小 p99=`1`。
- 三层 assignment 和完整三层 SID 与旧 30 轮 10k Gate 逐值相同；因此所有利用率、ESS、Gini 和碰撞指标相同。S1 final distortion 仅相差 `-5.96e-8`，S2/S3 相同。
- GPU 非确定性归约导致 codebook 不逐字节相同，但最大/平均绝对差仅 `1.40e-7/5.70e-9`；float16 residual 最大差 `6.10e-5`、平均差不超过 `6.60e-9`。这不影响离散 token，但不能宣称连续 artifact 逐值复现。

### 产物、校验与结论

- 输出：`outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0_hard60_topk5_v1/sample_010000/`，约 91 MiB。manifest SHA256=`29a918fb5d3b3727d393ad31dc69e6f95f61fcd49a3036af8d1981473784f4d7`，`_SUCCESS` 精确绑定该哈希。
- run/validate exit 均为 0；构建日志 SHA256=`a1326373c2cf0e4d282ba2e9642bb8fc6a3bea559bcb8c8c73c1ad7b7b3a1038`，独立校验日志 SHA256=`8a725a547630c996d1dfda8e52c8057014de7c7ab70c43f07252302f963da21b`。validator 重验来源、配置证据、全部 artifact 哈希、精确 assignment、projection residual、分布指标和禁止数据读取标志。
- 启动前 115 项 QG 回归、compileall、Ruff、CLI `--help` 和真实 dry-run 均通过。manifest 明确记录 Query graph、Category、Geo、raw Train、业务 Validation/Test 均未读取。
- 结论：新协议 10k 工程与结构 Gate 通过，没有离散退化；但三层仍在 15–16 轮收敛，故 10k 不能验证把上限提高到 60 的实际收益。当前停止于 `HOLD_FOR_P5_HARD60_TOPK5_SAMPLE_REVIEW`；唯一下一步是等待用户审核并决定是否运行同协议 100k，不自动进入 500k/full 或 P6。

## EXP-20260909-01：P5-CAT hard60+Top-k5 active POI full

### 目标与假设

- 目标：按用户 2026-09-09 的明确指令，跳过新协议 100k/500k Gate，直接在全部 716,245 条 active POI 上构建 P5 POI-only A0，并重点审核 S1 的收敛、码使用和 residual；本实验不进入 P6，也不读取 Query、类别、Geo 或业务 Validation/Test。
- 假设：三层 hard fit 应在 `max_iter=60` 内按冻结双停止规则收敛，512 个 code 全部激活，不产生零 residual；S1 不应出现 code collapse。POI-only A0 的全路径碰撞只作 P6/P7 前的内部对照，不代表最终 QG SID。

### 数据版本与协议覆盖

- active POI 共 716,245 行，full 使用冻结 active 原行序，selected rows SHA256=`3ff99cf4dda7e338504b61aad1ac9e2732b570db975c1393d209df587ed0c349`。
- POI embedding shape=`[716245,1024]`、dtype=`float16`，SHA256=`0f77fe64dce69e8d875228c747fcd742bb3852db20b657824e47e9d1a5b6d674`；POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- 参数仍为 512×3、cosine、kmeans++、seed=42、完整 active POI 公共方向去除、projection residual、min/max iter=8/60、双停止和 Top-k5/β15/5轮。配置 SHA256=`5dd8a3eb9f017460e4ff8da9c1aadb5f008313114228085c187af4a6998fae71`，resolved signature=`1398e5032f0b9137ff5bc048c7da4af18113c7d0e66f1650c2f0d09e52962b20`。
- 为落实用户覆盖，CLI 新增显式 `--direct-full-from-sample-user-authorized`；默认 Gate 顺序没有放宽。full manifest 绑定已验收 sample manifest SHA256=`29a918fb5d3b3727d393ad31dc69e6f95f61fcd49a3036af8d1981473784f4d7`，并记录 `USER_CONFIRMED_20260909_DIRECT_P5_FULL_FROM_SAMPLE`、仅限 `P5_A0_ONLY`、跳过 `medium100k/medium500k`。

### 代码、命令与环境

- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`；dirty 用户工作树保持不动，未 commit/push。成功运行使用 90 文件冻结快照 `outputs/run_control/p5_cat_active_512_hard60_topk5/full_716245_direct_from_sample_attempt02/source_snapshot/`，其 `MANIFEST.sha256` SHA256=`ec96b6d22ffd821c8f54490d667d0c09bdf3b7e428fba42294b22a36698e5f8c`。
- 环境为 Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、单卡 NVIDIA RTX A6000；启动前 GPU 0 MiB/0%，`TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5f` 长度 56 字节。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5f OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib /ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/outputs/run_control/p5_cat_active_512_hard60_topk5/full_716245_direct_from_sample_attempt02/source_snapshot/scripts/build_poi_prqk.py --config qg_prqk/configs/qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml --gate full --previous-gate-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0_hard60_topk5_v1/sample_010000/manifest.json --previous-gate-manifest-sha256 29a918fb5d3b3727d393ad31dc69e6f95f61fcd49a3036af8d1981473784f4d7 --direct-full-from-sample-user-authorized
```

### 核心指标

| 指标 | S1 | S2 | S3 |
|---|---:|---:|---:|
| hard 收敛轮次 | 57 | 52 | 57 |
| pre-Top-k hard distortion | 0.468387 | 0.556877 | 0.596786 |
| Top-k5 final distortion | 0.475771 | 0.561065 | 0.599759 |
| Top-k5 相对增幅 | 1.577% | 0.752% | 0.498% |
| active codes / utilization | 512 / 100% | 512 / 100% | 512 / 100% |
| Kish ESS | 444.82 | 431.33 | 427.15 |
| Gini | 0.2177 | 0.2367 | 0.2474 |
| residual retained energy mean | 0.7157 | 0.7952 | 0.8287 |
| residual zero rows | 0 | 0 | 0 |

- S1 cluster size min/p50/p90/p99/max=`180/1385.5/2079.7/2869.89/3430`，没有空 code 或极端单桶坍塌；最大 residual 正交误差=`3.22e-7`。
- 716,245 条 POI 产生 distinct SID=`606,045`（84.6142%），collision excess=`110,200`（15.3858%），碰撞 POI=`177,206`（24.7410%），桶大小 p50/p90/p95/p99/max=`1/2/2/4/208`。
- Top-k5 在 full 上仍逐轮提高 hard distortion，与 100k 诊断方向一致；本次没有构建 full Top-k-off 对照，因此不能声称 Top-k 在 full 上带来多少唯一率增益。P5 的解释仍是均衡正则，不是 distortion 优化。

### 产物、校验与结论

- 输出位于 `outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0_hard60_topk5_v1/full_716245/`，约 5.5 GiB。manifest SHA256=`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`，metrics SHA256=`557ee4a891baa109a64409d78e23dc32aa923d0c3f224ee3d7789234399a9c`，S1 metrics SHA256=`7c342864ae1cf1ca5a40dce8ec3c3a19b5ae19927be21d9ae479b33fca3b1423`；`_SUCCESS` 精确绑定 manifest。
- 构建从 11:33:18 到 11:37:30，外层 `run.exit=0`；manifest 内拟合、发布及内联校验前阶段 runtime=`133.73s`、峰值 RSS=`2662.56 MiB`。独立 `--validate-only` 重试从 11:49:45 到 11:53:15，`validate_retry01.exit=0`，日志 SHA256=`925e0ae91ec3948c3f33e46298d736ef3111fc09deafdde21e5d8b7685862d9c`。
- attempt01 仅在运行前源码快照校验阶段因工作目录错误退出 72；未启动 Python/GPU、未创建 full 输出，完整记录保留。attempt02 构建完成后的首次外层 validator 已输出完整结果，但 wrapper 未写退出码便消失，因此不采信其终态；上述 retry01 提供了明确退出 0 的独立验收。
- 启动前 16 项 P5 测试、compileall、CLI `--help` 和 full dry-run 均通过；完成后全部 116 项 QG 合成回归、Ruff 和 compileall 通过。validator 重验输入、全部 artifact 哈希、逐层精确 assignment、projection residual、分布指标和禁止数据读取标志。
- 结论：P5 full A0 构建与结构验收通过，S1 未坍塌；但 S1/S3 均在第 57 轮才收敛，60 轮只有 3 轮余量。P5 不重跑，实验完成时停止在 `HOLD_FOR_P5_HARD60_TOPK5_FULL_REVIEW`。后续用户已确认 P6/P7 复合目标上限为 60；该后续决策不修改本实验配置、manifest 或结论。

## EXP-20260909-02：P6-CAT S1/S2 sample 首次启动失败

### 目标、数据与配置

- 目标：在用户确认的图闭包 sample 上首次运行 P6 S1/S2 双视图交替优化，验证类别软代价、Query–POI 图对齐、Query 质心收缩和 projection residual；不进入 medium/full/P7。
- 数据：P5 seed=42 的 10,000 个基础 POI，加上 P4 前 1,000 个 Query 的全部 S1/S2 target，闭包后为 11,459 POI；完整保留 2,610 条 S1/S2 边。输入只来自 active POI、Train-only P4 图和 P5 full A0，不读取业务 Validation/Test。
- 配置：`configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml`；512×3、hard max_iter=60、Top-k5/β15/5轮及既定 S1/S2 权重不变。Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树 dirty，未 commit/push。

### 命令、环境与实际状态

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p6s1 \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/build_poi_query_category_prqk.py \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml \
  --gate sample
```

- 环境：单卡 NVIDIA RTX A6000；启动前显存占用 0 MiB、可用 46,080 MiB。进程进入 S1 后退出码为 2，GPU 正常释放；没有 OOM、输入缺失或图闭包错误。
- 最后已验证进度：S1 收到 11,459 POI、1,000 Query、1,495 条 S1 边；尚未完成 S1，因此没有 checkpoint、metrics、manifest 或 `_SUCCESS`，不得作为有效 sample 结果。
- 失败输出完整移至 `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_prqk_s1_s2_hard60_topk5_v1/sample_001000q_011459p.failed_exp_20260909_02/`；日志、PID 和退出码位于 `outputs/run_control/p6_sample_hard60_topk5_v1.failed_exp_20260909_02/`，日志 SHA256=`93e6f25f02799afb63e3e6d84b14378e748820ccab6e7806cb76a5385c3d8190`。

### 原因、修正与下一步

- 原因：warm-up 在第 3—6 轮逐步增大 Query/graph/category 权重，复合 objective 的标尺随轮次变化；首次实现仍把相邻轮原始 objective 直接比较，连续三次权重引起的上升被误判为固定目标恶化。
- 修正：连续升高保护只在第 6 轮 warm-up 结束、全部权重固定后开始计数；收敛判断、权重、数据、seed、初始化和其余算法均不变。新增回归测试明确第 3—6 轮不累计 increase streak、第 7 轮起恢复保护。
- 状态：`FAILED`，保留为真实失败实验。修正通过轻量测试后，以新的空输出目录重跑同一 P6 sample；不覆盖本次失败证据，不进入 medium/full/P7。

## EXP-20260909-03：P6-CAT S1/S2 graph-closed sample

### 目标与假设

- 目标：在固定图闭包 sample 上完整运行 P6 S1/S2，验证 POI/Query 双质心共享 token、层级图对齐、S1 coarse/S2 conditional-fine 类别软代价、Query 质心收缩、warm-up、Top-k5 和逐层 projection residual。
- 假设：相对完全相同 POI/Query/边上的 P5 A0，P6 应提高图一致性与类别结构，同时保持 Query distortion 不退化、512 code 不坍塌且两层 prefix 结构无灾难性退化。该 Gate 不使用 Geo，不运行 medium/full/P7。

### 数据、配置与工作树

- 基础 POI 为 P5 seed=42 确定性 10,000 行；Query 为 P4 稳定行序前 1,000 行，保留全部 S1/S2 边并将 target POI 并入闭包。最终为 11,459 POI、1,000 Query、2,610 条边，其中 S1/S2 为 1,495/1,115 条；未截边、未重归一。
- P4 full manifest SHA256=`f6f5a6b4a6f5ef78ff8eb1007e8b566a17fe5a911aae215185b0ed156ab6329d`；P5 full A0 manifest SHA256=`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`。只读取 Train-only Query 图、active POI 类别和 P5 初始化，不读取业务 Validation/Test。
- 配置 `configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml` SHA256=`0b6464c0c212a79b723b8c6718c66e7ab59f22bb0ebff7411b98f1be7f7c057c`，resolved signature=`9eeced7f05213c863df0b16d36377e8503573720b566fe1e1c4dae28e348fdac`。S1 Query/graph/category 权重 `0.05/0.05/0.08`，S2 为 `0.10/0.10/0.08`；category smoothing `32/16`、Query centroid shrinkage `32`、hard max_iter=60、Top-k5/β15/5轮。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`；dirty 用户工作树保持不动，未 commit/push。全部新源码、配置、测试、日志和产物均位于 `qg_prqk/`。

### 命令与环境

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p6s1 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/build_poi_query_category_prqk.py \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml \
  --gate sample
```

- 环境：Python 3.10、PyTorch 2.9.1+cu128、单卡 NVIDIA RTX A6000；启动前显存 0 MiB/可用 46,080 MiB。builder runtime=`49.17s`、GPU peak allocated=`335.60 MiB`、峰值 RSS=`1763.89 MiB`，`exit=0`。
- 独立构建 validator 与独立 Gate evaluation validator 均在新进程退出 0。启动前及修正后的 compileall、Ruff、14 项 P6 专项测试均通过；完整 QG 回归结果见本次状态更新验证记录。

### 核心指标与同闭包 P5 A0 对照

| 指标 | P5 A0 S1 | P6 S1 | P5 A0 S2 | P6 S2 |
|---|---:|---:|---:|---:|
| Query cosine distortion | 0.621731 | 0.683140 | 0.587027 | 0.654598 |
| 加权图一致率 | 28.8821% | 94.4393% | 32.4789% | 89.7598% |
| POI active codes | 512 | 512 | 512 | 512 |
| hard iterations / converged | — | 12 / 是 | — | 9 / 是 |
| POI residual zero rows | — | 0 | — | 0 |

- S1 coarse token purity 从 `77.7293%` 提高到 `81.2811%`（+3.5518pp），conditional entropy 从 `0.629971` 降到 `0.512859`。
- S1+S2 fine path purity 从 `96.3086%` 提高到 `96.6053%`（+0.2967pp），conditional entropy 从 `0.054520` 降到 `0.051314`。
- P5/P6 distinct S1+S2 prefix 为 `10,403/10,387`，P6 少 16；collision excess 为 `1,056/1,072`，P6 多 16。两者桶 p50/p99 都为 `1/3`，P6 最大桶由 12 降到 8，因此属于轻微唯一率代价，不是桶坍塌。
- P6 Query assignment 使用码数为 S1/S2 `351/390`；POI code 两层均 512 全激活。S1/S2 Query residual retained energy mean=`0.862825/0.834302`，零 residual 均为 0。

### 产物、失败前序与结论

- P6 输出：`outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_prqk_s1_s2_hard60_topk5_v1/sample_001000q_011459p/`，约 81.50 MiB；manifest SHA256=`10f40347f60e38289cb9b0405787433e176c0ab051b718773d9e3e2ab36a612a`，`_SUCCESS` 精确绑定。
- 对照输出：同一方法目录下 `evaluations/sample_001000q_011459p_vs_p5_a0_v1/`；evaluation manifest SHA256=`93b15fcd2fe559b838080492470bbaee798982bc1837374c6bea2c8e3e718075`，comparison SHA256=`9fcd0d59e2c57ea944e219d1ab56acd3d15b27f305cf88bd82fb48c6884ac242`。
- 成功日志位于 `outputs/run_control/p6_sample_hard60_topk5_v1_attempt02/`；构建日志 SHA256=`03f83b0dcc0224d7c4de9720223c2c77c51dfec29234bb5f38e64a44aa1e4e74`，evaluation validator 日志 SHA256=`a639b451b72790dd1d30ee52a22eba1e7a6ca87769e795eeacc05cfaa306699c`。首次失败 `EXP-20260909-02` 单独保留，未覆盖。
- 最终核验：训练产物和对照产物独立 validator 均退出 0；完整 QG 合成回归 `130 passed`，Ruff、compileall、两个新增 CLI 的 `--help` 与源码清单哈希校验均通过。
- 结论：图对齐和类别锚定生效，code 利用率与桶结构安全；但公平 P5 A0 对照下 Query distortion 在 S1/S2 分别增加 `0.061408/0.067571`，预注册的 Query distortion 改善条件未通过。实验标记为 `REVIEW_REQUIRED`，停在 `HOLD_FOR_P6_SAMPLE_REVIEW`；用户确认前不调整权重、不运行 medium/full，也不进入 P7/P8。

## EXP-20260909-04：P6-CAT sample Query distortion 六分支归因

### 目标与假设

- 目标：在与 `EXP-20260909-03` 完全相同的 POI、Query、S1/S2 边和 P5 初始化上，隔离 graph alignment、Query 质心 POI 先验、Top-k5 与 category 对 Query distortion 的影响；不调参、不发布候选 checkpoint、不进入 medium/full/P7。
- 假设：1,000 Query 分配到 512 个码后单码支持过稀，固定 `tau=32` 可能让 Query 质心主要由 POI 先验决定；graph 与 Top-k5 可能提供次级影响，category 不应直接主导 Query assignment。

### 数据、代码与分支

- 冻结数据仍为 11,459 POI、1,000 Query、2,610 条完整 S1/S2 边；canonical P6 manifest SHA256=`10f40347f60e38289cb9b0405787433e176c0ab051b718773d9e3e2ab36a612a`，同闭包 P5 A0 evaluation manifest SHA256=`93b15fcd2fe559b838080492470bbaee798982bc1837374c6bea2c8e3e718075`。未读取业务 Validation/Test，也未读取 Geo。
- 新增 `src/qg_prqk/p6_attribution.py`、`src/qg_prqk/p6_attribution_cli.py`、`scripts/diagnose_p6_sample_attribution.py` 和合成测试。诊断 role 固定为 `CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT`；canonical 配置、正式 P6 checkpoint/manifest 和全部历史产物未修改。
- 六个分支均从同一冻结 P5 endpoint 重新开始：`canonical`、`topk_off`、`graph_off`、`near_zero_query_prior`、`graph_off_near_zero_query_prior`、`category_off`。近零先验使用 `tau=1e-6`，只为保留空 Query code 的 POI 回退并近似移除占用码 prior，不代表候选参数。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户工作树保持不动。运行前源码快照位于 `outputs/run_control/p6_sample_attribution_v1_attempt01/source_snapshot/`，快照清单 SHA256=`e6343df0eb5bf08031b835cdbbb475cf212c7a1d047837ae2f203b1ab496f02c`。

### 命令与环境

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p6d \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
  PYTHONPATH=qg_prqk/outputs/run_control/p6_sample_attribution_v1_attempt01/source_snapshot/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p6_sample_attribution_v1_attempt01/source_snapshot/scripts/diagnose_p6_sample_attribution.py \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml
```

- 环境：Python 3.10、PyTorch 2.9.1+cu128、单卡 NVIDIA RTX A6000；启动前显存 0 MiB/可用 46,080 MiB。六分支计算段 `39.09s`，完整命令含 OrangeFS 冻结输入读取与内联验收为 `2:33.76`，GPU peak allocated=`333.89 MiB`、峰值 RSS=`1842.25 MiB`，`exit=0`。
- 冻结源码独立 `--validate-only` 在新进程退出 0，逐分支重验 artifact 哈希、assignment shape/range、POI/Query distortion、图一致率、projection residual 和 canonical parity。启动前 14 项 P6 相关测试、Ruff、compileall、CLI `--help` 与 dry-run 均通过；完成后完整 QG 合成回归 `133 passed`。

### 核心结果

| 分支 | S1 Query distortion | S1 图一致率 | S2 Query distortion | S2 图一致率 | Query active code S1/S2 |
|---|---:|---:|---:|---:|---:|
| P5 A0 probe | 0.621731 | 28.8821% | 0.587027 | 32.4789% | — |
| canonical | 0.683140 | 94.4393% | 0.654598 | 89.7598% | 351/390 |
| topk_off | 0.666788 | 94.1355% | 0.646359 | 90.3050% | 347/383 |
| graph_off | 0.588880 | 37.5139% | 0.582160 | 38.5289% | 348/384 |
| near_zero_query_prior | 0.262386 | 63.7304% | 0.230850 | 60.7922% | 492/500 |
| graph_off + near_zero prior | 0.207477 | 22.4137% | 0.212819 | 18.5371% | 490/504 |
| category_off | 0.682127 | 94.3294% | 0.655920 | 89.8049% | 352/393 |

- canonical 占用 Query code 的支持数中位数在 S1/S2 均为 2；`tau=32` 下 Query-weighted POI prior mass 为 `85.20%/90.85%`，Query 数据自身质量仅为 `14.80%/9.15%`。canonical 复跑与正式 P6 的两层 POI/Query assignment 逐值一致，distortion 差仅 `5.96e-8/0`。
- 在其他参数不变时，`tau=32` 相对近零 prior 增加 distortion `0.420753/0.423748`，同时增加图一致率 `30.7090/28.9676pp`；graph 相对 `graph_off` 增加 distortion `0.094260/0.072438`，同时增加图一致率 `56.9254/51.2309pp`。二者存在正交互，不能把总差值简单线性相加。
- Top-k5 相对 `topk_off` 增加 distortion `0.016352/0.008239`；S1 图一致率只增加 `0.304pp`，S2 反而降低 `0.545pp`。它是次要来源。关闭 category 后 distortion 只变化约 `-0.001013/+0.001321`、图一致率变化约 `-0.110/+0.045pp`；category 不是 Query distortion 的来源，S1 coarse purity 仍由 category 提高 `3.403pp`。
- 近零 prior 的低 distortion 伴随 Query active code 增至 `492/500`，接近 512 容量上限，说明它在 1k 样本上主要通过近似逐码记忆 Query 获益；不能据此把 canonical `tau` 改成近零。

### 产物、结论与下一步

- 输出：`outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_prqk_s1_s2_hard60_topk5_v1/diagnostics/sample_001000q_011459p_attribution_v1/`，约 49 MiB；manifest SHA256=`cd8e9e61274fd9632b48778fcac5827dfa3dd5773a4e33fa22726772743ac441`，comparison SHA256=`0412aa78cfbf033d23ff302693d90ad033f143ef5c3a361446158e7c3e167f2a`，`_SUCCESS` 精确绑定。
- 运行日志 SHA256=`2c06882a1958f20afce7836e876f6ec27e8c73b5cf965c1c0b18f55e35b50a13`，独立 validator 日志 SHA256=`d0fa9029df61af2b6854cf89bf8eeb62fd669f82d888c09d988b35cac0a0e9d3`；构建和 validator 的退出码文件均为 0。
- 结论：1k Gate 的 Query distortion 退化主要是“固定 `tau=32` × 每码 Query 支持过少”的尺度效应，graph 提供次级代价，Top-k5 较小，category 基本无关。50k medium 若码使用近似均衡，每码平均 Query 支持约 98，先验有效占比预计显著下降；这是待验证推断，不是已运行结论。
- 当时建议：不要用近零 prior 小样本分支改参数，先用更大规模验证。后续用户于同日明确把协议改为“小规模只跑通、随后直接 full、指标和调参只依据 full”，因此未启动 50k medium；该后续决策由 direct-full-v2 合同和下一条正式实验记录承接。

## EXP-20260909-05：P6-CAT S1/S2 active POI direct full

### 目标、数据与配置

- 目标：按用户明确授权跳过 P6 medium，在全部 716,245 条 active POI 和 342,879 个 Train-only Query 上构建 S1/S2 双视图码；完成后只运行同规模 P5 A0 对照和独立验收，停在 `HOLD_FOR_P6_FULL_REVIEW`，不进入 P7。
- 数据：固定读取 P4 full manifest SHA256=`f6f5a6b4a6f5ef78ff8eb1007e8b566a17fe5a911aae215185b0ed156ab6329d`、P5 full manifest SHA256=`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e` 和 P6 sample manifest SHA256=`10f40347f60e38289cb9b0405787433e176c0ab051b718773d9e3e2ab36a612a`。S1/S2 图共 912,980 条边，闭包新增 POI=0；不读取业务 Validation/Test。
- 配置：`configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml`，SHA256=`3ef96a0445e3e37e30e1c8d86c3a18b65fa0ac5eaae56cd81123fdc03b61acf7`，resolved signature=`2e7aa687648d3ba7e2737ade36d6d60f273c3dfe08efe20804476bb01dcc8fc0`。512×3、seed=42、S1/S2 权重、`tau=32`、warm-up、hard max_iter=60、Top-k5/β15/5轮与 sample 完全一致。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`；dirty 用户工作树保持不动，未 commit/push。所有配置、源码、运行控制和失败输出都位于 `qg_prqk/`。

### 启动、失败与内存修复

- attempt01 在 detached shell 进入 Python 前消失，console 和算法输出均为空；attempt02 完成输入准备并写出初始 artifact 后 detached shell 消失，未进入 S1、没有退出码、manifest 或 `_SUCCESS`。两次均不提供算法结果，运行控制和 attempt02 的 partial output 已分别原样保留。
- attempt03 使用已验证的 `nohup + setsid` 启动，进入 S1 后退出码=`137`，console 最后一条算法事件为 716,245 POI、342,879 Query、514,500 条 S1 边。外层 `/usr/bin/time` 墙钟 `3:03.99`、最大 RSS=`10,106,060 KiB`；同一时刻主机内核记录 memory-cgroup OOM 并杀死 Python。没有层级 checkpoint、metrics、manifest 或 `_SUCCESS`，不得视为 full 结果。
- attempt03 的 2.1 GiB 未完成输出已移至 `outputs/run_control/p6_full_hard60_topk5_direct_v2_attempt03/failed_partial_output/`；console、exit code、源码快照和 `launch_failure.txt` 同目录保存，未覆盖 attempt01/02 或历史 sample。
- 根因是工程实现同时保留全量 float16 源、float32 搬运/落盘副本和逐 POI 的 `N×512` dense category cost，而不是 GPU OOM 或算法发散。修复把输入搬运和 NPY 写入改成 8,192 行分块，并将类别代价存成“类别/父码-细类 pair → 512 cost”的紧凑表，在 assignment chunk 内按行查表；公式、float32 cost、assignment、objective 和超参数不变。
- S1/S2 新旧类别代价在固定随机合成数据上逐元素一致，S2 三组 smoothing 的最大绝对误差均为 0。完整 QG 合成回归 `140 passed`，Ruff、compileall、`git diff --check` 和历史 P6 sample validator 均通过。
- 修复后 full dry-run 退出 0，规模仍为 716,245 POI、342,879 Query、912,980 边，未写正式输出；墙钟 `22.14s`，最大 RSS 从修复前 dry-run 的约 4,979 MiB 降至 `1,302.08 MiB`。
- attempt04 把配置文件也放入快照路径，因 project root 参与 base signature 而在读数据前保护性退出 2；无正式输出。attempt05 改为冻结快照源码配 canonical 哈希锁定配置，成功完成。早期监控曾因父 shell PID 消失和日志延迟回填误判 attempt05 失败，随后已在同目录 `launch_failure.txt` 中保留并更正；attempt06 被已有 `_SUCCESS` 的 overwrite guard 安全拒绝，未修改产物。

### full 构建指标

| 指标 | S1 | S2 |
|---|---:|---:|
| hard iterations / converged | 39 / 是 | 12 / 是 |
| Top-k5 final POI distortion | 0.479498 | 0.559732 |
| Top-k5 final Query distortion | 0.623430 | 0.620627 |
| POI active codes | 512 | 512 |
| Query active codes | 512 | 512 |
| 加权图一致率 | 81.6757% | 80.2796% |
| POI assignment 相对 P5 A0 变化率 | 20.3397% | 15.3357% |
| POI residual zero rows | 0 | 0 |
| Query residual zero rows | 0 | 0 |

- 两层 hard fit 都按原双停止规则收敛，未触达 60 轮上限；POI/Query 两层均使用全部 512 个码，没有空码或 residual collapse。
- S1/S2 最终图加权一致率为 `0.816757/0.802796`；S1 coarse token purity=`0.868014`，S1+S2 fine path purity=`0.793472`。这里的 S2 类别结果使用层级 path partition，不把 S2 token 单独解释为全局 fine category。
- P6 S1+S2 distinct prefix=`147,981/716,245`（20.6607%），collision excess=`568,264`，桶 p50/p99/max=`3/32/469`。这是 S1/S2 中间前缀，不是三层最终 SID。

### 同闭包 P5 A0 对照、产物与结论

| 指标 | P5 A0 S1 | P6 S1 | Δ | P5 A0 S2 | P6 S2 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Query cosine distortion | 0.624580 | 0.623430 | -0.001151 | 0.592466 | 0.620627 | +0.028162 |
| 加权图一致率 | 29.9614% | 81.6757% | +51.7143pp | 33.9990% | 80.2796% | +46.2806pp |
| category purity | 76.3230% | 86.8014% | +10.4784pp | 73.2765% | 79.3472% | +6.0707pp |
| POI active codes | 512 | 512 | 0 | 512 | 512 | 0 |

- S1 Query distortion 在 full 上小幅改善 0.001151（约 0.184%）；S2 仍退化 0.028162（约 4.75%）。因此预注册的 `query_distortion_improved_both_levels` 为 false，不能把 full 标记为自动通过。
- 类别 conditional entropy 在 S1 coarse token 上由 `0.803047` 降到 `0.422594`，在 S1+S2 fine path 上由 `0.656858` 降到 `0.495814`。图一致率和类别结构均显著改善，说明复合监督确实生效，但 S2 存在明确的 Query 几何代价。
- P5/P6 distinct S1+S2 prefix 为 `150,144/147,981`，P6 少 2,163；collision excess 相应增加 2,163，p99 从 31 到 32，但最大桶从 522 降到 469。该变化没有码本坍塌，但属于需与 S2 distortion 一起审核的唯一率代价。
- 正式 P6 输出位于 `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_prqk_s1_s2_hard60_topk5_v1/full_342879q_716245p/`，约 6.2 GiB；manifest SHA256=`326e00d62e132bffdcbaa05ac21a839b886fda5c9d00608c4ebfcb9486bb1624`，metrics SHA256=`cac6e2c723d6d1d26fc149f019e35410a4352894e9b3f1f574fe8034919c1739`。builder 内计算/发布 runtime=`188.02s`，外层墙钟 `4:25.77`，最大 RSS=`6,517,656 KiB`，GPU peak allocated=`22,297.38 MiB`。
- 公平对照位于同方法目录的 `evaluations/full_342879q_716245p_vs_p5_a0_v1/`；evaluation manifest SHA256=`e9cb07063b6dad09845fb5ccd22afde234bf0e977fef728b24b933c2aa93c74d`，comparison SHA256=`4b957e0e2cbf9e38158c892a83264773da9e0cb8c349906c200802ad15d99a5c`。OrangeFS 全量校验/读取使外层评估墙钟为 `1:00:09`；这是 I/O 主导，不是重新训练。
- 成功 attempt05 的源码快照 `MANIFEST.sha256`/`SNAPSHOT.sha256` 文件哈希为 `b3eb166f81d0ae03ea14530078ad4056028134adfddb7c8b69dac2a645e93870`/`e52e67b4cfe8d5b844d8bc01364200fa727bcec52d3fe464db275824829d9387`。构建、构建独立 validator、公平对照和对照独立 validator 的退出码均为 0；四份日志 SHA256 依次为 `59c0b98d1402bb20e73acd53c91bc0a2d2da39014ce7b5be405c2ede140bb217`、`8194013bd562d947b2a2211465e88e04ffe9cffee968e3abc4dc8f8a7aa62703`、`8cc02b8c03e9a8e6f9fb0ec8c70fec437de274779ab1b5f92e70d304d8ab1b89`、`0120c4133d8c1de05ae36ef358c05874ba7d2c1ee09ca22f69a0011ac7fb8053`。
- 结论：`REVIEW_REQUIRED`。full 证明 sample 中 `tau=32` 的极端稀疏效应不能直接外推到 S1，但 S2 Query distortion 仍未通过；当前严格停在 `HOLD_FOR_P6_FULL_REVIEW`。不自动调参、不启动 P7/P8，等待用户决定接受该多目标权衡，还是先做仅针对 S2 的受控诊断。

## EXP-20260910-01：P6-CAT full S2-only 受控诊断

### 目标与假设

- 目标：冻结 `EXP-20260909-05` 的正式 P6-S1 codebook、assignment 和由其确定的 projection residual，只在全部 S2 节点与完整 S2 图上重跑 S2，隔离上游 S1 residual、graph alignment、Query 质心先验 `tau=32`、Top-k5 和 category 的影响。
- 假设：S2 的 `+0.028162` Query distortion 可能同时包含 S1 residual 几何迁移和 S2 本层复合 assignment 代价，不能用同时重跑 S1/S2 的 sample 消融直接归因。

### 数据、配置与运行状态

- 输入固定为 716,245 条 active POI、342,879 个 Train-only Query、其中 317,240 个 S2 active Query 及全部 398,480 条 S2 边；不读取业务 Validation/Test，不读取 Geo。
- 正式 P6 manifest SHA256=`326e00d62e132bffdcbaa05ac21a839b886fda5c9d00608c4ebfcb9486bb1624`；同图 P5/P6 evaluation manifest SHA256=`e9cb07063b6dad09845fb5ccd22afde234bf0e977fef728b24b933c2aa93c74d`。
- 配置仍为 `configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml`，resolved signature=`2e7aa687648d3ba7e2737ade36d6d60f273c3dfe08efe20804476bb01dcc8fc0`；除受控分支声明的单一因素外，S2 训练算法和参数不变。
- 六分支为 canonical、Top-k-off、graph-off、near-zero-query-prior、graph-off×near-zero-query-prior、category-off。另以四个静态点精确分解：P5 path、P6-S1 residual 上的 P5-S2、P6-S2 content-nearest、P6-S2 coupled assignment。
- Git 工作树 dirty，保留既有用户改动，未 commit/push。正式运行使用 `outputs/run_control/p6_full_s2_attribution_v1_attempt01/source_snapshot/` 的 122 文件冻结快照；其 `MANIFEST.sha256`/`SNAPSHOT.sha256` 文件 SHA256 为 `bb4ee511e2ddaded7684555774df8aa96604930c4e09b538a1c425b40c8cb402`/`88b90a266c4c4775eb300c1ec0c0ed822184a116d9061523655598cb40469145`。
- 诊断输出固定为 `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_prqk_s1_s2_hard60_topk5_v1/diagnostics/full_342879q_716245p_s2_attribution_v1/`，role 为 `CONTROLLED_S2_FULL_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT`。

### 当前指标、结论与下一步

四点静态分解使用同一个原始 Query view，且三项效应严格加和到 P5→P6 的 S2 总变化：

| 分解点/效应 | Query cosine distortion 或 Δ |
|---|---:|
| A：P5-S1 residual + P5-S2 content-nearest | 0.592465 |
| B：P6-S1 residual + P5-S2 content-nearest | 0.569664 |
| C：P6-S1 residual + P6-S2 centroid content-nearest | 0.515784 |
| D：P6-S1 residual + P6-S2 coupled assignment | 0.620627 |
| 上游 S1 residual 迁移，B-A | -0.022802 |
| S2 Query 质心适配，C-B | -0.053880 |
| S2 coupled assignment 代价，D-C | +0.104844 |
| 总变化，D-A | +0.028162 |

- A 与正式 P5 A0 指标只差 `4.17e-7`，D 与正式 P6 指标完全一致；因此分解闭合。P5/P6 的 S1 residual cosine mean/p10/p50/p90=`0.908172/0.802063/0.934597/0.977602`，S1 Query assignment 变化率为 `64.4656%`。几何确实发生重排，但其净效果是改善 S2 content distortion，而不是造成退化。

| S2 分支 | Query distortion | 图一致率 | fine path purity | distinct S1+S2 prefix |
|---|---:|---:|---:|---:|
| canonical | 0.620627 | 80.2796% | 79.3472% | 147,981 |
| Top-k-off | 0.631647 | 84.1939% | 78.6854% | 146,187 |
| graph-off | 0.510165 | 28.4846% | 79.7176% | 147,360 |
| near-zero-query-prior | 0.618518 | 79.3691% | 79.3546% | 147,988 |
| graph-off + near-zero-query-prior | 0.505654 | 26.3153% | 79.7215% | 147,348 |
| category-off | 0.620224 | 80.4407% | 77.9155% | 148,955 |

- graph 是明确主因：canonical 相对 graph-off 的 Query distortion 增加 `0.110462`，图一致率增加 `51.7950pp`。完全关闭 graph 会把几何做得远好于 P5，但图一致率从 canonical 的 `80.2796%` 降至 `28.4846%`，还低于 P5 A0 的 `33.9990%`，因此 graph-off 只能用于归因，不能直接替换主方法。
- `tau=32` 在 full S2 上的 Query-weighted POI prior mass 只有 `4.8425%`；降至近零只改善 distortion `0.002110`，同时损失图一致率 `0.9105pp`，不再是 sample 中的主因。graph×prior 的 distortion interaction 仅 `-0.002401`。
- Top-k5 在 full S2 上改善 distortion `0.011019`、fine path purity `0.6618pp`、distinct prefix 1,794，但牺牲图一致率 `3.9143pp`；它是有益的几何/结构补偿，不应因 sample 结果而直接关闭。
- category 只增加 distortion `0.000403`，却提高 fine path purity `1.4316pp`；类别监督不是 S2 几何退化来源，当前没有证据建议移除。
- canonical 复跑与正式 P6 的 Query assignment 逐值一致，POI assignment match=`99.99972%`（仅 2 行浮点 tie 差异），Query distortion 与图一致率差均为 0。六分支均收敛、POI/Query 均使用全部 512 codes。
- 新增 3 项 full S2-only 协议测试后，完整 QG 合成回归为 `143 passed`；Ruff 与 compileall 通过。
- 输出 manifest SHA256=`7e17ce4369046d8c2c9453ced59d7b47a50afbf23bd010cb924cae83c32d55d0`，comparison SHA256=`0900f9d17569bbce8128e603bf9c582393919e6bed7b8d3daf8521bdb38dbe4d`，31 个 artifact 与 `_SUCCESS` 一致。六分支计算合计约 154 秒，builder 内总 runtime=`202.75s`、GPU peak allocated=`20,955.09 MiB`；完整运行墙钟 `8:46.03`，独立 validator 墙钟 `5:29.61`，二者的 console SHA256 为 `a161368b6139175c621b716485c5eb339930cac2cb7442912919551728c08827`/`dd962bef45386c51c8315bfb7e24d9880a09ced5949d21ef70753c6c53da9d54`。
- attempt01 的后台进程最初跨会话不可见，随后实际成功完成；launcher 的 `exit_code` 捕获与 console 中 `/usr/bin/time Exit status=0` 冲突，已用 `exit_code_resolution.txt` 显式说明。attempt02 未进入 Python；attempt03 在发现 attempt01 已发布结果后被 overwrite guard 正常拒绝。所有现场均保留，canonical P6 未覆盖。
- 结论：`REVIEW_REQUIRED`。S2 退化的根因不是 residual 几何退化、`tau=32`、Top-k5 或 category，而是当前 `graph_alignment_weight=0.10` 下的 coupled assignment 对 Query content 几何施加过强约束。正式 P6 参数和 checkpoint 保持不变，状态停在 `HOLD_FOR_P6_S2_DIAGNOSTIC_REVIEW`；下一步如需调整，应由用户确认后仅对 S2 graph weight 做 full 受控权衡试验，不自动启动 P7。

## EXP-20260911-01：P7-CAT D3 Query + Local Geo + Hard Entity S3 full

### 目标与假设

- 目标：接受 P6 S2 诊断为已知多目标权衡且不修改正式 P6，冻结其 S1/S2 endpoint，只在 S3 引入 D3 Exact Query residual、局部 Geo 和 same-fine-category 困难实体图，先完成工程 sample，再按用户授权直接构建全部 active POI。
- 假设：P7 应在 `max_iter=60` 内按冻结双停止规则收敛，POI/Query S3 均激活全部 512 个码；D1/D2、S1/S2 类别分类代价和业务 Validation/Test 不得进入 S3；相对 P5 A0 三层路径，P7 应改善 SID/bucket 结构，但完整 A0/A4 方法结论留给 P8。

### 数据版本与来源

- active catalog 固定为 `data/beijing_poi_active_order14d_history10_20260715_json/` 的 716,245 行，catalog manifest SHA256=`dc13c3f137c57ba715f129ff2ccbbd8909d910da3cebc4a9fdbce2172f4d144a`，POI ID SHA256=`8b170fe38eb86a8018f54231676c201a930a525f66243f19e91cdbbf7f2cab81`。
- P4 full/P5 full/P6 full manifest SHA256 分别为 `f6f5a6b4a6f5ef78ff8eb1007e8b566a17fe5a911aae215185b0ed156ab6329d`、`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`、`326e00d62e132bffdcbaa05ac21a839b886fda5c9d00608c4ebfcb9486bb1624`；P4 false-negative mask SHA256=`7b9e7082c5bb3daadc459a7c3fc2ac0bba5073292fbcaff0a22ba1c9f9b48004`。
- sample 为 P6 sample 闭包中的 11,459 POI 和 832 条 D3 Query；full 为全部 716,245 POI、342,879 个 P6 Query 节点中的 291,590 条 D3 Query，以及一 Query 一行的 291,590 条 S3 Exact 边。只读取 Train-derived P4 图和 D3 view，不读取业务 Validation/Test Query 或标签。
- P5 A0 S3 codebook/assignment 只作 S3 初始化；正式 P6 的 POI/Query S1/S2 assignment 和 residual 逐值冻结。Geo 从 POI `lng/lat` 与 longitude-first GID6 计算，类别只用于困难边候选筛选。

### 代码、配置与工作树

- 新增 `configs/qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml`、`src/qg_prqk/p7_{config,data,geo,prqk,cli}.py`、`scripts/build_poi_query_category_geo_prqk.py` 及 3 个 P7 测试文件；全部位于 `qg_prqk/`，运行时不导入 `src/poi_gr`。
- 配置文件 SHA256=`81473643ae44851f4ef38f8f7819555731835eb6ecbe5e4036ecc03e3473089d`，resolved signature=`ebf00f1aad5e4dcb24da6f528f4c999202dd710295403f2914eff5d4f903b795`，授权标识=`USER_CONFIRMED_20260911_P7_SAMPLE_THEN_FULL`。
- S3 权重 Query distortion/graph alignment/Geo/hard collision=`0.20/0.20/0.10/0.05`；hard alternating 最大 60 轮，Top-k=`5`、beta=`15`、5 轮，projection residual；bounded local refinement 为 32 个候选码、3 轮。
- 困难图只在固定 `(GID6,S1,S2)` parent 内构造，因此第一版是同 GID6，不向相邻 cell 扩展；要求 same fine category。BGE/name/category/address/geo 复合权重=`0.35/0.30/0.15/0.10/0.10`，阈值 `0.60`，每 POI 最多 20 个有向邻居；排除 self-loop 与 D3 false-negative protected pair，不假设不存在的 canonical duplicate mapping。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，运行时工作树 dirty，用户改动保持不动，未 commit/push。成功 full 的 130 文件冻结源码快照位于 `outputs/run_control/p7_full_attempt01/source_snapshot/`，快照清单文件 SHA256=`fe0338c9244ce3b1ed90986bf75a2212c0360764d00c48ef6069b0542ae028b3`。

### 命令与环境

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTHONPATH=qg_prqk/outputs/run_control/p7_full_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p7a \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p7_full_attempt01/source_snapshot/scripts/build_poi_query_category_geo_prqk.py \
  --config qg_prqk/configs/qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml \
  --gate full

PYTHONPATH=qg_prqk/outputs/run_control/p7_full_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p7_full_attempt01/source_snapshot/scripts/build_poi_query_category_geo_prqk.py \
  --config qg_prqk/configs/qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml \
  --gate full --validate-only
```

- 环境为 Python 3.10.20、NumPy 1.26.4、PyTorch 2.9.1+cu128、CUDA 12.8、单张 NVIDIA RTX A6000。正式构建外层墙钟 `5:09.05`、最大 RSS `8,168,752 KiB`、GPU peak allocated=`20,755.21 MiB`，退出码 0；builder 内 runtime=`89.27s`。独立 validator 墙钟 `17.67s`、最大 RSS `2,411,444 KiB`，退出码 0。

### sample 工程闭环

- attempt01 已完成 11,459 POI、832 D3 Query 的 S3 计算，但 derived Arrow 字段被推断为 nullable，与冻结 non-null schema 不一致，validator 退出 2。该失败输出完整迁移并保留为 `sample_000832q_011459p.failed_attempt01_validator_schema/`，未覆盖。
- 修正只使用显式 `P7_POI_METADATA_SCHEMA` 构建最终表，并把 `_SUCCESS` 时序收紧为全部 artifact 校验通过后再写标记；不改变数据、seed、目标、权重或 assignment。
- attempt02 构建与独立 validator 均退出 0，外层墙钟 `2:01.50`，manifest SHA256=`5a18914d4236476dc2c0f037afa2699897938416faaa099d72a873b40419ea68`；hard edge=376、hard iterations=9、POI S3 512 码全激活。sample 只用于代码/数据/GPU/产物 smoke，不参与参数选择。
- 完整 QG 合成回归 `149 passed`；schema 修正与成功标记时序修正后的 P7 定向回归 `6 passed`，Ruff、compileall、sample/full dry-run 均通过。

### full 核心结果

| 指标 | full 结果 |
|---|---:|
| D3 Query / S3 Exact 边 | 291,590 / 291,590 |
| hard entity directed edges | 746,034 |
| hard iterations / converged | 40 / 是 |
| POI / Query active S3 codes | 512 / 512 |
| `Acc(S3 | GID6,S1,S2)` 未加权 / 加权 | 82.7810% / 85.2967% |
| hard collision 未加权 / 加权 | 16.6089% / 17.5233% |
| POI / Query semantic distortion | 0.603686 / 0.673663 |
| Query initial content-nearest distortion | 0.718783 |
| Geo distortion | 0.245575 |
| POI / Query zero residual rows | 0 / 0 |

- S3 POI cluster min/p50/p90/p99/max=`303/1304/2220/3200.89/4236`，Kish ESS=`426.87`，Gini=`0.24677`；Query S3 cluster min/p50/p90/p99/max=`84/518.5/947.6/1448.26/1978`，Kish ESS=`404.22`，Gini=`0.27827`。两侧均没有 code collapse。
- 纯三层 SID distinct=`617,887/716,245`（86.2675%），collision excess=`98,358`，桶 p50/p90/p95/p99/max=`1/2/2/4/95`。P5 A0 对应为 `606,045`（84.6142%）、collision excess `110,200`、最大桶 `208`；P7 增加 11,842 个唯一 SID并降低最大桶，但两条路径的上游 S1/S2 已不同，完整方法比较仍必须在 P8 统一完成。
- `GID4/GID5/GID6 + SID` distinct ratio=`88.2620%/90.7114%/93.8637%`，collision excess=`84,073/66,529/43,951`，最大桶=`95/95/56`。商场/医院/车站类别路径 proxy 子集的困难边碰撞率为 `10.5263%/14.2012%/20.2117%`；由于当前没有 parent-child mapping，这些不能解释为真实子 POI 指标。
- 533,168 个固定 parent 中 445,306 个为 singleton；singleton Geo feature 按合同置零。该稀疏性是 active catalog + GID6 + P6 前缀的真实结构，不是缺失数据，非 singleton Geo 向量均通过标准化/L2 校验。

### 产物、启动现场与结论

- 正式输出位于 `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_geo_prqk_s3_hard60_topk5_v1/full_291590q_716245p/`，约 2.1 GiB；包含 S3 三类 codebook、POI/Query assignment、冻结 S1/S2+S3 SID、Geo/困难图、projection residual、metrics、manifest 和 `_SUCCESS`。manifest SHA256=`dd395ab0d11321106455ad05b18503af9b7d3dda7cb7fd85e531655cece2133c`。
- 构建 console SHA256=`4930f9458a8d8150ac5235d1be7c7fc41c35084588fe2d13268cbd6eb4929997`，独立 validator console SHA256=`fd78cd3f5d44c19b6cf8c9efebd0078ccfaf68fd1a88377ad02777a93e564dbc`。validator 重验全部 artifact 哈希、shape/dtype/finite、D3 mask、P6 S1/S2 逐值冻结、Geo 归一化、困难边阈值/Top-20、自环排除和 manifest/metrics 一致性。
- attempt01 的 detached 父 PID 一度不可见，但 Python 子进程持续执行并最终保存 `run.exit=0` 与完整 time 日志；attempt02 在 attempt01 已创建 canonical 目录后由 overwrite guard 于加载数据前拒绝，退出 2，未覆盖产物。两次 run-control 目录及说明全部保留。
- 结论：`COMPLETED_AND_VALIDATED`。P7 的工程、结构、收敛和产物 Gate 均通过，且相对 P5 A0 的纯 SID 唯一数与最大桶方向改善；这些结果尚不替代 P8 的统一 A0/A4 静态评测，也不证明最终生成式检索收益。状态设置为 `HOLD_FOR_P7_FULL_REVIEW`，未经用户确认不得启动 P8、Final PID、SFT、外部基线或其他下游阶段。

## EXP-20260911-02：P8-CAT A0/A4 静态评测与 A4 SID 发布候选

### 目标与假设

- 目标：在相同 active POI、相同 Train-derived Query 图与相同 D1/D2/D3 Query view 上，只比较内部初始化 A0 与完整方法 A4，联合验收码本/路径、类别结构、Query–POI 对齐、Prefix Probe、GID6 局部分离、Distinct SID、桶分布和 residual energy，并发布 A4 三层 SID 候选。
- 假设：A4 应改善类别结构和局部/全局桶结构，同时不能明显损害 Query-only 可预测性。若结构改善而 Prefix Probe 退化，必须标记 `REVIEW_REQUIRED` 并停在 `HOLD_FOR_REVIEW`，不得自动调参或进入 Final PID/SFT/外部基线。

### 数据版本与评测口径

- P2.5/P4/P5/P6/P7 manifest SHA256 分别为 `5f203b94d567433f89ff25a7755516b211544b9be2c484ac28e2243ccd7ce55f`、`f6f5a6b4a6f5ef78ff8eb1007e8b566a17fe5a911aae215185b0ed156ab6329d`、`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`、`326e00d62e132bffdcbaa05ac21a839b886fda5c9d00608c4ebfcb9486bb1624`、`dd395ab0d11321106455ad05b18503af9b7d3dda7cb7fd85e531655cece2133c`。
- full 为 716,245 POI、342,879 个参与 SID 监督的 Query（D1/D2/D3=`25,639/25,650/291,590`）、S1/S2/S3 边=`514,500/398,480/291,590`；完整 P2.5 另记录 D0=`1,029,782`。只读取冻结 QG Train-derived 产物，不读取原始业务订单或业务 Validation/Test Query/标签。
- A0 使用 P5 POI-only 三层 codebook/assignment；A4 使用 P6 S1/S2 与 P7 S3 的 POI/Query 双质心和 assignment。两个分支共用 P6 的已去 POI-fitted 公共方向并 L2 normalize 的 Query residual S0。
- Prefix Probe 明确定义为 content-only cosine nearest-code，不把训练图 assignment 当作 probe：S1 在全部 512 码中预测；S2 用正确 S1 对 Query 做 projection residual，并限制到该 S1 下已出现的 S2；S3用正确 GID6/S1/S2 residual，并限制到该 parent 下已出现的 S3。另报告使用预测语义前缀、S3 仍给定正确 GID6 的 autoregressive cumulative 指标。多合理目标按冻结 edge weight 和未加权两种方式聚合，平局取最小 token ID。

### 代码、配置、命令与环境

- 新增 `configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml`、`src/qg_prqk/p8_{config,data,evaluation,cli}.py`、`scripts/evaluate_and_publish_p8_sid.py` 和两份 P8 测试；全部位于 `qg_prqk/`，运行时不导入根 `src/poi_gr`。配置 SHA256=`11cbda45fddad82608895ffc3613eaa86d59af9ca712bd95036f0e80464f9f59`，resolved signature=`a64d95c6bee68cf97b92fd4491cf3a700d7e1f63cc31c62019917a8474be45e0`。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，运行和更新文档时工作树 dirty，用户已有改动全部保留，未 commit/push。

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=qg_prqk/outputs/run_control/p8_full_attempt02/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p8_full \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p8_full_attempt02/source_snapshot/scripts/evaluate_and_publish_p8_sid.py \
  --config qg_prqk/configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml \
  --gate full --chunk-rows 8192

PYTHONPATH=qg_prqk/outputs/run_control/p8_full_attempt02/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p8_full_attempt02/source_snapshot/scripts/evaluate_and_publish_p8_sid.py \
  --config qg_prqk/configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml \
  --gate full --validate-only
```

- 环境沿用 Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1、PyTorch 2.9.1+cu128、CUDA 12.8 与单张 NVIDIA RTX A6000。corrected full 墙钟 `2:39.22`、最大 RSS `3,496,400 KiB`、GPU 分块无 OOM、退出码 0；冻结源码独立 validator 退出码 0。

### sample 与实现修正

- corrected sample 使用 11,459 POI、1,000 Query（D1/D2/D3=`95/73/832`），构建和独立 validator 均退出 0，manifest SHA256=`71677030dea72014662aa8fd593abd58901a83b7d6db11f4246f23f577813ae4`。sample 只验证 GPU Probe、Parquet/schema、assignment 冻结、manifest 和停止边界，不用于参数判断。
- 首个 sample/full 计算本身退出 0，但 GID6+SID 展示把 packed GID6 错当作第四个 512 码层计算理论容量；该字段无效，distinct、碰撞、purity 和 probe 数值不受影响。两份结果不覆盖，完整保留为 `.failed_attempt01_gid_capacity_metric`。
- 修正后不再对 GID6+SID 声称 512 容量归一，增加 `poi_sid.bucket_id`、Prefix Probe candidate-set 难度、按 D0–D3 的 concentration/entropy、POI/Query paired-codebook cosine 和显式 `evaluation_outcome`。corrected sample 再次跑通后才运行 corrected full；未改 A0/A4 模型、assignment 或 probe 算法。

### full 静态结构与类别

| 指标 | A0 | A4 | A4-A0 |
|---|---:|---:|---:|
| Distinct SID | 606,045（84.6142%） | 617,887（86.2675%） | +11,842 |
| collision excess | 110,200 | 98,358 | -11,842 |
| SID bucket max | 208 | 95 | -113 |
| coarse purity(S1) | 76.3230% | 86.8014% | +10.4784pp |
| coarse NMI(S1) | 0.42863 | 0.52696 | +0.09833 |
| fine purity(S1+S2) | 73.2765% | 79.3472% | +6.0707pp |
| fine NMI(S1+S2) | 0.49205 | 0.51665 | +0.02460 |
| GID6+SID distinct | 659,041（92.0133%） | 672,294（93.8637%） | +13,253 |
| same-GID6 pairwise separation | 99.8895% | 99.9264% | +0.0369pp |

- A0/A4 三层 code 均 512 全激活。Kish ESS 为 A0 `444.82/431.33/427.15`、A4 `426.81/431.47/426.87`；Gini 为 A0 `0.21768/0.23667/0.24735`、A4 `0.24930/0.23750/0.24677`。A4 S1 更偏斜，但没有 code collapse；S2/S3 均衡与 A0 接近。
- A4 paired POI/Query codebook cosine mean 为 S1/S2/S3=`0.73681/0.87832/0.88426`。A0/A4 residual retained-energy mean 为 S1 `0.71574/0.71867`、S2 `0.79517/0.79384`、S3 `0.82866/0.83117`，零 residual 行均为 0。

### Query 对齐与 Prefix Probe

- A4 训练 assignment 的 S1/S2/S3 加权 token 一致率=`81.6757%/80.2796%/85.2967%`，与 P6/P7 正式指标一致，排除了 P8 Query/POI 行映射错位。D1/D2/D3 的 S1=`68.6102%/74.6895%/84.2742%`；D2/D3 的 cumulative S1+S2=`41.2623%/73.0157%`；D3 cumulative S1–S3=`65.6254%`。

| content-only Prefix Probe（weighted） | A0 | A4 | A4-A0 |
|---|---:|---:|---:|
| Query → S1 | 29.9614% | 21.0191% | -8.9423pp |
| Query + correct S1 → S2 | 39.2334% | 36.2871% | -2.9463pp |
| Query + correct GID6/S1/S2 → S3 | 90.1722% | 89.7239% | -0.4483pp |
| autoregressive cumulative S1–S3 | 12.7183% | 7.7743% | -4.9440pp |

- 分 depth 的 S1 probe：A0 D1/D2/D3=`17.7015%/24.7686%/32.2315%`，A4=`18.2125%/16.2865%/21.9760%`；A4 只在 D1 小幅提高，在 D2/D3 明显降低。teacher-forced S2 的 A0/A4 D2=`28.5435%/22.6889%`、D3=`40.5364%/37.9446%`。
- S3 teacher-forced 候选数 p50/p90/p99 两个分支均为 `1/4/19`，singleton 比例 A0/A4=`65.12%/64.66%`；因此 S3 约 90% 的准确率包含强 parent 候选约束，必须连同候选规模看。free-running S3 candidate coverage A0/A4=`25.3435%/16.8195%`，A4 的预测前缀更常在正确 GID6 中不存在。

### 产物、验收与结论

- canonical 输出为 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/p8_static_evaluation_a0_vs_a4_v1/full_342879q_716245p/`，包含 A4 六份 POI/Query codebook、POI/Query assignment、`poi_sid.parquet`、`sid_bucket_index.parquet`、两份类别分布、static/probe/residual/data-quality 指标、报告、manifest 和 `_SUCCESS`；不复制数 GB 的 residual 矩阵，而以冻结来源哈希和 residual diagnostics 引用。manifest SHA256=`c66ea6c5fac499ed7c377b9f2ae7d73c3fde333c7842f0d79e9c84b141c82ab3`。
- validator 重验 P2.5–P7 manifest 哈希、全部 P8 artifact 哈希/schema/行数、A4 POI/Query assignment 逐值一致、SID/bucket 计数守恒、源码哈希、A0/A4-only 范围和下游停止边界。成功源码快照清单 SHA256=`60f2933334c0e0be4c21408e724bffa0fe0944cd6fff4d08282b1447edb94ae1`，构建/validator console SHA256=`568dc15881e28cd102fa317ad084bb669f41f78121f74d4f3d422104c444cfc5`/`dd530e592780422ec09d6a656304c4e17aeff7076040205580a20a1cc6eb37ad`。
- 阶段收尾时完整 QG 合成回归 `154 passed`（89.632 秒）；其中新增 5 项 P8 测试覆盖 A0/A4-only、下游停止边界、候选 parent 缺失、Purity/NMI/path/GID6 局部分离等核心合同。
- detached attempt02 launcher 初始短暂不可见，但实际持续计算并保存 `run.exit=0`；随后前台重试被已存在输出的 overwrite guard 拒绝，退出 2，未覆盖 canonical 结果。所有现场保留。
- 结论：`REVIEW_REQUIRED`。A4 的类别结构、SID 唯一数、最大桶和 GID6 局部分离改善，但当前冻结的 content-only Prefix Probe 全层与 cumulative 均低于 A0。该冲突与训练 assignment 高图一致率共同表明：图/类别/Geo 目标确实重塑了 token 结构，但脱离训练图后的 Query-only nearest-code 可预测性下降。本实验完成时状态固定为 `HOLD_FOR_REVIEW`，不自动改权重、不构建 Final PID/Dedup/Tokenizer/Trie、不启动 Qwen SFT、外部基线或其他消融；该停止点随后已由用户履行。

## EXP-20260911-03：NoGID `(S1,S2)` parent 的三位 SID 受控变体

### 目标与假设

- 用户要求再构建一版不引入 GID 的 S3：冻结已有 S1/S2，只用 `(S1,S2)` 作 S3 parent，最终 SID 严格为 `[S1,S2,S3]`。
- 这是对 canonical GID-parent A4 的受控变体，不修改 P4/P5/P6 冻结产物，不进入 Final PID、Dedup、Tokenizer、Trie、SFT 或外部基线。核心假设是：去除 GID parent 后，S3 能在更广的 `(S1,S2)` 范围内重新组织语义，并可能在三位 SID 结构或 Query 可预测性上改善。

### 数据版本与代码状态

- POI 为冻结 active Beijing catalog `716,245` 行；Query 只使用 Train-only D3 Exact-Core `291,590` 行。不读业务 Validation/Test、原始订单或 D1/D2 Query。
- 冻结 P4/P5/P6 manifest SHA256 依次为 `f6f5a6b4a6f5ef78ff8eb1007e8b566a17fe5a911aae215185b0ed156ab6329d` / `ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e` / `326e00d62e132bffdcbaa05b18503af9b886fda5c9d00608c4ebfcb9486bb1624`；原 GID-parent P7 manifest SHA256=`dd395ab0d11321106455ad05b18503af9b7d3dda7cb7fd85e531655cece2133c`。
- 实验时 Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，工作树已有用户改动；本次只在 `qg_prqk/` 内新增/修改变体代码、配置、测试、文档和输出，不覆盖上游。

### 方法、配置与命令

- P7 NoGID 配置为 `configs/qg_prqk_p7_nogid_s1s2_parent_hard60_topk5_direct_full_v1.yaml`，SHA256=`8606e0a4729353e5acb70b7cee38d5c2befea2d452353019a0b93524e6700ad9`，resolved signature=`b4cc8bb52aa1b94f4e355ddad14b3eee9321a02ed1b0557bde69100038b31422`。它沿用 P6 endpoint、P5 S3 初始化、D3 Query 图、projection residual、Query/graph/Geo/hard=`0.20/0.20/0.10/0.05`、warm-up、Top-k5/β=15、`max_iter=60` 和 512 码容量；只将 parent 从 `(GID6,S1,S2)` 改为 `(S1,S2)`。
- S3 连续 Geo 特征相对 `(S1,S2)` parent 中心构造 `dx/dy/log-distance/bearing-sin/cos`，不计算 geohash/GID；困难边候选改为 `(S1,S2,fine category)`。由于原图中 same-GID 贡献是候选内常数 `0.10`，阈值从 `0.60` 等价平移为语义分数 `0.50`，其余分数权重不变。
- P8 公平比较配置为 `configs/qg_prqk_p8_nogid_s1s2_parent_comparison_full_v1.yaml`，SHA256=`552c821656f8af23988bf64573f5ee947ec9df6ac8480dd4c51d0b29ef0548a5`。A0、原 GID-parent A4 和 NoGID A4 都输出/评估三位 `[S1,S2,S3]`，S3 content-only Prefix Probe 都只使用 correct `(S1,S2)` 选候，不给原 A4 提供 correct GID。
- sample/full 构建命令均为 `scripts/build_poi_query_category_nogid_prqk.py --config <P7-NoGID-YAML> --gate sample|full`；对比命令为 `scripts/evaluate_p8_nogid_sid.py --config <P8-NoGID-YAML>`。三者都使用 `/ofs/map_search/hudan/envs/poi-gr/bin/python`、`PYTHONPATH=qg_prqk/src`。

### 运行与核心结果

- sample smoke：`11,459 POI + 832 D3 Query`，1,290 条困难边，9 轮收敛，512 码全激活，S3 加权一致率=`92.3239%`，distinct SID=`11,358/11,459`。首次计算因 validator 误将文件名 `nogid` 中的字符串判为 GID 产物而退出 1；失败目录已保留，修复为只禁止真实 `gid6`/`geohash` 后重跑成功。sample 仅作代码/数据流 smoke，不作方法结论。
- P7 NoGID full：147,981 个 `(S1,S2)` parent，p50/p90/p99/max parent size=`3/11/32/469`；2,952,789 条有向困难边。43 轮在冻结停止条件下收敛，POI/Query 均 512 码全激活。S3 加权训练一致率=`84.8905%`，加权困难边碰撞率=`11.1675%`，POI/Query 语义 distortion=`0.609373/0.674562`，POI/Query residual retained energy=`0.834686/0.841935`。
- P7 NoGID 三位 SID：distinct=`613,198/716,245`（85.6129%），collision excess=`103,047`，bucket p50/p90/p95/p99/max=`1/2/2/4/159`。相对 A0 增加 7,153 个 distinct，最大桶 `208→159`；但相对原 GID-parent A4 减少 4,689 个 distinct，最大桶 `95→159`。

| 同一 `(S1,S2)` S3 候选口径（weighted） | A0 POI-only | 原 GID-parent A4 | NoGID A4 |
|---|---:|---:|---:|
| Query → S1 | 29.9614% | 21.0191% | 21.0191% |
| Query + correct S1 → S2 | 39.2334% | 36.2871% | 36.2871% |
| Query + correct S1/S2 → S3 | 67.7142% | 66.2934% | 65.8531% |
| autoregressive cumulative S1–S3 | 9.7802% | 5.8439% | 5.7867% |

- `(S1,S2)` parent 使 S3 teacher-forced 候选数 p50/p90/p99 约为 `7/22/62–63`，singleton 比例约 `7.6%`；这比原 P8 提供 correct GID 时的 `1/4/19` 明显更难，因此原 P8 的 `89.7239%` 与本表 `66.2934%` 不可直接比较。本表的原 A4/NoGID A4 是公平的同口径对照。
- 在该公平口径下，NoGID 相对原 A4 的 S3 下降 `0.4403pp`，cumulative 下降 `0.0572pp`；S1/S2 完全相同，符合只重训 S3 的合同。因此本次假设未得到支持。

### 产物、验收与结论

- P7 NoGID full 输出位于 `outputs/qg_prqk_512x3_v2_1_cat_active/poi_query_category_geo_nogid_prqk_s3_s1s2_parent_hard60_topk5_v1/full_291590q_716245p/`，manifest SHA256=`a6ce4ba8c20f79073e5a41fa2e516b26b2bdeb7c2d7c56e260f4a94f46bf1d6c`；墙钟 `6:11.87`，最大 RSS `8,424,268 KiB`，构建/冻结源码 validator 均退出 0。
- P8 NoGID 公平对比位于 `outputs/qg_prqk_512x3_v2_1_cat_active/p8_nogid_s1s2_parent_comparison_v1/full_342879q_716245p/`，manifest SHA256=`b22ab413558b1e147542135fd554fdf0b7a71082bdd1842fc787aa2b16ed4ed8`；墙钟 `3:33.44`，最大 RSS `2,606,984 KiB`，构建/冻结源码 validator 均退出 0。P7/P8 成功源码快照清单 SHA256 分别为 `c87d0079c58e7b2cf023679fdaa19730aafb4e9eea82167520387dc469e82443` / `8c9f97b30f99202a04d12a2cf7b1c98b79956f2be0088f98b1c7a5b873f4d29c`。
- validator 重验冻结输入/源码/输出哈希、行序、S1/S2 逐值冻结、SID 三列 schema、parent/困难边/Geo 合同、不存在 GID6/geohash 产物、bucket 计数守恒和下游停止边界。
- 阶段收尾时完整 QG 合成回归 `162 passed`（97.438 秒）；全目录 compileall、Ruff 和 `git diff --check` 均通过。NoGID 新增 7 项定向测试，另扩展 1 项原 P8 parent-key 合同，覆盖无 GID 配置、三位 SID、相对 Geo、`(S1,S2)` parent/困难图和公平 probe。
- 结论：`REVIEW_REQUIRED`。NoGID 版已完整实现并验证，且好于 A0 的纯结构基点；但它同时在 distinct SID、最大桶、S3 Prefix Probe 和 cumulative Prefix Probe 上略差于原 GID-parent A4，因此不建议自动替换 canonical A4。本实验完成时状态为 `HOLD_FOR_NOGID_S3_REVIEW`，等待用户确认保留哪个分支且不启动任何下游；该停止点随后已由用户履行，并决定两个分支均进入 SFT 数据准备。

### SID 三方并排可视化补充分析

- 目标：在不重跑聚类、不改变 SID 的前提下，对 `A0_POI_ONLY`、`A4_GID_PARENT` 和 `A4_NOGID_S1S2_PARENT` 做同一套全量可视化，回答码本是否坍塌、逐层 residual 与质心是否形成可见簇，以及共享前缀是否对应更高的原始 BGE/类别相似度。设计参考 [GenRet](https://irlab.science.uva.nl/wp-content/papercite-data/pdf/sun-2023-learning.pdf) 的 token 频率与 embedding/codebook t-SNE、[Better Generalization with Semantic IDs](https://arxiv.org/html/2306.08121v2) 的共享前缀层次分析，以及 [LAMIA](https://arxiv.org/html/2409.07276v3) 对 RQ-VAE 各层 token/残差语义的 t-SNE 检查；图片只作可解释性诊断，不作为训练或选模指标。
- 数据与合同：读取 716,245 条 active POI 的原始 BGE、三套冻结三位 SID/512×3 POI 码本，以及每一层实际 assignment 对应的 residual 输入（S1=`residual_s0`，S2=`residual_after_s1`，S3=`residual_after_s2`）。三方来源 manifest 均冻结在配置中；不读原始订单、业务 Validation/Test 或 Query embedding，不改写 SID，不启动下游。
- 代码与配置：新增 `configs/qg_prqk_sid_visualization_a0_a4_nogid_full_v1.yaml`、`src/qg_prqk/sid_visualization{_config,_cli}.py`、`scripts/visualize_sid.py` 和 `tests/test_sid_visualization.py`。配置 SHA256=`9a34b1854c250974d698626fc26df6dbedd75772682e26e33b2b5478dde9e7b0`，resolved signature=`86a9567db86e521104998aba15035f4b43515131ab366c838b8c33254dba8854`；Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户工作树保持不动，未 commit/push。
- 抽样与降维：seed=42；每种方法在 exact shared-prefix length `0/1/2/3` 上各抽 20,000 对 POI，pair 在满足该 exact-prefix 条件的全体 POI pair 中等概率抽取。语义相似度固定用原始 active-POI BGE cosine；coarse category 取可读 `category` 的冒号前路径，full 审计确认 19 个路径与 19 个 `category_code[0:2]` 恰好一一对应、没有一对多或多对一，因此逐行等价于 canonical coarse binding；fine category 使用冻结 `fine_category_index`。每层每种方法取占用最高的 5 个码、每码 100 条 POI；把三种方法的样本和码本放入该层共同的 PCA-50+t-SNE 坐标系，三列共享坐标范围。PCA 保留方差 S1/S2/S3=`66.2215%/50.0115%/46.2183%`。

| 方法 | S1 active / ESS / Gini | S2 active / ESS / Gini | S3 active / ESS / Gini |
|---|---:|---:|---:|
| A0 POI-only | 512 / 444.82 / 0.2177 | 512 / 431.33 / 0.2367 | 512 / 427.15 / 0.2474 |
| 原 GID-parent A4 | 512 / 426.81 / 0.2493 | 512 / 431.47 / 0.2375 | 512 / 426.87 / 0.2468 |
| NoGID A4 | 512 / 426.81 / 0.2493 | 512 / 431.47 / 0.2375 | 512 / 407.12 / 0.2680 |

| 方法 | exact-prefix 0/1/2/3 的 BGE cosine | exact-prefix 1 coarse/fine | exact-prefix 2 coarse/fine | exact-prefix 3 coarse/fine |
|---|---:|---:|---:|---:|
| A0 POI-only | 0.5649 / 0.6878 / 0.7857 / 0.8974 | 69.19% / 52.46% | 77.52% / 63.01% | 89.59% / 78.31% |
| 原 GID-parent A4 | 0.5653 / 0.6865 / 0.7850 / 0.8911 | 84.81% / 63.23% | 89.24% / 72.84% | 94.11% / 81.12% |
| NoGID A4 | 0.5661 / 0.6861 / 0.7852 / 0.8962 | 85.46% / 63.83% | 89.80% / 73.02% | 93.77% / 81.12% |

- 结果解释：三者每层均使用全部 512 码，未见 code collapse。三套 SID 的 exact-prefix 长度增加时，原始 BGE cosine 都严格单调上升，证明前缀层次确实承载由粗到细的内容相似性；但三方法间 cosine 差异很小。A4 在共享 S1/S1S2/S1S2S3 时的类别一致率显著高于 A0，说明 Query 图与类别软锚定主要改变的是类别组织，而非简单拉高通用 BGE cosine。
- NoGID 诊断：NoGID 与原 A4 的 S1/S2 完全冻结相同；差异只出现在 S3。NoGID S3 ESS 从 `426.87` 降为 `407.12`、Gini 从 `0.2468` 升为 `0.2680`、最大码占用从 `4,236` 升为 `5,468`，完整 SID distinct/max bucket 为 `613,198/159`，仍弱于原 A4 的 `617,887/95`。这与 P8 公平 Prefix Probe 的结论一致，不支持用 NoGID 自动替换 canonical A4。
- 运行与失败记录：首次未发布尝试在默认 OpenBLAS/OpenMP `128/64/128` 线程下发生严重线程过度竞争，S1 投影约 16 分钟后人工中止；该尝试没有正式输出，空 staging 已删除。随后只把 CPU 线程数冻结为 8，不减少 3,020/3,019/3,028 个联合点、不减少 1000 轮 t-SNE；正式构建墙钟 `160.46s`，退出 0，独立 `--validate-only` 退出 0。
- 产物：`outputs/figures/qg_prqk_sid_a0_a4_nogid_full_v1/` 保存 `sid_code_distribution.png`、`sid_residual_semantic_map.png`、`sid_prefix_semantics.png`、`metrics.json`、`semantic_examples.json`、`sampling.npz`、manifest 和 `_SUCCESS`。manifest SHA256=`3057f95601eff3441df5b5b2a67051a5babda489e8c8a7a9507e252f69e5b2f3`；三张图片 SHA256 依次为 `4fe66446f2af5b0ddc34ca5c3f36e4fc4eb18eef18b57b4d9122b71589edbdb4` / `f5cc8859582aac3e0b5768446632debafe6ef2141bf9ec782a100965b057f371` / `7ee3477d07a6cd1479e35690c07f76b5ff92c64d15750a65e16f9631c629ddd5`。新增 6 项定向合成测试覆盖 occupancy、转移守恒及 0–3 级 exact-prefix 确定性抽样；完整 QG 回归 `168 passed`（86.609 秒），compileall、Ruff、目录 manifest 与 `git diff --check` 均通过。
- 结论与停止点：该补充分析增强了三方结果的可解释性，但没有产生新的 checkpoint、SID 或方法选择证据；状态继续保持 `HOLD_FOR_NOGID_S3_REVIEW`，不推进 Final PID/Dedup/Tokenizer/Trie、SFT、外部基线或其他消融。

### 2026-09-14：GenPOI 风格同五类与具体前缀补充可视化（已完成）

本项延续上述三方 SID 的只读解释性分析，不重跑聚类、不创建新方法实验、不改变当前 SFT/Test 交接阶段。参考 [GenPOI（Chen 等，2026）图 4/5](https://arxiv.org/html/2605.03397v1)：图 4 用相同五类颜色观察向量分布，图 5 展示不同长度 SID 前缀中的类别/区域组成。原文未公开五类的具体名称与完整抽样、降维参数；以下是用户确认的本地适配，不声称逐项复现原文 GeoPE 前后 embedding。

- 目标与假设：控制样本与投影后，检查 A4 相对 A0 改变了多少类别组织，区分共同 S1/S2 与两种 S3 的作用；图只解释已冻结 SID，不充当检索提升或模型选择依据。
- 输入与边界：三版均为原有 716,245 行 active POI、512×3 POI 码本及三位 SID。来源通过 SHA256 引用旧可视化配置，再逐个核对 13 个实际读取的 SID/码本/metadata 产物哈希、行序、POI 唯一性及两种 A4 的 S1/S2 相同；不读原始 BGE/residual 大矩阵、订单、Query embedding 或业务 Validation/Test，不更新 SID/模型。
- 固定五类：按全库粗类别数量选 `房产小区 190,347`、`室内及附属设施 89,873`、`美食 66,495`、`购物 58,085`、`生活服务 52,867`。`seed=42` 各无放回抽 200 个，共 1,000 个 POI；三版的 POI ID、行号、类别颜色完全相同。粗类别显示名与 `category_code[:2]` 的一一对应关系已检查。
- t-SNE：对每条 SID 选择对应三层 POI 码向量，每层 L2 归一化后按 `1/sqrt(3)` 等权串联为 3,072 维表示；不把整数 code 当欧氏坐标，不使用类别/地理/Query 码向量作为投影输入。三版合计 3,000 点联合 randomized PCA50，再统一 t-SNE（seed=42、perplexity=30、max_iter=1000、init=pca、learning_rate=auto、Euclidean）；三面板共享坐标范围。PCA 保留方差 `45.0955%`，t-SNE KL=`0.9564364`，因此二维图只是受损压缩后的定性视图。
- 具体前缀：每类固定抽样的第一个 POI 为锚点，不根据纯度或图片效果选样。在每种方法中沿同一锚点展示 `[S1,*,*] → [S1,S2,*] → [S1,S2,S3]`，共 45 个桶；每条统计全库中匹配该前缀的全部 POI，不限定五类，也不添加 GID 条件。第一层显示粗类别，后两层显示完整细类别的叶名；区域用冻结经纬度计算标准 Geohash5 网格，明确不是行政区，亦未把 GID 加入 NoGID 方法。
- 图中类别/区域分别用蓝色/橙色，深浅表示各桶占比第 1/2/3 名，灰色为其余；这是排名颜色，不是跨桶固定类别颜色，区别于 t-SNE 的固定类别颜色。条内百分比四舍五入到整数，小片段省略文本，完整精确计数及细类别完整路径保存在 `metrics.json`。每个桶保留 POI 数，单例显式标记。

以下纯度是**全库按 POI 数加权的桶内最大类别/网格占比**，不是 5 个示例的平均，也不同于旧图“恰好共享指定长度前缀的随机 pair 一致率”。非单例指标重新以桶大小大于 1 的 POI 为分母。

| 指标 | A0 POI-only | A4 GID-parent | A4 NoGID |
|---|---:|---:|---:|
| 同五类样本高维 cosine silhouette | 0.051067 | 0.061592 | 0.061600 |
| S1 粗类别纯度 | 76.3230% | 86.8014% | 86.8014% |
| S1/S2 细类别纯度 | 73.2765% | 79.3472% | 79.3472% |
| S1/S2 细类别纯度，剔除单例 | 71.4871% | 77.9793% | 77.9793% |
| S1/S2/S3 细类别纯度 | 96.2665% | 97.1905% | 96.9938% |
| S1/S2/S3 细类别纯度，剔除单例 | 84.9097% | 87.4675% | 87.1173% |
| S1/S2/S3 Geohash5 纯度，剔除单例 | 81.7314% | 79.1900% | 82.6869% |
| 完整 SID 单例 POI 占比 | 75.2590% | 77.5821% | 76.6645% |

- 读图结论：A4 保留了 A0 的整体类别布局，类别边界有所集中，但不是五个完全分离的簇。S1 粗类别纯度提高约 10.48 个百分点、S1/S2 细类别提高约 6.07 个百分点；剔除单例后提升仍存在。两种 A4 的 S1/S2 完全相同，t-SNE 也高度相近，因此不能从这组图宣称某种 S3 带来大幅全局语义改善。
- S3 边界：15 个完整 SID 示例中 13 个是单例，其余为 2/3 个 POI；不能把图上的 100% 解释为“语义学习完美”。全库也有约 75%–78% POI 是完整 SID 单例，所以补充非单例指标。NoGID 在纯三位 SID 桶的 Geohash5 纯度更高，但 GID-parent 的外部 GID 没有参与本图分组，不能据此判定 NoGID 的完整 Final ID 地理组织或下游检索更好。类别本身参与过 A4 监督，聚集现象不是独立泛化证据。
- 实现与复现：`src/qg_prqk/sid/category_region_visualization.py` 负责合同、抽样、特征、投影、精确桶统计及哈希；`category_region_plots.py` 只负责绘图；`commands/visualize_sid_categories.py` 接入已有 CLI，不修改旧版绘图实现。配置为 `configs/qg_prqk_sid_category_region_v2.yaml`，命令为 `python qg_prqk/scripts/qg_prqk.py visualize-sid-categories --config qg_prqk/configs/qg_prqk_sid_category_region_v2.yaml`；完整环境及只读复核命令见 `../../README.md`。
- 环境与结果：使用 `/ofs/map_search/hudan/envs/poi-gr/bin/python`、scikit-learn 1.7.2、CPU 8 线程、QG 内短 TMPDIR 与字体/cache；实际构建 `150.88s`，`build.exit=0`，新图独立 `--validate-only` 退出 0。8 项新增合成测试与旧图/CLI/代码隔离回归合计 20 项通过，Ruff、compileall 通过。Git HEAD 仍为 `54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户工作树保留，未 commit/push。
- 产物与来源：`outputs/figures/qg_prqk_sid_category_region_v2/` 保存 `sid_category_tsne.{png,pdf}`、`sid_prefix_category_region.{png,pdf}`、`sample_and_coordinates.parquet`、`metrics.json`、`manifest.json` 与 `_SUCCESS`。manifest SHA256=`0b8b80254c98cd4240e5a935e392d59eefce080eac4f1ca052f0d89b126bf5a7`；保存 31 项配置/上游 manifest/实际数据/字体/运行代码来源哈希。字体由仓库既有 `outputs/ppt_0907/assets/NotoSansCJKsc-Regular.otf` 复制到 `qg_prqk/outputs/inputs/fonts/`，运行时全部代码和资产只从 QG 目录消费。日志、PID、退出码在 `outputs/run_control/sid_category_region_v2/`。
- 独立核对：1,000 个抽样行/POI ID 唯一、三版 SID 与冻结行逐项相符；45 个前缀桶均重新核对匹配行数，所有类别/区域计数及 Top-3 加其余占比守恒，坐标全部有限。metadata 的完整细类别名/完整 `category_code` 为 `397/397` 一一对应。旧图的 manifest、全部输出及冻结来源哈希仍一致；但旧 `visualize-sid --validate-only` 对“当前源码必须等于历史运行源码”的检查因之前目录/CLI 重构而失败。本次未修改这四个旧运行文件，以 `require_current_code=False` 单独核验旧产物/来源通过，未重写历史源码指纹；该历史严格核验限制与新图（当前代码核验通过）区分记录。
- 下一步：用户查看 PDF 与精确桶计数后再决定是否增补其他切面；本项不据图调参、不启动新训练或业务评测，保留当前 `HOLD_FOR_QG_PRQK_FULL_TEST_LAUNCH_REVIEW`。

#### 碰撞案例补充：末层全部为非单例（2026-09-14）

- 用户指出前版 15 个末层示例中 13 个是单例，无法据此观察区域结构。本次只增加碰撞条件视图，旧图和 t-SNE 不覆盖、不重算；不改变 SID、训练或评测阶段。
- 选例合同：仍使用相同五类与三版冻结 716,245 行 SID，先要求锚点在**三版**的完整三位 SID 桶都含 3–20 个不同 POI，再以 seed=42 在每类合格 POI 中均匀抽取一个。不根据类别纯度、区域分散程度或图形效果选例；共同合格 POI 共 36,410 个，五类各为 17,714/6,497/1,265/1,110/586 个。上限仅为便于完整展示成员，故此图不代表全库或全部碰撞桶的分布。
- 数据流与代码：新 `sid/collision_visualization.py` 计算完整 SID 桶大小与三版候选交集，复用 `category_region_visualization.py` 的输入核验、精确前缀统计和 `category_region_plots.py` 的条形图函数；沿同五个锚点展示 45 个前缀桶，并对末层 15 个桶保存全部 54 条成员记录（跨方法可重复）。明细保留 POI ID、名称、完整类别、Geohash5 和经纬度，不按五类/GID 过滤桶内成员。

| 锚点类别 | A0：POI 数 / 网格数 | A4 GID-parent：POI 数 / 网格数 | A4 NoGID：POI 数 / 网格数 |
|---|---:|---:|---:|
| 房产小区 | 3 / 2 | 3 / 2 | 3 / 3 |
| 室内及附属设施 | 5 / 2 | 4 / 2 | 4 / 2 |
| 美食 | 3 / 2 | 3 / 2 | 3 / 2 |
| 购物 | 3 / 3 | 6 / 5 | 5 / 5 |
| 生活服务 | 3 / 1 | 3 / 1 | 3 / 1 |

- 实际案例：NoGID `[508,425,323]` 由“至善家园6号楼-4单元”“杨庄北街6号院1号楼-4单元”“梨园东里北区6号楼-4单元”组成，类别均为楼栋号，但分别位于 `wx4gp/wx4fy/wx4fz`。购物 `[367,141,302]` 在三版均为“自行车专卖”，A4 GID-parent 包含 6 家爱玛门店、跨 5 个网格，说明纯三位 SID 可能聚合语义相近的异地店；不能把类别集中解读为地域相同。
- 保留反例：生活服务锚点的完整 SID `[316,189,432]` 在三版均含 2 个台球馆、1 个美容美发 POI，虽然都在 `wx4ff`，也不是类别纯桶。美食案例均含 2 个火锅、1 个自助餐 POI。图中的同网格不代表同坐标或同一地点；完整原始类别路径和坐标见明细。GID-parent 的外部 GID 未参与本图分组，因此不得把上述跨格案例直接当作其完整 Final ID 的地理碰撞。
- 复现：`python qg_prqk/scripts/visualize_sid_collisions.py`，默认复用 V2 来源配置、seed=42、min-size=3、max-size=20；`--validate-only` 复核已有输出，`--help` 说明参数。单独的薄入口避免修改 V2 manifest 已绑定的 CLI 源码；不新增一套命令调度器。全部实现、字体、缓存和产物仍在 QG 内，环境为 `poi-gr`、CPU、8 线程及 `outputs/tmp/sv2`，没有 GPU/训练/业务 Validation/Test 读取。
- 已完成：实际构建 `43.61s`，`build.exit=0`；输出为 `outputs/figures/qg_prqk_sid_collision_examples_v1/` 下的 `sid_collision_prefix_category_region.{png,pdf}`、`collision_members.html`、`collision_examples.json`、manifest 和 `_SUCCESS`。manifest SHA256=`378c905f358e1bb89b7cfb14e8e93eaeee9ea99a8eae97d5baf307047cc48e97`；日志/PID/退出码在 `outputs/run_control/sid_collision_examples_v1/`。8 项新增合成测试加既有类别可视化/代码隔离/CLI 回归共 22 项通过，Ruff、compileall 通过。Git HEAD 仍为 `54802e6674e283722ecee00fb862530df30cef9a`，dirty 工作树保留，未 commit/push。
- 碰撞版与原 V2 均已用各自 `--validate-only` 独立核验成功，原 V2 manifest SHA256 仍为 `0b8b80254c98cd4240e5a935e392d59eefce080eac4f1ca052f0d89b126bf5a7`，当前源码和产物都未变。
- 结论与下一步：用真实碰撞案例替代单例来观察末层结构，确认同 SID 既可能跨区域，也可能跨类别；这仅是条件案例分析，不据 5 个锚点给三种方法作整体排序。不启动新实验，继续等待用户查看图和成员明细。

#### A0→A4 前缀与成员迁移补充（2026-09-14）

用户认为共同碰撞案例中 A0/A4 过于相似，确认增加“实际改变了什么”的只读视图。本项沿用原 716,245 行 active POI 和三版冻结 SID，不重训、不改参数、不重算 t-SNE、不读订单/Query embedding/业务 Validation/Test，也不追加 GID/dedup。旧两版图原样保留，SFT/Test 主状态不变。

- 口径：分别统计每个 POI 的数值前缀是否改变，以及前后同桶的其他 POI 集合是否改变。通过前后桶的联合分组精确计算交集，不把码编号距离当语义距离；码号相同也可能换成员，整体重编号也可能完全不换成员。成员变化仅指至少一名同伴改变；平均同伴 Jaccard 的双方空集按 1 计，因此末层值受大量双侧单例影响，不能作为语义指标。
- 类别/区域变化：对每个 POI，比较 `(同锚点标签的桶内 POI 数−1)/(桶大小−1)`，即剔除自身后的同类别/同网格同伴比例。S1 看粗类别，S1/S2 与三位 SID 看完整细类别；区域是 Geohash5。前后均非单例才能比较，提高/持平/降低用整数交叉乘法精确判断；任一侧单例单列“不可比”，不视作 100%。该指标不是上文的“桶内最大标签占比”，不能混用数值。
- 选例：全库统计不做筛选；案例则要求前缀与成员集合均改变、前后均非单例，S2/S3 的前后桶各至多 20 个 POI（S1 不限）。在类别比例提高/持平/降低的候选池内分别均匀抽一个，固定 seed=42，每层使用 seed+depth；不取最大增益，也不按区域筛选。不局限旧五类。两版各 9 个案例，共 18 个展示项；前两层相同，末层候选池不同，不能把两张图的末层当成共同锚点配对实验。

| 对比与深度 | 前缀码号改变 POI | 同伴集合改变 POI | 码号未变但同伴改变 |
|---|---:|---:|---:|
| 两种 A4，S1 | 145,682（20.3397%） | 716,245（100%） | 570,563 |
| 两种 A4，S1/S2 | 201,516（28.1351%） | 611,346（85.3543%） | 412,771 |
| GID-parent，S1/S2/S3 | 300,605（41.9696%） | 172,017（24.0165%） | 70,231 |
| NoGID，S1/S2/S3 | 355,420（49.6227%） | 180,950（25.2637%） | 68,318 |

前两层同伴平均 Jaccard 为 `0.580454 / 0.504420`。末层分别有 `198,819 / 242,788` 个 POI 换码但同伴集合不变，说明不能将约 42%/50% 的换码比例直接当作语义重组幅度。

以下均在**同一对比内前后都非单例的同一批 POI**上取平均；两个末层对比的可比群体不同，不能直接横比其绝对值。

| 对比与深度 | 可比 POI 数 | 同类同伴比例 A0→A4 | 同网格同伴比例 A0→A4 |
|---|---:|---:|---:|
| 两种 A4，S1 | 716,245 | 66.2648%→80.9657% | 7.6185%→7.0071% |
| 两种 A4，S1/S2 | 652,311 | 57.0241%→65.6795% | 31.6931%→31.1687% |
| GID-parent，三位 SID | 115,619 | 75.9925%→77.9832% | 66.6988%→65.5112% |
| NoGID，三位 SID | 114,429 | 76.1558%→77.9507% | 71.6473%→72.5798% |

- S1 类别同伴比例提高/持平/降低的 POI 为 `646,173 / 349 / 69,723`，占全库约 `90.22% / 0.05% / 9.73%`；不能解释为 90% 的 Query 检索提升。S1/S2 相应为 `276,296 / 274,121 / 101,894`，另 `63,934` 个因单例不可比。
- 完整 SID 碰撞迁移（按 POI，不是桶）：GID-parent 的“旧碰撞→单例 / 旧单例→碰撞”为 `61,587 / 44,948`，碰撞 POI 净减少 `16,639`；NoGID 为 `62,777 / 52,710`，净减少 `10,067`。原碰撞 POI 共 `177,206`，两版最终为 `160,567 / 167,139`。这是碰撞结构，不是去重后的最终 ID 唯一性或检索指标。
- 实际案例：S1/S2 的“蓝精灵财务顾问(北京)有限公司”由 `[440,291]→[247,291]`，旧桶与新桶均 4 个 POI，仅保留锚点，其余 3 个全部更换，同细类别比例 `0%→100%`。反例“New Power新势力运动中心(知春路店)”由 5 人桶扩大至 8 人桶，原 5 个全部保留、新增 3 个，同细类别比例 `100%→85.7143%`。NoGID 末层“天通北苑1区22号楼-1单元”前后均为 2 人桶，类别比例同为 100%，但另一个成员被替换，同网格比例 `0%→100%`：类别图相似不意味着地域成员不变。
- 标签边界：“石化学院丽园小区-18号楼”的冻结粗类别实际为“教育学校”；其同类比例前后均为 0，不能只看名称便改成房产类别或把“持平”解读为良好。当前分析完全使用存量标签、不自行纠正，类别指标仍受标签质量影响。“蓝精灵财务顾问”移出的实际成员为蓝鲸控股集团、蓝驰创投、蓝色天际投资；移入的是北京市蓝鹏(西城区)律师事务所、北京蓝鹏律师事务所、北京市蓝石律师事务所。财务顾问与律师都标作“生活服务:事务所”，因此标签比例达到 100%，但业务语义未必比原金融/投资邻域更合适；此例只算类别指标提高，不可声称 Query 检索改善，提醒后续关注标签粒度和监督权衡。
- 数据流与源码：`scripts/analyze_sid_migration.py` 薄入口 → `sid/migration_analysis.py` 加载冻结合同、精确分组/选例/发布 → `sid/migration_plots.py` 绘图及 HTML。复用 QG 内既有来源核验、metadata、网格和保存函数；不改原 V2/碰撞版所绑定的 CLI 或运行源码。全部代码与输出仍在 QG 下。
- 复现：`python qg_prqk/scripts/analyze_sid_migration.py`；用 `--validate-only` 只读复核，用 `--output-dir qg_prqk/outputs/figures/<新目录>` 另存，已有目录拒绝覆盖。环境同上为 `poi-gr`、CPU 8 线程、QG 内 `outputs/tmp/sv2` 与字体缓存；完整环境命令见 `../../README.md`。Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，dirty 用户工作树保留，未 commit/push。
- 产物：`outputs/figures/qg_prqk_sid_migration_v1/`，包含 `sid_migration_overview.{png,pdf}`、`sid_migration_gid_examples.{png,pdf}`、`sid_migration_nogid_examples.{png,pdf}`、`metrics.json`、`examples.json`、`migration_examples.html`、`memberships.npz`、manifest 与 `_SUCCESS`。JSON/HTML 每个保留/移出/移入组预览至多 5 个，NPZ 保存全部成员行号与 POI ID，统计未截断。manifest SHA256=`40eee80dd82b4bb40bbfe45a3125b04d9b77486abef0757fc5285cd54f76b240`，绑定 34 项来源、10 个结果文件；日志/PID/退出码在 `outputs/run_control/sid_migration_v1/`。实际构建 `119.59s`，退出码 0。
- 核验：新增 10 项迁移测试与已有类别图/碰撞图/CLI/代码隔离共 `32` 项通过（`8.870s`），Ruff、compileall、`--help`、工作树 diff/空白检查通过。独立从冻结 SID 重新枚举 18 个案例的前后桶，逐个核对全部保留/移出/移入行号、POI ID、预览前缀与计数守恒；两种 A4 前两层统计完全相同。新迁移版、已有 V2 和碰撞版各自 `--validate-only` 均返回 `validated`，旧输出与运行源码哈希不变；三个新 PNG 已人工读图检查。
- 结论与下一步：A4 主要重组前两层的类别邻域；末层约四分之一 POI 的同伴发生变化，既拆开旧碰撞也引入新碰撞，保留了较多 A0 结构。类内组织改善不是 Query 监督的独立因果证据，因为类别也参与过监督；地理不全面改善，且本图未使用 GID-parent 的外部 GID。保留降低案例与既有 content-only Prefix Probe 下降的诊断，不据图宣称端到端优于 A0；后续须经用户确认做匹配协议的 A0 SFT/检索对比，本次不启动。

## EXP-20260911-04：A4 GID-parent / NoGID 双 Final ID 与全量 SFT Messages

### 目标与假设

- 用户在审核 SID 后确认进入 SFT，并要求同时保留两个训练分支：A4 GID-parent 使用 `G1..G6,S1,S2,S3,[D]`，第一种 NoGID 严格使用 `S1,S2,S3,[D]`，不在 NoGID history/target identifier 中引入 GID。
- 假设：两个冻结 SID 都能在全部 716,245 active POI 上通过“完整 Base ID 冲突桶内按 `poi_id` 字典序追加末位 D”得到全局唯一 Final ID；同一份 history10 语料可以只替换 history/target identifier，保持 Query、请求 GID、用户、时间切分和样本顺序逐行对齐。
- 本实验只构造 Final ID、全量 Messages 与后续词表/Cache/训练入口，不运行扩词表、Tokenized Cache、Qwen SFT、Trie 评测或外部基线。

### 数据版本与代码状态

- active POI=`716,245`。GID-parent SID 来自 `p8_static_evaluation_a0_vs_a4_v1/full_342879q_716245p/poi_sid.parquet`，其来源 manifest SHA256=`c66ea6c5fac499ed7c377b9f2ae7d73c3fde333c7842f0d79e9c84b141c82ab3`；NoGID SID 来自 `p8_nogid_s1s2_parent_comparison_v1/full_342879q_716245p/poi_sid.parquet`，来源 manifest SHA256=`b22ab413558b1e147542135fd554fdf0b7a71082bdd1842fc787aa2b16ed4ed8`。
- GID6 使用冻结的 `poi_gid6.npy`，与 active POI 同行同序。SFT 变换源为 `data/sft/tiger_active716k_bge_m3_512x3_history10_query_gid_v1/`，manifest SHA256=`80d39eb00269baffe0b5c72aa9ecbcaf6d832b3f52e18329390adceabaf088a1`；Train/Valid/Test=`7,586,410/597,421/606,682`，合计 8,790,513 条、43,208,167 个 history occurrence。
- 源 TIGER mapping 仅用于把历史/目标的旧唯一 ID 反查到 `poi_id`，mapping SHA256=`33d4e820f0d818d5a1d2d3b0290ef30f9a3254999b651e1f8f6293b48819e5b8`；旧 TIGER SID/C 不作为 QG 标签。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，工作树已有用户改动并保持不动；本次源码、配置、测试、文档和输出均位于 `qg_prqk/`。从根目录复用的 Dedup、Messages 改写、扩词表、Token 预检和训练编排已经复制并登记到 `configs/code_provenance.yaml`，QG 运行时不导入 `poi_gr` 或调用仓库其他项目脚本。

### 方法、配置与命令

- Final ID 由 `src/qg_prqk/final_identifier.py` 和 `scripts/build_final_identifiers.py` 构造；先冻结完整 Base ID，再只给碰撞桶成员分配 `D0..`，单例存储 `-1` 且序列中省略 D。mapping 保持 `poi_row_index/poi_id` 同行同序，输出 NPY、Parquet、metrics、manifest 和 `_SUCCESS`，存在目录拒绝覆盖。
- Messages 由 `src/qg_prqk/sft_data.py` 和 `scripts/build_sft_data.py` 一次扫描源 JSONL、同步写出两个分支。每行先严格核对旧 target 与 `target_poi_id`，再替换 history/target；NoGID assistant target 显式禁止 `<G_*>`。源三文件的行数和 SHA256 必须全部吻合后才原子发布。
- 共同词表、全量 Train/Valid 预检/Cache 与训练代码分别为 `sft_vocab.py`、`sft_preflight.py`、`sft_training.py`。两个 YAML 除 variant、dataset/cache/output 路径外算法参数完全相同：Qwen3-0.6B full SFT、3 epoch、4 GPU、global batch 512、LR=`5e-5`、seed=42、`qwen3_nothink`、packing、`train_on_prompt=false`、cutoff 1024。`dataset_info.json` 只注册两个分支的 Train/Valid，不注册 Test。
- 正式 Messages 命令使用 `/ofs/map_search/hudan/envs/poi-gr/bin/python -u qg_prqk/scripts/build_sft_data.py`，传入上述冻结源、两个 Final-ID 目录和 `qg_prqk/outputs/sft_data/` 下两个输出目录；完整可复制参数见 `qg_prqk/README.md`。运行环境为 Python 3.10.20、NumPy 1.26.4、PyArrow 19.0.1，纯 CPU 顺序读写。

### 核心结果

| 指标 | A4 GID-parent | A4 NoGID |
|---|---:|---:|
| Base ID 结构 | GID6+SID3 | SID3 |
| Base distinct | 672,294（93.8637%） | 613,198（85.6129%） |
| collision excess | 43,951 | 103,047 |
| colliding POI | 73,174（10.2163%） | 167,139（23.3355%） |
| max Base bucket | 56 | 159 |
| Dedup Token | D0–D55 | D0–D158 |
| Final unique | 716,245（100%） | 716,245（100%） |
| 需要 D 的订单目标 | 1,618,136（18.4078%） | 2,589,694（29.4601%） |
| Messages 目录大小 | 约 14 GiB | 约 12 GiB |

- 两个分支均完整写出 Train/Valid/Test=`7,586,410/597,421/606,682`，retained total=`8,790,513`，history occurrence=`43,208,167`。两份 `special_tokens.json` SHA256 都为 `b560b14a33d4e52a9b6d91956c90844d58ca45a0779ca2bfc42d14fc4413edb9`，共 3,743 个普通原子 Token：结构 16、GID 32、用户桶 2000、SID `512×3`、Dedup 159。
- GID-parent Train/Valid/Test JSONL SHA256=`42c51be6cebaa0b3b318a58f72296165fef4927fc0ef54111159c87e516141b3` / `0dcee9823fd645dcd8562c5a498f56f185692f3ece689915a0653c65ce0413ab` / `8aa7616fad72d48b0103481aae135e129b0f3d7856fef4b2f13456191c6a54d9`。
- NoGID Train/Valid/Test JSONL SHA256=`0276ac4c9f6dd9d818423e1b22c2425a374a7340bf4961aae3facdb1508538ef` / `e278408a4488a073439d38d0533c0968120324c5f4f79b4fd1754948d1ce2298` / `87d478f19039de7474a62ca7859317759e1917990b504d8c1d2bfc730bba7a30`。
- 首尾边界抽样覆盖 Train/Valid/Test 六个位置：两个分支的 sample/order/search/target POI/history length/split 全部相同，移除 identifier 后的输入上下文相同；GID target 恰有 GID6+SID3，NoGID target 恰有 SID3 且无 GID，两者均无旧 TIGER/C Token。

### 产物、失败记录与结论

- GID-parent Final-ID 目录为 `outputs/final_identifiers/a4_gid_parent_order_a_v1/`，manifest SHA256=`7c87a9d7e1c94874ad4ebf8fd61504f0bd52176e48f6c20ad88a5f72af9e5b86`；NoGID 为 `outputs/final_identifiers/a4_nogid_sid3_v1/`，manifest SHA256=`794a6069585b0f8c81fd5ddc646a2be82f4be2667816b060c4ba6dd5436309d8`。
- GID-parent SFT 目录为 `outputs/sft_data/a4_gid_parent_order_a_history10_v1/`，manifest SHA256=`c05810df785952b5e4412d4e67d31324587e2e02fb940ebb005331d9ea09e114`；NoGID 为 `outputs/sft_data/a4_nogid_sid3_history10_v1/`，manifest SHA256=`b173fe1411b6043727bd138a88b93bb9456b2bff89522d1c46b75e6304d21e65`。
- 正式构造从约 21:20 到 22:14 CST，墙钟约 54 分钟，包含完成写出后对约 26 GiB JSONL 的一次收尾文件复读 SHA256。最终代码保留写入时流式 SHA256 并去掉这次冗余复读，同时把顺序读写缓冲提高到 8 MiB；不改变 JSONL 内容或 manifest 口径，后续重跑会更快。
- 第一次日志重定向会话实际仍在运行但无终端回显；误判后曾启动第二个相同任务。发现两组 staging 后立即中止后启动任务，当时约完成 175 万 Train；其两个未发布临时目录共约 5.6 GiB 已删除，不能恢复。最早启动的正式任务未受影响，最终只有上述两个 canonical 目录。
- 定向验收为 Final ID `3 passed`、SFT 数据/预检/训练入口 `7 passed`；加入四卡平台 launcher 与分支独立短 TMPDIR 后，完整 QG 合成回归 `178 passed`（100.303 秒）。全目录 compileall、Ruff、184 文件目录 manifest 校验、两个 launcher 的 `bash -n`/`--help` 和 `git diff --check` 均通过。测试显式覆盖 NoGID target 不含 GID、末位 D、双分支上下文对齐、Train/Valid-only 预检、共同词表、算法配置一致性和运行时临时目录隔离。
- 结论：`COMPLETED_AND_VALIDATED`。两个 Final ID 与两套全量 Messages 均满足定义；NoGID 确实只有三位 SID 加可选末位 D，请求 GID 仍作为共同输入。本实验结束时固定停在 `HOLD_FOR_QG_PRQK_SFT_DATA_REVIEW`，当时共同扩词表、两套 1024 Tokenized Cache、两个 SFT checkpoint 和训练指标均未生成；后续词表与 Cache 结果见 `EXP-20260912-01`。

## EXP-20260912-01：共享扩词表与双分支全量 Tokenized Cache

### 目标与假设

- 在不启动 Qwen SFT、不读取业务 Test 的前提下，为已冻结的 A4 GID-parent 与纯三位 NoGID Messages 构造同一个扩词表初始模型，以及两个不可覆盖的 Train/Valid Tokenized Cache，使两套四卡训练入口达到可启动状态。
- 假设：两份逐字节相同的 3,743 Token 清单可得到同一 tokenizer/model 初始状态；两个分支全部 Train+Valid 样本在 `qwen3_nothink`、`train_on_prompt=false` 口径下都不超过 1024，且 assistant target 不会被截断。

### 数据版本与代码状态

- GID-parent / NoGID Messages manifest SHA256 分别为 `c05810df785952b5e4412d4e67d31324587e2e02fb940ebb005331d9ea09e114` / `b173fe1411b6043727bd138a88b93bb9456b2bff89522d1c46b75e6304d21e65`。本实验只读取两者的 Train/Valid=`7,586,410/597,421`，不读取或注册各自 606,682 条 Test。
- GID-parent Train/Valid SHA256=`42c51be6cebaa0b3b318a58f72296165fef4927fc0ef54111159c87e516141b3` / `0dcee9823fd645dcd8562c5a498f56f185692f3ece689915a0653c65ce0413ab`；NoGID Train/Valid SHA256=`0276ac4c9f6dd9d818423e1b22c2425a374a7340bf4961aae3facdb1508538ef` / `e278408a4488a073439d38d0533c0968120324c5f4f79b4fd1754948d1ce2298`。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，工作树已有用户改动并保持不动；新增/修改的 QG 代码、配置、测试、launcher、日志和产物均位于 `qg_prqk/`。未 commit/push。

### 配置、命令与环境

- 共享词表由 `scripts/prepare_sft_vocab.py` 从 `models/Qwen3-0.6B` 和两个 `special_tokens.json` 构造。两个 cache 均由 `scripts/prepare_sft_cache.py` 构造，固定 `cutoff_len=1024`、`packing=true`、`qwen3_nothink`、`train_on_prompt=false`、seed/data_seed 42、preflight batch 4096、16 个预处理 worker、预处理 batch 1000。
- 正式 cache 在宿主机运行，使用 `/ofs/map_search/hudan/envs/poi-gr/bin/python`、Python 3.10.20、LLaMA-Factory 0.9.4、Transformers 4.52.4、Hugging Face Datasets 3.6.0，纯 CPU + OrangeFS；短 `TMPDIR` 分别位于 `qg_prqk/outputs/tmp/`。使用宿主机是因为沙箱禁止 multiprocessing 的 AF_UNIX listener，并非算法或数据要求。
- 训练 launcher 固定为 `launchers/run_train_qg_prqk_a4_gid_parent_sft_4gpu_3epoch.sh` 与 `launchers/run_train_qg_prqk_a4_nogid_sft_4gpu_3epoch.sh`。两者 `--dry-run` 检查共同模型、对应数据/cache manifest、1024 零截断门禁和冻结算法配置；正式配置均为 4 GPU、full SFT、3 epoch、per-device batch 8、gradient accumulation 16、global batch 512、bf16、LR `5e-5`。

### 核心结果

- 共享模型输出为 `outputs/models/Qwen3-0.6B-QGPRQK-A4-Vocab-v1/`，约 1.2 GiB。base tokenizer=`151,669`，新增普通原子 Token=`3,743`，最终 tokenizer=`155,412`；`qg_prqk_token_mapping.json` SHA256=`ced19fd4025662e0faabc3e8f032c1b89e3bc28bfba64c6402bfff3ad41f9d1b`，extended tokenizer SHA256=`e4013655c39ae69d1e8808ec412168c8553d86c84f9b6cc32b757cf58f55ec54`。扩展前 embedding 行逐值保持，新增模型行均 finite，重载后 input/output embedding 仍 tied。

| 指标 | A4 GID-parent | A4 NoGID |
|---|---:|---:|
| Train+Valid 原始行 | 8,183,831 | 8,183,831 |
| Train packed 行 | 1,527,130 | 1,263,217 |
| Valid packed 行 | 114,056 | 94,446 |
| total length max | 988 | 947 |
| total length p50/p90/p95/p99/p99.9 | 178/364/372/404/513 | 149/299/307/339/448 |
| target length max | 14 | 8 |
| 超 1024 | 0 | 0 |
| target truncation | 0 | 0 |
| Cache 大小 | 约 33 GiB | 约 28 GiB |
| Cache manifest SHA256 | `5f0ac5f1ca64e3faf7ee1c20acff669071df395ff3fb50369ce5e889c7351576` | `5ddc9d12f59df2b6f65d2931e45e9a3a39e3dab5db6fbb642c47e79d22906d59` |

- 两个 cache 分别发布到 `outputs/sft_tokenized/a4_gid_parent_history10_v1/` 与 `outputs/sft_tokenized/a4_nogid_history10_v1/`，均有 `cache_manifest.json` 和 `_SUCCESS`，临时 `.building-*` 目录已清理。两个 cache 绑定同一个模型路径和 extended tokenizer SHA，绑定各自 Train/Valid 原始 SHA。
- 两套 launcher 的 `bash -n` 和完整 `--dry-run` 均退出 0；dry-run 显示两个分支除 dataset/cache/output/variant 外训练算法配置一致。训练输出目录没有创建，没有 QG SFT 训练进程，checkpoint 和训练指标仍为 `NOT_STARTED`。

### 失败记录、验收与结论

- GID attempt01 使用禁用 tokenizer 并行的慢速 preflight，在 50 万条时人工中止，退出 130；未产生 preflight state 或 cache，现场保存在 `outputs/run_control/sft_cache_a4_gid_parent_v1_attempt01_slow_interrupted/`。
- GID attempt02 完成全部 8,183,831 条 preflight 并写入可复用 state，但在沙箱启动 16 worker 时因 `PermissionError: [Errno 1] Operation not permitted` 创建 AF_UNIX listener 失败，退出 1；没有发布 cache。attempt03 在宿主机复用该 preflight state 后成功，退出 0。NoGID attempt01 在宿主机完成 full preflight 与 cache，退出 0。所有成功/失败日志均保存在各自 `outputs/run_control/` 目录，不覆盖历史记录。
- 完整门禁验证了 Train/Valid 行数、源 SHA、共享 tokenizer、长度分位数、零超长、零 target truncation、packed 非空、manifest 和 `_SUCCESS`；Test 未读取。两个训练入口的 `--dry-run` 再次验证 cache/data manifest 和算法配置，均退出 0。
- 全目录 compileall、Ruff 和正确 `PYTHONPATH=qg_prqk/src` 下的完整 QG 合成回归均通过，测试结果为 `178 passed`（191.92 秒）。
- 结论：`P10_SHARED_VOCAB_AND_DUAL_TOKENIZED_CACHE_COMPLETED_AND_VALIDATED`。本实验结束时停在 `HOLD_FOR_QG_PRQK_DUAL_SFT_LAUNCH_REVIEW`；当时没有自动启动 SFT，不产生效果指标，也不进入 Test、端到端评测、外部基线、Order-B 或其他消融。后续用户启动的 SFT 见下条记录。

## EXP-20260913-01：双分支四卡 A100 三轮 SFT

### 目标、数据与配置

- 目标：用完全相同的初始 Qwen3-0.6B、共同词表和优化协议，分别学习 GID-parent 与 NoGID Final ID，随后用同一 Validation 生成检索协议判断 SID 是否可学习。本记录于 2026-09-14 按用户已运行的产物补齐，不重新训练。
- 数据与来源沿用 `EXP-20260912-01`：原始 Train/Valid=`7,586,410/597,421`；GID-parent / NoGID packed Train=`1,527,130/1,263,217`、packed Valid=`114,056/94,446`，两个分支分别绑定该条列出的完整 JSONL/cache/model 哈希。
- 配置为 `configs/sft/a4_gid_parent_history10_v1.yaml` 与 `configs/sft/a4_nogid_history10_v1.yaml`（相对 `qg_prqk/`）。统一 full SFT、qwen3_nothink、cutoff=1024、packing、bf16、3 epoch、per-device batch=8、accumulation=16、global batch=512、LR=5e-5、AdamW、cosine、warmup=0.03、seed/data_seed=42；save/eval 按 epoch，不加载最佳模型、不生成评测、不注册 Test。
- 代码：`src/qg_prqk/sft/training.py` 调用本地 LLaMA-Factory；交接时 Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，工作树为 dirty。未将当前 HEAD 伪称为已提交的训练快照；源配置和实际训练命令以保留的 console log 为依据。
- 环境：训练平台 4×A100、`/ofs/map_search/hudan/envs/poi-gr`；LLaMA-Factory 0.9.4、日志确认 Transformers 4.52.4、599,609,344 个参数全部可训练。

仓库根目录启动命令（历史已执行，不要重复）：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a4_gid_parent_sft_4gpu_3epoch.sh
bash qg_prqk/launchers/run_train_qg_prqk_a4_nogid_sft_4gpu_3epoch.sh
```

### 已核验结果

| 指标 | A4 GID-parent | A4 NoGID |
|---|---:|---:|
| epoch 1 Valid loss | 0.2780051231 | 0.4551329613 |
| epoch 2 Valid loss | 0.2099997401 | 0.3522982299 |
| epoch 3 Valid loss | 0.2006811798 | 0.3390350640 |
| 训练平均 loss | 0.3775110184 | 0.5687756923 |
| final step / max_steps | 8949 / 8949 | 7404 / 7404 |
| Trainer train_runtime 秒 | 71,681.3223（约 19.91 小时） | 59,749.5220（约 16.60 小时） |
| 平台 launcher 退出码 | 0 | 0 |

- 最终权重：`outputs/sft/a4_gid_parent_history10_v1_gpu4_e3/checkpoint-8949/` 与 `outputs/sft/a4_nogid_history10_v1_gpu4_e3/checkpoint-7404/`。根目录 `train_results.json`、末步 `trainer_state.json`、三个 epoch 的 checkpoint 和 tokenizer 均保留。
- 日志与退出码：`outputs/run_control/sft_a4_gid_parent_history10_v1_gpu4_e3/`、`outputs/run_control/sft_a4_nogid_history10_v1_gpu4_e3/` 下的 `train_console_a100.log`、`train_a100.exit`。
- 文件时间核验：GID / NoGID 的 `train_results.json` 修改时间分别为 `2026-09-13T14:52:45Z`、`2026-09-13T20:05:39Z`（北京时间 9 月 13 日 22:52、9 月 14 日 04:05）；时间是产物元数据，不替代 Trainer 的实际运行时长。
- 外层 `qg_prqk_training_manifest.json` 未落盘，不能声称存在该成功标记。使用 exit=0、完整末步和 Trainer 结果交叉确认训练完成；不修改历史训练目录。评测准备产生的 `plan.json` 会记录实际 checkpoint/model SHA256、词表、数据来源与评测代码哈希。

### 结论与下一步

- 两个分支均可训练且三轮 Valid loss 持续下降。由于目标长度与 Token 分布不同，不能直接由 loss 更低断言 GID 的精确 POI 召回优于 NoGID；目前没有生成检索指标。
- 2026-09-14 用户先要求四卡 6000D 评测，随后因资源变化改为**一台双卡节点统一评测两个分支**。当前入口是 `run_evaluate_qg_prqk_a4_dual_epoch3_fixed10k_generalization_2x6000d.sh`；配置仍为 `configs/sft/evaluation_epoch3_fixed10k_generalization_v1.yaml`，命令/输出路径见 `../../README.md`。固定 epoch 3，既有随机 10k 加四类互斥泛化各 10k，按 `(order_id, searchid)` 对齐；无约束 Beam=10，非法/重复 beam 保持排名，不用 Test 或五组集合选模。只改变总并发/调度，不改变数据、单卡 batch、生成或指标；旧四卡入口已移入可恢复归档，未触碰已训练 checkpoint。
- 评测代码/合成测试/CPU 预检属于当时交接准备，不登记为已完成生成实验。当时状态停在 `HOLD_FOR_QG_PRQK_DUAL_SFT_EVALUATION`；用户随后启动的正式推理及最终结果见 `EXP-20260914-01`。

## EXP-20260914-01：双分支 epoch-3 固定 10k 与四类泛化评测

### 目标、数据与协议

- 目标与假设：验证两套已训练 Final ID 的端到端精确 POI 检索能力，观察 GID-parent 的地理前缀与 NoGID 较短路径的收益和代价；不使用 SFT loss 代替生成检索比较。
- 模型沿用 `EXP-20260913-01`：GID-parent 为 `checkpoint-8949`，NoGID 为 `checkpoint-7404`，均固定 epoch 3，不再选模。两套 Final ID 均覆盖北京 active POI 716,245 条；目标分别为 `GID6+S1+S2+S3+[D]` 与 `S1+S2+S3+[D]`，S3 parent 和碰撞后缀也不同，不能将差异仅归因于增删六个 GID token。NoGID 输入仍保留当前请求的用户 GID。
- 数据是既有 2026-07-13 Validation 固定随机 10k，以及已见 Query/未见 Query–POI、新 Query/已见目标、长尾目标、冷目标四组各 10k。两个分支按 `(order_id, searchid)` 同行同序对齐，五组互斥，共 50,000 个业务键、100,000 次模型评测。没有重新抽样或读取 Test。
- 配置：`configs/sft/evaluation_epoch3_fixed10k_generalization_v1.yaml`（以下路径均相对 `qg_prqk/`）。使用相同 LLaMA-Factory `qwen3_nothink` 模板、history10、cutoff=1024、BF16、无约束 Beam=10、batch=32、chunk=500、length_penalty=1、early_stopping=true、renormalize_logits=true；GID / NoGID 的 max_new_tokens=13 / 7。完整 Final ID（含必要的末位 D）精确查表；非法和重复 beam 不删除、不前移排名，不做 Bucket 展开或 Trie 约束。
- 泛化宏平均只对四个专项集等权平均，不包含固定随机 10k；合法 ID 率的分母是所有生成候选，而不是 Query 数。

### 代码、命令与环境

- 实现：`src/qg_prqk/sft/evaluation_data.py` 对齐冻结样本，`evaluation.py` 负责生成/完整 ID 解析/指标累计，`evaluation_suite.py` 负责双卡共享十项队列和汇总；入口为 `commands/evaluate_sft.py`。
- Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，工作树 dirty；实际六个运行源码 SHA256 绑定在各分支 `plan_2x6000d.json`，结果复核时与当前源码完全一致，不将 HEAD 冒充已提交实验快照。
- 平台为同一节点 2×NVIDIA RTX 6000D，日志显示启动前每卡空闲约 82.6/83.0 GiB；使用 `/ofs/map_search/hudan/envs/poi-gr/bin/python` 与本地 LLaMA-Factory。每个 worker 独占一张可见 GPU，空闲卡领取下一项，无四卡依赖。十项实际 batch 均为 32，无 OOM 降 batch 日志。
- 用户已执行的命令（仅记录，不重复启动）：

```bash
bash qg_prqk/launchers/run_evaluate_qg_prqk_a4_dual_epoch3_fixed10k_generalization_2x6000d.sh
```

- 启动/完成文件时间为北京时间 2026-09-14 11:57:55 / 13:24:44，总墙钟约 1 小时 26 分 49 秒（包含预检、初始化和调度）。GID / NoGID 五项累计 chunk 推理耗时为 4,886.20 / 3,175.09 秒；两者并行，不能把两项相加作为平台墙钟。

### 最终结果

下表数值均为百分数；HR@K 在本次单目标精确 POI 协议下即 Recall@K。

| 分支 | 评测集 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | MRR@10 | 合法 ID 率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GID | 固定 10k | 50.3100 | 74.4600 | 80.7400 | 85.1900 | 68.7252 | 63.3241 | 65.4020 |
| GID | 已见 Query/未见 Query–POI | 19.0300 | 40.4200 | 50.1100 | 59.0500 | 38.3725 | 31.8073 | 62.9320 |
| GID | 新 Query/已见目标 | 46.9300 | 62.7800 | 67.6700 | 72.1400 | 59.8341 | 55.8536 | 41.8470 |
| GID | 长尾目标 | 18.9100 | 29.5600 | 34.4800 | 41.3900 | 29.3539 | 25.5942 | 32.3100 |
| GID | 冷目标 | 5.8300 | 9.3800 | 11.3500 | 14.1800 | 9.5811 | 8.1598 | 26.3360 |
| GID | 四类泛化宏平均 | 22.6750 | 35.5350 | 40.9025 | 46.6900 | 34.2854 | 30.3537 | 40.8563 |
| NoGID | 固定 10k | 50.1900 | 73.5700 | 79.9700 | 84.6700 | 68.2651 | 62.9015 | 69.8440 |
| NoGID | 已见 Query/未见 Query–POI | 17.6100 | 37.0700 | 46.4700 | 55.8200 | 35.8329 | 29.5248 | 69.2100 |
| NoGID | 新 Query/已见目标 | 44.4500 | 61.3500 | 66.1300 | 70.8700 | 57.9197 | 53.7371 | 51.0720 |
| NoGID | 长尾目标 | 18.8100 | 29.2900 | 34.9200 | 41.5600 | 29.3676 | 25.5546 | 41.1870 |
| NoGID | 冷目标 | 6.4400 | 10.4100 | 12.9200 | 17.0800 | 11.0923 | 9.2589 | 35.0900 |
| NoGID | 四类泛化宏平均 | 21.8275 | 34.5300 | 40.1100 | 46.3325 | 33.5531 | 29.5189 | 49.1398 |

### 产物与独立核验

- 总运行控制：`outputs/run_control/eval_a4_dual_epoch3_2x6000d/`，`full.exit=0`，十个 worker 的 `.exit` 均为 0；总日志包含两个“五项评测全部完成”标记。逐 500 条进度只写各 worker 的 `.console.log`，平台主日志只显示启动/完成，不代表无进度。
- 结果根目录：`outputs/eval/sft_epoch3_fixed10k_generalization_v1/`；两个分支各保存 `plan_2x6000d.json`、`data/manifest.json`、五组 JSONL、`results/<subset>/{progress,result}.json`、`results/summary.json` 和 `results/summary.csv`。未修改既有 checkpoint、SID、样本、配置或评分源码。
- GID / NoGID 的 plan signature 为 `9fce855da15ac383e78419dc94fe6742a29de21d8374843acf21e6f15e57e3ea` / `86cc2ac0f618b1129f1072ac0a4c4f0de4bad717eaad518969b800e63095d5a7`。
- GID / NoGID 的 `summary.json` SHA256 为 `e225cf8d9a5c08fa5c75f821ecd9ac3a5e0bb34ae0bc75f7f9ed845ba244f35c` / `744918f6ef6d23ed594ed22f176fe40c3388f0d316eca7ce0822a0927e2a0fb1`。
- plan 绑定的 GID / NoGID `model.safetensors` SHA256 为 `fa24f17687b1c89876abb5bde57abe94865f7fc62d72b9b08649e49f179d81b7` / `b544f2c58ceecde3940dd44d1b9ee652d2889dd5f7551b3349cd2046762f71ca`；此次复核其及 tokenizer 等 checkpoint 文件的大小和纳秒修改时间均与启动时完整哈希记录一致，不重复声称重新计算了模型全文件哈希。
- 独立只读复核通过：十份评测 JSONL 全文件 SHA256、每组 10,000 个唯一业务键、分支间逐项同序及五组互斥；plan/run/progress 来源签名与源码哈希；每组 100,000 个候选、hit/invalid 计数守恒；从目标排名直方图重新计算 HR@1/3/5/10、NDCG@1/3/5/10、MRR@10，并从原始合法候选计数复算合法率，与 result 和 summary 一致；四类宏平均复算一致。此项是累计统计复算，没有重新生成或逐条重放全部 beam。

### 结论与下一步

- 两个分支均完成从 SID 到 SFT 再到生成检索的工程闭环。GID 相对 NoGID 的固定 HR@1/HR@10/NDCG@10 高 0.12/0.52/0.4602 个百分点；四类宏平均高 0.8475/0.3575/0.7323 个百分点，属于当前单次实验的数值优势，不宣称统计显著。
- GID 的优势主要在已见 Query/未见 Query–POI 和新 Query/已见目标；长尾基本持平，冷目标 NoGID 的 HR@1/HR@10/NDCG@10 反而高 0.61/2.90/1.5112 个百分点。不能说 GID 全场景更优。
- NoGID 的合法 ID 率各组均更高。两版五组分别有 271,173 / 233,597 个非法候选，其中 271,024 / 233,448 个是 `corpus_miss`（超过 99.9%），主要表现为完整 ID 组合不在目录，而不是 wrapper/EOS 格式错误；尚不能定位具体哪一层或 D 后缀导致，也不能断言约束解码一定提高召回。
- 本次只比较两个 QG 任务，不新增外部基线实验或对全部方法宣称胜出。状态停在 `HOLD_FOR_QG_PRQK_DUAL_SFT_RESULTS_REVIEW`；下一步建议先审核两版结果，再由用户确认同 active 目录基线比较或合法性诊断。未启动 Test、Trie、RL、额外训练或消融。

### 2026-09-14 后续交接：最后一天全量 Test 入口（正式推理未运行）

- 用户随后确认补充同一台双卡 6000D 的全量 Test 评测脚本。参考既有 `launchers/run_evaluate_tiger_mmbert_active716k_epoch3_full_test_2x6000d.sh` 的最后一天协议与原 JSONL 字节偏移视图；复用实现已适配复制到 QG，并登记 `configs/code_provenance.yaml`，不调用根目录脚本。
- QG 入口为 `launchers/run_evaluate_qg_prqk_a4_dual_epoch3_full_test_2x6000d.sh`，配置为 `configs/sft/evaluation_epoch3_full_test_20260714_v1.yaml`；两版各读取 2026-07-14 全部 606,682 条 Test，分别固定 `checkpoint-8949 / checkpoint-7404`。沿用本实验模型、模板、无约束 Beam=10 和完整 Final ID 指标口径，不从 Test 选模或改参，原 Validation 源码指纹和结果保留。
- 数据流：`sft/full_test_data.py` 流式核验两版原始 Test 并提供惰性行视图；`sft/full_test_evaluation.py` 建立冻结计划、双卡各跑一个分支、分块生成与续跑，进度同时进入 worker 和平台主日志；最终两版指标及差值写入 `outputs/eval/sft_epoch3_full_test_20260714_v1/`，与 Validation 和 smoke 隔离。
- 真实 CPU `--dry-run` 退出 0，全量行数、文件哈希、业务键唯一性/同序和当前请求/目标对齐均通过；两版各 716,245 条 Final ID 与前 100 条训练同模板 Token 检查通过，剩余 Token 长度在实际生成 chunk 内检查。39 项相关合成测试、Ruff、compileall、shell 语法/帮助和代码来源检查通过。环境仍为 `poi-gr`，工作树 dirty 保留；完整来源哈希、两份 plan 和 CPU 日志均在 QG 输出目录。
- 本项是可复现入口与 CPU 预检交接，不登记新正式实验编号，也没有 Test 效果指标。当前状态更新为 `HOLD_FOR_QG_PRQK_FULL_TEST_LAUNCH_REVIEW`，GPU 正式推理待用户启动；最短命令、输入输出及验收方式见 `../../README.md` 的“最后一天全量 Test 双卡评测”。

### 2026-09-15 后续交接：双卡约束解码（正式 GPU 评测未运行）

后续用户另确认增加双卡约束解码评测，A0-GID 入口保持独立，不由该评测任务启动。约束入口是 `launchers/run_evaluate_qg_prqk_a4_dual_epoch3_constrained_fixed10k_generalization_2x6000d.sh`；配置 `configs/sft/evaluation_epoch3_constrained_fixed10k_generalization_v1.yaml` 按 SHA256 继承两份已完成的无约束 plan，完整复用固定 10k 和四类泛化样本、末轮模型、词表与生成参数。`sft/constrained_decoding.py` 只在 `generate` 注入完整 716,245 POI 的合法前缀 callback，包含可选 D、闭合和 EOS，不按 Query/区域剪枝；`sft/constrained_evaluation.py` 负责独立断点、两卡调度、进度转发及相对无约束差值。`renormalize_logits=true` 保持原值，mask 后合法后继概率会重新归一化，不能把效果解释成简单过滤旧候选。

实际完成的是代码与 CPU 预检交接：真实 launcher `--dry-run` 退出 0，两版全目录和十份冻结样本的哈希、数量、业务键顺序/互斥、目标与 CURRENT 对齐通过；31 项合成回归通过，含小型随机 Qwen 的真实 Beam=10（不是业务 checkpoint 推理）。旧调度测试的随机长临时目录曾触发路径门禁，已仅在两项调度夹具中 mock 路径编码；另有真实长路径拒绝测试，生产 64 字节门禁不改。运行环境为 `poi-gr`、QG 内短 TMPDIR，工作树 dirty，未 commit/push。宿主仅单卡 A6000 且无平台提交接口，正式双卡 GPU 评测未启动；不登记正式实验编号或业务指标。平台默认先两版各 2 条 GPU smoke，通过后再完整十项；产物进入 `outputs/eval/sft_epoch3_constrained_fixed10k_generalization_v1/`，日志进入 `outputs/run_control/eval_a4_dual_epoch3_constrained_2x6000d/`，与旧无约束/Test/A0 完全隔离。下一步由用户提交双卡脚本，完成后比较逐组 Recall、NDCG、MRR 和合法 ID 率，四类宏平均不含固定集。

### 2026-09-15 后续交接：A0-GID 平台准备与四卡 SFT（未全量运行）

- 用户确认增加 A0-GID 端到端对照，并要求把全量数据构造、长度预检和 Tokenized Cache 移到训练平台执行，开发机只完成代码和轻量核验。本项不登记正式实验编号，不补写 A0 SFT 指标。
- 数据流：冻结 A0 `full_716245` 的三位 SID，前置与 A4 相同的六位 GID，按完整九位前缀碰撞重新分配末位可选 D；以 A4-GID 冻结语料逐条替换历史和目标 identifier。保留当前 Query/GID、历史事件、业务主键、顺序和 split；预期 Train 7,586,410 行、Valid 597,421 行，准备阶段不读取 Test。所有新数据与原 A4 隔离。
- 代码：`sft/a0_gid_data.py` 负责目录与语料转换；`sft/a0_gid_pipeline.py` 负责来源合同、单进程准备、缓存验收和训练调度；`scripts/a0_gid_sft.py` 是薄入口。复用 QG 内已有的 Dedup/格式、Token 预检和缓存实现，不修改已冻结的 A4 CLI、训练和评测源码。
- 配置：`configs/sft/a0_gid_pipeline_v1.yaml` 保存来源 manifest/词表 SHA256 和预期行数；`configs/sft/a0_gid_history10_v1.yaml` 与 A4-GID 使用同一个扩词初始 Qwen3-0.6B、LLaMA-Factory 后端和训练超参，非 A4 SFT checkpoint。四卡、BF16、3 epoch、cutoff=1024、packing、每卡 batch=8、累积 16、global batch=512、seed=42、学习率 5e-5。不按 A100/6000D 型号名称限制，但要求四张可用 BF16 CUDA 卡并检查空闲显存；不自动改变 batch 或词表。
- 平台命令：`bash qg_prqk/launchers/run_train_qg_prqk_a0_gid_sft_4gpu_3epoch.sh`。启动器继承已有平台环境与挂载，先屏蔽 CUDA、由一个 CPU 主进程顺序完成全量准备，再恢复四卡训练。`--dry-run` 仅检查小型合同，`--prepare-only` 仅完成平台准备；任何阶段失败都停止后续训练。首次仍需 CPU/I/O 时间，GPU 在准备期间空闲，不声称缓存改由 GPU 加速。
- 环境与工作树：`poi-gr` Python，项目内短 TMPDIR=`qg_prqk/outputs/tmp/qgsa0`，Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`、工作树 dirty；未 commit/push。启动器保留在忽略目录，不把平台敏感字段写入文档或源码清单。
- 实际核验：相关单元测试 25 项通过；另用真实共享 tokenizer 与 LLaMA-Factory 对 4 条合成 Train、4 条合成 Valid 完成零截断检查、packed Cache 生成及复用，1 项通过、耗时 177.159 秒。此测试无真实业务样本、无 GPU 训练。真实启动器 `--dry-run` 退出 0，报告 `planned_not_prepared`，A0 全量 Final ID、Messages、Cache、训练输出均未生成；不能将其解读为全量预检已通过。
- 准备成功的验收标记为 `A0_GID_PREPARATION_COMPLETED`，产物分别进入 `outputs/final_identifiers/a0_gid_order_a_v1/`、`outputs/sft_data/a0_gid_order_a_history10_v1/`、`outputs/sft_tokenized/a0_gid_history10_v1/`；训练进入 `outputs/sft/a0_gid_history10_v1_gpu4_e3/`，阶段日志与缓存凭证进入 `outputs/run_control/sft_a0_gid_history10_v1_gpu4_e3/`。首次计算来源/产物 SHA256，已完成阶段按相同合同及文件集合、大小、纳秒修改时间复用；半成品不逐行续跑，非空训练输出不覆盖、不隐式续训。
- 当前状态停在 `HOLD_FOR_A0_GID_PLATFORM_LAUNCH_REVIEW`。下一步由用户在四卡训练平台启动此脚本；待正式完成后，再登记实际数据/缓存/训练指标与来源哈希，之后按同一生成评测协议比较 A0-GID 与 A4-GID。原 A4 产物及全量 Test 入口保持不变。

### 2026-09-16 后续交接：PRQ-KMeans 单卡 A100 双解码评测入口

- 用户报告 A0-GID SFT 完成，本次只读核实末轮 trainer_state：训练日志记录 epoch 1/2/3 对应 step `2985/5970/8955`；末轮状态为 `epoch=3.0`、`global_step=max_steps=8955`。记录的 Validation loss 为 `0.277653/0.208086/0.198839`。固定末轮评测，不依据待跑五组指标选择模型；未重训、未修改 SID，也不据 loss 宣称检索胜出。
- 来源绑定：数据 manifest SHA256=`4605137fefe865f655d42398dcabfe34cab170df6a62842f2474e652be1a003e`；A0-GID 标识 manifest SHA256=`1237c138783054c426fde41521c2964b2132b231a6c2b150865e9a962dc4f82a`；末轮 trainer_state SHA256=`1ef7dad9df283f7ccc62473e5d3a5e897c30f0516971e4301bc96a6844d32bfb`。标识 manifest 记录 716,245 个全局唯一 Final ID，结构为 `GID6+S1/S2/S3+[D]`。
- 新代码为 `sft/a0_evaluation.py` 与薄入口 `scripts/evaluate_a0_gid_sft.py`，配置 `configs/sft/evaluation_a0_gid_epoch3_dual_decode_v1.yaml`；直接调用旧 QG 的模板、checkpoint 检查、业务键对齐、Beam 生成/计分与全目录 Trie，不修改冻结 A4 评测源码。新增部分只处理 A0 来源/九位格式适配、单模型双解码调度和断点汇总，保留真实 `variant=a0_gid`，不冒充 A4。
- 资源随后调整为单卡 A100，当前平台命令为 `bash qg_prqk/launchers/run_evaluate_prqk_a0_gid_epoch3_dual_decode_1xa100.sh`。同一张卡先运行无约束 worker，再运行约束 worker；每个 worker 加载一次模型后顺序评测固定 10k 与四类泛化。默认先在平台准备缓存和核验全目录，再十项各 2 条 smoke，全部通过后十项各 10,000 条正式评测。Beam=10、cutoff=1024、max_new_tokens=13、length_penalty=1、renormalize_logits=true，与 QG-GID 一致；不读 Test，非法/重复候选保持原排名，四类宏平均不含固定集。旧双卡 6000D 入口只读保留用于资源恢复时复现，不与单卡任务同时运行。
- 目前完成的是启动器静态 `--dry-run`、语法/帮助/compileall 与轻量合成测试；静态预检不等于全量 SHA256/Token/GPU 门禁通过。真实 checkpoint 推理、50k 对齐扫描和单卡 smoke 由平台启动时执行，尚无 PRQ 正式检索指标，不登记新评测实验编号。所有产物进入 `outputs/eval/a0_gid_epoch3_dual_decode_v1/`，日志进入 `outputs/run_control/eval_a0_gid_dual_decode_v1/`；环境为 `poi-gr`，工作树 dirty 保留，未 commit/push。最短命令和输出结构见 `../../README.md` 的 PRQ 双解码章节。

## EXP-20260917-02：A0/A4 配对输入与全量 S1 内容/图分配诊断

### 目标、范围与数据

用户要求继续核验 A0 消融收益较小及 Query 可预测性下降的原因。先排除配对语料和标识替换错误，再检查 S1 是否只是 Top-1 换位，以及图分配与 content-only 预测的差距。使用冻结五组 Validation 各 10,000 条、Train-derived 342,879 个 Query 和 514,500 条 S1 边；不读取业务 Test、不更新 Query 向量、码本、SID 或模型。A0/A4 为同 active 716,245 库的 POI-only PRQ 与完整 GID-parent 方法。

### 实现、配置、环境与复现

入口 `scripts/diagnose_a0_a4.py` 复用 P8 的输入加载、边映射和加权计分，以及 A0 Messages 的 remap_record；NumPy 只新增分块 S1 目标排名和固定参数的 Query 分配公式复算。配置沿用 P8 `configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml` 和两个已完成评测 plan。当前 Git HEAD=`54802e6674e283722ecee00fb862530df30cef9a`，工作树 dirty，未 commit/push；诊断源码 SHA256=`d8b7e07e7383e4cbaae10a2bb40f659d17f26d78cfdefbd07a7e71309acc6c93`。运行环境为 poi-gr Python、CPU BLAS 8 线程、chunk_rows=2048，短 TMPDIR 为项目内 `qg_prqk/outputs/tmp/diag`。

```bash
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/diag \
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' \
/ofs/map_search/hudan/envs/poi-gr/bin/python qg_prqk/scripts/diagnose_a0_a4.py \
  --output qg_prqk/outputs/eval/a0_a4_diagnostic_v1/result_verified.json
```

输出已存在时拒绝覆盖；复跑需换一个未使用的 JSON 路径。输入各实际使用的静态数组、边表、配置和配对 Messages 已做完整 SHA256 核验；哈希清单写入结果。没有重新哈希 SFT 大模型权重，因为本轮未加载 Qwen。

### 核验结果

- 五组共 50,000 条的业务键、目标 POI、CURRENT、非标识历史输入与历史长度全部一致；从冻结 A4 目录向 A0 重新执行标识映射，50,000 个目标及 232,409 个历史事件与实际 Messages 全部一致。十份 Messages 文件完整哈希与原计划一致。
- 除 variant、数据/输出路径等必要字段外，训练 YAML 无超参差异；冻结计划的解码参数与 token mapping 指纹一致。Cache manifest 均记录 Train 7,586,410、Valid 597,421，cutoff=1024、零超长与零 target 截断。A0/A4 packed Train 为 1,528,165/1,527,130，epoch 3 步数为 8955/8949；训练预算是相同 epoch，不是完全相同步数。Cache 指标本次只读已有 manifest，未重新构建或扫描 Train。

| S1 content-only 加权目标命中率 | A0 | A4 |
|---|---:|---:|
| Top-1 | 29.9614% | 21.0191% |
| Top-5 | 50.1852% | 38.6014% |
| Top-10 | 57.4874% | 47.5105% |
| Top-50 | 74.9414% | 70.0352% |

- 两版 Top-1 与冻结 P8 结果复算一致。D3 的 Top-10 为 A0 60.9750%、A4 50.7228%，说明不是仅第一名在近似等价候选间换位。
- A4 训练分配的 token 是内容 Top-1 的 Query 占 40.3413%，在内容 Top-10 内占 67.4594%；其相对最优码字的 cosine gap 均值 0.134112，p50/p90/p99 为 0.061869/0.357723/0.541585。这些是按 Query 计数的指标，与加权边准确率分母不同。
- 加权边四格分解：图分配正确且内容预测正确 20.9699%；图分配正确但内容预测错误 60.7058%；图分配错误但内容正确 0.0492%；二者都错误 18.2751%。图分配约 81.6757% 的一致率主要不能由同一冻结向量的纯内容 Top-1 复现。
- 使用同一 S1 Query 码本、冻结 POI labels、完整图边和正式 query_distortion_weight=graph_alignment_weight=0.05，复算 argmax(0.05*cosine + 0.05*同码邻边权重和)，与导出 Query 分配一致率为 99.9976668%（342,879 条中 8 条不同）；不是完全逐值一致，不把剩余差异未经核验归因于浮点误差。本轮不重训，也不据此评价其他 graph weight 的效果。

### 状态、产物、结论与下一步

两次 CPU 核验均退出 0；第一份 `result.json` 仅包含非标识配对和 S1 排名，完整补核验结果为 `outputs/eval/a0_a4_diagnostic_v1/result_verified.json`，SHA256=`883ceea402a1a6945312e047a6a7b622a3ac22e9a0fa1e5b64de0e3b0d6c090b`。PID、日志和退出码在 `outputs/run_control/a0_a4_diagnostic/`，最终以 `verified.exit=0` 为准。2 项合成测试（排名平局、配对范围与历史计数门禁）、compileall、CLI 帮助和 git diff --check 通过；没有生成图表。

结论：本轮未发现所核验范围内的数据配对/历史标识替换错误，且全量 S1 显示明确的图耦合分配与内容可预测性差距。公式近乎逐值复现训练分配，支持这是当前目标的实际优化结果，而不是 P8 简单读错码本；结合 EXP-20260910-01 的 S2 受控诊断，优先怀疑图对齐收益未充分转化为脱离图的内容预测能力。不过这是静态量化层证据，尚不能证明它是 Qwen 最终收益较小的全部原因，也不能单独归因于 category/Geo 或宣称新权重会改善检索。

原评测没有保存逐条 beam，无法从聚合排名还原 Qwen 的分层准确率。宿主 GPU 两次读取均为利用率 100%，且有计算进程，故本轮不追加 GPU 推理、不影响用户已启动的四模型 Test；Qwen teacher-forced 分层和累计生成诊断尚未运行。下一步应在可用 GPU 上对冻结 Validation 配对样本补该诊断，不使用正在运行的 Test 调参。

### 2026-09-17 后续交接：单卡 Qwen 逐层诊断入口（GPU 待运行）

用户要求继续核验，并在宿主 A6000 持续报告 100% 利用率时选择单卡平台脚本。新增 `scripts/diagnose_qwen_prefix.py` → `sft/prefix_diagnostic.py`，平台入口为仓库根 `launchers/run_diagnose_qg_a0_a4_qwen_prefix_1gpu.sh`，完整复制四模型评测入口的环境/挂载/认证，任务资源改为单卡串行 A0/A4。冻结 A0-GID checkpoint-8955 与 A4-GID checkpoint-8949；原模型和评测代码不改。

输入为两版原五组 Validation 固定 10k 缓存，先核对十份完整文件 SHA256 和逐行非标识输入，再按固定业务键哈希每组选 100 条，共 500 条/模型；选样独立于模型结果，不读取 Test。默认先五组各 2 条（10 条/模型）GPU smoke，两版都通过后才运行每组 100 条。目标是在相同上下文上比较正确前缀下 S1/S2/S3 的预测能力，以及原 Beam=10 双解码最终候选中的目标前缀覆盖；不把小样本作为正式全量排名，不用 Test 调参。

正确前缀评分保持实际 Query/GID/历史 prompt，目标包含 wrapper、GID6、SID3、可选 D、闭合和 EOS，使用 left-padding position_ids 和最后最多 13 个位置的 logits。记录全词表/合法后继排名与 NLL、合法概率质量和候选数，显式区分 teacher-forced 与自由生成。自由生成直接调用原 evaluate_chunk，经只读 RecordingModel 捕获原始候选及分数；约束模式直接复用 FinalIdPrefixIndex / ConstrainedGenerationModel。Top-1/10 累计前缀为 GID6 → GID6+S1 → GID6+S1/S2 → GID6+SID3 → 完整 ID；首次丢失分段由相邻前缀覆盖差值计算，Top-10 表示十个最终候选中已无正确前缀，不是逐 token greedy。

实际完成的核验：5 项合成测试通过，包括真实随机小型 Qwen 的批量左 padding/变长目标评分与逐条无 padding next-token forward 对照；还覆盖业务键选样、mask 概率与排名、可选 D/EOS 位置、原 Beam 排序/返回对象保持不变。CLI 帮助、dry-run、compileall、shell 语法、可执行权限、短 TMPDIR、平台 bootstrap 一致性和 git diff --check 通过。真实 CPU --prepare-only 已完成两版所有样本的 Token/目录检查、checkpoint 完整文件哈希与原评测记录逐项比较；sample2 为各 10 条，sample100 为各 500 条，二者均 prepared，业务键逐行一致。

首轮运行期间修正了启动器 dry-run 的样本数量参数转发；两个 CPU 准备子阶段均退出 0，但活动父 Shell 随后读到残缺命令，退出 127。固定脚本后重新执行完整 --prepare-only，两个阶段复用均退出 0、父脚本退出 0；失败与成功日志都保留，不覆盖。最终成功日志标签为 `20260917_191949_2`。本项没有真实 checkpoint 的 GPU 逐层指标，不登记新正式实验编号。

复现命令（仓库根目录）：

```bash
bash launchers/run_diagnose_qg_a0_a4_qwen_prefix_1gpu.sh --dry-run
bash launchers/run_diagnose_qg_a0_a4_qwen_prefix_1gpu.sh
```

产物根目录 `outputs/eval/a0_a4_qwen_prefix_diagnostic_v1/`（相对 qg_prqk），分 `sample2/` 与 `sample100/`：plan、配对缓存、每模型 progress/result 和合并 summary。sample2/sample100 plan SHA256 分别为 `f1e642c1a9fc651a8daab630ac936de3c658ca7dccd432b4f5c61347e7f31aaf` / `f4b0eed5a7ea7ea278c18380813df5ec5415abd5cc7c450ee2fd76e9b2e7bf7f`。日志/PID/退出码为 `outputs/run_control/a0_a4_qwen_prefix_diagnostic_v1/`。环境为 poi-gr、CPU 预检、项目内 TMPDIR=`outputs/tmp/qpd1`（相对仓库根），未提交 GPU 任务，未 commit/push。下一步由用户在空闲单卡平台执行，随后依据配对原始生成记录判断静态 S1 差距是否传递到 Qwen。

## EXP-20260917-03：active 四模型全量 Test 评测

- **目标与假设**：对比 GNPR、GenPOI、QG-HRQ(GID/NoGID) 在同一完整 Test 的检索能力，并核验两版 QG 全目录约束相对已完成无约束评测的收益。
- **日期与状态**：2026-09-17 启动，9 月 18 日完成并验收；四个正式任务均 completed、各 606,682 条。准备、四个 smoke2、smoke 汇总、四个 full、full 汇总及父启动器共 12 个退出码全部为 0。
- **数据与配置**：2026-07-14 全量 Test，active 716,245 POI；四模型业务键、目标、CURRENT 和历史长度由准备阶段全量配对。配置 `configs/sft/active_four_models_full_test_20260714_v1.yaml`（相对仓库根）；epoch 3、Beam/返回数 10、cutoff 1024、batch 32、chunk 500。GNPR 无约束且非法候选保留原排名；GenPOI SSP γ=2 + TCG；QG 两版采用全目录合法路径约束，不做地理剪枝。
- **代码与环境**：基于 `54802e6` 的未提交工作树，实际源码指纹见 plan.source_hashes；平台四卡入口每卡一个模型任务，6000D 配置、poi-gr Python 环境。CPU 验收未启动新 GPU 推理。
- **命令**：仓库根目录执行 `bash launchers/run_evaluate_active_four_models_full_test_4x6000d.sh`；入口依次完成 prepare → smoke2 → full → summarize。本次验收只调用已有 `load_config/verify_plan/summarize`，核对配置、来源指纹及原始结果；没有重跑全量数据扫描或推理。
- **Checkpoint**：GNPR/GenPOI/QG-GID/QG-NoGID 分别为 7377/8868/8949/7404；完整模型路径及冻结 SHA256 见 plan.weights。统一 plan 签名 `9f639dc355ce3f1395553830dcde76bc3595414f5ec835ed10119f9ef15990d1`，业务键 SHA256 `c2e71487ab3b539d3468667a9e1bdcd8fdc53066d71de754aab1d30c532177e3`。
- **验收**：重新汇总与 summary.json 完全一致；四个原始 result 与 suite_result 内结果一致；各 progress.next_line 为 606,682、生成候选为 6,066,820；逐项用累积命中数/NDCG 和重算指标，并对 GNPR/QG 排名直方图独立复算。

| 方法 | 解码 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | 合法率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GNPR | 无约束 | 49.0443% | 71.9766% | 78.0274% | 82.3832% | 62.6495% | 65.1553% | 66.5933% | 71.0600% |
| GenPOI | SSP γ=2 + TCG | 50.7726% | 74.9351% | 81.5248% | 86.7847% | 65.0993% | 67.8268% | 69.5561% | 100.0000% |
| QG-HRQ(GID) | 全目录合法路径约束 | 51.1482% | 75.4580% | 82.0277% | 87.2035% | 65.5656% | 68.2830% | 69.9869% | 100.0000% |
| QG-HRQ(NoGID) | 全目录合法路径约束 | 50.5710% | 74.5747% | 81.0940% | 86.3446% | 64.8113% | 67.5084% | 69.2324% | 100.0000% |

与已有全量 Test 无约束结果比较：两版源 Test 文件 SHA256 和 checkpoint 权重 SHA256 均与本次一致。

| 方法 | 无约束 HR@1 → 约束 | 无约束 HR@10 → 约束 | 无约束 NDCG@10 → 约束 |
|---|---:|---:|---:|
| QG-HRQ(GID) | 50.9631% → 51.1482%（+0.1851pp） | 85.5120% → 87.2035%（+1.6915pp） | 69.1641% → 69.9869%（+0.8228pp） |
| QG-HRQ(NoGID) | 50.1919% → 50.5710%（+0.3791pp） | 85.1705% → 86.3446%（+1.1741pp） | 68.5076% → 69.2324%（+0.7248pp） |

- **结论**：QG-GID > GenPOI > QG-NoGID > GNPR。GID 相对 GenPOI HR@1/HR@10/NDCG@10 高 0.3756/0.4188/0.4307pp；NoGID 则低 0.2016/0.4401/0.3238pp。GID 相对 NoGID 高 0.5772/0.8589/0.7545pp。QG 约束提高候选覆盖，合法率均变为 100%；合法率是候选级指标，不是请求命中率。
- **解释边界与下一步**：本表是指定解码协议下的完整系统比较，尚无显著性检验；没有 TIGER 或 A0 Test 单元，不能据此认定超过三篇论文或证明 QG 图/Query 创新组件的独立收益。A0/A4 的原因继续依赖已交付的 Qwen 配对逐层诊断，不用本次完整系统结果替代消融。
- **产物与日志**：根目录 `outputs/eval/active_four_models_full_test_20260714_v1/` 下 summary.json/csv、plan.json、各模型 full/suite_result.json 及原始 result/progress；日志 `outputs/run_control/active_four_models_full_test_20260714_v1/`，运行标签 `20260917_171327_854`。summary.json SHA256 `eb1d41b693a8113fa88bc4acd48fbf01ec256a32ecc301a4bd9cad531ab9a055`。旧 QG 无约束结果为 `qg_prqk/outputs/eval/sft_epoch3_full_test_20260714_v1/`。

## EXP-20260918-01：A0/A4 Qwen 配对逐层诊断验收

### 目标、协议与完成状态

承接 EXP-20260917-02，检验静态 S1 content-only Query 可预测性下降是否传递到训练后的 Qwen，并区分正确前缀条件下的 token 排序、自由生成候选覆盖和后缀损失。冻结 A0-GID checkpoint-8955 与 A4-GID checkpoint-8949；五个原 Validation 集各按业务键哈希选取 100 条，两版同序共 500 条，另有每组 2 条 smoke。样本选择独立于模型结果；不包含 Test，五组等权合并只是诊断混合样本，不能当作自然流量分布或泛化宏平均（泛化单独排除 fixed10k）。

2026-09-18 单卡平台运行标签 `20260918_085501_800`，两个准备、四个模型任务（smoke/full）、两次汇总及父启动器共 9 个退出码全部为 0；A0/A4 正式日志均到 500/500。9 月 17 日准备交接中已有的退出 127 单独保留，不混入本次退出码。

### 实现、来源与核验

- 工作树基于 `54802e6`，未提交；调用既有 `src/qg_prqk/sft/prefix_diagnostic.py`，模型、数据、原评测 plan 和源码指纹绑定到诊断 plan。平台入口为仓库根目录 `bash launchers/run_diagnose_qg_a0_a4_qwen_prefix_1gpu.sh`，poi-gr 环境、显式单 GPU 串行 A0/A4。当前 CPU 验收没有新推理或训练。
- 数据流：配对缓存 → gold-prefix Teacher-Forcing（全词表及真实目录合法后继排名/NLL）→ 原 Beam=10 evaluator 的无约束/全目录约束 → 原始 token/score → 逐层联合前缀与完整 ID 汇总。这里的正确前缀包含真实 GID6，S2/S3 还包含真实先前 SID token。
- 复核原计划/源码哈希、模型文件预检后的大小/mtime、两份缓存完整 SHA256；smoke/full 的 result/progress 行及业务键一致。对全部记录从原始 beam tokens 重算联合前缀，核对 10 候选数量与原分数排序，重新 aggregate 与 summary 逐值一致；teacher 的全词表/合法排名、候选范围及 NLL 大小关系均通过。
- 正式结果 SHA256：A0 `bb5bf7d27fb4c01a689fb63ded4caaff2cd8baa7b38b596987771b2864bace36`；A4 `6a7c8f1a4c5c6a319fae6d6e67a3f529c8b641b28b66a40f29f1ff0276b98a8f`；summary `7c98c8bf96d15b1660820ee76d963abd39fdcf533d119237bbc81ebbd8b8890b`；计划签名 `2de45c791cf5eb21bf5650cbaaa3fff4f178db652627538e81e118bca8703648`。
- CPU 轻量复验：`PYTHONPATH=qg_prqk/src CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /ofs/map_search/hudan/envs/poi-gr/bin/python -m unittest discover -s qg_prqk/tests -p test_qwen_prefix_diagnostic.py -v`，5 项全部通过，包含微型 Qwen 批量左 padding 与逐条无 padding 的 next-token 对齐；入口与模块 compileall、git diff --check 通过。

### 正确前缀条件下的 Qwen 预测

| 正确前缀下的指标（500 条） | A0 | A4 | A4−A0 |
|---|---:|---:|---:|
| S1 全词表 Top-1 | 61.40% | 60.00% | -1.40pp |
| S1 合法后继 Top-1 | 63.00% | 61.20% | -1.80pp |
| S2 全词表 Top-1 | 74.80% | 76.60% | +1.80pp |
| S2 合法后继 Top-1 | 83.20% | 85.80% | +2.60pp |
| S3 全词表 Top-1 | 74.00% | 73.80% | -0.20pp |
| S3 合法后继 Top-1 | 92.20% | 91.40% | -0.80pp |
| S1/S2/S3 三层全词表 Top-1 同时正确 | 43.40% | 43.40% | 0.00pp |

- S1 全词表 NLL 为 1.8241→1.9481（低为好），S2 为 1.3669→1.3243，S3 为 1.5050→1.5347。S1 有负向迹象，S2 局部正向，整段 SID 同时正确没有净增益；不能把三个单层百分比直接相加或相乘。
- S3 合法 Top-1 的 92.2%/91.4% 含 62.0%/61.0% 的单一合法后继。只看两版都多分支的同 166 条，合法 Top-1 为 79.52%/78.31%；在四类泛化都多分支的同 120 条上，两者均 74.17%。高 S3 条件准确率并不证明局部 S3 创新已有效。
- A0/A4 需要 Dedup 的样本分别 81/61 条，不能直接把不同条件人群的 Dedup 准确率当配对收益。

### 自由生成的联合前缀覆盖

| 最终约束 Top-10 中的正确联合前缀覆盖（500 条） | A0 | A4 |
|---|---:|---:|
| GID6 | 75.80% | 76.80% |
| GID6+S1 | 64.80% | 65.40% |
| GID6+S1+S2 | 60.60% | 59.20% |
| GID6+SID3 | 59.00% | 57.00% |
| 完整 Final ID | 57.40% | 55.80% |

- A4 221 个完整 Top-10 miss 中，116 条没有正确 GID6 前缀、57 条有 GID6 但没有联合 S1，合计 173/221=78.28%；S2、S3、最终后缀新增缺失分别 31、11、6 条。A0 对应为 121、55、21、8、8 条，共 213 个 miss。尾部 Dedup/闭合不是这批样本的主要损失位置。
- 这些计数是**最终返回十条完整序列的前缀覆盖分解**，不是解码每一步的活跃 beam 轨迹；因此不能认定目标“在该步被剪枝”，更不能直接把全部 GID miss 归因于地理模块。
- 小样本约束完整 HR@1 为 28.0%→28.4%，HR@10 为 57.4%→55.8%；无约束分别 28.0%→28.2%、52.2%→51.6%。这是机制诊断结果，不替换原 10k/40k 正式指标。

| 诊断切片（各 100 条） | S1 全词表 Top-1 A0→A4 | 约束 HR@10 A0→A4 |
|---|---:|---:|
| 固定集 | 86% → 85% | 89% → 90% |
| 已见 Query / 未见配对 | 62% → 54% | 63% → 55% |
| 新 Query / 已见目标 | 68% → 66% | 71% → 70% |
| 长尾目标 | 55% → 60% | 43% → 45% |
| 冷目标 | 36% → 35% | 21% → 19% |

### 配对不确定性与完整评测交叉检查

在 500 条同业务键结果上按五组各自重采样、保持每组 100 条，做 10,000 次配对 bootstrap（NumPy RNG seed=20260918；探索性、未作多重比较校正）：S1 全词表 Top-1 差值 −1.4pp，95% 区间约 [−4.8,+2.0]pp；约束 HR@10 差值 −1.6pp，区间约 [−4.6,+1.4]pp。前者 A4-only 正确 32 条、A0-only 正确 39 条；后者 A4-only 命中 28 条、A0-only 命中 36 条。该规模不足以确认 1—2pp 级的总体差异。

原完整五集结果重新只读核对：约束固定 10k A0→A4 的 HR@1/HR@10/NDCG@10 为 50.65→50.62%、87.06→87.34%、69.7518→69.8333%；四类泛化各 10k 的宏平均为 24.1400→24.5575%、53.7250→53.9075%、38.1769→38.5413%。无约束固定三项变化 −0.10/−0.15/−0.1533pp，泛化宏平均变化 +0.3125/−0.0350/+0.1575pp。

特别是已见 Query/未见配对：本次 100 条约束 HR@10 为 63→55%，原 10k 却为 60.99→61.48%。不能把小切片的 −8pp 外推为该组正式结果。完整评测仍支持新增组件的边际收益小，不支持 A4 大幅退化。

### 判断与下一步

1. 原静态 probe 是冻结 Query 向量对码本做内容相似度排序；这里是训练后的 Qwen 使用 Query 文本、用户 GID、历史、真实目标前缀进行条件预测，且数据/加权口径也不同。静态 Top-1 的 29.96→21.02% 不能等同于 Qwen 掉 8.94pp；本次 Qwen 只有 −1.4pp 的点估计。
2. 已核验范围内未发现数据配对或本次诊断汇总错误。证据更接近“图目标重塑了编码，但局部结构收益没有稳定变成整条可生成路径收益”；A0 共享的 PRQ/GID/SFT 框架已很强。各组件的独立因果贡献、不同 seed 和 S1/S2/S3 各自作用仍未隔离。
3. 下一最小建议是保持 checkpoint/解码不变，将五组确定性诊断扩大到每组 1,000 条，再检验 S1 负向、S2 正向和整段正确率是否稳定；暂不据 500 条结果选图权重或重训，也不根据 Test 调参。扩样推理尚未启动。

产物：`outputs/eval/a0_a4_qwen_prefix_diagnostic_v1/sample{2,100}/`；日志：`outputs/run_control/a0_a4_qwen_prefix_diagnostic_v1/`，均相对 qg_prqk。完整效果参照 `outputs/eval/a0_gid_epoch3_dual_decode_v1/results/summary.json` 与 `outputs/eval/sft_epoch3_constrained_fixed10k_generalization_v1/summary.json`。本步仅更新已有正式文档，无新报告、图表、模型或代码修改，未 commit/push。

## 2026-09-18 后续交接：A0-GID 全量 Test 双卡入口

用户要求使用原 A0-GID 模型补齐与四模型相同的全量 Test，无约束和约束都跑。此处仅记录代码交接，不是已完成的正式实验，不新增 EXP 编号或指标。工作树未提交，旧冻结评测源码及结果不改。

- 输入固定为 active 716,245 库的 A0-GID epoch-3 `checkpoint-8955`，模型权重 SHA256 `282ac677f0962f65e5f691dbeff10d4d5f3459bbeb0924f36287db30283d716a`；Test 为 2026-07-14 的 606,682 条，与原 QG-HRQ(GID) Test 同序，源 JSONL SHA256 `8aa7616fad72d48b0103481aae135e129b0f3d7856fef4b2f13456191c6a54d9`。配置为 `configs/sft/evaluation_a0_gid_epoch3_full_test_dual_decode_v1.yaml`（相对 qg_prqk），并绑定原 A0 和全量 Test 配置哈希。
- 原 A0 没有 Test JSONL。新 `sft/a0_full_test.py` 将冻结 A4-GID Test 的目标/历史 ID 回到 POI 行，再换成 A0 ID；保留业务键、CURRENT、历史顺序和 Test 划分。平台准备核验全量源哈希、业务键唯一性和每条映射，并计算映射后记录哈希；仅建立只读索引，不额外物化全量 JSONL。前 100 条验证格式化 Token，剩余逐 chunk 沿用原编码器的检查。
- poi-gr 环境、2×RTX PRO 6000D，两卡分别无约束/完整 active 目录约束；复用 `full_test_evaluation.evaluate_test_chunk` 和现有 Trie，Beam=10、cutoff=1024、max_new_tokens=13、batch=32、chunk=500，不增加地理剪枝。默认先两模式各 2 条 smoke，通过后各跑全量；OOM 降批，来源一致才复用断点，任一 worker 失败即终止本套件另一 worker。
- 在仓库根目录运行 `bash qg_prqk/launchers/run_evaluate_prqk_a0_gid_epoch3_full_test_dual_decode_2x6000d.sh`。按用户指定放在方法目录的 launchers；平台初始化复用既有双卡入口。支持 `--dry-run`、`--prepare-only` 和 `--smoke-limit N`，详见 `qg_prqk/README.md`。
- 本地验证：`bash -n`、入口 `--help`/`--dry-run`、compileall、9 项 `test_a0_full_test.py` 合成测试通过（含映射拒错、OOM、恢复、解码隔离、完整汇总和失败收尾）；CPU 只读核验真实 716,245 POI 两份目录及前 8 条 Test 的目标/历史映射、目标 Token 全部通过。未加载模型进行推理，全量模型哈希与 606,682 条映射扫描将在平台准备阶段完成。
- 正式输出预定为 `outputs/eval/a0_gid_epoch3_full_test_20260714_dual_decode_v1/`（相对 qg_prqk），两种模式结果在 `{unconstrained,constrained}/results/`，统一汇总在 `results/summary.json` / `summary.csv`；日志在 `outputs/run_control/eval_a0_gid_full_test_dual_decode_v1/`。待两模式各完成 606,682 条并验收后，再登记正式实验及与 A4 的配对比较；不根据 Test 调参。本步未 commit/push。

## EXP-20260918-02：A0-GID 全量 Test 双解码验收

### 目标、配置与完成状态

补齐 POI-only PRQ-KMeans（A0）的全量 Test，检验 A4 新增 Query、类别与局部地理等组件在相同 GID、生成模型与解码协议下的整体增量。2026-09-18 启动，2026-09-19 验收；运行标签 `full_20260918_141018_839`。两个 full worker 和父启动器退出码均为 0，日志出现 `A0_FULL_TEST_COMPLETED`；两模式均完成 606,682/606,682 条，smoke 不混入正式指标。

- 工作树基于 `54802e6674e283722ecee00fb862530df30cef9a`，未提交；本次验收只读结果并更新已有实验文档，没有新训练或 GPU 推理。
- 平台入口：在仓库根目录运行 `bash qg_prqk/launchers/run_evaluate_prqk_a0_gid_epoch3_full_test_dual_decode_2x6000d.sh`。poi-gr 环境，2×RTX PRO 6000D 分别运行两种解码；配置为 `qg_prqk/configs/sft/evaluation_a0_gid_epoch3_full_test_dual_decode_v1.yaml`。
- active 目录 716,245 POI，A0 epoch-3 `checkpoint-8955`，权重 SHA256 `282ac677f0962f65e5f691dbeff10d4d5f3459bbeb0924f36287db30283d716a`。三层 PRQ SID 前置 GID6 并使用条件 Dedup；不是去掉 GID 的裸 PRQ。
- Test 为 2026-07-14 的全部 606,682 条。冻结 A4-GID 源 Test SHA256 `8aa7616fad72d48b0103481aae135e129b0f3d7856fef4b2f13456191c6a54d9`；经 POI 行映射替换目标和历史 ID，保留当前请求及业务键。准备回执记录 2,869,373 个历史事件，映射记录 SHA256 `5fa3acfc3b2bc59d8cd99b2000f2f30f5e90efb31515dfa4bd3ba34f21b7f39f`，未物化新全量 JSONL。
- Beam=10、cutoff=1024、max_new_tokens=13、batch=32、chunk=500。约束为完整 active 目录合法路径 Trie，不额外使用地理剪枝；无约束非法 ID 保持原排名并计 miss。累计推理计时无约束 15.94 小时、约束 16.87 小时，两卡并行，不能相加当作墙钟耗时。

### 产物核验与指标

2026-09-19 CPU 核验：配置哈希、源码指纹、计划及结果签名一致；两份 progress 的 next_line/sample_count 均为 606,682，候选总数均为 6,066,820。调用原 `finalize_metrics` 从累计统计重建结果，与 result 完全一致；独立从目标排名直方图复算 HR@1/3/5/10、NDCG@1/3/5/10 和 MRR@10 均通过，summary 与 result 逐项一致。此处未重新扫描全部模型权重或业务 JSONL，使用平台准备回执及来源指纹核对。

| A0 解码 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | MRR@10 | 候选合法率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无约束 | 50.9641% | 74.7317% | 81.0514% | 85.6261% | 69.2049% | 63.8270% | 65.9327% |
| 全目录约束 | 51.1357% | 75.3390% | 81.9713% | 87.2192% | 69.9686% | 64.3408% | 100.0000% |
| 约束−无约束（pp） | +0.1716 | +0.6072 | +0.9199 | +1.5931 | +0.7637 | +0.5138 | +34.0673 |

约束使 Top-1 净命中数增加 1,041，Top-10 净命中数增加 9,665。候选合法率不是请求命中率，且约束结果不是简单删除无约束结果里的非法候选。

### 与 A4 及 baseline 比较

A0 与 EXP-20260917-03 四模型汇总的业务键 SHA256 一致，均为 `c2e71487ab3b539d3468667a9e1bdcd8fdc53066d71de754aab1d30c532177e3`。旧 A4 无约束汇总的业务键哈希使用 tab 拼接，A0/四模型使用紧凑 JSON 行，因此哈希不同；已核对旧 A4 plan 引用完全相同源 Test 文件、SHA256、行数和 checkpoint-8949，不把序列化差异误判为换了测试集。

| 解码 | 方法 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| 无约束 | A0-GID | 50.9641% | 74.7317% | 81.0514% | 85.6261% | 69.2049% |
| 无约束 | A4-GID | 50.9631% | 74.7878% | 81.0045% | 85.5120% | 69.1641% |
| 全目录约束 | A0-GID | 51.1357% | 75.3390% | 81.9713% | 87.2192% | 69.9686% |
| 全目录约束 | A4-GID | 51.1482% | 75.4580% | 82.0277% | 87.2035% | 69.9869% |

- A4−A0 无约束 HR@1/HR@10/NDCG@10 为 −0.0010/−0.1141/−0.0408pp，Top-1/Top-10 净命中数分别少 6/692。
- A4−A0 约束 HR@1/HR@10/NDCG@10 为 +0.0125/−0.0157/+0.0183pp，Top-1 净多 76 条、Top-10 净少 95 条；HR@3/5 分别 +0.1190/+0.0564pp。这些是总命中数之差，不是逐条胜负分解。
- GenPOI SSP+TCG 的 HR@1/HR@10/NDCG@10 为 50.7726%/86.7847%/69.5561%；A0 全目录约束对应高 +0.3631/+0.4345/+0.4124pp。GNPR 无约束为 49.0443%/82.3832%/66.5933%。跨方法比较采用各自冻结解码协议，属于系统比较；不能把差值单独归因于量化器，也没有本次 TIGER 全量 Test 单元。

### 结论、边界与产物

全量 Test 点估计支持 A0/A4 整体接近，A4 新增组件组合尚未产生清晰的最终指标增益；不能用 A4 超过 GenPOI 的系统结果代替组件有效性的证据，因为 A0 约束同样超过 GenPOI。无逐条配对显著性检验或多 seed 复跑，不能宣称统计等价，也不能认定所有组件无效。该结果不定位 Query、类别或地理中任何单个组件的因果贡献；后续方法选择和参数调整仍使用 Validation，Test 仅留作最终确认。

正式产物为 `qg_prqk/outputs/eval/a0_gid_epoch3_full_test_20260714_dual_decode_v1/`：`plan.json`、`{unconstrained,constrained}/results/{result,progress}.json` 和 `results/summary.json` / `summary.csv`。summary SHA256 `86a870c4a583875972f600c000e002aeef0faed297d46541fd198ef199d5b769`；plan SHA256 `0bc5a2d4beb7fdd93fb779492291a10dc7affbc7d416c3116bd5fc7240bbb7e0`；计划签名 `f37acd87a2bfcd8a76d05bd520ad93c79698013e89a256c6e50af5511c6d24be`。日志与退出码位于 `qg_prqk/outputs/run_control/eval_a0_gid_full_test_dual_decode_v1/`。未 commit/push。
