# QG-PRQK v2.1-CAT Codex 启动与续跑命令

> 版本说明（2026-09-19）：本文保存 v2.1-CAT 的已实现命令与历史恢复示例；其中阶段状态不是当前任务指令。v3.0 已完成[方案登记](methods/QG_PRQK_METHOD_SPEC.md)，尚无训练/评测命令；实际下一步见[实施状态](experiments/QG_PRQK_IMPLEMENTATION_STATUS.md)。旧方法定义见 [v2.1-CAT 归档](methods/QG_PRQK_V2_1_CAT_SPEC.md)。

> 目标仓库：`/ofs/map_search/hudan/poi_genret`  
> 工作方式：canonical 状态驱动，一次只执行一个最小阶段。

## 1. Canonical 文件

```text
qg_prqk/docs/methods/QG_PRQK_METHOD_SPEC.md
qg_prqk/docs/methods/QG_PRQK_V2_1_CAT_SPEC.md
qg_prqk/docs/methods/QG_PRQK_CODEX_EXECUTION_PLAN.md
qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md
qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS_TEMPLATE.md
qg_prqk/prompts/QG_PRQK_CODEX_BOOTSTRAP.md
qg_prqk/prompts/QG_PRQK_CODEX_NEXT_PHASE.md
```

QG 文档不复制到仓库根 `docs/`。源码、脚本、配置、测试和方法级入口也必须在 `qg_prqk/`；所有 QG 新产物只能写入 `qg_prqk/outputs/`。

当前源码统一通过 `qg_prqk/scripts/qg_prqk.py <command>` 进入。本文中直接调用当前源码的命令已切换为语义化子命令；只有指向 `outputs/run_control/**/source_snapshot/` 的历史命令继续使用冻结快照入口。新运行先用 `qg_prqk/scripts/qg_prqk.py --help` 查看命令，不再新增一项任务一个 Python wrapper。

## 2. 开始前检查

```bash
cd /ofs/map_search/hudan/poi_genret
git status --short
sed -n '1,80p' qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md
```

重点核对：

- `CURRENT_PHASE` 和唯一 `NEXT_ACTION`；
- 区分 `DESIGN_METHOD_VERSION: v3.0` 与 `IMPLEMENTED_METHOD_VERSION: v2.1-CAT`；复核旧任务时使用其冻结配置/源码，不能把旧命令当作 v3 实现；
- `P2_FULL_QUERY_STATS: COMPLETED`；
- `CATEGORY_AUDIT` 与 active P2.5 的真实状态；
- `P3_EXACT_ADAPTER: FULL_IN_PROGRESS | FULL_COMPLETED`；50k Gate 不得误写为 full；
- 用户工作树是否有未解释的修改。

## 3. 新会话启动

```bash
cd /ofs/map_search/hudan/poi_genret

codex \
  -C /ofs/map_search/hudan/poi_genret \
  -s workspace-write \
  -a on-request \
  "$(cat qg_prqk/prompts/QG_PRQK_CODEX_BOOTSTRAP.md)"
```

不要使用 `--yolo` 或 `--dangerously-bypass-approvals-and-sandbox`。

## 4. 恢复已有会话

```bash
cd /ofs/map_search/hudan/poi_genret
codex resume --last
```

进入 TUI 后发送：

```text
读取 qg_prqk/prompts/QG_PRQK_CODEX_NEXT_PHASE.md，并严格只执行 qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md 中 NEXT_ACTION 指定的一个阶段；完成后更新状态文档并停止。
```

无法恢复时，可直接用 next-phase prompt 开新会话：

```bash
cd /ofs/map_search/hudan/poi_genret

codex \
  -C /ofs/map_search/hudan/poi_genret \
  -s workspace-write \
  -a on-request \
  "$(cat qg_prqk/prompts/QG_PRQK_CODEX_NEXT_PHASE.md)"
```

## 5. 历史阶段与复核命令

P3A-FULL、P4–P8 canonical 链路，以及 NoGID `(S1,S2)` parent S3 受控变体和三方公平比较均已完成并独立验收。NoGID 输出严格三位 `[S1,S2,S3]`，但全量结构和 Query Prefix Probe 均略差于原 GID-parent A4；历史 `HOLD_FOR_NOGID_S3_REVIEW` 已由用户履行，并批准两个分支进入 SFT。两套 Final ID、全量 history10 Messages、共同扩词表与两套 Train/Valid Tokenized Cache 均已完成，两条四卡 launcher 的 `--dry-run` 已通过；当前停在 `HOLD_FOR_QG_PRQK_DUAL_SFT_LAUNCH_REVIEW`，不得自动启动 SFT、读取 Test、运行外部基线或其他下游阶段。

P8 full 只读复核：

```bash
PYTHONPATH=qg_prqk/outputs/run_control/p8_full_attempt02/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p8_full_attempt02/source_snapshot/scripts/evaluate_and_publish_p8_sid.py \
  --config qg_prqk/configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml \
  --gate full --validate-only
```

预期 `status=p8_validated`、`poi_rows=716245`、`query_rows=342879`、`evaluation_outcome=REVIEW_REQUIRED`、`next_status=HOLD_FOR_REVIEW`。成功 manifest SHA256=`c66ea6c5fac499ed7c377b9f2ae7d73c3fde333c7842f0d79e9c84b141c82ab3`；输出已有 `_SUCCESS`，不得执行不带 `--validate-only` 的同路径命令。

NoGID P7/P8 full 只读复核：

```bash
PYTHONPATH=qg_prqk/outputs/run_control/p7_nogid_full_attempt02/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p7_nogid_full_attempt02/source_snapshot/build_poi_query_category_nogid_prqk.py \
  --config qg_prqk/configs/qg_prqk_p7_nogid_s1s2_parent_hard60_topk5_direct_full_v1.yaml \
  --gate full --validate-only

PYTHONPATH=qg_prqk/outputs/run_control/p8_nogid_full_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p8_nogid_full_attempt01/source_snapshot/evaluate_p8_nogid_sid.py \
  --config qg_prqk/configs/qg_prqk_p8_nogid_s1s2_parent_comparison_full_v1.yaml \
  --validate-only
```

预期分别输出 `status=p7_nogid_validated` / `status=p8_nogid_validated`，两者 `next_status=HOLD_FOR_NOGID_S3_REVIEW`。P7/P8 manifest SHA256 分别为 `a6ce4ba8c20f79073e5a41fa2e516b26b2bdeb7c2d7c56e260f4a94f46bf1d6c` / `b22ab413558b1e147542135fd554fdf0b7a71082bdd1842fc787aa2b16ed4ed8`；命令只读，不覆盖产物。

P7 full 只读复核：

```bash
PYTHONPATH=qg_prqk/outputs/run_control/p7_full_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p7a \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p7_full_attempt01/source_snapshot/scripts/build_poi_query_category_geo_prqk.py \
  --config qg_prqk/configs/qg_prqk_p7_active_512x3_hard60_topk5_direct_full_v1.yaml \
  --gate full --validate-only
```

预期 `status=p7_validated`、`poi_rows=716245`、`d3_query_rows=291590`、`hard_edge_rows=746034`、`next_status=HOLD_FOR_P7_FULL_REVIEW`。成功 manifest SHA256=`dd395ab0d11321106455ad05b18503af9b7d3dda7cb7fd85e531655cece2133c`；输出已有 `_SUCCESS`，不得执行不带 `--validate-only` 的同路径命令。

P6/P7 配置和真实输入 header 只读预检：

```bash
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py inspect-relational-inputs \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml
```

预期 `status=p6_input_headers_validated`、`hard_max_iter=60`、POI/Query/S1+S2 边为 `716245/342879/912980`，并且 `output_written=false`。

P6 sample 已完成，复核命令：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py build-relational-codebook \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml \
  --gate sample --validate-only

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py evaluate-relational-codebook \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml \
  --gate sample --validate-only
```

上面的 header 预检针对当前 direct-full-v2 合同。历史 v1 sample 已有 `_SUCCESS`，不得重复运行无 `--validate-only` 的命令；历史 medium 合同保留但不再执行。

P6 full 已完成；以下命令当前都只用于只读复核，不能移除 `--validate-only` 重复构建或评估：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=qg_prqk/outputs/run_control/p6_full_hard60_topk5_direct_v2_attempt05/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p6f \
/ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p6_full_hard60_topk5_direct_v2_attempt05/source_snapshot/scripts/build_poi_query_category_prqk.py \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml \
  --gate full --validate-only

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py build-relational-codebook \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml \
  --gate full --validate-only

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=qg_prqk/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p6f \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py evaluate-relational-codebook \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml \
  --gate full --validate-only
```

full 固定使用 716,245 POI、342,879 Query 和 912,980 条 S1/S2 边，输出为 `poi_query_category_prqk_s1_s2_hard60_topk5_v1/full_342879q_716245p/`。配置显式绑定已完成 sample manifest 并记录跳过 medium；除规模与 Gate 顺序外不改算法。

P6 sample 六分支归因诊断已完成，只读复核命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=qg_prqk/outputs/run_control/p6_sample_attribution_v1_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p6d \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p6_sample_attribution_v1_attempt01/source_snapshot/scripts/diagnose_p6_sample_attribution.py \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_v1.yaml \
  --validate-only
```

预期 `status=p6_sample_attribution_validated`、manifest SHA256=`cd8e9e61274fd9632b48778fcac5827dfa3dd5773a4e33fa22726772743ac441`、`next_status=HOLD_FOR_P6_SAMPLE_ATTRIBUTION_REVIEW`。诊断入口默认禁止覆盖，不能再次执行无 `--validate-only` 的命令。

P6 full S2-only 六分支诊断已完成，只读复核命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=qg_prqk/outputs/run_control/p6_full_s2_attribution_v1_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p6s2diag \
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p6_full_s2_attribution_v1_attempt01/source_snapshot/scripts/diagnose_p6_full_s2.py \
  --config qg_prqk/configs/qg_prqk_p6_p7_active_512x3_hard60_topk5_direct_full_v2.yaml \
  --validate-only
```

预期 `status=p6_full_s2_diagnostic_validated`、manifest SHA256=`7e17ce4369046d8c2c9453ced59d7b47a50afbf23bd010cb924cae83c32d55d0`、comparison SHA256=`0900f9d17569bbce8128e603bf9c582393919e6bed7b8d3daf8521bdb38dbe4d`。该入口冻结正式 S1 endpoint，只重算 S2 指标；默认 overwrite=false，不能移除 `--validate-only`。

新协议 10k 历史构建命令（已完成；只读复核应追加 `--validate-only`）：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=qg_prqk/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/scripts/qg_prqk.py build-base-codebook \
  --config qg_prqk/configs/qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml \
  --gate sample
```

正式运行必须使用 `qg_prqk/outputs/run_control/p5_cat_active_512_hard60_topk5/sample_010000_attempt01/source_snapshot/` 内冻结源码，命令和源码哈希分别见该目录的 `command.txt` 与 `source_snapshot/MANIFEST.sha256`。上述短命令用于说明入口；不得用旧 30 轮配置写入新 namespace。

P5 full 已完成；以下命令只读复核全部 716,245 行，不重新拟合、不覆盖产物：

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5f \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p5_cat_active_512_hard60_topk5/full_716245_direct_from_sample_attempt02/source_snapshot/scripts/build_poi_prqk.py \
  --config qg_prqk/configs/qg_prqk_p5_active_512x3_hard60_topk5_v1.yaml \
  --gate full --validate-only
```

预期 `status=validated`、`next_status=HOLD_FOR_P5_HARD60_TOPK5_FULL_REVIEW`；成功复核日志和退出码位于 attempt02 的 `validate_retry01.log` 与 `validate_retry01.exit`。

历史 P4 v1 sample 独立只读校验（使用已冻结快照，不修改原合同）：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p4_cat_active_512/sample_1000_attempt01/source_snapshot/scripts/build_query_graph.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --limit 1000 --validate-only
```

输入为冻结 P2/P2.5、D3 cache、FINAL 来源合同与新 512 namespace 下的 `query_graph/sample_001000/`；不加载 BGE/Adapter，不重读业务 Query。预期 `status=validated`、`query_rows=1000`、`next_status=HOLD_FOR_P4_SAMPLE_REVIEW`。把 `--validate-only` 换成 `--resume` 可复核已完成目录的复用，当前不会重新编码或写入产物。来源或源码不一致会拒绝恢复，后续代码扩展须保留当前版本快照。

旧 sample 的 `next_status` 只表示历史停止点，不回退 canonical 进度。当前源码 schema 为 v3，v1/v2 历史目录必须使用对应的冻结入口复核，不通过删除哈希校验来迁就新源码。

50k medium 构建/恢复命令（已经存在的输出必须追加 `--resume`，禁止覆盖）：

```bash
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p4m \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
HF_HOME=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p4_cat_active_512/medium_50000_attempt01/source_snapshot/scripts/build_query_graph.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate medium --limit 50000 \
  --reuse-sample qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/sample_001000/manifest.json \
  --reuse-sample-sha256 20dfdbd7a49159236694afbd9afee68261b67b37ec52e0326f696464315cb2c6
```

运行前必须检查宿主 GPU 空闲，确认短 TMPDIR 已存在、可写且绝对路径不超过 64 字节。模型 batch 256 不变，缓存块为 8192；只读复用旧 sample 的 Raw/view 和冻结 D3 Raw，不重训 Adapter。输出独立写 `query_graph/medium_050000/`，来源哈希由外部明确锁定，不通过读取当前文件重新生成“期望哈希”。

medium 完成后独立复核（不加载模型、不训练；从已完成 manifest 读取冻结 sample 来源）：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p4_cat_active_512/medium_50000_attempt01/source_snapshot/scripts/build_query_graph.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate medium --limit 50000 --validate-only
```

预期 `status=validated`、`query_rows=50000`、`next_status=HOLD_FOR_P4_MEDIUM_REVIEW`。该 `next_status` 是历史停止点；它不回退已完成的 full。

342,879-query full 历史构建/完成态恢复命令（已存在的输出必须追加 `--resume`，禁止覆盖）：

```bash
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p4f \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p4_cat_active_512/full_342879_attempt02/source_snapshot/scripts/build_query_graph.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate full --limit 342879 \
  --reuse-medium qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/query_graph/medium_050000/manifest.json \
  --reuse-medium-sha256 e9a5411167a2d39b685f61c5fde85ccff64a74905e3b68f26f144d45551140db
```

full 完成后独立复核（不加载模型、不使用 GPU）：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/outputs/run_control/p4_cat_active_512/full_342879_attempt02/source_snapshot/scripts/build_query_graph.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate full --limit 342879 --validate-only
```

预期 `status=validated`、`query_rows=342879`、`next_status=HOLD_FOR_P4_FULL_REVIEW`，manifest SHA256=`f6f5a6b4a6f5ef78ff8eb1007e8b566a17fe5a911aae215185b0ed156ab6329d`。`--dry-run` 只核验图输入，不写输出、不校验 D3 chunk 数值或加载模型；它不替代独立产物验收。P4 不启动 P5 或 SID 聚类。

P5 10k sample 已完成。以下命令使用成功 attempt02 的冻结源码只读复核，不重新拟合、不覆盖输出；不能用已经包含 100k I/O 修订的 live source 校验旧 manifest：

```bash
cd /ofs/map_search/hudan/poi_genret

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/run_control/p5_cat_active_512/sample_010000_attempt02/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p5_cat_active_512/sample_010000_attempt02/source_snapshot/scripts/build_poi_prqk.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate sample --validate-only
```

预期 `status=validated`、`poi_rows=10000`、`next_status=HOLD_FOR_P5_SAMPLE_REVIEW`；正式 manifest SHA256=`c9d18df7d1029c9b80442965f31943a31cf4c3ef076de2aa6b9fd5b6d437f4e2`。validator 重验完整 active 公共方向、全部 artifact 哈希、精确 assignment 和 projection residual，且确认 Query/Category/Geo/业务 Validation/Test 未被读取。

100k medium 已在用户确认后按以下命令完成，并显式绑定 sample；这是历史构建命令，不表示重跑授权：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ofs/map_search/hudan/poi_genret/qg_prqk/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/scripts/qg_prqk.py build-base-codebook \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate medium100k \
  --previous-gate-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/sample_010000/manifest.json \
  --previous-gate-manifest-sha256 c9d18df7d1029c9b80442965f31943a31cf4c3ef076de2aa6b9fd5b6d437f4e2
```

正式输出位于 `poi_prqk_a0/medium_100000/`，manifest SHA256=`8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0`。独立复核必须使用 attempt01 冻结源码：

```bash
cd /ofs/map_search/hudan/poi_genret

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_attempt01/source_snapshot/scripts/build_poi_prqk.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --gate medium100k --validate-only
```

预期 `status=validated`、`poi_rows=100000`、`next_status=HOLD_FOR_P5_MEDIUM100K_REVIEW`。三层利用率、residual 和桶结构正常，但 hard fit 都达到 30 轮上限且 Top-k refinement 逐轮恶化 hard distortion。这是原 100k 的历史停止点，后续受控诊断已在独立目录完成。

100k hard endpoint 受控诊断的历史构建命令（不表示授权重跑）：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_hard_diag_attempt01/source_snapshot/src \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_hard_diag_attempt01/source_snapshot/scripts/diagnose_p5_medium100k.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --reference-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/medium_100000/manifest.json \
  --reference-manifest-sha256 8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0
```

正式诊断输出为 `poi_prqk_a0/diagnostics/medium_100000_hard_endpoint_v1/`，manifest SHA256=`00bb291a4f58ed37788382ae7d5abd70ee9294a0749c41c27277afe82d8f1e29`。只读独立复核使用上述相同命令追加 `--validate-only`，预期 `status=validated`、`next_status=HOLD_FOR_P5_MEDIUM100K_DIAGNOSTIC_REVIEW`。validator 重验冻结来源/诊断 artifact 哈希、精确 hard assignment 和逐层 projection residual，不修改输出。

诊断已证实 hard30 不足，hard60-off 在 43/39/33 轮收敛；同时原 Top-k5 的 distinct SID 高于两个 Top-k-off 分支。后续获用户确认补充的 hard60+Top-k5 配对诊断也已完成，历史构建命令如下，不表示授权重跑：

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/ofs/map_search/hudan/poi_genret/qg_prqk/outputs/tmp/p5m \
  /ofs/map_search/hudan/envs/poi-gr/bin/python -u \
  qg_prqk/outputs/run_control/p5_cat_active_512/medium_100000_hard60_topk5_attempt02/source_snapshot/scripts/diagnose_p5_medium100k_hard60_topk5.py \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml \
  --reference-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/medium_100000/manifest.json \
  --reference-manifest-sha256 8ccaf6383e1a89c62780190f96222fac6a6d2508002bc5af30f7c7dc0cbdf3d0 \
  --endpoint-diagnostic-manifest qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/diagnostics/medium_100000_hard_endpoint_v1/manifest.json \
  --endpoint-diagnostic-manifest-sha256 00bb291a4f58ed37788382ae7d5abd70ee9294a0749c41c27277afe82d8f1e29
```

正式输出为 `poi_prqk_a0/diagnostics/medium_100000_hard60_topk5_v1/`，manifest SHA256=`f33bf2573c141b2ae47bf5e578be3cbe0debd0aed03f23e834a9c982814dbd90`。只读复核使用相同命令追加 `--validate-only`，预期 `status=validated`、`next_status=HOLD_FOR_P5_MEDIUM100K_PARAMETER_REVIEW`；这是历史诊断合同，不代表当前状态。用户随后已确认新 10k，并于 2026-09-09 另行授权 direct-full；P5 full、P6 sample 和 P6 sample 归因现均已完成。

512 下游配置只读核验（不创建产物、不读取业务数据、不启动训练）：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py inspect-sid-config \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml
```

预期 `status=config_validated`、容量 `[512,512,512]`、向量维数 1024、`downstream_started=false`；追加 `--resolved` 查看全部继承参数。输入是新 YAML 及其冻结上游配置；输出为终端 JSON，无输出文件。此命令只查配置，实际产物内容校验另属 P4 Gate。新 P4–P8 输出固定在 `qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/`，旧 `1024x3` 配置/产物原样保留。

已完成 P3A-FULL 产物的复核命令（不会训练或读取业务 Validation/Test）：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py select-query-adapter \
  --config qg_prqk/configs/qg_prqk_p3a_full_active_1024x3_v1.yaml \
  --validate-only
```

预期输出 `status=validated`、`best_epoch=2`、`next_status=HOLD_FOR_P3A_FULL_REVIEW`。其中 `next_status` 是冻结 P3A 协议的历史停止标记，不会将用户后续确认的 canonical 状态退回 HOLD；不为修改这个标记而改动已冻结 P3A 实现。日志位于 `qg_prqk/outputs/run_control/p3a_full_active/attempt03_full/validate_console.log`；完整指标与哈希见实施状态文档。以下保留已执行阶段的历史构建命令，不表示授权重跑；完成目录禁止覆盖。

后续候选宇宙固定为 716,245 条 active POI。QG 自己的 BGE 与类别精确行子集构建/复核入口：

```bash
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py build-active-poi \
  --config qg_prqk/configs/qg_prqk_active_poi_assets_v1.yaml

PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py build-active-poi \
  --config qg_prqk/configs/qg_prqk_active_poi_assets_v1.yaml \
  --validate-only
```

active P2.5-CAT 的复核命令：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py build-query-supervision \
  --config qg_prqk/configs/qg_prqk_v2_1_category_active_1024x3.yaml \
  --validate-only --no-progress
```

该命令会完整复核输出 hash/schema/行数、active 类别行序、D3 守恒以及每个 Query–层的概率和与边权和。active manifest SHA256=`5f203b94d567433f89ff25a7755516b211544b9be2c484ac28e2243ccd7ce55f`；旧 2,337,178 行 P2.5 输出继续保留，但不能作为 active P3 输入。

P3A 50,000 条 D3 Query 的历史 Train-only Gate 已完成并通过。用户已经确认 active P3A-FULL，并于 2026-09-05 确认建模前 seed=42、SELECT/FINAL 共用新 identity 初始 state（不声称逐值复现历史 Gate 初始权重）；以下为本次已完成任务的正式启动命令，完成目录会拒绝重复构建：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py select-query-adapter \
  --config qg_prqk/configs/qg_prqk_p3a_full_active_1024x3_v1.yaml
```

2026-09-04 首次 active FULL 在 Query 编码阶段中断。已确认的前缀恢复命令（只生成 cache，不训练）为：

```bash
LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py select-query-adapter \
  --config qg_prqk/configs/qg_prqk_p3a_full_active_1024x3_v1.yaml \
  --prepare-query-cache-only \
  --recover-query-prefix-from qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_adapter_exact_full_interrupted_20260904
```

新 cache 输出到 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat_active/query_cache_exact_full/`。逐块顺序写入 NPY、fsync 后记录 hash 和进度；恢复只接受已提交块，最终全量顺序合并为 `embeddings.npy` 并发布 manifest/`_SUCCESS`。可重复执行同命令校验并复用完成块，不能依据大 NPY 的预分配大小判断完成。正式 FULL 入口自动复核并读取这一 cache。`LD_LIBRARY_PATH` 显式使用 poi-gr 的 C++ runtime，避免环境未激活时 FAISS 报 `GLIBCXX_3.4.29` 缺失。

历史 FULL 已使用 `--validate-only` 独立复核并按当时要求停在 `HOLD_FOR_P3A_FULL_REVIEW`；后续授权与当前动作以本页第 5/8 节和 canonical 状态文档为准。

P3A 正式 Gate 的独立复核命令：

```bash
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py gate-query-adapter \
  --config qg_prqk/configs/qg_prqk_p3a_1024x3_v1.yaml \
  --gate medium \
  --validate-only
```

正式输出位于 `qg_prqk/outputs/qg_prqk_1024x3_v2_1_cat/query_adapter_exact/`，manifest SHA256 为 `12a29f1ff90d16f619231da18b99916da0afc7a2dfa4d138a18b9bb4bac96c2f`。该命令只做完整性复核，不重新训练。

当前 A0/原 A4/NoGID A4 三方 SID 可视化已生成，通常只需运行只读 validator：

```bash
MPLCONFIGDIR=qg_prqk/outputs/cache/matplotlib \
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py visualize-sid \
  --config qg_prqk/configs/qg_prqk_sid_visualization_a0_a4_nogid_full_v1.yaml \
  --validate-only
```

预期 `status=sid_visualization_validated`、`next_status=HOLD_FOR_NOGID_S3_REVIEW`。不带 `--validate-only` 是首次生成命令；因 `overwrite=false`，不能覆盖现有正式图片。输出位于 `qg_prqk/outputs/figures/qg_prqk_sid_a0_a4_nogid_full_v1/`。

## 6. 已有代码的轻量验证

```bash
cd /ofs/map_search/hudan/poi_genret

LD_LIBRARY_PATH=/ofs/map_search/hudan/envs/poi-gr/lib \
PYTHONPATH=qg_prqk/src \
  /ofs/map_search/hudan/envs/poi-gr/bin/python \
  -m unittest discover -s qg_prqk/tests -v

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py preflight \
  --config qg_prqk/configs/qg_prqk_1024x3.yaml \
  --limit 2 \
  --dry-run
```

第一条命令运行全部 QG 合成回归（当前 168 项，覆盖 P4 full、P5 sample/100k I/O 与诊断、P6 sample/full/归因/S2 诊断、P7/P8 canonical、NoGID 及 SID 可视化配置/抽样/统计合同），第二条只验证历史 v1.1 preflight；v2.1-CAT 的 active P2.5、P3A-FULL、P4–P8、NoGID 受控变体和三方可视化入口均已实现。历史产物校验继续使用各阶段对应的冻结 YAML/源码。

P3 合成可训性 smoke：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py smoke-query-adapter \
  --config qg_prqk/configs/qg_prqk_1024x3.yaml \
  --steps-as-epochs 60
```

该 smoke 不读取业务数据、不产生真实 checkpoint，也不能把 `P3_EXACT_ADAPTER` 改成 Gate/full。

## 7. 阶段结束检查

```bash
cd /ofs/map_search/hudan/poi_genret
git diff --stat
git status --short
sed -n '1,100p' qg_prqk/docs/experiments/QG_PRQK_IMPLEMENTATION_STATUS.md
```

若阶段涉及代码，还必须执行状态文档中记录的 compileall、unit test、`--help` 和 sample/smoke。若阶段涉及正式运行，还要核对 `qg_prqk/outputs/` 中的 manifest、schema、行数、SHA256、日志和 checkpoint。

## 8. 当前预期顺序

```text
M1 v2.1-CAT 文档/配置迁移（已完成）
-> active 资产与 P2.5-CAT 类别 Query 粒度标注（已完成）
-> P3A Exact Adapter Gate/FULL-SELECT/FULL-FINAL（已完成）
-> HOLD_FOR_P3A_FULL_REVIEW（已履行，2026-09-07 用户确认继续完整方法）
-> 512×3 配置/文档迁移（已完成）
-> P4-CAT 最小实现与 1,000-query sample（已完成并独立验收）
-> HOLD_FOR_P4_SAMPLE_REVIEW（已履行，用户确认继续）
-> P4-CAT 50,000-query medium（已完成并独立验收）
-> HOLD_FOR_P4_MEDIUM_REVIEW（已履行，用户确认继续）
-> P4-CAT full 类别层级 Query 图与 embedding（已完成并独立验收）
-> HOLD_FOR_P4_FULL_REVIEW（已履行，用户确认进入 P5）
-> P5-CAT POI-only PRQK 512×3 10k sample（已完成并独立验收）
-> HOLD_FOR_P5_SAMPLE_REVIEW（已履行，用户确认继续）
-> P5-CAT 100k medium（已完成并独立验收，算法 Gate 待审）
-> HOLD_FOR_P5_MEDIUM100K_REVIEW（已履行，用户批准受控诊断）
-> P5-CAT 100k hard30/60+Top-k-off 受控诊断（已完成并独立验收）
-> HOLD_FOR_P5_MEDIUM100K_DIAGNOSTIC_REVIEW（已履行，用户批准配对诊断）
-> P5-CAT 100k hard60+Top-k5 配对诊断（已完成并独立验收）
-> HOLD_FOR_P5_MEDIUM100K_PARAMETER_REVIEW（已履行，用户确认先试 hard60+Top-k5）
-> P5-CAT hard60+Top-k5 独立 10k sample（已完成并独立验收）
-> HOLD_FOR_P5_HARD60_TOPK5_SAMPLE_REVIEW（已履行，用户授权跳过新协议 100k/500k）
-> P5-CAT hard60+Top-k5 active POI full（已完成并独立验收）
-> HOLD_FOR_P5_HARD60_TOPK5_FULL_REVIEW（已履行，用户确认 P6/P7 上限为 60）
-> P6/P7 hard60 overlay 与 P6 输入/闭包合同（已完成）
-> HOLD_FOR_P6_SAMPLE_PROTOCOL_CONFIRMATION（已履行，用户确认闭包规模）
-> P6-CAT S1/S2 sample 与同闭包 P5 A0 对照（已完成并独立验收，Query distortion Gate 未通过）
-> HOLD_FOR_P6_SAMPLE_REVIEW（已履行，用户批准归因诊断）
-> P6-CAT sample 六分支 graph/prior/Top-k/category 归因（已完成并独立验收）
-> HOLD_FOR_P6_SAMPLE_ATTRIBUTION_REVIEW（已履行，用户改为 sample smoke 后 direct full）
-> P6-CAT full 716,245 POI / 342,879 Query（已完成并独立验收）
-> HOLD_FOR_P6_FULL_REVIEW（已履行，用户批准先诊断 S2）
-> P6-CAT full S2-only 六分支与四点几何分解（已完成并独立验收）
-> HOLD_FOR_P6_S2_DIAGNOSTIC_REVIEW（已履行，用户接受诊断留存并确认继续 P7）
-> P7-CAT D3 Query + Local Geo + Hard Entity S3 sample（已完成并独立验收）
-> P7-CAT full 716,245 POI / 291,590 D3 Query（已完成并独立验收）
-> HOLD_FOR_P7_FULL_REVIEW（已履行，用户确认继续 P8）
-> P8-CAT A0/A4 静态评测与完整 SID sample（已完成工程闭环）
-> P8-CAT A0/A4 静态评测与完整 SID full（已完成并独立验收，结论为 REVIEW_REQUIRED）
-> HOLD_FOR_REVIEW（原 P8 停止点，已获用户授权追加 NoGID 受控变体）
-> NoGID `(S1,S2)` parent S3 sample/full（已完成并独立验收）
-> A0/原 A4/NoGID A4 统一 `(S1,S2)` S3 候选比较（已完成，`REVIEW_REQUIRED`）
-> A0/原 A4/NoGID A4 SID 三方并排可视化（已完成并独立验收）
-> HOLD_FOR_NOGID_S3_REVIEW（已履行，用户批准两个分支均准备 SFT）
-> P9 双 Final ID + 双 history10 Messages（已完成并验证）
-> HOLD_FOR_QG_PRQK_SFT_DATA_REVIEW（已履行）
-> P10 共同扩词表 + 双 Train/Valid Tokenized Cache（已完成并验证）
-> HOLD_FOR_QG_PRQK_DUAL_SFT_LAUNCH_REVIEW（当前停止点）
```

P5 用户授权的 direct-full 已绑定 sample manifest SHA256=`29a918fb5d3b3727d393ad31dc69e6f95f61fcd49a3036af8d1981473784f4d7`，并在 full manifest 中记录跳过 100k/500k；默认 Gate 保护没有放宽。P5 full manifest SHA256=`ec84eb8fa2c1c65059caab1de94d43095d82048c25310564e6950ab0a244789e`。P8 完成后下列阶段保持冻结：

```text
-> Final PID / collision Dedup ID
-> Tokenizer / Trie / Qwen SFT
-> 外部基线、Order-B 和其他消融
```

NoGID 和 P9 停止点已由用户明确履行；共同扩词表与两套 Train/Valid Cache 已完成，两个分支均满足超长 0、target truncation 0。当前只等待训练启动审核，训练、checkpoint 和指标尚未生成。

## 9. 双分支 SFT 入口

目标定义：

```text
A4 GID-parent = G1 G2 G3 G4 G5 G6 S1 S2 S3 [D]
A4 NoGID      = S1 S2 S3 [D]
```

完整命令见 `qg_prqk/README.md` 的“当前 SFT 启动方式”。最短只读检查为：

```bash
/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py train-sft --help

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py train-sft --variant a4_gid_parent --dry-run

/ofs/map_search/hudan/envs/poi-gr/bin/python \
  qg_prqk/scripts/qg_prqk.py train-sft --variant a4_nogid --dry-run
```

统一训练命令不会自行生成或绕过 Cache；缺少共享扩词表、对应 full Cache、`_SUCCESS`、1024 cutoff 或零截断结果时必须失败。Test 没有注册到 `qg_prqk/configs/sft/dataset_info.json`。

训练平台 launcher 直接位于 QG 方法目录，两个脚本都接受 4×A100 或 4×6000D，但不改变冻结训练配置：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a4_gid_parent_sft_4gpu_3epoch.sh --help
bash qg_prqk/launchers/run_train_qg_prqk_a4_nogid_sft_4gpu_3epoch.sh --help
```

Cache 门禁通过后依次使用 `--dry-run` 和无参数正式启动。两个分支使用不同的短 TMPDIR、run-control 日志和输出目录；launcher 不支持隐式续训，也不会覆盖非空输出。平台文件继承已有 launcher 的环境、挂载和认证骨架，因含本地平台字段被 Git 忽略，不得把内容粘贴到文档或提交。

当前两条 `--dry-run` 已退出 0。审核后在训练平台选择恰好 4 张同型号 A100 或 4 张 RTX PRO 6000D，正式启动其中一个分支：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a4_gid_parent_sft_4gpu_3epoch.sh
```

或：

```bash
bash qg_prqk/launchers/run_train_qg_prqk_a4_nogid_sft_4gpu_3epoch.sh
```

若两个分支都训练，建议等待第一个分支退出且 GPU 释放后再启动第二个。不要同时向同一输出目录启动重复任务。

## 10. 会话占用故障

若 `codex resume --last` 报 `already has an active writer`：

1. 先确认是否还有其他 Codex TUI/进程；
2. 不手工修改 session JSONL；
3. 不反复强制 resume；
4. 旧进程不可恢复时，按第 4 节用 canonical 状态开启新会话。

只读查看进程：

```bash
ps -ef | grep '[c]odex'
```

不要在未确认进程归属时批量 `kill -9`。
