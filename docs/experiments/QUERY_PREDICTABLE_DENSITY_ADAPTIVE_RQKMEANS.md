# QD-RQ：面向 Query 可预测性的密度自适应 RQ-KMeans 构想

> 文档状态：方法构想，尚未实现、尚未运行实验。
> 记录日期：2026-08-11。
> 暂用名称：Query-Predictable Density-Adaptive RQ-KMeans，简称 QD-RQ。
> 本文只冻结当前讨论形成的研究假设，不代表方法已经有效，也不改变现有 SFT 与实验执行计划。

## 1. 研究动机

北京全量 POI 上已经得到以下实验现象：

1. 在相同 Embedding、相同三层 `1024×1024×1024` 码本和相同评测协议下，RQ-KMeans 的静态 SID 唯一性、碰撞覆盖、重构误差和码本利用率整体优于当前 TIGER RQ-VAE，因此 RQ-KMeans 更适合作为后续 SID 优化的基础量化器。
2. 在固定 30-bit 总容量下，把容量从 S3 前移到 S1，会单调提高静态 SID 唯一性并缩小热点桶；但同时显著扩大第一步生成词表和低支持 S1 数量，使生成模型第一 Token 更难预测。
3. 把容量后移到 S3 更符合“由粗到细”的自回归生成直觉，但较小的 S1 会把更多差异留给后续层，导致完整路径唯一性下降和热点路径增大。
4. 第一层硬对齐 `category_code` 虽然使 S1 类别纯度达到 100%，但北京 POI 类别规模严重不均衡。楼栋号、门牌信息等超大类别被压入单一根节点后，共享的全局 S2/S3 残差码本无法恢复早期容量损失，最终静态结构明显退化。
5. E4 已经证明 Train-only 正向 Query 可以显著改善连续 POI 向量检索；但 E4 在量化前就把内容向量与 Query 残差融合为同一个连续向量，普通 RQ-KMeans 无法决定 Query 信息应进入粗层还是细层，也没有直接优化“Query 是否容易预测目标 SID”。

这些结果说明，当前核心问题不只是码本是否均匀，也不只是连续向量重构是否准确，而是：

> 如何构建一个既保留稳定粗语义、又把真实用户检索表达逐步放入细层，同时避免密集 POI 区域垄断有限码本容量的 SID。

普通 RQ-KMeans 只最小化 POI 表示的量化误差：

$$
\min_{\mathcal{C},\mathbf{s}}
\sum_i
\left\|
e_i-\sum_{l=1}^{L}c_{s_i^l}^{l}
\right\|_2^2
$$

这个目标不知道哪些码字容易从 Query 预测，也不知道不同层应该承载何种信息。QD-RQ 的基本方向是把“Query 可预测性”和“密度自适应容量”直接加入离线 SID 构建，而不是只在后续 SFT 中被动学习。

## 2. 创新边界

以下方法可以作为相关工作、强基线或消融，但不能作为本项目的主要创新声明：

- 普通 RQ-KMeans；
- OneRec 的全层等容量 Balanced K-Means；
- OneSearch 的 L3-only Balanced K-Means；
- 在最后残差上追加 OPQ；
- Faiss ResidualQuantizer 的 Beam Search、progressive-dimension 或 codebook refinement；
- 固定码本布局的容量前移或容量后移。

QD-RQ 计划形成的差异是：

1. 不只重构 POI 内容，而是让同一条离散 SID 路径同时对齐 POI 内容视图和真实 Query 视图；
2. Query 对量化决策的影响由粗到细逐层增强，使粗层保持稳定语义，细层表达别名、楼号、门牌、局部地址等 Query 可区分信息；
3. 不强制所有簇等容量，而是用密度温度化目标压缩热点、保留真实语义密度；
4. 只使用现有 Train 正向 `Query → target POI` 样本，不要求在 SID 离线构建阶段额外挖掘困难负样本。

当前只能把这些内容称为“候选创新”。在形成论文创新声明前，仍需继续进行相关工作查新，并用完整消融和下游 SFT 结果证明各组件的必要性。

## 3. 方法总览

QD-RQ 为每个 POI 保留两个独立视图，但让两个视图共享同一组离散 Token assignment：

```text
POI 名称/地址/别名              Train-only 正向 Query–POI
          ↓                                ↓
  POI 内容向量 c_i              POI Query 聚合与类别公共方向残差 q_i
          └──────────────┬─────────────────┘
                         ↓
             双视图共享 assignment
                         ↓
           S1：稳定粗粒度内容语义
                         ↓
           S2：内容 + 部分 Query 语义
                         ↓
           S3：强化 Query 可区分残差
                         ↓
        密度温度化的后层容量正则
                         ↓
                  三层基础 SID
```

方法由两个核心组件组成：

1. 分层 Query 对齐的双视图残差量化；
2. 密度温度化的软容量约束。

## 4. 分层 Query 对齐的双视图残差量化

### 4.1 两个输入视图

对 POI $i$，定义：

$$
c_i\in\mathbb{R}^{d}
$$

为原始 BGE POI 内容向量，内容来自名称、地址和别名。

定义：

$$
q_i\in\mathbb{R}^{d}
$$

为只使用 Train 正向 Query 构造的 POI Query 视图。第一版计划复用 E4 已冻结的 Query 聚合逻辑：先对 Query 按订单频次与跨 POI 的 IDF 加权聚合，再减去部分 `category_code` 公共 Query 方向，使 $q_i$ 更偏向同类 POI 之间可区分的真实检索表达。

没有 Train Query 覆盖的 POI 不伪造 Query 表示，令其 Query 可靠度为：

$$
\rho_i=0
$$

有 Train Query 覆盖时，第一版可先使用统一的：

$$
\rho_i=1
$$

避免在主方法尚未验证时再次引入 E3 式逐 POI 动态融合。只有统一权重通过后，才研究由 Query 数量、一致性或离散度决定的可靠度。

### 4.2 双中心、共享 Token

第 $l$ 层的每个离散 Token $k$ 同时维护两个中心：

$$
u_k^l\in\mathbb{R}^{d}
$$

表示内容视图中心；

$$
v_k^l\in\mathbb{R}^{d}
$$

表示 Query 视图中心。

两个视图不分别生成两套 SID，而是共享同一个离散 assignment $s_i^l$。第 $l$ 层的基础距离为：

$$
D_{ik}^{l}
=
\left\|r_{i,c}^{l}-u_k^l\right\|_2^2
+
\lambda_l\rho_i
\left\|r_{i,q}^{l}-v_k^l\right\|_2^2
$$

其中：

- $r_{i,c}^{l}$ 为内容视图在第 $l$ 层的残差；
- $r_{i,q}^{l}$ 为 Query 视图在第 $l$ 层的残差；
- $\lambda_l$ 控制该层对 Query 可预测性的关注强度。

在暂不加入容量正则时，共享 assignment 为：

$$
s_i^l=\arg\min_k D_{ik}^{l}
$$

两个视图分别更新残差：

$$
r_{i,c}^{l+1}
=
r_{i,c}^{l}-u_{s_i^l}^{l}
$$

$$
r_{i,q}^{l+1}
=
r_{i,q}^{l}-v_{s_i^l}^{l}
$$

这使一个 SID Token 同时具有两种可解释性：它在 POI 内容空间中表示一类相似实体，在真实用户 Query 空间中也对应一类相似检索表达。

### 4.3 Query 强度由粗到细递增

QD-RQ 不让 Query 信号以相同强度进入所有层，而是要求：

$$
0\leq\lambda_1<\lambda_2<\lambda_3
$$

第一版最保守的结构是：

$$
\lambda_1=0,\qquad \lambda_2>0,\qquad \lambda_3>\lambda_2
$$

其含义为：

- S1 只学习稳定的 POI 内容粗语义，避免细粒度 Query 表达扰乱根节点；
- S2 在内容残差上引入一部分真实检索语义；
- S3 更强地表达别名、父实体、道路、楼栋、门牌、出入口等可区分 Query 信息。

这与 E4 的区别不是“是否使用 Query”，而是“能否控制 Query 信息进入哪一层”。E4 把内容和 Query 预先融合为一个向量，所有层看到的输入相同；QD-RQ 保留双视图，并通过 $\lambda_l$ 显式控制层级信息分工。

## 5. 密度温度化的软容量约束

### 5.1 为什么不直接等容量

北京 POI 的语义密度天然不均匀。住宅楼栋、门牌、公司企业等类别规模远大于稀有景点或特定设施。完全等容量会把真实密集区域中的 POI 强制推向较远中心，可能提高表面利用率，却损伤语义层级和 Query 可预测性。

因此，QD-RQ 不直接要求：

$$
n_k=\frac{N}{K}
$$

而是从普通 RQ-KMeans 的自然 occupancy 出发，对其进行连续、可控的压平。

### 5.2 密度温度化目标

设普通 RQ-KMeans 基线中第 $l$ 层码字 $k$ 的全量占用为：

$$
n_{k}^{l,(0)}
$$

定义目标容量：

$$
\tilde n_k^l
=
N
\frac{
\left(n_k^{l,(0)}+\epsilon\right)^{\tau_l}
}{
\sum_j\left(n_j^{l,(0)}+\epsilon\right)^{\tau_l}
}
$$

其中：

- $\tau_l=1$：保持原始自然密度，相当于不做 balance；
- $\tau_l=0$：退化为完全等容量；
- $0<\tau_l<1$：压缩热点簇，但保留不同语义区域的真实密度差异。

第一版只在 L3 加入容量正则：

$$
\tau_1=1,\qquad \tau_2=1,\qquad \tau_3\in\{0.50,0.75\}
$$

这样不改变 S1/S2 的自然粗语义，只限制最终残差层被极少数热点码字垄断。

### 5.3 容量价格与 assignment

为每个码字维护容量价格 $\pi_k^l$，将 assignment 改为：

$$
s_i^l
=
\arg\min_k
\left[
D_{ik}^{l}+\mu_l\pi_k^l
\right]
$$

其中 $\mu_l$ 控制容量正则强度。完成一轮 assignment 后，根据实际占用更新价格：

$$
\pi_k^l
\leftarrow
\pi_k^l
+
\eta
\frac{n_k^l-\tilde n_k^l}{\tilde n_k^l}
$$

超出目标容量的中心价格升高，下一轮会有一部分语义迁移代价较小的 POI 转向其他候选中心；容量不足的中心价格降低。该过程不会像硬等容量一样无条件把每个簇填到完全相同大小。

初始层级计划为：

$$
\mu_1=0,\qquad \mu_2=0,\qquad \mu_3>0
$$

只有 L3 软容量验证通过后，才考虑在 L2 使用更弱的正则。全层强 balance 不进入第一版候选。

## 6. 联合目标的概念形式

QD-RQ 可概括为以下交替优化问题：

$$
\min_{mathbf{s},\mathcal{U},\mathcal{V}}
\sum_{l=1}^{L}\sum_i
\left[
\left\|r_{i,c}^{l}-u_{s_i^l}^{l}\right\|_2^2
+
\lambda_l\rho_i
\left\|r_{i,q}^{l}-v_{s_i^l}^{l}\right\|_2^2
\right]
+
\sum_{l=1}^{L}\mu_l
\mathcal{R}_{\mathrm{capacity}}^{l}
$$

其中：

- 第一项保证 POI 内容语义和连续向量重构；
- 第二项保证 SID 对真实 Query 表达可预测；
- 第三项抑制后层热点容量，但不强制所有语义区域等密度。

该目标的重点不是得到最低的 POI 向量 MSE，而是获得更适合作为自回归生成标签的层级离散路径。

## 7. 与现有 E4 + RQ-KMeans 的区别

| 维度 | 当前 E4 + 普通 RQ-KMeans | QD-RQ 候选 |
|---|---|---|
| Query 使用方式 | 量化前与内容融合为单一连续向量 | 内容与 Query 保持双视图 |
| 离散 assignment | 只按融合向量到中心的欧氏距离 | 内容距离 + 分层 Query 距离 + 容量价格 |
| Query 层级 | 三层看到同一种融合结果 | $\lambda_1<\lambda_2<\lambda_3$ |
| 无 Query POI | 使用原始内容向量 | $\rho_i=0$，自然退化为内容量化 |
| 码本分布 | 自然分布，无容量约束 | L3 密度温度化软容量 |
| 类别信息 | E4 中用于去除 Query 类别公共方向 | 继续作为 Query 视图去公共方向，不硬编码为 S1 |
| 优化目标 | 连续向量重构 | 内容重构、Query 可预测性与容量利用的联合折中 |

## 8. 当前可主张的候选创新点

### 8.1 Query-aware tied assignment

同一个离散 SID assignment 同时对齐 POI 内容和真实检索 Query，而不是只对 Query 增强后的单一混合向量做量化。

### 8.2 Layer-wise query scheduling

通过逐层增大的 $\lambda_l$，把稳定内容语义放在前层，把真实 Query 中的细粒度可区分信息放在后层，直接服务于由粗到细的自回归生成。

### 8.3 Density-tempered capacity regularization

使用 $\tau_l$ 在自然密度与完全均匀之间连续插值，避免等容量 K-Means 对北京超大 POI 类别造成不合理切分。

三者组合后，方法针对的是地图检索特有的矛盾：Query 意图通常明确，但 POI 目录包含大量同类、同名、楼栋和门牌实体；SID 既要共享语义前缀，又必须保留 Query 能够表达的细粒度差异。

## 9. 待验证研究假设

当前需要通过实验验证以下假设，不能提前写成结论：

### H1：双视图 assignment 提高 Query 可预测性

相较于 E4 单向量量化，QD-RQ 应提高 Validation Query 与目标 SID 的逐层一致性，尤其是给定正确 S1/S2 后的 S3 可预测性。

### H2：层级 Query 调度优于全层统一 Query 融合

若细粒度 Query 信号主要进入 S2/S3，S1 的类别纯度、支持度和生成难度不应明显恶化，同时后层 Query 簇内一致性应提高。

### H3：密度温度化优于完全等容量

相较于 L3 exact balance，$\tau_3\in(0,1)$ 应以较小的额外量化误差和较低的强制迁移比例，获得大部分热点压缩收益。

### H4：静态收益能够转化为 SFT 收益

只有当完整 identifier HR、NDCG、逐层 Teacher-Forcing 和合法率实际提高时，才能证明方法比普通 RQ-KMeans 更适合生成式检索。静态唯一率或重构 MSE 单独改善不构成方法成功。

## 10. 最小消融计划（尚未执行）

为了避免把新量化方法与非对称码本布局混为一个变量，第一轮建议固定：

- 北京全量 POI 行序；
- Train/Validation/Test 时间切分；
- 原始 BGE 内容向量；
- E4 已冻结的 Train-only Query 聚合与类别公共方向残差；
- 三层对称 `1024×1024×1024`；
- 相同 30-bit 名义容量、训练样本、随机种子和全量评测协议。

候选矩阵为：

| 组别 | 双视图 Query assignment | L3 容量 | 角色 |
|---|---|---|---|
| R0 | 否，复用当前 E4 单向量 | 无 | 当前强基线 |
| R1 | 否 | 完全等容量 | 已有 balance 参考，不属于提出方法 |
| R2 | 否 | $\tau_3=0.75$ 或 $0.50$ | 单独验证密度温度化 |
| R3 | 是，$\lambda_1=0<\lambda_2<\lambda_3$ | 无 | 单独验证分层 Query 对齐 |
| R4 | 是 | R2 胜出的软容量 | 完整 QD-RQ |

第一阶段只运行离线量化和轻量 Query-to-SID 代理评测。只有 R3/R4 至少一组在预先定义的主要指标上稳定优于 R0，才构建新的唯一 identifier 并进入同协议 SFT。

## 11. 评测指标

### 11.1 现有 SID 指标

- 全量不同 SID 数与唯一 SID 比例；
- 碰撞 POI、Excess Collision、P99 和最大桶；
- 每层码本利用率、归一化熵、最小/平均/最大 occupancy；
- D1/D2/D3 类别 micro/macro purity；
- Validation 重构 MSE 和 cosine；
- Train 加权 $H(S_1)$、$H(S_2\mid S_1)$、$H(S_3\mid S_{1:2})$；
- 固定 Validation 中每层低支持目标数。

### 11.2 QD-RQ 新增指标

- Train-only Query 中心到目标 SID Query 中心的逐层 Top-K；
- 固定 Validation Query 的 S1/S2/S3 代理预测准确率；
- 每层 Query 簇内 cosine 与簇间分离度；
- Query 覆盖和无 Query 覆盖 POI 的分桶结果；
- 容量约束导致的非最近中心 assignment 比例；
- 被迁移 POI 的平均、P95 和最大额外距离；
- 实际 occupancy 与 $\tilde n_k^l$ 的偏差；
- 最终同协议 SFT 的逐层 Teacher-Forcing、完整 identifier HR@1/3/5/10、NDCG@10、合法率和解码时延。

代理 Query-to-SID 指标只能用于低成本筛选，不能替代生成模型 SFT。

## 12. 实现与资源约束

若后续确认进入实现，必须满足：

1. Query 视图和类别统计严格只使用 Train，Validation/Test 不参与聚合、中心计算或参数选择；
2. 复用现有 491,213 条 Query 覆盖 POI 行映射、E2/E4 聚合产物和冻结哈希，不重新制造一套统计口径；
3. 全量 2,337,178 条 POI 使用 mmap 和固定 chunk 流式处理；
4. 不构造 `N×K×d` 或全量 dense distance 常驻矩阵；每个 POI 只保留有限 Top-T 候选及距离；
5. 中间缓存、memmap、日志和产物只能放在 `/ofs/map_search/hudan/poi_genret/outputs/` 下，不使用开发机系统 `/tmp`；
6. GPU 只负责分块距离计算和必要的中心更新，CPU 内存只保存紧凑 assignment、容量统计与小规模中心；
7. 在真实全量实验前先定义合成数据契约并完成确定性、无 Query 退化、容量价格和残差更新测试；
8. 不覆盖现有 E4、RQ-KMeans 或 RQ-VAE 产物，使用独立输出目录和 manifest。

## 13. 已知风险

### 13.1 Query 覆盖稀疏

Train Query 当前只覆盖约 21% 的全量 POI。QD-RQ 必须保证无 Query POI 能自然退化为内容量化，且不能为了覆盖 POI 把 Validation/Test Query 引入 SID 构建。

### 13.2 Query 聚合质量不稳定

真实用户 Query 可能包含错拼、简称和上下文省略。E4 已通过类别公共方向残差减轻泛语义，但双视图量化仍可能放大噪声。第一版不增加逐 POI 动态可靠度，先验证统一权重。

### 13.3 Query 可预测性与目录唯一性冲突

把相似 Query 的 POI聚到一起可能让 Token 更容易生成，却可能增加同路径 POI。必须联合观察逐层可预测性、静态路径结构和最终 identifier 指标，不能只优化其中一个。

### 13.4 容量正则引入语义迁移

任何 balance 都会让一部分 POI离开最近中心。需要把迁移比例和额外距离作为主要约束，而不是只报告码本熵提高。

### 13.5 方法复杂度与创新收益不匹配

如果分层 Query 对齐只有很小收益，而普通 E4 + RQ-KMeans 已能达到相同效果，则双中心和容量价格不值得增加工程复杂度。完整消融必须证明 R3 和 R4 的独立贡献。

### 13.6 相关工作新颖性

当前方法是基于已阅读的 TIGER、OneRec、OneSearch、GNPR、GenPOI 和项目实验提出的候选。正式写论文前仍需专项检索多视图量化、query-aware semantic ID、balanced hierarchical clustering 和 generative retrieval tokenization，确认创新边界。

## 14. 暂不并入第一版的扩展方向

如果全局 L3 软容量仍不能处理密集前缀，可以研究 prefix-aware adaptive branching：根据前缀 POI 数、残差方差和 Query 离散度，为不同前缀分配不同的局部分支数。

可定义前缀复杂度：

$$
A_p
=
N_p^{\alpha}
\cdot
\operatorname{Var}(r_p)^{\beta}
\cdot
H(Q\mid p)^{\gamma}
$$

再在总中心存储预算下按 $A_p$ 分配局部子分支。该方向更直接地解决楼栋号等超大分支，但会从全局 residual codebook 转向条件化层级量化，并引入局部码字复用、Token 语义和增量更新问题，因此不与 QD-RQ 第一版同时实现。

OPQ、Beam RQ 和局部图消歧也暂不并入 QD-RQ 第一版：

- OPQ 可作为最后残差压缩的已有强基线；
- Beam RQ 可作为重构优化消融；
- 局部图与碰撞后缀属于基础 SID 完成后的实体消歧问题。

保持变量隔离后，才能判断 QD-RQ 的收益究竟来自 Query-aware assignment、密度自适应容量，还是额外编码容量。

## 15. 后续决策闸门

当前只记录构想，不启动代码或实验。若后续确认继续，执行顺序应为：

1. 补充相关工作查新，确认方法命名与创新边界；
2. 冻结双视图数据契约、$\lambda_l$ 候选和 $\tau_3$ 候选；
3. 实现合成数据上的双视图共享 assignment 与软容量单元测试；
4. 在固定小样本上验证无 Query 退化、容量收敛和确定性；
5. 执行对称 `1024³` 的 R0—R4 最小消融；
6. 只让离线指标通过的最优候选进入唯一 identifier 与同协议 SFT；
7. 最后再判断是否迁移到容量后移、对称和容量前移布局。

在第 5 步得到正式结果前，不新增 QD-RQ 方法实验文档，不在 `EXPERIMENT_LOG.md` 登记实验编号，也不把 QD-RQ 写入 `PROJECT_STATUS.md` 的已完成进展。

## 16. `EXP-20260813-01`：R3 强 Query 权重首轮固定样本筛选

### 16.1 目标与协议

本轮只验证 R3 的分层双视图共享 assignment，不加入 R2/R4 的容量约束。固定条件为：

- 内容视图：原始 BGE-M3 POI 向量；
- Query 视图：Train-only E2 聚合减去 `category_code` 公共方向，固定 $\beta=0.85$；
- Query 覆盖：491,213 / 2,337,178 条 POI，未覆盖 POI 的 $\rho_i=0$；
- 码本：`1024×1024×1024`；
- 固定样本：100,000 条，索引 SHA256 为 `2ff976ba4c216782fb79027fd5dae95e9fb23b5d0a93885345e18d3c62126afe`；
- 初始化：原始 BGE `1024³` RQ-KMeans 的同样本码本与 assignment；
- 层级权重：$\lambda=[0,0.25,0.75]$；
- Lloyd 迭代：S2/S3 各 10 次；
- 实现：内容中心由全部样本更新，Query 中心只由 20,908 条覆盖样本更新；S1 完全冻结，两个视图共享 Token、分别更新残差。

正式命令为：

```bash
export TMPDIR=/ofs/map_search/hudan/poi_genret/outputs/tmp/qdr3
PYTHONPATH=src /ofs/map_search/hudan/envs/poi-gr/bin/python \
  scripts/sid/build_query_predictable_rqkmeans.py \
  --config configs/sid/query_predictable_rqkmeans_bge_m3_1024x3.yaml
```

代码运行时工作树非干净，本轮只新增独立 QD-RQ 实现、配置、入口和合成测试，不改写已完成的 E4/RQ-KMeans 产物。

### 16.2 运行与资源

任务在开发服务器单张 RTX A6000 上完成，退出状态为 0，`_SUCCESS` 存在。GPU 峰值 allocated/reserved 显存为 80.16/102.00 MiB，没有 OOM。总耗时为 54,349.51 秒；其中 GPU Lloyd 计算很短，主要耗时来自服务器文件系统读取全量 4.8 GiB BGE 和 1 GiB Query 聚合。首个随机 mmap 读取版本曾陷入不可中断 I/O 等待，未产生有效指标，已停止并改为顺序分块扫描；该调试过程不单独编号。

### 16.3 结果

| 固定 100,000 条训练样本内统计 | 原始 BGE RQ-KMeans 样本码 | R3 强权重样本码 | 变化 |
|---|---:|---:|---:|
| S1 assignment 变化 | — | 0.000% | 符合冻结约束 |
| S2 assignment 变化 | — | 18.213% | Query 已进入中层 |
| S3 assignment 变化 | — | 32.046% | Query 后层作用更强 |
| S1/S2/S3 使用码字 | 1024/1024/1024 | 1024/1015/996 | S2/S3 出现空码字 |
| 样本内 distinct SID ratio | 96.686% | 93.016% | -3.670pp |
| 样本内 collision excess ratio | 3.314% | 6.984% | +3.670pp |
| 样本内最大重复桶 | 13 | 27 | +14 |
| D1 micro purity | 67.099% | 67.099% | 0.000pp |
| D2 micro purity | 90.607% | 89.824% | -0.783pp |
| D3 micro purity | 99.438% | 98.139% | -1.299pp |

上表的 `96.686% / 13` 只是对固定 100,000 条码本训练样本的三层 code 去重统计，不是 2,337,178 条 POI 的全量 SID 评估，也不能与全量碰撞指标混用。已冻结的原始 BGE + RQ-KMeans `1024³` 全量基线使用 500,000 训练样本和 3 次重启：完整 SID 唯一率为 76.0725%，collision excess 为 23.9275%，碰撞 POI 为 36.1501%，最大碰撞桶为 467。本轮 100,000 样本结果只用于同样本、同初始化下筛查 R3 是否引发结构恶化；未做全量编码前，不得宣称 R3 的全量唯一率或最大碰撞桶。

对 20,908 条 Query 覆盖样本，S2/S3 Token 相对内容基线的变化率分别为 36.718%/66.200%；无 Query 样本虽然不直接使用 Query 距离，但由于共享内容中心被覆盖样本重新估计，S2/S3 仍有 13.321%/23.017% 的 Token 变化。S2 的覆盖 Query 残差均方距离在 10 轮中由 0.69646 降至 0.66623；S3 由 0.59164 降至 0.54627，说明优化目标按设计下降，但结构损失明显过大。第 10 轮 S2/S3 仍有 0.737%/1.866% assignment 变化，尤其 S3 尚未完全收敛。

### 16.4 结论与下一步

本组为正式负向筛查结果，不扩大到 500,000 样本，也不进入全量编码、唯一 identifier 或 SFT。它验证了“Query 信号后移”的实现有效，但否定了全码本自由重分配下的 $\lambda=[0,0.25,0.75]$：覆盖稀疏时，共享中心迁移会连带扰动无 Query POI，并使固定样本内的重复显著增加。这一负向结论是严格的同样本相对比较，但不是 R3 的全量碰撞结论。

下一步仍属于 R3，不引入 R2/R4：把 S2/S3 assignment 限制在内容距离的 Top-T 候选内，先使用 `Top-32` 与更弱的 $\lambda=[0,0.05,0.15]$。在固定 100,000 样本上，只有 Query 残差改善且 distinct SID ratio 下降不超过 0.5pp、样本内最大重复桶不超过对照的 1.25 倍、D2/D3 micro purity 分别下降不超过 0.3/0.5pp，才继续第二档权重或 500,000 码本训练；方法候选最终必须对 2,337,178 条 POI 做全量编码与统一 evaluator 评测。
