# 生成式 POI 检索创新方案与后续实验规划

## 1. 研究背景

当前项目已经在北京全量 POI 数据上完成 V1、TIGER 和 GenPOI 的复现，并获得了以下初步结论：

- 北京全量 POI 数量为 2,337,178；
- BGE-M3 是当前效果最好的共享文本编码器；
- TIGER 的生成式检索效果优于当前 GenPOI 复现；
- TIGER 基础三层 SID 唯一率为 71.6766%；
- 约 953,321 个 POI 位于碰撞桶中；
- TIGER 通过追加随机 Collision Token 保证最终标识唯一；
- 随机 Collision Token 只能区分 POI，无法表达楼号、单元、商铺号、分店等真实语义；
- GenPOI 将地理信息直接加入量化输入后，出现第一层码本利用不足和语义层次较差的问题；
- 当前固定 Train 包含大量高置信度的正向 `Query → POI` 发单样本；严格按 7 月 1—12 日 Train-only 统计，共有 491,213 个 POI 成为过目标，占全量目录 21.0174%。

这些结果反映出地图 POI 的 Semantic ID 构建存在一个核心矛盾：

> 相似 POI 应当共享语义前缀，以降低生成难度；但同名分店、同小区不同楼号、同商场不同商铺等实体又必须能够被唯一识别。

因此，后续方法需要分别解决两个问题：

1. 如何利用真实 Query 日志，让基础 SID 更符合用户实际检索语义；
2. 如何在不破坏共享语义结构的情况下，解决基础 SID 的实体碰撞。

---

## 2. 总体方法

拟采用以下整体框架：

```text
训练集正向 Query–POI 日志
              +
       原始 BGE POI 向量
              ↓
   Query-Augmented POI Embedding
              ↓
           RQ-KMeans
              ↓
        基础共享语义 SID
              ↓
       检测基础 SID 碰撞
              ↓
    碰撞桶局部关系图与路径选择
              ↓
      变长关系语义消歧后缀
              ↓
         最终唯一 POI SID
```

可以将最终标识形式化为：

$$
\operatorname{SID}_i = Q\left(z_i^{\mathrm{sem}}\right) \oplus R\left(i, B_i\right)
$$

其中：

- $z_i^{\mathrm{sem}}$ 为 Query 增强后的 POI 语义向量；
- $Q$ 为 RQ-KMeans 等基础语义量化器；
- $B_i$ 为 POI 所属的基础 SID 碰撞桶；
- $R(i,B_i)$ 为只在发生碰撞时构建的关系消歧后缀；
- $\oplus$ 表示基础 SID 与可选变长后缀的连接。

---

## 3. 创新点一：正向 Query 增强的 POI Embedding

### 3.1 研究动机

当前 BGE POI embedding 主要由以下内容构建：

```text
POI 名称 + 地址 + 别名
```

但真实用户检索还包含大量 POI 原始文本中不突出或不存在的信息，例如：

- 用户常用简称；
- 俗称和别名；
- 商场、小区、园区等父实体；
- 楼号、单元号、商铺号；
- 用户更关注的功能和场景；
- 真实检索中常用的地址表达方式。

例如：

```text
POI：北京南站

正向 Query：
北京南站
北京南站进站口
南站停车场
北京南火车站
北京南站出发层
```

这些正向 Query 可以作为 POI 的第二个语义视角，从而补充原始 POI 文本表示。

### 3.2 基本方法

对于每个 POI $i$，从训练集中收集其所有正向 Query：

$$
Q_i = \left\{q_1,q_2,\ldots,q_n\right\}
$$

使用固定的 BGE 模型编码 Query：

$$
e_q = \operatorname{Encoder}_{\mathrm{BGE}}(q)
$$

然后对该 POI 的 Query embedding 进行加权聚合：

$$
\bar{q}_i =
\frac{\sum_{q\in Q_i} w(q,i)e_q}
{\sum_{q\in Q_i} w(q,i)}
$$

最后与原始 BGE POI embedding $c_i$ 融合：

$$
z_i = \operatorname{Normalize}\left((1-\alpha_i)c_i+\alpha_i\bar{q}_i\right)
$$

当前阶段不进行以下操作：

- 不更新 BGE 参数；
- 不构造困难负样本；
- 不进行 Query–POI 对比学习；
- 不进行 RL；
- 不修改评测 Query embedding；
- 只利用训练集正向 Query 对 POI embedding 做离线增强。

### 3.3 泛 Query 降权

用户 Query 的质量和粒度并不稳定。

例如：

```text
奶茶
附近
吃饭
停车
```

这些 Query 可能关联大量不同 POI，对具体 POI 的区分价值较低。

而以下 Query 通常具有更强的 POI 指向性：

```text
百子湾东里409号楼
凯德Mall大峡谷店
北京南站进站口
合生汇B1层奶茶
```

因此，可以只利用正向 Query–POI 关系计算 Query 权重：

$$
w(q,i) =
\log\left(1+\operatorname{count}(q,i)\right)
\cdot
\log\frac{N+1}{\operatorname{df}(q)+1}
$$

其中：

- $\operatorname{count}(q,i)$ 表示 Query $q$ 对 POI $i$ 的发单次数；
- $\operatorname{df}(q)$ 表示 Query $q$ 正向关联过的不同 POI 数量；
- $N$ 表示具有训练 Query 的 POI 总数。

该权重具有以下作用：

- 高频重复发单 Query 有更高可信度；
- 同一个 Query 关联大量 POI 时自动降权；
- 明确指向少量 POI 的 Query 权重更高；
- 不需要人为构造任何负样本。

### 3.4 Coverage-Aware 融合

全量目录包含 2,337,178 个 POI，但只有约 520,333 个 POI 成为过训练目标。

因此，不能让 Query embedding 完全主导 POI 表示，否则会造成：

- 有历史 Query 和无历史 Query 的 POI 分布不一致；
- 头部 POI 和长尾 POI 表示不一致；
- 新增 POI 无法构建相同质量的表示；
- SID 随日志变化而大幅漂移。

可以为每个 POI 计算 Query 可靠度：

$$
\rho_i =
\frac{n_i}{n_i+\tau}
\cdot
\operatorname{Consistency}(Q_i)
$$

其中：

- $n_i$ 为该 POI 的有效训练 Query 数量；
- $\tau$ 为平滑参数；
- $\operatorname{Consistency}(Q_i)$ 衡量该 POI 的历史 Query embedding 是否集中。

融合权重定义为：

$$
\alpha_i = \alpha_{\max}\rho_i
$$

最终效果为：

```text
无历史 Query：
z_i = c_i

Query 数量少或语义不一致：
z_i ≈ c_i

Query 数量充分且语义一致：
z_i = Content Embedding + Query Augmentation
```

### 3.5 Query Residual

简单平均 Query embedding 可能引入大量公共查询方向，例如：

```text
北京
附近
门店
导航
奶茶
餐厅
```

为了突出 POI 特有查询语义，可以减去同类别 POI 的公共 Query 均值：

$$
r_i^q = \bar{q}_i-\mu_{\operatorname{category}(i)}
$$

再进行融合：

$$
z_i = \operatorname{Normalize}\left(c_i+\alpha_i r_i^q\right)
$$

例如，同一类别奶茶店的公共语义可能包括：

```text
奶茶、饮品、附近、奶茶店
```

而某个具体 POI 的特有 Query 语义可能包括：

```text
合生汇店
九龙山店
B1层
```

减去类别公共 Query 均值后，保留下来的主要是具体 POI 的特有检索语义。

### 3.6 创新性判断

以下方法本身创新性较弱：

```text
POI Embedding + 所有历史 Query 的简单平均
```

因为已有工作已经使用 Query–Item 对齐或行为语义增强。

更有价值的设计是：

> Coverage-Aware Positive Query Augmentation

其主要特点包括：

1. 只使用正向 Query–POI 发单关系；
2. 不依赖困难负样本和额外模型训练；
3. 根据 Query 关联 POI 数自动抑制泛 Query；
4. 根据 Query 数量和一致性动态控制融合强度；
5. 使用类别 Query 残差保留 POI 特有检索语义；
6. 对无历史 Query 的长尾 POI 自动退化为原始内容 embedding；
7. 直接服务于后续 SID 构建。

该模块适合作为论文中的 Embedding 支持创新，但不建议单独作为整篇论文的唯一核心贡献。

---

## 4. 创新点二：碰撞触发的最小关系 SID

### 4.1 研究动机

TIGER 当前使用三层基础 SID：

```text
<S1><S2><S3>
```

当多个 POI 具有相同基础 SID 时，追加随机 Collision Token：

```text
<S1><S2><S3><C0>
<S1><S2><S3><C1>
<S1><S2><S3><C2>
```

这种方式可以保证最终标识唯一，但存在以下问题：

- `C0/C1/C2` 没有真实语义；
- 大模型无法从 Query 中推断随机编号；
- 同一碰撞桶内的楼号、单元、商铺和父实体信息被浪费；
- 随机编号只解决唯一性，不解决可生成性和可解释性；
- 新增或删除 POI 后，碰撞编号可能发生变化。

### 4.2 基本方法

基础 SID 构建完成后，对发生碰撞的桶进行局部分析：

```text
基础 SID
   ↓
是否碰撞？
   ├── 否：直接结束
   └── 是：构建局部关系图
                 ↓
          选择最短区分路径
                 ↓
          构建变长关系后缀
```

普通无碰撞 POI：

```text
<S1><S2><S3><EOS>
```

通过地理区域区分：

```text
<S1><S2><S3><DISAMB_GEO><G1><G2><EOS>
```

通过楼号区分：

```text
<S1><S2><S3><BUILDING_NO><N409><EOS>
```

通过楼号和单元区分：

```text
<S1><S2><S3><BUILDING_NO><N5><UNIT><N3><EOS>
```

通过商铺号和楼层区分：

```text
<S1><S2><S3><SHOP_NO><N258><FLOOR><B3><EOS>
```

### 4.3 局部关系图

只对发生碰撞的 POI 构建局部关系，不对全量 233 万 POI 建立完整知识图谱。

局部关系可以包括：

```text
POI
├── located_in → 行政区
├── located_in → GID 网格
├── located_on → 道路
├── child_of → 小区
├── child_of → 商场
├── child_of → 园区
├── has_building_no → 楼号
├── has_unit_no → 单元号
├── has_floor → 楼层
├── has_room_no → 房间号
├── has_shop_no → 商铺号
├── has_entrance → 出入口
└── alias_of → 规范实体
```

局部图的优势包括：

- 构建范围小；
- 可以流式处理碰撞桶；
- 新增 POI 时只更新局部桶；
- 不需要部署完整知识图谱系统；
- 能够利用地图 POI 已有的结构化地址和父子关系。

### 4.4 最短关系路径选择

关系路径不依赖某一条具体历史 Query，而只依赖稳定的 POI 属性。

可以定义：

$$
\operatorname{Score}(p) =
\operatorname{Discrimination}(p)
-\lambda\operatorname{Length}(p)
+\mu\operatorname{TokenReuse}(p)
$$

其中：

- $\operatorname{Discrimination}(p)$ 表示路径对碰撞桶 POI 的区分能力；
- $\operatorname{Length}(p)$ 表示新增 Token 数量；
- $\operatorname{TokenReuse}(p)$ 表示这些关系 Token 能否在大量 POI 中共享。

选择目标为：

> 在保证不同实体最终唯一的条件下，寻找总代价最小的关系路径集合。

具体规则如下：

1. GID 能完全区分且路径较短时，使用 GID；
2. GID 只能区分部分 POI 时，可以继续追加局部关系；
3. GID 在当前碰撞桶中没有信息增益时，跳过 GID；
4. 同一区域、同一小区内优先使用楼号、单元号等关系；
5. 同一商场内优先使用楼层、商铺号等关系；
6. 真实重复或别名实体进行 canonicalize；
7. 元数据完全相同但确实属于不同实体时，最后才使用 arbitrary dedup code。

### 4.5 创新性判断

以下单独都不足以构成强创新：

- 变长 SID；
- 碰撞后追加 Token；
- 使用 GID；
- 使用局部图；
- 随机 Dedup；
- RQ-OPQ。

真正具有创新潜力的是：

> 对有害碰撞桶构建局部异构关系，并自动编译出最短、共享、可解释的实体消歧路径。

核心不是“使用知识图谱”，而是：

1. 只处理有害碰撞；
2. 只构建碰撞桶局部关系；
3. 自动选择最短区分路径；
4. 使用可共享的关系类型和数值 Token；
5. 将 arbitrary dedup 降为最后兜底手段；
6. 支持局部增量更新。

---

## 5. 创新点三：共享语义与实体身份解耦

### 5.1 核心问题

传统 SID 通常要求同一组固定长度 Token 同时承担：

1. 将语义相似 POI 聚集在一起；
2. 唯一标识每一个具体 POI。

这两个目标存在天然冲突。

例如：

```text
百子湾东里209号楼
百子湾东里409号楼
百子湾东里115号楼
```

从共享语义角度，它们应该具有相同或相近的 SID 前缀。

从实体身份角度，它们又必须能够被唯一识别。

如果在基础 embedding 中强行把它们完全分开，就会破坏共享语义结构；如果只保留共享语义，就会发生 SID 碰撞。

### 5.2 解耦方法

拟采用以下分工：

```text
Query-Augmented Embedding
        ↓
学习共享检索语义
        ↓
RQ-KMeans 基础 SID

碰撞桶局部关系
        ↓
保存实体区分信息
        ↓
关系语义后缀
```

即：

- 基础 SID 负责表达“它是什么”；
- 关系后缀负责表达“它具体是哪一个”。

例如：

```text
共享语义：
<S_RESIDENTIAL><S_BAIZIWAN><S_BUILDING>

实体身份：
<BUILDING_NO><N409>
```

最终：

```text
<S_RESIDENTIAL>
<S_BAIZIWAN>
<S_BUILDING>
<BUILDING_NO>
<N409>
```

### 5.3 论文核心观点

可以将论文的核心观点总结为：

> 现有 Semantic ID 将语义组织与实体唯一性混合在同一个固定长度量化空间中。对于地图 POI，这会导致共享语义与细粒度实体身份发生冲突。本文通过 Query 增强的基础语义 SID 与碰撞触发的关系后缀，将两类信息分开编码。

这是目前最值得作为论文核心主线的创新。

---

## 6. 空间模块的定位

后续可以为所有 POI 保留：

```text
GID
经纬度
行政区
空间层次
```

并在 SFT 模型中增加轻量空间意图 Head，完成：

- 用户位置中心识别；
- Query 地点中心识别；
- 附近、行政区、跨域等空间意图识别；
- 语义候选与空间候选合并；
- 软距离重排。

但空间模块不建议在当前阶段同时展开。

原因包括：

- 当前首先需要验证 Embedding 和 SID 两个核心假设；
- 同时加入空间模块会引入过多实验变量；
- 难以判断收益来自 Embedding、SID 还是空间排序；
- 论文主线容易变得松散。

空间模块可以作为：

1. 当前论文的后续扩展；
2. 上线系统优化；
3. 后续独立研究方向。

---

## 7. 创新强度评估

| 模块 | 单独创新性 | 建议定位 |
|---|---:|---|
| RQ-KMeans | 弱 | 必要基线 |
| RQ-OPQ | 弱 | 对照方法 |
| 正向 Query 简单平均 | 弱 | Embedding 消融 |
| Query TF-IDF 加权 | 中弱 | Embedding 消融 |
| Coverage-Aware Query Fusion | 中 | Embedding 支持创新 |
| Category Query Residual | 中 | Embedding 支持创新 |
| 变长 SID | 弱 | 方法表现形式 |
| 碰撞后追加 GID | 弱 | 对照实验 |
| 碰撞桶局部关系图 | 中 | 方法基础 |
| 最短关系路径编译 | 中强 | SID 核心创新 |
| 共享语义与实体身份解耦 | 强潜力 | 论文核心观点 |
| 轻量空间意图 Head | 中 | 后续扩展 |

---

## 8. 后续实验规划

### 8.1 阶段一：冻结 Embedding 评测协议

使用 SFT 已固定的同一组 10,000 条 Validation 样本作为 Embedding 评测集。

训练侧不依赖 GPU 的数据准备已经完成：7,586,410 条正向样本按原始 Query 的 BLAKE2b-64 哈希写入 256 个 Parquet 分片；进一步逐分片生成 1,408,778 个唯一 Query、2,063,175 个唯一 Query–POI 对及 491,213 个 POI 的 Train-only `count/df/coverage` 统计。聚合耗时 174.63 秒、峰值 RSS 221.80 MiB，全部计数守恒和文件 SHA256 复核通过；尚未编码 Train Query 或生成增强向量。

已完成：

1. 校验固定评测集与训练集在 `order_id/searchid` 上不存在重叠；
2. 校验 Train 与 Validation 时间切分；
3. 固定这 10,000 条 Query 的 BGE embedding；
4. 后续所有实验复用同一份 Query embedding；
5. 只替换全量 POI embedding；
6. 重新运行纯 BGE POI embedding 基线。

冻结的 BGE Query 向量 SHA256 为 `98e2032b86b76cf42991dc1d1f4967ff456661fb32fbae185652163c85bf70e0`，行映射 SHA256 为 `d512aa206468ea672b473ce38c353eca703c64289ac890d9a935cd3c9b2f855f`。E0 原始 BGE-M3 的 Hit@10/20 为 32.39%/40.25%，完整记录见 `docs/experiments/V1.md`；后续 E1—E4 统一记录到 `docs/experiments/EMBEDDING_OPTIMIZATION.md`。

统一指标：

```text
HR@1
HR@5
HR@10
HR@20
MRR@10
NDCG@10
```

同时按以下维度分组：

```text
训练 Query 数量 1–5
训练 Query 数量 6–20
训练 Query 数量大于 20
头部 POI
长尾 POI
```

该阶段的主要目标是建立无泄漏、固定、可复现的向量检索基线。

### 8.2 阶段二：正向 Query 增强 Embedding

在 Query embedding、评测集和 Faiss 参数全部固定的情况下，对比：

```text
E0：原始 BGE POI Embedding

E1：BGE + 正向 Query 简单均值

E2：BGE + Query TF-IDF 加权均值

E3：BGE + Coverage-Aware Query Fusion

E4：BGE + Coverage-Aware Category Query Residual
```

实验中只使用训练集 Query–POI 正样本，不构造困难负样本。

需要重点观察：

- 总体向量检索指标是否提升；
- 提升是否只来自头部 POI；
- Query 较少的 POI 是否稳定；
- 不同融合系数对结果的影响；
- POI embedding 分布是否出现异常聚集；
- Query 增强后是否出现明显的类别或地域偏置。

### 8.3 阶段三：比较量化器

选择阶段二效果最好的 POI embedding，统一比较：

```text
RQ-VAE
RQ-KMeans
RQ-KMeans + 最后一层 Balance
可选 RQ-OPQ
```

固定以下条件：

- 相同 POI embedding；
- 相同训练样本；
- 相同随机种子；
- 相同层数；
- 可比较的码本容量；
- 相同全量 SID 评测协议。

评测指标包括：

```text
码本利用率
每层 Token 熵
前缀类别纯度
完整 SID 唯一率
碰撞桶数量
碰撞 POI 数量
最大碰撞桶
重构误差
后续固定 10k 检索效果
```

RQ-KMeans 本身不是创新，但需要验证它在北京全量 POI 数据上是否比 RQ-VAE 更稳定、更适合大规模 SID 构建。

### 8.4 阶段四：验证关系 SID

在固定的基础 SID 上对比：

```text
随机 Collision Token
所有 POI 固定追加 GID
仅碰撞 POI 追加 GID
RQ-OPQ 唯一残差
碰撞桶局部最小关系后缀
```

按碰撞类型分别评测：

```text
同名不同分店
同小区不同楼号
同楼不同单元
同商场不同商铺
不同地理区域
同一地理区域
元数据完全相同
真实重复或别名
```

评测指标包括：

```text
基础 SID 唯一率
最终标识唯一率
Arbitrary Dedup 使用率
关系后缀平均长度
关系后缀 P95 长度
关系 Token 覆盖率
碰撞子集 HR@1
碰撞子集 HR@10
全量固定 10k 检索指标
Trie 节点数量
约束解码时延
增量 POI 更新成本
```

### 8.5 阶段五：决定是否加入空间模块

如果前四阶段已经证明：

- Query 增强可以稳定改善基础 embedding；
- RQ-KMeans 能构建更好的基础 SID；
- 局部关系后缀优于随机 Dedup 和固定 GID；

则可以先完成 SID 论文闭环。

空间模块可以根据论文篇幅和实验结果决定：

- 作为附加模块；
- 作为上线优化；
- 作为后续独立论文方向。

---

## 9. 关键风险

### 9.1 Query 覆盖率不足

只有约 22.26% 的全量 POI 成为过目标。

应对方法：

- 原始 BGE embedding 始终作为主干；
- Query 只作为残差增强；
- 使用 coverage-aware 融合；
- 对无 Query POI 单独报告指标；
- 避免直接使用 Query embedding 替换 POI embedding。

### 9.2 头部 POI 偏置

高频 POI 有大量 Query，容易获得更强增强。

应对方法：

- 对发单次数使用对数压缩；
- 对 Query 数量设置上限；
- 使用 Query 一致性而不是单纯数量；
- 按头部和长尾分组评测。

### 9.3 泛 Query 污染

同一个泛 Query 可能关联大量 POI。

应对方法：

- 使用 Query–POI 正向图上的 IDF；
- 使用类别 Query 残差；
- 对关联 POI 数量过多的 Query 降权；
- 比较简单均值和加权方法的差异。

### 9.4 SID 依赖行为日志

如果 Query 直接决定 SID，新增日志可能导致 SID 频繁变化。

应对方法：

- Query 只增强基础 embedding；
- 基础内容向量始终保留；
- 局部关系后缀只依赖稳定 POI 属性；
- 对 embedding 版本进行冻结和版本化；
- 增量更新时设置变化阈值。

### 9.5 局部关系退化为规则工程

如果只手工规定“先 GID，再楼号”，创新性不足。

应对方法：

- 将关系路径选择形式化为优化问题；
- 统一计算区分能力、路径长度和 Token 复用率；
- 提供固定算法而不是 Case-by-Case 规则；
- 与随机 Dedup、固定 GID、RQ-OPQ 做系统对照；
- 报告局部更新成本和标识长度。

---

## 10. 推荐论文方向

暂定方法名称：

> Query-Augmented Relational Semantic IDs for Generative POI Retrieval

可以概括为三个贡献点：

1. 提出只使用正向查询日志、兼容长尾目录的 Coverage-Aware POI 表征增强方法；
2. 揭示地图 POI Semantic ID 中共享语义与实体唯一身份之间的目标冲突；
3. 提出碰撞触发的局部关系编译方法，以最短、可解释的关系后缀替代随机 Collision Token。

核心论文叙事为：

> 正向 Query 增强负责让基础 SID 符合真实用户检索语义，局部关系后缀负责保留量化过程中丢失的实体身份信息。二者共同实现从“语义相似”到“实体唯一”的分层 POI 标识。

---

## 11. 当前推荐优先级

### 最高优先级

1. 已使用冻结的 BGE-M3 对 1,408,778 个唯一 Train Query 完成一次编码并通过全量复检；
2. 实现正向 Query 简单均值和 TF-IDF 加权；
3. 在固定 BGE 评测 Query 向量上验证 Query 增强是否带来稳定收益。

### 第二优先级

1. 实现 Coverage-Aware Query Fusion；
2. 实现 Category Query Residual；
3. 分析有 Query 和无 Query POI 的效果；
4. 选择最佳 POI embedding。

### 第三优先级

1. 在最佳 embedding 上运行 RQ-KMeans；
2. 与现有 RQ-VAE 做公平比较；
3. 分析码本利用、语义纯度和碰撞结构。

### 第四优先级

1. 构建碰撞桶局部关系；
2. 实现最短关系路径选择；
3. 与随机 Dedup、GID 和 RQ-OPQ 对比；
4. 完成关系 SID 的下游 SFT 和固定 10k 评测。

### 暂缓

1. 空间意图 Head；
2. 路线和多中心空间查询；
3. RL/GRPO/DPO；
4. 困难负样本训练；
5. 完整全量知识图谱。

---

## 12. 当前状态

目前已完成无泄漏评测集、BGE 评测 Query 向量、E0 原始 BGE 精确召回、Train Query 哈希分片、频次/DF、POI 覆盖统计及评测分组；尚未计算 Train Query embedding 或增强后的 POI 向量。

后续正式开始时，应从以下最小闭环进入：

```text
唯一 Train Query 的 BGE-M3 编码
        ↓
实现正向 Query 简单聚合
        ↓
在同一评测集上比较向量检索指标
```

只有在 Query 增强向量表示得到稳定收益后，再继续推进 RQ-KMeans 和碰撞关系 SID。
