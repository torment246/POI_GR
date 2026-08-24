# QGR-SID：Query 引导的纯离散变长关系 SID

更新时间：2026-08-14。

当前状态：M1 与 M2-A 已完成；纯词法关系代理相对桶内流行度 HR@1 提高 1.7405 个百分点。M2-B 精确 GID 与 M2-C Query residual 均显著负向；M2-D 可靠词法仍正向但只保留 full lexical 82.47% 的 HR@1 增量、92.77% 的 early 碰撞请求仍需 fallback，未通过预注册门禁。Query 引导路线已停止；后续不使用 Query 构码的“粗地理→关系→细地理”重构独立记录在 [GHR-SID](GHR_SID.md)，不回写或调参本方法的 holdout 结果。

## 1. 方法定位

QGR-SID（Query-Guided Relational Semantic ID）面向地图 POI 生成式检索中的热点碰撞问题。方法不重新训练基础量化器，第一轮固定使用当前 TIGER/BGE-M3/RQ-VAE `1024×3` 的三层语义 SID：

```text
<S1_i><S2_j><S3_k>
```

基础 SID 负责表达“它是什么”，只在同一基础 SID 对应多个 POI 时，追加由稳定 POI 属性决定的最短关系后缀，表达“它具体是哪一个”。Train Query 与请求 GID 只用于估计某种关系是否能从真实检索请求中预测，不允许某一条请求动态改变 POI 标识。

第一轮只替换 TIGER 当前按 `poi_id` 字典序分配的随机 `<C_k>`，不改变以下变量：

- Qwen3-0.6B 基础模型；
- Train/Valid/Test 时间切分与样本顺序；
- 用户 ID、最近 10 条历史、当前 Query 和请求 Geohash6 GID；
- 三轮全参数 SFT 超参数；
- 固定普通 Validation 10,000 条业务键与 Beam=10 主评测；
- 无约束生成中非法 identifier 保留原 Beam 排名并计为 miss 的口径。

第一轮不在 E4 SID 上开发。只有 QGR-SID 在 TIGER 上超过 TIGER 随机 collision token 后，才迁移到 E4 RQ-KMeans `2048×1024×512`，避免基础 SID 和碰撞方法同时变化。

## 2. 现有证据与核心假设

TIGER `1024×3` 当前全量结果为：

| 指标 | 数值 |
|---|---:|
| 全量 POI | 2,337,178 |
| 基础 SID 唯一率 | 71.6766% |
| 位于碰撞桶的 POI | 953,321（40.7894%） |
| 最大碰撞桶 | 306 |
| 固定 10k 目标位于碰撞桶 | 38.54% |
| 固定 10k 中 `C>0` 目标 | 23.56% |
| `C>0` 条件 collision token Teacher-Forcing Top-1 | 81.03% |
| epoch 3 HR@1 / HR@10 / NDCG@10 | 51.87% / 87.16% / 70.4190% |
| epoch 3 Valid ID Rate | 74.105% |

现有热点碰撞桶包含大量具有明确数字关系的 POI，例如：

- 同一小区的不同分区、楼号和单元；
- 同一商场的不同楼层、室号和商铺号；
- 同一道路或小区内的不同门牌号；
- 同一设施的不同编号入口；
- 跨区域同名分店或同类设施。

核心假设为：

> 随机 `<C_k>` 只保证唯一性，无法由 Query 和请求位置稳定推断；如果把同桶 POI 编译成短的“关系类型 + 规范化数字”路径，并用 Train Query/GID 优先选择可预测关系，则能提高碰撞后缀准确率，最终提高完整 identifier HR/NDCG。

最终结论只由下游 SFT 评测决定。关系覆盖率、唯一率、平均长度或可解释性都只是训练前门禁，不能单独算作方法提升。

## 3. 纯离散数值序列化

### 3.1 冻结原则

QGR-SID 的 Assistant Target 不包含 POI 名称、道路、小区、商场、`号楼`、`单元`、`层`、`室`、`店`等自然语言文本。关系类型和关系值都编码为新增的原子离散 token。

标识语法为：

```text
identifier := S1 S2 S3 [relation_pair] [relation_pair] [relation_pair]
relation_pair := relation_type numeric_value
```

约束如下：

1. 无碰撞 POI 在 `S3` 后直接结束，不再追加与碰撞 POI共用的 `C0`；
2. 碰撞 POI 后缀最多包含 3 个“关系—数值”对；
3. 残差兜底若出现，必须是最后一对，并计入最多 3 对的限制；
4. 如果两层真实关系后仍不能保证唯一，第三对直接使用残差原型，不继续无限增加路径；
5. 只有第三个真实关系能直接完成唯一化时，才允许使用三个真实关系对；
6. 最长 identifier 为 3 个语义 token + 6 个关系 token；加上 `<TARGET_POI>`、`</TARGET_POI>` 和 EOS，最长生成长度为 12；
7. 每个 POI 的最终标识必须静态、确定、全局唯一，并可由同一冻结输入复建出相同 SHA256。

### 3.2 序列示例

无碰撞 POI：

```text
<TARGET_POI><S1_465><S2_543><S3_609></TARGET_POI>
```

`4区5号楼3单元` 只在离线抽取阶段保留原始解释，进入 SID 后序列化为：

```text
<TARGET_POI>
<S1_465><S2_543><S3_609>
<R_ZONE_NUM><V_4>
<R_BUILDING><V_5>
<R_UNIT><V_3>
</TARGET_POI>
```

`B3层258号铺` 序列化为：

```text
<TARGET_POI>
<S1_438><S2_599><S3_833>
<R_BASEMENT><V_3>
<R_SHOP><V_258>
</TARGET_POI>
```

`204-1号` 作为两个结构化数字分量：

```text
<R_ADDRESS_NO><V_204><R_SUBNO><V_1>
```

关系无法完全区分时：

```text
<S1_i><S2_j><S3_k>
<R_ZONE_NUM><V_4>
<R_RESIDUAL><P_137>
```

上述换行只用于阅读，实际 Messages 中保持连续 token 序列。

### 3.3 第一版关系类型

第一版只输出能够高置信度规范化为数字的关系，不把父实体名、道路名、品牌名或分店名直接写入 SID。

| Token | 关系 | 数值规范化示例 |
|---|---|---|
| `<R_PHASE>` | 期数 | 二期 → `<V_2>` |
| `<R_ZONE_NUM>` | 数字分区 | 4区 → `<V_4>` |
| `<R_ZONE_ALPHA>` | 字母分区 | A区/B区 → `<V_1>`/`<V_2>` |
| `<R_ZONE_DIR>` | 方位分区 | 北/东北/东/东南/南/西南/西/西北/中 → `<V_0>`—`<V_8>` |
| `<R_BUILDING>` | 楼、栋、幢、座编号 | 409号楼 → `<V_409>` |
| `<R_ADDRESS_NO>` | 裸门牌、院号 | 219号/204号 → `<V_219>`/`<V_204>` |
| `<R_QUALIFIER>` | 门牌或楼号前限定符 | 甲/乙/丙 → `<V_1>`/`<V_2>`/`<V_3>` |
| `<R_SUBNO>` | 连字符附加编号 | `204-1` 的 `1` → `<V_1>` |
| `<R_UNIT>` | 单元 | 3单元 → `<V_3>` |
| `<R_FLOOR>` | 地上楼层 | 2层/F2 → `<V_2>` |
| `<R_BASEMENT>` | 地下楼层 | B3/地下三层 → `<V_3>` |
| `<R_ROOM>` | 房间号 | 301室 → `<V_301>` |
| `<R_SHOP>` | 商铺或摊位号 | 258号铺 → `<V_258>` |
| `<R_ENTITY_NO>` | 名称开头的独立数字实体号 | `258(凯德Mall)` → `<V_258>` |
| `<R_ENTRANCE_DIR>` | 入口方位 | 使用与 `<R_ZONE_DIR>` 相同的 0—8 方位码 |
| `<R_ENTRANCE_ALPHA>` | 字母入口 | A口/B口 → `<V_1>`/`<V_2>` |
| `<R_ENTRANCE_NO>` | 入口序号 | 北2门/A2口中的序号 2 → `<V_2>` |
| `<R_GEO>` | 桶内最短地理差异 | 最短 1—2 个 Geohash 差异字符合并成一个 `<V_k>` |
| `<R_RESIDUAL>` | 关系不足时的可预测残差原型 | 后接 `<P_k>` |

父实体、小区名、商场名和道路名仍可以参与离线解析、公共上下文识别和残差特征构造，但第一版不输出对应文本，也不为它们分配桶内随机数字。直接把“凯德Mall”映射成某个局部整数，与当前随机 `C` 本质接近，缺少跨桶复用和 Query 可预测性，暂不采用。

### 3.4 数值与地理 token

第一版候选数值词表暂定为：

```text
<V_0> ... <V_1055>
<P_0> ... <P_511>
```

其中：

- 对非 GEO 关系，`<V_n>` 表示真实规范化整数 `n`，在所有关系类型之间共享，不按桶重新编号；
- 对 GEO 关系，一个差异字符的编码为其 alphabet 下标 `k∈[0,31]`；两个差异字符的合并编码为 `32 + 32×k1 + k2∈[32,1055]`，因此 `<R_GEO>` 后仍然只生成一个 `<V_k>`；
- `<P_k>` 表示第 `k` 个全局残差原型，不是 `poi_id` 字典序；
- 非 GEO 原始数值 0—1055 的上限只是训练前候选，必须由全量碰撞 POI 覆盖审计决定是否足够；
- 超出范围、歧义大或低置信度的数字不拆成自然语言，不在第一版临时扩出长尾 token，直接交给后续关系或残差兜底；
- `<V_n>` 的新增模型 embedding 和 tied LM Head 行优先使用基础 Qwen 中十进制字符串 `str(n)` 的原 token embedding 均值初始化，减少从 Query 数字到离散值 token 的冷启动；关系类型 token 仍按统一可复现规则初始化。

## 4. 稳定关系抽取

### 4.1 输入字段

当前 POI 原始表实际提供：

```text
poi_id, displayname, address, alias, text,
category, category_code, lng, lat, area, city,
layer, click_score, source_dt
```

没有可靠的 `parent_id/building_no/unit_no/floor/shop_no` 结构化列，因此第一阶段必须先实现确定性抽取器，不能假设存在完整知识图谱。

### 4.2 规范化规则

抽取器从 `displayname + address + alias` 生成候选关系，并保留来源和置信度：

1. Unicode、全半角、大小写、空格和连接符统一；
2. 只在明确的关系触发词上下文内转换中文数字，避免把普通名称误转为号码；
3. `5号楼`、`5单元`、`5层`、`5室`、`5号铺`分别映射到不同关系类型；
4. `B3` 与地上 `3层` 分开编码；
5. `204-1号`、`甲219-1号`拆成主体号、附加号和限定符，不拼成不可解释大整数；
6. alias 只用于补充或交叉验证，多个 alias 冲突时降低置信度；
7. 父实体和道路公共片段只用于确认数字属于哪个实体，不直接输出文本；
8. 经纬度只用于 GEO 候选和残差特征，不从坐标反推虚假的楼号关系；
9. 完全相同元数据的多个 `poi_id` 不擅自合并，继续作为不同评测标签并进入残差兜底。

每条抽取结果至少包含：

```text
poi_id
relation_type
numeric_value
source_field
source_span
confidence
normalization_rule
```

### 4.3 使用门槛

一个关系候选进入局部编译器前必须满足：

- 数值位于冻结词表范围；
- 置信度达到统一门槛；
- 在当前碰撞节点至少产生两个非空分支；
- 请求加权信息增益为正；
- 缺失值不能被当作高质量语义分支；
- 同一 POI 对该关系只有一个稳定规范值，或存在明确的复合编号拆解规则；
- 关系不能仅依赖 Validation/Test 中出现的信息。

## 5. Query 的使用方式

### 5.1 不让请求改变 SID

QGR-SID 的最终映射是 POI 级静态映射。任意 Query 指向同一 POI 时，目标 identifier 完全相同。Query 只用于离线回答：

> 在多个都能静态区分 POI 的关系中，哪一个最容易由真实检索请求预测？

SID 构建严格只使用 Train 订单。Validation/Test Query、目标和请求位置不参与关系选择、参数选择、原型训练或分配。

### 5.2 复用现有 Train-only Query 资产

直接复用现有 E2 资产：

- 1,408,778 个唯一 Train Query 的 BGE-M3 embedding；
- 2,063,175 个唯一 Query–POI 对的 `count(q,i)` 与 `df(q)`；
- 491,213 个有 Train Query 的 POI 的 IDF 加权 Query 聚合 `qbar_i`；
- 每个 POI 的 Train 订单数、唯一 Query 数和聚合可靠度。

不重新编码相同 Query，不使用 E4 最终融合向量作为关系特征。

### 5.3 关系分支可预测性

先按时间把 Train 划成前 90% 和后 10%：

- 前 90% 构建 Query 统计、关系分支中心和轻量全局分支预测器；
- 后 10% 只做训练内代理评测，不回流选择后的统计；
- 方法冻结后，再使用完整 Train 重建最终关系树和 SID。

对碰撞树节点 `U`、候选关系 `r` 和数值分支 `v`，分支预测特征包括：

1. Query 是否抽取出相同类型与数值；
2. Query 是否包含边界明确的相同数字；
3. Query embedding 与该分支 Train Query 中心的 cosine；
4. Query embedding 相对当前节点公共 Query 中心的残差 cosine；
5. 请求 GID 与该分支 POI/Train 请求地理中心的一致性；
6. POI Query 覆盖、有效 Query 数和分支支持度 mask。

预测器只需按关系类型训练一个全局轻量模型，不为每个碰撞桶单独拟合。它只输出该关系在后 10% Train 上的分支 NLL、Top-1 和覆盖率，不参加在线推理。

POI/请求权重采用平方根压缩并在 P99 截断：

$$
w_i=\sqrt{1+\min\left(n_i,P99(\{n\})\right)}
$$

避免纯 POI 等权指标忽略热点碰撞，也避免极少数超级头部完全支配关系树。

## 6. 地理信息的引入

GID 需要引入，但只作为可选关系和可预测性信号：

1. 保留现有 Prompt 中的当前请求 `<USER_GID>`，不增加新输入；
2. 不把完整目标 POI GID 放进 Prompt；
3. 不给所有 POI 固定生成六个目标 GID token；
4. 对碰撞节点先求静态 POI Geohash6 的公共前缀；
5. 只取公共前缀之后、能够区分当前节点的最短 1—2 个字符，并按固定公式合并编码为一个 `<R_GEO><V_k>` 关系对；
6. Geohash 字符先按固定 alphabet 映射为 0—31，两个字符再合并成 32—1055 的单一整数，目标 SID 不出现字母；
7. 同时统计目标 POI 的 Train 请求 GID 稳健中心；无 Train 请求覆盖时回退为静态 POI 坐标；
8. 如果 GEO 在 Train 后 10% 上的分支可预测性低，或者同商场/同楼内没有信息增益，编译器自动跳过 GEO；
9. 第一轮不加入 SSP、地理前缀预填或距离裁剪，以免把 SID 收益和约束解码收益混在一起。

## 7. 碰撞桶局部关系树编译

### 7.1 编译单位

只对基础 `S1/S2/S3` 碰撞桶构建局部树，不对 233 万 POI 构建全局知识图谱。每个节点保存当前未区分 POI 集、已用关系、累计长度和剩余后缀预算。

### 7.2 候选关系评分

候选关系 `r` 在节点 `U` 上的代价定义为：

$$
J(r\mid U)=
\operatorname{NLL}_{proxy}(r\mid U)
-\lambda_{gain}\operatorname{InfoGain}_w(r\mid U)
+\lambda_{miss}\operatorname{MissingRate}_w(r\mid U)
+\lambda_{len}\operatorname{TokenCost}(r)
$$

树级目标为：

$$
J(T)=
\operatorname{NLL}_{proxy}(T)
+\lambda_{len}\mathbb{E}_w[|R_i|]
+\lambda_{fallback}\frac{W_{fallback}}{W}
$$

这里的信息增益和长度只负责约束结构，`NLL_proxy` 衡量真实 Query/GID 能否生成正确关系分支。超参数只允许在 Train 90/10 代理集上冻结，不使用正式 Validation。

### 7.3 搜索过程

每个碰撞桶执行深度不超过 3、beam size 8 的确定性树搜索：

```text
输入：同一 S1/S2/S3 下的 POI 集 U

1. 枚举当前节点的高置信数字关系与 GEO 关系；
2. 过滤无信息增益、高缺失、高冲突和低代理可预测性候选；
3. 为候选计算请求加权代价并保留最优若干棵部分树；
4. 某分支只有一个 POI时立即结束，不继续增加后缀；
5. 某分支仍碰撞且尚有真实关系预算时继续递归；
6. 两个真实关系后仍碰撞时，第三对必须使用残差原型；
7. 若第三个真实关系可直接使所有叶子唯一，可不用残差；
8. 选择满足全局唯一约束的最小总代价树；
9. 所有并列情况使用固定关系优先级、规范值和 poi_id 依次打破，poi_id 只作完全并列的确定性 tie-break，不决定语义编号。
```

同一节点共享关系类型，不同节点可以选择不同关系顺序。例如：

```text
楼栋桶：ZONE_NUM → BUILDING → UNIT
商场桶：BASEMENT/FLOOR → SHOP
门牌桶：BUILDING → SUBNO
跨区域同名桶：GEO
关系不足桶：某个数字关系 → RESIDUAL
```

## 8. 关系不足时的残差原型

### 8.1 桶内残差签名

对真实关系仍无法唯一化的叶子，构造静态、多视角的桶内残差签名：

$$
h_i=\operatorname{Normalize}
\left[
\operatorname{PCA}_{32}(r_i^q);
\operatorname{PCA}_{32}(r_i^c);
r_i^{poi\_geo};
r_i^{request\_geo};
mask_i
\right]
$$

其中：

$$
r_i^q=\operatorname{Normalize}(\bar q_i-\bar q_U)
$$

$$
r_i^c=\operatorname{Normalize}(c_i-\bar c_U)
$$

- `qbar_i` 为现有 Train-only E2 Query 聚合；
- `c_i` 为原始 BGE-M3 POI 内容向量；
- `poi_geo` 为 POI 静态坐标相对叶子中心的位置；
- `request_geo` 为 Train 请求 GID 稳健中心相对叶子中心的位置；
- 无 Query/请求地理覆盖的视角置零，并通过 mask 明确标记；
- 各有效视角先独立归一化，再等权拼接，第一轮不为融合权重消耗 SFT 试验预算。

### 8.2 全局原型与桶内唯一分配

在所有需要兜底的 POI 上学习 512 个全局 spherical K-Means 原型。对每个未解决叶子计算：

$$
D_{ik}=1-\cos(h_i,p_k)
$$

再用矩形 Hungarian matching 为叶子内 POI 分配互不重复的 `<P_k>`。TIGER 最大基础桶为 306，因此 512 个原型具备足够的一对一容量。

这个设计与随机 `C` 的区别是：

- `<P_k>` 在不同桶之间复用近似一致的多视角残差方向；
- 分配首先最小化 POI 与原型的语义/Query/地理距离；
- `poi_id` 仅在签名和代价完全相同时保证复建稳定；
- 需要单独报告残差使用率，不能靠大量 `<P_k>` 掩盖关系抽取失败。

## 9. 与 E4 的关系

E4 与 QGR-SID 可以使用相同的 Train Query 原始资产，但作用层次不同：

| 模块 | Query 残差参照 | 影响位置 | 目标 |
|---|---|---|---|
| E4 | 类别公共 Query 中心 | 全库 POI embedding 与 S1/S2/S3 | 跨桶语义组织 |
| QGR-SID | 当前碰撞节点/叶子 Query 中心 | 仅变长关系后缀和残差原型 | 桶内实体识别 |

E4 使用：

$$
\bar q_i-0.85\mu_{category(i)}
$$

QGR-SID 使用：

$$
\bar q_i-\bar q_{collision\ node(i)}
$$

后续迁移到 E4 时必须：

1. 选用当前下游更强的 E4 RQ-KMeans `2048×1024×512`，不再迁移到已被下游结果支配的 E4 `1024³`；
2. 根据 E4 的新碰撞桶重新抽取候选、训练节点统计并编译关系树；
3. 不把 E4 最终融合向量输入 QGR-SID 残差模块，继续使用原始 BGE 内容向量和 E2 Query 聚合；
4. 保持关系类型、数值词表、代理协议和 SFT 超参数不变，检验方法能否跨基础 SID 迁移。

## 10. SFT 数据与评测协议

### 10.1 Messages

User Prompt 继续沿用 TIGER：

```text
USER_ID
HISTORY:
  request GID + Query + history POI identifier
CURRENT:
  request GID + Query
```

只替换历史 POI identifier 和 Assistant Target，不加入：

- 目标 POI 名称、地址或别名；
- 目标 POI GID；
- 父实体文本或关系提示；
- 关系抽取标签；
- 距离标签或额外空间意图 Head。

训练继续使用 Qwen3-0.6B、Assistant-only CE、相同三轮 SFT 和相同 global batch。变长 identifier 会增加部分历史和目标长度，因此构建前必须报告 cutoff 512 下的 Prompt/Target 截断变化；不能为本方法单独增大 cutoff 后直接与 TIGER 比较。

### 10.2 变长索引与无约束主评测

当前 TIGER 四列 NPY、四层 pack key 和固定 7-token 评测器不兼容 QGR-SID，需要独立实现：

- 变长 token sequence 存储；
- `poi_id → identifier` 与 `identifier → poi_id` 全局双向唯一映射；
- 支持中间节点提前 `</TARGET_POI>` 的变长 Trie；
- 变长目标严格解析、合法性检查和错误类型统计；
- 最大生成长度 12；
- 按真实序列长度计算 missing EOS、无效结构和不在 corpus 的路径。

主结果仍使用无约束 Beam=10、`length_penalty=1.0`，非法路径保留原排名并计 miss。Trie 约束结果只作为合法路径上限诊断，不能替换无约束结果与 TIGER 主表比较。

### 10.3 必报指标

静态与数据指标：

- 最终 identifier 唯一率；
- 关系类型覆盖率、数值覆盖率和字段冲突率；
- 请求加权的纯关系解决率与 residual 使用率；
- 后缀长度 0/2/4/6 token 分布、均值和 P95；
- Trie 节点数、Tokenizer 新增 token 数和 cutoff 截断变化。

Teacher-Forcing 指标：

- S1/S2/S3 Top-1/Top-10 与三级累计准确率；
- 第一个关系类型、每种关系值和 EOS 时机准确率；
- 完整变长 identifier exact match；
- 按 suffix 长度、关系类型、是否 GEO、是否 residual 分组；
- 与 TIGER `C>0` Top-1 81.03% 的碰撞难例对照。

生成检索指标：

- HR@1/3/5/10 与 NDCG@10；
- Valid ID Rate、missing EOS、结构错误和 identifier-not-in-corpus；
- 全量固定 10k、目标位于碰撞桶、原 TIGER `C>0` 三组；
- 关系路径长度分组和头部/长尾目标分组；
- 吞吐、生成 token 数与解码时延。

## 11. 只消耗一次正式 SFT 前的门禁

先完成不需要完整 SFT 的四种 Train 90/10 代理对比：

1. 静态区分度关系树；
2. GEO-only；
3. Query 引导数字关系树；
4. Query + GID + residual 的完整 QGR-SID。

正式构建 SFT 数据前必须同时满足：

- 2,337,178 条最终 identifier 全局唯一；
- 相同输入完整复建 mapping SHA256 一致；
- 无超过 3 个关系对或最长 12 个生成 token 的目标；
- 请求加权纯关系解决率达到 60% 或以上；
- residual 不能成为碰撞请求的主要默认路径；
- 完整 QGR-SID 在 Train 后 10% 的已知 gold bucket 代理 HR@1/NDCG@10 明显优于静态树和 GEO-only；
- 非 GEO 数值 0—1055 对最终选中真实关系值有足够覆盖，越界部分全部有确定性处理；
- cutoff 512 下 Assistant Target 截断为 0，Prompt 保留率相对 TIGER 无明显恶化；
- 变长映射、Tokenizer、Trie、数据格式和评测器的合成测试与小样本 smoke 全部通过。

如果门禁未通过，先修关系抽取、数值范围或编译器，不启动完整 SFT；不通过代理门禁时也不直接回退为“所有碰撞 POI加一个固定关系 token”，因为那无法验证核心假设。

## 12. 正式成败标准

首轮正式实验直接与已经完成的 TIGER epoch 3 比较，不重复训练 TIGER。主要成功标准为：

| 指标 | TIGER 基线 | QGR-SID 要求 |
|---|---:|---:|
| HR@1 | 51.87% | 严格高于 51.87% |
| HR@10 | 87.16% | 不低于 87.16% |
| NDCG@10 | 70.4190% | 严格高于 70.4190% |
| Valid ID Rate | 74.105% | 不低于 74.105% |
| `C>0`/对应碰撞难例 TF Top-1 | 81.03% | 明确提高 |

同时必须满足：

- 收益主要来自碰撞目标，而不是无碰撞短序列单独贡献；
- 长关系路径不出现明显 missing EOS 或重复关系 token；
- GEO 收益主要出现在跨区域碰撞，数字楼栋/商铺关系主要解决同区域碰撞；
- residual 子集不能掩盖真实关系子集下降；
- 如果全量指标未超过 TIGER，即使关系覆盖、唯一率和解释性更好，也记为下游负结果。

## 13. 创新边界

QGR-SID 不把“变长”“追加 GID”“碰撞后追加 token”单独作为创新。与近期方法的边界为：

- OneSearch 的 RQ-OPQ 用额外量化 token 编码残差；
- Purely Semantic Indexing、ZCR 等方法通过重新分配码字减少或消除碰撞；
- FORGE 对碰撞组做后处理分配；
- QuaSID 区分有害与良性碰撞并优化量化间隔；
- SA²CRQ 根据样本重要性调整 SID 长度；
- Gryphon 扩展碰撞候选后做 item-level rerank；
- CRID 把行为或业务排序信息编码到末端 token。

QGR-SID 要验证的组合创新是：

> 由稳定 POI 数字关系提供可区分性，由真实 Train Query 和请求地理信号估计可生成性，再在每个碰撞桶中自动编译请求加权、最短、纯离散、全局唯一的关系路径。

因此论文证据必须包含关系可预测性、局部树编译、残差使用率和端到端 SFT 四部分，不能只展示 SID 唯一率或若干可解释案例。

## 14. 实现分阶段与当前进展

| 阶段 | 内容 | 当前状态 | 正式实验编号 |
|---|---|---|---|
| M0 | 方法选择、TIGER 主基线、变长上限和纯数值序列化冻结 | 已完成方案确认 | 无；未运行实验 |
| M1 | 确定性数字关系抽取器、全量碰撞字段覆盖审计 | 已完成；产物 SHA 复检通过 | `EXP-20260813-05` |
| M2 | Train 90/10 Query/GID 分支预测与局部树编译器 | M2-A 纯词法正向；M2-B/C 负向；M2-D 可靠词法未过完整门禁 | `EXP-20260813-07/08/09/10` |
| M3 | residual 原型、Hungarian 分配、全量唯一变长 mapping 与 Trie | 未开始 | 待定 |
| M4 | Messages、扩词表、初始化、packed Cache 和训练前门禁 | 未开始 | 待定 |
| M5 | 一次 TIGER+QGR-SID 三轮正式 SFT、Teacher-Forcing 和固定 10k 评测 | 未开始 | 待定 |
| M6 | 若 M5 正向，再迁移到 E4 RQ-KMeans `2048×1024×512` | 条件阶段 | 待定 |

当前离线门禁已完成并停止：不构建最终 mapping、Tokenizer 或 SFT。若后续继续，必须先重构关系目标或学习目标，使热点 fallback 比例显著低于当前 92.77%，且在冻结代理上同时保持 full lexical 收益；不能通过降低既定门槛或继续增加静态 GEO/Query residual 覆盖来绕过本轮负证据。

### 14.1 `EXP-20260813-05`：TIGER 全量纯数字关系覆盖审计

#### 目标与输入

本实验只回答 M1 问题：当前 POI `displayname/address/alias` 能为 TIGER 三层 SID 的碰撞 POI 稳定提供多少纯数字关系，以及这些关系覆盖多少 Train 请求权重。它不编译最终关系路径，不声称已解决碰撞，也不使用 Validation/Test。

冻结输入为：

- 原始 POI JSONL 16 个分片，共 2,337,178 行；
- TIGER/BGE-M3/RQ-VAE `1024×3 / epoch 20` 的 `poi_tiger_id_mapping.parquet`，SHA256 为 `7998acbfc5dcd7222370bc258ef52fa2cf79ae22cd43d47758b473a6aa2250a8`；
- Train-only Query 聚合 `covered_poi_rows/train_order_count/unique_query_count`，不读取 Query embedding 本体；
- 数值词表候选固定为 `<V_0>...<V_1055>`；
- 代码基线为 `main@7dd52b3437a9db41834aef638e446010d0df3610` 的 dirty worktree，QGR-SID 新实现尚未提交；
- Python 3.10.20、NumPy 1.26.4，CPU 流式运行，无 GPU。

关系抽取使用 `displayname > address > alias` 的确定性来源优先级。最高优先级若只有一个值则保留；低优先级不同值只记录冲突而不能覆盖；最高优先级本身多值则不输出该关系。M1 新增的 `R_ADDRESS_NO/R_QUALIFIER` 将裸门牌、甲乙丙限定符与真正楼号分开，避免“甲219-1号院5号楼”产生伪楼号冲突。Smoke 后还修正了长数字边界：`B2101` 不再解释为地下 2101 层，`三六零三门诊部` 不再解释为 3603 号门。

正式命令：

```bash
python scripts/qgr_sid/audit_relations.py \
  --poi-dir data/beijing_poi_clean_20260715_json \
  --identifier-dir outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/tiger_ids/epoch_20 \
  --query-aggregate-dir outputs/embeddings/query_augmented_bge_m3/query_poi_aggregates_v1 \
  --output-dir outputs/qgr_sid/EXP-20260813-05_tiger_numeric_relation_audit_v1 \
  --value-max 1055 \
  --batch-rows 65536 \
  --examples-per-relation 20
```

运行耗时 448.53 秒。全量扫描后 2,337,178 行、953,321 个碰撞 POI、291,352 个碰撞桶与 TIGER 指标严格守恒，16 个 POI 分片行数之和也为 2,337,178；逐行 `poi_id` 对齐无错位。

#### 全量覆盖结果

| 指标 | 结果 | 解释 |
|---|---:|---|
| 有任意稳定词法数字关系的碰撞 POI | 678,215 / 953,321（71.1424%） | 尚未检查该关系能否区分同桶 POI |
| 有任意词表内关系的碰撞 POI | 674,557 / 953,321（70.7586%） | 至少一个值位于 0—1055 |
| 词表内覆盖 / 词法覆盖 | 99.4606% | 当前数值上限不是 M1 主瓶颈 |
| 无词表内关系的碰撞 POI | 278,764（29.2414%） | 必须由 GEO、残差或后续补充关系处理 |
| 恰有 1 个词表内关系 | 222,629（23.3530%） | 局部树可选空间有限 |
| 至少 2 个词表内关系 | 451,928（47.4056%） | 具备变长关系组合候选 |
| 至少 3 个词表内关系 | 180,267（18.9094%） | 不能直接等同于三对即可唯一化 |
| 出现字段值冲突的碰撞 POI | 73,574（7.7177%） | M2 必须显式降权或过滤 |
| 至少含一个词表内关系 POI 的碰撞桶 | 221,565 / 291,352（76.0472%） | 只是“桶内有覆盖”，不是“桶已解决” |

主要关系覆盖为：

| 关系 | 稳定 POI | 词表内 POI | 越界 POI | 冲突 POI |
|---|---:|---:|---:|---:|
| `R_BUILDING` | 400,491 | 395,591 | 4,900 | 12,401 |
| `R_ADDRESS_NO` | 306,313 | 304,425 | 1,888 | 43,486 |
| `R_UNIT` | 213,256 | 213,248 | 8 | 98 |
| `R_ZONE_DIR` | 83,834 | 83,834 | 0 | 2,688 |
| `R_ZONE_NUM` | 83,524 | 83,522 | 2 | 1,640 |
| `R_ENTRANCE_DIR` | 69,337 | 69,337 | 0 | 12,563 |
| `R_FLOOR` | 35,173 | 35,150 | 23 | 752 |
| `R_SUBNO` | 29,533 | 29,272 | 261 | 112 |
| `R_QUALIFIER` | 28,402 | 28,402 | 0 | 638 |

所有关系共有 1,340,735 次稳定 POI—关系选择，其中 8,454 次关系值越界。越界主要来自真实长门牌、楼号、室号或商铺号；M2 不扩张全局 `<V>` 词表，而是优先选择同一 POI 的其他词表内关系，仍无法处理时交给 GEO/残差，防止长尾值显著扩大输出空间。

#### Query 加权结果

953,321 个碰撞 POI 中 169,323 个在现有 Train-only Query 聚合内有订单支持；其中 138,460 个具有词表内关系，POI 覆盖率为 81.7727%。进一步按请求权重统计：

| 权重口径 | 词表内关系覆盖 |
|---|---:|
| Train 原始订单数 | 2,492,745 / 2,823,361（88.2900%） |
| `sqrt(1 + min(order_count, P99))`，P99=179 | 83.1019% |

加权覆盖高于全量 POI 等权 70.7586%，说明可抽取数字关系更集中在有请求的热点 POI，方向值得进入 M2；但这仍只是“目标 POI 有关系”，不代表 Query 能正确预测关系值，也不能替代 Train 后 10% 的分支 Top-1/NLL。

类别差异支持组合式设计：楼栋号、门牌信息的词表内覆盖分别为 98.6290%/95.6570%，门/出入口为 72.0560%；停车场只有 30.6077%，路口、道路名和公交站仅 1.5249%/2.4081%/14.2644%。因此 M2 不应强制所有桶使用数字关系：楼栋/门牌优先数字关系，路口/道路/公交和跨区域同名桶优先验证 GEO，剩余叶子再进入残差原型。

#### 产物与核验

正式产物目录：

```text
outputs/qgr_sid/EXP-20260813-05_tiger_numeric_relation_audit_v1/
```

- `manifest.json` SHA256：`9946e9fbb7401c828734f418901109fcc1629713fbe198b26f50114fb97d29f0`；
- `metrics.json` SHA256：`389e42dfb010c02e8795828e5ef6504ca907701d4b760615d8987ac3877d28c8`；
- `relation_examples.jsonl` SHA256：`d6fac9c2590b5cf4c9100f49fdcef4e82f2f1a9d93c3a3e1540affcd2cea54e6`；
- 输入签名：`6e10067e812148b4bb4ee2d210002e3e2dbcb896cf03102eb2de21d9ef209ba7`；
- `--validate-only` 已逐文件复核两个受 manifest 管理的输出 SHA256；
- 14 个 QGR-SID 抽取/审计合成测试在正式运行前通过。

`relation_examples.jsonl` 中保留的原始中文名称、地址和别名只用于本地审计抽取是否正确；它们不会进入最终 identifier、Tokenizer 或 Assistant Target。最终 SID 仍只允许关系类型 token 与数字值 token。

#### M1 结论

M1 通过“值得继续做 M2”的门槛，但没有通过“可以直接启动 SFT”的门槛。积极证据是词法数字关系覆盖 88.29% 的碰撞 Train 订单、平方根压缩后仍为 83.10%，且 47.41% 的碰撞 POI 有至少两个词表内关系可供局部树选择。限制是 29.24% 的碰撞 POI完全没有词表内关系，7.72% 存在字段冲突，而且本实验没有验证同桶区分度与 Query/GID 可预测性。

M2 必须先报告每个关系的桶内信息增益、缺失分支、Train 后 10% 分支 Top-1/NLL、纯关系唯一叶比例、GEO 增量和 residual 请求权重；在这些指标出来前，不构建全量 SFT Messages，不申请训练资源。

### 14.2 `EXP-20260813-07`：真实时间 90/10 纯词法关系树代理

#### 目标与冻结协议

M2-A 回答两个比 M1 覆盖率更接近下游的问题：数字关系在同一个 TIGER 碰撞桶内能否改善目标排序，以及用 early Train Query 可预测性选择关系是否优于只按静态区分度选择。协议在运行前固定为：

- 仅使用 2026-07-01—12 的 7,586,410 条 Train 原始订单；
- 按 `create_time` 做事件数前 90% / 后 10% 的真实时间切分；切分分钟内按 `BLAKE2b(create_time, order_id, searchid)` 确定性排序，精确得到 6,827,769 / 758,641 条；
- early 90% 统计 POI 请求权重和 Query 对关系值的匹配率并编译树；holdout 10% 只评测，不参与选树；
- 字段存在任意值冲突时，该 POI 的对应关系视为缺失；数值必须位于 0—1055；
- 静态树按请求加权信息增益选择关系；Query 引导树再乘以 early 关系值可预测性，采用 20 个请求的全局先验平滑；
- 每个 POI 最多发射三个“关系类型索引 + 数字值”，缺失分支不消耗关系对预算；
- 只在已知 gold TIGER 碰撞桶内比较 popularity、静态树和 Query 引导树。该结果不是无约束生成指标，也不是 SFT 证据。

精确时间边界为 `2026-07-11 18:35`。此前有 6,827,350 条订单，边界分钟有 682 条，排序后的前 419 条归 early，最终严格闭合为 6,827,769 / 758,641。

#### 实现与命令

新增实现包括：

- `src/poi_gr/methods/qgr_sid/compiler.py`：深度不超过 3 的确定性变长关系树编译器；
- `src/poi_gr/methods/qgr_sid/proxy.py`：POI/TIGER 严格对齐、两遍订单流式切分、early 统计、双树编译、holdout 排序与全产物复检；
- `scripts/qgr_sid/evaluate_relation_proxy.py`：正式运行及 `--validate-only` 入口；
- `tests/qgr_sid/test_proxy.py`：真实事件数切分与端到端合成产物测试。

正式命令：

```bash
python scripts/qgr_sid/evaluate_relation_proxy.py \
  --output-dir outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1

python scripts/qgr_sid/evaluate_relation_proxy.py \
  --output-dir outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1 \
  --validate-only
```

运行环境为 Python 3.10.20、NumPy 1.26.4，代码基线为 `main@7dd52b3437a9db41834aef638e446010d0df3610` 的 dirty worktree；正式耗时 1,311.59 秒。运行前 20 个 QGR-SID 抽取、编译与代理测试全部通过，静态检查通过。

#### 严格关系目录与树结构

在 953,321 个 TIGER 碰撞 POI、291,352 个碰撞桶上，严格无冲突、词表内关系至少一项的 POI 为 665,678，占 69.8273%；287,643 个 POI 没有严格关系，73,574 个 POI 至少有一个冲突类型。该口径低于 M1 的 674,557，是因为 M2-A 对发生冲突的关系类型一律置缺失，不使用低优先级冲突中的选中值。

| 指标 | 静态树 | Query 引导树 |
|---|---:|---:|
| 决策节点数 | 143,972 | 144,926 |
| 纯关系唯一 POI | 377,638（39.6129%） | 378,007（39.6516%） |
| early 请求加权纯关系解决率 | 33.3635% | 32.4487% |
| 后缀 0/1/2/3 个关系对的 POI | 499,382 / 374,599 / 72,803 / 6,537 | 498,700 / 371,932 / 75,640 / 7,049 |

两棵树有 51,719 个 POI 的路径不同。Query 引导略增加 POI 等权解决数，但降低 early 请求加权纯关系解决率，说明它为了可表达性主动放弃了一部分热点桶的最大静态拆分；这需要由 holdout 排序而不是唯一率判断是否值得。

#### Holdout 已知桶代理结果

holdout 中有 273,937 条目标位于 TIGER 碰撞桶的请求。结果如下：

| 方法 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | Mean Rank |
|---|---:|---:|---:|---:|---:|---:|
| early popularity | 77.5642% | 96.1710% | 98.2303% | 99.3039% | 90.0425% | 1.5144 |
| 静态关系树 | 79.2803% | 97.0077% | 98.7552% | 99.5630% | 91.0359% | 1.4060 |
| Query 引导关系树 | **79.3047%** | **97.0292%** | 98.7552% | **99.5656%** | **91.0535%** | **1.4048** |

Query 引导树相对 popularity 的 HR@1/NDCG@10 增量为 `+1.7405pp/+1.0110pp`，对应 HR@1 命中从 212,477 增至 217,245；相对静态树的增量为 `+0.0245pp/+0.0176pp`，只多 67 条 HR@1 命中。纯词法路径在 holdout 上覆盖全部路径且命中目标值的请求仅 13,961 条，占 5.0964%；Query 可预测性最高的高支持直接数值关系主要是 `UNIT` 41.83%、`ZONE_NUM` 33.47%、`BUILDING` 28.01% 和 `PHASE` 22.24%，而门牌号只有 6.13%。

#### 产物与核验

正式目录为：

```text
outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1/
```

目录保存严格关系矩阵、early 请求/关系命中计数、静态与 Query 引导路径类型/值、纯关系解决 mask、有限改进/退化 Case、metrics 和 manifest，总大小约 139 MiB。`manifest.json` SHA256 为 `8d69059e807de9e01d40b905abdee5c94a6d624141d0ff807f38640a439a4c53`，`metrics.json` SHA256 为 `9e3fafad82597ecabe8e77fc1ab45ea9e5a782dd6461835077226456e9330000`，输入签名为 `12456b86a6c1d0319d3136d51c7d6dfc85de6bc3856e19169a806b6ee5ef4d9d`。`--validate-only` 已复核 12 个 NPY、metrics 和 cases 共 14 个产物的字节数、shape、dtype 与 SHA256。

#### M2-A 结论与 M2-B 决策

纯词法关系树通过“继续 GEO/GID 代理”的门槛：它在真实未来 10% Train 上同时改善 HR@1 与 NDCG@10，说明关系区分度不是纯静态假象。Query 引导相对静态树也是正向，但只有 67 条 HR@1 增量，证据强度不足以把当前简单数字匹配视为主创新；29.24% 以上 POI 没有可用严格词法关系，请求加权纯关系解决率也只有约 32%—33%，未达到启动 SFT 所要求的 60%。

因此下一步固定为 M2-B：加入静态 POI Geohash 的最短纯数字 GEO 关系，并用请求 GID 评估其分支可预测性；同时报告 GEO-only 与词法+GEO，相同 holdout 不重新调切分。Query embedding 与 residual 暂不先于 GEO/GID，以便先隔离现有 Prompt 已包含的地理信号是否真正补足词法盲区。

### 14.3 `EXP-20260813-08`：静态 GEO / 请求 GID 代理负结果

#### 目标与编码

M2-B 沿用 M2-A 的同一 Train 时间切分、TIGER 碰撞桶和 273,937 条 holdout 碰撞请求，不重新选择样本。静态 POI Geohash6 来自既有 `gid_codes.npy`，先对每个 TIGER 碰撞桶求公共前缀，再构造：

- 一字符候选：公共前缀后的第一个 Geohash code，数值范围 0—31；
- 两字符候选：`32 + 32*c1 + c2`，数值范围 32—1055；
- 同一 POI 最终仍只保存静态 `R_GEO + 数字值`，请求 GID 只在 early 统计精确分支匹配率和 holdout 代理排序。

正式实现位于 `src/poi_gr/methods/qgr_sid/geo_proxy.py` 与 `scripts/qgr_sid/evaluate_geo_proxy.py`。运行前 23 个 QGR-SID 测试和静态检查通过；正式耗时 993.06 秒。命令为：

```bash
python scripts/qgr_sid/evaluate_geo_proxy.py \
  --output-dir outputs/qgr_sid/EXP-20260813-08_tiger_geo_gid_proxy_v1
```

完整 2,337,178 行 GID POI ID、TIGER mapping 与 M2-A 碰撞行序均严格对齐；early POI 请求计数逐行复现 M2-A。M2-B 还在最终输出前强制精确复现 M2-A popularity、静态词法与 Query 引导词法四项排名指标。

#### 静态覆盖与请求可预测性

622,111 个碰撞 POI（65.2572%）所在桶存在可区分 GEO 候选，331,210 个 POI 的同桶 Geohash6 完全一致。请求 GID 对一字符/两字符目标 POI GEO 的 early 精确匹配率只有 22.1419%/11.3531%。GID 引导选择 166,589 个一字符桶、6,292 个两字符桶；其选中 GEO 的整体 early 精确匹配率为 21.3570%。

GEO 显著提高静态区分率但可生成性不足：

| 结构 | 纯关系唯一 POI | early 请求加权解决率 |
|---|---:|---:|
| M2-A Query 引导纯词法 | 378,007（39.6516%） | 32.4487% |
| 静态词法+GEO | 643,180（67.4673%） | 50.6594% |
| Query+GID 引导词法+GEO | 616,829（64.7032%） | 47.0551% |

这再次验证“静态消碰撞更高”不能直接当成生成检索提升。

#### Holdout 已知桶结果

| 方法 | HR@1 | HR@3 | HR@10 | NDCG@10 | Mean Rank |
|---|---:|---:|---:|---:|---:|
| popularity | 77.5642% | 96.1710% | 99.3039% | 90.0425% | 1.5144 |
| Query 引导纯词法 | **79.3047%** | **97.0292%** | **99.5656%** | **91.0535%** | **1.4048** |
| 静态 GEO-only | 75.8222% | 95.9407% | 99.3499% | 89.3255% | 1.5343 |
| GID 引导 GEO-only | 75.5739% | 95.7760% | 99.3455% | 89.1728% | 1.5469 |
| 静态词法+GEO | 77.0969% | 96.0407% | 99.4948% | 89.8998% | 1.4854 |
| Query+GID 词法+GEO | 75.3995% | 94.5338% | 99.3641% | 88.7791% | 1.5836 |

Query+GID 组合相对纯词法 HR@1/NDCG@10 下降 `3.9053pp/2.2744pp`，甚至低于 GEO-only `0.1745pp/0.3937pp`。尽管组合树在 holdout 上的静态解决请求占比达到 50.1418%，完整路径被 Query+GID 精确表达的请求只有 12.8088%，不能通过可生成性门禁。

#### 产物与结论

正式目录为 `outputs/qgr_sid/EXP-20260813-08_tiger_geo_gid_proxy_v1/`，保存碰撞 POI GID、公共前缀、GEO 候选/early 匹配、一/两字符选择、两套组合树和解决 mask，总大小约 49 MiB。`manifest.json`/`metrics.json` SHA256 为 `9f022ac4872ccc5a245031381a3f58a7c9ec7e973d3368b02375150ee07f79dd`/`b1e7491aa88e5e8e5d3f6cace8484c030da687f41a6ddbb068117730cb824d8b`，输入签名为 `51516168df1dc53120e95b79c722ecc69204d494ab67f313eb45532844c0032f`；17 个输出的字节数、shape、dtype 与 SHA256 已完整复检。

M2-B 记为负结果：不把精确目标 GEO 作为第一版关系 SID，也不因静态唯一率提高而启动 SFT。现有 Prompt 中的请求 GID继续保留，但后续只能作为距离/区域一致性的软预测特征，不能与目标 POI Geohash 直接等同。下一步进入 M2-C 的 Query embedding/桶内 residual：先验证 Query 语义能否预测词法未解决叶子，再决定 residual 原型与最终唯一化。

### 14.4 `EXP-20260813-09`：early-only Query residual 代理负结果

#### 目标与无泄漏重建

M2-C 验证现有 BGE Query embedding 能否在词法关系未解决时提供桶内 residual 信号。既有 1,408,778 条 Query embedding 由冻结预训练 BGE-M3 逐 Query 独立编码，不依赖目标标签；但既有 POI Query 聚合混合了完整 Train，不能直接使用。本实验因此：

1. 按 M2-A 同一时间边界扫描 7,586,410 条 Train；
2. 统计 758,641 条 holdout 的精确 raw Query–POI count；
3. 从 2,063,175 个 full Train Query–POI pair 逐分片扣除 holdout count；
4. 用 early-only `log1p(count) × IDF` 重建 161,582 个有 early 请求的碰撞 POI Query center；
5. 分别评测 raw center cosine、当前 TIGER 桶公共 Query 方向扣除后的 residual cosine，以及“词法证据优先、residual 只打破词法并列”的保守融合。

两条独立守恒路径均得到 full/early/holdout `7,586,410/6,827,769/758,641`；early-only 统计包含 1,305,104 个唯一 Query、1,913,124 个唯一 Query–POI 对和 471,595 个覆盖 POI。holdout 标签没有进入任何 POI center。

实现位于 `src/poi_gr/methods/qgr_sid/embedding_proxy.py` 与 `scripts/qgr_sid/evaluate_embedding_proxy.py`。运行前 25 个 QGR-SID 测试与静态检查通过；正式耗时 592.32 秒。命令为：

```bash
python scripts/qgr_sid/evaluate_embedding_proxy.py \
  --output-dir outputs/qgr_sid/EXP-20260813-09_tiger_query_residual_proxy_v1
```

#### Holdout 结果

目标在 early 有 Query center 的 holdout 请求为 264,638/273,937（96.6054%），有非零桶内 residual 的为 220,559（80.5145%），因此负结果不是由大面积无覆盖造成。

| 方法 | HR@1 | HR@3 | HR@10 | NDCG@10 | Mean Rank |
|---|---:|---:|---:|---:|---:|
| popularity | 77.5642% | 96.1710% | 99.3039% | 90.0425% | 1.5144 |
| Query 引导纯词法 | **79.3047%** | **97.0292%** | **99.5656%** | **91.0535%** | **1.4048** |
| raw POI Query center | 66.6916% | 93.2185% | 99.4316% | 85.1299% | 1.7046 |
| 桶内 Query residual | 65.9998% | 92.5107% | 99.3177% | 84.5368% | 1.7685 |
| 词法优先 + residual tie-break | 67.0534% | 93.0586% | 99.5105% | 85.1646% | 1.6856 |

保守融合相对纯词法 HR@1/NDCG@10 下降 `12.2514pp/5.8888pp`。这说明 POI 历史 Query 聚合主要编码了多请求分布与热点别名，并不能作为单条请求在高度相似 TIGER 桶中的稳定实体判别中心；扣除桶公共方向也没有把它变成可靠 residual。

#### 产物与决策

正式目录 `outputs/qgr_sid/EXP-20260813-09_tiger_query_residual_proxy_v1/` 保存 early center 碰撞行、161,582×1,024 float16 center、E2 权重和有限 Case，总大小约 318 MiB。`manifest.json`/`metrics.json` SHA256 为 `19c264a30473cfb7082acc783cf08ad9698d9066f8b549e704b83215ce99f20c`/`c8903025cbf0890d40c2dc0b97a327d30bf9d8ec80c6b2bc5f0b5cfd81ea033a`，输入签名为 `8a55cfa97b4a69abd7a9c383d54e3e5d53d21c20683b7143e226646d7e7ef53d`；metrics、cases 和三个数组共五个产物已完整复检。

M2-C 记为负结果，不继续训练 512 个 Query residual 原型，也不把 Query embedding 融合进第一版 SID。M2-B/M2-C 共同说明，直接提高静态消碰撞覆盖会破坏请求可生成性。下一步收缩为 M2-D 可靠词法关系：只保留 early 全局支持充分且数值匹配率高的楼栋、单元、分区、期数等关系，未解决部分继续使用 TIGER fallback，以最小 SFT 风险验证关系 SID 核心贡献。

### 14.5 `EXP-20260813-10`：可靠词法 + TIGER fallback 门禁未通过

#### 预注册门槛

M2-D 不做阈值网格，只运行一套在查看本实验 holdout 前冻结的可靠性门槛：

- 关系在 early 碰撞请求中的全局支持数不少于 10,000；
- early Query 对关系值的 value-aware 精确匹配率不低于 15%；
- 最多三个关系对；未被关系唯一化的叶子保留原 TIGER collision code；
- 成功要求相对 popularity 的 HR@1/NDCG@10 同时正向，并保留 full lexical 至少 95% 的 HR@1 增量。

early-only 门禁选出六类：`R_PHASE`、`R_ZONE_NUM`、`R_ZONE_ALPHA`、`R_ZONE_DIR`、`R_BUILDING`、`R_UNIT`。支持数/匹配率分别为 42,072/19.54%、86,102/31.95%、35,282/20.47%、134,388/17.02%、385,947/26.40%、25,658/40.56%。门牌、楼层、出入口、房间、商铺等关系被剔除。

实现位于 `src/poi_gr/methods/qgr_sid/reliable_proxy.py` 与 `scripts/qgr_sid/evaluate_reliable_proxy.py`。正式运行前全量 QGR-SID 28 个测试和静态检查通过；耗时 734.48 秒。命令为：

```bash
python scripts/qgr_sid/evaluate_reliable_proxy.py \
  --output-dir outputs/qgr_sid/EXP-20260813-10_tiger_reliable_lexical_proxy_v1
```

#### 结果

可靠树使 263,890 个碰撞 POI（27.6811%）得到纯关系唯一路径，但只覆盖 7.2285% 的 early 碰撞请求；689,431 个 POI、92.7715% 的 early 碰撞请求仍须使用 TIGER fallback。路径长度 0/1/2/3 个关系对的 POI 为 627,089/254,419/66,928/4,885。

| 方法 | HR@1 | HR@3 | HR@10 | NDCG@10 | Mean Rank |
|---|---:|---:|---:|---:|---:|
| popularity | 77.5642% | 96.1710% | 99.3039% | 90.0425% | 1.5144 |
| full Query 引导词法 | **79.3047%** | **97.0292%** | **99.5656%** | **91.0535%** | **1.4048** |
| reliability-gated 词法 | 78.9996% | 96.8942% | 99.5386% | 90.8838% | 1.4181 |

可靠版本相对 popularity 的 HR@1/NDCG@10 仍提高 `1.4354pp/0.8413pp`，但相对 full lexical 下降 `0.3052pp/0.1697pp`，只保留 full lexical `82.47%` 的 HR@1 增量，未达到 95% 预注册门槛。有可靠路径的 31,834 条 holdout 请求中，完整路径可由 Query 表达的为 9,235 条，条件表达率 29.0099%；相比 full lexical 的全局 5.0964% 有明显改善，但覆盖太窄。

#### 产物与最终门禁决策

正式目录 `outputs/qgr_sid/EXP-20260813-10_tiger_reliable_lexical_proxy_v1/` 保存可靠路径类型、值、解决 mask、metrics 和有限 Case，总大小约 12 MiB。`manifest.json`/`metrics.json` SHA256 为 `7405d0c6b5fdfc95818c2353d2db96b757446f66329a30c381112b5e30a98776`/`b17e9b56d81e84e0f2b56dea3b1b682f4ff5f3743b3330c09fd592eed478e5f0`，输入签名为 `85ee0f815586dc5497a6d9a54d7e32c73f12c078b3edd30152d391f8fa01885a`；五个输出已完整复检。

按预注册规则，M2-D 未通过。可靠词法关系确实提供正向信号，但热点覆盖不足，若 92.77% 请求仍生成原 TIGER fallback，预计难以用一次昂贵 SFT 稳定超过 TIGER；full lexical 虽代理更强，目标完整表达率又过低。至此不构建最终 mapping、不扩词表、不启动 SFT，也不在相同 holdout 上继续调 10k/15% 门槛。后续只有在引入新的关系目标学习机制、而非继续堆静态特征时，才重开该路线。

## 15. 方案决策记录

### 2026-08-13：冻结第一版主方法

- 基础 SID 选择 TIGER/BGE-M3/RQ-VAE `1024×3`；
- 采用真正的变长关系 SID，不先做固定长度过渡实验；
- Query 只用于关系可预测性和桶内残差，不让单条请求改变 SID；
- 请求 GID 保留为输入，只在有信息增益时生成最短 GEO 数字关系；
- Assistant Target 不混入中文楼号、单元、商场或道路文本；
- 关系后缀统一为“关系类型 token + 纯离散数值 token”；
- 第一个正式长训练只在通过 Train 90/10 离线门禁后启动；
- 端到端成功条件是超过 TIGER SFT 指标，而不是静态 SID 指标更好。
