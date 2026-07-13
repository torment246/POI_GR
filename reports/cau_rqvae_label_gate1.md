# CAU-RQ-VAE Gate 1 Coarse Category Labels

This gate builds high-confidence coarse category labels for CAU-RQ-VAE. It does not train a model and does not drop any POI.

## Policy

- `UNK` / unmatched POIs are retained for reconstruction and final SID evaluation.
- `UNK` / unmatched POIs do not participate in category supervision (`L_tag`).
- Training labels use high-confidence rules with field priority `category_l1 -> tags -> name`.
- Full `sid_eval.py` inferred semantic labels remain the baseline-compatible evaluation metric and are not used directly as training labels.
- High-confidence supervised labels are split by label into 80% tag-train and 20% tag-heldout validation using a fixed seed.

## Artifacts

- Input: `data/sid/poi_sid_train.parquet`
- Labels: `outputs/experiments/cau_rqvae/labels/cau_coarse_category_labels.parquet`
- Vocabulary: `outputs/experiments/cau_rqvae/labels/cau_category_vocab.json`
- Distribution: `outputs/experiments/cau_rqvae/labels/cau_label_distribution.csv`
- Report: `reports/cau_rqvae_label_gate1.md`

## Summary

- rows: 109385
- supervised rows: 65147 / 0.5956
- tag train rows: 52120
- tag held-out rows: 13027
- unsupervised rows: 44238
- number of supervised labels: 21
- seed: 42
- heldout_ratio: 0.2
- min_label_count: 20
- low-count labels removed from supervision: []

## Split Names

- category loss train split: `tag_train`
- held-out category validation split: `tag_heldout`
- no category supervision split: `unsupervised`

## Label Distribution

| label | total | train | heldout | heldout_rate | category_l1_source | tags_source | name_source | example_keywords |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ev_charging | 5968 | 4774 | 1194 | 0.2001 | 2682 | 0 | 3286 | 汽车充电站 / 充电站 / 充电桩 / 小桔充电 / 星星充电 |
| bank | 5520 | 4416 | 1104 | 0.2000 | 2049 | 0 | 3471 | 银行 / ATM |
| transit | 5307 | 4246 | 1061 | 0.1999 | 2010 | 0 | 3297 | 公交站 / 地铁站 / 火车站 / 汽车站 / 机场 |
| hotel | 4792 | 3834 | 958 | 0.1999 | 3070 | 0 | 1722 | 酒店 / 宾馆 / 民宿 / 旅馆 / 住宿 |
| residential | 4689 | 3751 | 938 | 0.2000 | 1647 | 0 | 3042 | 小区 / 花园 / 家园 / 社区 / 公寓 |
| gas_station | 4227 | 3382 | 845 | 0.1999 | 1517 | 0 | 2710 | 加油站 / 加气站 / cng加气站 / 加油加气站 / CNG加气站 |
| company | 4012 | 3210 | 802 | 0.1999 | 787 | 0 | 3225 | 有限公司 / 公司 / 产业园 / 工厂 / 工业园 |
| shopping | 3839 | 3071 | 768 | 0.2001 | 1407 | 0 | 2432 | 超市 / 便利店 / 农贸市场 / 百货 / 商场 |
| parking | 3742 | 2994 | 748 | 0.1999 | 2062 | 0 | 1680 | 停车场 / 停车位 / 地下停车 / 停车库 |
| government | 3336 | 2669 | 667 | 0.1999 | 858 | 0 | 2478 | 人民政府 / 政府 / 派出所 / 村委会 / 公安局 |
| school | 3147 | 2518 | 629 | 0.1999 | 1691 | 0 | 1456 | 中学 / 大学 / 小学 / 学院 / 学校 |
| restaurant | 3014 | 2411 | 603 | 0.2001 | 1485 | 0 | 1529 | 火锅 / 饭店 / 小吃 / 美食 / 餐厅 |
| medical | 2796 | 2237 | 559 | 0.1999 | 1600 | 0 | 1196 | 医院 / 诊所 / 卫生院 / 社区卫生服务中心 / 卫生服务站 |
| scenic | 2422 | 1938 | 484 | 0.1998 | 1261 | 0 | 1161 | 公园 / 景区 / 博物馆 / 景点 / 旅游区 |
| public_toilet | 2389 | 1911 | 478 | 0.2001 | 431 | 0 | 1958 | 公共厕所 / 卫生间 / 洗手间 / 公厕 |
| telecom_electronics | 1789 | 1431 | 358 | 0.2001 | 335 | 0 | 1454 | 中国移动 / 中国电信 / 小米之家 / 中国联通 / 手机维修 |
| auto_service | 1301 | 1041 | 260 | 0.1998 | 320 | 0 | 981 | 洗车 / 汽修 / 汽车维修 / 轮胎 / 养车 |
| logistics | 951 | 761 | 190 | 0.1998 | 416 | 0 | 535 | 物流 / 快递 / 菜鸟驿站 / 驿站 / 配送站 |
| entertainment | 806 | 645 | 161 | 0.1998 | 455 | 0 | 351 | 台球 / 网吧 / 棋牌 / 健身房 / 影城 |
| pharmacy | 726 | 581 | 145 | 0.1997 | 304 | 0 | 422 | 药房 / 药店 |
| beauty | 374 | 299 | 75 | 0.2005 | 168 | 0 | 206 | 美容 / 理发店 / 美发 / 美甲 |

## Gate Decision

- Gate 1 status: `PASS`
- all POIs retained; supervised labels have disjoint train/held-out splits; UNK is excluded from L_tag.
- split counts: {"tag_heldout": 13027, "tag_train": 52120, "unsupervised": 44238}
