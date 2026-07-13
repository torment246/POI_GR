# POI Geo Feature Quality Report

## Basic Statistics
- input_sid_rows: 109385
- input_embedding_meta_rows: 109385
- output_geo_rows: 109385
- output_geo_features_shape: [109385, 54]
- unique_poi_id: 109385
- row_order_aligned_with_embedding_meta: True
- row_id_aligned_with_embedding_meta: True
- input_sid_initial_row_order_aligned_with_embedding_meta: True
- input_sid_initial_row_id_aligned_with_embedding_meta: True
- lat_missing_count: 0
- lon_missing_count: 0
- invalid_latlon_count: 0

## Coordinate Range
- lat_min: 16.330358
- lat_max: 53.475645
- lat_mean: 33.192344
- lat_p50: 32.340032
- lon_min: 74.969215
- lon_max: 134.753338
- lon_mean: 114.780835
- lon_p50: 116.187976

## Geohash Coverage
- geohash5_non_empty_count / rate: 109385 / 1.0000
- geohash6_non_empty_count / rate: 109385 / 1.0000
- geohash7_non_empty_count / rate: 109385 / 1.0000
- unique_geohash5: 19997
- unique_geohash6: 59990
- unique_geohash7: 96552
- top_geohash5:
- wtw3s: 369
- wx4g1: 365
- wqj6z: 309
- yb4h3: 304
- wx4ep: 298
- wtw3e: 254
- wwgqd: 253
- wx4g0: 252
- ws0e9: 245
- wx4fb: 238
- top_geohash6:
- wm6n2k: 41
- wm7b1h: 40
- yb4h3k: 38
- wtsqpj: 35
- wtw3sq: 34
- wtw3sm: 33
- wm7b1j: 32
- wqj6zn: 31
- wx4g14: 31
- ws10kb: 30

## Anchor Summary
- n_anchors: 16
- anchor coordinates table:
| anchor_id | count | ratio | anchor_lat | anchor_lon |
| --- | ---: | ---: | ---: | ---: |
| 0 | 9033 | 0.0826 | 34.420594 | 111.665125 |
| 1 | 10116 | 0.0925 | 23.003884 | 113.241335 |
| 2 | 5375 | 0.0491 | 45.238923 | 126.164394 |
| 3 | 18716 | 0.1711 | 31.028400 | 120.467435 |
| 4 | 12876 | 0.1177 | 39.608788 | 116.993606 |
| 5 | 2139 | 0.0196 | 43.091777 | 85.436100 |
| 6 | 4181 | 0.0382 | 28.021643 | 113.126447 |
| 7 | 4025 | 0.0368 | 25.346599 | 103.488731 |
| 8 | 7586 | 0.0694 | 30.260387 | 104.716892 |
| 9 | 3368 | 0.0308 | 26.146359 | 118.777791 |
| 10 | 4431 | 0.0405 | 38.881892 | 112.491003 |
| 11 | 10749 | 0.0983 | 35.917656 | 117.594001 |
| 12 | 6650 | 0.0608 | 31.354783 | 115.193734 |
| 13 | 2567 | 0.0235 | 19.855915 | 109.748844 |
| 14 | 5145 | 0.0470 | 41.728665 | 122.798423 |
| 15 | 2428 | 0.0222 | 37.031824 | 103.827408 |
- anchor assignment distribution:
  - anchor_0: 9033 / 0.0826
  - anchor_1: 10116 / 0.0925
  - anchor_2: 5375 / 0.0491
  - anchor_3: 18716 / 0.1711
  - anchor_4: 12876 / 0.1177
  - anchor_5: 2139 / 0.0196
  - anchor_6: 4181 / 0.0382
  - anchor_7: 4025 / 0.0368
  - anchor_8: 7586 / 0.0694
  - anchor_9: 3368 / 0.0308
  - anchor_10: 4431 / 0.0405
  - anchor_11: 10749 / 0.0983
  - anchor_12: 6650 / 0.0608
  - anchor_13: 2567 / 0.0235
  - anchor_14: 5145 / 0.0470
  - anchor_15: 2428 / 0.0222
- min_anchor_distance_km_mean: 163.264143
- min_anchor_distance_km_p50: 141.318978
- min_anchor_distance_km_p95: 353.825291

## Feature Numeric Checks
- feature_dim: 54
- has_nan: False
- has_inf: False
- standardized_mean_abs_max: 0.00000000
- standardized_std_min: 1.00000000
- standardized_std_max: 1.00000000

## Extra CSV Reports
- poi_geo_manual_check_samples.csv
- geohash6_distribution.csv
- anchor_assignment_distribution.csv
- qrels_sample_used: True
- manual_sample_group_counts: {'random': 100, 'high_frequency_geohash5': 100, 'coordinate_boundary': 50, 'qrels_poi': 50}

## Manual Samples
| row_id | poi_id | name | city | address | lat | lon | geohash6 |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | mobilitybench:8fdde3d05b6ad11892 | 通运新村 | 张家港市 | 江苏省 张家港市 杨舍镇龙潭路316号 | 31.875793 | 120.544950 | wtttr4 |
| 1 | mobilitybench:0f926c78958912b11d | 景洪志远康宁客客栈(西双版纳嘎洒国际机场高铁站店) | 景洪市 | 云南省 景洪市 嘎洒镇曼占宰村委会曼喃村民小组149号 | 21.982162 | 100.777037 | w5ztj1 |
| 2 | mobilitybench:be2b489cffbdf62e5c | 站前客栈(西双版纳高铁站店) | 景洪市 | 云南省 景洪市 嘎洒镇曼占宰村委会曼喃村142号 | 21.982250 | 100.776717 | w5ztj1 |
| 3 | mobilitybench:fcb26cc7b2c57cebb8 | 景洪高铁大酒店(西双版纳嘎洒机场高铁站店) | 景洪市 | 云南省 景洪市 嘎洒镇机场东路188米 | 21.979975 | 100.775450 | w5ztj1 |
| 4 | mobilitybench:9b176d8ad3db7afe84 | 维也纳国际酒店(西双版纳机场高铁站店) | 景洪市 | 云南省 景洪市 机场公路oyo号 | 21.982729 | 100.777287 | w5ztj1 |
| 5 | mobilitybench:2f8d1eb63cdef08e55 | 景洪艾宰泰客栈(嘎洒机场高铁店) | 景洪市 | 云南省 景洪市 曼喃新村96号 | 21.981886 | 100.777686 | w5ztj3 |
| 6 | mobilitybench:c2b784365519ae2846 | 景洪简雅民宿(嘎洒机场高铁站店) | 景洪市 | 云南省 景洪市 嘎洒镇曼喃新村68号 | 21.980676 | 100.776964 | w5ztj1 |
| 7 | mobilitybench:05d77782734a0a4dc3 | 西双版纳景洪雨之林民宿 | 景洪市 | 云南省 景洪市 允景洪街道镜与森林别墅西143米 | 21.988338 | 100.772831 | w5ztj4 |
| 8 | mobilitybench:7b967c2033af732cb4 | 景洪嘎洒云尚民宿 | 景洪市 | 云南省 景洪市 曼喃佛寺东南(俊熙烟酒楼上) | 21.979256 | 100.775197 | w5ztj1 |
| 9 | mobilitybench:95ef67c469f2fec104 | 壹品顶苑美宿 | 景洪市 | 云南省 景洪市 勐腊路88号1栋13楼 | 21.988280 | 100.775991 | w5ztj4 |
| 10 | mobilitybench:966fc600e7cc635c87 | 合欢苑民宿(机场公路2号分店) | 景洪市 | 云南省 景洪市 曼喃新村121号 | 21.981925 | 100.777468 | w5ztj1 |
| 11 | mobilitybench:f22f1333b4ef7237ce | 西双版纳站 | 景洪市 | 云南省 景洪市 嘎洒镇曼暖龙村 | 21.983691 | 100.773372 | w5ztj4 |
| 12 | mobilitybench:2747e4bf5f34cc0d70 | 安徽建筑大学紫云路校区 | 合肥市 | 安徽省 蜀山区 经济技术开发区紫云路292号 | 31.744052 | 117.223461 | wteke6 |
| 13 | mobilitybench:973b0b829b016a1cb9 | 广汽能源汽车充电站(昊铂西安民乐园万达超充站) | 西安市 | 陕西省 新城区 西安民乐园万达广场地下停车场 | 34.269060 | 108.963660 | wqj6zw |
| 14 | mobilitybench:ce6aeb9ea21acc5f1c | 极氪能源超级充电站(西安万达广场民乐园店极充站) | 西安市 | 陕西省 新城区 解放路111号万达广场民乐园店地下停车场001 | 34.268964 | 108.963793 | wqj6zw |
| 15 | mobilitybench:08eb4b10a3c1d1b23d | 特斯拉超级充电站(民乐园万达) | 西安市 | 陕西省 新城区 解放路111号万达广场地下停车场B2层D区D200车位旁 | 34.267830 | 108.963760 | wqj6zw |
| 16 | mobilitybench:afe39c5afa99108541 | 新电途汽车充电站(民乐园万达广场青鸟超级充电站) | 西安市 | 陕西省 新城区 东四路41号地面停车场 | 34.267487 | 108.963390 | wqj6zw |
| 17 | mobilitybench:90ad5d3229fa1efc5a | 小桔充电站(五路口民安大厦) | 西安市 | 陕西省 新城区 东六路145号地下停车场B区B068(五路口地铁站D东北口步行300米) | 34.271779 | 108.964407 | wqj6zw |
| 18 | mobilitybench:6d3e2e04d60f2cb6c7 | 延长壳牌汽车充电站(延长壳牌新城区革命公园充电站) | 西安市 | 陕西省 新城区 尚德路165号小区西门西60米 | 34.270740 | 108.960140 | wqj6zq |
| 19 | mobilitybench:0b2bf8d7714d05cc96 | 星星充电汽车充电站(联志花苑市体育场) | 西安市 | 陕西省 新城区 西一路街道西五路47号西安城墙·碑林历史文化景区(东北角) | 34.269535 | 108.958180 | wqj6zq |
