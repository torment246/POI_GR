# CAU-RQ-VAE Cluster Examples

The cases use baseline clusters and inspect the same POI sets under CAU Final P1. They are diagnostic examples, not cherry-picked success-only evidence.

| case | baseline_key | final_dominant_sid | baseline_cluster_size | final_max_same_poi_bucket | final_split_count | baseline_top_category | baseline_purity | final_top_category | final_purity | improvement | still_failed | trade_off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| good_category_cluster | S0_25 S1_127 S2_83 | S0_25 S1_127 S2_83 | 92 | 21 | 38 | 公共厕所 | 1.000000 | 公共厕所 | 1.000000 | yes | no | good semantic cluster stays pure, but is split across many final SIDs |
| severely_mixed_prefix1_cluster | S0_97 | S0_97 S1_66 S2_52 | 1669 | 11 | 1141 | 政府机构 | 0.026363 | 康布乡 | 0.090909 | partial | yes | category/name ambiguity remains severe even though the large bucket is split |
| largest_full_sid_collision | S0_0 S1_72 S2_2 | S0_0 S1_72 S2_2 | 211 | 178 | 18 | UNK | 0.488152 | UNK | 0.533708 | partial | yes | largest collision shrinks but remains a large UNK/brand-style collision |
| government_admin_mixed_case | S0_97 | S0_97 S1_66 S2_52 | 1669 | 11 | 1141 | 政府机构 | 0.026363 | 康布乡 | 0.090909 | partial | yes | administrative names remain mixed with towns, services, and other POIs |
| brand_or_chain_case | S0_98 S1_121 | S0_98 S1_121 S2_108 | 608 | 72 | 102 | 汽车充电站 | 0.521382 | 汽车充电站 | 0.458333 | partial | yes | chain/brand pattern improves collision size but keeps category ambiguity |

## Representative POIs

### good_category_cluster
- representative_pois: mobilitybench:1adea9e0603fcb819b:公共厕所(公共厕所) | mobilitybench:c13015f384432897fd:公共厕所(公共厕所) | mobilitybench:1a07464de34fe4239d:公共厕所(公共厕所) | mobilitybench:0889433479ce4731fc:公共厕所(公共厕所) | mobilitybench:3f09143becf8269514:公共厕所(公共厕所)
- improvement: yes
- still_failed: no
- trade_off: good semantic cluster stays pure, but is split across many final SIDs

### severely_mixed_prefix1_cluster
- representative_pois: mobilitybench:9eb44010ff77b95a76:黄家镇(UNK) | mobilitybench:2a3ec05a5a24a5de4a:巴彦查干乡人民政府(政府机构) | mobilitybench:b88d0fa55fd04d0859:永乐镇人民政府(政府机构) | mobilitybench:6065a499858ab212bb:陈家镇裕安农贸市场(UNK) | mobilitybench:c718b45cc696abb5c3:寨河镇人民政府(政府机构)
- improvement: partial
- still_failed: yes
- trade_off: category/name ambiguity remains severe even though the large bucket is split

### largest_full_sid_collision
- representative_pois: mobilitybench:a0f885204dde3d60f0:小米之家(邢台临西县临西镇阳光北大街授权店)(UNK) | mobilitybench:3bcac8bc945f09f73c:小米之家(邢台临西玉兰路联通授权店)(UNK) | mobilitybench:e53f1a1122a3848c8f:小米之家(吕尖线店)(UNK) | mobilitybench:22ac8c893f89ed5244:小米之家(邢台市临西县家乐园授权店)(UNK) | mobilitybench:1d96bd379e0a92c1c2:小米之家(贵阳花溪区花溪沃尔玛专卖店)(UNK)
- improvement: partial
- still_failed: yes
- trade_off: largest collision shrinks but remains a large UNK/brand-style collision

### government_admin_mixed_case
- representative_pois: mobilitybench:9eb44010ff77b95a76:黄家镇(UNK) | mobilitybench:2a3ec05a5a24a5de4a:巴彦查干乡人民政府(政府机构) | mobilitybench:b88d0fa55fd04d0859:永乐镇人民政府(政府机构) | mobilitybench:6065a499858ab212bb:陈家镇裕安农贸市场(UNK) | mobilitybench:c718b45cc696abb5c3:寨河镇人民政府(政府机构)
- improvement: partial
- still_failed: yes
- trade_off: administrative names remain mixed with towns, services, and other POIs

### brand_or_chain_case
- representative_pois: mobilitybench:90ad5d3229fa1efc5a:小桔充电站(五路口民安大厦)(汽车充电站) | mobilitybench:8073f917bde99ab91b:小桔充电汽车充电站(清洋超充站)(汽车充电站) | mobilitybench:a351e7e29998fbe2f4:小桔充电汽车充电站(万马爱充少儿公园站)(汽车充电站) | mobilitybench:20de4bde1e71d9011c:小桔充电汽车充电站(红星文化大酒店充电站)(酒店住宿) | mobilitybench:25ecba6d9fbb3fbd51:小桔充电汽车充电站(上城区平海路中旅充电站)(汽车充电站)
- improvement: partial
- still_failed: yes
- trade_off: chain/brand pattern improves collision size but keeps category ambiguity
