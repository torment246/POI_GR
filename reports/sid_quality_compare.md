# SID Quality Compare Report

## Summary Table
| mode | unique_sid_rate | unique_pid_gid6_sid_rate | max_pois_per_sid | codebook_usage_mean | prefix1_weighted_semantic_purity | prefix2_weighted_semantic_purity | prefix3_weighted_semantic_purity | prefix1_weighted_city_purity | prefix2_weighted_city_purity | prefix3_weighted_city_purity | prefix1_weighted_geohash5_purity | prefix2_weighted_geohash5_purity | prefix3_weighted_geohash5_purity | qrels_targets_in_sid_collision_rate | qrels_targets_in_pid_collision_rate | queries_with_sid_collision_rate | queries_with_pid_collision_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic | 0.5617 | 0.9676 | 211 | 1.0000 | 0.3527 | 0.4461 | 0.7651 | 0.0915 | 0.2765 | 0.6799 | 0.0147 | 0.1572 | 0.6120 | 0.4943 | 0.0082 | 0.2417 | 0.1073 |
| geo_fused | 0.5073 | 0.9472 | 158 | 1.0000 | 0.3201 | 0.4375 | 0.7518 | 0.2144 | 0.4914 | 0.8290 | 0.0309 | 0.1936 | 0.6175 | 0.5858 | 0.0149 | 0.3072 | 0.1557 |

## Recommendation
- 两者接近，优先推荐 semantic 作为第一版生成式检索 target，geo_fused 作为消融对照。
