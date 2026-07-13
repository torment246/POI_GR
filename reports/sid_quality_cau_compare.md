# CAU Final SID Quality Compare

Compared with the same SID export/eval logic used by Gate 0 and pilots.

| run | prefix1_semantic_purity | prefix2_semantic_purity | prefix3_semantic_purity | unique_sid_rate | sid_collision_group_count | max_pois_per_sid | qrels_sid_collision_rate | unique_pid_rate | qrels_pid_collision_rate | reconstruction_cosine | category_heldout_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic_baseline | 0.352700 | 0.446100 | 0.765100 | 0.561700 | 15547 | 211 | 0.494300 | 0.967600 | 0.008200 | 0.515736 | NA |
| P1_tag0025_unique010 | 0.353300 | 0.447700 | 0.777600 | 0.586100 | 15430 | 209 | 0.472600 | 0.969600 | 0.007400 | 0.512452 | 0.672819 |
| P2_tag0025_unique015 | 0.349800 | 0.446300 | 0.782000 | 0.597300 | 15433 | 207 | 0.461400 | 0.971000 | 0.007100 | 0.511648 | 0.639928 |
| CAU_Final_P1 | 0.364182 | 0.453719 | 0.786022 | 0.600887 | 15719 | 202 | 0.468927 | 0.971852 | 0.007499 | 0.511913 | 0.939720 |

## Formal CAU Output

- checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`
- SID indices: `data/sid/poi_sid_indices_cau.npy`
- SID mapping: `data/sid/poi_sid_mapping_cau.parquet`
- SID report: `reports/sid_quality_cau.md`
- experiment copy: `outputs/experiments/cau_rqvae/final_p1/poi_sid_indices_cau.npy`, `outputs/experiments/cau_rqvae/final_p1/poi_sid_mapping_cau.parquet`

## Final Notes

- P1 remains the promoted configuration; P2 is kept as a weight ablation only.
- Final checkpoint selection used protected SID metrics before validation loss.
- Final qrels SID collision rate: 0.468927
- Final unique SID rate: 0.600887
