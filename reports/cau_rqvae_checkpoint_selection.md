# CAU-RQ-VAE Checkpoint Selection

## Protection Baseline

- prefix1_semantic_purity: 0.35270000
- prefix2_semantic_purity: 0.44610000
- prefix3_semantic_purity: 0.76510000
- unique_sid_rate: 0.56170000
- unique_pid_rate: 0.96760000
- qrels_sid_collision_rate: 0.49430000
- qrels_pid_collision_rate: 0.00820000
- reconstruction_cosine: 0.51573642
- max_pois_per_sid: 211.00000000
- sid_collision_group_count: 15547.00000000

## SID Evaluation History

| epoch | prefix1_semantic_purity | prefix3_semantic_purity | unique_sid_rate | qrels_sid_collision_rate | unique_pid_rate | reconstruction_cosine | min_codebook_usage_rate | category_heldout_macro_f1 | passes_protection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.364822 | 0.775253 | 0.577529 | 0.489163 | 0.969164 | 0.513215 | 1.000000 | 0.919027 | True |
| 10 | 0.364620 | 0.785720 | 0.598025 | 0.475090 | 0.971221 | 0.511864 | 1.000000 | 0.934797 | True |
| 13 | 0.364182 | 0.786022 | 0.600887 | 0.468927 | 0.971852 | 0.511913 | 1.000000 | 0.939720 | True |

## Decision

- selected_epoch: 13
- selected_checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`
- qrels_sid_collision_rate: 0.468927
- unique_sid_rate: 0.600887
- prefix1_semantic_purity: 0.364182
- category_heldout_macro_f1: 0.939720
