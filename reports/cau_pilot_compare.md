# CAU Pilot Compare

Compared Semantic baseline, P1, and P2 using the same SID export/eval logic. CAU pilot checkpoints were exported only under their experiment directories; no formal `data/sid/poi_sid_indices_cau.npy` or `data/sid/poi_sid_mapping_cau.parquet` was written.

## Artifacts

- Metrics CSV: `reports/cau_pilot_metrics.csv`
- P1 SID report: `outputs/experiments/cau_rqvae/pilot_tag0025_unique010/sid_quality.md`
- P2 SID report: `outputs/experiments/cau_rqvae/pilot_tag0025_unique015/sid_quality.md`

## Summary Table

| run | lambda_tag | lambda_unique | prefix1_semantic_purity | prefix2_semantic_purity | prefix3_semantic_purity | unique_sid_rate | sid_collision_group_count | max_pois_per_sid | qrels_sid_collision_rate | unique_pid_rate | qrels_pid_collision_rate | reconstruction_cosine | min_codebook_usage_rate | category_heldout_macro_f1 | passes_protection | primary_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic_baseline | NA | NA | 0.352700 | 0.446100 | 0.765100 | 0.561700 | 15547 | 211 | 0.494300 | 0.967600 | 0.008200 | 0.515736 | 1.000000 | NA | True | 0.000000 |
| P1_tag0025_unique010 | 0.025000 | 0.100000 | 0.353300 | 0.447700 | 0.777600 | 0.586100 | 15430 | 209 | 0.472600 | 0.969600 | 0.007400 | 0.512452 | 1.000000 | 0.672819 | True | 0.238500 |
| P2_tag0025_unique015 | 0.025000 | 0.150000 | 0.349800 | 0.446300 | 0.782000 | 0.597300 | 15433 | 207 | 0.461400 | 0.971000 | 0.007100 | 0.511648 | 1.000000 | 0.639928 | True | 0.317500 |

## Deltas vs Semantic Baseline

| run | delta Prefix1 | delta Prefix2 | delta Prefix3 | delta unique SID | delta max POIs/SID | delta qrels SID collision | delta unique PID | delta recon cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1_tag0025_unique010 | 0.000600 | 0.001600 | 0.012500 | 0.024400 | -2 | -0.021700 | 0.002000 | -0.003284 |
| P2_tag0025_unique015 | -0.002900 | 0.000200 | 0.016900 | 0.035600 | -4 | -0.032900 | 0.003400 | -0.004088 |

## Selection

- Recommended pilot: `P1_tag0025_unique010`
- Reason: P1 is the only protected pilot with a Prefix1 purity gain over the Semantic baseline. P2 has the stronger unique-SID and qrels-collision improvements, but its Prefix1 purity regresses below baseline.

Important caveat: these pilots trained only 3 epochs from the Semantic checkpoint. They improved collision/uniqueness metrics strongly, but Prefix1 gains were small or negative, so this is not yet a final model decision.
