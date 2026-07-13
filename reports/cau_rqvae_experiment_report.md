# CAU-RQ-VAE Experiment Report

## Final P1 Status

- status: success
- output_dir: `outputs/experiments/cau_rqvae/final_p1`
- command: `CUDA_VISIBLE_DEVICES=0 python scripts/09_train_cau_final.py --config configs/rqvae_cau.yaml --output-dir outputs/experiments/cau_rqvae/final_p1 --lambda-tag 0.025 --lambda-unique 0.10 --unique-margin 0.70 --warmup-epochs 3 --max-total-epochs 80 --early-stop-patience 10 --evaluation-interval 5 --device auto --run-name final_p1`
- device: `cuda`
- GPU: GPU 0, NVIDIA RTX 5880 Ada Generation
- runtime_seconds: 175.25
- epochs_completed: 13
- training_schedule: 3 warmup epochs, then joint fine-tuning; early stopped after 10 joint epochs without validation-loss improvement.
- best_val_loss_epoch: 1
- best_sid_epoch: 13
- best_checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`
- best_val_loss_checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_val_loss.pt`

## Formal Result

- prefix1_semantic_purity: 0.364182
- prefix2_semantic_purity: 0.453719
- prefix3_semantic_purity: 0.786022
- unique_sid_rate: 0.600887
- sid_collision_group_count: 15719.000000
- max_pois_per_sid: 202.000000
- qrels_sid_collision_rate: 0.468927
- unique_pid_rate: 0.971852
- qrels_pid_collision_rate: 0.007499
- reconstruction_cosine: 0.511913
- category_heldout_macro_f1: 0.939720
- codebook_usage_rates: 1.000000 / 1.000000 / 1.000000
- candidates_pid_collision_rate: 0.101370
- dedup_pid_unique_rate: 1.000000

## Baseline/Pilot/Final Compare

| run | prefix1_semantic_purity | prefix3_semantic_purity | unique_sid_rate | qrels_sid_collision_rate | unique_pid_rate | reconstruction_cosine | category_heldout_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| semantic_baseline | 0.352700 | 0.765100 | 0.561700 | 0.494300 | 0.967600 | 0.515736 | NA |
| P1_tag0025_unique010 | 0.353300 | 0.777600 | 0.586100 | 0.472600 | 0.969600 | 0.512452 | 0.672819 |
| P2_tag0025_unique015 | 0.349800 | 0.782000 | 0.597300 | 0.461400 | 0.971000 | 0.511648 | 0.639928 |
| CAU_Final_P1 | 0.364182 | 0.786022 | 0.600887 | 0.468927 | 0.971852 | 0.511913 | 0.939720 |

## Conclusion Wording

Use the final measured values. Do not claim Prefix1 significant improvement, complete collision resolution, hierarchical category semantics, or full HiD-VAE/CQ-SID reproduction unless later evidence supports it.

## Success Criteria

CAU Final P1 met the predefined promotion/protection criteria:

- Prefix1 semantic purity improved from `0.3527` to `0.3642`.
- Prefix3 semantic purity improved from `0.7651` to `0.7860`.
- unique SID rate improved from `0.5617` to `0.6009`.
- qrels SID collision decreased from `0.4943` to `0.4689`.
- max POIs per SID decreased from `211` to `202`.
- unique PID rate improved from `0.9676` to `0.9719`.
- reconstruction cosine decreased by only `0.0038`, within the allowed `0.03`.
- codebook usage stayed at `128/128` for all three codebooks.

## Differences From Pilots

- Final P1 uses the same `lambda_tag=0.025`, `lambda_unique=0.10`, `unique_margin=0.70`, label split, collision sampler, optimizer, learning rate, and batch size as P1 Pilot.
- Final P1 adds the requested 3-epoch classifier warmup and longer joint training with SID-metric checkpoint selection.
- Final P1 improves Prefix1 and held-out Macro-F1 over both pilots.
- P2 still has lower qrels SID collision (`0.4614`) than Final P1 (`0.4689`), but P2 regressed Prefix1 purity and held-out Macro-F1 in the pilot comparison.

## Remaining Issues

- `sid_collision_group_count` increased from `15547` baseline to `15719` final, even though unique SID rate and qrels collision improved.
- Largest full-SID collision decreased from `211` to `202`, but cluster examples still show a large `UNK`/brand-style collision bucket.
- Government/administrative-name and chain/brand cases remain semantically ambiguous.
- Query-to-PID, Trie, LLM category completion, Geo_fused retraining, and embedding rebuild were not run.
