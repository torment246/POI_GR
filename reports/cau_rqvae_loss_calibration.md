# CAU-RQ-VAE Loss Calibration

Calibration run used the same 10k-row smoke setup as the original smoke: seed 42, batch size 512, 2 epochs, Semantic baseline checkpoint initialization, unchanged collision sampler, `unique_margin=0.70`, `label_smoothing=0.05`, and `classifier_dropout=0.10`.

Only weights changed:

- `lambda_tag = 0.025`
- `lambda_unique = 0.15`

## Artifacts

- Output directory: `outputs/experiments/cau_rqvae/smoke_calibrated/`
- Training log: `outputs/experiments/cau_rqvae/smoke_calibrated/train_log.csv`
- Run report: `outputs/experiments/cau_rqvae/smoke_calibrated/report.md`

## Per-Epoch Metrics

| epoch | train_base_total_loss | train_tag_loss | train_weighted_tag_loss | train_weighted_tag_to_base_ratio | train_unique_loss | train_weighted_unique_loss | train_weighted_unique_to_base_ratio | train_valid_collision_pair_count | train_mean_valid_collision_pairs_per_batch | train_zero_pair_batch_rate | train_active_margin_pair_count | train_active_margin_pair_rate | train_collision_pair_mean_cosine | val_recon_cosine | codebook0_usage_rate | codebook1_usage_rate | codebook2_usage_rate | train_tag_accuracy | tag_heldout_accuracy | tag_heldout_macro_f1 | val_online_unique_sid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.776701 | 3.025139 | 0.075602 | 0.097337 | 0.299773 | 0.045146 | 0.058125 | 1600.000000 | 64.000000 | 0.000000 | 728.000000 | 0.455000 | 0.798961 | 0.537469 | 0.804688 | 0.890625 | 0.898438 | 0.072840 | 0.156764 | 0.109683 | 0.958678 |
| 2 | 0.769801 | 2.852497 | 0.071312 | 0.092637 | 0.300741 | 0.044761 | 0.058147 | 1600.000000 | 64.000000 | 0.000000 | 597.000000 | 0.373125 | 0.785576 | 0.537493 | 0.804688 | 0.875000 | 0.914062 | 0.220505 | 0.312813 | 0.220802 | 0.964876 |

## Decision

- First-epoch weighted tag/base ratio: `0.097337`; target `0.05-0.15`.
- First-epoch weighted uniqueness/base ratio: `0.058125`; target `0.04-0.10`.
- Valid collision pairs per batch: `64.00`.
- Zero-pair batch rate: `0.000000`.
- Active margin pair rate: `0.455000`.
- Collision pair mean cosine: `0.798961`.
- Codebook usage rates at final epoch: `0.8047`, `0.8750`, `0.9141`.
- Calibration status: `PASS`.
- Proceed to pilots P1 (`lambda_tag=0.025`, `lambda_unique=0.10`) and P2 (`lambda_tag=0.025`, `lambda_unique=0.15`).
