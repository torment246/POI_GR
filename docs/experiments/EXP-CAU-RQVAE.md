# Category-Aware and Uniqueness-Regularized RQ-VAE

Short name: CAU-RQ-VAE

Status: completed. Gate 0, Gate 1, smoke calibration, pilots, final P1 training, formal CAU SID export, and final reports have been run.

Update on 2026-07-12: Gate 0 and Gate 1 passed. The original 10k-row smoke run completed, but the first-epoch weighted uniqueness loss / base loss ratio was `0.020082`, below the requested range. A calibrated smoke with `lambda_tag=0.025` and `lambda_unique=0.15` passed with first-epoch ratios `0.097337` and `0.058125`. Two pilots were run and compared; P1 (`lambda_tag=0.025`, `lambda_unique=0.10`) is the recommended candidate. See `reports/cau_rqvae_experiment_report.md` and `reports/cau_pilot_compare.md`.

Final update on 2026-07-12: P1 was promoted as the only final configuration. The selected final checkpoint is epoch 13 at `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`. Formal CAU SID files were exported to `data/sid/poi_sid_indices_cau.npy` and `data/sid/poi_sid_mapping_cau.parquet`. See `reports/sid_quality_cau_compare.md`, `reports/cau_rqvae_checkpoint_selection.md`, and `reports/cau_rqvae_cluster_examples.md`.

## Motivation

The semantic baseline is the current main line, but its SID quality report shows:

- Prefix1 semantic purity is low: `0.3527`.
- Full SID unique rate is `0.5617`.
- qrels target SID collision rate is `0.4943`.
- max POIs per SID is `211`.
- Many low-purity Prefix1 and collision groups mix missing-category / `UNK` POIs.

Geo_fused improves city purity but worsens collision metrics, so the next experiment should improve coarse semantic structure and full-SID uniqueness while staying close to the semantic baseline.

## Goal

Starting from the existing Semantic RQ-VAE setup, add:

1. First-layer coarse category supervision.
2. Full SID collision uniqueness regularization.

Primary optimization targets:

- Prefix1 semantic purity.
- unique SID rate.
- qrels SID collision rate.
- max POIs per SID.

Protected metrics:

- Prefix3 semantic purity.
- unique PID rate.
- qrels PID collision rate.
- reconstruction cosine.
- codebook usage.

## Baseline To Compare Against

Semantic baseline:

- Input: `data/rqvae/rqvae_input_semantic.npy`
- Checkpoint: `outputs/rqvae/rqvae_semantic.pt`
- Best epoch: 142
- Best val loss: 0.81053485
- Recon cosine: 0.51573642
- unique SID rate: 0.5617
- unique PID rate: 0.9676
- Prefix1/2/3 semantic purity: 0.3527 / 0.4461 / 0.7651
- qrels SID collision rate: 0.4943
- qrels PID collision rate: 0.0082
- max POIs per SID: 211

Geo_fused remains an ablation reference, not the starting point.

## Proposed Method

### Category-Aware Prefix1 Supervision

Final implementation uses high-confidence coarse labels from existing POI metadata:

- Prefer `category_l1` when present and meaningful.
- Fall back to `tags` when usable.
- Fall back to strong `name` keyword rules for high-confidence cases.
- Keep `UNK` in reconstruction training and final SID evaluation.
- Exclude `UNK` and unsupervised rows from category supervision.
- Use an 80/20 category-stratified split of high-confidence labels; held-out labels do not participate in `L_tag`.

Implemented model changes:

- Add a lightweight classifier head from encoder latent `h` to coarse category labels.
- Apply category supervision through the encoder latent classifier.
- Keep reconstruction, commitment, and codebook losses from the existing baseline.

Potential loss form:

```text
loss = recon_loss
     + beta * commitment_loss
     + codebook_loss_weight * codebook_loss
     + lambda_category * prefix1_category_loss
     + lambda_unique * uniqueness_loss
```

### Full SID Collision Uniqueness Regularization

Goal: reduce large groups sharing identical `(sid0, sid1, sid2)`.

Implemented uniqueness regularization:

- Build offline sampling groups from Semantic baseline full-SID collisions.
- Use collision-aware sampling for about 25% of each batch.
- Compute uniqueness loss only for sampled pairs that still share the current full SID.
- Exclude possible duplicates using normalized name/address/coordinate signatures.
- Do not build a global pair matrix.

`qrels` and `candidates` were used only for evaluation, not training.

## Planned Files And Output Isolation

Do not overwrite baseline files.

Implemented files:

- Config: `configs/rqvae_cau.yaml`
- Label builder: `scripts/07_build_cau_labels.py`
- Pilot trainer: `scripts/08_train_cau_rqvae.py`
- Final trainer/exporter: `scripts/09_train_cau_final.py`
- Reused input: `data/rqvae/rqvae_input_semantic.npy`
- Final output directory: `outputs/experiments/cau_rqvae/final_p1/`
- Formal SID indices: `data/sid/poi_sid_indices_cau.npy`
- Formal SID mapping: `data/sid/poi_sid_mapping_cau.parquet`
- Final reports: `reports/sid_quality_cau.md`, `reports/sid_quality_cau_compare.md`, `reports/cau_rqvae_experiment_report.md`, `reports/cau_rqvae_ppt_metrics.csv`, `reports/cau_rqvae_checkpoint_selection.md`, `reports/cau_rqvae_training_curves.csv`, `reports/cau_rqvae_cluster_examples.md`

## Smoke And Pilot Plan

Before any long training:

1. Re-check `poi_id` and `row_id` alignment across SID train data, embedding meta, geo meta, and semantic input.
2. Build or load category labels and report label coverage.
3. Run a smoke test with a small row subset and a separate output path.
4. Verify checkpoint load, train log, report writing, and SID export on the smoke output.
5. Run a short pilot with fixed seed and compare against semantic baseline on the same evaluation scripts.
6. Only then consider full training.

## Final Run

Final P1 configuration:

- `lambda_tag = 0.025`
- `lambda_unique = 0.10`
- `unique_margin = 0.70`
- `label_smoothing = 0.05`
- `classifier_dropout = 0.10`
- `seed = 42`
- optimizer: AdamW
- learning rate: `0.0005`
- batch size: `1024`
- scheduler: none
- warmup epochs: 3
- max total epochs: 80
- early stop patience: 10
- evaluation interval: 5 epochs

Actual run:

- command: `CUDA_VISIBLE_DEVICES=0 python scripts/09_train_cau_final.py --config configs/rqvae_cau.yaml --output-dir outputs/experiments/cau_rqvae/final_p1 --lambda-tag 0.025 --lambda-unique 0.10 --unique-margin 0.70 --warmup-epochs 3 --max-total-epochs 80 --early-stop-patience 10 --evaluation-interval 5 --device auto --run-name final_p1`
- device: GPU 0, NVIDIA RTX 5880 Ada Generation
- runtime: 175.25 seconds
- epochs completed: 13
- best validation-loss epoch: 1
- best SID-metrics epoch: 13
- selected checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`

Final metrics:

| Metric | Semantic baseline | P1 Pilot | P2 Pilot | CAU Final P1 |
|---|---:|---:|---:|---:|
| Prefix1 semantic purity | 0.3527 | 0.3533 | 0.3498 | 0.3642 |
| Prefix2 semantic purity | 0.4461 | 0.4477 | 0.4463 | 0.4537 |
| Prefix3 semantic purity | 0.7651 | 0.7776 | 0.7820 | 0.7860 |
| unique SID rate | 0.5617 | 0.5861 | 0.5973 | 0.6009 |
| SID collision group count | 15547 | 15430 | 15433 | 15719 |
| max POIs per SID | 211 | 209 | 207 | 202 |
| qrels SID collision rate | 0.4943 | 0.4726 | 0.4614 | 0.4689 |
| unique PID rate | 0.9676 | 0.9696 | 0.9710 | 0.9719 |
| qrels PID collision rate | 0.0082 | 0.0074 | 0.0071 | 0.0075 |
| reconstruction cosine | 0.5157 | 0.5125 | 0.5116 | 0.5119 |
| min codebook usage | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| held-out category Macro-F1 | NA | 0.6728 | 0.6399 | 0.9397 |

## Acceptance Criteria

A CAU run is worth promoting only if it improves primary metrics without damaging protected metrics.

Primary desired direction:

- Prefix1 semantic purity increases over `0.3527`.
- unique SID rate increases over `0.5617`.
- qrels SID collision rate decreases below `0.4943`.
- max POIs per SID decreases below `211`.

Protection thresholds to define before full training:

- Prefix3 semantic purity should not materially drop from `0.7651`.
- unique PID rate should stay close to `0.9676`.
- qrels PID collision rate should stay close to `0.0082`.
- codebook usage should not collapse.
- reconstruction cosine should not materially drop from `0.51573642`.

Final result against criteria:

- Prefix1 semantic purity improved from `0.3527` to `0.3642`.
- Prefix3 semantic purity improved from `0.7651` to `0.7860`.
- unique SID rate improved from `0.5617` to `0.6009`.
- qrels SID collision decreased from `0.4943` to `0.4689`.
- max POIs per SID decreased from `211` to `202`.
- unique PID rate improved from `0.9676` to `0.9719`.
- qrels PID collision decreased from `0.0082` to `0.0075`.
- reconstruction cosine decreased only from `0.5157` to `0.5119`.
- codebook usage remained `128/128` for all codebooks.
- `sid_collision_group_count` increased from `15547` to `15719`, so collision structure is not fully solved.

Conclusion: CAU Final P1 met the promotion criteria. The most accurate claim is that, in this run, category supervision plus uniqueness regularization improved SID uniqueness and qrels collision while preserving coarse semantic purity and reconstruction quality. It did not fully solve large `UNK`, brand, or administrative-name collision cases.

## Resolved Decisions And Remaining Issues

- `UNK` remains in reconstruction/evaluation but is excluded from category supervision.
- Coarse training labels use category_l1 -> tags -> strong name keyword high-confidence rules, not full `sid_eval.py` inferred labels.
- P1 is the only final configuration; P2 is retained as a weight ablation.
- Formal CAU SID was exported only after `best_sid_metrics.pt` passed protection conditions.
- Remaining issue: final `sid_collision_group_count` increased despite improvements in unique SID rate, qrels collision, max POIs per SID, and PID metrics.
- Remaining issue: cluster examples still show unresolved `UNK`, government/administrative-name, and brand/chain ambiguity.
