# Experiment Log

Append new experiments to this file. Dates below are based on repository file modification times where available, not guessed wall-clock run dates.

| Experiment ID | Date | Method | Status | Input | Output | Key Result | Notes |
|---|---|---|---|---|---|---|---|
| EXP-RQVAE-SEMANTIC | 2026-06-27 | Semantic RQ-VAE baseline | completed | `data/rqvae/rqvae_input_semantic.npy` (`[109385, 256]`) | `outputs/rqvae/rqvae_semantic.pt` | best epoch 142, best val loss 0.81053485, final recon cosine 0.51573642 | Main-line SID source. Early stopped. Seed 42. |
| EXP-RQVAE-GEO-FUSED | 2026-06-27 | Geo_fused RQ-VAE ablation | completed | `data/rqvae/rqvae_input_geo_fused.npy` (`[109385, 310]`) | `outputs/rqvae/rqvae_geo_fused.pt` | best epoch 177, best val loss 0.65069439, final recon cosine 0.53907679 | Spatially enhanced ablation. Early stopped. Seed 42. |
| EXP-SID-EVAL | 2026-06-28 | SID/PID export and quality evaluation | completed | semantic and geo_fused checkpoints plus aligned metadata | `data/sid/poi_sid_mapping_*.parquet`, `reports/sid_quality_*.md` | semantic recommended as first target; geo_fused city purity higher but collisions worse | `reports/sid_quality_compare.md` is the main comparison report. |
| EXP-SID-VIS | 2026-06-28 | SID quality visualization | completed | SID reports and mappings | `outputs/figures/sid_quality/` | generated seven PPT-ready figures | Figure README lists PNG/SVG paths and usage notes. |
| EXP-CAU-RQVAE | 2026-07-12 | Category-Aware and Uniqueness-Regularized RQ-VAE | completed | semantic baseline input, Gate 1 coarse labels, baseline collision groups | `outputs/experiments/cau_rqvae/final_p1/`, `data/sid/poi_sid_mapping_cau.parquet` | CAU Final P1 selected epoch 13; Prefix1 0.3642, unique SID 0.6009, qrels SID collision 0.4689 | P1 final succeeded; P2 remains weight ablation. See `reports/cau_rqvae_experiment_report.md`. |

## EXP-RQVAE-SEMANTIC

- Command shape: `python scripts/04_train_rqvae.py --config configs/rqvae_train.yaml --mode semantic`
- Config: `configs/rqvae_train.yaml`
- Checkpoint: `outputs/rqvae/rqvae_semantic.pt`
- Last checkpoint: `outputs/rqvae/rqvae_semantic_last.pt`
- Train log: `outputs/rqvae/train_log_semantic.csv`
- Report: `outputs/rqvae/rqvae_train_report_semantic.md`
- Preprocess: `outputs/rqvae/preprocess_semantic.pkl`
- Reproducible: yes, if local dependencies and inputs are unchanged.
- Git commit/hash: unavailable in current inspection because `.git/` is empty in the sandbox and `git status` fails.

## EXP-RQVAE-GEO-FUSED

- Command shape: `python scripts/04_train_rqvae.py --config configs/rqvae_train.yaml --mode geo_fused`
- Config: `configs/rqvae_train.yaml`
- Checkpoint: `outputs/rqvae/rqvae_geo_fused.pt`
- Last checkpoint: `outputs/rqvae/rqvae_geo_fused_last.pt`
- Train log: `outputs/rqvae/train_log_geo_fused.csv`
- Report: `outputs/rqvae/rqvae_train_report_geo_fused.md`
- Preprocess: `outputs/rqvae/preprocess_geo_fused.pkl`
- Reproducible: yes, if local dependencies and inputs are unchanged.
- Git commit/hash: unavailable in current inspection because `.git/` is empty in the sandbox and `git status` fails.

## EXP-SID-EVAL

- Command shape: `python scripts/05_export_and_eval_sid.py --mode both`
- Semantic mapping: `data/sid/poi_sid_mapping_semantic.parquet`
- Geo_fused mapping: `data/sid/poi_sid_mapping_geo_fused.parquet`
- Semantic report: `reports/sid_quality_semantic.md`
- Geo_fused report: `reports/sid_quality_geo_fused.md`
- Compare report: `reports/sid_quality_compare.md`
- Reproducible: yes, from the saved checkpoints and aligned input artifacts.

## EXP-CAU-RQVAE

- Plan: `docs/experiments/EXP-CAU-RQVAE.md`
- Status: completed; final P1 training, formal CAU SID export, and final reports completed.
- Gate 0 report: `reports/cau_rqvae_baseline_recheck.md`
- Gate 1 report: `reports/cau_rqvae_label_gate1.md`
- Smoke report: `outputs/experiments/cau_rqvae/smoke/report.md`
- Calibrated smoke report: `reports/cau_rqvae_loss_calibration.md`
- Pilot compare report: `reports/cau_pilot_compare.md`
- Pilot metrics: `reports/cau_pilot_metrics.csv`
- Experiment report: `reports/cau_rqvae_experiment_report.md`
- Final output dir: `outputs/experiments/cau_rqvae/final_p1/`
- Final command: `CUDA_VISIBLE_DEVICES=0 python scripts/09_train_cau_final.py --config configs/rqvae_cau.yaml --output-dir outputs/experiments/cau_rqvae/final_p1 --lambda-tag 0.025 --lambda-unique 0.10 --unique-margin 0.70 --warmup-epochs 3 --max-total-epochs 80 --early-stop-patience 10 --evaluation-interval 5 --device auto --run-name final_p1`
- Final checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`
- Best validation-loss checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_val_loss.pt` at epoch 1.
- Best SID-metrics checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt` at epoch 13.
- Formal SID indices: `data/sid/poi_sid_indices_cau.npy`
- Formal SID mapping: `data/sid/poi_sid_mapping_cau.parquet`
- Formal SID report: `reports/sid_quality_cau.md`
- Final comparison report: `reports/sid_quality_cau_compare.md`
- Checkpoint selection report: `reports/cau_rqvae_checkpoint_selection.md`
- Training curves: `reports/cau_rqvae_training_curves.csv`
- PPT metrics: `reports/cau_rqvae_ppt_metrics.csv`
- Cluster examples: `reports/cau_rqvae_cluster_examples.md`
- Gate 0 result: PASS; baseline export/eval was reproducible with exact indices and mapping key-column matches.
- Gate 1 result: PASS; 65147 high-confidence supervised labels, 52120 tag-train, 13027 tag-heldout, 44238 unsupervised.
- Smoke result: category loss scale was within target, uniqueness loss scale was below target (`0.020082` first-epoch weighted uniqueness/base ratio), so P1/P2/P3 pilots were not started.
- Calibrated smoke result: PASS with `lambda_tag=0.025`, `lambda_unique=0.15`; first-epoch ratios were tag/base `0.097337` and uniqueness/base `0.058125`.
- Pilot result: P1 (`lambda_tag=0.025`, `lambda_unique=0.10`) is recommended. P1 Prefix1 purity `0.3533`, unique SID `0.5861`, qrels SID collision `0.4726`; P2 improves uniqueness/collision more but Prefix1 drops to `0.3498`.
- Final result: CAU Final P1 used `lambda_tag=0.025`, `lambda_unique=0.10`, `unique_margin=0.70`, seed 42, GPU 0 NVIDIA RTX 5880 Ada Generation. It completed 13 epochs in 175.25 seconds and early-stopped after 10 joint epochs without validation-loss improvement.
- Final metrics: Prefix1/2/3 semantic purity `0.3642 / 0.4537 / 0.7860`, unique SID rate `0.6009`, max POIs per SID `202`, qrels SID collision `0.4689`, unique PID rate `0.9719`, qrels PID collision `0.0075`, reconstruction cosine `0.5119`, min codebook usage `1.0000`, held-out category Macro-F1 `0.9397`.
- Reproducible: yes, from saved config, command, environment, code checksums, label artifacts, Semantic baseline checkpoint, and aligned input artifacts.
- Git commit/hash: unavailable because current `.git/` metadata is invalid/empty in this sandbox.
