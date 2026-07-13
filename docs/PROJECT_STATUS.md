# Project Status

Last verified from repository files on 2026-07-12.

## Research Goal

Current research goal:

POI metadata -> text/geo representation -> RQ-VAE -> SID -> geohash GID -> PID -> later query-to-PID generative retrieval.

The current repository state is still at the POI-side ID construction and evaluation stage. Query-to-PID formal training has not started in the checked artifacts.

## Completed Pipeline

1. MobilityBench data processing and validation.
2. POI catalog construction for SID training.
3. POI text embedding with local `models/Qwen3-Embedding-0.6B`.
4. Geographic feature construction.
5. Semantic RQ-VAE baseline.
6. Geo_fused RQ-VAE ablation.
7. SID/PID export.
8. SID quality evaluation.
9. SID visualization / PPT material generation.
10. CAU-RQ-VAE P1 final training, formal CAU SID export, and final comparison reports.

## Current Dataset Statistics

Current main MobilityBench files and downstream artifacts agree on these values:

| Item | Value | Verified From |
|---|---:|---|
| POI catalog | 109385 | `data/processed/mobilitybench/pois.csv`, `data/sid/poi_sid_train.parquet` |
| queries | 18011 | `data/processed/mobilitybench/queries.csv` |
| qrels rows | 9735 | `data/processed/mobilitybench/qrels.csv` |
| unique qrel POI | 9657 | `data/processed/mobilitybench/qrels.csv` |
| candidates rows | 63804 | `data/processed/mobilitybench/candidates.csv` |
| unique candidate POI | 62512 | `data/processed/mobilitybench/candidates.csv` |
| candidate queries | 16277 | `data/processed/mobilitybench/candidates.csv`, SID reports |
| text embedding shape | `[109385, 1024]` | `data/embeddings/poi_text_embeddings.npy` |
| embedding meta rows | 109385 | `data/embeddings/poi_embedding_meta.parquet` |
| geo feature shape | `[109385, 54]` | `data/geo/poi_geo_features.npy` |
| geo meta rows | 109385 | `data/geo/poi_geo_meta.parquet` |
| semantic RQ-VAE input shape | `[109385, 256]` | `data/rqvae/rqvae_input_semantic.npy` |
| geo_fused RQ-VAE input shape | `[109385, 310]` | `data/rqvae/rqvae_input_geo_fused.npy` |

Split files currently present under `data/processed/mobilitybench/`:

| Split | queries | qrels |
|---|---:|---:|
| train | 14410 | 7791 |
| dev | 1800 | 965 |
| test | 1801 | 979 |

The file `outputs/data_report/dataset_report.md` contains an older/general MobilityBench summary with `pois=154401`, `queries=27372`, `qrels=10366`, and `candidates=109000`. That does not match the current SID pipeline artifacts. For current SID/RQ-VAE work, use `data/processed/mobilitybench/*`, `outputs/data_report/mobilitybench_quality_report.md`, and downstream reports as the source of truth.

## Baseline Models

| Mode | Checkpoint | Last Checkpoint | Input Path | Input Dim | Best Epoch | Best Val Loss | Recon Cosine | Codebook Usage | Early Stop |
|---|---|---|---|---:|---:|---:|---:|---|---|
| semantic | `outputs/rqvae/rqvae_semantic.pt` | `outputs/rqvae/rqvae_semantic_last.pt` | `data/rqvae/rqvae_input_semantic.npy` | 256 | 142 | 0.81053485 | 0.51573642 | train report final: 128/128, 128/128, 128/128 | True |
| geo_fused | `outputs/rqvae/rqvae_geo_fused.pt` | `outputs/rqvae/rqvae_geo_fused_last.pt` | `data/rqvae/rqvae_input_geo_fused.npy` | 310 | 177 | 0.65069439 | 0.53907679 | train report final: 128/128, 127/128, 128/128; full export report: 128/128 each | True |

Both baseline checkpoints include `model_state_dict`, `optimizer_state_dict`, `model_config`, `mode`, `input_dim`, `best_epoch`, `best_val_loss`, `train_log_path`, `preprocess_path`, `rqvae_input_path`, `random_seed`, `epoch`, `config`, and `preprocess_summary`.

## Baseline SID Metrics

| Metric | semantic | geo_fused |
|---|---:|---:|
| unique SID rate | 0.5617 | 0.5073 |
| unique PID GID6+SID rate | 0.9676 | 0.9472 |
| max POIs per SID | 211 | 158 |
| Prefix1 semantic purity | 0.3527 | 0.3201 |
| Prefix2 semantic purity | 0.4461 | 0.4375 |
| Prefix3 semantic purity | 0.7651 | 0.7518 |
| Prefix1 city purity | 0.0915 | 0.2144 |
| Prefix2 city purity | 0.2765 | 0.4914 |
| Prefix3 city purity | 0.6799 | 0.8290 |
| qrels SID collision rate | 0.4943 | 0.5858 |
| qrels PID collision rate | 0.0082 | 0.0149 |
| candidates PID collision query rate | 0.1073 | 0.1557 |
| dedup PID unique rate | 1.0000 | 1.0000 |

Source reports:

- `reports/sid_quality_semantic.md`
- `reports/sid_quality_geo_fused.md`
- `reports/sid_quality_compare.md`

## Current Conclusion

- Semantic SID is the first-version main line.
- Geo_fused has higher city purity, especially at Prefix2 and Prefix3, but worse SID/PID collision rates.
- CAU Final P1 is the current strongest Semantic-line SID variant for collision-sensitive evaluation. It improves Prefix1/2/3 semantic purity, unique SID rate, qrels SID collision, max POIs per SID, unique PID rate, and qrels/candidates PID collision over the Semantic baseline while preserving reconstruction cosine and full codebook usage.
- CAU Final P1 does not solve all collision structure. Its `sid_collision_group_count` is higher than the Semantic baseline, and large `UNK` / brand / administrative-name clusters remain.
- Main remaining quality issues are mixed `UNK` clusters, administrative-name/category ambiguity, chain/brand collisions, and incomplete coarse category semantics.
- PID is not learned by retraining. It is constructed by concatenating geohash6 GID and SID, with a dedup suffix when needed.
- Formal query-to-PID generative retrieval training has not started.

## CAU Final P1

Formal CAU output was generated on 2026-07-12 under `outputs/experiments/cau_rqvae/final_p1/`.

| Item | Value |
|---|---:|
| lambda_tag | 0.025 |
| lambda_unique | 0.10 |
| unique_margin | 0.70 |
| seed | 42 |
| device | GPU 0, NVIDIA RTX 5880 Ada Generation |
| runtime | 175.25 seconds |
| epochs completed | 13 |
| best validation-loss epoch | 1 |
| best SID-metrics epoch | 13 |
| selected checkpoint | `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt` |
| formal SID indices | `data/sid/poi_sid_indices_cau.npy` |
| formal SID mapping | `data/sid/poi_sid_mapping_cau.parquet` |

Final comparison:

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
| candidates PID collision query rate | 0.1073 | not recorded in pilot CSV | not recorded in pilot CSV | 0.1014 |
| reconstruction cosine | 0.5157 | 0.5125 | 0.5116 | 0.5119 |
| min codebook usage | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| held-out category Macro-F1 | NA | 0.6728 | 0.6399 | 0.9397 |

## Known Path And Report Differences

- The physical workspace root is `/mnt/data2/hudan/poi_genret`, but `pwd` may display the symlink path `/mnt/data1/home/hudan/poi_genret`.
- `.git/` is empty in the inspected sandbox, so `git status`, `git branch --show-current`, and `git diff --stat` fail even though a `.git` directory name exists.
- `README.md` still describes an earlier "data layer only, no training" stage. Current repository artifacts include completed embedding, geo feature, RQ-VAE, SID/PID, evaluation, and visualization stages.
- `scripts/02_embed_poi_text.py` defaults to `--model /model/Qwen3-Embedding-0.6B`, while the actual embedding quality report records `models/Qwen3-Embedding-0.6B`.
- `outputs/data_report/dataset_report.md` contains older/general MobilityBench counts (`154401`, `27372`, `10366`, `109000`) that differ from the current SID pipeline counts (`109385`, `18011`, `9735`, `63804`).

## Current Experiment State

Current CAU-RQ-VAE progress: [Category-Aware and Uniqueness-Regularized RQ-VAE](experiments/EXP-CAU-RQVAE.md).

- Gate 0 baseline recheck: PASS, report at `reports/cau_rqvae_baseline_recheck.md`.
- Gate 1 coarse category labels: PASS, report at `reports/cau_rqvae_label_gate1.md`.
- Original smoke run: stopped before pilots because weighted uniqueness loss / base loss was below the requested initial target range.
- Calibrated smoke: PASS, report at `reports/cau_rqvae_loss_calibration.md`.
- Pilot comparison: completed, reports at `reports/cau_pilot_compare.md` and `reports/cau_pilot_metrics.csv`.
- Recommended pilot candidate: P1 (`lambda_tag=0.025`, `lambda_unique=0.10`), because it preserves protection metrics and is the only pilot with a small Prefix1 purity gain.
- Final P1 training: completed, report at `reports/cau_rqvae_experiment_report.md`.
- Checkpoint selection: `reports/cau_rqvae_checkpoint_selection.md`.
- Formal CAU SID report: `reports/sid_quality_cau.md`.
- Final comparison: `reports/sid_quality_cau_compare.md`.
- PPT metrics: `reports/cau_rqvae_ppt_metrics.csv`.

Do not start query-to-PID, Trie, LLM category completion, new lambda tuning, Geo_fused retraining, or embedding rebuild unless explicitly requested.
