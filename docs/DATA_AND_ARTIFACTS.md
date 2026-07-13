# Data and Artifacts

Last verified from repository files on 2026-07-12.

## Alignment Contract

The current SID pipeline relies on row-wise alignment across POI metadata, embeddings, geo features, RQ-VAE inputs, and SID mappings.

The following must hold before training or SID export:

- Row counts match across metadata and feature arrays.
- `poi_id` order matches.
- `row_id` order matches.
- `poi_id` is unique in POI catalog, SID train data, embedding meta, geo meta, and SID mapping files.
- Numeric arrays contain no NaN or Inf.
- `qrels` and `candidates` POI references exist in the SID catalog.
- `qrels` and `candidates` are not used for training unless the experiment plan explicitly changes the contract.

Current inspection found these alignment checks true:

- `data/sid/poi_sid_train.parquet` and `data/embeddings/poi_embedding_meta.parquet` have the same `poi_id` and `row_id` order.
- `data/embeddings/poi_embedding_meta.parquet` and `data/geo/poi_geo_meta.parquet` have the same `poi_id` and `row_id` order.
- Both semantic and geo_fused SID mappings keep the same `poi_id` order as embedding metadata.
- Duplicate `poi_id` count is 0 in SID train, embedding meta, geo meta, and both SID mappings.

## Main Data Files

| Path | Shape / Rows | dtype | Main Fields | Generating Script | Rebuild Allowed | Git Policy |
|---|---:|---|---|---|---|---|
| `data/processed/mobilitybench/pois.csv` | 109385 rows | CSV object columns | `poi_id`, `name`, `address`, `city`, `lat`, `lon`, `category_l1`, `tags` | `scripts/process_mobilitybench.py` | Yes, only into controlled output and after backup/confirmation | Do not commit |
| `data/processed/mobilitybench/queries.csv` | 18011 rows | CSV object columns | `query_id`, `query_text`, `city`, `user_lat`, `user_lon`, `query_type` | `scripts/process_mobilitybench.py` | Yes, with split integrity checks | Do not commit |
| `data/processed/mobilitybench/qrels.csv` | 9735 rows | CSV object columns | `query_id`, `poi_id`, `label`, `source` | `scripts/process_mobilitybench.py` | Yes, with reference checks | Do not commit |
| `data/processed/mobilitybench/candidates.csv` | 63804 rows | CSV object columns | `query_id`, `poi_id`, `rank`, `score`, `candidate_source` | `scripts/process_mobilitybench.py` | Yes, with reference checks | Do not commit |
| `data/sid/poi_sid_train.parquet` | 109385 rows | parquet mixed | `row_id`, `poi_id`, POI fields, `parse_warning`, `poi_json`, `poi_text` | `scripts/01_build_poi_sid_train_data.py` | Yes, from processed POIs | Do not commit |
| `data/embeddings/poi_text_embeddings.npy` | `[109385, 1024]` | `float32` | text embedding matrix | `scripts/02_embed_poi_text.py` | Expensive; only rebuild intentionally | Do not commit |
| `data/embeddings/poi_embedding_meta.parquet` | 109385 rows | parquet mixed | `row_id`, `poi_id`, POI fields, `embedding_text`, `embedding_text_len` | `scripts/02_embed_poi_text.py` | Yes, must stay aligned with embeddings | Do not commit |
| `data/geo/poi_geo_features.npy` | `[109385, 54]` | `float32` | standardized geo feature matrix | `scripts/03_build_geo_features.py` | Yes, deterministic with current seed/anchors | Do not commit |
| `data/geo/poi_geo_meta.parquet` | 109385 rows | parquet mixed | `row_id`, `poi_id`, `lat`, `lon`, `geohash5/6/7` | `scripts/03_build_geo_features.py` | Yes, must stay aligned with features | Do not commit |
| `data/rqvae/rqvae_input_semantic.npy` | `[109385, 256]` | `float32` | PCA/scaled semantic input | `scripts/04_train_rqvae.py` via `src/rqvae_preprocess.py` | Yes, only with saved preprocess/config | Do not commit |
| `data/rqvae/rqvae_input_geo_fused.npy` | `[109385, 310]` | `float32` | semantic input plus `0.1 * geo_features` | `scripts/04_train_rqvae.py` via `src/rqvae_preprocess.py` | Yes, only with saved preprocess/config | Do not commit |
| `data/sid/poi_sid_indices_semantic.npy` | `[109385, 3]` | `int64` | RQ-VAE code indices | `scripts/05_export_and_eval_sid.py` | Yes, from checkpoint and input | Do not commit |
| `data/sid/poi_sid_indices_geo_fused.npy` | `[109385, 3]` | `int64` | RQ-VAE code indices | `scripts/05_export_and_eval_sid.py` | Yes, from checkpoint and input | Do not commit |
| `data/sid/poi_sid_indices_cau.npy` | `[109385, 3]` | `int64` | CAU Final P1 RQ-VAE code indices | `scripts/09_train_cau_final.py` using `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt` | Yes, from checkpoint and input | Do not commit |
| `data/sid/poi_sid_mapping_semantic.parquet` | 109385 rows | parquet mixed | POI fields, geohash, SID, GID, PID, dedup PID | `scripts/05_export_and_eval_sid.py` | Yes, from checkpoint/input/meta | Do not commit |
| `data/sid/poi_sid_mapping_geo_fused.parquet` | 109385 rows | parquet mixed | POI fields, geohash, SID, GID, PID, dedup PID | `scripts/05_export_and_eval_sid.py` | Yes, from checkpoint/input/meta | Do not commit |
| `data/sid/poi_sid_mapping_cau.parquet` | 109385 rows | parquet mixed | CAU Final P1 POI fields, geohash, SID, GID, PID, dedup PID | `scripts/09_train_cau_final.py` using `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt` | Yes, from checkpoint/input/meta | Do not commit |
| `outputs/experiments/cau_rqvae/labels/cau_coarse_category_labels.parquet` | 109385 rows | parquet mixed | high-confidence coarse label, train/heldout masks, source fields | `scripts/07_build_cau_labels.py` | Yes, from current POI metadata/rules | Do not commit |

Numeric sample checks on `.npy` arrays found no NaN or Inf in inspected samples. Existing reports also record full numeric checks with `has_nan: False` and `has_inf: False`.

## Checkpoints

| Path | Mode | Input Dim | Best Epoch | Best Val Loss | Epoch Stored | Role | Git Policy |
|---|---|---:|---:|---:|---:|---|---|
| `outputs/rqvae/rqvae_semantic.pt` | semantic | 256 | 142 | 0.8105348544030112 | 142 | baseline best checkpoint | Do not commit |
| `outputs/rqvae/rqvae_semantic_last.pt` | semantic | 256 | 142 | 0.8105348544030112 | 152 | baseline last checkpoint | Do not commit |
| `outputs/rqvae/rqvae_geo_fused.pt` | geo_fused | 310 | 177 | 0.6506943919580627 | 177 | ablation best checkpoint | Do not commit |
| `outputs/rqvae/rqvae_geo_fused_last.pt` | geo_fused | 310 | 177 | 0.6506943919580627 | 187 | ablation last checkpoint | Do not commit |
| `outputs/rqvae/rqvae_semantic_debug.pt` | semantic | 256 | 2 | 1.0698330402374268 | 2 | debug checkpoint | Do not commit |
| `outputs/rqvae/rqvae_semantic_debug_last.pt` | semantic | 256 | 2 | 1.0698330402374268 | 3 | debug last checkpoint | Do not commit |
| `outputs/experiments/cau_rqvae/final_p1/best_val_loss.pt` | cau | 256 | 1 | 0.8105348282789787 | 1 | CAU Final P1 best validation-loss checkpoint; classifier warmup only | Do not commit |
| `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt` | cau | 256 | 13 | 0.8105348282789787 | 13 | CAU Final P1 selected formal SID checkpoint by protected SID metrics | Do not commit |
| `outputs/experiments/cau_rqvae/final_p1/last.pt` | cau | 256 | 1 | 0.8105348282789787 | 13 | CAU Final P1 last checkpoint | Do not commit |

Checkpoint top-level keys include:

`model_state_dict`, `optimizer_state_dict`, `model_config`, `mode`, `input_dim`, `best_epoch`, `best_val_loss`, `train_log_path`, `preprocess_path`, `rqvae_input_path`, `random_seed`, `epoch`, `config`, `preprocess_summary`.

## Reports

Important current reports:

- `outputs/data_report/mobilitybench_quality_report.md`: current processed MobilityBench validation summary.
- `reports/poi_parse_coverage_report.md`: POI JSON/text coverage and missing field report.
- `reports/poi_id_reference_integrity_report.md`: alignment and qrels/candidates reference integrity.
- `reports/poi_embedding_quality_report.md`: embedding shape, model path, field coverage, and numeric checks.
- `reports/poi_geo_quality_report.md`: geo feature shape, coordinate ranges, geohash coverage, anchors, and numeric checks.
- `outputs/rqvae/rqvae_train_report_semantic.md`: semantic RQ-VAE train report.
- `outputs/rqvae/rqvae_train_report_geo_fused.md`: geo_fused RQ-VAE train report.
- `reports/sid_quality_semantic.md`: semantic SID quality report.
- `reports/sid_quality_geo_fused.md`: geo_fused SID quality report.
- `reports/sid_quality_compare.md`: side-by-side SID comparison and recommendation.
- `reports/cau_rqvae_baseline_recheck.md`: Gate 0 baseline re-export/eval reproducibility report.
- `reports/cau_rqvae_label_gate1.md`: Gate 1 coarse label coverage and split report.
- `reports/cau_rqvae_loss_calibration.md`: calibrated smoke loss-scale report.
- `reports/cau_pilot_compare.md`: P1/P2 pilot comparison and P1 recommendation.
- `reports/cau_pilot_metrics.csv`: Semantic baseline, P1 pilot, and P2 pilot metrics.
- `reports/sid_quality_cau.md`: formal CAU Final P1 SID quality report.
- `reports/sid_quality_cau_compare.md`: Semantic baseline, P1 pilot, P2 pilot, and CAU Final P1 comparison.
- `reports/cau_rqvae_experiment_report.md`: final CAU experiment report with command, runtime, GPU, best checkpoint, and final metrics.
- `reports/cau_rqvae_checkpoint_selection.md`: SID-metric checkpoint selection history and protection checks.
- `reports/cau_rqvae_training_curves.csv`: CAU Final P1 epoch-level training metrics.
- `reports/cau_rqvae_ppt_metrics.csv`: PPT-ready metric table for Semantic baseline, P1, P2, and CAU Final P1.
- `reports/cau_rqvae_cluster_examples.md`: diagnostic cluster examples including successful and still-failed cases.
- `outputs/figures/sid_quality/README.md`: generated figure list and PPT usage notes.

CAU Final P1 experiment directory:

- `outputs/experiments/cau_rqvae/final_p1/config.yaml`
- `outputs/experiments/cau_rqvae/final_p1/command.txt`
- `outputs/experiments/cau_rqvae/final_p1/environment.txt`
- `outputs/experiments/cau_rqvae/final_p1/train_log.csv`
- `outputs/experiments/cau_rqvae/final_p1/sid_eval_history.csv`
- `outputs/experiments/cau_rqvae/final_p1/metrics.json`
- `outputs/experiments/cau_rqvae/final_p1/best_val_loss.pt`
- `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`
- `outputs/experiments/cau_rqvae/final_p1/last.pt`
- `outputs/experiments/cau_rqvae/final_p1/stdout.log`
- `outputs/experiments/cau_rqvae/final_p1/README.md`
- `outputs/experiments/cau_rqvae/final_p1/code_checksums.txt`
- `outputs/experiments/cau_rqvae/final_p1/poi_sid_indices_cau.npy`
- `outputs/experiments/cau_rqvae/final_p1/poi_sid_mapping_cau.parquet`

Note: `outputs/data_report/dataset_report.md` contains an older/general dataset summary with MobilityBench counts that differ from the current SID artifacts. Do not use it as the current SID pipeline source of truth without reconciling paths.

## Git Policy

Default not to commit:

- `data/`
- `models/`
- `outputs/`
- `*.npy`
- `*.npz`
- `*.parquet`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.safetensors`
- model binaries and embedding files
- large CSVs
- large training logs
- cache directories such as `__pycache__/` and notebook checkpoints

Small source files, configs, and docs are intended to be commit candidates when Git metadata is available and the user asks for a commit.
