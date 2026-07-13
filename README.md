# POI Generative Retrieval: Semantic ID Construction

This repository studies POI-side ID construction for generative retrieval. The current focus is the pipeline:

```text
MobilityBench POI metadata
-> POI text embedding
-> geographic features
-> RQ-VAE
-> Semantic ID (SID)
-> geohash GID + SID PID
-> SID/PID quality evaluation
```

The current main line is Semantic SID. CAU-RQ-VAE P1 is the latest completed experiment and is the strongest Semantic-line SID variant in the checked reports. Query-to-PID generative model training has not started in this repository.

## Repository Contents

Uploaded intentionally:

- Source code: `src/`, `scripts/`
- Configs: `configs/`
- Project documentation: `AGENTS.md`, `docs/`, `reports/*.md`
- Small comparison CSVs: `reports/cau_pilot_metrics.csv`, `reports/cau_rqvae_ppt_metrics.csv`, `reports/cau_rqvae_training_curves.csv`
- Processed MobilityBench dataset: `data/processed/mobilitybench/`

Not uploaded:

- Raw datasets under `data/raw/`
- Local embedding model files under `models/`
- Generated embeddings, geo features, RQ-VAE inputs, SID parquet/npy artifacts
- Training outputs and model checkpoints under `outputs/`
- Large generated report CSVs such as prefix summaries and collision group dumps

This keeps the repository reproducible without committing large intermediate artifacts or model weights.

## Current Results

Final CAU-RQ-VAE P1 was trained from the Semantic RQ-VAE baseline checkpoint locally, using:

- `lambda_tag = 0.025`
- `lambda_unique = 0.10`
- `unique_margin = 0.70`
- seed `42`
- 3 classifier warmup epochs, then joint fine-tuning
- best checkpoint selected by protected SID metrics, not validation loss alone

| Run | Prefix1 purity | Prefix3 purity | Unique SID | Max POIs/SID | qrels SID collision | Unique PID | Recon cosine | Held-out Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Semantic baseline | 0.3527 | 0.7651 | 0.5617 | 211 | 0.4943 | 0.9676 | 0.5157 | NA |
| P1 Pilot | 0.3533 | 0.7776 | 0.5861 | 209 | 0.4726 | 0.9696 | 0.5125 | 0.6728 |
| P2 Pilot | 0.3498 | 0.7820 | 0.5973 | 207 | 0.4614 | 0.9710 | 0.5116 | 0.6399 |
| CAU Final P1 | 0.3642 | 0.7860 | 0.6009 | 202 | 0.4689 | 0.9719 | 0.5119 | 0.9397 |

Interpretation: CAU Final P1 improves SID uniqueness and qrels collision while preserving coarse semantic purity and reconstruction quality. It does not fully solve collision structure: `sid_collision_group_count` increased, and `UNK`, government/administrative-name, and brand/chain clusters remain mixed in some cases.

See:

- `docs/PROJECT_STATUS.md`
- `docs/experiments/EXP-CAU-RQVAE.md`
- `reports/sid_quality_cau_compare.md`
- `reports/cau_rqvae_experiment_report.md`
- `reports/cau_rqvae_cluster_examples.md`

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching your CUDA driver if the default wheel is not suitable.

The embedding step expects a local sentence-transformer model. The checked experiment used:

```text
models/Qwen3-Embedding-0.6B
```

Model weights are intentionally not committed. Place the local model there or pass another local path to `scripts/02_embed_poi_text.py --model`.

## Data

The repository includes the processed MobilityBench CSV dataset:

```text
data/processed/mobilitybench/
  pois.csv
  queries.csv
  qrels.csv
  candidates.csv
  train_queries.csv
  dev_queries.csv
  test_queries.csv
  train_qrels.csv
  dev_qrels.csv
  test_qrels.csv
  split_manifest.json
  dataset_summary.json
```

Current checked counts:

| Item | Count |
|---|---:|
| POIs | 109385 |
| queries | 18011 |
| qrels rows | 9735 |
| candidates rows | 63804 |

Raw MobilityBench input files are not committed.

## Reproducing The POI-Side Pipeline

Run commands from the repository root.

Build the POI training catalog:

```bash
python scripts/01_build_poi_sid_train_data.py
```

Build text embeddings with a local embedding model:

```bash
python scripts/02_embed_poi_text.py --model models/Qwen3-Embedding-0.6B --device auto
```

Build geographic features:

```bash
python scripts/03_build_geo_features.py
```

Train baseline RQ-VAE models:

```bash
python scripts/04_train_rqvae.py --config configs/rqvae_train.yaml --mode semantic
python scripts/04_train_rqvae.py --config configs/rqvae_train.yaml --mode geo_fused
```

Export and evaluate baseline SID/PID:

```bash
python scripts/05_export_and_eval_sid.py --mode both
```

Build CAU labels:

```bash
python scripts/07_build_cau_labels.py --config configs/rqvae_cau.yaml
```

Run CAU pilots if needed:

```bash
python scripts/08_train_cau_rqvae.py \
  --config configs/rqvae_cau.yaml \
  --output-dir outputs/experiments/cau_rqvae/pilot_tag0025_unique010 \
  --lambda-tag 0.025 \
  --lambda-unique 0.10
```

Run the latest final P1 experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/09_train_cau_final.py \
  --config configs/rqvae_cau.yaml \
  --output-dir outputs/experiments/cau_rqvae/final_p1 \
  --lambda-tag 0.025 \
  --lambda-unique 0.10 \
  --unique-margin 0.70 \
  --warmup-epochs 3 \
  --max-total-epochs 80 \
  --early-stop-patience 10 \
  --evaluation-interval 5 \
  --device auto \
  --run-name final_p1
```

Generated artifacts will appear under `data/embeddings/`, `data/geo/`, `data/rqvae/`, `data/sid/`, `outputs/`, and `reports/`. Most of those are ignored by Git.

## Documentation Map

- `AGENTS.md`: stable repository working rules for future Codex sessions.
- `docs/PROJECT_STATUS.md`: latest project status and current metrics.
- `docs/REPO_MAP.md`: source files, scripts, commands, and dependencies.
- `docs/DATA_AND_ARTIFACTS.md`: data contracts, artifacts, and Git policy.
- `docs/EXPERIMENT_LOG.md`: append-only experiment history.
- `docs/experiments/EXP-CAU-RQVAE.md`: CAU-RQ-VAE plan, decisions, and final outcome.

## Git Policy

The repository is configured to avoid committing raw data, local model files, generated arrays, parquet artifacts, checkpoints, and large output directories. Only processed MobilityBench CSV/JSON files are intentionally tracked under `data/`.

Do not commit:

- `models/`
- `outputs/`
- `data/embeddings/`
- `data/geo/`
- `data/rqvae/`
- `data/sid/`
- `*.npy`, `*.parquet`, `*.pt`, `*.safetensors`
