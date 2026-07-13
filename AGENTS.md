# Repository Purpose

- This repository studies Semantic ID / POI ID construction for POI generative retrieval.
- The current pipeline has completed MobilityBench data cleaning, POI text embedding, geographic features, RQ-VAE training, SID/PID export, and quality evaluation.
- The current main line is Semantic SID. Geo_fused is kept as a spatially enhanced ablation.

# Required Reading

Before starting any task, read these files in order:

1. `docs/PROJECT_STATUS.md`
2. `docs/REPO_MAP.md`
3. `docs/DATA_AND_ARTIFACTS.md`
4. `docs/EXPERIMENT_LOG.md`
5. The task-specific file under `docs/experiments/*.md`

# Working Rules

- Locate the existing implementation before changing code; do not duplicate an existing function or script.
- Do not overwrite existing baseline checkpoints, SID/PID mappings, reports, figures, or experiment results.
- New experiments must write to an independent output directory.
- New experiments must save config, command, log, metrics, and comparison report.
- Fix random seeds for all stochastic experiments.
- Before any training run, verify `poi_id`, `row_id`, and row order alignment.
- `qrels` and `candidates` are for evaluation only unless an experiment plan explicitly uses weak supervision.
- Do not commit raw data, model weights, embeddings, checkpoints, or large generated artifacts.
- Do not fabricate experiment results.
- Run a smoke test or pilot before long training.
- After a task or experiment completes, update `docs/PROJECT_STATUS.md` and `docs/EXPERIMENT_LOG.md`.
- Do not create a Git commit unless the user explicitly asks for one.

# Current Conventions

- Python environment: `GR`, observed at `/mnt/data1/home/hudan/anaconda3/envs/GR/bin/python`.
- Project root: physical path `/mnt/data2/hudan/poi_genret`; the shell may display the symlink path `/mnt/data1/home/hudan/poi_genret`.
- Config directory: `configs/`.
- Script directory: `scripts/`.
- Source directory: `src/`.
- Data directory: `data/`.
- Output directory: `outputs/`.
- Report directory: `reports/`.
- Main RQ-VAE config: `configs/rqvae_train.yaml`.
- Common execution style: run scripts from the repository root, for example `python scripts/05_export_and_eval_sid.py --mode both`.
- GPU convention: scripts accept `--device auto`, `--device cpu`, or explicit CUDA devices such as `--device cuda` / `--device cuda:0`. On the 2026-07-12 inspection, `torch.cuda.is_available()` returned `False`; existing embedding reports record earlier CUDA usage.
