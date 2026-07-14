# Experiment Log

Keep one compact row per meaningful experiment. Large logs, checkpoints and generated reports belong in ignored experiment output directories.

| Experiment | Dataset | Config | Commit | Main result | Artifact status |
|---|---|---|---|---|---|
| Semantic RQ-VAE | MobilityBench | `configs/rqvae_train.yaml` | pre-baseline history | best epoch 142; reconstruction cosine 0.5157; unique SID 0.5617 | legacy; do not resume |
| Geo-fused RQ-VAE | MobilityBench | `configs/rqvae_train.yaml` | pre-baseline history | stronger city purity but worse collision metrics than Semantic | legacy ablation |
| SID/PID evaluation | MobilityBench | Semantic/geo-fused export | pre-baseline history | Semantic selected as first main line; PID dedup reached full uniqueness | results summarized only |
| CAU pilots P1/P2 | MobilityBench | `configs/rqvae_cau.yaml` | pre-baseline history | P1 preserved quality best; P2 improved uniqueness with Prefix1 trade-off | metrics in `reports/metrics/cau_pilot_metrics.csv` |
| CAU Final P1 | MobilityBench | `configs/rqvae_cau.yaml` | pre-baseline history | Prefix1 0.3642; unique SID 0.6009; qrels SID collision 0.4689; held-out Macro-F1 0.9397 | legacy; retrain for Didi |

## Logging Convention

- Add a row or a short subsection only after a reproducible experiment finishes.
- Record dataset version, config, commit, primary result and artifact status.
- Keep command, environment, full metrics and logs inside the experiment output directory.
- Do not create a separate tracked Markdown report for every run unless explicitly requested.
