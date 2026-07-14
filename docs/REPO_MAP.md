# Repository Map

Only stable directories and responsibilities are listed here.

```text
README.md                         project entry point
AGENTS.md                         Codex collaboration rules
requirements.txt                  Python dependencies
configs/                          experiment configuration
src/                              reusable library modules
scripts/                          command-line pipeline entry points
docs/                             long-lived project documentation
reports/metrics/                  small stable historical metrics
data/processed/mobilitybench/     tracked public baseline
models/                           ignored local models
outputs/                          ignored generated experiment outputs
_local_legacy/                    ignored inactive local artifacts
```

## Source Modules

- `src/poi_genret/`: processed-data schema, normalization and retrieval evaluation.
- `src/embedding_utils.py`, `src/text_normalize.py`: POI text and embedding helpers.
- `src/geo_features.py`: coordinate, geohash and anchor features.
- `src/rqvae.py`, `src/rqvae_preprocess.py`, `src/train_utils.py`: RQ-VAE model, preprocessing and training utilities.
- `src/cau_labels.py`: CAU coarse-category labels.
- `src/sid_eval.py`, `src/sid_visualization.py`: SID/PID evaluation and optional visualization.

## Script Groups

- Dataset layer: `process_*.py`, `prepare_*.py`, `analyze_all_datasets.py`, `validate_mobilitybench_data.py`.
- POI SID pipeline: `01_build_poi_sid_train_data.py` through `06_visualize_sid_quality.py`.
- CAU pipeline: `07_build_cau_labels.py`, `08_train_cau_rqvae.py`, `09_train_cau_final.py`.
- Retrieval evaluation: `scripts/evaluate.py`.

Generated reports and figures default to ignored `outputs/` paths. New scripts should follow the same convention and expose configurable input/output paths.
