# Repository Map

Only stable directories and responsibilities are listed here.

```text
README.md                         project entry point
AGENTS.md                         Codex collaboration rules
requirements.txt                  Python dependencies
configs/                          experiment configuration
src/                              reusable library modules
scripts/data/                     dataset preparation, processing and validation
scripts/sid/                      POI feature, RQ-VAE, SID and CAU pipelines
scripts/eval/                     retrieval evaluation
docs/                             long-lived project documentation
reports/metrics/                  small stable historical metrics
data/processed/mobilitybench/     tracked public baseline
models/                           ignored local models
outputs/                          ignored generated experiment outputs
```

## Source Modules

- `src/poi_genret/`: processed-data schema, normalization and retrieval evaluation.
- `src/embedding_utils.py`, `src/text_normalize.py`: POI text and embedding helpers.
- `src/geo_features.py`: coordinate, geohash and anchor features.
- `src/rqvae.py`, `src/rqvae_preprocess.py`, `src/train_utils.py`: RQ-VAE model, preprocessing and training utilities.
- `src/cau_labels.py`: CAU coarse-category labels.
- `src/sid_eval.py`, `src/sid_visualization.py`: SID/PID evaluation and optional visualization.

## Script Groups

- Dataset layer: `scripts/data/process_*.py`, `scripts/data/prepare_*.py`, `scripts/data/analyze_all_datasets.py`, `scripts/data/validate_mobilitybench_data.py`.
- POI SID pipeline: `scripts/sid/01_build_poi_sid_train_data.py` through `scripts/sid/06_visualize_sid_quality.py`.
- CAU pipeline: `scripts/sid/07_build_cau_labels.py`, `scripts/sid/08_train_cau_rqvae.py`, `scripts/sid/09_train_cau_final.py`.
- Retrieval evaluation: `scripts/eval/evaluate.py`.

Generated reports and figures default to ignored `outputs/` paths. New scripts should follow the same convention and expose configurable input/output paths.
