# Data and Artifacts

## What Git Stores

- Source code, scripts, configs and tests.
- README and long-lived documentation.
- Public processed MobilityBench baseline under `data/processed/mobilitybench/`.
- Small, stable historical metrics under `reports/metrics/`.

## What Git Does Not Store

- Didi real POI data or any internal business fields.
- Raw/private datasets, local model directories and downloaded weights.
- Embedding arrays, NPY/NPZ, generated Parquet, checkpoints and ONNX/binary model files.
- Training outputs, logs, generated reports, plots and temporary analysis tables.
- Ad-hoc legacy copies inside the repository; use approved external archival storage instead.

## Stable Data Contract

Generated POI features and SID mappings must preserve deterministic row alignment:

- `poi_id` is non-empty and unique.
- `row_id` is stable and ordered.
- Metadata, Embedding, geo features, RQ-VAE inputs and SID mappings have matching row counts/order.
- Numeric arrays contain no NaN or Inf.
- `qrels` and `candidates` are evaluation inputs unless an approved experiment explicitly states otherwise.

## Public Baseline

Tracked MobilityBench files live in `data/processed/mobilitybench/`. They support public code validation. Other content under `data/` is local and ignored.

## Didi Data Policy

- Didi POI data must remain in approved internal storage and must never be pushed to the public repository.
- Do not place access tokens, internal hosts or sensitive absolute paths in code/config/docs.
- Commit only schemas, synthetic fixtures or explicitly approved anonymized samples.

## Local Artifact Layout

| Path | Purpose | Git policy |
|---|---|---|
| `models/` | local pretrained models | ignored |
| `data/embeddings/` | text embeddings and metadata | ignored |
| `data/geo/` | geographic features and metadata | ignored |
| `data/rqvae/` | model input arrays | ignored |
| `data/sid/` | SID indices/mappings | ignored |
| `outputs/` | checkpoints, logs, metrics, reports and figures | ignored |

## Regeneration and Backup

- Regenerable: parsed intermediates, Embedding, feature arrays, SID mappings, analysis tables and figures when source data/config/model versions are available.
- Must be backed up externally when needed: irreplaceable raw snapshots, approved training checkpoints, exact model versions and final experiment manifests.
- A completed experiment output should contain config, command, environment, commit, log, metrics and selected checkpoint in one isolated directory.
