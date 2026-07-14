# Project Status

Last updated: 2026-07-14.

## Current Goal

Develop a reusable POI-side pipeline for generative retrieval:

```text
POI fields -> text/geo features -> RQ-VAE -> SID + GID -> PID -> retrieval evaluation
```

The next production-facing stage uses Didi POI data. Historical MobilityBench features, models and experiment outputs are reference results only and must not be reused as initialization or training input.

## Implemented Code

- Common processed-data schemas and dataset processors.
- MobilityBench cleaning, split validation and reference checks.
- POI text construction and local SentenceTransformer embedding.
- Geohash, coordinate and anchor-based geographic features.
- Semantic and geo-fused RQ-VAE training.
- SID/GID/PID export, collision metrics and prefix-purity evaluation.
- CAU-RQ-VAE category supervision, uniqueness regularization, pilot/final training and protected checkpoint selection.
- SID quality visualization utilities. Generated figures remain under ignored output directories.

## Available Public Baseline

`data/processed/mobilitybench/` is the tracked public baseline. Current main-table counts are:

| Table | Rows |
|---|---:|
| `pois.csv` | 109385 |
| `queries.csv` | 18011 |
| `candidates.csv` | 63804 |
| `qrels.csv` | 9735 |

MobilityBench is useful for code validation and public comparison, not as a substitute for the next Didi dataset.

## Artifact Status

- Old local `models/`, Embedding, NPY, checkpoint, SID and output content is retained under ignored `_local_legacy/` only.
- Active generated directories are empty until a new controlled run creates artifacts.
- Historical RQ-VAE/CAU results remain summarized in `docs/EXPERIMENT_LOG.md`; their old artifacts are not guaranteed to exist or remain compatible.
- Do not resume the old CAU checkpoint. Rebuild features and retrain after the Didi schema and data contract are finalized.

## Next Stage

1. Define Didi POI field contract, privacy boundary and stable `poi_id`/row-order rules.
2. Define text feature composition and offline model source.
3. Define geographic GID and semantic SID objectives.
4. Build a small aligned sample and run data/feature smoke tests.
5. Re-run Embedding, Semantic RQ-VAE and CAU-RQ-VAE from scratch with isolated outputs.
6. Evaluate SID/PID quality before starting query-to-PID training.
