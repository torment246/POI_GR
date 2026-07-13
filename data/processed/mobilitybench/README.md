# Processed MobilityBench Data

This directory contains the processed MobilityBench POI retrieval dataset used by the SID/RQ-VAE pipeline.

Raw MobilityBench source files are not committed. These CSV/JSON files are the normalized, split-ready artifacts consumed by the repository scripts.

## Core Files

| File | Description |
|---|---|
| `pois.csv` | POI catalog with `poi_id`, name/address/city/category/tags and coordinates. |
| `queries.csv` | Query set with `query_id`, text, city, user coordinates, and query type. |
| `qrels.csv` | Query-to-target-POI relevance rows. Used for evaluation only. |
| `candidates.csv` | Query candidate POIs. Used for evaluation only. |
| `train_queries.csv`, `dev_queries.csv`, `test_queries.csv` | Query splits. |
| `train_qrels.csv`, `dev_qrels.csv`, `test_qrels.csv` | qrels splits aligned with query splits. |
| `split_manifest.json` | Split metadata. |
| `dataset_summary.json` | Dataset counts and summary metadata. |
| `task_distribution.csv` | Query task distribution summary. |

Additional files such as `clean_nearby_queries.csv`, `tool_intent_queries.csv`, and split-specific tool-intent query files are retained because they are part of the checked processed MobilityBench export.

## Current Counts

| Item | Count |
|---|---:|
| POIs | 109385 |
| queries | 18011 |
| qrels rows | 9735 |
| candidates rows | 63804 |

The POI-side SID pipeline assumes `poi_id` values in `qrels.csv` and `candidates.csv` are present in `pois.csv`. `qrels` and `candidates` must not be used for training unless a future experiment explicitly changes that contract.
