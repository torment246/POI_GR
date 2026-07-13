# CAU-RQ-VAE Gate 0 Baseline Recheck

Generated on 2026-07-12 from repository-local artifacts. This gate did not train a model and did not overwrite Semantic/Geo_fused baseline files.

## Command

```bash
python scripts/05_export_and_eval_sid.py --mode semantic --checkpoint outputs/rqvae/rqvae_semantic.pt --rqvae-input data/rqvae/rqvae_input_semantic.npy --embedding-meta data/embeddings/poi_embedding_meta.parquet --geo-meta data/geo/poi_geo_meta.parquet --qrels data/processed/mobilitybench/qrels.csv --candidates data/processed/mobilitybench/candidates.csv --out-mapping outputs/experiments/cau_rqvae/gate0_baseline_recheck/poi_sid_mapping_semantic_recheck.parquet --out-indices outputs/experiments/cau_rqvae/gate0_baseline_recheck/poi_sid_indices_semantic_recheck.npy --report outputs/experiments/cau_rqvae/gate0_baseline_recheck/sid_quality_semantic_recheck.md --batch-size 8192 --device auto --max-report-groups 1000
```

## Output Paths

- Recheck report: `outputs/experiments/cau_rqvae/gate0_baseline_recheck/sid_quality_semantic_recheck.md`
- Recheck indices: `outputs/experiments/cau_rqvae/gate0_baseline_recheck/poi_sid_indices_semantic_recheck.npy`
- Recheck mapping: `outputs/experiments/cau_rqvae/gate0_baseline_recheck/poi_sid_mapping_semantic_recheck.parquet`
- Baseline report: `reports/sid_quality_semantic.md`
- Baseline indices: `data/sid/poi_sid_indices_semantic.npy`
- Baseline mapping: `data/sid/poi_sid_mapping_semantic.parquet`

## Exact Artifact Checks

- indices shape baseline/recheck: `(109385, 3)` / `(109385, 3)`
- indices exact match: `True`
- mapping shape baseline/recheck: `(109385, 25)` / `(109385, 25)`
- mapping shape match: `True`
- mapping key columns exact match: `True`
- missing key columns: `[]`

## Metric Comparison

| Metric | Baseline | Recheck | Delta | Tolerance | Pass |
|---|---:|---:|---:|---:|---|
| input_rows | 109385 | 109385 | 0 | 0 | True |
| unique_poi_id | 109385 | 109385 | 0 | 0 | True |
| unique SID count | 61437 | 61437 | 0 | 0 | True |
| unique SID rate | 0.5617 | 0.5617 | +0.000000 | 0.0001 | True |
| max POIs per SID | 211 | 211 | 0 | 0 | True |
| unique PID count | 105841 | 105841 | 0 | 0 | True |
| unique PID rate | 0.9676 | 0.9676 | +0.000000 | 0.0001 | True |
| dedup PID unique count | 109385 | 109385 | 0 | 0 | True |
| dedup PID unique rate | 1.0 | 1.0 | +0.000000 | 0.0001 | True |
| Prefix1 semantic purity | 0.3527 | 0.3527 | +0.000000 | 0.0001 | True |
| Prefix2 semantic purity | 0.4461 | 0.4461 | +0.000000 | 0.0001 | True |
| Prefix3 semantic purity | 0.7651 | 0.7651 | +0.000000 | 0.0001 | True |
| Prefix1 city purity | 0.0915 | 0.0915 | +0.000000 | 0.0001 | True |
| Prefix2 city purity | 0.2765 | 0.2765 | +0.000000 | 0.0001 | True |
| Prefix3 city purity | 0.6799 | 0.6799 | +0.000000 | 0.0001 | True |
| qrels rows | 9735 | 9735 | 0 | 0 | True |
| qrels unique POI | 9657 | 9657 | 0 | 0 | True |
| qrels SID collision count | 4812 | 4812 | 0 | 0 | True |
| qrels SID collision rate | 0.4943 | 0.4943 | +0.000000 | 0.0001 | True |
| qrels PID collision count | 80 | 80 | 0 | 0 | True |
| qrels PID collision rate | 0.0082 | 0.0082 | +0.000000 | 0.0001 | True |
| candidate rows | 63804 | 63804 | 0 | 0 | True |
| candidate queries SID collision rate | 0.2417 | 0.2417 | +0.000000 | 0.0001 | True |
| candidate queries PID collision rate | 0.1073 | 0.1073 | +0.000000 | 0.0001 | True |

## Gate Decision

- metric comparison pass: `True`
- exact artifact comparison pass: `True`
- Gate 0 status: `PASS`
- Decision: baseline export/eval is reproducible. Proceed to Gate 1 coarse category label construction.
