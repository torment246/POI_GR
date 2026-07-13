# Repository Map

Last verified from repository files on 2026-07-12.

## Core Source Files

- `src/rqvae.py`: PyTorch RQ-VAE model, residual vector quantizer, encode/decode helpers, and `model_config()`.
- `src/cau_labels.py`: high-confidence coarse category labeling helpers for CAU-RQ-VAE.
- `src/rqvae_preprocess.py`: deterministic RQ-VAE input construction, PCA/scaler fitting, geo_fused concatenation, train/val index split, preprocess pickle load/save, and input numeric checks.
- `src/train_utils.py`: YAML loading, seed setup, device resolution, numpy dataset/loader, train/eval loops, checkpoint save, train log writing, and CPU thread summary.
- `src/geo_features.py`: geohash encoding, coordinate validation, lat/lon features, anchor distance/bearing features, and standardization.
- `src/sid_eval.py`: SID/PID mapping construction, metadata alignment checks, semantic label inference, SID/PID collision metrics, prefix purity metrics, qrels/candidates analysis, and report builders.
- `src/embedding_utils.py`: embedding text construction and validation, local SentenceTransformer loading, offline flags, device resolution, and embedding numeric checks.
- `src/sid_visualization.py`: matplotlib-based SID quality figures and HTML index generation.
- `src/data_utils.py`: shared CSV/parquet/report helpers.
- `src/text_normalize.py`: text cleanup helpers.
- `src/poi_genret/schema.py`: common processed-data schemas and CSV IO helpers.
- `src/poi_genret/process_utils.py`: stable hashing, parsing, normalization, and dataset processing helpers.
- `src/poi_genret/evaluation.py`: retrieval qrels/run conversion and ranking metrics.
- `src/poi_genret/sid.py`: earlier SID utilities for processed smoke data.

## Pipeline Scripts

General data layer scripts:

- `scripts/process_llm4poi.py`: process LLM4POI next-POI / trajectory data.
- `scripts/process_yelp.py`: process Yelp business metadata and template queries.
- `scripts/process_mobilitybench.py`: process MobilityBench POI Search / Nearby Search into processed CSVs.
- `scripts/process_all_datasets.py`: one-command runner for the dataset processors.
- `scripts/analyze_all_datasets.py`: build cross-dataset summaries, reports, and figures.
- `scripts/validate_mobilitybench_data.py`: validate processed MobilityBench references, candidates, qrels, splits, and quality issues without modifying processed files.
- `scripts/prepare_mvp_data.py`, `scripts/prepare_all_datasets.py`, `scripts/prepare_mobilitybench_data.py`, `scripts/analyze_datasets.py`: older/compatibility preparation and analysis entry points still present in the repository.
- `scripts/evaluate.py`: evaluate a run file against qrels with ranking metrics.

Current MobilityBench SID pipeline order:

1. `scripts/01_build_poi_sid_train_data.py`: builds `data/sid/poi_sid_train.parquet`, preview CSV, and POI parse coverage reports from `data/processed/mobilitybench/pois.csv`.
2. `scripts/02_embed_poi_text.py`: builds `data/embeddings/poi_text_embeddings.npy`, embedding metadata parquet, and embedding quality report from SID train data and a local embedding model.
3. `scripts/03_build_geo_features.py`: builds `data/geo/poi_geo_features.npy`, geo metadata parquet, anchors/stat files, and geo quality reports.
4. `scripts/04_train_rqvae.py`: trains or resumes semantic / geo_fused RQ-VAE models, writes checkpoints, logs, preprocess pickles, and train reports.
5. `scripts/05_export_and_eval_sid.py`: loads RQ-VAE checkpoints, encodes POI SID indices, exports SID/PID mappings, and writes SID quality reports.
6. `scripts/06_visualize_sid_quality.py`: generates SID quality PNG/SVG figures and `outputs/figures/sid_quality/index.html`.
7. `scripts/07_build_cau_labels.py`: builds CAU high-confidence coarse category labels and train/heldout masks.
8. `scripts/08_train_cau_rqvae.py`: runs CAU smoke and pilot training from the Semantic baseline checkpoint.
9. `scripts/09_train_cau_final.py`: runs CAU Final P1 with classifier warmup, periodic SID evaluation, protected checkpoint selection, formal CAU SID export, and final reports.

## Config Files

- `configs/rqvae_train.yaml`: baseline RQ-VAE config. It defines seed `42`, input paths, PCA dim `256`, `geo_alpha=0.1`, `val_ratio=0.05`, encoder hidden dims `[512, 256]`, latent dim `64`, `num_codebooks=3`, `codebook_size=128`, batch size `1024`, default epochs `50`, early stop patience `10`, and device `auto`.
- `configs/rqvae_cau.yaml`: CAU-RQ-VAE config. It reuses semantic RQ-VAE input and Semantic baseline checkpoint, points to CAU label artifacts, and defines category/uniqueness loss settings.

No `requirements.txt`, `pyproject.toml`, `setup.py`, or environment YAML was found during inspection.

## Important Commands

Commands recorded in `README.md`:

```bash
python scripts/process_llm4poi.py
python scripts/process_yelp.py
python scripts/process_mobilitybench.py
python scripts/analyze_all_datasets.py
python scripts/process_all_datasets.py
```

Main SID/RQ-VAE commands supported by current argparse defaults:

```bash
python scripts/01_build_poi_sid_train_data.py
python scripts/02_embed_poi_text.py --model models/Qwen3-Embedding-0.6B --device auto
python scripts/03_build_geo_features.py
python scripts/04_train_rqvae.py --config configs/rqvae_train.yaml --mode semantic
python scripts/04_train_rqvae.py --config configs/rqvae_train.yaml --mode geo_fused
python scripts/05_export_and_eval_sid.py --mode both
python scripts/06_visualize_sid_quality.py
python scripts/07_build_cau_labels.py --config configs/rqvae_cau.yaml
python scripts/08_train_cau_rqvae.py --config configs/rqvae_cau.yaml --output-dir outputs/experiments/cau_rqvae/pilot_tag0025_unique010 --lambda-tag 0.025 --lambda-unique 0.10
CUDA_VISIBLE_DEVICES=0 python scripts/09_train_cau_final.py --config configs/rqvae_cau.yaml --output-dir outputs/experiments/cau_rqvae/final_p1 --lambda-tag 0.025 --lambda-unique 0.10 --unique-margin 0.70 --warmup-epochs 3 --max-total-epochs 80 --early-stop-patience 10 --evaluation-interval 5 --device auto --run-name final_p1
```

Path note: `scripts/02_embed_poi_text.py` has an argparse default of `--model /model/Qwen3-Embedding-0.6B`, but the current embedding report records the actual model path as `models/Qwen3-Embedding-0.6B`. Prefer the explicit repo-local path unless the user points to another local model directory.

Useful script-specific options found in argparse:

- `scripts/04_train_rqvae.py`: `--epochs`, `--min-epochs`, `--batch-size`, `--device`, `--debug-max-rows`, `--force-rebuild-input`, `--resume`, `--resume-from`.
- `scripts/05_export_and_eval_sid.py`: `--mode semantic|geo_fused|both`, `--checkpoint`, `--rqvae-input`, `--batch-size`, `--device`, `--max-report-groups`, custom mapping/index/report paths.
- `scripts/08_train_cau_rqvae.py`: `--config`, `--output-dir`, `--epochs`, `--batch-size`, `--debug-max-rows`, `--device`, `--lambda-tag`, `--lambda-unique`, `--run-name`.
- `scripts/09_train_cau_final.py`: `--config`, `--output-dir`, `--lambda-tag`, `--lambda-unique`, `--unique-margin`, `--warmup-epochs`, `--max-total-epochs`, `--early-stop-patience`, `--evaluation-interval`, `--batch-size`, `--sid-batch-size`, `--device`, `--run-name`, `--max-report-groups`.
- `scripts/03_build_geo_features.py`: `--n-anchors`, `--random-state`, `--qrels`, custom input/output/report paths.
- `scripts/02_embed_poi_text.py`: `--model`, `--batch-size`, `--device`, `--max-length`, `--normalize`, `--no-normalize`.

Do not run formal training unless the user explicitly asks for it. For long experiments, first run a smoke or pilot command with `--debug-max-rows` or a separate small config/output directory.

## Dependency Notes

Dependencies are inferred from actual imports because no environment file is present:

- Core data stack: `pandas`, `numpy`, `pyarrow` or another parquet backend, `yaml`.
- Model/training stack: `torch`, `torch.nn`, `torch.utils.data`, `tqdm`.
- Preprocessing: `sklearn.decomposition.PCA`, `sklearn.preprocessing.StandardScaler`.
- Embedding: `sentence_transformers` with local model loading and offline environment flags.
- Visualization/reporting: `matplotlib`, `matplotlib.font_manager`, `numpy`, `pandas`.
- Standard library usage: `argparse`, `json`, `csv`, `pathlib`, `hashlib`, `re`, `math`, `subprocess`, `collections`.

Observed environment on 2026-07-12:

- Python executable: `/mnt/data1/home/hudan/anaconda3/envs/GR/bin/python`
- Python: `3.10.20`
- numpy: `2.2.6`
- pandas: `2.3.3`
- torch: `2.12.0+cu130`
- scikit-learn: `1.7.2`
- PyYAML: `6.0.3`
- CUDA available in this session: `False`
