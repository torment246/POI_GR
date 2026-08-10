---
name: poi-genret-workflow
description: Safely organize, inspect, run, monitor, and document experiments in /ofs/map_search/hudan/poi_genret. Use for this repository's Embedding, SID, RQ-VAE, RQ-KMeans, GenPOI, TIGER, GNPR, SFT, retrieval evaluation, GPU training-platform launchers, or development-server GPU jobs. Enforce the repository scope, poi-gr environment, host GPU checks, streaming data handling, project-local outputs, launcher conventions, and reproducible experiment records.
---

# POI GenRet Workflow

Follow these constraints for every task in `/ofs/map_search/hudan/poi_genret`.

## Establish scope and protocol

1. Work only inside `/ofs/map_search/hudan/poi_genret` unless the user explicitly expands scope. Do not inspect, edit, stop, or reuse sibling-project jobs.
2. Read the repository `AGENTS.md`, `README.md`, `方案.md`, `docs/PROJECT_STATUS.md`, and task-relevant implementation before changing or launching anything.
3. Preserve the dirty worktree and other Codex processes. Before a launch, identify relevant processes by full command. Never stop a process unless its exact PID and task belong to the authorized experiment.
4. Reuse existing modules, configs, launch conventions, frozen splits, sample indices, and evaluators. Do not silently change the comparison protocol.

## Use the server GPU correctly

1. Treat the development server's physical GPU as the default resource when the user says to run locally or use the server GPU.
2. The filesystem sandbox may hide the NVIDIA driver. Check GPU state in the host environment with `nvidia-smi`; do not conclude that the server lacks a GPU from a sandbox-only driver error.
3. Use `/ofs/map_search/hudan/envs/poi-gr/bin/python` or explicitly activate `/ofs/map_search/hudan/envs/poi-gr`. Never launch a project job with an unverified default Python environment.
4. Before launch, check host GPU free memory/utilization and relevant running processes. If the required GPU is occupied, do not oversubscribe it.
5. Never fall back from a requested GPU implementation to CPU merely because the GPU is hidden or busy. Report the condition or wait as appropriate.

## Prevent OOM and storage misuse

1. Stream or mmap full POI embeddings and datasets. Keep only bounded samples or chunks in RAM; use existing `chunk_rows`, `sample_chunk_rows`, and GPU temporary-memory controls.
2. Use one GPU unless the user explicitly requests a multi-GPU layout. Estimate peak memory before increasing batch size, sample size, codebook size, or concurrent jobs.
3. Put outputs, caches, logs, PID files, exit-status files, and run-control artifacts under `/ofs/map_search/hudan/poi_genret/outputs` or another explicit path under `/ofs/map_search/hudan`.
4. Do not use system `/tmp`, `/dev/shm`, a home-directory cache, or an implicit framework temp directory for project data. Set project-local temp/cache variables when a tool otherwise requires them.

## Launch without script clutter

1. Do not create a shell script for each development-server run. Invoke the existing Python entry point directly and use CLI overrides for one-off layouts when supported.
2. Create or modify a YAML config only when it represents a durable protocol, platform launcher, or reusable experiment—not merely to wrap one command.
3. For a long job, write its log, PID, and eventual exit code under `outputs/run_control/<experiment>/`. Use a detached host process only after all input and output-path gates pass.

## Organize and derive platform launchers

1. Keep all durable training-platform/HDFS launcher files directly under the repository-level `launchers/` directory. Keep this directory flat: do not create method subdirectories. Preserve method, task, GPU type/count, and schedule in descriptive filenames such as `run_train_gnpr_sft_4a100_3epoch.sh`. Put reusable non-launcher Python entry points under `scripts/<method>/`; do not add `run_*` files directly at the repository root.
2. Keep the entire local `launchers/` directory covered by `.gitignore` because launcher files may contain platform-specific and sensitive fields. Create a launcher by copying the closest existing file in that directory, preferring the same method, task type, GPU model, and GPU count. Do not reconstruct the platform boilerplate from memory or start from an empty shell file.
3. Copy the full platform skeleton unchanged: shebang and strict shell mode, environment activation, repository working directory, Java/Hadoop/OFS setup, mount commands, resource and CUDA variables, cache/temp/thread settings, preflight checks, and explicit authentication fields such as `hadoop_name`, `user`, and `password`.
4. Existing plaintext user names, passwords, and related authentication values may be copied verbatim from the authorized reference launcher into the new ignored local launcher. Never replace required values with placeholders merely for the new launcher. Never print, echo, summarize, paste into documentation, expose in tool output, or commit those values.
5. Change only task-specific values: config/model/data/output paths, experiment name, GPU type/count and matching batch or accumulation parameters, task arguments, and the final start command. Preserve the established environment and platform initialization sequence unless the user explicitly requests a platform change.
6. Recalculate and verify global batch size, process count, checkpoint/evaluation cadence, and output isolation whenever resource parameters change. Diff the new launcher against its reference so every unrelated change is intentional.
7. Before handoff or launch, run `bash -n`, verify executable permission, resolve every referenced config/entrypoint/path, and use the launcher's `--dry-run` or preflight mode when available. Syntax validation must not start training.

## Validate and record

1. Monitor logs, process status, GPU utilization, and OOM signals. Communicate at least once per minute while a job is actively being supervised.
2. On completion, require exit code 0 and the method's success marker. Validate row count, shape, dtype, token ranges, frozen fingerprints/hashes, and unified metrics before interpreting results.
3. Compare only protocol-compatible cells. Separate reconstruction, code utilization, collision distribution, prefix semantics, and downstream retrieval instead of declaring a winner from one metric.

## Close every formal experiment in documentation

1. Treat documentation as an experiment exit gate. As soon as a formal run reaches a verified terminal state—completed, failed, or interrupted—update its records before declaring the experiment finished or starting the next formal experiment.
2. Route the full record through the method table in `docs/EXPERIMENT_LOG.md`: shared V1 work to `docs/experiments/V1.md`, Query/Embedding innovations to `EMBEDDING_OPTIMIZATION.md`, RQ-KMeans work to `RQKMEANS.md`, and paper-method work to `TIGER.md`, `GNPR_SID.md`, or `GENPOI.md`. Create one new method document only when that method begins its first formal experiment; never create one Markdown file per run.
3. Assign a repository-wide unique `EXP-YYYYMMDD-NN` after searching the index and method documents. Use the actual run date and increment `NN` from `01` within the day; never reuse an ID across methods.
4. Append the detailed record to the owning method document first. Include the objective and hypothesis, data version and frozen fingerprints, commit or dirty-worktree state, configuration, exact command, environment and resources, verified status and core metrics, artifact/log paths, conclusion, and next step.
5. Add one concise row to `docs/EXPERIMENT_LOG.md` only after the detailed record exists. Link to the method record and keep the index to method, status, and key result; do not duplicate the full configuration or analysis there.
6. Record failures, negative results, and interrupted runs honestly. State the last verified progress, exit status, error evidence, usable artifacts, impact, and retry decision; never infer missing metrics or rewrite an unsuccessful run as completed.
7. Update `docs/PROJECT_STATUS.md` only when the project stage, frozen candidate, comparison conclusion, or next formal step changes. Update `docs/DATA_AND_ARTIFACTS.md` when a durable dataset, checkpoint, mapping, evaluation set, artifact path, version, or fingerprint is added or changed. Update `docs/REPO_MAP.md` when stable directory ownership or entry points change.
8. Update `README.md` only for a material top-level status or documentation-entry change. Do not modify `方案.md` as routine experiment bookkeeping; it remains the approved technical baseline unless the user confirms a direction change.
9. Before handoff, cross-check experiment IDs, links, dates, commands, paths, hashes, statuses, and headline metrics against the actual outputs. Keep all project documents under `docs/` in Chinese.
10. Do not create formal experiment records for code-only cleanup, synthetic smoke tests, transient performance probes, or ordinary debugging. Fold only the necessary failure cause and adopted fix into the related formal experiment record.
11. Do not commit or push unless the user explicitly requests it.
