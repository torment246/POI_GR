"""Evaluate A0-GID with the unchanged QG generation, scoring and prefix constraints."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft.a0_gid_data import VARIANT, WIRE_FORMAT, checked_artifact
from qg_prqk.sft.constrained_decoding import ConstrainedGenerationModel, FinalIdPrefixIndex
from qg_prqk.sft.constrained_evaluation import immutable_json
from qg_prqk.sft.evaluation_data import (
    SUBSETS, SftEvaluationError, align_subsets, load_config, load_json,
    load_references, read_records, require_hash, resolve, signature,
)
from qg_prqk.sft.training import project_root
from qg_prqk.sid.identifiers import identifier_content, identifier_key

DEFAULT_CONFIG = 'qg_prqk/configs/sft/evaluation_a0_gid_epoch3_dual_decode_v1.yaml'
MODES = ('unconstrained', 'constrained')
METRICS = ('hr@1', 'hr@3', 'hr@5', 'hr@10', 'ndcg@1', 'ndcg@3',
           'ndcg@5', 'ndcg@10', 'mrr@10', 'valid_id_rate')


def load_protocol(path: Path) -> dict[str, Any]:
    """Validate small frozen receipts only; leave full scans to platform preparation."""
    settings = yaml.safe_load(path.read_text())
    if settings.get('schema_version') != 'qg-prqk-a0-evaluation-v1':
        raise SftEvaluationError('不是 PRQ-KMeans 评测配置')
    reference = resolve(settings['reference_config'])
    require_hash(reference, settings['reference_config_sha256'])
    config = load_config(reference)
    source_spec = config['variants']['a4_gid_parent']
    spec = settings['variant_spec']
    if spec['expected_step'] != 8955 or spec['max_new_tokens'] != 13:
        raise SftEvaluationError('必须固定 epoch 3 / step 8955 与 13 token 上限')
    for key in ('output_root', 'run_control'):
        value = resolve(settings[key])
        if not value.is_relative_to(resolve('qg_prqk/outputs')) or value == resolve('qg_prqk/outputs'):
            raise SftEvaluationError('产物和日志只能写到 qg_prqk/outputs 的独立子目录')
    if resolve(settings['output_root']) == resolve(config['output_root']):
        raise SftEvaluationError('禁止覆盖 QG-HRQ 结果目录')
    for key, hash_key in (('data_dir', 'data_manifest_sha256'), ('identifier_dir', 'identifier_manifest_sha256')):
        directory = resolve(spec[key])
        require_hash(directory / 'manifest.json', spec[hash_key])
        manifest = load_json(directory / 'manifest.json')
        if manifest.get('status') != 'completed' or manifest.get('variant') != VARIANT:
            raise SftEvaluationError('PRQ 训练数据/标识未完成或分支错误')
        if load_json(directory / '_SUCCESS').get('manifest_sha256') != spec[hash_key]:
            raise SftEvaluationError('完成标记与 manifest 不一致')
    data = load_json(resolve(spec['data_dir']) / 'manifest.json')
    source = data['contract']['sources']['a4_data_manifest']
    require_hash(resolve(source['path']), source_spec['data_manifest_sha256'])
    source_data = load_json(resolve(source['path']))
    if (source['sha256'] != source_spec['data_manifest_sha256']
            or source_data['source_sft']['time_split']['valid'] != config['date']
            or data['final_identifier']['manifest_sha256'] != spec['identifier_manifest_sha256']
            or data['outputs']['valid.jsonl']['rows'] != config['source_rows']
            or data.get('scan_mode') != 'full'
            or data['paired_contract'].get('only_history_and_target_poi_identifiers_differ') is not True):
        raise SftEvaluationError('PRQ 数据与 QG 配对或日期契约不一致')
    checkpoint = resolve(spec['checkpoint'])
    state = load_json(checkpoint / 'trainer_state.json')
    if (state.get('epoch'), state.get('global_step'), state.get('max_steps')) != (3.0, 8955, 8955):
        raise SftEvaluationError('指定 epoch-3 checkpoint 未完成')
    for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'generation_config.json'):
        if not (checkpoint / name).is_file():
            raise SftEvaluationError(f'checkpoint 缺少 {name}')
    config.update(variants={VARIANT: spec}, output_root=settings['output_root'],
                  run_control=settings['run_control'], config_sha256=sha256_file(path),
                  reference_config_sha256=settings['reference_config_sha256'])
    return config


def fingerprints() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name('a0_gid_data.py'),
             Path(__file__).with_name('constrained_decoding.py'),
             Path(__file__).with_name('constrained_evaluation.py'),
             Path(__file__).with_name('training.py'), Path(__file__).with_name('vocabulary.py'),
             resolve('qg_prqk/scripts/evaluate_a0_gid_sft.py')]
    return {**ev.source_fingerprints(), **{str(p.relative_to(resolve('qg_prqk'))): sha256_file(p) for p in paths}}


def build_index(config: dict, tokenizer: Any, *, expected_pois: int = 716245) -> ev.FinalIdIndex:
    """Read A0 provenance while reusing the established GID-first wire format."""
    spec = config['variants'][VARIANT]
    directory = resolve(spec['identifier_dir'])
    require_hash(directory / 'manifest.json', spec['identifier_manifest_sha256'])
    manifest = load_json(directory / 'manifest.json')
    if manifest.get('variant') != VARIANT or manifest.get('status') != 'completed':
        raise SftEvaluationError('索引来源必须是已完成的 A0-GID，不可冒充 A4')
    base, dedup, required = [np.load(checked_artifact(directory, manifest, name), mmap_mode='r')
                              for name in ('base_identifier_codes.npy', 'dedup_codes.npy', 'requires_dedup.npy')]
    if (base.shape != (expected_pois, 9) or base.dtype != np.int32
            or dedup.shape != (expected_pois,) or dedup.dtype != np.int32
            or required.dtype != np.bool_ or not np.array_equal(required, dedup >= 0)):
        raise SftEvaluationError('A0-GID 行数、九位结构或 Dedup 数组错误')
    mapping = load_json(resolve(config['tokenizer']) / 'qg_prqk_token_mapping.json')['tokens']
    for token, value in mapping.items():
        if tokenizer.encode(token, add_special_tokens=False) != [value]:
            raise SftEvaluationError('共同词表不再原子或 token ID 变化')
    row_by_tokens, tokens_by_poi = {}, {}
    parquet = pq.ParquetFile(checked_artifact(directory, manifest, 'poi_final_id_mapping.parquet'))
    row = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=['poi_row_index', 'poi_id', 'final_id_key']):
        for item in batch.to_pylist():
            if row >= expected_pois or item['poi_row_index'] != row:
                raise SftEvaluationError('POI 行序变化')
            content = identifier_content(base[row], int(dedup[row]), variant=WIRE_FORMAT)
            tokens = tuple(mapping[t] for t in re.findall(r'<[^>]+>', content))
            if (item['final_id_key'] != identifier_key(base[row], int(dedup[row]), variant=WIRE_FORMAT)
                    or tokens in row_by_tokens or not item['poi_id'] or item['poi_id'] in tokens_by_poi):
                raise SftEvaluationError('POI/Final ID 不唯一或不匹配')
            row_by_tokens[tokens] = row
            tokens_by_poi[item['poi_id']] = tokens
            row += 1
    if row != expected_pois:
        raise SftEvaluationError('全目录索引不完整')
    return ev.FinalIdIndex(row_by_tokens, tokens_by_poi, mapping['<TARGET_POI>'],
                           mapping['</TARGET_POI>'], tokenizer.eos_token_id, 9, row)


def prepare(config: dict) -> dict:
    print('[准备] 核验 epoch-3 模型与共同词表；不选择 checkpoint', flush=True)
    checkpoint = ev.validate_checkpoint(config, VARIANT)
    spec = config['variants'][VARIANT]
    manifest = load_json(resolve(spec['data_dir']) / 'manifest.json')
    valid = manifest['outputs']['valid.jsonl']
    print('[准备] 一次扫描 PRQ Validation，对齐五组固定业务键；不读取 Train/Test', flush=True)
    data = align_subsets(resolve(spec['data_dir']) / 'valid.jsonl', load_references(config),
                         resolve(config['output_root']) / 'data', variant=VARIANT,
                         source_rows=config['source_rows'], source_sha256=valid['sha256'],
                         source_manifest_sha256=spec['data_manifest_sha256'])
    tokenizer, template = ev.load_lf_tokenizer_and_template(Path(checkpoint['path']), project_root=project_root())
    index = build_index(config, tokenizer)
    trie = FinalIdPrefixIndex(index)
    if trie.max_length != spec['max_new_tokens']:
        raise SftEvaluationError('完整合法路径长度与生成上限不一致')
    for subset in SUBSETS:
        records = read_records(Path(data['outputs'][subset]['file']))
        for record in records[:8]:
            ev.encode_record(record, variant=VARIANT, tokenizer=tokenizer, template=template, index=index)
        print(f'[准备/{subset}] 10,000 条业务键/目标一致，样例 token 校验通过', flush=True)
    plan = dict(schema_version='qg-prqk-a0-dual-decode-plan-v1', variant=VARIANT, config=config,
                checkpoint=checkpoint, data=data, source_sha256=fingerprints(),
                preflight=dict(catalog_pois=index.poi_count, max_path_length=trie.max_length,
                               no_test_read=True, resampling=False, checked_formatted_samples=40))
    immutable_json(resolve(config['output_root']) / 'plan.json', plan)
    return plan


def verify_plan(plan: dict) -> None:
    if (plan.get('schema_version') != 'qg-prqk-a0-dual-decode-plan-v1'
            or plan.get('variant') != VARIANT or plan['source_sha256'] != fingerprints()):
        raise SftEvaluationError('PRQ 计划或代码变化，禁止混合断点')
    # The superset above binds A0/constraint code; this checks the unchanged core and weight file states.
    ev.verify_worker_inputs({**plan, 'source_sha256': ev.source_fingerprints()})


def evaluate_subset(plan: dict, mode: str, subset: str, *, model: Any, tokenizer: Any,
                    template: Any, index: ev.FinalIdIndex, smoke_limit: int | None) -> dict:
    """Use the existing chunk scorer for both modes, preserving illegal beam ranks."""
    import torch
    config = plan['config']
    data = plan['data']['outputs'][subset]
    require_hash(Path(data['file']), data['sha256'])
    records = read_records(Path(data['file']))
    if len(records) != 10000:
        raise SftEvaluationError('不允许丢失冻结评测行')
    if smoke_limit:
        records = records[:smoke_limit]
    tag = f'smoke{smoke_limit}' if smoke_limit else 'results'
    directory = resolve(config['output_root']) / mode / tag / subset
    settings = dict(plan_signature=signature(plan), decoding=mode, subset=subset,
                    smoke_limit=smoke_limit, num_beams=10, cutoff_len=1024, max_new_tokens=13,
                    data_sha256=data['sha256'], invalid_ids_keep_original_rank=True)
    run_signature = signature(settings)
    result_path, progress_path = directory / 'result.json', directory / 'progress.json'
    if result_path.exists():
        result = load_json(result_path)
        if (result.get('status') != 'completed' or result.get('signature') != run_signature
                or result['metrics']['sample_count'] != len(records)
                or (mode == 'constrained' and result['metrics']['valid_id_rate'] != 1)):
            raise SftEvaluationError('已存在不同协议或不完整结果，拒绝覆盖')
        print(f'[{mode}/{subset}] 已完成，复用', flush=True)
        return result
    progress = dict(signature=run_signature, next_line=0, metrics=ev.empty_metrics(),
                    actual_batch_size=config['decoding']['batch_size'], inference_seconds=0.)
    if progress_path.exists():
        progress = load_json(progress_path)
        done = progress['next_line']
        if (progress['signature'] != run_signature or not 0 <= done <= len(records)
                or progress['metrics']['sample_count'] != done
                or progress['metrics']['candidate_count'] != done * 10
                or (mode == 'constrained' and progress['metrics']['valid_candidate_count'] != done * 10)):
            raise SftEvaluationError('断点签名或累计计数错误')
    directory.mkdir(parents=True, exist_ok=True)
    while progress['next_line'] < len(records):
        start = progress['next_line']
        stop = min(start + config['decoding']['chunk_size'], len(records))
        began = time.perf_counter()
        try:
            metrics = ev.evaluate_chunk(records[start:stop], variant=VARIANT, model=model,
                                         tokenizer=tokenizer, template=template, index=index,
                                         batch_size=progress['actual_batch_size'], max_new_tokens=13)
        except RuntimeError as error:
            if not ev.is_cuda_oom(error) or progress['actual_batch_size'] <= 1:
                raise
            progress['actual_batch_size'] //= 2
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f'[{mode}/{subset}] OOM，batch 减半为 {progress["actual_batch_size"]}，重试未入账 chunk', flush=True)
            continue
        if (metrics['sample_count'] != stop - start or metrics['candidate_count'] != (stop - start) * 10
                or (mode == 'constrained' and metrics['valid_candidate_count'] != metrics['candidate_count'])):
            raise SftEvaluationError('生成候选不完整或约束解码存在非法 ID')
        ev.merge_metrics(progress['metrics'], metrics)
        progress.update(next_line=stop, inference_seconds=progress['inference_seconds'] + time.perf_counter() - began)
        write_json_atomic(progress_path, progress, overwrite=True)
        print(f'[{mode}/{subset}] {stop:,}/{len(records):,}，batch={progress["actual_batch_size"]}', flush=True)
    result = dict(status='completed', signature=run_signature, config=settings,
                  metrics=ev.finalize_metrics(progress['metrics']),
                  performance={k: progress[k] for k in ('actual_batch_size', 'inference_seconds')})
    immutable_json(result_path, result)
    return result


def worker(plan: dict, mode: str, smoke_limit: int | None) -> None:
    verify_plan(plan)
    checkpoint = Path(plan['checkpoint']['path'])
    tokenizer, template = ev.load_lf_tokenizer_and_template(checkpoint, project_root=project_root())
    index = build_index(plan['config'], tokenizer)
    model = ev.load_generation_model(checkpoint, expected_vocab_size=len(tokenizer))
    if mode == 'constrained':
        model = ConstrainedGenerationModel(model, FinalIdPrefixIndex(index))
    # Load weights/catalog once per decoding-mode worker, then reuse them for five subsets.
    for subset in SUBSETS:
        evaluate_subset(plan, mode, subset, model=model, tokenizer=tokenizer, template=template,
                        index=index, smoke_limit=smoke_limit)


def validate_hardware(*, single_a100: bool) -> None:
    """Require either one A100 or the retained legacy two-6000D layout."""
    import torch

    expected = 1 if single_a100 else 2
    count = torch.cuda.device_count()
    if count != expected:
        raise SftEvaluationError(f'本入口需要恰好 {expected} 张可见 GPU，实际 {count}')
    for gpu in range(count):
        name = torch.cuda.get_device_name(gpu).upper()
        free, total = torch.cuda.mem_get_info(gpu)
        print(f'GPU {gpu}: {name}, 空闲 {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB', flush=True)
        if single_a100:
            if 'A100' not in name or free < 28 * 1024**3:
                raise SftEvaluationError('单卡入口要求 A100，且至少 28 GiB 空闲显存')
        elif not ('6000D' in name or 'RTX PRO 6000' in name) or free < 36 * 1024**3:
            raise SftEvaluationError('双卡兼容入口要求两张 RTX PRO 6000D，每张至少 36 GiB 空闲')


def schedule(plan: dict, smoke_limit: int | None, *, single_a100: bool = False) -> None:
    """Run both decoders sequentially on one A100 or concurrently on legacy dual GPUs."""
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    expected = 1 if single_a100 else 2
    if len(devices) != expected or len(set(devices)) != expected or not all(v.strip() for v in devices):
        raise SftEvaluationError(f'CUDA_VISIBLE_DEVICES 必须指定 {expected} 张不同的 GPU')
    log_dir = resolve(plan['config']['run_control']) / (f'smoke{smoke_limit}' if smoke_limit else 'full')
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []

    def forward(process: Any, mode: str) -> int:
        with (log_dir / f'{mode}.console.log').open('a') as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(f'[{mode}] {line}', end='', flush=True)
        code = process.wait()
        (log_dir / f'{mode}.exit').write_text(f'{code}\n')
        return code

    def start(mode: str, gpu: int) -> Any:
        temporary = resolve(f'qg_prqk/outputs/tmp/a0e{gpu}')
        temporary.mkdir(parents=True, exist_ok=True)
        if len(os.fsencode(temporary)) > 64 or not os.access(temporary, os.W_OK):
            raise SftEvaluationError('TMPDIR 不可写或超过 64 字节')
        command = [sys.executable, str(resolve('qg_prqk/scripts/evaluate_a0_gid_sft.py')),
                   '--worker-mode', mode, '--plan', str(resolve(plan['config']['output_root']) / 'plan.json')]
        if smoke_limit:
            command += ['--smoke-limit', str(smoke_limit)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': devices[gpu], 'TMPDIR': str(temporary),
               'TMP': str(temporary), 'TEMP': str(temporary), 'PYTHONUNBUFFERED': '1'}
        process = subprocess.Popen(command, env=env, cwd=project_root(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        processes.append(process)
        (log_dir / f'{mode}.pid').write_text(f'{process.pid}\n')
        print(f'[{mode}] GPU {devices[gpu]} 已启动，PID={process.pid}', flush=True)
        return process

    pool = ThreadPoolExecutor(max_workers=2) if not single_a100 else None
    try:
        if single_a100:
            for mode in MODES:
                if forward(start(mode, 0), mode) != 0:
                    raise SftEvaluationError(f'{mode} 失败；停止后续模式并保留 chunk 断点')
            return
        futures = {}
        for gpu, mode in enumerate(MODES):
            process = start(mode, gpu)
            futures[pool.submit(forward, process, mode)] = mode
        for future in as_completed(futures):
            if future.result() != 0:
                raise SftEvaluationError(f'{futures[future]} 失败，终止本套件另一子进程；保留 chunk 断点')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if pool is not None:
            pool.shutdown(wait=True)


def summarize(plan: dict, smoke_limit: int | None) -> dict:
    root = resolve(plan['config']['output_root'])
    tag = f'smoke{smoke_limit}' if smoke_limit else 'results'
    rows, macro = [], {}
    for mode in MODES:
        group = []
        for subset in SUBSETS:
            path = root / mode / tag / subset / 'result.json'
            result = load_json(path)
            settings = result['config']
            if (result.get('status') != 'completed' or settings['plan_signature'] != signature(plan)
                    or settings['decoding'] != mode or settings['subset'] != subset
                    or settings['smoke_limit'] != smoke_limit
                    or result['metrics']['sample_count'] != (smoke_limit or 10000)
                    or (mode == 'constrained' and result['metrics']['valid_id_rate'] != 1)):
                raise SftEvaluationError('汇总协议或样本数不一致')
            row = dict(method='PRQ-KMeans (POI-only, GID-first)', decoding=mode, subset=subset,
                       sample_count=result['metrics']['sample_count'], **{k: result['metrics'][k] for k in METRICS})
            group.append(row)
            rows.append(row)
        macro[mode] = {key: sum(row[key] for row in group[1:]) / 4 for key in METRICS}
    summary = dict(status='completed', plan_signature=signature(plan), rows=rows,
                   generalization_macro_average=macro, fixed10k_excluded_from_macro=True,
                   metric_scale='fraction_0_to_1', smoke_limit=smoke_limit)
    directory = root / tag
    immutable_json(directory / 'summary.json', summary)
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description='PRQ-KMeans GID-first：五组固定 Validation，约束/无约束同口径评测。')
    parser.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG), help='冻结评测配置')
    parser.add_argument('--dry-run', action='store_true', help='只核对小文件与 checkpoint 状态，不扫描数据或启动 GPU')
    parser.add_argument('--prepare-only', action='store_true', help='完成全目录/数据/模型核验与五组缓存，不启动 GPU')
    parser.add_argument('--smoke-limit', type=int, help='每组 GPU 冒烟条数（1..100），输出独立')
    parser.add_argument('--single-a100', action='store_true', help='单张 A100 串行运行无约束和约束评测')
    parser.add_argument('--worker-mode', choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument('--plan', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
        parser.error('--smoke-limit 仅允许 1..100')
    if bool(args.worker_mode) != bool(args.plan) or (args.worker_mode and (args.dry_run or args.prepare_only or args.single_a100)):
        parser.error('worker 必须成对提供 --worker-mode/--plan，不可混用准备模式')
    if args.dry_run and args.prepare_only:
        parser.error('--dry-run 和 --prepare-only 不可同时使用')
    try:
        if args.worker_mode:
            worker(load_json(args.plan), args.worker_mode, args.smoke_limit)
            return 0
        config = load_protocol(resolve(args.config))
        if args.dry_run:
            layout = '单卡 A100 串行' if args.single_a100 else '双卡 6000D 并行'
            print(f'静态预检通过：epoch 3 / checkpoint-8955；5×10,000×2 模式；{layout}；Beam=10；cutoff=1024。')
            print('尚未扫描全量数据或核验全部模型字节；启动时自动准备、GPU smoke，再正式评测。')
            return 0
        root = resolve(config['output_root'])
        root.mkdir(parents=True, exist_ok=True)
        with (root / '.suite.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SftEvaluationError('同一 PRQ 评测套件正在运行，禁止重复启动') from error
            if not args.prepare_only:
                validate_hardware(single_a100=args.single_a100)
            plan = prepare(config)
            if args.prepare_only:
                print('完整 CPU 准备通过，未启动 GPU 推理。', flush=True)
                return 0
            if args.smoke_limit is None:
                schedule(plan, 2, single_a100=args.single_a100)
                summarize(plan, 2)
            schedule(plan, args.smoke_limit, single_a100=args.single_a100)
            summarize(plan, args.smoke_limit)
            print(f'全部完成：{root}', flush=True)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'PRQ 评测失败：{error}\n')
    return 0
