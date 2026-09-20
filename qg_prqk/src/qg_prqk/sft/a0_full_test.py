"""Run the frozen A0-GID checkpoint on paired full Test with two decoders."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import yaml

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sft import a0_evaluation as a0, evaluation as ev
from qg_prqk.sft.a0_gid_data import HISTORY_ID, VARIANT, WIRE_FORMAT, checked_artifact
from qg_prqk.sft.constrained_decoding import ConstrainedGenerationModel, FinalIdPrefixIndex
from qg_prqk.sft.constrained_evaluation import immutable_json
from qg_prqk.sft.data import load_final_identifier_lookup
from qg_prqk.sft.evaluation_data import business_key, current_context, load_json, require_hash, resolve, signature
from qg_prqk.sft.full_test_data import JsonlRecordSequence, file_stat, load_test_config, parse_test_record, test_sources
from qg_prqk.sft.full_test_evaluation import encode_test_record, evaluate_test_chunk
from qg_prqk.sft.training import project_root
from qg_prqk.sid.identifiers import identifier_content, identifier_key

DEFAULT_CONFIG = 'qg_prqk/configs/sft/evaluation_a0_gid_epoch3_full_test_dual_decode_v1.yaml'
OUTPUT = 'qg_prqk/outputs/eval/a0_gid_epoch3_full_test_20260714_dual_decode_v1'
MODES = a0.MODES


def load_protocol(path: Path) -> dict[str, Any]:
    """Bind the existing A0 checkpoint and the same full Test as the A4 runs."""
    settings = yaml.safe_load(path.read_text())
    if (settings['schema_version'], settings['split'], settings['date'], settings['source_rows']) != (
            'qg-prqk-a0-full-test-v1', 'test', '2026-07-14', 606682):
        raise ValueError('必须固定 A0 epoch-3 和 2026-07-14 的 606,682 条 Test')
    for key in ('a0_protocol', 'reference_test_protocol'):
        require_hash(resolve(settings[key]), settings[key + '_sha256'])
    config = a0.load_protocol(resolve(settings['a0_protocol']))
    reference = load_test_config(resolve(settings['reference_test_protocol']))
    if config['decoding'] != reference['decoding']:
        raise ValueError('A0 与已完成 QG Test 的生成参数不同')
    if resolve(settings['output_root']) != resolve(OUTPUT):
        raise ValueError('A0 Test 必须使用独立冻结输出目录')
    if resolve(settings['run_control']) != resolve('qg_prqk/outputs/run_control/eval_a0_gid_full_test_dual_decode_v1'):
        raise ValueError('A0 Test 日志目录不符合冻结协议')
    source = test_sources(reference)[WIRE_FORMAT]
    spec = reference['variants'][WIRE_FORMAT]
    checkpoint = resolve(config['variants'][VARIANT]['checkpoint'])
    # A lightweight dry run reads only metadata and a few source lines.
    with Path(source['file']).open('rb') as stream:
        for line_number in range(1, 3):
            parse_test_record(stream.readline(), line_number=line_number, variant=WIRE_FORMAT)
    return {**config, **settings, 'config_path': str(path), 'config_sha256': sha256_file(path),
            'source': source, 'source_identifier': spec,
            'checkpoint_path': str(checkpoint)}


def fingerprints() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name('full_test_data.py'),
             Path(__file__).with_name('full_test_evaluation.py'),
             resolve('qg_prqk/scripts/evaluate_a0_gid_test.py')]
    return {**a0.fingerprints(), **{str(p): sha256_file(p) for p in paths}}


def remap_test_record(record: Mapping[str, Any], *, old_rows: Mapping[str, int],
                      row_by_poi: Mapping[str, int], content: Callable[[int], str],
                      key: Callable[[int], str], requires_dedup: Sequence[bool]) -> dict[str, Any]:
    """Change only target/history identifiers; never reinterpret Test as Validation."""
    if (record.get('split'), record.get('identifier_variant')) != ('test', WIRE_FORMAT):
        raise ValueError('只允许冻结 A4-GID Test 输入')
    current = current_context(record)
    row = row_by_poi.get(record['target_poi_id'])
    target = record['messages'][1]['content']
    if (row is None or not target.startswith('<TARGET_POI>') or not target.endswith('</TARGET_POI>')
            or old_rows.get(target[12:-13]) != row):
        raise ValueError('源 Test 目标标识与 POI 不一致')
    def replace(match: Any) -> str:
        history_row = old_rows.get(match.group(1))
        if history_row is None:
            raise ValueError('源 Test 历史标识不在目录，禁止跳过')
        return f'<POI_QGPRQK_ID>{content(history_row)}</POI_QGPRQK_ID>'
    user = record['messages'][0]['content']
    transformed, count = HISTORY_ID.subn(replace, user)
    if count != record['history_length'] or user.count('<POI_QGPRQK_ID>') != count:
        raise ValueError('Test 历史标识或事件数量不一致')
    result = {**record, 'identifier_variant': VARIANT, 'requires_dedup': bool(requires_dedup[row]),
              'target_qg_prqk_id_key': key(row),
              'messages': [dict(role='user', content=transformed),
                           dict(role='assistant', content=f'<TARGET_POI>{content(row)}</TARGET_POI>')]}
    if current_context(result) != current:
        raise ValueError('Test CURRENT 被意外修改')
    return result


def make_remapper(config: dict) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Verify both frozen catalogs and build a bounded-cache identifier remapper."""
    spec = config['source_identifier']
    require_hash(resolve(spec['identifier_dir']) / 'manifest.json', spec['identifier_manifest_sha256'])
    old = load_final_identifier_lookup(resolve(spec['identifier_dir']), WIRE_FORMAT)
    spec = config['variants'][VARIANT]
    directory = resolve(spec['identifier_dir'])
    require_hash(directory / 'manifest.json', spec['identifier_manifest_sha256'])
    manifest = load_json(directory / 'manifest.json')
    base, dedup = [np.load(checked_artifact(directory, manifest, name), mmap_mode='r')
                   for name in ('base_identifier_codes.npy', 'dedup_codes.npy')]
    pois = pq.read_table(checked_artifact(directory, manifest, 'poi_final_id_mapping.parquet'), columns=['poi_id'])['poi_id'].to_pylist()
    if (pois != old.poi_ids or base.shape != old.base_codes.shape or dedup.shape != old.dedup_codes.shape
            or not np.array_equal(base[:, :6], old.base_codes[:, :6])):
        raise ValueError('A0/A4 POI 行序、数量或确定性 GID6 不一致')
    old_rows = {old.content(i): i for i in range(len(pois))}
    if len(old_rows) != len(pois):
        raise ValueError('源目录标识非唯一')
    @lru_cache(maxsize=65536)
    def content(row: int) -> str:
        return identifier_content(base[row], int(dedup[row]), variant=WIRE_FORMAT)
    def key(row: int) -> str:
        return identifier_key(base[row], int(dedup[row]), variant=WIRE_FORMAT)
    required = dedup >= 0
    return lambda record: remap_test_record(record, old_rows=old_rows, row_by_poi=old.row_by_poi_id,
                                             content=content, key=key, requires_dedup=required)


class A0TestRecords(Sequence):
    """Read the frozen source by byte offset and remap only the requested chunk."""
    def __init__(self, source: dict, remap: Callable):
        self.records = JsonlRecordSequence(Path(source['file']), variant=WIRE_FORMAT,
                                          expected_rows=source['rows'], expected_sha256=source['sha256'])
        self.remap = remap

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: int | slice) -> Any:
        rows = self.records[key]
        return [self.remap(r) for r in rows] if isinstance(key, slice) else self.remap(rows)


def scan_source(source: dict, remap: Callable, *, inspect: Callable | None = None) -> dict:
    """Validate every Test row and its remapped history without writing another JSONL."""
    digest, keys_digest, mapped_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    seen, count, history_count = set(), 0, 0
    with Path(source['file']).open('rb') as stream:
        for count, raw in enumerate(stream, 1):
            if count > source['rows']:
                raise ValueError('Test 超出冻结行数')
            digest.update(raw)
            record = parse_test_record(raw, line_number=count, variant=WIRE_FORMAT)
            keys = json.dumps(business_key(record), ensure_ascii=False, separators=(',', ':')).encode()
            hashed = hashlib.sha256(keys).digest()
            if hashed in seen:
                raise ValueError('Test 业务键重复，禁止去重后继续')
            seen.add(hashed)
            keys_digest.update(keys + b'\n')
            mapped = remap(record)
            mapped_digest.update(json.dumps(mapped, sort_keys=True, ensure_ascii=False).encode() + b'\n')
            history_count += record['history_length']
            if inspect is not None and count <= 100:
                inspect(mapped)
            if count % 50000 == 0:
                print(f'[A0 Test 准备] {count:,}/{source["rows"]:,}，目标和历史映射通过', flush=True)
    if count != source['rows'] or digest.hexdigest() != source['sha256']:
        raise ValueError('Test 全量行数或 SHA256 不一致')
    if any(file_stat(Path(source['file']))[k] != source[k] for k in ('bytes', 'mtime_ns')):
        raise ValueError('扫描期间 Test 文件改变')
    return dict(rows=count, history_events=history_count, business_keys_sha256=keys_digest.hexdigest(),
                mapped_records_sha256=mapped_digest.hexdigest(), materialized=False)


def verify_plan(plan: dict, config: dict | None = None) -> None:
    config = config or load_protocol(resolve(plan['config']['config_path']))
    if (plan.get('schema_version') != 'qg-prqk-a0-full-test-plan-v1'
            or plan.get('status') != 'prepared' or plan['config'] != config
            or plan['source_sha256'] != fingerprints()):
        raise ValueError('A0 Test 配置/代码/来源变化，拒绝混用断点')
    for name, receipt in plan['checkpoint']['files'].items():
        if any(file_stat(Path(plan['checkpoint']['path']) / name)[k] != receipt[k] for k in ('bytes', 'mtime_ns')):
            raise ValueError('预检后 checkpoint 文件改变')


def prepare(config: dict) -> dict:
    path = resolve(config['output_root']) / 'plan.json'
    if path.exists():
        plan = load_json(path)
        verify_plan(plan, config)
        return plan
    checkpoint = ev.validate_checkpoint(config, VARIANT)
    if checkpoint['files']['model.safetensors']['sha256'] != config['checkpoint_sha256']:
        raise ValueError('模型不是冻结的 A0 epoch-3 checkpoint')
    tokenizer, template = ev.load_lf_tokenizer_and_template(Path(checkpoint['path']), project_root=project_root())
    index = a0.build_index(config, tokenizer)
    trie = FinalIdPrefixIndex(index)
    if trie.max_length != 13:
        raise ValueError('A0 路径长度与 13 token 上限不一致')
    data = scan_source(config['source'], make_remapper(config), inspect=lambda row: encode_test_record(
        row, variant=VARIANT, tokenizer=tokenizer, template=template, index=index))
    plan = dict(schema_version='qg-prqk-a0-full-test-plan-v1', status='prepared', config=config,
                checkpoint=checkpoint, data=data, source_sha256=fingerprints(),
                preflight=dict(catalog_pois=index.poi_count, checked_formatted_samples=100,
                               remaining_rows_checked_per_chunk=True))
    immutable_json(path, plan)
    return plan


def run_settings(plan: dict, mode: str, smoke_limit: int | None) -> dict:
    """Bind each run to the full protocol, mode and smoke size."""
    config = plan['config']
    settings = dict(plan_signature=signature(plan), decoding=mode, subset='test', split='test', date=config['date'], variant=VARIANT,
                    checkpoint=plan['checkpoint']['path'], geographic_pruning=False,
                    smoke_limit=smoke_limit, num_beams=10, cutoff_len=1024, max_new_tokens=13,
                    data_sha256=config['source']['sha256'], invalid_ids_keep_original_rank=True)
    return settings


def evaluate_test(plan: dict, mode: str, *, records: Sequence, model: Any, tokenizer: Any,
                    template: Any, index: ev.FinalIdIndex, smoke_limit: int | None) -> dict:
    """Use the existing chunk scorer for both modes, preserving illegal beam ranks."""
    import torch
    config = plan['config']
    subset = 'test'
    target_rows = smoke_limit or config['source_rows']
    if len(records) != config['source_rows']:
        raise ValueError('Test 行数不完整')
    tag = f'smoke{smoke_limit}' if smoke_limit else 'results'
    directory = resolve(config['output_root']) / mode / tag
    settings = run_settings(plan, mode, smoke_limit)
    run_signature = signature(settings)
    result_path, progress_path = directory / 'result.json', directory / 'progress.json'
    if result_path.exists():
        result = load_json(result_path)
        if (result.get('status') != 'completed' or result.get('signature') != run_signature or result.get('config') != settings
                or result['metrics']['sample_count'] != target_rows
                or (mode == 'constrained' and result['metrics']['valid_id_rate'] != 1)):
            raise ValueError('已存在不同协议或不完整结果，拒绝覆盖')
        print(f'[{mode}/{subset}] 已完成，复用', flush=True)
        return result
    progress = dict(signature=run_signature, next_line=0, metrics=ev.empty_metrics(),
                    actual_batch_size=config['decoding']['batch_size'], inference_seconds=0.)
    if progress_path.exists():
        progress = load_json(progress_path)
        done = progress['next_line']
        if (progress['signature'] != run_signature or not 0 <= done <= target_rows
                or progress['metrics']['sample_count'] != done
                or progress['metrics']['candidate_count'] != done * 10
                or not 1 <= progress.get('actual_batch_size', 0) <= config['decoding']['batch_size']
                or (mode == 'constrained' and progress['metrics']['valid_candidate_count'] != done * 10)):
            raise ValueError('断点签名或累计计数错误')
    directory.mkdir(parents=True, exist_ok=True)
    while progress['next_line'] < target_rows:
        start = progress['next_line']
        stop = min(start + config['decoding']['chunk_size'], target_rows)
        began = time.perf_counter()
        try:
            metrics = evaluate_test_chunk(records[start:stop], variant=VARIANT, model=model,
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
            raise ValueError('生成候选不完整或约束解码存在非法 ID')
        ev.merge_metrics(progress['metrics'], metrics)
        progress.update(next_line=stop, inference_seconds=progress['inference_seconds'] + time.perf_counter() - began)
        write_json_atomic(progress_path, progress, overwrite=True)
        print(f'[{mode}/{subset}] {stop:,}/{target_rows:,}，batch={progress["actual_batch_size"]}', flush=True)
    result = dict(status='completed', signature=run_signature, config=settings,
                  metrics=ev.finalize_metrics(progress['metrics']),
                  performance={k: progress[k] for k in ('actual_batch_size', 'inference_seconds')})
    immutable_json(result_path, result)
    return result



def schedule(plan: dict, smoke_limit: int | None) -> None:
    """Run the two decoding modes on separate GPUs with isolated logs."""
    devices = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')]
    expected = 2
    if len(devices) != expected or len(set(devices)) != expected or not all(v.strip() for v in devices):
        raise ValueError(f'CUDA_VISIBLE_DEVICES 必须指定 {expected} 张不同的 GPU')
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
        temporary = resolve(f'qg_prqk/outputs/tmp/a0t{gpu}')
        temporary.mkdir(parents=True, exist_ok=True)
        if len(os.fsencode(temporary)) > 64 or not os.access(temporary, os.W_OK):
            raise ValueError('TMPDIR 不可写或超过 64 字节')
        command = [sys.executable, str(resolve('qg_prqk/scripts/evaluate_a0_gid_test.py')),
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

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = {}
        for gpu, mode in enumerate(MODES):
            process = start(mode, gpu)
            futures[pool.submit(forward, process, mode)] = mode
        for future in as_completed(futures):
            if future.result() != 0:
                raise ValueError(f'{futures[future]} 失败，终止本套件另一子进程；保留 chunk 断点')
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



def worker(plan: dict, mode: str, smoke_limit: int | None) -> None:
    verify_plan(plan)
    checkpoint = Path(plan['checkpoint']['path'])
    tokenizer, template = ev.load_lf_tokenizer_and_template(checkpoint, project_root=project_root())
    index = a0.build_index(plan['config'], tokenizer)
    records = A0TestRecords(plan['config']['source'], make_remapper(plan['config']))
    model = ev.load_generation_model(checkpoint, expected_vocab_size=len(tokenizer))
    if mode == 'constrained':
        model = ConstrainedGenerationModel(model, FinalIdPrefixIndex(index))
    evaluate_test(plan, mode, records=records, model=model, tokenizer=tokenizer,
                  template=template, index=index, smoke_limit=smoke_limit)


def summarize(plan: dict, smoke_limit: int | None) -> dict:
    """Require both decoders to cover the same complete frozen Test."""
    root = resolve(plan['config']['output_root'])
    tag = f'smoke{smoke_limit}' if smoke_limit else 'results'
    rows = []
    for mode in MODES:
        result = load_json(root / mode / tag / 'result.json')
        settings = run_settings(plan, mode, smoke_limit)
        metrics = result['metrics']
        if (result.get('status') != 'completed' or result['config'] != settings
                or result['signature'] != signature(settings)
                or metrics['sample_count'] != (smoke_limit or plan['config']['source_rows'])
                or (mode == 'constrained' and metrics['valid_id_rate'] != 1.0)):
            raise ValueError('A0 Test 汇总来源、完整性或约束合法率不一致')
        if any(not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in a0.METRICS):
            raise ValueError('指标不是合法的 0—1 比例')
        rows.append(dict(method='A0-GID (POI-only PRQ-KMeans)', decoding=mode,
                         sample_count=metrics['sample_count'], **{k: metrics[k] for k in a0.METRICS}))
    summary = dict(status='completed', split='test', date=plan['config']['date'],
                   plan_signature=signature(plan), smoke_limit=smoke_limit,
                   business_keys_sha256=plan['data']['business_keys_sha256'], rows=rows,
                   delta_constrained_minus_unconstrained={k: rows[1][k]-rows[0][k] for k in a0.METRICS})
    destination = root / tag
    immutable_json(destination / 'summary.json', summary)
    with (destination / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description='A0-GID 双卡全量 Test：无约束与全目录约束各 606,682 条。')
    parser.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG), help='冻结 Test 配置')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='只读配置、来源和 checkpoint 元数据，不扫描全量或启动 GPU')
    modes.add_argument('--prepare-only', action='store_true', help='CPU 完整模型哈希与全量 Test 标识映射预检，不启动推理')
    parser.add_argument('--smoke-limit', type=int, help='两种解码各生成前 1—100 条，隔离输出')
    parser.add_argument('--worker-mode', choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument('--plan', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
        parser.error('--smoke-limit 必须在 1—100 之间')
    if bool(args.worker_mode) != bool(args.plan) or (args.worker_mode and (args.dry_run or args.prepare_only)):
        parser.error('worker 必须同时给出 --worker-mode/--plan，不能混用准备参数')
    try:
        if args.worker_mode:
            worker(load_json(resolve(args.plan)), args.worker_mode, args.smoke_limit)
            return 0
        config = load_protocol(resolve(args.config))
        if args.dry_run:
            print('静态预检通过：A0 checkpoint-8955；2026-07-14 全量 Test 606,682 条 × 双解码；2×6000D。')
            print('平台准备时核验完整模型哈希及逐条目标/历史映射；未进行 GPU 推理。')
            return 0
        root = resolve(config['output_root'])
        root.mkdir(parents=True, exist_ok=True)
        with (root / '.suite.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.prepare_only:
                a0.validate_hardware(single_a100=False)
            plan = prepare(config)
            if args.prepare_only:
                print('A0_FULL_TEST_PREPARED：完整 Test 映射和 checkpoint 哈希核验通过，未启动 GPU。')
                return 0
            if args.smoke_limit is None:
                schedule(plan, 2)
                summarize(plan, 2)
            schedule(plan, args.smoke_limit)
            summarize(plan, args.smoke_limit)
            tag = f'smoke{args.smoke_limit}' if args.smoke_limit else 'results'
            print(f'A0_FULL_TEST_COMPLETED：两种解码已完成，汇总 {root / tag / "summary.json"}')
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f'A0 全量 Test 失败：{error}', file=sys.stderr)
        return 2
