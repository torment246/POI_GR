"""Visualize frozen S3 changes with identical geographic/semantic parent groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sid.geo import EARTH_RADIUS_METERS, geohash_strings, parent_group_ids

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active'
BEFORE = RUNS / 'poi_prqk_a0_hard60_topk5_v1/full_716245'
AFTER = RUNS / 'poi_query_category_geo_prqk_s3_hard60_topk5_v1/full_291590q_716245p'
OUTPUT = ROOT / 'qg_prqk/outputs/figures/qg_prqk_s3_local_v1'


def parent_collisions(groups: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count unordered colliding POI pairs and colliding POIs in each fixed parent."""
    groups, codes = np.asarray(groups), np.asarray(codes)
    if (groups.ndim != 1 or groups.shape != codes.shape or not len(groups)
            or groups.dtype.kind not in 'iu' or codes.dtype.kind not in 'iu'
            or np.any(groups < 0) or np.any(codes < 0) or np.any(codes >= 512)):
        raise ValueError('父组与 S3 须为对齐的一维整数数组，S3 范围为 0..511')
    keys, counts = np.unique(groups.astype(np.int64) * 512 + codes, return_counts=True)
    size = int(groups.max()) + 1
    pairs = np.bincount(keys // 512, weights=counts * (counts - 1) // 2, minlength=size)
    pois = np.bincount(keys // 512, weights=np.where(counts > 1, counts, 0), minlength=size)
    return pairs.astype(np.int64), pois.astype(np.int64)


def choose_parents(groups: np.ndarray, initial: np.ndarray, hard_parents: np.ndarray,
                   *, seed: int = 42, count: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose examples without inspecting final labels or improvement magnitude."""
    sizes = np.bincount(groups)
    collisions, _ = parent_collisions(groups, initial)
    hard = np.bincount(hard_parents, minlength=len(sizes))
    eligible = np.flatnonzero((sizes >= 4) & (sizes <= 8) & (collisions > 0) & (hard > 0))
    if len(eligible) < count:
        raise ValueError('符合展示条件的父组不足，不回退到按改进幅度选例')
    chosen = np.random.default_rng(seed).choice(eligible, count, replace=False)
    return chosen, {'seed': seed, 'eligible_parent_count': len(eligible), 'chosen_parents': chosen.tolist(),
                    'conditions': '4..8 POIs; initial S3 collision; at least one frozen hard edge',
                    'selection_uses_final_s3': False}


def local_coordinates(longitudes: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    """Express actual locations in meters; use neither t-SNE nor point jitter."""
    lng, lat = np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float)
    if lng.shape != lat.shape or not np.isfinite(lng).all() or not np.isfinite(lat).all():
        raise ValueError('经纬度未对齐或包含非有限值')
    return np.column_stack((EARTH_RADIUS_METERS * np.deg2rad(lng - lng.mean()) * np.cos(np.deg2rad(lat.mean())),
                            EARTH_RADIUS_METERS * np.deg2rad(lat - lat.mean())))


def same_code_pairs(codes: np.ndarray) -> np.ndarray:
    """Return each same-code pair exactly once, without self pairs."""
    codes = np.asarray(codes)
    left, right = np.triu_indices(len(codes), 1)
    return np.column_stack((left, right))[codes[left] == codes[right]]


def _checked(directory: Path, manifest: dict, name: str, sources: dict) -> Path:
    path = directory / name
    actual = sha256_file(path)
    expected = manifest['artifacts'][name]['sha256']
    if actual != expected:
        raise ValueError(f'冻结产物哈希不匹配：{path}')
    sources[str(path.relative_to(ROOT))] = actual
    return path


def analyze() -> dict[str, Any]:
    """Read only frozen assignments, catalog metadata and hard edges."""
    sources: dict[str, str] = {}
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in (BEFORE, AFTER)]
    for directory in (BEFORE, AFTER):
        if not (directory / '_SUCCESS').is_file():
            raise ValueError(f'冻结阶段没有成功标记：{directory}')
        sources[str((directory / 'manifest.json').relative_to(ROOT))] = sha256_file(directory / 'manifest.json')
    def old(name):
        return _checked(BEFORE, manifests[0], name, sources)
    def new(name):
        return _checked(AFTER, manifests[1], name, sources)
    groups = np.load(new('poi_parent_groups.npy'), mmap_mode='r')
    selected = np.load(new('selected_poi_rows.npy'), mmap_mode='r')
    initial = np.load(old('poi_assignments_s3.npy'), mmap_mode='r')[selected]
    final = np.load(new('poi_assignments_s3.npy'), mmap_mode='r')
    sid = np.load(new('poi_sid_s1_s2_s3.npy'), mmap_mode='r')
    gid = np.load(new('poi_gid6.npy'), mmap_mode='r')
    assert np.array_equal(selected, np.arange(716245))
    assert np.array_equal(sid[:, 2], final)
    assert np.array_equal(parent_group_ids(gid, sid[:, 0], sid[:, 1]), groups)
    edges = pq.read_table(new('hard_entity_edges.parquet'), columns=[
        'poi_row_index', 'neighbor_poi_row_index', 'parent_group', 'composite_similarity'])
    src, dst = [edges[k].to_numpy() for k in ('poi_row_index', 'neighbor_poi_row_index')]
    edge_parents = edges['parent_group'].to_numpy()
    weights = edges['composite_similarity'].to_numpy().astype(np.float64)
    assert np.array_equal(groups[src], groups[dst]) and np.array_equal(groups[src], edge_parents)
    old_pairs, old_pois = parent_collisions(groups, initial)
    new_pairs, new_pois = parent_collisions(groups, final)
    sizes = np.bincount(groups)
    denominator = int(np.sum(sizes * (sizes - 1) // 2))
    before_edge, after_edge = initial[src] == initial[dst], final[src] == final[dst]
    metrics = {
        'poi_count': len(groups), 'parent_count': len(sizes), 'non_singleton_parents': int(np.sum(sizes > 1)),
        'fixed_parent_pair_count': denominator, 'hard_directed_edge_count': len(src),
        'before': {'collision_pairs': int(old_pairs.sum()), 'colliding_pois': int(old_pois.sum()),
                   'hard_collision_rate': float(before_edge.mean()),
                   'hard_weighted_collision_rate': float(weights[before_edge].sum() / weights.sum())},
        'after': {'collision_pairs': int(new_pairs.sum()), 'colliding_pois': int(new_pois.sum()),
                  'hard_collision_rate': float(after_edge.mean()),
                  'hard_weighted_collision_rate': float(weights[after_edge].sum() / weights.sum())},
        'parent_outcomes': {'fewer_collisions': int(np.sum(new_pairs < old_pairs)),
                            'unchanged': int(np.sum((new_pairs == old_pairs) & (sizes > 1))),
                            'more_collisions': int(np.sum(new_pairs > old_pairs))},
    }
    saved = json.loads(new('metrics.json').read_text())
    assert abs(metrics['after']['hard_weighted_collision_rate'] - saved['hard_entity_collision']['weighted_collision_rate']) < 1e-6
    chosen, selection = choose_parents(groups, initial, edge_parents)
    metadata_path = new('poi_geo_metadata.parquet')
    cases = []
    columns = ['poi_row_index', 'poi_id', 'displayname', 'category', 'category_code', 'lat', 'lng', 'parent_group']
    for group in chosen:
        members = pq.read_table(metadata_path, columns=columns, filters=[('parent_group', '=', int(group))]).to_pylist()
        members.sort(key=lambda p: p['poi_row_index'])
        rows = np.asarray([p['poi_row_index'] for p in members])
        assert np.array_equal(rows, np.flatnonzero(groups == group))
        xy = local_coordinates(np.asarray([p['lng'] for p in members]), np.asarray([p['lat'] for p in members]))
        for i, (member, row) in enumerate(zip(members, rows)):
            member.update(number=i + 1, initial_s3=int(initial[row]), final_s3=int(final[row]),
                          x_m=float(xy[i, 0]), y_m=float(xy[i, 1]))
        cases.append({'parent_group': int(group), 'gid6': next(geohash_strings(gid[rows[:1]])),
                      's1': int(sid[rows[0], 0]), 's2': int(sid[rows[0], 1]), 'poi_count': len(rows),
                      'before_collision_pairs': int(old_pairs[group]), 'after_collision_pairs': int(new_pairs[group]),
                      'members': members})
    return {'schema_version': 'qg-prqk-s3-local-visualization-v1', 'sources': sources, 'metrics': metrics,
            'selection': selection, 'cases': cases,
            'comparison': 'Fixed final GID6/S1/S2 and POI membership; P7 actual PRQ-KMeans S3 initialization vs final S3.',
            'limits': ['Not a full POI-only vs full-method ablation: S1/S2 are fixed to the final prefix.',
                       'Not Geo-only attribution: Query, Geo and hard-entity terms update S3 together.',
                       'No Query/SFT inference or evaluation was performed; fewer collisions do not prove retrieval improvement.'],
            'source_access': {'embedding_read': False, 'validation_test_read': False, 'training': False}}


def draw(data: dict[str, Any], output: Path) -> None:
    """Use actual coordinates and unchanged point positions across both panels."""
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'qg_prqk/outputs/tmp/s3viz_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = ROOT / 'qg_prqk/outputs/inputs/fonts/NotoSansCJKsc-Regular.otf'
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font_path)).get_name(),
                         'font.size': 17, 'axes.unicode_minus': False})
    names = ['PRQ-KMeans 初始化', 'QG-HRQ 优化后']
    metrics = data['metrics']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, title, multiplier in [
        (axes[0], 'collision_pairs', '固定父组内同码 POI 对数', 1),
        (axes[1], 'hard_weighted_collision_rate', '困难边加权同码率 (%)', 100),
    ]:
        values = [metrics[k][key] * multiplier for k in ('before', 'after')]
        bars = ax.bar(names, values, width=.50, color=['#9BAFC4', '#3977B8'])
        ax.bar_label(bars, labels=[f'{v:,.0f}' if multiplier == 1 else f'{v:.2f}' for v in values], padding=8, fontsize=20)
        ax.set_ylim(0, max(values) * 1.22); ax.set_title(title, fontsize=20, pad=18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=17)
    fig.tight_layout(w_pad=3)
    fig.savefig(output / 's3_local_overview.png', dpi=180); plt.close(fig)
    for case_index, case in enumerate(data['cases'], 1):
        members = case['members']
        xy = np.asarray([[p['x_m'], p['y_m']] for p in members])
        codes = [np.asarray([p[key] for p in members]) for key in ('initial_s3', 'final_s3')]
        unique_codes = np.unique(np.concatenate(codes))
        palette = plt.get_cmap('tab20')
        colors = {int(c): palette(2 * i if i < 10 else 2 * (i - 10) + 1) for i, c in enumerate(unique_codes)}
        low, high = xy.min(axis=0), xy.max(axis=0)
        # Equal physical scales, shared limits, and no coordinate perturbation.
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])) * 1.8, 40.0)
        center = (low + high) / 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.35), sharex=True, sharey=True)
        for ax, values, name in zip(axes, codes, names):
            for left, right in same_code_pairs(values):
                ax.plot(xy[[left, right], 0], xy[[left, right], 1], color=colors[int(values[left])], alpha=.55, linewidth=1.3, zorder=1)
            for i, (point, code) in enumerate(zip(xy, values)):
                ax.scatter(*point, s=105, color=colors[int(code)], edgecolor='white', linewidth=.8, zorder=3)
                # Only the labels are offset, identically in both panels.
                distances = np.linalg.norm(xy - point, axis=1)
                distances[i] = np.inf
                nearest = int(np.argmin(distances))
                delta = point - xy[nearest]
                offset = (8 if delta[0] >= 0 else -17, 10 if delta[1] >= 0 else -20)
                if point[1] < center[1] - span * .25:
                    offset = (offset[0], 10)
                ax.annotate(str(i + 1), point, xytext=offset, textcoords='offset points', fontsize=17)
            ax.set_xlim(center[0] - span * .62, center[0] + span * .62)
            ax.set_ylim(center[1] - span * .34, center[1] + span * .34)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('东西向距离（米）', fontsize=16)
            ax.set_title(name, fontsize=20, pad=12)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('南北向距离（米）', fontsize=16)
        fig.subplots_adjust(left=.075, right=.975, top=.85, bottom=.18, wspace=.20)
        fig.savefig(output / f's3_local_case_{case_index}.png', dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='固定 GID/S1/S2，绘制 S3 初始化与优化后的真实坐标和同码关系。')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT, help='新输出目录，禁止覆盖已有分析')
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError('输出目录已存在，请指定新目录')
        if not args.output_dir.resolve().is_relative_to(ROOT / 'qg_prqk'):
            raise ValueError('输出必须在 qg_prqk 下')
        data = analyze()
        args.output_dir.mkdir(parents=True)
        write_json_atomic(args.output_dir / 'analysis.json', data)
        draw(data, args.output_dir)
        write_json_atomic(args.output_dir / 'manifest.json', {
            'schema_version': data['schema_version'], 'status': 'completed', 'sources': data['sources'],
            'implementation_sha256': sha256_file(Path(__file__)),
            'artifacts': {p.name: sha256_file(p) for p in args.output_dir.iterdir() if p.is_file()},
        })
        print(json.dumps({'output': str(args.output_dir), 'metrics': data['metrics'], 'selection': data['selection']}, ensure_ascii=False))
    except (ValueError, OSError, AssertionError, KeyError) as error:
        parser.exit(2, f'S3 可视化失败：{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
