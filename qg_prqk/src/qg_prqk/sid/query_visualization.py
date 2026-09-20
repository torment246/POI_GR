"""Measure target-prefix concentration on frozen Train D1/D2 query edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sid.local_visualization import ROOT, RUNS, BEFORE, AFTER, _checked

GRAPH = RUNS / 'poi_query_category_prqk_s1_s2_hard60_topk5_v1/full_342879q_716245p'
OUTPUT = ROOT / 'qg_prqk/outputs/figures/qg_prqk_query_prefix_v1'
METRICS = ('top_prefix_share', 'prefix_count', 'entropy_bits', 'expected_bucket_size')


def prefix_codes(sid: np.ndarray, depth: int) -> np.ndarray:
    """Encode a semantic prefix without adding GID or Dedup."""
    sid = np.asarray(sid)
    if depth not in (1, 2) or sid.ndim != 2 or sid.shape[1] < depth:
        raise ValueError('仅支持 D1/S1 与 D2/S1+S2 前缀')
    if sid.dtype.kind not in 'iu' or np.any(sid[:, :depth] < 0) or np.any(sid[:, :depth] >= 512):
        raise ValueError('SID 必须是范围 0..511 的整数')
    codes = sid[:, 0].astype(np.int64)
    return codes if depth == 1 else codes * 512 + sid[:, 1]


def distribution(codes: np.ndarray, counts: np.ndarray, bucket_sizes: np.ndarray) -> dict[str, float]:
    """Normalize order counts within a query; keep catalog bucket size separate."""
    codes, counts = np.asarray(codes), np.asarray(counts, dtype=np.float64)
    if (codes.ndim != 1 or codes.shape != counts.shape or len(codes) == 0
            or not np.isfinite(counts).all() or np.any(counts <= 0)):
        raise ValueError('每个 Query 需要对齐的有效前缀与正订单计数')
    unique, inverse = np.unique(codes, return_inverse=True)
    if np.any(unique < 0) or np.any(unique >= len(bucket_sizes)) or np.any(bucket_sizes[unique] <= 0):
        raise ValueError('前缀不在全目录桶索引中')
    mass = np.bincount(inverse, weights=counts) / counts.sum()
    return {'top_prefix_share': float(mass.max()), 'prefix_count': int(len(unique)),
            'entropy_bits': float(-np.sum(mass * np.log2(mass))),
            'expected_bucket_size': float(np.dot(mass, bucket_sizes[unique]))}


def select_case(candidates: list[dict], rng: np.random.Generator, outcome: str = 'any') -> dict:
    """Sample within an explicitly declared outcome stratum, never by best delta."""
    if outcome not in ('any', 'improved', 'worsened'):
        raise ValueError('未知选例分层')
    eligible = [q for q in candidates if 3 <= len(q['poi_rows']) <= 4
                and len(q['query']) <= 12 and q['query_count'] >= 5
                and q['before']['prefix_count'] > 1]
    if outcome != 'any':
        sign = 1 if outcome == 'improved' else -1
        eligible = [q for q in eligible if sign * (q['after']['top_prefix_share'] - q['before']['top_prefix_share']) > 1e-12]
    if not eligible:
        raise ValueError('没有符合预设排版条件的 Query；不按改进幅度回退选例')
    selected = dict(eligible[int(rng.integers(len(eligible)))])
    selected['eligible_case_count'] = len(eligible)
    selected['selection_outcome'] = outcome
    return selected


def select_projection_cases(candidates: list[dict], seed: int = 42) -> list[dict]:
    """Select five improved D2 examples, with disjoint targets and no delta ranking."""
    eligible = [q for q in candidates if 5 <= len(q['poi_rows']) <= 20
                and len(q['query']) <= 12 and q['query_count'] >= 5
                and q['after']['top_prefix_share'] > q['before']['top_prefix_share'] + 1e-12]
    selected, used = [], set()
    for index in np.random.default_rng(seed).permutation(len(eligible)):
        candidate = eligible[index]
        if used.intersection(candidate['poi_rows']):
            continue
        selected.append(dict(candidate))
        used.update(candidate['poi_rows'])
        if len(selected) == 5:
            return selected
    raise ValueError('不足五组不重叠的 D2 改善样例，不按图形效果回退选例')


def analyze(case_selection: str = 'random', with_tsne: bool = False) -> dict[str, Any]:
    """Read frozen labels/edges only, with exact artifact hash verification."""
    sources: dict[str, str] = {}
    manifests = {}
    for directory in (BEFORE, AFTER, GRAPH):
        if not (directory / '_SUCCESS').is_file():
            raise ValueError(f'冻结数据缺少成功标记：{directory}')
        path = directory / 'manifest.json'
        manifests[directory] = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = sha256_file(path)

    def checked(directory, name):
        return _checked(directory, manifests[directory], name, sources)

    old_sid = np.column_stack([np.load(checked(BEFORE, f'poi_assignments_s{i}.npy'), mmap_mode='r') for i in (1, 2)])
    new_sid = np.load(checked(AFTER, 'poi_sid_s1_s2_s3.npy'), mmap_mode='r')
    selected_rows = np.load(checked(AFTER, 'selected_poi_rows.npy'), mmap_mode='r')
    if len(old_sid) != 716245 or not np.array_equal(selected_rows, np.arange(len(old_sid))):
        raise ValueError('POI 行映射或目录规模与冻结全库不一致')
    nodes = pq.read_table(checked(GRAPH, 'query_nodes.parquet'),
                          columns=['node_row', 'query_id', 'normalized_query', 'query_count', 'supervision_depth'],
                          filters=[('supervision_depth', 'in', [1, 2])]).to_pylist()
    node_index = {q['node_row']: q for q in nodes}
    edge_file = checked(GRAPH, 'query_poi_edges_s1_s2.parquet')
    depth_metrics, cases, projection_cases = {}, [], []
    rng = np.random.default_rng(42)
    for depth in (1, 2):
        edges = pq.read_table(edge_file, columns=['node_row', 'poi_row_index', 'pair_count', 'edge_probability'],
                              filters=[('supervision_depth', '=', depth), ('layer', '=', depth)])
        node, poi, counts, probability = [edges[k].to_numpy() for k in edges.column_names]
        order = np.argsort(node, kind='stable')
        node, poi, counts, probability = (a[order] for a in (node, poi, counts, probability))
        cuts = np.r_[0, np.flatnonzero(np.diff(node)) + 1, len(node)]
        codes = [prefix_codes(sid, depth) for sid in (old_sid, new_sid)]
        buckets = [np.bincount(c, minlength=512 ** depth) for c in codes]
        queries = []
        for start, stop in zip(cuts[:-1], cuts[1:]):
            rows, mass = poi[start:stop], counts[start:stop]
            q = node_index[int(node[start])]
            if len(np.unique(rows)) != len(rows) or mass.sum() > q['query_count']:
                raise ValueError('重复 Query–POI 边或订单计数不守恒')
            if not np.allclose(mass / mass.sum(), probability[start:stop], atol=1e-12, rtol=1e-10):
                raise ValueError('有效边的概率与订单计数不一致')
            if len(rows) < 2:
                continue
            record = {'query_id': q['query_id'], 'node_row': q['node_row'], 'query': q['normalized_query'],
                      'query_count': q['query_count'], 'depth': depth, 'poi_rows': rows.tolist(),
                      'pair_counts': mass.tolist(), 'retained_orders': int(mass.sum()),
                      'before': distribution(codes[0][rows], mass, buckets[0]),
                      'after': distribution(codes[1][rows], mass, buckets[1])}
            queries.append(record)
        if not queries:
            raise ValueError(f'D{depth} 没有有效多目标 Query')
        delta = np.asarray([q['after']['top_prefix_share'] - q['before']['top_prefix_share'] for q in queries])
        depth_metrics[f'D{depth}'] = {
            'all_depth_queries': sum(q['supervision_depth'] == depth for q in nodes),
            'multi_target_queries': len(queries), 'effective_edges': sum(len(q['poi_rows']) for q in queries),
            'before': {key: float(np.mean([q['before'][key] for q in queries])) for key in METRICS},
            'after': {key: float(np.mean([q['after'][key] for q in queries])) for key in METRICS},
            'top_share_outcomes': {'improved': int(np.sum(delta > 1e-12)),
                                   'unchanged': int(np.sum(np.abs(delta) <= 1e-12)),
                                   'worsened': int(np.sum(delta < -1e-12))},
        }
        outcome = ('improved' if case_selection == 'positive' else
                   ('improved' if depth == 1 else 'worsened') if case_selection == 'contrast' else 'any')
        cases.append(select_case(queries, rng, outcome))
        if depth == 2 and with_tsne:
            projection_cases = select_projection_cases(queries)
        print(f'D{depth} 已完成：{len(queries):,} 条有效多目标 Query', flush=True)
    rows = sorted({row for case in cases + projection_cases for row in case['poi_rows']})
    metadata = pq.read_table(checked(AFTER, 'poi_geo_metadata.parquet'),
                            columns=['poi_row_index', 'poi_id', 'displayname', 'category'],
                            filters=[('poi_row_index', 'in', rows)]).to_pylist()
    metadata = {m['poi_row_index']: m for m in metadata}
    for case in cases + projection_cases:
        case['members'] = [{**metadata[row], 'pair_count': count,
                            'before_prefix': old_sid[row, :case['depth']].tolist(),
                            'after_prefix': new_sid[row, :case['depth']].tolist()}
                           for row, count in zip(case.pop('poi_rows'), case.pop('pair_counts'))]
    return {'schema_version': 'qg-prqk-query-prefix-visualization-v1', 'sources': sources,
            'catalog_pois': len(old_sid), 'metrics': depth_metrics, 'cases': cases,
            'projection_cases': projection_cases,
            'selection': {'seed': 42, 'uses_final_codes': case_selection != 'random', 'case_selection': case_selection,
                          'rule': 'D1 then D2; 3..4 effective targets, query length <=12, orders >=5, initial prefix count >1',
                          'outcome_strata': ('D1 and D2 improved, random within stratum' if case_selection == 'positive' else
                                             'D1 improved / D2 worsened top-prefix share, random within stratum' if case_selection == 'contrast' else 'none'),
                          'interpretation': 'Directional examples are not a population estimate; use the complete macro statistics.'},
            'contract': {'split': 'Train', 'targets': 'frozen dominant-category effective edges, not all raw order targets',
                         'within_query_weight': 'pair_count / retained_pair_count_sum', 'across_queries': 'uniform macro mean',
                         'prefixes': 'D1: S1; D2: S1/S2; excludes GID and Dedup',
                         'bucket_size': 'sum_prefix P(prefix|query) * full_catalog_prefix_size',
                         'comparison': 'POI-only PRQ-KMeans vs complete QG-HRQ (GID-parent version)',
                         'limits': 'Train structural description, not Query-only attribution or held-out retrieval; D3 excluded'},
            'source_access': {'embedding_read': False, 'validation_test_read': False, 'training': False}}


def prefix_features(sid: np.ndarray, books: list[np.ndarray]) -> np.ndarray:
    """Concatenate unit S1/S2 codewords; numeric token IDs are not coordinates."""
    if sid.ndim != 2 or sid.shape[1] != 2 or len(books) != 2:
        raise ValueError('需要两层前缀与两层码本')
    prefix_codes(sid, 2)
    parts = []
    for level, book in enumerate(books):
        values = np.asarray(book[sid[:, level]], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all() or np.any(norms <= 1e-12):
            raise ValueError('前缀码向量包含零范数或无效值')
        parts.append(values / norms / np.sqrt(2.0))
    return np.concatenate(parts, axis=1)


def build_tsne(data: dict, output: Path) -> dict:
    """Fit one joint projection of unique prefix vectors, without outcome tuning."""
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'qg_prqk/outputs/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    cases = data['projection_cases']
    labels = np.concatenate([np.full(len(q['members']), i) for i, q in enumerate(cases)])
    rows = [m['poi_row_index'] for q in cases for m in q['members']]
    features = []
    for directory, variant in ((BEFORE, 'before'), (GRAPH, 'after')):
        manifest = json.loads((directory / 'manifest.json').read_text())
        books = [np.load(_checked(directory, manifest, f'poi_codebook_s{i}.npy', data['sources'])) for i in (1, 2)]
        sid = np.asarray([m[f'{variant}_prefix'] for q in cases for m in q['members']])
        features.append(prefix_features(sid, books))
    # Identical prefix vectors are projected once, then drawn with count-scaled area.
    # This prevents random t-SNE separation of POIs that actually share a prefix.
    unique, inverse = np.unique(np.concatenate(features), axis=0, return_inverse=True)
    perplexity = min(10., (len(unique) - 1) / 3.)
    with threadpool_limits(limits=4):
        reduced = PCA(n_components=min(50, len(unique) - 1), random_state=42).fit_transform(unique)
        model = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                     init='pca', learning_rate='auto', random_state=42, n_jobs=4)
        coordinates = model.fit_transform(reduced)[inverse].reshape(2, len(rows), 2)
    if not np.isfinite(coordinates).all():
        raise ValueError('投影包含无效坐标')
    colors = ['#4878A8', '#E69F45', '#58A57D', '#B36B96', '#8D80BB']
    low, high = coordinates.min(axis=(0, 1)), coordinates.max(axis=(0, 1))
    margin = np.maximum((high - low) * .10, 1.)
    for method, variant in enumerate(('before', 'after')):
        figure, axis = plt.subplots(figsize=(6.0, 4.4), dpi=200)
        for label, color in enumerate(colors):
            points, counts = np.unique(coordinates[method, labels == label], axis=0, return_counts=True)
            axis.scatter(points[:, 0], points[:, 1], s=45 * counts, c=color, alpha=.85,
                         edgecolors='white', linewidths=.6)
        axis.set_xlim(low[0] - margin[0], high[0] + margin[0])
        axis.set_ylim(low[1] - margin[1], high[1] + margin[1])
        axis.set_aspect('equal', adjustable='box')
        axis.axis('off')
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        figure.savefig(output / f'query_tsne_{variant}.png', facecolor='white')
        plt.close(figure)
    record = {'poi_rows': rows, 'query_labels': labels.tolist(), 'coordinates': coordinates.tolist(),
              'colors': colors, 'queries': [q['query'] for q in cases],
              'seed': 42, 'perplexity': perplexity, 'max_iter': 1000, 'unique_vectors': len(unique),
              'kl_divergence': float(model.kl_divergence_),
              'representation': 'concatenated unit POI S1/S2 codewords, equal level weights; no GID/S3/Dedup',
              'projection': 'one joint PCA+t-SNE fit on unique vectors; identical vectors share coordinates; common axes; no jitter',
              'selection': 'D2 improved top-prefix share, 5..20 targets, query length <=12, orders>=5; seed42 permutation; first five disjoint target sets; not ranked by improvement',
              'scope': 'selected Train improvement examples, not population evidence or Query-only attribution'}
    write_json_atomic(output / 'tsne.json', record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description='分析冻结 D1/D2 Query 的目标前缀分布，为 PPT 提供真实树图和全量统计。')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT, help='qg_prqk 内的新目录，不覆盖已有结果')
    parser.add_argument('--case-selection', choices=['random', 'contrast', 'positive'], default='random',
                        help='random 为不看结果抽样；contrast 为正反例；positive 为 D1/D2 改善样例')
    parser.add_argument('--with-tsne', action='store_true', help='补充五组 D2 改善样例的联合前缀码向量 t-SNE')
    args = parser.parse_args()
    try:
        if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(ROOT / 'qg_prqk'):
            raise ValueError('请指定 qg_prqk 下尚不存在的输出目录')
        data = analyze(args.case_selection, args.with_tsne)
        args.output_dir.mkdir(parents=True)
        artifacts = ['analysis.json']
        if args.with_tsne:
            build_tsne(data, args.output_dir)
            artifacts.extend(['tsne.json', 'query_tsne_before.png', 'query_tsne_after.png'])
        write_json_atomic(args.output_dir / 'analysis.json', data)
        write_json_atomic(args.output_dir / 'manifest.json', {
            'status': 'completed', 'sources': data['sources'], 'contract': data['contract'],
            'implementation_sha256': sha256_file(Path(__file__)),
            'artifacts': {name: sha256_file(args.output_dir / name) for name in artifacts},
        })
        print(json.dumps({'output': str(args.output_dir), 'metrics': data['metrics'],
                          'cases': [{k: c[k] for k in ('query', 'depth', 'before', 'after')} for c in data['cases']]}, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Query 可视化分析失败：{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
