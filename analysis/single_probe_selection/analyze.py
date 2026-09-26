"""Fixed top-20% selection from archived individual Exp271 probes."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/single_probe_selection_20260923'
ARCHIVES = {m: ROOT / 'results/remote_runs' / name for m, name in {
    'sd3': 'exp271_sd3_complete_20260913',
    'flux': 'exp271_flux_complete_20260914',
    'qwen': 'exp271_qwen_complete_20260916'}.items()}
SUFFIX = Path('results/heldout_prediction/runtime')
SOURCE = ARCHIVES['qwen'] / SUFFIX / 'analysis.json'
CALIBRATION = ARCHIVES['qwen'] / 'pipelines/heldout_prediction/development_calibration.json'
PREVIOUS = ROOT / 'results/frozen_selection/results.json'
STEPS = {'sd3': [6, 10], 'flux': [2, 3], 'qwen': [10, 16]}
METRICS = ['R', 'stereotype_reduction', 'matching_change']
DRAWS, SEED = 10000, 20260913
PROVENANCE = {}


def read(path):
    # A bounded file at a time; never emit raw records into conversation.
    assert path.stat().st_size < 8_000_000, path
    blob = path.read_bytes()
    PROVENANCE[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def summary(point, samples):
    return {'estimate': float(point), 'ci95': np.quantile(samples, [.025, .975]).tolist()}


def rho(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def load_model(model, data, calibration):
    records = {}
    for path in sorted((ARCHIVES[model] / SUFFIX / model).glob('shard_[0-9][0-9][0-9].json')):
        shard = read(path)
        done = read(path.with_name(path.stem + '_complete.json'))
        assert done['manifest_sha256'] == PROVENANCE[str(path.relative_to(ROOT))]
        for key, record in shard['cases'].items():
            assert key not in records
            records[key] = record['probe']
    cases = sorted([c for c in data['cases'] if c['status'] == 'PRIMARY_VALID'
                    and all(c['S'][a] is not None for a in ['clean', 'steer', 'random0'])
                    and all(k in c for k in ['G', 'M', 'H', 'A0'])], key=lambda c: c['triplet_key'])
    assert len(cases) == 1295 and len(records) == len(data['cases']) == 1314
    cells = [(seed, step) for seed in [6, 7, 8] for step in STEPS[model]]
    rows = []
    for c in cases:
        key = c['triplet_key']
        p = records[key]
        assert p['status'] == 'PRIMARY_VALID'
        mapped = {(r['seed'], r['timestep']): r for r in p['cells']}
        assert len(mapped) == len(p['cells']) == 6 and set(mapped) == set(cells)
        assert all(r['primary_status'] == 'VALID' for r in mapped.values())
        cos = [mapped[cell]['cosine'] for cell in cells]
        mag = [mapped[cell]['magnitude'] for cell in cells]
        assert np.isfinite(cos + mag).all()
        assert max(map(abs, cos)) <= 1.000001 and min(mag) >= 0
        assert math.isclose((np.mean(cos)-calibration['mu'])/calibration['sd'], c['G'], abs_tol=1e-9)
        assert math.isclose(np.mean(mag), c['M'], abs_tol=1e-9)
        assert math.isclose(c['R'], c['S']['steer']-c['S']['clean'], abs_tol=1e-12)
        rows.append({'key': key, 'R': c['R'], 'G': c['G'], 'M': c['M'],
                     'stereotype_reduction': c['components']['clean']['bias']-c['components']['steer']['bias'],
                     'matching_change': c['components']['steer']['alignment']-c['components']['clean']['alignment'],
                     'cosines': cos, 'magnitudes': mag})
    assert len(set(r['key'] for r in rows)) == 1295
    return {'cells': cells, 'rows': rows, 'coverage': len(records)}


def analyze(model, inputs, original, previous):
    rows, cells = inputs['rows'], inputs['cells']
    n, k = len(rows), math.ceil(.2*len(rows))
    y = np.array([[r[m] for m in METRICS] for r in rows])
    x = {'G6': np.array([r['G'] for r in rows]), 'M6': np.array([r['M'] for r in rows])}
    for j in range(6):
        x[f'G1_{j}'] = np.array([r['cosines'][j] for r in rows])
        x[f'M1_{j}'] = np.array([r['magnitudes'][j] for r in rows])
    ix = np.arange(n)

    def select(indices, values):
        return indices[np.lexsort((indices, -values[indices]))][:k]

    selected = {s: select(ix, scores) for s, scores in x.items()}
    point = {s: y[v].mean(axis=0) for s, v in selected.items()}
    point['ALL'] = y.mean(axis=0)
    for label, orig in [('G6', 'G'), ('M6', 'M')]:
        assert math.isclose(rho(x[label], y[:, 0]), original['G_M'][orig]['rho'], abs_tol=1e-12)
        assert [rows[i]['key'] for i in selected[label]] == previous['strategies'][orig]['selected_keys']
        assert math.isclose(point[label][0], previous['strategies'][orig]['metrics']['R']['estimate'], abs_tol=1e-12)
    cos_average = np.array([np.mean(r['cosines']) for r in rows])
    assert np.array_equal(select(ix, cos_average), selected['G6'])
    # At full coverage all orderings retain the same outcomes.
    for values in x.values():
        assert np.allclose(y[np.lexsort((ix, -values))].mean(axis=0), point['ALL'])

    boot = {s: np.empty((DRAWS, len(METRICS))) for s in [*x, 'ALL']}
    rng = np.random.default_rng(20260913)
    for b in range(DRAWS):
        indices = rng.integers(0, n, n)
        boot['ALL'][b] = y[indices].mean(axis=0)
        for s, values in x.items():
            boot[s][b] = y[select(indices, values)].mean(axis=0)
    # The exact same estimator/seed reproduces the existing six-probe CIs.
    for label, orig in [('G6', 'G'), ('M6', 'M')]:
        assert np.allclose(np.quantile(boot[label][:, 0], [.025, .975]),
                           previous['strategies'][orig]['metrics']['R']['ci95'], atol=1e-10)
    for name, prefix in [('G1_mean', 'G1'), ('M1_mean', 'M1')]:
        point[name] = np.mean([point[f'{prefix}_{j}'] for j in range(6)], axis=0)
        boot[name] = np.mean([boot[f'{prefix}_{j}'] for j in range(6)], axis=0)
    results = {s: {'metrics': {m: summary(point[s][j], boot[s][:, j]) for j, m in enumerate(METRICS)}} for s in point}
    for s, indices in selected.items():
        results[s]['selected_keys'] = [rows[i]['key'] for i in indices]
        results[s]['rho'] = rho(x[s], y[:, 0])
        if s.startswith(('G1_', 'M1_')):
            results[s]['probe_seed'], results[s]['probe_step'] = cells[int(s[-1])]
    for s, prefix in [('G1_mean', 'G1'), ('M1_mean', 'M1')]:
        results[s]['configuration_range'] = {m: [float(min(point[f'{prefix}_{i}'][j] for i in range(6))),
                                                float(max(point[f'{prefix}_{i}'][j] for i in range(6)))]
                                           for j, m in enumerate(METRICS)}
    contrasts = {}
    for a, b in [('G1_mean', 'G6'), ('G1_mean', 'M1_mean'), ('G1_mean', 'M6'),
                 ('G1_mean', 'ALL'), ('G6', 'M6')]:
        contrasts[f'{a}_minus_{b}'] = {m: summary(point[a][j]-point[b][j], boot[a][:, j]-boot[b][:, j])
                                     for j, m in enumerate(METRICS)}
    return {'n': n, 'n_selected': k, 'n_coverage': inputs['coverage'], 'n_excluded': inputs['coverage']-n,
            'strategies': results, 'contrasts': contrasts,
            'retained_mean_R_percent': float(100*point['G1_mean'][0]/point['G6'][0]),
            'retained_lift_over_ALL_percent': float(100*(point['G1_mean'][0]-point['ALL'][0])/(point['G6'][0]-point['ALL'][0])),
            'single_probe_rho_mean': float(np.mean([results[f'G1_{j}']['rho'] for j in range(6)])),
            'six_probe_reproduction': 'Passed correlations, selected keys, mean R and bootstrap intervals'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original, calibration, previous = read(SOURCE), read(CALIBRATION), read(PREVIOUS)
    inputs, result = {}, {'simulated': False, 'fraction': .2, 'seed': SEED, 'bootstrap': DRAWS,
                         'judge': 'Qwen3-VL', 'outcome_seeds': [3, 4], 'models': {},
                         'scope': 'Supplementary cached-data analysis; all six single-probe configurations retained'}
    for model in ['sd3', 'flux', 'qwen']:
        inputs[model] = load_model(model, original['models'][model], calibration[model])
        print(model, 'inputs verified', flush=True)
        result['models'][model] = analyze(model, inputs[model], original['models'][model], previous['models'][model])
        d = result['models'][model]
        print(json.dumps({'model': model, 'G6': d['strategies']['G6']['metrics']['R'],
                          'G1_mean': d['strategies']['G1_mean'], 'G1_minus_M1': d['contrasts']['G1_mean_minus_M1_mean']['R'],
                          'retained_percent': d['retained_mean_R_percent']}), flush=True)
        write(OUT / f'{model}_results.json', d)
    write(OUT / 'inputs.json', inputs)
    result['provenance'] = PROVENANCE
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['plan_sha256'] = hashlib.sha256((HERE/'ANALYSIS_PLAN.md').read_bytes()).hexdigest()
    result['inputs_sha256'] = hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest()
    write(OUT/'results.json', result)
    with (OUT/'summary.csv').open('w', newline='') as f:
        fields = ['model', 'G6_R', 'G1_mean_R', 'G1_min_R', 'G1_max_R', 'M1_mean_R', 'ALL_R', 'retained_mean_R_percent']
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n");w.writeheader()
        for model, d in result['models'].items():
            s = d['strategies']
            w.writerow({'model': model, 'G6_R': s['G6']['metrics']['R']['estimate'],
                        'G1_mean_R': s['G1_mean']['metrics']['R']['estimate'],
                        'G1_min_R': s['G1_mean']['configuration_range']['R'][0],
                        'G1_max_R': s['G1_mean']['configuration_range']['R'][1],
                        'M1_mean_R': s['M1_mean']['metrics']['R']['estimate'],
                        'ALL_R': s['ALL']['metrics']['R']['estimate'],
                        'retained_mean_R_percent': d['retained_mean_R_percent']})


if __name__ == '__main__':
    main()
