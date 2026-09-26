"""Measured, fixed top-20% audit. No fitting, model calls or cutoff search."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MODELS = ['sd3', 'flux', 'qwen']
NAMES = ['ALL', 'G', 'M', 'H', 'B', 'B_G', 'B_M', 'B_G_M']
METRICS = ['R', 'bias_reduction', 'clean_alignment', 'steer_alignment',
           'alignment_change', 'clean_bias', 'steer_bias', 'content_pass',
           'joint_success', 'retained_R', 'retained_bias_reduction']
CONTRASTS = [('G_minus_M', 'G', 'M'), ('G_minus_H', 'G', 'H'),
             ('G_minus_ALL', 'G', 'ALL'), ('B_G_minus_B', 'B_G', 'B'),
             ('full_minus_B_M', 'B_G_M', 'B_M'), ('B_G_minus_B_M', 'B_G', 'B_M')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def scores(cases, fits):
    x = {n: np.array([c[n] for c in cases]) for n in ['G', 'M', 'H']}
    for name, f in fits.items():
        features = np.array([[c[k] for k in f['features']] for c in cases])
        x[name] = ((features-np.array(f['mean']))/np.array(f['scale'])) @ np.array(f['coef']) + f['intercept']
    assert all(np.isfinite(v).all() for v in x.values())
    return x


def order(indices, values):
    return indices[np.lexsort((indices, -values[indices]))]


def endpoints(cases):
    r = np.array([c['R'] for c in cases], dtype=float)
    ac, ast, bc, bs = [np.array([c['components'][a][v] for c in cases])
                       for a, v in [('clean', 'alignment'), ('steer', 'alignment'),
                                    ('clean', 'bias'), ('steer', 'bias')]]
    assert set(ac) <= {0, .5, 1} and set(ast) <= {0, .5, 1}
    assert np.allclose(r, [c['S']['steer']-c['S']['clean'] for c in cases])
    assert ((bc >= 0) & (bc <= 5) & (bs >= 0) & (bs <= 5)).all()
    d = bc-bs
    passed = ast == 1
    return np.column_stack([r, d, ac, ast, ast-ac, bc, bs, passed, passed & (d > 0)])


def means(values, indices):
    a = values[indices]
    base = a.mean(axis=0)
    keep = a[:, 7] == 1
    conditional = a[keep, :2].mean(axis=0) if keep.any() else [np.nan, np.nan]
    return np.r_[base, conditional]


def finite(value):
    return float(value) if np.isfinite(value) else None


def summarize(point, boot):
    keep = np.isfinite(boot)
    return dict(estimate=finite(point), ci95=np.quantile(boot[keep], [.025, .975]).tolist()
                if keep.any() else None, undefined_draws=int((~keep).sum()))


def analyze_model(model, data, fits, draws):
    allcases = data['cases']
    cases = sorted([c for c in allcases if c['status'] == 'PRIMARY_VALID'
                    and all(c['S'][a] is not None for a in ['clean', 'steer', 'random0'])
                    and all(k in c for k in ['G', 'M', 'H', 'A0'])], key=lambda c: c['triplet_key'])
    n = len(cases)
    assert n == 1295, (model, n)
    keys = [c['triplet_key'] for c in cases]
    assert len(set(keys)) == n
    k = math.ceil(.2*n)
    x, y = scores(cases, fits), endpoints(cases)
    indices = np.arange(n)
    selected = {'ALL': indices, **{s: order(indices, v)[:k] for s, v in x.items()}}
    for values in x.values():
        assert np.allclose(means(y, order(indices, values)), means(y, indices), equal_nan=True)
    point = {s: means(y, selected[s]) for s in NAMES}
    boot = {s: np.empty((draws, len(METRICS))) for s in NAMES}
    rng = np.random.default_rng(20260913)
    for b in range(draws):
        ix = rng.integers(0, n, n)
        boot['ALL'][b] = means(y, ix)
        for s, v in x.items():
            boot[s][b] = means(y, order(ix, v)[:k])
    results = {}
    for s in NAMES:
        ix = selected[s]
        results[s] = dict(n_selected=len(ix), n_content_pass=int(y[ix, 7].sum()),
                          selected_keys=[keys[i] for i in ix],
                          metrics={metric: summarize(point[s][j], boot[s][:, j]) for j, metric in enumerate(METRICS)})
    comparisons = {label: {metric: summarize(point[a][j]-point[b][j], boot[a][:, j]-boot[b][:, j])
                          for j, metric in enumerate(METRICS)} for label, a, b in CONTRASTS}
    return dict(n_coverage=len(allcases), n_valid=n, n_excluded=len(allcases)-n,
                case_keys_sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
                strategies=results, comparisons=comparisons,
                frozen_prediction=data['frozen_prediction'], G_M=data['G_M'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--predictors', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    source, predictors = read(args.source), read(args.predictors)
    assert source['predictors_sha256'] == sha(args.predictors)
    result = dict(simulated=False, scope='Exploratory measured reanalysis of historically exposed Exp271 cases',
                  fraction=.2, seed=20260913, bootstrap=10000, content_threshold=1.0,
                  confidence='Pointwise 95% percentile intervals; not multiplicity-adjusted',
                  provenance=dict(source=str(args.source), source_sha256=sha(args.source),
                                  predictors=str(args.predictors), predictors_sha256=sha(args.predictors),
                                  script_sha256=sha(__file__), plan_sha256=sha(Path(__file__).with_name('ANALYSIS_PLAN.md'))),
                  models={})
    for model in MODELS:
        result['models'][model] = analyze_model(model, source['models'][model], predictors['models'][model]['fits'], 10000)
        print(json.dumps(dict(model=model, status='ANALYZED', G_minus_M=result['models'][model]['comparisons']['G_minus_M'])), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
