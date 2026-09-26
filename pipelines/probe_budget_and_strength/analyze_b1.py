"""Exp280 B1 analysis: does the probe design matter?

Three variants, all evaluated against the same outcome R (Exp271 two-seed composite,
Qwen3-VL judge) on the frozen 300-case subset:

  timestep      rho(G, R) for G built from each single probe timestep, and from the
                deployed pair, at the deployed steering strength
  reference     cos(dv, anti - stereotype)  vs  cos(dv, anti - neutral)
  statistic     cosine (deployed) vs projection (cosine x magnitude) vs magnitude alone

G is the mean cosine over the cells entering it; the published G additionally applies a
development affine calibration, which is monotone and so leaves every Spearman unchanged.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(os.environ.get("DG_ROOT", "."))
RES = ROOT / 'results'
OUT = RES / 'probe_budget_and_strength'
MODELS = ['sd3', 'flux', 'qwen']
NBOOT = int(os.environ.get('NBOOT', 5000))
SEED = 20260921
PLAN = json.loads((ROOT / 'pipelines/probe_budget_and_strength/plan.json').read_text())
rng = np.random.default_rng(SEED)


def spear(x, y):
    x = rankdata(x, axis=-1).astype(float)
    y = rankdata(y, axis=-1).astype(float)
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    return np.divide((x * y).sum(-1), den, out=np.full_like(den, np.nan), where=den > 1e-12)


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def load_cells(model):
    rows = []
    for f in sorted(glob.glob(str(OUT / 'probe' / model / 'shard_*.json'))):
        d = json.loads(Path(f).read_text())
        for k, v in d['cases'].items():
            if v['status'] != 'PRIMARY_VALID':
                continue
            for cc in v['cells']:
                rows.append(dict(key=k, **cc))
    df = pd.DataFrame(rows)
    assert not df.duplicated(['key', 'seed', 'timestep', 'dose']).any()
    return df


def outcomes(model):
    a = json.loads((RES / 'heldout_prediction/runtime/analysis.json').read_text())
    return pd.DataFrame([dict(key=c['triplet_key'], R=c['R'], G_pub=c['G'], M_pub=c['M'])
                         for c in a['models'][model]['cases'] if c['status'] == 'PRIMARY_VALID'])


def agg(df, col, mask=None):
    d = df if mask is None else df[mask]
    return d.groupby('key')[col].mean()


rows, ref_rows, stat_rows, repro_rows = [], [], [], []
for m in MODELS:
    cells = load_cells(m)
    out = outcomes(m)
    dep_d = PLAN[m]['deployed_dose']
    dep_t = PLAN[m]['deployed_timesteps']
    at_dep_dose = cells.dose == dep_d
    keys = sorted(cells.key.unique())
    base = out[out.key.isin(keys)].set_index('key').loc[keys]
    R = base.R.to_numpy(float)
    n = len(keys)
    draws = rng.integers(0, n, size=(NBOOT, n))

    def series(s):
        return s.reindex(keys).to_numpy(float)

    # --- reference: deployed pair vs anti-minus-neutral, on the deployed cells ---
    dep_mask = at_dep_dose & cells.timestep.isin(dep_t)
    g_pair = series(agg(cells, 'cosine', dep_mask))
    g_anti = series(agg(cells, 'cosine_anti_neutral', dep_mask))
    mag = series(agg(cells, 'magnitude', dep_mask))
    # Published G is the mean of these same six cosines under a monotone affine map, but a
    # cell recomputed while the surrounding probe grid differs is not bit-identical: only Qwen
    # (whose timesteps were left at the deployed pair) reproduces exactly. Record the
    # cross-run agreement and keep every B1 contrast inside this run.
    g_pub = series(base.G_pub)
    repro_rows.append(dict(model=m, n=n,
                           rho_G_this_run_vs_published=float(spear(g_pair, g_pub)),
                           rho_G_R_this_run=float(spear(g_pair, R)),
                           rho_G_R_published=float(spear(g_pub, R)),
                           timesteps_probed=len(sorted(cells.timestep[at_dep_dose].unique())),
                           identical_timesteps=sorted(cells.timestep[at_dep_dose].unique()) == sorted(dep_t)))
    d_ref = [float(spear(g_pair[i], R[i]) - spear(g_anti[i], R[i])) for i in draws]
    ref_rows.append(dict(model=m, n=n,
                         rho_pair=float(spear(g_pair, R)), rho_anti_neutral=float(spear(g_anti, R)),
                         diff=float(spear(g_pair, R) - spear(g_anti, R)), diff_ci95=ci(d_ref),
                         rho_between=float(spear(g_pair, g_anti))))

    # --- statistic: cosine vs projection vs magnitude ---
    proj = series(agg(cells.assign(p=cells.cosine * cells.magnitude), 'p', dep_mask))
    for name, v in [('cosine', g_pair), ('projection', proj), ('magnitude', mag)]:
        d = [float(spear(v[i], R[i]) - spear(g_pair[i], R[i])) for i in draws]
        stat_rows.append(dict(model=m, statistic=name, n=n, rho=float(spear(v, R)),
                              diff_vs_cosine=float(spear(v, R) - spear(g_pair, R)),
                              diff_ci95=ci(d)))

    # --- timestep: each single step, and the deployed pair ---
    for t in sorted(cells.timestep[at_dep_dose].unique()):
        v = series(agg(cells, 'cosine', at_dep_dose & (cells.timestep == t)))
        d = [float(spear(v[i], R[i]) - spear(g_pair[i], R[i])) for i in draws]
        rows.append(dict(model=m, probe=f'step {t}', n=n, rho=float(spear(v, R)),
                         deployed=bool(t in dep_t), diff_vs_deployed=float(spear(v, R) - spear(g_pair, R)),
                         diff_ci95=ci(d)))
    rows.append(dict(model=m, probe='deployed pair', n=n, rho=float(spear(g_pair, R)),
                     deployed=True, diff_vs_deployed=0.0, diff_ci95=[0.0, 0.0]))
    # all six timesteps pooled
    v = series(agg(cells, 'cosine', at_dep_dose))
    d = [float(spear(v[i], R[i]) - spear(g_pair[i], R[i])) for i in draws]
    rows.append(dict(model=m, probe='all steps pooled', n=n, rho=float(spear(v, R)), deployed=False,
                     diff_vs_deployed=float(spear(v, R) - spear(g_pair, R)), diff_ci95=ci(d)))
    print(f'[{m}] n={n} cells={len(cells)} rho_deployed={spear(g_pair, R):.4f}', flush=True)

rp = pd.DataFrame(repro_rows)
rp.to_csv(OUT / 'b1_reproducibility.csv', index=False)
ts = pd.DataFrame(rows)
rf = pd.DataFrame(ref_rows)
st = pd.DataFrame(stat_rows)
OUT.mkdir(parents=True, exist_ok=True)
ts.to_csv(OUT / 'b1_timesteps.csv', index=False)
rf.to_csv(OUT / 'b1_reference.csv', index=False)
st.to_csv(OUT / 'b1_statistic.csv', index=False)
print()
print(ts.to_string(index=False))
print()
print(rf.to_string(index=False))
print()
print(st.to_string(index=False))
print()
print(rp.to_string(index=False))
