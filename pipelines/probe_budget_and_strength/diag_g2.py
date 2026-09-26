import os
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(os.environ.get("DG_ROOT", ".")); RES = ROOT/'results'
OUT = RES/'probe_budget_and_strength'
for m, dep_d, dep_t in [('sd3', 2.0, [6, 10]), ('flux', 1.0, [2, 3]), ('qwen', 2.0, [10, 16])]:
    cal = json.loads((RES/f'dev_calibration/runtime/development_g_primary/{m}/calibration.json').read_text())
    mu, sd = cal['mu'], cal['sd']
    rows = []
    for f in sorted(glob.glob(str(OUT/f'probe/{m}/shard_*.json'))):
        d = json.loads(Path(f).read_text())
        for k, v in d['cases'].items():
            if v['status'] != 'PRIMARY_VALID': continue
            for cc in v['cells']: rows.append(dict(key=k, **cc))
    df = pd.DataFrame(rows).drop_duplicates(['key', 'seed', 'timestep', 'dose'])
    dep = df[(df.dose == dep_d) & (df.timestep.isin(dep_t))]
    ncell = dep.groupby('key').cosine.size()
    nnull = dep.cosine.isna().groupby(dep.key).sum()
    g = (dep.groupby('key').cosine.mean() - mu) / sd
    a = json.loads((RES/'heldout_prediction/runtime/analysis.json').read_text())
    pub = {c['triplet_key']: c['G'] for c in a['models'][m]['cases'] if c['status'] == 'PRIMARY_VALID'}
    keys = [k for k in g.index if k in pub]
    resid = np.array([g[k] - pub[k] for k in keys])
    bad = [(k, round(g[k], 4), round(pub[k], 4), int(ncell[k]), int(nnull.get(k, 0)))
           for k in keys if abs(g[k] - pub[k]) > 1e-6]
    print(f'{m}: mu={mu:.6g} sd={sd:.6g} n={len(keys)} cells/case={sorted(set(ncell))} '
          f'nulls={int(nnull.sum())} exact={int((np.abs(resid) <= 1e-6).sum())} bad={len(bad)} '
          f'max|resid|={np.abs(resid).max():.4g}')
    for row in bad[:6]:
        print('   key', row[0][:10], 'mine', row[1], 'pub', row[2], 'ncells', row[3], 'nulls', row[4])
