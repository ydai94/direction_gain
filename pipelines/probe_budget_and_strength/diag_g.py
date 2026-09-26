import os
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(os.environ.get("DG_ROOT", ".")); RES = ROOT/'results'
OUT = RES/'probe_budget_and_strength'
rows = []
for f in sorted(glob.glob(str(OUT/'probe/sd3/shard_*.json'))):
    d = json.loads(Path(f).read_text())
    for k, v in d['cases'].items():
        if v['status'] != 'PRIMARY_VALID': continue
        for cc in v['cells']: rows.append(dict(key=k, **cc))
df = pd.DataFrame(rows)
dep = df[(df.dose == 2.0) & (df.timestep.isin([6, 10]))]
g = dep.groupby('key').cosine.mean()
a = json.loads((RES/'heldout_prediction/runtime/analysis.json').read_text())
pub = {c['triplet_key']: c['G'] for c in a['models']['sd3']['cases'] if c['status'] == 'PRIMARY_VALID'}
keys = [k for k in g.index if k in pub]
x = g.loc[keys].to_numpy(); y = np.array([pub[k] for k in keys])
print('n', len(keys), 'spearman(mine,pub)', round(float(spearmanr(x, y).statistic), 6))
print('pearson', round(float(np.corrcoef(x, y)[0, 1]), 6))
A = np.polyfit(x, y, 1); print('affine fit', np.round(A, 4), 'resid max', round(float(np.abs(np.polyval(A, x)-y).max()), 6))
st = {}
for f in sorted(glob.glob(str(RES/'heldout_prediction/runtime/sd3/shard_*.json'))):
    d = json.loads(Path(f).read_text())
    for k, v in d.get('cases', {}).items():
        cc = (v.get('probe') or {}).get('cells')
        if cc: st[k] = cc
ov = [k for k in keys if k in st][:3]
print('stored-cell overlap', len([k for k in keys if k in st]))
for k in ov:
    mine = sorted([(c.seed, c.timestep, round(c.cosine, 6)) for c in dep[dep.key == k].itertuples()])
    theirs = sorted([(c['seed'], c['timestep'], round(c['cosine'], 6)) for c in st[k]])
    print(k[:10], 'mine  ', mine)
    print(' ' * 10, 'theirs', theirs)
    print(' ' * 10, 'mean mine', round(np.mean([t[2] for t in mine]), 6),
          'mean theirs', round(np.mean([t[2] for t in theirs]), 6), 'pub G', round(pub[k], 6))
    print(' ' * 10, 'theirs keys', sorted(st[k][0].keys()))
