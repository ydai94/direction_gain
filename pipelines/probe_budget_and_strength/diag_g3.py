import os
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(os.environ.get("DG_ROOT", ".")); RES = ROOT/'results'
OUT = RES/'probe_budget_and_strength'
for m, dep_d, dep_t in [('sd3', 2.0, [6, 10]), ('flux', 1.0, [2, 3]), ('qwen', 2.0, [10, 16])]:
    rows = []
    for f in sorted(glob.glob(str(OUT/f'probe/{m}/shard_*.json'))):
        d = json.loads(Path(f).read_text())
        for k, v in d['cases'].items():
            if v['status'] != 'PRIMARY_VALID': continue
            for cc in v['cells']: rows.append(dict(key=k, **cc))
    mycells = pd.DataFrame(rows).drop_duplicates(['key', 'seed', 'timestep', 'dose'])
    dep = mycells[(mycells.dose == dep_d) & (mycells.timestep.isin(dep_t))]
    mine = dep.groupby('key').cosine.mean()
    st = {}
    for f in sorted(glob.glob(str(RES/f'heldout_prediction/runtime/{m}/shard_*.json'))):
        d = json.loads(Path(f).read_text())
        for k, v in d.get('cases', {}).items():
            cc = (v.get('probe') or {}).get('cells')
            if cc: st[k] = cc
    stored = pd.Series({k: float(np.mean([c['cosine'] for c in v])) for k, v in st.items()})
    a = json.loads((RES/'heldout_prediction/runtime/analysis.json').read_text())
    cal = json.loads((RES/f'dev_calibration/runtime/development_g_primary/{m}/calibration.json').read_text())
    pubmean = pd.Series({c['triplet_key']: c['G']*cal['sd'] + cal['mu']
                         for c in a['models'][m]['cases'] if c['status'] == 'PRIMARY_VALID'})
    keys = [k for k in mine.index if k in stored.index and k in pubmean.index]
    A, B, C = mine[keys].to_numpy(), stored[keys].to_numpy(), pubmean[keys].to_numpy()
    print(f'{m}: n={len(keys)}  mine==stored: {int((np.abs(A-B)<1e-9).sum())}  '
          f'stored==pub: {int((np.abs(B-C)<1e-6).sum())}  mine==pub: {int((np.abs(A-C)<1e-6).sum())}  '
          f'max|mine-stored|={np.abs(A-B).max():.4g}  max|stored-pub|={np.abs(B-C).max():.4g}')
    bad = [k for k in keys if abs(mine[k]-stored[k]) > 1e-9]
    if bad:
        k = bad[0]
        mm = sorted([(c.seed, c.timestep, round(c.cosine, 8)) for c in dep[dep.key == k].itertuples()])
        tt = sorted([(c['seed'], c['timestep'], round(c['cosine'], 8)) for c in st[k]])
        print('   first divergent case', k[:12])
        print('     mine  ', mm)
        print('     stored', tt)
        print('   n divergent:', len(bad))
