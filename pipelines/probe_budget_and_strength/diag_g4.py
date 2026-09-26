import os
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
ROOT = Path(os.environ.get("DG_ROOT", ".")); RES = ROOT/'results'
OUT = RES/'probe_budget_and_strength'
a = json.loads((RES/'heldout_prediction/runtime/analysis.json').read_text())
print(f"{'model':5} {'n':>4} {'rho(mine,stored)':>17} {'r(mine,stored)':>15} "
      f"{'rho(mine,R)':>12} {'rho(stored,R)':>14} {'percell rho':>12}")
for m, dep_d, dep_t in [('sd3', 2.0, [6, 10]), ('flux', 1.0, [2, 3]), ('qwen', 2.0, [10, 16])]:
    rows = []
    for f in sorted(glob.glob(str(OUT/f'probe/{m}/shard_*.json'))):
        d = json.loads(Path(f).read_text())
        for k, v in d['cases'].items():
            if v['status'] != 'PRIMARY_VALID': continue
            for cc in v['cells']: rows.append(dict(key=k, **cc))
    my = pd.DataFrame(rows).drop_duplicates(['key', 'seed', 'timestep', 'dose'])
    dep = my[(my.dose == dep_d) & (my.timestep.isin(dep_t))]
    mine = dep.groupby('key').cosine.mean()
    st = {}
    for f in sorted(glob.glob(str(RES/f'heldout_prediction/runtime/{m}/shard_*.json'))):
        d = json.loads(Path(f).read_text())
        for k, v in d.get('cases', {}).items():
            cc = (v.get('probe') or {}).get('cells')
            if cc: st[k] = cc
    stored = pd.Series({k: float(np.mean([c['cosine'] for c in v])) for k, v in st.items()})
    R = pd.Series({c['triplet_key']: c['R'] for c in a['models'][m]['cases'] if c['status'] == 'PRIMARY_VALID'})
    keys = [k for k in mine.index if k in stored.index and k in R.index]
    A, B, y = mine[keys].to_numpy(), stored[keys].to_numpy(), R[keys].to_numpy()
    # per-cell agreement across the 6 deployed cells
    pc_mine, pc_st = [], []
    for k in keys:
        mm = {(c.seed, c.timestep): c.cosine for c in dep[dep.key == k].itertuples()}
        for cc in st[k]:
            pc_mine.append(mm[(cc['seed'], cc['timestep'])]); pc_st.append(cc['cosine'])
    print(f'{m:5} {len(keys):4d} {spearmanr(A,B).statistic:17.4f} {pearsonr(A,B)[0]:15.4f} '
          f'{spearmanr(A,y).statistic:12.4f} {spearmanr(B,y).statistic:14.4f} '
          f'{spearmanr(pc_mine,pc_st).statistic:12.4f}')
