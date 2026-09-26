"""Post-hoc Qwen selection utility; local CPU, no model execution."""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'results/remote_runs/exp271_qwen_complete_20260916/results/heldout_prediction/runtime/analysis.json'
frozen=ROOT/'pipelines/heldout_prediction/predictors_frozen.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
allcases=read(source)['models']['qwen']['cases']
cases=sorted([c for c in allcases if all(c['S'][a] is not None for a in ['clean','steer','random0']) and all(k in c for k in ['G','M','H','A0'])],key=lambda c:c['triplet_key'])
n=len(cases);assert n==1295
keys=[c['triplet_key'] for c in cases];y=np.array([c['R'] for c in cases])
x={name:np.array([c[name] for c in cases]) for name in ['G','M','H']}
for name in ['B','B_G']:
 f=read(frozen)['models']['qwen']['fits'][name];X=np.array([[c[k] for k in f['features']] for c in cases]);x[name]=((X-f['mean'])/f['scale'])@f['coef']+f['intercept']
fractions=[.1,.2,.25,.5,1.0];ks=[math.ceil(n*q) for q in fractions]
def selections(indices):
 # Original integer index is the outcome-independent sorted key tie break.
 return {name:indices[np.lexsort((indices,-values[indices]))] for name,values in x.items()}
point=selections(np.arange(n));boot={name:np.empty((10000,len(ks))) for name in x};random=np.empty(10000)
rng=np.random.default_rng(20260913)
for b in range(10000):
 idx=rng.integers(0,n,n);random[b]=y[idx].mean()
 for name,order in selections(idx).items():
  sums=np.cumsum(y[order]);boot[name][b]=[sums[k-1]/k for k in ks]
def ci(v):return np.quantile(v,[.025,.975]).tolist()
rows=[]
for j,(q,k) in enumerate(zip(fractions,ks)):
 strategies={}
 for name,order in point.items():
  v=float(y[order[:k]].mean());strategies[name]=dict(mean_gain=v,ci95=ci(boot[name][:,j]),lift_vs_random=v-float(y.mean()),lift_ci95=ci(boot[name][:,j]-random),selected_keys=[keys[i] for i in order[:k]])
 comparisons={}
 for label,a,b in [('G_minus_M','G','M'),('G_minus_H','G','H'),('B_G_minus_B','B_G','B')]:
  comparisons[label]=dict(difference=strategies[a]['mean_gain']-strategies[b]['mean_gain'],ci95=ci(boot[a][:,j]-boot[b][:,j]))
 rows.append(dict(fraction=q,n_selected=k,strategies=strategies,comparisons=comparisons))
for name in x:assert np.allclose(boot[name][:,-1],random)
result=dict(scope='Post-hoc exploratory Qwen held-out selection; pointwise intervals, no selection of best cutoff',n=n,seed=20260913,bootstrap=10000,source_sha256=sha(source),predictors_sha256=sha(frozen),script_sha256=sha(Path(__file__)),random_expected_mean=float(y.mean()),random_expected_ci95=ci(random),rows=rows)
out=ROOT/'results/selection_qwen/results.json';out.write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
 print(json.dumps(dict(fraction=r['fraction'],n=r['n_selected'],strategies={k:{a:b for a,b in v.items() if a!='selected_keys'} for k,v in r['strategies'].items()},comparisons=r['comparisons'])))
