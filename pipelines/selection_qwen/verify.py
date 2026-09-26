"""Independent stdlib point-selection reconstruction; no bootstrap replication."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2]
a=json.loads((root/'results/remote_runs/exp271_qwen_complete_20260916/results/heldout_prediction/runtime/analysis.json').read_text())['models']['qwen']['cases']
d=json.loads((root/'results/selection_qwen/results.json').read_text())
cases=[c for c in a if c['R'] is not None and c['S']['random0'] is not None and all(k in c for k in ['G','M','H','A0'])]
assert len(cases)==d['n']==1295
fits=json.loads((root/'pipelines/heldout_prediction/predictors_frozen.json').read_text())['models']['qwen']['fits']
for row in d['rows']:
 for name,r in row['strategies'].items():
  def prediction(c):
   if name in ['G','M','H']:return c[name]
   f=fits[name]
   return f['intercept']+sum((c[k]-mu)/sd*coef for k,mu,sd,coef in zip(f['features'],f['mean'],f['scale'],f['coef']))
  chosen=sorted(cases,key=lambda c:(-prediction(c),c['triplet_key']))[:row['n_selected']]
  assert [c['triplet_key'] for c in chosen]==r['selected_keys']
  assert math.isclose(sum(c['S']['steer']-c['S']['clean'] for c in chosen)/len(chosen),r['mean_gain'],abs_tol=1e-10)
print('Verified all25 selected sets and gain estimates with independent stdlib sorting and prediction')
