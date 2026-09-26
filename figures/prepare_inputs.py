"""Copy existing experimental summaries, without fitting or resampling."""
import argparse,hashlib,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('workspace',type=Path);args=parser.parse_args()
root=args.workspace;out=Path(__file__).resolve().parent/'inputs';out.mkdir(exist_ok=True)
files={'sd3':'SD3_VERIFIED_20260913.json','flux':'FLUX_VERIFIED_20260914.json','qwen':'QWEN_VERIFIED_20260916.json'}
folders={'sd3':'exp273_selection','flux':'exp273_flux_selection','qwen':'selection_qwen'}
provenance={};data={'models':{},'note':'Archived measured outputs, copied without changing estimates or interval bounds.'}
def read(rel):
 p=root/rel;assert p.stat().st_size<2_000_000
 provenance[str(rel)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 return json.loads(p.read_text())
sel=read(Path('results/frozen_selection/results.json'))
for model,file in files.items():
 d=read(Path('results/heldout_prediction')/file)
 curve=read(Path('results')/folders[model]/'results.json')
 rows=[]
 for row in curve['rows']:
  rows.append(dict(fraction=row['fraction'],n_selected=row['n_selected'],strategies={k:{a:b for a,b in v.items() if a!='selected_keys'} for k,v in row['strategies'].items()},comparisons=row['comparisons']))
 data['models'][model]={'n':d['n_prediction'],'correlation':d['G_M'],'summary':d['summary'],'components':d['components'],'prediction':d['frozen_prediction'],'selection':{k:sel['models'][model]['strategies'][k]['metrics'] for k in ['G','M','ALL']},'selection_comparisons':sel['models'][model]['comparisons'],'curves':rows,'all_prompt_mean':curve['random_expected_mean'],'selection_source_seed':curve['seed'],'component_source_seed':sel['seed']}
 assert d['n_prediction']==curve['n']==1295
 top20=next(r for r in rows if r['fraction']==.2)
 for name in ['G','M']:
  assert abs(top20['strategies'][name]['mean_gain']-data['models'][model]['selection'][name]['R']['estimate'])<1e-10
(out/'plotted_values.json').write_text(json.dumps(data,indent=2)+'\n')
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
