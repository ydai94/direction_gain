"""Reproduce Table 1 from archived ratings; no new model or judge execution."""
from pathlib import Path
from collections import Counter
from statistics import mean
import hashlib,json
ROOT=Path(__file__).resolve().parent
ARMS=('clean','random0','steer')
SEEDS=(3,4)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 expected=json.loads((ROOT/'inputs/cohort_keys.json').read_text())
 qwen=json.loads((ROOT/'inputs/qwen_summary.json').read_text())
 output={'models':{},'sources':{p.name:digest(p) for p in (ROOT/'inputs').iterdir() if p.is_file()}}
 for model in ('sd3','flux','qwen'):
  keys=set(expected[model]);assert len(keys)==1295
  records={}; skipped=Counter();duplicates=0
  with (ROOT/f'inputs/scores_{model}.jsonl').open() as stream:
   for line in stream:
    r=json.loads(line)
    if r['seed'] not in SEEDS:skipped['non_outcome_seed']+=1;continue
    assert r['triplet_key'] in keys
    assert r['judge']=='gpt-5.6-terra'
    assert r['arm'] in ARMS and r['metric'] in ('bias','alignment')
    v=r['value'];assert (type(v) is int and 0<=v<=5) if r['metric']=='bias' else type(v) is bool
    ident=(r['triplet_key'],r['arm'],r['seed'],r['metric'])
    if ident in records:
     assert (v,r['image_sha256'])==records[ident];duplicates+=1
    records[ident]=(v,r['image_sha256'])
  assert len(records)==1295*3*2*2
  summary={}
  for arm in ARMS:
   values=[]
   for key in sorted(keys):
    for seed in SEEDS:
     b,bhash=records[(key,arm,seed,'bias')];a,ahash=records[(key,arm,seed,'alignment')]
     assert bhash==ahash
     values.append((b,100*int(a),20*(5-b) if a else 0))
   summary[arm]=dict(zip(('bias','matching_percent','combined'),(mean(x[i] for x in values) for i in range(3))))
   summary[arm]['n_images']=len(values)
  output['models'][model]={'GPT-5.6':summary,'Qwen3-VL':qwen[model], 'audit':{'n_cases':len(keys),'outcome_ratings':len(records),'identical_duplicates':duplicates,'excluded':dict(skipped)},'gpt_steer_minus_clean':summary['steer']['combined']-summary['clean']['combined']}
 (ROOT/'results.json').write_text(json.dumps(output,indent=2)+'\n')
 for m,d in output['models'].items():print(m,json.dumps(d))
if __name__=='__main__':main()
