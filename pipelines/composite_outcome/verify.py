"""Check exact composite arithmetic and independently reproduce point ranks."""
import json,statistics
from pathlib import Path
from analyze import ROOT,BASE,RUNTIME,check_history
check_history()
r=json.loads((ROOT/'results/composite_outcome/results.json').read_text())
scores=json.loads((RUNTIME/'scores.json').read_text())['rows']
by={(s['model'],s['triplet_key'],s['arm'],s['seed'],s['metric']):s['value'] for s in scores}
def rank(v):
    values=sorted(set(v));mapping={x:sum(i+1 for i,y in enumerate(sorted(v)) if x==y)/v.count(x) for x in values}
    return [mapping[x] for x in v]
for model,block in r['models'].items():
    p=BASE/'results/dev_calibration/runtime/development_g_primary'/model/'calibration.json'
    cal={x['triplet_key']:x for x in json.loads(p.read_text())['rows']}
    valid=[]
    for c in block['cases']:
        if c['R'] is None:continue
        k=c['triplet_key'];vals={}
        for arm in ['clean','steer','random0']:
            ss=[(100-20*by[model,k,arm,s,'bias']) if by[model,k,arm,s,'alignment'] else 0 for s in [3,4]]
            vals[arm]=sum(ss)/2;assert vals[arm]==c['S'][arm]
        assert vals['steer']-vals['clean']==c['R'];valid.append(c)
    assert len(valid)==498
    for pred in ['G','M']:
        observed=statistics.correlation(rank([cal[c['triplet_key']][pred] for c in valid]),rank([c['R'] for c in valid]))
        assert abs(observed-block['associations']['S'][pred]['rho'])<1e-12
print('Verified all composite case scores and six primary G/M rank correlations; historical formula equivalence passes')
