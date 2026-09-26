"""Reconstruct historical per-image composite on fixed existing image scores."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
BASE=ROOT/'results/remote_runs/exp269_complete_20260912/mirror'
RUNTIME=BASE/'results/dev_scoring/runtime'
OUT=ROOT/'results/composite_outcome'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def composite(aligned,bias):
    if aligned is None or bias is None:return None
    assert type(aligned) is bool and type(bias) is int and 0<=bias<=5
    return float(20*int(aligned)*(5-bias))
def check_history():
    tree=ast.parse((CODE/'source_history/exp39_auto_anti_cdg.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_scomp')
    ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'historical_scomp','exec'),ns)
    for a in [False,True]:
        for b in range(6):assert abs(ns['_scomp'](a,b)-composite(a,b))<1e-12
    assert composite(False,0)==0 and composite(True,0)==100 and composite(True,5)==0
    # Composition precedes averaging: asymmetric examples expose accidental product-of-means.
    assert np.mean([composite(False,0),composite(True,5)])==0
    assert 100*.5*(1-2.5/5)==25

def corr(x,y):
    x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
    den=np.sqrt((x*x).sum(axis=-1)*(y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1),den,out=np.full_like(den,np.nan),where=den>1e-12)
def number(x):return float(x) if np.isfinite(x) else None
def interval(x):
    good=x[np.isfinite(x)];return np.quantile(good,[.025,.975]).tolist() if len(good) else None

def main():
    check_history();scores=read(RUNTIME/'scores.json');inputs=read(RUNTIME/'inputs.json')
    assert sha(RUNTIME/'scores.json')==read(RUNTIME/'analysis.json')['score_sha256']
    assert scores['binding']['inputs_sha256']==sha(RUNTIME/'inputs.json')
    index={(r['model'],r['triplet_key'],r['arm'],r['seed'],r['metric']):r for r in scores['rows']}
    assert len(index)==len(scores['rows'])==24084
    for r in scores['rows']:
        assert r['status']=='SCORED'
        raw=r['attempts'][-1]['raw'].strip()
        if r['metric']=='bias':assert raw in list('012345') and int(raw)==r['value']
        else:assert raw.lower() in ['true','false'] and (raw.lower()=='true')==r['value']
    sources={str(p.relative_to(ROOT)):sha(p) for p in [RUNTIME/'scores.json',RUNTIME/'inputs.json',CODE/'source_history/exp39_auto_anti_cdg.py',CODE/'source_history/exp156_score.py',CODE/'ANALYSIS_PLAN.md',Path(__file__)]}
    rng=np.random.default_rng(20260912);results={}
    for model in ['sd3','flux','qwen']:
        cp=BASE/'results/dev_calibration/runtime/development_g_primary'/model/'calibration.json'
        assert sha(cp)==inputs['source_hashes'][str(cp.relative_to(BASE))]
        sources[str(cp.relative_to(ROOT))]=sha(cp);cal={r['triplet_key']:r for r in read(cp)['rows']};cases=[]
        for key,coverage in inputs['coverage'][model].items():
            means={};products={};components={}
            for arm,seeds in [('clean',[3,4]),('steer',[3,4]),('random0',[3,4]),('baseline',[0,1])]:
                actual='clean' if arm=='baseline' else arm
                a=[index.get((model,key,actual,s,'alignment'),{}).get('value') for s in seeds]
                b=[index.get((model,key,actual,s,'bias'),{}).get('value') for s in seeds]
                if any(v is None for v in a+b):means[arm]=products[arm]=None;continue
                means[arm]=float(np.mean([composite(x,y) for x,y in zip(a,b)]))
                products[arm]=float(5*sum(a)*(10-sum(b)))
                components[arm]=dict(alignment=float(np.mean(a)),bias=float(np.mean(b)))
            c=dict(triplet_key=key,operator_status=coverage['status'],S=means,product_of_seed_means=products,components=components)
            c['R']=means['steer']-means['clean'] if means['steer'] is not None and means['clean'] is not None else None
            c['headroom']=100-means['baseline'] if means['baseline'] is not None else None
            cases.append(c)
        assert len(cases)==513
        valid=[c for c in cases if all(c['S'][a] is not None for a in ['clean','steer','random0'])];n=len(valid);assert n==498
        draws=rng.integers(0,n,size=(10000,n));summary={}
        for convention in ['S','product_of_seed_means']:
            x=np.array([[c[convention][a] for a in ['clean','steer','random0']] for c in valid]);block={'arm_means':dict(zip(['clean','steer','random0'],x.mean(axis=0).tolist()))}
            for label,a,b in [('steer_vs_clean',1,0),('steer_vs_random',1,2),('random_vs_clean',2,0)]:
                d=x[:,a]-x[:,b];block[label]=dict(mean=float(d.mean()),ci95=interval(d[draws].mean(axis=1)))
            summary[convention]=block
        globalproduct={a:100*np.mean([c['components'][a]['alignment'] for c in valid])*(1-np.mean([c['components'][a]['bias'] for c in valid])/5) for a in ['clean','steer','random0']}
        matched=sorted([c for c in valid if cal[c['triplet_key']]['status']=='PRIMARY_VALID' and c['headroom'] is not None],key=lambda c:c['triplet_key']);n=len(matched);idx=rng.integers(0,n,size=(10000,n));associations={}
        for convention in ['S','product_of_seed_means']:
            y=np.array([c[convention]['steer']-c[convention]['clean'] for c in matched]);yr=rankdata(y);yb=rankdata(y[idx],axis=1);res={};boots={}
            controls=np.column_stack([np.ones(n),rankdata([100-c[convention]['baseline'] for c in matched]),rankdata([c['components']['baseline']['alignment'] for c in matched])])
            ry=yr-controls@np.linalg.lstsq(controls,yr,rcond=None)[0]
            for pred in ['G','M']:
                x=np.array([cal[c['triplet_key']][pred] for c in matched]);xr=rankdata(x);boots[pred]=corr(rankdata(x[idx],axis=1),yb);rx=xr-controls@np.linalg.lstsq(controls,xr,rcond=None)[0]
                res[pred]=dict(rho=number(corr(xr,yr)),ci95=interval(boots[pred]),undefined_draws=int((~np.isfinite(boots[pred])).sum()),partial_rank_rho=number(corr(rx,ry)))
            res['G_minus_M']=dict(estimate=res['G']['rho']-res['M']['rho'] if res['G']['rho'] is not None and res['M']['rho'] is not None else None,ci95=interval(boots['G']-boots['M']))
            associations[convention]=res
        results[model]=dict(n_coverage=len(cases),n_valid=len(valid),n_predictor_cases=len(matched),n_improved=sum(c['R']>0 for c in valid),n_unchanged=sum(c['R']==0 for c in valid),n_worse=sum(c['R']<0 for c in valid),summary=summary,historical_global_product=globalproduct,associations=associations,cases=cases)
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(scope='Post-hoc development reconstruction of historical composite; no p-values or held-out inference',formula='per image S=100*aligned*(1-bias/5), average seeds then edited-minus-clean',seed=20260912,n_draws=10000,sources=sources,models=results)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for m,r in results.items():print(m,json.dumps({k:v for k,v in r.items() if k not in ['cases','historical_global_product']}))
if __name__=='__main__':main()
