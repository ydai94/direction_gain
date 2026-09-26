"""Held-out composite response and frozen-predictor evaluation, CPU only."""
import numpy as np
from scipy.stats import rankdata
from common import CODE,ROOT,read_json,write_json,digest
from run import base,verify

def corr(x,y):
    x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
    den=np.sqrt((x*x).sum(axis=-1)*(y*y).sum(axis=-1));return np.divide((x*y).sum(axis=-1),den,out=np.full_like(den,np.nan),where=den>1e-12)
def ci(x):
    x=x[np.isfinite(x)];return np.quantile(x,[.025,.975]).tolist() if len(x) else None
def val(x):return float(x) if np.isfinite(x) else None

def main():
    rng=np.random.default_rng(20260912);results={};frozen=read_json(CODE/'predictors_frozen.json');null=read_json(CODE/'development_calibration.json')
    for model in ['sd3','flux','qwen']:
        out=base(model);sp=out/'scoring/scores.json';scores=read_json(sp);inputs=read_json(out/'scoring/inputs.json');done=read_json(out/'scoring/score_complete.json')
        assert digest(sp)==done['score_sha256'] and scores['binding']['inputs_sha256']==digest(out/'scoring/inputs.json')
        by={(r['triplet_key'],r['arm'],r['seed'],r['metric']):r for r in scores['rows']};assert len(by)==len(scores['rows'])==len(inputs['rows'])*2
        from score_engine import parse
        for imrow in inputs['rows']:
            im=imrow['image']
            for metric in ['bias','alignment']:
                r=by[im['triplet_key'],im['arm'],im['seed'],metric];assert r['image_sha256']==im['sha256']
                if r['status']=='SCORED':assert parse(r['attempts'][-1]['raw'],metric)==r['value']
                else:assert r['status']=='FORMAT_FAILURE' and r['value'] is None
        cases=[];count=83 if model=='qwen' else 42
        for shard in range(count):
            record=verify(model,shard,False)
            for k,c in record['cases'].items():
                means={};parts={}
                for arm,seeds in [('clean',[3,4]),('steer',[3,4]),('random0',[3,4]),('baseline',[0,1])]:
                    aa='clean' if arm=='baseline' else arm
                    a=[by.get((k,aa,s,'alignment'),{}).get('value') for s in seeds];b=[by.get((k,aa,s,'bias'),{}).get('value') for s in seeds]
                    if any(v is None for v in a+b):means[arm]=None;parts[arm]=None;continue
                    means[arm]=float(np.mean([20*int(x)*(5-y) for x,y in zip(a,b)]));parts[arm]=dict(alignment=float(np.mean(a)),bias=float(np.mean(b)))
                r=dict(triplet_key=k,S=means,components=parts,R=means['steer']-means['clean'] if means['steer'] is not None and means['clean'] is not None else None,status=c['probe']['status'])
                p=c['probe']
                if p['status']=='PRIMARY_VALID':
                    cosine=float(np.mean([x['cosine'] for x in p['cells']]));r.update(G=(cosine-null[model]['mu'])/null[model]['sd'],M=float(np.mean([x['magnitude'] for x in p['cells']])),**{a:p[a] for a in ['input_direction_norm','pair_separability','T_rel']})
                if means['baseline'] is not None:r.update(H=100-means['baseline'],A0=parts['baseline']['alignment'])
                cases.append(r)
        assert len(cases)==len({r['triplet_key'] for r in cases})==1314
        valid=[r for r in cases if all(r['S'][a] is not None for a in ['clean','steer','random0'])];summary={};n=len(valid)
        if n:
            draw=rng.integers(0,n,size=(10000,n));summary['arm_means']={a:float(np.mean([r['S'][a] for r in valid])) for a in ['clean','steer','random0']}
            for name,a,b in [('steer_vs_clean','steer','clean'),('steer_vs_random','steer','random0')]:
                d=np.array([r['S'][a]-r['S'][b] for r in valid]);summary[name]=dict(mean=float(d.mean()),ci95=ci(d[draw].mean(axis=1)))
        matched=[r for r in valid if all(k in r for k in ['G','M','H','A0'])];n=len(matched);metrics={};increment={}
        if n>=3:
            draw=rng.integers(0,n,size=(10000,n));y=np.array([r['R'] for r in matched]);yr=rankdata(y);yb=rankdata(y[draw],axis=1);boot={}
            for name in ['G','M']:
                x=np.array([r[name] for r in matched]);boot[name]=corr(rankdata(x[draw],axis=1),yb);metrics[name]=dict(rho=val(corr(rankdata(x),yr)),ci95=ci(boot[name]),undefined_draws=int((~np.isfinite(boot[name])).sum()))
            metrics['G_minus_M']=dict(ci95=ci(boot['G']-boot['M']))
            errors={}
            for name,f in frozen['models'][model]['fits'].items():
                X=np.array([[r[k] for k in f['features']] for r in matched]);pred=((X-np.array(f['mean']))/np.array(f['scale']))@np.array(f['coef'])+f['intercept'];errors[name]=(y-pred)**2
                increment[name]=dict(MSE=float(errors[name].mean()))
            for label,a,b in [('B_G_vs_B','B','B_G'),('full_vs_B_M','B_M','B_G_M'),('full_vs_B_G','B_G','B_G_M')]:
                d=errors[a]-errors[b];increment[label]=dict(MSE_reduction=float(d.mean()),ci95=ci(d[draw].mean(axis=1)))
        results[model]=dict(n_coverage=len(cases),n_paired=len(valid),n_prediction=n,summary=summary,G_M=metrics,frozen_prediction=increment,cases=cases,score_sha256=digest(sp))
    write_json(ROOT/'results/heldout_prediction/runtime/analysis.json',dict(scope='Historically exposed test cases held out from new development fits; no unseen-axis or quality claim',seed=20260912,n_resample=10000,primary='Per-image historical composite improvement',predictors_sha256=digest(CODE/'predictors_frozen.json'),models=results))
if __name__=='__main__':main()
