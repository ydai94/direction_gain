"""Freeze development-only ridge predictors for continuous composite improvement."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[1]
p=ROOT/'results/composite_outcome/results.json';comp=json.loads(p.read_text());fits={}
for model in ['sd3','flux','qwen']:
    cp=ROOT/'results/dev_calibration/runtime/development_g_primary'/model/'calibration.json'
    cal={r['triplet_key']:r for r in json.loads(cp.read_text())['rows']};cases=[c for c in comp['models'][model]['cases'] if c['R'] is not None and cal[c['triplet_key']]['status']=='PRIMARY_VALID']
    rows=[dict(cal[c['triplet_key']],H=c['headroom'],A0=c['components']['baseline']['alignment'],R=c['R']) for c in cases];B=['input_direction_norm','pair_separability','T_rel','H','A0'];saved={}
    for name,features in [('B',B),('B_G',B+['G']),('B_M',B+['M']),('B_G_M',B+['G','M'])]:
        X=np.array([[r[k] for k in features] for r in rows]);y=np.array([r['R'] for r in rows]);sc=StandardScaler();Z=sc.fit_transform(X);fit=Ridge(alpha=1.,solver='svd').fit(Z,y)
        saved[name]=dict(features=features,mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=fit.coef_.tolist(),intercept=float(fit.intercept_))
    fits[model]=dict(n=len(rows),triplet_keys=[c['triplet_key'] for c in cases],fits=saved,calibration_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
out=CODE/'predictors_frozen.json';result=dict(scope='development-only fixed ridge alpha1; no tuning, no test labels',composite_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),models=fits)
if out.exists():assert json.loads(out.read_text())==result
else:out.write_text(json.dumps(result,indent=2)+'\n')
print('Frozen composite predictors from development only')
