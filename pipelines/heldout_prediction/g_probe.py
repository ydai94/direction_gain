"""Primary pair response and random nulls; unrelated semantic controls deferred."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
from common import CODE, ROOT, binding, digest, read_json, write_json
from headroom import selected, check_audit


def base(model):
    return ROOT/'results/dev_calibration/runtime/development_g_primary'/model


def safe_cosine(a,b):
    import torch
    x,y=a.float().flatten(),b.float().flatten()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('nonfinite response')
    nx,ny=float(x.norm()),float(y.norm())
    if not nx or not ny:return None
    value=float((x@y)/(nx*ny))
    if not -1.00001<=value<=1.00001:raise AssertionError('cosine out of range')
    return max(-1.,min(1.,value))


def probe(adapter,row,p):
    import torch
    from base_model import norm_matched,NotWritable
    cells=[];adapter.trajectory_audits=[]
    for seed in [6,7,8]:
        snapshots=adapter.trajectory(p['neutral'],seed)
        for t in adapter.spec['timesteps']:
            z=snapshots[t]
            def response(pack,reference=False):return adapter.velocity(z,pack,p['neutral'],reference)
            v0=response(p['neutral']);ref=response(p['anti'],True)-response(p['stereo'],True);delta=response(p['edited'])-v0
            c=safe_cosine(delta,ref);mag=float(delta.norm());rn=float(ref.norm())
            cell=dict(seed=seed,timestep=t,cosine=c,magnitude=mag,reference_norm=rn,
                primary_status='VALID' if c is not None else ('ZERO_PRIMARY_REFERENCE' if rn==0 else 'ZERO_PRIMARY_RESPONSE'),random_write_cosines=[],random_reference_cosines=[])
            for v in p['nulls']:
                try:value=safe_cosine(response(adapter.write(p['neutral'],v,p['anchors']))-v0,ref)
                except NotWritable:value=None
                cell['random_write_cosines'].append(value)
            for k in range(4):
                random_ref=norm_matched(ref.detach().cpu(),f"reference:{adapter.model}:{row['id']}:{seed}:{t}:{k}").to(delta.device)
                cell['random_reference_cosines'].append(safe_cosine(delta,random_ref));del random_ref
            cells.append(cell);del v0,ref,delta
        del snapshots
    for a in adapter.trajectory_audits:
        if adapter.model=='qwen':
            from engineering import check_audit as qwen_check
            qwen_check(a,'trajectory')
        else:check_audit(a,adapter.model)
    return cells


def verify_case(r,model):
    if r['status']=='NOT_WRITABLE':
        if not r.get('reason'):raise AssertionError('coverage reason missing')
        return
    import math
    if r['status'] not in ['PRIMARY_VALID','PRIMARY_UNDEFINED']:raise AssertionError('unknown status')
    indices={'sd3':[6,10],'flux':[2,3],'qwen':[10,16]}[model]
    if len(r['cells'])!=6 or {(c['seed'],c['timestep']) for c in r['cells']}!={(s,t) for s in [6,7,8] for t in indices}:raise AssertionError('probe rectangle')
    valid=True
    for c in r['cells']:
        for k in ['magnitude','reference_norm']:
            if not math.isfinite(c[k]) or c[k]<0:raise AssertionError('invalid norm')
        if c['cosine'] is None:valid=False
        for k in ['random_write_cosines','random_reference_cosines']:
            if len(c[k])!=4:raise AssertionError('control count')
        for v in [c['cosine']]+c['random_write_cosines']+c['random_reference_cosines']:
            if v is not None and (not math.isfinite(v) or abs(v)>1):raise AssertionError('invalid cosine')
    if valid!=(r['status']=='PRIMARY_VALID'):raise AssertionError('validity mismatch')


def run(model,shard):
    rows=selected(shard);out=base(model);out.mkdir(parents=True,exist_ok=True)
    lock=(out/f'shard_{shard:03d}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bound=dict(code=binding(),model=model,shard=shard,scope='primary_G_magnitude_random_controls; unrelated_donors_not_measured')
    mp=out/f'shard_{shard:03d}.json';record=read_json(mp) if mp.exists() else dict(binding=bound,rows=[])
    if record['binding']!=bound:raise AssertionError('resume binding')
    done={r['triplet_key']:r for r in record['rows']}
    if len(done)!=len(record['rows']) or not set(done)<={r['triplet_key'] for r in rows}:raise AssertionError('duplicate/unexpected rows')
    for r in done.values():verify_case(r,model)
    import torch
    from adapter import FixedAdapter
    from base_model import NotWritable
    if torch.cuda.get_device_properties(0).total_memory<{'qwen':70,'flux':40,'sd3':23}[model]*1024**3:raise RuntimeError('GPU capacity')
    start=time.perf_counter();adapter=FixedAdapter(model);job=os.environ['SLURM_JOB_ID']
    prov=out/'provenance'/f'run_{job}_{shard:03d}';prov.mkdir(parents=True,exist_ok=True);sources={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_file() and path.suffix=='.py' and path.is_relative_to(ROOT):
                rel=str(path.relative_to(ROOT));sources[rel]=digest(path);dst=prov/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    pipe=Path(inspect.getfile(type(adapter.pipe)));shutil.copy2(pipe,prov/pipe.name)
    write_json(prov/'environment.json',dict(binding=bound,source_sha256=sources,python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,pipeline_sha256=digest(pipe)))
    for row in rows:
        if row['triplet_key'] in done:continue
        adapter.reset_cache();t=time.perf_counter();r=dict(id=row['id'],triplet_key=row['triplet_key'],job_id=job)
        try:
            p=adapter.prepare(row);cells=probe(adapter,row,p)
            r.update(status='PRIMARY_VALID' if all(c['cosine'] is not None for c in cells) else 'PRIMARY_UNDEFINED',cells=cells,trajectory_audits=adapter.trajectory_audits,
                **{n:p[n] for n in ['anchors','input_direction_norm','executed_write_norm','T_rel','pair_separability']});del p
        except NotWritable as e:r.update(status='NOT_WRITABLE',reason=str(e))
        r['seconds']=time.perf_counter()-t;verify_case(r,model);record['rows'].append(r);write_json(mp,record)
        print(json.dumps(dict(model=model,shard=shard,n_cases=len(record['rows']),status=r['status'])),flush=True)
    if {r['triplet_key'] for r in record['rows']}!={r['triplet_key'] for r in rows}:raise AssertionError('incomplete shard')
    for n,h in sources.items():
        if digest(ROOT/n)!=h:raise AssertionError('source changed')
    write_json(out/f'shard_{shard:03d}_complete.json',dict(status='VERIFIED_COMPLETE',binding=bound,manifest_sha256=digest(mp),n_cases=len(rows),seconds=time.perf_counter()-start,job_id=job))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['sd3','flux','qwen'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run(a.model,a.shard)
