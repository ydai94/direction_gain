"""Fixed held-out generation/probes and compact scoring, without test-time tuning."""
import argparse,fcntl,inspect,os,shutil,sys,time
from pathlib import Path
from common import CODE,ROOT,binding,digest,read_json,write_json
OUT=ROOT/'results/heldout_prediction/runtime'

def selected(model,shard):
    rows=read_json(CODE/'cohort.json');size=16 if model=='qwen' else 32
    assert len(rows)==1314 and all(r['split']=='test' for r in rows)
    assert len({r['triplet_key'] for r in rows})==1314
    assert 0<=shard<(len(rows)+size-1)//size
    return rows[shard*size:(shard+1)*size]

def base(model):return OUT/model

def provenance(adapter,dest,bound):
    import torch
    dest.mkdir(parents=True,exist_ok=True);sources={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_file() and p.suffix=='.py' and p.is_relative_to(ROOT):
                rel=p.relative_to(ROOT);sources[str(rel)]=digest(p);q=dest/'code'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    write_json(dest/'environment.json',dict(binding=bound,sources=sources,python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
    return sources

def verify(model,shard,check_images=True):
    from PIL import Image
    from g_probe import verify_case
    from headroom import check_audit
    rows=selected(model,shard);p=base(model)/f'shard_{shard:03d}.json';r=read_json(p)
    assert r['binding']==dict(code=binding(),model=model,shard=shard)
    assert set(r['cases'])=={x['triplet_key'] for x in rows}
    seen=set()
    for im in r['images']:
        key=(im['triplet_key'],im['arm'],im['seed']);assert key not in seen;seen.add(key)
        check_audit(im['audit'],model)
        if check_images:
            path=ROOT/im['path'];assert digest(path)==im['sha256']
            with Image.open(path) as image:assert image.size==(1024,1024);image.verify()
    expected=set()
    for k,c in r['cases'].items():
        verify_case(c['probe'],model)
        expected|={(k,'clean',s) for s in [0,1]}
        expected|={(k,a,s) for a in c['arms'] for s in [3,4]}
    assert seen==expected
    return r

def generate(model,shard):
    import torch
    from adapter import FixedAdapter
    from base_model import NotWritable
    from g_probe import probe,verify_case
    from headroom import check_audit
    assert torch.cuda.get_device_properties(0).total_memory>={'qwen':70,'flux':40,'sd3':23}[model]*1024**3
    out=base(model);out.mkdir(parents=True,exist_ok=True)
    lock=(out/f'shard_{shard:03d}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Duplicate worker: another allocation owns this shard',flush=True);return
    mp=out/f'shard_{shard:03d}.json';donefile=out/f'shard_{shard:03d}_complete.json'
    if donefile.exists():
        d=read_json(donefile);assert d['manifest_sha256']==digest(mp);verify(model,shard);return
    # Cancel only still-pending explicitly registered replicas of this exact shard.
    manifest=OUT.parent/'submissions.json'
    if manifest.exists():
        import subprocess
        for j in read_json(manifest).get('replicas',{}).get(f'{model}:{shard}',[]):
            if j==os.environ['SLURM_JOB_ID']:continue
            state=subprocess.run(['squeue','-j',j,'-h','-o','%T'],capture_output=True,text=True)
            if state.stdout.strip()=='PENDING':subprocess.run(['scancel',j],check=True)
    bound=dict(code=binding(),model=model,shard=shard);r=read_json(mp) if mp.exists() else dict(binding=bound,cases={},images=[])
    assert r['binding']==bound
    existing={(x['triplet_key'],x['arm'],x['seed']):x for x in r['images']}
    assert len(existing)==len(r['images'])
    for im in existing.values():assert digest(ROOT/im['path'])==im['sha256']
    started=time.perf_counter();adapter=FixedAdapter(model);job=os.environ['SLURM_JOB_ID']
    prov=out/'provenance'/f'run_{job}_{shard:03d}';sources=provenance(adapter,prov,bound)
    for row in selected(model,shard):
        k=row['triplet_key'];old=r['cases'].get(k)
        if old:
            expected={(k,'clean',s) for s in [0,1]}|{(k,a,s) for a in old['arms'] for s in [3,4]}
            if expected<=set(existing):continue
        adapter.reset_cache();packs={};p=None
        try:
            p=adapter.prepare(row);packs={'clean':p['neutral'],'steer':p['edited']}
            try:packs['random0']=adapter.write(p['neutral'],p['nulls'][0],p['anchors'])
            except NotWritable:pass
            if old:pr=old['probe']
            else:
                cells=probe(adapter,row,p)
                pr=dict(triplet_key=k,status='PRIMARY_VALID' if all(c['cosine'] is not None for c in cells) else 'PRIMARY_UNDEFINED',cells=cells,trajectory_audits=adapter.trajectory_audits,**{a:p[a] for a in ['input_direction_norm','pair_separability','T_rel','anchors','executed_write_norm']})
            c=dict(arms=list(packs),probe=pr)
        except NotWritable as err:
            packs={'clean':adapter.encode(row['prompt_neutral'])};c=dict(arms=[],probe=dict(triplet_key=k,status='NOT_WRITABLE',reason=str(err)))
        verify_case(c['probe'],model)
        if old:assert c==old
        r['cases'][k]=c;write_json(mp,r)
        specs=[('clean',0),('clean',1)]+[(a,s) for a in c['arms'] for s in [3,4]]
        for arm,seed in specs:
            if (k,arm,seed) in existing:continue
            path=out/f'{k}_{arm}_{seed}.png';tmp=path.with_suffix('.png.partial')
            # Recover only this shard's known unmanifested destination after interruption.
            if path.exists():path.rename(path.with_suffix(f'.orphan_{job}.png'))
            t=time.perf_counter();image=adapter.generate(packs[arm],seed);check_audit(adapter.last_audit,model)
            with tmp.open('wb') as f:image.save(f,format='PNG');f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
            im=dict(triplet_key=k,id=row['id'],arm=arm,seed=seed,path=str(path.relative_to(ROOT)),sha256=digest(path),audit=adapter.last_audit,seconds=time.perf_counter()-t,job_id=job)
            r['images'].append(im);existing[k,arm,seed]=im;write_json(mp,r)
        del p,packs
        print(model,shard,'completedcase',k,flush=True)
    verify(model,shard)
    for name,h in sources.items():assert digest(ROOT/name)==h
    write_json(donefile,dict(status='VERIFIED_COMPLETE',manifest_sha256=digest(mp),n_images=len(r['images']),n_cases=len(r['cases']),seconds=time.perf_counter()-started,job_id=job))

def gate(model,shard):
    verify(model,shard)
    print('CANARY VERIFIED',model,shard)

def prepare_score(model,_):
    todo=[];hashes={};coverage={};cohort={r['triplet_key']:r for r in read_json(CODE/'cohort.json')}
    count=83 if model=='qwen' else 42
    for i in range(count):
        r=verify(model,i);p=base(model)/f'shard_{i:03d}.json';done=read_json(base(model)/f'shard_{i:03d}_complete.json');assert done['manifest_sha256']==digest(p)
        hashes[str(p.relative_to(ROOT))]=digest(p);coverage.update(r['cases'])
        todo.extend(dict(model=model,image=im,case=cohort[im['triplet_key']]) for im in r['images'])
    assert len(coverage)==1314
    import hashlib
    todo.sort(key=lambda r:hashlib.sha256(('20260911:'+model+r['image']['path']).encode()).hexdigest())
    write_json(base(model)/'scoring/inputs.json',dict(code=binding(),source_hashes=hashes,coverage={model:coverage},rows=todo))

def score(model,_):
    import score_engine
    score_engine.OUT=base(model)/'scoring';score_engine.score()
    write_json(base(model)/'scoring/score_complete.json',dict(status='COMPLETE',score_sha256=digest(base(model)/'scoring/scores.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','gate','prepare_score','score']);p.add_argument('model',choices=['sd3','flux','qwen']);p.add_argument('shard',type=int,nargs='?',default=0);a=p.parse_args();globals()[a.stage](a.model,a.shard)
