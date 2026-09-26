"""Only frozen development clean0/1; no probes, writes, test outcomes or scoring."""
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


def selected(shard):
    freeze = read_json(CODE/'headroom_freeze.json')
    if digest(CODE/'development_headroom_cohort.json') != freeze['development_cohort_sha256']:
        raise AssertionError('development text changed')
    if digest(CODE/'protocol.json') != freeze['protocol_sha256']:
        raise AssertionError('sampling protocol changed')
    rows = read_json(CODE/'development_headroom_cohort.json')
    if len(rows) != 513 or any(r['split'] != 'development' for r in rows):
        raise AssertionError('headroom stage restricted to frozen development')
    if len({r['triplet_key'] for r in rows}) != 513 or not 0 <= shard < 17:
        raise AssertionError('duplicate case or invalid shard')
    return rows[shard*32:(shard+1)*32]


def expected(rows):
    return {(r['triplet_key'], s) for r in rows for s in (0, 1)}


def check_audit(a, model):
    settings = read_json(CODE/'protocol.json')['settings'][model]
    want = dict(forward_calls={'sd3':28,'flux':8,'qwen':100}[model],
                batch_size=2 if model=='sd3' else 1,
                unconditional_executed=model!='flux', guidance_mode=settings['guidance_mode'],
                guidance_scale=settings['guidance_scale'])
    if any(a.get(k) != v for k,v in want.items()):
        raise AssertionError('generation forward audit mismatch')


def verify_images(record, rows, model, complete):
    from PIL import Image
    seen = set()
    for r in record['images']:
        k = (r['triplet_key'], r['seed'])
        if k in seen or k not in expected(rows) or r['arm'] != 'clean':
            raise AssertionError('duplicate, unexpected seed, case or arm')
        seen.add(k)
        path = ROOT/r['path']
        if digest(path) != r['sha256']:
            raise AssertionError('image hash changed')
        with Image.open(path) as im:
            if im.size != (1024,1024): raise AssertionError('image dimensions')
            im.verify()
        check_audit(r['audit'], model)
    if complete and seen != expected(rows):
        raise AssertionError('incomplete headroom shard')
    return seen


def run(model, shard):
    rows = selected(shard)
    base = ROOT/'results/dev_calibration/runtime/development_headroom'/model
    base.mkdir(parents=True, exist_ok=True)
    lock = (base/f'shard_{shard:03d}.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('shard already running')
    stage_binding = dict(code=binding(),freeze_sha256=digest(CODE/'headroom_freeze.json'),model=model,shard=shard)
    mp = base/f'shard_{shard:03d}.json'
    record = read_json(mp) if mp.exists() else dict(binding=stage_binding,images=[])
    if record['binding'] != stage_binding: raise AssertionError('resume binding changed')
    done = verify_images(record,rows,model,False)
    if done == expected(rows):
        write_json(base/f'shard_{shard:03d}_complete.json',dict(status='VERIFIED_COMPLETE',binding=stage_binding,manifest_sha256=digest(mp),n_images=len(record['images']),job_id=os.environ.get('SLURM_JOB_ID'),resumed_complete=True))
        print(json.dumps(dict(status='ALREADY_VERIFIED',model=model,shard=shard)));return
    import torch
    from adapter import FixedAdapter
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    minimum = {'qwen':70,'flux':40,'sd3':23}[model]
    if torch.cuda.get_device_properties(0).total_memory < minimum*1024**3:
        raise RuntimeError('insufficient GPU memory')
    started = time.perf_counter()
    adapter = FixedAdapter(model)
    job = os.environ['SLURM_JOB_ID']
    prov = base/'provenance'/f'run_{job}_{shard:03d}'
    prov.mkdir(parents=True,exist_ok=True)
    sources = {}
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix=='.py' and path.is_relative_to(ROOT):
                rel = str(path.relative_to(ROOT));sources[rel]=digest(path)
                dst=prov/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    pipe_source=Path(inspect.getfile(type(adapter.pipe)))
    shutil.copy2(pipe_source,prov/pipe_source.name)
    write_json(prov/'environment.json',dict(binding=stage_binding,source_sha256=sources,
        python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        model_load_seconds=time.perf_counter()-started,pipeline_sha256=digest(pipe_source)))
    for row in rows:
        todo=[s for s in (0,1) if (row['triplet_key'],s) not in done]
        if not todo: continue
        adapter.reset_cache()
        pack=adapter.encode(row['prompt_neutral'])
        for seed in todo:
            image_path=base/f"{row['triplet_key']}_clean_{seed}.png"
            if image_path.exists(): raise RuntimeError('unmanifested image; explicit recovery needed')
            t=time.perf_counter();im=adapter.generate(pack,seed);check_audit(adapter.last_audit,model)
            tmp=image_path.with_suffix('.png.partial')
            with tmp.open('wb') as f:
                im.save(f,format='PNG');f.flush();os.fsync(f.fileno())
            os.replace(tmp,image_path)
            record['images'].append(dict(triplet_key=row['triplet_key'],id=row['id'],arm='clean',seed=seed,
                path=str(image_path.relative_to(ROOT)),sha256=digest(image_path),audit=adapter.last_audit,
                seconds=time.perf_counter()-t,job_id=job))
            write_json(mp,record)
            print(json.dumps(dict(model=model,shard=shard,image_complete=len(record['images']),expected=len(rows)*2)),flush=True)
        del pack
    verify_images(record,rows,model,True)
    for name,h in sources.items():
        if digest(ROOT/name)!=h: raise AssertionError('dependency changed during run')
    write_json(base/f'shard_{shard:03d}_complete.json',dict(status='VERIFIED_COMPLETE',binding=stage_binding,
        manifest_sha256=digest(mp),n_images=len(record['images']),seconds=time.perf_counter()-started,job_id=job))
    print(json.dumps(dict(status='VERIFIED_COMPLETE',model=model,shard=shard,n_images=len(record['images']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['sd3','flux','qwen'],required=True);p.add_argument('--shard',type=int,required=True)
    a=p.parse_args();run(a.model,a.shard)
