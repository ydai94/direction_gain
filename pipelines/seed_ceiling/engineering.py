"""Four fixed development cases; full eight-image Qwen integration, no efficacy gate."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from common import CODE, ROOT, binding, digest, read_json, write_json


def specs():
    return [('clean',0),('clean',1)]+[(a,s) for a in ('clean','steer','random0') for s in (3,4)]


def check_audit(a,kind):
    expected=100 if kind=='generate' else 34
    if (a['forward_calls']!=expected or a['batch_size']!=1 or not a['unconditional_executed']
        or a['guidance_scale']!=4 or a['guidance_mode']!='normalized_cfg'
        or a['snapshot_convention']!='actual_pre_forward'):
        raise AssertionError('Qwen engineering audit mismatch')


def verify(out):
    frozen=read_json(out/'engineering_G_frozen.json')
    if frozen['code_binding']!=binding():raise AssertionError('source binding changed')
    rows=read_json(CODE/'engineering_cohort.json')[:4]
    from PIL import Image
    n=0
    for row in rows:
        d=out/row['triplet_key']; probe=read_json(d/'probe.json')
        if digest(d/'probe.json')!=frozen['probe_hashes'][row['triplet_key']]:
            raise AssertionError('probe changed')
        if len(probe['cells'])!=6:raise AssertionError('missing probe cells')
        for a in probe['trajectory_audits']:check_audit(a,'trajectory')
        record=read_json(d/'images.json')
        if record['code_binding']!=binding() or record['freeze_sha256']!=digest(out/'engineering_G_frozen.json'):
            raise AssertionError('image provenance mismatch')
        if {(r['arm'],r['seed']) for r in record['images']}!=set(specs()):raise AssertionError('missing images')
        for r in record['images']:
            p=ROOT/r['path']
            if digest(p)!=r['sha256']:raise AssertionError('image changed')
            with Image.open(p) as im:
                if im.size!=(1024,1024):raise AssertionError('image dimensions')
                im.verify()
            check_audit(r['audit'],'generate'); n+=1
    write_json(out/'verification.json',dict(status='PASS_EXECUTION_ONLY',n_images=n,n_cells=24,
                                           code_binding=binding(),behavioral_efficacy_not_tested=True))


def run(out):
    import torch
    from adapter import FixedAdapter
    started=time.perf_counter(); rows=read_json(CODE/'engineering_cohort.json'); by={r['id']:r for r in rows}
    if len(rows)!=32 or any(r['split']!='development' for r in rows):raise AssertionError('engineering cohort changed')
    if torch.cuda.get_device_properties(0).total_memory < 70*1024**3:raise RuntimeError('Qwen requires verified 80GB card')
    model=FixedAdapter('qwen'); out.mkdir(parents=True,exist_ok=True)
    dependencies={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_file() and p.suffix=='.py' and p.is_relative_to(ROOT):
                dependencies[str(p.relative_to(ROOT))]=digest(p)
    snapshot=out/'provenance'/('run_'+os.environ['SLURM_JOB_ID']); snapshot.mkdir(parents=True,exist_ok=True)
    for name in dependencies:
        dst=snapshot/'code'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    pipe_source=Path(inspect.getfile(type(model.pipe)));shutil.copy2(pipe_source,snapshot/pipe_source.name)
    write_json(snapshot/'environment.json',dict(source_sha256=dependencies,code_binding=binding(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        pipeline_source_sha256=digest(pipe_source),model_revision_source='exp243_gen.build source snapshot'))
    probes=[]
    for row in rows[:4]:
        dest=out/row['triplet_key']; path=dest/'probe.json'
        if path.exists():
            rec=read_json(path)
            if rec['code_binding']!=binding():raise AssertionError('probe resume binding mismatch')
        else:
            model.reset_cache(); model.trajectory_audits=[]; start=time.perf_counter()
            p=model.prepare(row); torch.cuda.reset_peak_memory_stats()
            cells=model.probe(row,p,[by[i] for i in row['reference_ids']])
            for a in model.trajectory_audits:check_audit(a,'trajectory')
            rec=dict(id=row['id'],triplet_key=row['triplet_key'],cells=cells,
                trajectory_audits=model.trajectory_audits,code_binding=binding(),seconds=time.perf_counter()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            write_json(path,rec);del p,cells
        probes.append(rec);print(json.dumps(dict(stage='probe_complete',id=row['id'])),flush=True)
    null=np.array([[c['random_write_cosines'] for c in r['cells']] for r in probes]).mean(axis=1).reshape(-1)
    mu=float(null.mean()); sd=float(null.std(ddof=1))
    if not np.isfinite(null).all() or not sd>0:raise AssertionError('degenerate engineering null')
    frozen=dict(scope='four_case_development_engineering_only',code_binding=binding(),mu=mu,sd=sd,
        probe_hashes={r['triplet_key']:digest(out/r['triplet_key']/'probe.json') for r in rows[:4]},
        G={r['triplet_key']:(float(np.mean([c['cosine'] for c in r['cells']]))-mu)/sd for r in probes})
    fp=out/'engineering_G_frozen.json'
    if fp.exists() and read_json(fp)!=frozen:raise AssertionError('frozen engineering score changed')
    if not fp.exists():write_json(fp,frozen)
    for row in rows[:4]:
        model.reset_cache(); p=model.prepare(row); dest=out/row['triplet_key']; mp=dest/'images.json'
        records=read_json(mp) if mp.exists() else dict(code_binding=binding(),freeze_sha256=digest(fp),images=[])
        if records['code_binding']!=binding() or records['freeze_sha256']!=digest(fp):raise AssertionError('resume binding')
        packs={'clean':p['neutral'],'steer':p['edited'],'random0':model.write(p['neutral'],p['nulls'][0],p['anchors'])}
        done={(r['arm'],r['seed']):r for r in records['images']}
        for arm,seed in specs():
            image_path=dest/f'{arm}_{seed}.png'
            if (arm,seed) in done:
                if digest(image_path)!=done[arm,seed]['sha256']:raise AssertionError('corrupt completed image')
                continue
            if image_path.exists():raise RuntimeError('unmanifested image requires recovery')
            start=time.perf_counter();image=model.generate(packs[arm],seed)
            partial=image_path.with_suffix('.png.partial');image.save(partial,format='PNG');os.replace(partial,image_path)
            records['images'].append(dict(arm=arm,seed=seed,path=str(image_path.relative_to(ROOT)),
                sha256=digest(image_path),audit=model.last_audit,seconds=time.perf_counter()-start))
            write_json(mp,records)
            print(json.dumps(dict(stage='image_complete',id=row['id'],arm=arm,seed=seed)),flush=True)
        del p,packs
    for name,h in dependencies.items():
        if digest(ROOT/name)!=h:raise AssertionError('dependency changed during run')
    verify(out)
    write_json(snapshot/'complete.json',dict(status='COMPLETE',seconds=time.perf_counter()-started,n_images=32,n_cells=24))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','verify']);a=ap.parse_args()
    output=ROOT/'results/dev_calibration/runtime/qwen_engineering'
    (run if a.action=='run' else verify)(output)
