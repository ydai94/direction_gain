"""Exp268: output-only revision of historical bias/alignment rubrics."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DG_ROOT', '.'))
OUT = ROOT/'results/dev_scoring/runtime'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    with t.open('w') as f:
        json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def binding():
    return {p.name:sha(p) for p in sorted(CODE.iterdir()) if p.is_file()}

def parse(raw,metric):
    s=raw.strip()
    if metric=='bias' and s in ['0','1','2','3','4','5']:return int(s)
    if metric=='alignment' and s.lower() in ['true','false']:return s.lower()=='true'
    raise ValueError('FORMAT_FAILURE')

def prepare():
    from PIL import Image
    rows=read(CODE/'cohort.json');by={r['triplet_key']:r for r in rows}
    todo=[];hashes={};coverage={}
    for model in read(CODE/'judge_config.json')['models']:
        for stage in ['development_headroom','development_outcomes']:
            base=ROOT/'results/dev_calibration/runtime'/stage/model
            for shard in range(17):
                p=base/f'shard_{shard:03d}.json';cp=base/f'shard_{shard:03d}_complete.json'
                d=read(p);c=read(cp)
                assert c['status']=='VERIFIED_COMPLETE' and c['manifest_sha256']==sha(p)
                for f in [p,cp]:hashes[str(f.relative_to(ROOT))]=sha(f)
                if stage=='development_outcomes':coverage.setdefault(model,{}).update(d['cases'])
                for im in d['images']:
                    assert im['triplet_key'] in by
                    path=ROOT/im['path'];assert path.resolve().is_relative_to(ROOT.resolve())
                    assert sha(path)==im['sha256']
                    with Image.open(path) as image:
                        assert image.size==(1024,1024);image.verify()
                    todo.append(dict(model=model,stage=stage,image=im,case=by[im['triplet_key']]))
    keys=[(r['model'],r['image']['triplet_key'],r['image']['arm'],r['image']['seed']) for r in todo]
    assert len(keys)==len(set(keys))
    for model,cases in coverage.items():
        assert set(cases)==set(by)
        for k,c in cases.items():
            expected={('clean',0),('clean',1)}|{(a,s) for a in c['arms'] for s in [3,4]}
            actual={(r['image']['arm'],r['image']['seed']) for r in todo if r['model']==model and r['image']['triplet_key']==k}
            assert expected==actual,(model,k,expected,actual)
    todo.sort(key=lambda r:hashlib.sha256(('20260911:'+r['model']+r['image']['path']).encode()).hexdigest())
    write(OUT/'inputs.json',dict(code=binding(),source_hashes=hashes,coverage=coverage,rows=todo))
    print('VERIFIED',len(todo),'images',flush=True)

def score():
    import torch
    from transformers import AutoProcessor,Qwen3VLMoeForConditionalGeneration
    cfg=read(CODE/'judge_config.json');prompts=read(CODE/'prompts.json');inputs=read(OUT/'inputs.json')
    assert inputs['code']==binding()
    for p,h in inputs['source_hashes'].items():assert sha(ROOT/p)==h
    assert torch.cuda.get_device_properties(0).total_memory>=70*1024**3
    lock=(OUT/'score.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'scores.json';bind=dict(code=binding(),inputs_sha256=sha(OUT/'inputs.json'))
    result=read(dest) if dest.exists() else dict(binding=bind,rows=[])
    assert result['binding']==bind
    cached={r['key'] for r in result['rows']};assert len(cached)==len(result['rows'])
    prov=OUT/'provenance'/('run_'+os.environ['SLURM_JOB_ID']);prov.mkdir(parents=True,exist_ok=True)
    for p in CODE.iterdir():
        if p.is_file():shutil.copy2(p,prov/p.name)
    torch.manual_seed(cfg['seed']);torch.set_grad_enabled(False)
    t0=time.perf_counter()
    model=Qwen3VLMoeForConditionalGeneration.from_pretrained(cfg['judge_path'],dtype=torch.bfloat16,device_map='auto').eval()
    processor=AutoProcessor.from_pretrained(cfg['judge_path'])
    write(prov/'environment.json',dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),judge_config=model.config.to_dict(),code=binding(),load_seconds=time.perf_counter()-t0))
    for row in inputs['rows']:
        im=row['image'];path=ROOT/im['path'];assert sha(path)==im['sha256']
        for metric in ['bias','alignment']:
            key='|'.join(map(str,[row['model'],im['triplet_key'],im['arm'],im['seed'],metric]))
            if key in cached:continue
            question=prompts[metric]['compact'].format(**row['case']);attempts=[];value=None
            for attempt in range(cfg['attempts']):
                messages=[{'role':'user','content':[{'type':'image','image':str(path)},{'type':'text','text':question}]}]
                t=time.perf_counter()
                batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
                generated=model.generate(**batch,max_new_tokens=cfg['max_new_tokens'],do_sample=False)
                trimmed=generated[:,batch.input_ids.shape[1]:]
                raw=processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                attempts.append(dict(raw=raw,output_tokens=int(trimmed.shape[1]),seconds=time.perf_counter()-t))
                del batch,generated,trimmed
                try:value=parse(raw,metric);break
                except ValueError:question=prompts[metric]['compact'].format(**row['case'])+' Output only the requested value.'
            result['rows'].append(dict(key=key,model=row['model'],triplet_key=im['triplet_key'],arm=im['arm'],seed=im['seed'],metric=metric,value=value,status='SCORED' if value is not None else 'FORMAT_FAILURE',image_sha256=im['sha256'],attempts=attempts))
            write(dest,result)
            if len(result['rows'])%32==0:print('scores',len(result['rows']),flush=True)
    assert binding()==bind['code']
    expected=len(inputs['rows'])*2;assert len(result['rows'])==expected
    write(prov/'complete.json',dict(status='COMPLETE',n_scores=expected,score_sha256=sha(dest),seconds=time.perf_counter()-t0))

def analyze():
    import numpy as np
    inp=read(OUT/'inputs.json');scores=read(OUT/'scores.json')
    assert scores['binding']==dict(code=binding(),inputs_sha256=sha(OUT/'inputs.json'))
    index={r['key']:r for r in scores['rows']};assert len(index)==len(scores['rows'])==2*len(inp['rows'])
    for imrow in inp['rows']:
        im=imrow['image']
        for metric in ['bias','alignment']:
            key='|'.join(map(str,[imrow['model'],im['triplet_key'],im['arm'],im['seed'],metric]));r=index[key]
            assert r['image_sha256']==im['sha256']
            if r['status']=='SCORED':assert parse(r['attempts'][-1]['raw'],metric)==r['value']
            else:
                assert r['status']=='FORMAT_FAILURE' and r['value'] is None and len(r['attempts'])==2
                for a in r['attempts']:
                    try:parse(a['raw'],metric)
                    except ValueError:continue
                    raise AssertionError('false format failure')
    by={(r['model'],r['triplet_key'],r['arm'],r['seed'],r['metric']):r['value'] for r in scores['rows']}
    def mean(model,k,arm,seeds,metric):
        v=[by.get((model,k,arm,s,metric)) for s in seeds]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    result=dict(provenance='measured compact-rubric development on existing Exp267 development images',simulated=False,score_sha256=sha(OUT/'scores.json'),models={},note='Descriptive development study; no p-values, no held-out claim; bias reduction is not anti-target attainment.')
    rng=np.random.default_rng(20260911)
    for model in read(CODE/'judge_config.json')['models']:
        cases=[]
        for k in inp['coverage'][model]:
            v={a:{metric:mean(model,k,a,[3,4],metric) for metric in ['bias','alignment']} for a in ['clean','steer','random0']}
            clean_bias=mean(model,k,'clean',[0,1],'bias')
            cases.append(dict(triplet_key=k,means=v,bias_headroom=clean_bias,alignment_baseline=mean(model,k,'clean',[0,1],'alignment'),bias_reduction={a:v['clean']['bias']-v[a]['bias'] if v['clean']['bias'] is not None and v[a]['bias'] is not None else None for a in ['steer','random0']}))
        summary={}
        for metric in ['bias','alignment']:
            valid=[c for c in cases if all(c['means'][a][metric] is not None for a in ['clean','steer','random0'])]
            n=len(valid);entry=dict(n_common_valid=n)
            if n:
                x=np.array([[c['means'][a][metric] for a in ['clean','steer','random0']] for c in valid]);draw=rng.integers(0,n,size=(10000,n))
                entry['arm_means']=dict(zip(['clean','steer','random0'],x.mean(axis=0).tolist()))
                sign=-1 if metric=='bias' else 1
                for label,a,b in [('steer_vs_clean',1,0),('random_vs_clean',2,0),('steer_vs_random',1,2)]:
                    delta=sign*(x[:,a]-x[:,b]);entry[label]=dict(mean=float(delta.mean()),ci95=np.quantile(delta[draw].mean(axis=1),[.025,.975]).tolist())
            summary[metric]=entry
        rows=[r for r in scores['rows'] if r['model']==model]
        result['models'][model]=dict(cases=cases,summary=summary,n_scores=len(rows),format_failures=sum(r['status']!='SCORED' for r in rows),total_decode_seconds=sum(a['seconds'] for r in rows for a in r['attempts']),total_output_tokens=sum(a['output_tokens'] for r in rows for a in r['attempts']))
    write(OUT/'analysis.json',result);print(json.dumps({m:v['summary'] for m,v in result['models'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score','analyze']);a=p.parse_args();globals()[a.stage]()
