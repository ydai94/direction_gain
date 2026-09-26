"""exp243 generation — four registered arms on the frozen held-out cohort.
Machinery verbatim exp235_dose.py (which is verbatim exp230_ladder.py): same build(),
diff_rows, anchor search, injection. Differences ONLY:
  - cases = cohort_frozen.parquet (240 fresh prompts); sign = +1 by construction (registered:
    zero overlap with signs_shard0; behavioural sign table NOT consulted);
  - dose = deployment alpha x 1 (NO escalation);
  - arms per prompt x 2 seeds: clean | steer (one steered set serves BOTH steer-all and
    G-gated/random-gated arms at analysis time — gating decides which image is USED, so we
    generate clean+steer for every prompt and never generate per-gate duplicates);
  - random-gate assignment: per model, a registered permutation (seed 20260818) of the frozen
    G-gate coverage — written to disk here, BEFORE scoring.
--model qwen|sd3|flux [--shard k --nshards n]
"""
import os, sys, re, argparse
import numpy as np, pandas as pd, torch
BASE=os.environ.get("DG_ROOT", ".")
sys.path.insert(0,BASE)
ROOT=os.path.join(BASE,"results/exp243_prospective")
SEEDS=[0,1]
BASE_ALPHA={"qwen":2.0,"sd3":2.0,"flux":1.0}
GENERIC=["person","man","woman","people","girl","boy","child","worker"]

def diff_rows(pa,pb):
    L=min(pa.shape[0],pb.shape[0])
    d=(pa[:L]-pb[:L]).norm(dim=-1).float().cpu().numpy()
    thr=max(1e-3,0.05*float(d.max()))
    return [int(i) for i in np.where(d>thr)[0]]
def emb(pack): return (pack[0] if isinstance(pack,tuple) else pack)

def build(model):
    if model=="qwen":
        from experiments.causal_patching.run_three_methods import load_pipe
        pipe=load_pipe(); torch.set_grad_enabled(False)
        def enc(pr):
            pe,pm=pipe.encode_prompt(prompt=pr,device=pipe._execution_device); return pe,pm
        def gen(pe,pm,sd):
            return pipe(prompt_embeds=pe,prompt_embeds_mask=pm,num_inference_steps=50,
                        true_cfg_scale=4.0,negative_prompt=" ",
                        generator=torch.Generator("cuda").manual_seed(sd)).images[0]
    elif model=="sd3":
        from diffusers import StableDiffusion3Pipeline
        MODEL=os.path.join(os.environ.get("MODEL_ROOT", "models"), "stable-diffusion-3.5-medium")
        pipe=StableDiffusion3Pipeline.from_pretrained(MODEL,torch_dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True); torch.set_grad_enabled(False)
        def enc(pr):
            pe,ne,pp,np_=pipe.encode_prompt(prompt=pr,prompt_2=None,prompt_3=None,device="cuda",
                                            negative_prompt="",do_classifier_free_guidance=True)
            return (pe,ne,pp,np_),None
        def gen(pack,_,sd):
            pe,ne,pp,np_=pack
            return pipe(prompt_embeds=pe,negative_prompt_embeds=ne,pooled_prompt_embeds=pp,
                        negative_pooled_prompt_embeds=np_,num_inference_steps=28,guidance_scale=7.0,
                        generator=torch.Generator("cuda").manual_seed(sd)).images[0]
    else:
        from experiments.interp_flux.flux_concept_control import _load_flux as load_flux
        pipe=load_flux(); torch.set_grad_enabled(False)
        def enc(pr):
            e=pipe.encode_prompt(prompt=pr,device="cuda")
            return (e[0] if isinstance(e,tuple) else e),None
        def gen(pe,_,sd):
            return pipe(prompt_embeds=emb(pe).to(torch.bfloat16),num_inference_steps=8,
                        generator=torch.Generator("cuda").manual_seed(sd)).images[0]
    return pipe,enc,gen

def main(model,shard,nshards):
    DEC=os.path.join(ROOT,model,"png"); os.makedirs(DEC,exist_ok=True)
    cases=pd.read_parquet(os.path.join(ROOT,"cohort_frozen.parquet"))
    # write the registered random-gate ONCE (rank-0 shard only, before any scoring exists)
    gates=pd.read_csv(os.path.join(ROOT,"gates_frozen.csv"))
    gm=gates[gates.model==model].set_index("cid12")
    rgp=os.path.join(ROOT,f"randomgate_{model}.csv")
    if shard==0 and not os.path.exists(rgp):
        rng=np.random.default_rng(20260818)
        ids=gm.index.to_numpy(); k=int(gm.gate.sum())
        pick=set(rng.choice(ids,size=k,replace=False))
        pd.DataFrame(dict(cid12=list(ids),rand_gate=[int(i in pick) for i in ids])).to_csv(rgp,index=False)
        print(f"[e243-{model}] random gate written: {k}/{len(ids)}",flush=True)
    pipe,enc,gen=build(model)
    rows=[]
    todo=[(n_,r) for n_,(_,r) in enumerate(cases.iterrows()) if n_%nshards==shard]
    for n_,r in todo:
        cid=str(r.cid12)
        try:
            pA,_=enc(str(r.prompt_stereotype)); pB,_=enc(str(r.prompt_anti_stereotype))
        except Exception as e:
            print(f"[e243-{model}] ENC FAIL {cid}: {e}",flush=True); continue
        eA=emb(pA)[0]; eB=emb(pB)[0]
        drs=diff_rows(eA,eB)
        if not drs: continue
        v=torch.stack([((eB[k]-eA[k])).float().cpu() for k in drs[:2]]).mean(0)  # sign +1 registered
        neutral=str(r.prompt_neutral)
        cands=[]
        for k in ("head","target"):
            vv=r.get(k)
            if isinstance(vv,str): cands+=[w for w in re.findall(r"[A-Za-z]+",vv) if len(w)>2][:2]
        cands+=GENERIC
        arows=[]
        for w in cands:
            if not re.search(rf"\b{w}\b",neutral,flags=re.I): continue
            p1,_=enc(neutral); p2,_=enc(re.sub(rf"\b{w}\b","thing",neutral,count=1,flags=re.I))
            arows=diff_rows(emb(p1)[0],emb(p2)[0])
            if arows: break
        if not arows:
            print(f"[e243-{model}] SKIP no-anchor {cid[:8]}",flush=True); continue
        packN,pmN=enc(neutral); peN=emb(packN)
        dev,dt=peN.device,peN.dtype
        al=BASE_ALPHA[model]
        peM=peN.clone()
        for cc in arows[:2]:
            if cc<peM.shape[1]: peM[0,cc]=peM[0,cc]+al*v.to(dev).to(dt)
        packM=(peM,)+packN[1:] if isinstance(packN,tuple) else peM
        for arm,pack,pm in [("clean",packN,pmN),("steer",packM,pmN)]:
            for sd in SEEDS:
                p=os.path.join(DEC,f"{cid}__{arm}__s{sd}.png")
                rows.append(dict(cid12=cid,arm=arm,seed=sd,image_path=p,
                                 prompt_stereotype=r.prompt_stereotype,
                                 prompt_anti_stereotype=r.prompt_anti_stereotype))
                if os.path.exists(p): continue
                gen(pack,pm,sd).save(p)
        print(f"[e243-{model}] {n_+1}/240 {cid[:8]}",flush=True)
        if len(rows)%40==0:
            pd.DataFrame(rows).to_parquet(os.path.join(ROOT,model,f"manifest_shard{shard}.parquet"),index=False)
    pd.DataFrame(rows).to_parquet(os.path.join(ROOT,model,f"manifest_shard{shard}.parquet"),index=False)
    print(f"[e243-{model}] done {len(rows)} rows",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True)
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
    a=ap.parse_args(); main(a.model,a.shard,a.nshards)

