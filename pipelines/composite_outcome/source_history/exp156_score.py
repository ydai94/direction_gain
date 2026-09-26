import os,sys,glob,re
import pandas as pd
BASE=os.environ.get("DG_ROOT", ".")
sys.path.insert(0,os.path.join(BASE,"experiments"))
ROOT=os.path.join(BASE,"results/exp156_fullwidth"); DEC=os.path.join(ROOT,"inject")
MODELP="Qwen/Qwen3-VL-30B-A3B-Instruct"
bench=pd.read_csv(os.path.join(BASE,"data/benchmark_prompts.csv")); bench["cid12"]=bench["id"].astype(str).str[:12]
brow={r["cid12"]:r for _,r in bench.iterrows()}
from experiments.causal_patching.score_images import load_qwen_vl, judge_single as bias_judge
import experiments.evaluate_alignment_local as EAL
EAL.MODEL_PATH=MODELP
model,proc=load_qwen_vl(); rows=[]
for p in sorted(glob.glob(os.path.join(DEC,"*.png"))):
    m=re.match(r"(.+?)__([a-z0-9_]+)__s(\d+)",os.path.basename(p)[:-4])
    if not m: continue
    cid,arm,seed=m.group(1),m.group(2),int(m.group(3)); b=brow.get(cid)
    if b is None: continue
    bj=bias_judge(model,proc,p,b["target"],b["bias_type"],str(b["head"]),b["relation"],b["stereotype_tails"],b["anti_stereotype_tails"])
    bias=bj[0] if isinstance(bj,(tuple,list)) else bj
    aj=EAL.judge_single(model,proc,p,b["prompt_neutral"])
    rows.append(dict(cid=cid,arm=arm,seed=seed,bias=int(bias),aligned=bool(aj[0] if isinstance(aj,(tuple,list)) else aj)))
    print(f"[e156s] {cid[:8]} {arm} s{seed} bias={bias}",flush=True)
df=pd.DataFrame(rows); df.to_parquet(os.path.join(ROOT,"scores.parquet"),index=False)
g=df.groupby("arm").agg(bias_mean=("bias","mean"),aligned_rate=("aligned","mean"),n=("bias","size"))
g["S_comp"]=g.aligned_rate*(1-g.bias_mean/5)*100
print("=== EXP156 Qwen FULL-WIDTH + shared axis vector (L0-12) ===")
print(g.round(3).to_string())
print("\n=== per-pair (bias; primary read) ===")
print(df.pivot_table(index="cid",columns="arm",values="bias",aggfunc="mean").round(2).to_string())
