"""Build data/per_prompt/{test,dev,meta}_<model>.* from the held-out run outputs (usage: python extract_per_prompt.py MODEL)."""

import json, glob, csv, os, sys, hashlib
R=os.environ.get("DG_ROOT", ".")
EXP=R+"/results/heldout_prediction/runtime"
REL=R+"/pipelines/heldout_prediction"
OUT=os.path.join(R,"data","per_prompt"); os.makedirs(OUT, exist_ok=True)
model=sys.argv[1]
null=json.load(open(REL+"/development_calibration.json"))
an=json.load(open(EXP+"/analysis.json"))["models"][model]

rows=json.load(open("%s/%s/scoring/scores.json"%(EXP,model)))["rows"]
by={}
for r in rows: by[(r["triplet_key"],r["arm"],r["seed"],r["metric"])]=r["value"]

cells={}
for f in sorted(glob.glob("%s/%s/shard_*.json"%(EXP,model))):
    if f.endswith("_complete.json"): continue
    d=json.load(open(f))
    for k,cse in d["cases"].items():
        p=cse["probe"]
        cells[k]=[(x["seed"],x["timestep"],x["cosine"],x["magnitude"],x["primary_status"]) for x in p.get("cells",[])]

def U(k,arm,s):
    a=by.get((k,arm,s,"alignment")); b=by.get((k,arm,s,"bias"))
    if a is None or b is None: return None
    return 20*int(a)*(5-b)

recs=[]
for cse in an["cases"]:
    k=cse["triplet_key"]
    rec={"triplet_key":k,"status":cse["status"]}
    for fld in ["G","M","H","A0","R","input_direction_norm","pair_separability","T_rel"]:
        rec[fld]=cse.get(fld)
    for arm,seeds in [("clean",[3,4]),("steer",[3,4]),("random0",[3,4])]:
        for s in seeds: rec["U_%s_%d"%(arm,s)]=U(k,arm,s)
    for s in [0,1]: rec["U_base_%d"%s]=U(k,"clean",s)
    for (sd,ts,cos,mag,st) in sorted(cells.get(k,[])):
        rec["cos_s%d_t%d"%(sd,ts)]=cos; rec["mag_s%d_t%d"%(sd,ts)]=mag
        if st!="VALID": rec["cell_flag"]=st
    recs.append(rec)
keys=[]
for r_ in recs:
    for k_ in r_:
        if k_ not in keys: keys.append(k_)
p=OUT+"/test_%s.csv"%model
with open(p,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=keys); w.writeheader()
    for r_ in recs: w.writerow(r_)

# development rows (same construction as freeze_predictors.py)
comp=json.load(open(R+"/results/composite_outcome/results.json"))
cp="%s/results/dev_calibration/runtime/development_g_primary/%s/calibration.json"%(R,model)
cal={r["triplet_key"]:r for r in json.load(open(cp))["rows"]}
dev=[]
for cse in comp["models"][model]["cases"]:
    k=cse["triplet_key"]
    if cse["R"] is None or cal[k]["status"]!="PRIMARY_VALID": continue
    c0=cal[k]
    dev.append({"triplet_key":k,"G":c0["G"],"M":c0["M"],
                "input_direction_norm":c0["input_direction_norm"],
                "pair_separability":c0["pair_separability"],"T_rel":c0["T_rel"],
                "H":cse["headroom"],"A0":cse["components"]["baseline"]["alignment"],"R":cse["R"]})
dk=list(dev[0].keys())
pd_=OUT+"/dev_%s.csv"%model
with open(pd_,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=dk); w.writeheader()
    for r_ in dev: w.writerow(r_)
meta={"model":model,"n_test":len(recs),"n_dev":len(dev),"null":null[model],
      "cal_keys":sorted(cal[list(cal)[0]].keys()),
      "sha_scores":hashlib.sha256(open("%s/%s/scoring/scores.json"%(EXP,model),"rb").read()).hexdigest()[:16],
      "sha_analysis":hashlib.sha256(open(EXP+"/analysis.json","rb").read()).hexdigest()[:16]}
json.dump(meta, open(OUT+"/meta_%s.json"%model,"w"), indent=1)
print(json.dumps(meta))
