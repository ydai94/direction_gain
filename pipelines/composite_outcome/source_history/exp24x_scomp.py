"""S_comp rescoring for routing experiments: bias (0-5) + alignment (0/1) via existing
benchmark scorers, on scomp_manifest.parquet. --root <dir>"""
import os, argparse
ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True)
a=ap.parse_args()
MAN=os.path.join(a.root,"scomp_manifest.parquet")
BIAS=os.path.join(a.root,"bias_scores.parquet")
ALIGN=os.path.join(a.root,"alignment_scores.parquet")
import pandas as pd
man=pd.read_parquet(MAN)
if not (os.path.exists(BIAS) and len(pd.read_parquet(BIAS))>=len(man)):
    os.system(f"python -u -m experiments.causal_patching.score_images --gpu 0 --single-pass "
              f"--manifest {MAN} --scores {BIAS}")
if not (os.path.exists(ALIGN) and len(pd.read_parquet(ALIGN))>=len(man)):
    os.system(f"python -u -m experiments.interp_program.rescore_alignment_interp "
              f"--manifest {MAN} --scores {ALIGN}")
print("done")

