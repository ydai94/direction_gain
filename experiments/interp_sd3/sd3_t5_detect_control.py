"""SD3.5 — T5 per-LAYER detection vs control (the FAITHFUL Qwen Exp-19 replication).

Fixes the prior per-ENCODER shortcut. Focus on T5 ONLY (the semantic encoder; CLIP is auxiliary/visual
and weak for distributional bias). Within T5's 24 layers:
  DETECTION = stereo-vs-anti probe accuracy per layer (already in stereo_control/probe_stereo.parquet, 1831 set).
  CONTROL   = project the per-prompt stereo−anti direction out of the NEUTRAL prompt at T5 block L, regenerate,
              standard S_comp (VLM 0-5 bias + alignment) -> per-layer ΔS_comp.
Then compare detection-best layer vs control-best layer, and ρ(probe_acc, ΔS_comp) across layers, per category
+ pooled — exactly Qwen Exp 19 (detection L17 vs control L10, ρ(probe,ΔS_comp)=−0.14 there).

  python -m experiments.interp_sd3.sd3_t5_detect_control --mode gen --shard i --nshards N
  python -m experiments.interp_sd3.sd3_t5_detect_control --mode score
  python -m experiments.interp_sd3.sd3_t5_detect_control --mode finalize
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd, torch
from experiments.interp_sd3.sd3_concept_control import _load_sd3, _layer_reps, _get_blocks, _ProjHook, ENC_CFG

ROOT = os.path.join(os.path.dirname(__file__), "t5_detect_control")
PNG = os.path.join(ROOT, "png")
MANIFEST = os.path.join(ROOT, "manifest.parquet")
BIAS = os.path.join(ROOT, "bias_scores.parquet")
ALIGN = os.path.join(ROOT, "alignment_scores.parquet")
PROBE = os.path.join(os.path.dirname(__file__), "stereo_control", "probe_stereo.parquet")
COHORT = "${DG_ROOT}/results/percat_layer_robust/cohort_e25.parquet"

CFG = ENC_CFG["t5"]
LAYERS = [2, 6, 10, 14, 18, 20, 22]   # T5 24 layers; detection peaks L22 -> see if control-best is shallower
ALPHA = 8.0
STEPS, GUID = 28, 4.5


@torch.no_grad()
def _t5_dir(pipe, stereo, anti):
    """per-prompt unit stereo−anti direction at every T5 layer -> [nL+1, hid]."""
    s = _layer_reps(pipe, [stereo], CFG)[0]
    a = _layer_reps(pipe, [anti], CFG)[0]
    d = s - a
    return d / (d.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def _gen(pipe, neutral, L, d_unit):
    blocks = _get_blocks(pipe, CFG)
    h = None
    if L != "clean":
        h = blocks[L].register_forward_hook(_ProjHook(d_unit[L + 1], ALPHA))
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        return pipe(prompt=neutral, num_inference_steps=STEPS, guidance_scale=GUID, generator=g).images[0]
    finally:
        if h is not None:
            h.remove()


def run_gen(shard=0, nshards=1):
    os.makedirs(PNG, exist_ok=True)
    coh = pd.read_parquet(COHORT)
    pipe = _load_sd3()
    jobs = [(i, L) for i in range(len(coh)) for L in (["clean"] + LAYERS)]
    mine = [j for n, j in enumerate(jobs) if n % nshards == shard]
    by_i = {}
    for i, L in mine:
        by_i.setdefault(i, []).append(L)
    rows = []
    for i, Ls in by_i.items():
        p = coh.iloc[i]
        du = _t5_dir(pipe, p["prompt_stereotype"], p["prompt_anti_stereotype"]) if any(L != "clean" for L in Ls) else None
        print(f"[gen] {p['id']} ({p['bias_type']}) Ls={Ls}", flush=True)
        for L in Ls:
            tag = "clean" if L == "clean" else f"L{L:02d}"
            ip = os.path.join(PNG, f"{p['id']}_{tag}.png")
            if not os.path.exists(ip):
                _gen(pipe, p["prompt_neutral"], L, du).save(ip)
            rows.append(dict(id=p["id"], bias_type=p["bias_type"], hook_layer=tag,
                             layer=(-1 if L == "clean" else L), seed=0, image_path=ip,
                             prompt_neutral=p["prompt_neutral"], stereotype_tails=p["stereotype_tails"],
                             anti_stereotype_tails=p["anti_stereotype_tails"], head=p["head"], relation=p["relation"],
                             variant=tag))
    pd.DataFrame(rows).to_parquet(MANIFEST.replace(".parquet", f"_shard{shard}.parquet"), index=False)
    print(f"[gen] shard {shard}: {len(rows)} rows")


def _merge():
    coh = pd.read_parquet(COHORT); meta = {r["id"]: r for _, r in coh.iterrows()}
    rows = []
    for ip in sorted(glob.glob(os.path.join(PNG, "*.png"))):
        base = os.path.basename(ip)[:-4]
        pid, tag = base.rsplit("_", 1)
        if pid not in meta:
            continue
        r = meta[pid]
        rows.append(dict(id=pid, bias_type=r["bias_type"], hook_layer=tag,
                         layer=(-1 if tag == "clean" else int(tag[1:])), seed=0, image_path=ip,
                         prompt_neutral=r["prompt_neutral"], stereotype_tails=r["stereotype_tails"],
                         anti_stereotype_tails=r["anti_stereotype_tails"], head=r["head"], relation=r["relation"],
                         variant=tag))
    man = pd.DataFrame(rows); man.to_parquet(MANIFEST, index=False); return man


def run_score():
    man = _merge()
    print(f"[score] {len(man)} images")
    if not (os.path.exists(BIAS) and len(pd.read_parquet(BIAS)) >= len(man)):
        os.system(f"python -u -m experiments.causal_patching.score_images --gpu 0 --single-pass "
                  f"--manifest {MANIFEST} --scores {BIAS}")
    if not (os.path.exists(ALIGN) and len(pd.read_parquet(ALIGN)) >= len(man)):
        os.system(f"python -u -m experiments.interp_program.rescore_alignment_interp "
                  f"--manifest {MANIFEST} --scores {ALIGN}")


def _scomp(ar, bm):
    return float(ar * (1.0 - bm / 5.0) * 100.0) if (ar == ar and bm == bm) else float("nan")


def run_finalize():
    man = pd.read_parquet(MANIFEST)
    b = pd.read_parquet(BIAS); man["bias"] = man["image_path"].map(dict(zip(b["image_path"], b["stereotype_score"])))
    a = pd.read_parquet(ALIGN); man["aligned"] = man["image_path"].map(dict(zip(a["image_path"], a["aligned"])))
    man["s_comp"] = man.apply(lambda r: _scomp(r["aligned"], r["bias"]), axis=1)
    man.to_csv(os.path.join(ROOT, "sd3_t5_detect_control.csv"), index=False)
    clean = man[man.hook_layer == "clean"]["s_comp"].mean()
    print(f"\n===== SD3.5 T5 per-LAYER debias (control), standard S_comp (clean={clean:.1f}) =====")
    print(f"  {'layer':6s} " + "  ".join(f"L{L:02d}" for L in LAYERS))
    print("  pooled " + "  ".join(f"{man[man.layer==L]['s_comp'].mean():4.0f}" for L in LAYERS))
    print("  Δvsclean " + " ".join(f"{man[man.layer==L]['s_comp'].mean()-clean:+4.0f}" for L in LAYERS))

    # detection probe (T5, pooled + per cat) vs control ΔS_comp across layers
    pb = pd.read_parquet(PROBE); t5 = pb[pb.encoder == "t5"]
    print("\n===== DETECTION (probe acc) vs CONTROL (ΔS_comp) across T5 layers, pooled =====")
    poolacc = t5[t5.category == "__pooled__"].set_index("layer")["acc"]
    dctrl = {L: man[man.layer == L]["s_comp"].mean() - clean for L in LAYERS}
    print("  layer:   " + "  ".join(f"L{L:02d}" for L in LAYERS))
    print("  probe:   " + "  ".join(f"{poolacc.get(L, float('nan')):.2f}" for L in LAYERS))
    print("  ΔS_comp: " + "  ".join(f"{dctrl[L]:+.1f}" for L in LAYERS))
    pa = np.array([poolacc.get(L, np.nan) for L in LAYERS]); dc = np.array([dctrl[L] for L in LAYERS])
    rho = pd.Series(pa).corr(pd.Series(dc), method="spearman")
    det_best = int(poolacc.idxmax()); ctrl_best = max(dctrl, key=dctrl.get)
    print(f"\n  detection-best layer L{det_best} (acc {poolacc.max():.3f}) | control-best layer L{ctrl_best} (Δ{dctrl[ctrl_best]:+.1f})")
    print(f"  ρ(probe_acc, ΔS_comp) across T5 layers = {rho:.2f}   (≈0 / negative => detection≠control)")

    print("\n  per-category control ΔS_comp by T5 layer:")
    print(f"    {'cat':12s} " + " ".join(f"L{L:02d}" for L in LAYERS))
    for cat, g in man.groupby("bias_type"):
        cc = g[g.hook_layer == "clean"]["s_comp"].mean()
        print(f"    {cat:12s} " + " ".join(f"{g[g.layer==L]['s_comp'].mean()-cc:+4.0f}" for L in LAYERS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gen", "score", "finalize"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    {"gen": lambda: run_gen(a.shard, a.nshards), "score": run_score, "finalize": run_finalize}[a.mode]()


if __name__ == "__main__":
    main()
