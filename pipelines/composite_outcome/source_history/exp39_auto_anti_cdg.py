"""Exp 39 — auto-anti CDG: make the DiT lever DEPLOYABLE (no golden anti/stereo prompts).

The decisive debias lever is CDG in the DiT loop (S_comp ~69), but it needs GOLDEN
anti/stereo prompt pairs per prompt -> not deployable. The deployable line stalled at
embed_swap@15 = 47.5. This experiment removes the golden requirement: synthesize the
anti & stereo conditionings from the NEUTRAL embed plus a precomputed stereotype
direction d (learned offline from golden pairs, NOT from the test prompt's own pair):

    pe_a = pe_n + (alpha/2) * d        # toward anti
    pe_s = pe_n - (alpha/2) * d        # toward stereo
then run the standard CDG loop with these synthetic a/s. At inference a new prompt
needs only pe_n and d (a fixed asset) — deployable.

d is built two ways, both LEAVE-ONE-OUT (a test prompt never uses its own golden pair):
  global : d_i = mean over ALL j!=i of [pool(enc(anti_j)) - pool(enc(stereo_j))]
           (the common "de-stereotype" component; analog of the Exp 19 +0.93 global dir)
  percat : d_i = mean over j!=i in the SAME category
(pool = mean over valid tokens -> one (D,) vector, broadcast over pe_n's tokens.)

Compare auto-anti CDG (deployable) vs clean 26.4, golden CDG 69.2 (ceiling, uses golden
pairs), embed_swap@15 47.5 (prior deployable winner). Reuses cdg_loop.generate_cdg and
the standard scorers; no shared code modified.

  python -m experiments.interp_program.exp39_auto_anti_cdg --mode pilot
  python -m experiments.interp_program.exp39_auto_anti_cdg --mode dirs        # build+cache directions
  python -m experiments.interp_program.exp39_auto_anti_cdg --mode generate --dirsrc global --alpha 4 --shard i --nshards N
  python -m experiments.interp_program.exp39_auto_anti_cdg --mode manifest
  python -m experiments.interp_program.exp39_auto_anti_cdg --mode finalize
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from experiments.interp_program import config_interp as CFG
from experiments.contrastive_denoising_guidance_pilot import cdg_loop

ROOT = os.path.join(CFG.OUT_DIR, "auto_anti")
DEC = os.path.join(CFG.DECODED_DIR, "auto_anti")
MANIFEST = os.path.join(ROOT, "manifest_e39.parquet")
BIAS = os.path.join(ROOT, "bias_scores_e39.parquet")
ALIGN = os.path.join(ROOT, "alignment_scores_e39.parquet")
DIRS_NPZ = os.path.join(ROOT, "auto_anti_dirs.npz")

SELECTED = os.path.join(CFG.OUT_DIR, "robust", "selected.parquet")
ROBUST_ROOT = os.path.join(CFG.OUT_DIR, "robust")
CDG_ROOT = os.path.join(CFG.OUT_DIR, "cdg_robust")

ALPHAS = [2.0, 3.0, 4.0]   # per-token perturbation magnitude (unit dir); ~2.75 flipped in pilot
SCHEDULES = ["all", "ramp_down"]   # Exp 37: ramp_down ≈ all, better aligned
RHO = 4.0
SEEDS = [0, 1]
NUM_STEPS = 50


def _pool(pe, pm):
    """Mean over valid tokens -> (D,)."""
    pe = pe.detach()
    if pm is None:
        return pe[0].mean(0)
    m = pm[0].bool()
    return pe[0][m].mean(0)


def build_dirs(pipe):
    """Encode golden pairs, build LOO global & per-category directions + cache neutral embeds."""
    sel = pd.read_parquet(SELECTED)
    enc = lambda s: pipe.encode_prompt(prompt=s, device=pipe._execution_device)
    diffs, cats, ids = [], [], []
    for _, p in sel.iterrows():
        pe_a, pm_a = enc(p["prompt_anti_stereotype"])
        pe_s, pm_s = enc(p["prompt_stereotype"])
        d = (_pool(pe_a, pm_a) - _pool(pe_s, pm_s)).float().cpu().numpy()
        diffs.append(d); cats.append(p["bias_type"]); ids.append(p["id"])
    diffs = np.stack(diffs); cats = np.array(cats); ids = np.array(ids)
    n = len(ids)
    d_global = np.zeros_like(diffs); d_percat = np.zeros_like(diffs)
    for i in range(n):
        others = np.arange(n) != i
        d_global[i] = diffs[others].mean(0)
        same = others & (cats == cats[i])
        d_percat[i] = diffs[same].mean(0) if same.any() else diffs[i]
    os.makedirs(ROOT, exist_ok=True)
    np.savez(DIRS_NPZ, ids=ids, cats=cats, d_global=d_global, d_percat=d_percat,
             raw_norm_global=np.linalg.norm(d_global, axis=1),
             raw_norm_percat=np.linalg.norm(d_percat, axis=1))
    print(f"[dirs] built global+percat LOO dirs for {n} prompts; "
          f"mean ‖d_global‖={np.linalg.norm(d_global,axis=1).mean():.2f}, "
          f"‖d_percat‖={np.linalg.norm(d_percat,axis=1).mean():.2f}")
    return ids, cats, d_global, d_percat


def _synth_embeds(pipe, pe_n, pm_n, d_vec, alpha, pe_u, pm_u):
    """pe_a/pe_s = pe_n ± alpha·d̂  (d UNIT-normalized so global/percat are comparable;
    alpha is the per-token perturbation magnitude added, broadcast over tokens)."""
    dev = pipe._execution_device
    d = torch.as_tensor(d_vec, dtype=pe_n.dtype, device=dev)
    d = d / (d.norm() + 1e-8)
    step = (alpha * d).view(1, 1, -1)
    return {"n": (pe_n, pm_n), "a": (pe_n + step, pm_n),
            "s": (pe_n - step, pm_n), "uncond": (pe_u, pm_u)}


def _load_dirs():
    z = np.load(DIRS_NPZ, allow_pickle=True)
    return {str(i): {"global": g, "percat": c}
            for i, g, c in zip(z["ids"], z["d_global"], z["d_percat"])}


def run_pilot():
    """Calibrate alpha on 2 prompts (global dir, schedule=all), eyeball debias vs destroy."""
    pdir = os.path.join(DEC, "pilot"); os.makedirs(pdir, exist_ok=True)
    from experiments.causal_patching.run_three_methods import load_pipe
    pipe = load_pipe()
    pe_u, pm_u = pipe.encode_prompt(prompt=" ", device=pipe._execution_device)
    ids, cats, d_global, d_percat = build_dirs(pipe)
    sel = pd.read_parquet(SELECTED)
    pick = [sel.index[sel.id == ids[k]][0] for k in (1, 5)]  # 2 prompts
    for ridx in pick:
        p = sel.loc[ridx]; i = int(np.where(ids == p["id"])[0][0])
        pe_n, pm_n = pipe.encode_prompt(prompt=p["prompt_neutral"], device=pipe._execution_device)
        ip = os.path.join(pdir, f"{p['id']}_clean.png")
        if not os.path.exists(ip):
            emb = _synth_embeds(pipe, pe_n, pm_n, d_global[i], 0.0, pe_u, pm_u)
            cdg_loop.generate_cdg(pipe, emb, 0, schedule="all", orthogonalize=False, rho=0.0).save(ip)
        for a in (1.0, 2.0, 4.0, 8.0):
            ip = os.path.join(pdir, f"{p['id']}_global_a{a}.png")
            if not os.path.exists(ip):
                print(f"[pilot] {p['id']} ({p['bias_type']}) global a={a}", flush=True)
                emb = _synth_embeds(pipe, pe_n, pm_n, d_global[i], a, pe_u, pm_u)
                cdg_loop.generate_cdg(pipe, emb, 0, schedule="all", orthogonalize=False, rho=RHO).save(ip)
    print(f"[pilot] images in {pdir} — pick alpha: debias visible, image intact")


def run_generate(dirsrc, alpha, schedule, shard=0, nshards=1, seeds=SEEDS):
    os.makedirs(ROOT, exist_ok=True); os.makedirs(DEC, exist_ok=True)
    from experiments.causal_patching.run_three_methods import load_pipe
    sel = pd.read_parquet(SELECTED)
    if not os.path.exists(DIRS_NPZ):
        pipe = load_pipe(); build_dirs(pipe)
    else:
        pipe = load_pipe()
    dirs = _load_dirs()
    pe_u, pm_u = pipe.encode_prompt(prompt=" ", device=pipe._execution_device)
    variant = f"{dirsrc}_a{alpha}_{schedule}"
    jobs = [(i, s) for i in range(len(sel)) for s in seeds]
    mine = [j for n, j in enumerate(jobs) if n % nshards == shard]
    print(f"[exp39] {variant} shard {shard}/{nshards}: {len(mine)}/{len(jobs)} imgs", flush=True)
    sub = os.path.join(DEC, variant); os.makedirs(sub, exist_ok=True)
    for i, s in mine:
        p = sel.iloc[i]
        ip = os.path.join(sub, f"{p['id']}_s{s}.png")
        if os.path.exists(ip):
            continue
        pe_n, pm_n = pipe.encode_prompt(prompt=p["prompt_neutral"], device=pipe._execution_device)
        d_vec = dirs[p["id"]][dirsrc]
        emb = _synth_embeds(pipe, pe_n, pm_n, d_vec, alpha, pe_u, pm_u)
        print(f"[exp39] {p['id']} {variant} s{s}", flush=True)
        cdg_loop.generate_cdg(pipe, emb, int(s), schedule=schedule,
                              orthogonalize=False, rho=RHO, num_steps=NUM_STEPS, cfg=CFG.CFG).save(ip)
    print(f"[exp39] {variant} shard {shard}/{nshards} done", flush=True)


def run_manifest(seeds=SEEDS):
    import glob
    sel = pd.read_parquet(SELECTED)
    meta = {r["id"]: (r["bias_type"], r["prompt_neutral"]) for _, r in sel.iterrows()}
    rows = []
    for sub in sorted(glob.glob(os.path.join(DEC, "*"))):
        variant = os.path.basename(sub)
        if variant == "pilot":
            continue
        for ip in sorted(glob.glob(os.path.join(sub, "*.png"))):
            pid, _, stag = os.path.basename(ip)[:-4].rpartition("_s")
            if pid not in meta:
                continue
            bt, neutral = meta[pid]
            rows.append(dict(exp="auto_anti", id=pid, bias_type=bt, layer=-1,
                             variant=variant, seed=int(stag), step=-1,
                             image_path=ip, prompt_neutral=neutral))
    pd.DataFrame(rows).drop_duplicates("image_path").to_parquet(MANIFEST, index=False)
    print(f"[exp39/manifest] {len(rows)} images -> {MANIFEST}")


def _scomp(ar, bm):
    return float(ar * (1.0 - bm / 5.0) * 100.0) if (ar == ar and bm == bm) else float("nan")


def _bootstrap(arr, n=2000, seed=0):
    arr = np.asarray([a for a in arr if a == a], float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bm = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n)]
    return float(arr.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def _load(mp, bp, ap):
    m = pd.read_parquet(mp)
    if os.path.exists(bp):
        b = pd.read_parquet(bp); m["bias"] = m["image_path"].map(dict(zip(b["image_path"], b["stereotype_score"])))
    if os.path.exists(ap):
        a = pd.read_parquet(ap); m["aligned"] = m["image_path"].map(dict(zip(a["image_path"], a["aligned"])))
    return m


def run_finalize():
    man = _load(MANIFEST, BIAS, ALIGN)
    rman = _load(os.path.join(ROBUST_ROOT, "manifest_robust.parquet"),
                 os.path.join(ROBUST_ROOT, "bias_scores_rb.parquet"),
                 os.path.join(ROBUST_ROOT, "alignment_scores_rb.parquet"))
    cman = _load(os.path.join(CDG_ROOT, "manifest_cdg.parquet"),
                 os.path.join(CDG_ROOT, "bias_scores_cdg.parquet"),
                 os.path.join(CDG_ROOT, "alignment_scores_cdg.parquet"))
    frames = []
    for sub, key in [(rman[rman.variant == "clean"], "clean"),
                     (cman[cman.variant == "cdg_all_rho4"], "golden_cdg_rho4")]:
        if not sub.empty:
            s = sub[["id", "bias_type", "seed", "bias", "aligned"]].copy(); s["method"] = key
            frames.append(s)
    for v, sub in man.groupby("variant"):
        s = sub[["id", "bias_type", "seed", "bias", "aligned"]].copy(); s["method"] = v
        frames.append(s)
    allf = pd.concat(frames, ignore_index=True)
    allf["s_comp"] = allf.apply(lambda r: _scomp(r["aligned"], r["bias"]), axis=1)
    allf.to_csv(os.path.join(ROOT, "per_image_e39.csv"), index=False)
    rows = []
    for m, sub in allf.groupby("method"):
        bmean, _, _ = _bootstrap(sub["bias"]); amean, _, _ = _bootstrap(sub["aligned"].astype(float))
        smean, slo, shi = _bootstrap(sub["s_comp"])
        rows.append(dict(method=m, n=len(sub), bias=bmean, aligned=amean,
                         s_comp=smean, s_comp_ci=f"[{slo:.1f},{shi:.1f}]"))
    tbl = pd.DataFrame(rows).sort_values("s_comp", ascending=False)
    tbl.to_csv(os.path.join(ROOT, "e39_summary.csv"), index=False)
    print("\n===== EXP 39 — auto-anti CDG (deployable; clean 26.4 / golden CDG 69.2 / embed_swap 47.5) =====")
    print(tbl.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "dirs", "generate", "manifest", "finalize"])
    ap.add_argument("--dirsrc", default="global", choices=["global", "percat"])
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--schedule", default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "pilot":
        run_pilot()
    elif args.mode == "dirs":
        from experiments.causal_patching.run_three_methods import load_pipe
        build_dirs(load_pipe())
    elif args.mode == "generate":
        run_generate(args.dirsrc, args.alpha, args.schedule, shard=args.shard, nshards=args.nshards)
    elif args.mode == "manifest":
        run_manifest()
    else:
        run_finalize()


if __name__ == "__main__":
    main()
