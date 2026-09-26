"""Exp 38 — DiT DEPTH map of control: which of the 60 DiT blocks carries debias?

Exp 37 mapped the DiT TIME axis (when, across 50 steps): control is front-loaded.
But CDG injects its contrastive residual at the DiT OUTPUT (after block 60), so it
collapses the DEPTH axis. The DiT has 60 transformer blocks; this experiment asks
WHERE in that depth the debias lever lives — the DiT analog of the encoder L8–L12
band, and the missing third panel of the control map (encoder-depth / DiT-time /
DiT-depth).

Mechanism — block-localized contrastive injection (the DiT analog of output-CDG):
within each denoising step, on the SAME current latent,
    h_k^a = block_k image-stream output with the ANTI prompt
    h_k^s = block_k image-stream output with the STEREO prompt
    r_k   = h_k^a - h_k^s          (contrastive direction in block-k residual space)
inject into the NEUTRAL forward at block k:
    h_k^n  <-  h_k^n + w_t * lam_k * r_k,   lam_k = rho * ||h_k^n|| / (||r_k|| + eps)
(per-token norm; lam normalizes the perturbation to a fixed RELATIVE size at every
block, so ΔS_comp across blocks reflects "does intervening here propagate to a
debiased image", not "is the a/s signal big here"). Blocks k+1..59 then propagate
it. At k=last this ~reduces to output-CDG; sweeping k gives a depth curve.

Time is held at the strong regime (schedule controls which steps inject; default
"all" for max detection SNR — Exp 37 showed early≈all so time is not the variable
here). rho is calibrated in --mode pilot before the full sweep.

  python -m experiments.interp_program.exp38_dit_depth --mode pilot
  python -m experiments.interp_program.exp38_dit_depth --mode generate --shard i --nshards N
  python -m experiments.interp_program.exp38_dit_depth --mode manifest
  python -m experiments.interp_program.exp38_dit_depth --mode finalize
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps

from experiments.interp_program import config_interp as CFG
from experiments.contrastive_denoising_guidance_pilot import cdg_loop

ROOT = os.path.join(CFG.OUT_DIR, "dit_depth")
DEC = os.path.join(CFG.DECODED_DIR, "dit_depth")
MANIFEST = os.path.join(ROOT, "manifest_e38.parquet")
BIAS = os.path.join(ROOT, "bias_scores_e38.parquet")
ALIGN = os.path.join(ROOT, "alignment_scores_e38.parquet")

SELECTED = os.path.join(CFG.OUT_DIR, "robust", "selected.parquet")
ROBUST_ROOT = os.path.join(CFG.OUT_DIR, "robust")

# 60 DiT blocks -> coarse 16-point grid; refine around any responsive band later.
BLOCKS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 59]
RHO = 0.5          # set from --mode pilot
SCHEDULE = "all"   # inject at every step (Exp 37: early≈all; time not the variable)
SEEDS = [0, 1]
NUM_STEPS = 50
_EPS = 1e-8


class BlockHook:
    """Forward hook on one DiT block: capture image-stream output, or inject r_k."""
    def __init__(self):
        self.mode = "off"        # off | capture | inject
        self.slot = None
        self.store = {}
        self.r = None
        self.rho = 0.0
        self.w = 1.0

    def __call__(self, module, inp, out):
        enc, hid = out           # (encoder_hidden_states, hidden_states[image stream])
        if self.mode == "capture":
            self.store[self.slot] = hid.detach()
            return out
        if self.mode == "inject" and self.r is not None and self.rho > 0.0:
            lam = self.rho * cdg_loop._per_token_norm(hid) / (cdg_loop._per_token_norm(self.r) + _EPS)
            return (enc, hid + (self.w * lam * self.r).to(hid.dtype))
        return out


@torch.no_grad()
def generate_dit_depth(pipe, embeds, seed, block_k, *, rho, schedule, num_steps=NUM_STEPS, cfg=4.0):
    """Block-k contrastive injection denoise -> PIL image. Mirrors cdg_loop setup."""
    device = pipe._execution_device
    transformer = pipe.transformer
    scheduler = pipe.scheduler

    pe_n, pm_n = embeds["n"]; pe_u, pm_u = embeds["uncond"]
    pe_a, pm_a = embeds["a"]; pe_s, pm_s = embeds["s"]
    dtype = pe_n.dtype
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = height

    generator = torch.Generator(device=device).manual_seed(int(seed))
    num_channels_latents = transformer.config.in_channels // 4
    latents = pipe.prepare_latents(1, num_channels_latents, height, width, dtype, device, generator, None)
    img_shapes = [[(1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)]]

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_steps = retrieve_timesteps(scheduler, num_steps, device, sigmas=sigmas, mu=mu)
    guidance = None

    def forward(pe, pm, ts, ctx):
        with transformer.cache_context(ctx):
            return transformer(
                hidden_states=latents, timestep=ts / 1000, guidance=guidance,
                encoder_hidden_states_mask=pm, encoder_hidden_states=pe,
                img_shapes=img_shapes, attention_kwargs={}, return_dict=False)[0]

    H = BlockHook()
    handle = transformer.transformer_blocks[block_k].register_forward_hook(H)
    try:
        scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            ts = t.expand(latents.shape[0]).to(latents.dtype)
            w_t = cdg_loop.schedule_weight(i, num_steps, schedule)
            active = w_t > 0.0 and rho > 0.0
            if active:
                H.mode = "capture"; H.slot = "a"; forward(pe_a, pm_a, ts, "cond_a")
                H.mode = "capture"; H.slot = "s"; forward(pe_s, pm_s, ts, "cond_s")
                H.r = H.store["a"] - H.store["s"]; H.rho = rho; H.w = float(w_t)
                H.mode = "inject"; cond_n = forward(pe_n, pm_n, ts, "cond")
                H.mode = "off"
            else:
                cond_n = forward(pe_n, pm_n, ts, "cond")
            uncond = forward(pe_u, pm_u, ts, "uncond")
            comb = uncond + cfg * (cond_n - uncond)
            pred = comb * (cdg_loop._per_token_norm(cond_n) / cdg_loop._per_token_norm(comb))
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    finally:
        handle.remove()

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    lm = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    ls = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / ls + lm
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _embeds(pipe, p, pe_u, pm_u):
    enc = lambda s: pipe.encode_prompt(prompt=s, device=pipe._execution_device)
    pe_n, pm_n = enc(p["prompt_neutral"])
    pe_a, pm_a = enc(p["prompt_anti_stereotype"])
    pe_s, pm_s = enc(p["prompt_stereotype"])
    return {"n": (pe_n, pm_n), "a": (pe_a, pm_a), "s": (pe_s, pm_s), "uncond": (pe_u, pm_u)}


def run_pilot():
    """Verify the hook works + calibrate rho: 2 prompts x {early/mid/late block} x rho grid."""
    pdir = os.path.join(DEC, "pilot"); os.makedirs(pdir, exist_ok=True); os.makedirs(ROOT, exist_ok=True)
    from experiments.causal_patching.run_three_methods import load_pipe
    sel = pd.read_parquet(SELECTED).iloc[[0, 1]]
    pipe = load_pipe()
    pe_u, pm_u = pipe.encode_prompt(prompt=" ", device=pipe._execution_device)
    rows = []
    for _, p in sel.iterrows():
        emb = _embeds(pipe, p, pe_u, pm_u)
        # clean reference (rho=0)
        ip = os.path.join(pdir, f"{p['id']}_clean.png")
        if not os.path.exists(ip):
            generate_dit_depth(pipe, emb, 0, 30, rho=0.0, schedule=SCHEDULE).save(ip)
        for k in (8, 30, 56):
            for rho in (0.25, 0.5, 1.0, 2.0):
                ip = os.path.join(pdir, f"{p['id']}_L{k}_r{rho}.png")
                if not os.path.exists(ip):
                    print(f"[pilot] {p['id']} L{k} rho={rho}", flush=True)
                    generate_dit_depth(pipe, emb, 0, k, rho=rho, schedule=SCHEDULE).save(ip)
                rows.append(dict(id=p["id"], block=k, rho=rho, image_path=ip))
    pd.DataFrame(rows).to_parquet(os.path.join(ROOT, "pilot_manifest.parquet"), index=False)
    print(f"[pilot] {len(rows)} images in {pdir} — inspect for: not-destroyed, debias visible at some (k,rho)")


def run_generate(shard=0, nshards=1, rho=RHO, schedule=SCHEDULE, seeds=SEEDS):
    os.makedirs(ROOT, exist_ok=True); os.makedirs(DEC, exist_ok=True)
    from experiments.causal_patching.run_three_methods import load_pipe
    sel = pd.read_parquet(SELECTED)
    jobs = [(i, k, s) for i in range(len(sel)) for k in BLOCKS for s in seeds]
    mine = [j for n, j in enumerate(jobs) if n % nshards == shard]
    by_prompt = {}
    for i, k, s in mine:
        by_prompt.setdefault(i, []).append((k, s))
    print(f"[exp38] shard {shard}/{nshards}: {len(mine)}/{len(jobs)} imgs, rho={rho}, schedule={schedule}", flush=True)
    pipe = load_pipe()
    pe_u, pm_u = pipe.encode_prompt(prompt=" ", device=pipe._execution_device)
    for i, tasks in by_prompt.items():
        p = sel.iloc[i]; emb = _embeds(pipe, p, pe_u, pm_u)
        for k, s in tasks:
            sub = os.path.join(DEC, f"L{k:02d}"); os.makedirs(sub, exist_ok=True)
            ip = os.path.join(sub, f"{p['id']}_s{s}.png")
            if not os.path.exists(ip):
                print(f"[exp38] {p['id']} L{k} s{s}", flush=True)
                generate_dit_depth(pipe, emb, int(s), k, rho=rho, schedule=schedule).save(ip)
    print(f"[exp38] shard {shard}/{nshards} done", flush=True)


def run_manifest(seeds=SEEDS):
    import glob
    sel = pd.read_parquet(SELECTED)
    meta = {r["id"]: (r["bias_type"], r["prompt_neutral"]) for _, r in sel.iterrows()}
    rows = []
    for k in BLOCKS:
        for ip in sorted(glob.glob(os.path.join(DEC, f"L{k:02d}", "*.png"))):
            fn = os.path.basename(ip)[:-4]
            pid, _, stag = fn.rpartition("_s")
            if pid not in meta:
                continue
            bt, neutral = meta[pid]
            rows.append(dict(exp="dit_depth", id=pid, bias_type=bt, layer=k,
                             variant=f"L{k:02d}", seed=int(stag), step=-1,
                             image_path=ip, prompt_neutral=neutral))
    pd.DataFrame(rows).drop_duplicates("image_path").to_parquet(MANIFEST, index=False)
    print(f"[exp38/manifest] {len(rows)} images -> {MANIFEST}")


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
        b = pd.read_parquet(bp)
        m["bias"] = m["image_path"].map(dict(zip(b["image_path"], b["stereotype_score"])))
    if os.path.exists(ap):
        a = pd.read_parquet(ap)
        m["aligned"] = m["image_path"].map(dict(zip(a["image_path"], a["aligned"])))
    return m


def run_finalize():
    man = _load(MANIFEST, BIAS, ALIGN)
    rman = _load(os.path.join(ROBUST_ROOT, "manifest_robust.parquet"),
                 os.path.join(ROBUST_ROOT, "bias_scores_rb.parquet"),
                 os.path.join(ROBUST_ROOT, "alignment_scores_rb.parquet"))
    frames = []
    clean = rman[rman.variant == "clean"]
    if not clean.empty:
        s = clean[["id", "bias_type", "seed", "bias", "aligned"]].copy(); s["method"] = "clean"
        frames.append(s)
    for v, sub in man.groupby("variant"):
        s = sub[["id", "bias_type", "seed", "bias", "aligned"]].copy(); s["method"] = v
        frames.append(s)
    allf = pd.concat(frames, ignore_index=True)
    allf["s_comp"] = allf.apply(lambda r: _scomp(r["aligned"], r["bias"]), axis=1)
    allf.to_csv(os.path.join(ROOT, "per_image_e38.csv"), index=False)

    rows = []
    for m, sub in allf.groupby("method"):
        bmean, _, _ = _bootstrap(sub["bias"])
        amean, _, _ = _bootstrap(sub["aligned"].astype(float))
        smean, slo, shi = _bootstrap(sub["s_comp"])
        rows.append(dict(method=m, n=len(sub), bias=bmean, aligned=amean,
                         s_comp=smean, s_comp_ci=f"[{slo:.1f},{shi:.1f}]"))
    tbl = pd.DataFrame(rows)
    tbl["__o"] = tbl["method"].apply(lambda v: -1 if v == "clean" else int(v[1:]) if v.startswith("L") else 99)
    tbl = tbl.sort_values("__o").drop(columns="__o")
    tbl.to_csv(os.path.join(ROOT, "e38_summary.csv"), index=False)
    print("\n========== EXP 38 — DiT DEPTH map (S_comp per injected block; clean baseline) ==========")
    print(tbl.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "generate", "manifest", "finalize"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--rho", type=float, default=RHO)
    ap.add_argument("--schedule", default=SCHEDULE)
    args = ap.parse_args()
    if args.mode == "pilot":
        run_pilot()
    elif args.mode == "generate":
        run_generate(shard=args.shard, nshards=args.nshards, rho=args.rho, schedule=args.schedule)
    elif args.mode == "manifest":
        run_manifest()
    else:
        run_finalize()


if __name__ == "__main__":
    main()
