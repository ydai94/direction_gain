"""Exp 21 — per-prompt encoder steering.

Closes the granularity ladder: pooled (Exp 16, S_comp 70.3) → per-category
(Exp 17, 67.5) → per-prompt (here). The per-prompt direction is the natural
counterpart to CDG's per-prompt contrastive embedding, but applied at the
encoder layer projection lever rather than in the DiT loop.

For each prompt p and layer L, the steering direction is
    d_hat_p[L] = unit(reps_stereo[p, L] - reps_anti[p, L])
read directly from the cached `reps_mean.h5` (no extra encoder forward).

Two sub-experiments, both vs. pooled and per-category baselines:

  --mode survival           : encoder-only edit-survival on all 1831 prompts × 28
                              layers using per-prompt d_hat. Mirrors
                              layer_probing/edit_survival.py but swaps pooled
                              d_hat[L] → d_hat_p[L].
  --mode aggregate_survival : aggregate per-prompt parquet → ratio-of-means CIs
                              per (pooling, category, layer); merge with pooled
                              baseline for comparison.
  --mode generate           : TL-CDG with per-prompt d_hat at L10+L12 + golden
                              CDG ρ=4 on 24-robust × 3 seeds. Same pipeline as
                              Exp 16/17, only direction granularity differs.
  --mode finalize           : bootstrap CIs + per-category breakdown vs Exp 16
                              (pooled), Exp 17 (per-category), Exp 20a (per-cat
                              alpha-scaled), and CDG baseline.

NOTE: per-prompt steering requires labeled stereo/anti for each prompt
(non-deployable, like Exp 16/17). The point is to see whether direction
granularity is the missing lever — and what it implies for the survival metric.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch

from experiments.causal_patching import encoder_runner as ER
from experiments.causal_patching.projection_patcher import ProjectionPatcher
from experiments.contrastive_denoising_guidance_pilot import cdg_loop
from experiments.interp_program import config_interp as CFG
from experiments.layer_probing import config_local as LC


# ----- output paths --------------------------------------------------------

ROOT = os.path.join(CFG.OUT_DIR, "perprompt_encoder")
DEC = os.path.join(CFG.DECODED_DIR, "perprompt_encoder")
MANIFEST = os.path.join(ROOT, "manifest_e21.parquet")
BIAS = os.path.join(ROOT, "bias_scores_e21.parquet")
ALIGN = os.path.join(ROOT, "alignment_scores_e21.parquet")

PER_PROMPT_SURV = os.path.join(ROOT, "edit_survival_perprompt_dir.parquet")
SURV_RESULTS = os.path.join(ROOT, "edit_survival_perprompt_dir_results.parquet")

SELECTED = os.path.join(CFG.OUT_DIR, "robust", "selected.parquet")
TLCDG_ROOT = os.path.join(CFG.OUT_DIR, "tlcdg_robust")        # Exp 16 (pooled)
TLCDG_PC_ROOT = os.path.join(CFG.OUT_DIR, "tlcdg_percat")     # Exp 17 (per-cat)
TLCDG_PCAS_ROOT = os.path.join(CFG.OUT_DIR, "tlcdg_percat_alpha_scaled")  # 20a
CDG_ROOT = os.path.join(CFG.OUT_DIR, "cdg_robust")
ROBUST_ROOT = os.path.join(CFG.OUT_DIR, "robust")

VARIANT = "tlcdg_multi_r4_perprompt"
LAYERS = [10, 12]
ALPHA = 2.0
RHO = 4.0
SEEDS = [0, 1, 2]

NLAYERS = LC.NUM_HIDDEN_LAYERS  # 28
N_BOOT = 1000
SEED = LC.BOOTSTRAP_SEED


# ============================================================
# Per-prompt direction tables from cached reps
# ============================================================

def load_perprompt_directions():
    """Return:
       d_pp     : (N_ids, 29, D) unit per-prompt direction
       mu       : (29, D)        global per-layer mean (same μ as Exp 3/16/17)
       id_to_idx: {id_str: row in d_pp}
       bt_of_id : {id_str: bias_type}

    Each row of d_pp is `unit(reps[stereo_p, L] − reps[anti_p, L])` for the
    prompt p (variant rows from prompt_index.parquet).
    """
    pi = pd.read_parquet(LC.PROMPT_INDEX_PATH)
    with h5py.File(LC.REPS_PATH["mean"], "r") as f:
        reps = f["reps"][...].astype(np.float32)  # (N, 29, D)
    piv = pi.pivot(index="id", columns="variant", values="row_idx")
    piv = piv[list(LC.VARIANTS)].dropna().astype(int)
    bt_of_id = (pi.drop_duplicates("id")
                .set_index("id")["bias_type"]
                .to_dict())
    ids = list(piv.index)

    h_s = reps[piv["stereo"].values]   # (N_ids, 29, D)
    h_a = reps[piv["anti"].values]
    d_raw = h_s - h_a                  # (N_ids, 29, D)
    norms = np.linalg.norm(d_raw, axis=-1, keepdims=True) + 1e-8
    d_pp = d_raw / norms
    mu = reps.mean(0)
    id_to_idx = {str(pid): i for i, pid in enumerate(ids)}

    print(f"[exp21] loaded {len(ids)} per-prompt directions × {d_pp.shape[1]} layers "
          f"× D={d_pp.shape[2]}")
    return d_pp, mu, id_to_idx, bt_of_id


# ============================================================
# Mode 1: per-prompt direction edit-survival sweep
# ============================================================

def _pool_mean(pe): return pe[0].float().mean(0).cpu().numpy()
def _pool_last(pe): return pe[0, -1].float().cpu().numpy()


def _encode(pipe, prompt, project_layer=None, d=None, mu=None):
    enc = ER.tokenize_wrapped(pipe.tokenizer, prompt, device="cuda")
    inner = ER._text_decoder(pipe.text_encoder)
    patcher = None
    try:
        if project_layer is not None:
            patcher = ProjectionPatcher()
            patcher.mode = "project"
            patcher.direction = torch.from_numpy(np.ascontiguousarray(d))
            patcher.mu = torch.from_numpy(np.ascontiguousarray(mu))
            patcher.alpha = ALPHA
            T = enc["input_ids"].shape[1]
            patcher.patch_mask = torch.ones((1, T), dtype=torch.bool)
            patcher.install(inner.layers[project_layer - 1])
        with torch.no_grad():
            out = inner(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        output_hidden_states=False, use_cache=False)
        pe, _ = ER.post_process_for_dit(
            out.last_hidden_state, enc["attention_mask"],
            target_dtype=torch.bfloat16)
        return pe
    finally:
        if patcher is not None:
            patcher.remove()


def run_survival():
    """Per-prompt direction survival: 1831 prompts × 28 layers.

    For each (id, L_e):
      1. Encode neutral clean → vn = pool(out)
      2. Encode neutral with ProjectionPatcher(d=d_hat_p[L_e]) → ve = pool(out)
      3. lean = v · d_out where d_out = unit(pool(stereo_out) − pool(anti_out))
      4. survival = (lean_clean − lean_edit) / |lean_clean|

    Steps 1 & 3 reuse the same clean encoder outputs across L_e (cached per id).
    Step 2 needs one encoder forward per (id, L_e) — 1831 × 28 = ~51k forwards
    total, batchable per id. Reusing encoded clean stereo/anti for d_out means
    NO extra encoder runs for the direction itself; only the edit pass.

    Resumable via incremental parquet flush.
    """
    from experiments.causal_patching.run_three_methods import load_pipe

    d_pp, mu, id_to_idx, _ = load_perprompt_directions()

    pi = pd.read_parquet(LC.PROMPT_INDEX_PATH)
    wide = (pi.pivot(index="id", columns="variant", values="prompt")
              .reset_index())
    wide = wide.merge(
        pi.drop_duplicates("id")[["id", "bias_type"]], on="id")

    existing = (pd.read_parquet(PER_PROMPT_SURV)
                if os.path.exists(PER_PROMPT_SURV) else None)
    done = (set(zip(existing["id"].astype(str), existing["edit_L"].astype(int)))
            if existing is not None else set())
    if done:
        print(f"[resume] {len(done)} (id, edit_L) cells already done.")

    os.makedirs(ROOT, exist_ok=True)
    pipe = load_pipe()
    t0 = time.time(); n_new = 0
    new_rows = []

    for ii, row in enumerate(wide.itertuples(index=False)):
        pid = str(row.id); btype = row.bias_type
        n_txt, s_txt, a_txt = row.neutral, row.stereo, row.anti
        if pid not in id_to_idx:
            continue
        p_idx = id_to_idx[pid]

        pe_n = _encode(pipe, n_txt)
        pe_s = _encode(pipe, s_txt)
        pe_a = _encode(pipe, a_txt)
        vn_m, vs_m, va_m = _pool_mean(pe_n), _pool_mean(pe_s), _pool_mean(pe_a)
        vn_l, vs_l, va_l = _pool_last(pe_n), _pool_last(pe_s), _pool_last(pe_a)
        d_m = vs_m - va_m; d_m = d_m / (np.linalg.norm(d_m) + 1e-8)
        d_l = vs_l - va_l; d_l = d_l / (np.linalg.norm(d_l) + 1e-8)
        lean_clean_m = float(vn_m @ d_m); lean_clean_l = float(vn_l @ d_l)

        for Le in range(1, NLAYERS + 1):
            if (pid, Le) in done:
                continue
            pe_e = _encode(pipe, n_txt,
                           project_layer=Le,
                           d=d_pp[p_idx, Le], mu=mu[Le])
            ve_m, ve_l = _pool_mean(pe_e), _pool_last(pe_e)
            le_m = float(ve_m @ d_m); le_l = float(ve_l @ d_l)
            new_rows.append(dict(
                id=pid, bias_type=btype, edit_L=int(Le),
                lean_clean_mean=lean_clean_m, lean_edit_mean=le_m,
                lean_clean_last=lean_clean_l, lean_edit_last=le_l,
                survival_mean=(lean_clean_m - le_m) / (abs(lean_clean_m) + 1e-8),
                survival_last=(lean_clean_l - le_l) / (abs(lean_clean_l) + 1e-8),
            ))
            n_new += 1

        if (ii + 1) % 25 == 0 or ii + 1 == len(wide):
            elapsed = time.time() - t0
            rate = n_new / max(elapsed, 1e-6)
            print(f"[exp21/surv] {ii+1}/{len(wide)} prompts "
                  f"({n_new} new cells, {rate:.1f}/s, {elapsed:.0f}s elapsed)")
            if new_rows:
                df_new = pd.DataFrame(new_rows)
                if os.path.exists(PER_PROMPT_SURV):
                    df_new = pd.concat(
                        [pd.read_parquet(PER_PROMPT_SURV), df_new],
                        ignore_index=True)
                df_new = df_new.drop_duplicates(["id", "edit_L"], keep="last")
                df_new.to_parquet(PER_PROMPT_SURV, index=False)
                new_rows = []

    print(f"[exp21/surv] DONE in {time.time()-t0:.0f}s -> {PER_PROMPT_SURV}")


# ============================================================
# Mode 2: aggregate survival to ratio-of-means table + compare with pooled
# ============================================================

def _bootstrap_ratio_of_means(num, den, n_boot, seed):
    num = np.asarray(num, float); den = np.asarray(den, float)
    m = np.isfinite(num) & np.isfinite(den)
    num, den = num[m], den[m]
    if len(num) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(num.mean() / (np.abs(den).mean() + 1e-12))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(num), len(num))
        boot.append(num[idx].mean() / (np.abs(den[idx]).mean() + 1e-12))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def aggregate_survival():
    df = pd.read_parquet(PER_PROMPT_SURV)
    print(f"[exp21/agg] {len(df)} (id, L) rows; "
          f"{df['id'].nunique()} prompts × {df['edit_L'].nunique()} layers")
    for pool in LC.POOLINGS:
        df[f"num_{pool}"] = df[f"lean_clean_{pool}"] - df[f"lean_edit_{pool}"]
        df[f"den_{pool}"] = df[f"lean_clean_{pool}"]

    rows = []
    for pooling in LC.POOLINGS:
        nc, dc = f"num_{pooling}", f"den_{pooling}"
        for L in range(1, NLAYERS + 1):
            sub = df[df["edit_L"] == L]
            m, lo, hi = _bootstrap_ratio_of_means(
                sub[nc].values, sub[dc].values, N_BOOT, SEED)
            rows.append(dict(metric="edit_survival_perprompt_dir",
                             pooling=pooling, category="__pooled__", layer=L,
                             mean=m, ci_lo=lo, ci_hi=hi, n=len(sub)))
        for cat in sorted(df["bias_type"].dropna().unique()):
            cat_df = df[df["bias_type"] == cat]
            for L in range(1, NLAYERS + 1):
                sub = cat_df[cat_df["edit_L"] == L]
                if len(sub) < 2:
                    m = lo = hi = float("nan")
                else:
                    m, lo, hi = _bootstrap_ratio_of_means(
                        sub[nc].values, sub[dc].values, N_BOOT, SEED)
                rows.append(dict(metric="edit_survival_perprompt_dir",
                                 pooling=pooling, category=cat, layer=L,
                                 mean=m, ci_lo=lo, ci_hi=hi, n=len(sub)))

    out = pd.DataFrame(rows)
    out.to_parquet(SURV_RESULTS, index=False)
    print(f"[exp21/agg] wrote {SURV_RESULTS} ({len(out)} rows)")

    # Compare pooled vs per-prompt survival side by side
    pp = out[(out.pooling == "mean") & (out.category == "__pooled__")][
        ["layer", "mean", "ci_lo", "ci_hi"]].rename(columns={
            "mean": "surv_pp", "ci_lo": "pp_lo", "ci_hi": "pp_hi"})

    pooled_path = os.path.join(LC.CACHE_DIR, "edit_survival_results.parquet")
    if os.path.exists(pooled_path):
        po = pd.read_parquet(pooled_path)
        po = po[(po.pooling == "mean") & (po.category == "__pooled__")][
            ["layer", "mean", "ci_lo", "ci_hi"]].rename(columns={
                "mean": "surv_pool", "ci_lo": "po_lo", "ci_hi": "po_hi"})
        comp = pp.merge(po, on="layer")
        comp["delta"] = comp["surv_pp"] - comp["surv_pool"]
        comp.to_csv(os.path.join(ROOT, "survival_compare.csv"), index=False)
        print("\nPer-prompt vs pooled survival (mean pool, n=1831):")
        print(comp.round(3).to_string(index=False))
    else:
        print(f"[warn] pooled baseline not found at {pooled_path}; "
              f"skipping side-by-side comparison.")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.0))
        ax.plot(pp["layer"], pp["surv_pp"], "-o", color="C3",
                label="per-prompt direction (Exp 21)")
        ax.fill_between(pp["layer"], pp["pp_lo"], pp["pp_hi"],
                        color="C3", alpha=0.18)
        if os.path.exists(pooled_path):
            po = pd.read_parquet(pooled_path)
            po = po[(po.pooling == "mean") & (po.category == "__pooled__")]
            ax.plot(po["layer"], po["mean"], "-s", color="C0",
                    label="pooled direction (LP+)")
            ax.fill_between(po["layer"], po["ci_lo"], po["ci_hi"],
                            color="C0", alpha=0.18)
        ax.set_xlabel("encoder edit layer L_e")
        ax.set_ylabel("edit-survival (ratio-of-means)")
        ax.set_title("Per-prompt vs pooled direction at the encoder "
                     "(1831 prompts × 28 layers)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        os.makedirs(LC.PLOTS_DIR, exist_ok=True)
        out_png = os.path.join(LC.PLOTS_DIR, "edit_survival_perprompt_vs_pooled.png")
        fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f"[exp21/agg] wrote {out_png}")
    except Exception as e:
        print(f"[exp21/agg] plot skipped: {e}")


# ============================================================
# Mode 3: TL-CDG generation with per-prompt direction
# ============================================================

def project_multi_perprompt(pipe, prompt, layers, alpha, d_for_layers, mu):
    """Same as exp13_methods_robust.project_multi(mode='project') except
    `d_for_layers` is a dict {L: per-prompt direction (D,)} so each layer
    uses the SAME prompt's direction (not a per-layer pooled vector)."""
    enc = ER.tokenize_wrapped(pipe.tokenizer, prompt, device="cuda")
    inner = ER._text_decoder(pipe.text_encoder)
    T = enc["input_ids"].shape[1]
    mask = torch.ones((1, T), dtype=torch.bool)
    patchers = []
    try:
        for L in layers:
            p = ProjectionPatcher()
            p.mode = "project"
            p.alpha = float(alpha)
            p.mu = torch.from_numpy(np.ascontiguousarray(mu[L]))
            p.patch_mask = mask
            p.direction = torch.from_numpy(
                np.ascontiguousarray(d_for_layers[L]))
            p.install(inner.layers[L - 1])
            patchers.append(p)
        with torch.no_grad():
            out = inner(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        output_hidden_states=False, use_cache=False)
    finally:
        for p in patchers:
            p.remove()
    return ER.post_process_for_dit(
        out.last_hidden_state, enc["attention_mask"],
        target_dtype=torch.bfloat16)


def run_generate(seeds):
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(DEC, exist_ok=True)
    from experiments.causal_patching.run_three_methods import load_pipe

    sel = pd.read_parquet(SELECTED)
    d_pp, mu, id_to_idx, _ = load_perprompt_directions()
    pipe = load_pipe()
    pe_u, pm_u = pipe.encode_prompt(prompt=" ",
                                    device=pipe._execution_device)

    rows = []
    d_var = os.path.join(DEC, VARIANT)
    os.makedirs(d_var, exist_ok=True)

    for _, p in sel.iterrows():
        pid = str(p["id"])
        neutral = p["prompt_neutral"]
        stereo = p["prompt_stereotype"]
        anti = p["prompt_anti_stereotype"]
        bias_type = p["bias_type"]
        if pid not in id_to_idx:
            raise KeyError(f"id {pid} not in reps cache")
        p_idx = id_to_idx[pid]
        d_for_layers = {L: d_pp[p_idx, L] for L in LAYERS}

        pe_gold_a, pm_gold_a = pipe.encode_prompt(
            prompt=anti, device=pipe._execution_device)
        pe_gold_s, pm_gold_s = pipe.encode_prompt(
            prompt=stereo, device=pipe._execution_device)

        pe_neu, pm_neu = project_multi_perprompt(
            pipe, neutral, LAYERS, ALPHA, d_for_layers, mu)

        for s in seeds:
            ip = os.path.join(d_var, f"{pid}_s{s}.png")
            if not os.path.exists(ip):
                print(f"[exp21/gen] {pid} ({bias_type}) {VARIANT} s{s} rho={RHO}")
                embeds = {
                    "n": (pe_neu, pm_neu),
                    "a": (pe_gold_a, pm_gold_a),
                    "s": (pe_gold_s, pm_gold_s),
                    "uncond": (pe_u, pm_u),
                }
                img = cdg_loop.generate_cdg(
                    pipe, embeds, int(s),
                    schedule="all", orthogonalize=False, rho=float(RHO),
                    num_steps=CFG.NUM_STEPS, cfg=CFG.CFG)
                img.save(ip)
            else:
                print(f"[exp21/gen] skip (exists) {ip}")
            rows.append(dict(exp="perprompt_encoder", id=pid,
                             bias_type=bias_type, layer=-1, variant=VARIANT,
                             seed=int(s), step=-1, image_path=ip,
                             prompt_neutral=neutral))

    if os.path.exists(MANIFEST):
        prev = pd.read_parquet(MANIFEST).to_dict("records")
        seen = {r["image_path"] for r in rows}
        rows += [r for r in prev if r["image_path"] not in seen]
    pd.DataFrame(rows).drop_duplicates("image_path").to_parquet(
        MANIFEST, index=False)
    print(f"[exp21/gen] manifest now {len(pd.read_parquet(MANIFEST))} images")


# ============================================================
# Mode 4: finalize generation results
# ============================================================

def _scomp(ar, bm):
    return float(ar * (1.0 - bm / 5.0) * 100.0) \
        if (ar == ar and bm == bm) else float("nan")


def _bootstrap(arr, n=2000, seed=0):
    arr = np.asarray([a for a in arr if a == a], float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bm = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n)]
    return (float(arr.mean()),
            float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)))


def _load_compare(man_path, bias_path, align_path, variant_filter=None):
    if not os.path.exists(man_path):
        return None
    m = pd.read_parquet(man_path)
    if variant_filter is not None:
        m = m[m["variant"] == variant_filter]
    bm = (dict(zip(*[pd.read_parquet(bias_path)[c]
                     for c in ["image_path", "stereotype_score"]]))
          if os.path.exists(bias_path) else {})
    am = (dict(zip(*[pd.read_parquet(align_path)[c]
                     for c in ["image_path", "aligned"]]))
          if os.path.exists(align_path) else {})
    m = m.copy()
    m["bias"] = m["image_path"].map(bm)
    m["aligned"] = m["image_path"].map(am)
    return m


def run_finalize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = _load_compare(MANIFEST, BIAS, ALIGN)
    if man is None:
        raise FileNotFoundError(MANIFEST)

    tm = _load_compare(
        os.path.join(TLCDG_ROOT, "manifest_tlcdgr.parquet"),
        os.path.join(TLCDG_ROOT, "bias_scores_tlcdgr.parquet"),
        os.path.join(TLCDG_ROOT, "alignment_scores_tlcdgr.parquet"),
        variant_filter="tlcdg_multi_r4")
    tpc = _load_compare(
        os.path.join(TLCDG_PC_ROOT, "manifest_tlcdgpc.parquet"),
        os.path.join(TLCDG_PC_ROOT, "bias_scores_tlcdgpc.parquet"),
        os.path.join(TLCDG_PC_ROOT, "alignment_scores_tlcdgpc.parquet"),
        variant_filter="tlcdg_multi_r4_percat")
    tpcas = _load_compare(
        os.path.join(TLCDG_PCAS_ROOT, "manifest_e20a.parquet"),
        os.path.join(TLCDG_PCAS_ROOT, "bias_scores_e20a.parquet"),
        os.path.join(TLCDG_PCAS_ROOT, "alignment_scores_e20a.parquet"),
        variant_filter="tlcdg_multi_r4_percat_alphascaled")
    cm = _load_compare(
        os.path.join(CDG_ROOT, "manifest_cdg.parquet"),
        os.path.join(CDG_ROOT, "bias_scores_cdg.parquet"),
        os.path.join(CDG_ROOT, "alignment_scores_cdg.parquet"),
        variant_filter="cdg_all_rho4")
    rb = _load_compare(
        os.path.join(ROBUST_ROOT, "manifest_robust.parquet"),
        os.path.join(ROBUST_ROOT, "bias_scores_rb.parquet"),
        os.path.join(ROBUST_ROOT, "alignment_scores_rb.parquet"),
        variant_filter="clean")

    frames = []
    for label_df, label in [(man, VARIANT),
                            (tm, "tlcdg_multi_r4"),
                            (tpc, "tlcdg_multi_r4_percat"),
                            (tpcas, "tlcdg_multi_r4_percat_alphascaled"),
                            (cm, "cdg_all_rho4"),
                            (rb, "clean")]:
        if label_df is None:
            continue
        s = label_df[["id", "bias_type", "seed", "bias", "aligned"]].copy()
        s["method"] = label
        frames.append(s)

    allf = pd.concat(frames, ignore_index=True)
    allf["s_comp"] = allf.apply(
        lambda r: _scomp(r["aligned"], r["bias"]), axis=1)
    allf.to_csv(os.path.join(ROOT, "per_image_e21.csv"), index=False)

    rows = []
    for m, sub in allf.groupby("method"):
        bmean, blo, bhi = _bootstrap(sub["bias"])
        amean, alo, ahi = _bootstrap(sub["aligned"].astype(float))
        smean, slo, shi = _bootstrap(sub["s_comp"])
        rows.append(dict(method=m, n=len(sub),
                         bias_mean=bmean,
                         bias_ci=f"[{blo:.2f},{bhi:.2f}]",
                         aligned_rate=amean,
                         aligned_ci=f"[{alo:.2f},{ahi:.2f}]",
                         s_comp=smean,
                         s_comp_ci=f"[{slo:.1f},{shi:.1f}]"))
    order = ["clean", "cdg_all_rho4",
             "tlcdg_multi_r4",
             "tlcdg_multi_r4_percat",
             "tlcdg_multi_r4_percat_alphascaled",
             VARIANT]
    tbl = pd.DataFrame(rows)
    tbl["__o"] = tbl["method"].apply(
        lambda v: order.index(v) if v in order else 99)
    tbl = tbl.sort_values("__o").drop(columns="__o")
    tbl.to_csv(os.path.join(ROOT, "e21_summary.csv"), index=False)

    catrows = []
    for m, sub in allf.groupby("method"):
        for c, g in sub.groupby("bias_type"):
            mean, lo, hi = _bootstrap(g["s_comp"])
            catrows.append(dict(method=m, bias_type=c, n=len(g),
                                s_comp=mean, ci_lo=lo, ci_hi=hi))
    catdf = pd.DataFrame(catrows)
    catdf.to_csv(os.path.join(ROOT, "e21_by_category.csv"), index=False)

    piv = catdf[catdf.method.isin(order)].pivot(
        index="bias_type", columns="method", values="s_comp")
    piv = piv.reindex(columns=[m for m in order if m in piv.columns])
    if VARIANT in piv.columns and "tlcdg_multi_r4" in piv.columns:
        piv["Δ_perprompt_vs_pooled"] = piv[VARIANT] - piv["tlcdg_multi_r4"]
    if VARIANT in piv.columns and "tlcdg_multi_r4_percat" in piv.columns:
        piv["Δ_perprompt_vs_percat"] = piv[VARIANT] - piv["tlcdg_multi_r4_percat"]
    if VARIANT in piv.columns and "cdg_all_rho4" in piv.columns:
        piv["Δ_perprompt_vs_cdg"] = piv[VARIANT] - piv["cdg_all_rho4"]
    piv.to_csv(os.path.join(ROOT, "e21_delta.csv"))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ms = [m for m in order if m in set(tbl.method)]
    bs = [_bootstrap(allf[allf.method == m]["s_comp"]) for m in ms]
    y = [x[0] for x in bs]
    lo = [x[0] - x[1] for x in bs]
    hi = [x[2] - x[0] for x in bs]
    colors = {"clean": "gray", "cdg_all_rho4": "C2",
              "tlcdg_multi_r4": "C0",
              "tlcdg_multi_r4_percat": "C1",
              "tlcdg_multi_r4_percat_alphascaled": "C4",
              VARIANT: "C3"}
    ax.bar(ms, y, yerr=[lo, hi], capsize=5,
           color=[colors.get(m, "C7") for m in ms])
    ax.set_ylabel("S_comp (95% CI)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Per-prompt encoder steering vs prior TL-CDG variants "
                 f"({allf['id'].nunique()} prompts × "
                 f"{allf['seed'].nunique()} seeds)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.FIG_DIR, "exp21_perprompt_compare.png"),
                dpi=130)
    plt.close(fig)

    print("[exp21/finalize] summary:")
    print(tbl.to_string(index=False))
    print("\n[exp21/finalize] per-category Δ vs pooled/percat/cdg:")
    print(piv.round(1).to_string())


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["survival", "aggregate_survival",
                             "generate", "finalize"])
    ap.add_argument("--seeds",
                    default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    if args.mode == "survival":
        run_survival()
    elif args.mode == "aggregate_survival":
        aggregate_survival()
    elif args.mode == "generate":
        run_generate([int(s) for s in args.seeds.split(",") if s != ""])
    else:
        run_finalize()


if __name__ == "__main__":
    main()
