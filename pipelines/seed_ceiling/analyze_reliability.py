"""A1-full: outcome reliability and noise-free correlation with 8 outcome seeds
(the original 3,4 plus new 10-15) on the frozen 300-case subset.

Reads: data/seed_ceiling_scores/scores_<model>.jsonl (new seeds), data/per_prompt/test_<model>.csv (original seeds, G, M),
       Exp271 per-cell cosines (already in test csv).
Reports: rho(G, R_k) as a function of k seeds; r_RR from k seeds (split-half, SB); rho*.
"""
import glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS = ["sd3", "flux", "qwen"]
NEW_SEEDS = [10, 11, 12, 13, 14, 15]
OLD_SEEDS = [3, 4]
NBOOT = 5000
SEED = 20260912


def spear(x, y):
    x = rankdata(x, axis=-1).astype(float); y = rankdata(y, axis=-1).astype(float)
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    return np.divide((x * y).sum(-1), den, out=np.full_like(den, np.nan), where=den > 1e-12)


def ci(v):
    v = np.asarray(v); v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def sb(r, k):
    return k * r / (1 + (k - 1) * r)


rows_curve, rows_main = [], []
for m in MODELS:
    # --- new-seed scores -> per-seed U ---
    recs = []
    for f in glob.glob(str(ROOT / "data" / "seed_ceiling_scores" / f"scores_{m}.jsonl")):
        recs += [json.loads(l) for l in open(f)]
    d = {}
    for r in recs:
        if r["value"] is None: continue
        d.setdefault((r["triplet_key"], r["arm"], r["seed"]), {})[r["metric"]] = r["value"]
    U = {k: 20 * int(v["alignment"]) * (5 - v["bias"]) for k, v in d.items() if len(v) == 2}

    t = pd.read_csv(ROOT / "data" / "per_prompt" / f"test_{m}.csv").set_index("triplet_key")
    keys = sorted({k for (k, _, _) in U})
    # per-seed improvement matrix: rows = cases, cols = seeds (3,4,10..15)
    Rmat, G, M, kept = [], [], [], []
    for k in keys:
        if k not in t.index or pd.isna(t.loc[k, "G"]): continue
        row = []
        ok = True
        for s in OLD_SEEDS:
            a, b = t.loc[k, f"U_steer_{s}"], t.loc[k, f"U_clean_{s}"]
            if pd.isna(a) or pd.isna(b): ok = False; break
            row.append(a - b)
        for s in NEW_SEEDS:
            if (k, "steer", s) not in U or (k, "clean", s) not in U: ok = False; break
            row.append(U[(k, "steer", s)] - U[(k, "clean", s)])
        if not ok: continue
        Rmat.append(row); G.append(t.loc[k, "G"]); M.append(t.loc[k, "M"]); kept.append(k)
    Rmat = np.array(Rmat); G = np.array(G); M = np.array(M); n = len(kept)
    nseeds = Rmat.shape[1]
    assert nseeds == 8, Rmat.shape
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, n, size=(NBOOT, n))

    # --- rho(G, mean of k seeds), averaged over seed subsets ---
    for k in [1, 2, 4, 8]:
        subs = list(itertools.combinations(range(nseeds), k))
        if len(subs) > 40: subs = [subs[i] for i in rng.choice(len(subs), 40, replace=False)]
        vals = [float(spear(G, Rmat[:, list(s)].mean(1))) for s in subs]
        rows_curve.append(dict(model=m, k_seeds=k, rho_G=round(np.mean(vals), 4),
                               rho_G_min=round(min(vals), 4), rho_G_max=round(max(vals), 4), n_subsets=len(subs)))

    # --- reliability from 8 seeds: split-half (4 vs 4), SB to 8 and to 2 ---
    halves = []
    for _ in range(200):
        p = rng.permutation(nseeds); a, b = p[:4], p[4:]
        halves.append(float(spear(Rmat[:, a].mean(1), Rmat[:, b].mean(1))))
    r_hh = float(np.mean(halves))             # reliability of a 4-seed mean
    r_single = r_hh / (4 - 3 * r_hh)          # invert SB: single-seed reliability
    r_RR8 = sb(r_single, 8); r_RR2 = sb(r_single, 2)
    R8 = Rmat.mean(1); R2 = Rmat[:, :2].mean(1)
    rho8 = float(spear(G, R8)); rho2 = float(spear(G, R2))
    # G reliability from stored cells (same as A1-lite)
    cos_cols = sorted([c for c in t.columns if c.startswith("cos_s")])
    C = t.loc[kept, cos_cols].to_numpy(float)
    g_h = [sb(float(spear(C[:, p[:3]].mean(1), C[:, p[3:]].mean(1))), 2) for p in (rng.permutation(6) for _ in range(200))]
    r_GG = float(np.mean(g_h))
    rho_star8 = rho8 / np.sqrt(r_RR8 * r_GG)
    rho_star2 = rho2 / np.sqrt(r_RR2 * r_GG)
    # bootstrap rho_star8
    bs = []
    for row in draw[:2000]:
        Rb, Gb, Cb = Rmat[row], G[row], C[row]
        p = rng.permutation(nseeds)
        rh = float(spear(Rb[:, p[:4]].mean(1), Rb[:, p[4:]].mean(1)))
        rs = rh / (4 - 3 * rh); rr8 = sb(rs, 8)
        q = rng.permutation(6); gg = sb(float(spear(Cb[:, q[:3]].mean(1), Cb[:, q[3:]].mean(1))), 2)
        with np.errstate(invalid="ignore"):
            bs.append(float(spear(Gb, Rb.mean(1))) / np.sqrt(rr8 * gg))
    rows_main.append(dict(model=m, n=n, rho_G_2seed=round(rho2, 4), rho_G_8seed=round(rho8, 4),
                          rho_M_8seed=round(float(spear(M, R8)), 4),
                          r_single_seed=round(r_single, 4), r_RR_2=round(r_RR2, 4), r_RR_8=round(r_RR8, 4),
                          r_GG=round(r_GG, 4), rho_star_from2=round(rho_star2, 4),
                          rho_star_from8=round(float(rho_star8), 4),
                          rho_star8_ci=[round(x, 3) for x in ci(bs)],
                          sd_R8=round(float(R8.std(ddof=1)), 2), sd_R2=round(float(R2.std(ddof=1)), 2)))

curve = pd.DataFrame(rows_curve); main = pd.DataFrame(rows_main)
OUT = ROOT / "results" / "seed_ceiling"; OUT.mkdir(parents=True, exist_ok=True)
curve.to_csv(OUT / "seed_curve.csv", index=False); main.to_csv(OUT / "reliability.csv", index=False)
print(curve.to_string(index=False)); print(); print(main.to_string(index=False))
