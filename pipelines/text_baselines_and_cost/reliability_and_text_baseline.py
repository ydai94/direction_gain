"""Prediction reliability (disattenuated rho*), text-only baseline B_text, and one-seed image-pair baseline.

Reads data/per_prompt/{test,dev,meta}_<model>.csv|json; writes results/text_baselines_and_cost/a1_a3_results.json.
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "per_prompt"
OUTDIR = ROOT / "results" / "text_baselines_and_cost"
OUTDIR.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


MODELS = ["sd3", "flux", "qwen"]
TEXT_FEATS = ["input_direction_norm", "pair_separability", "T_rel"]
B_FEATS = TEXT_FEATS + ["H", "A0"]
NBOOT = 10000
SEED = 20260912


def spear(x, y):
    x = rankdata(x, axis=-1).astype(float)
    y = rankdata(y, axis=-1).astype(float)
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt((x * x).sum(axis=-1) * (y * y).sum(axis=-1))
    return np.divide((x * y).sum(axis=-1), den, out=np.full_like(den, np.nan), where=den > 1e-12)


def ci(v):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else None


def sb(r, k=2.0):
    return k * r / (1.0 + (k - 1.0) * r)


def fit_ridge(dev, feats):
    X = dev[feats].to_numpy(float)
    y = dev["R"].to_numpy(float)
    sc = StandardScaler()
    Z = sc.fit_transform(X)
    f = Ridge(alpha=1.0, solver="svd").fit(Z, y)
    return dict(features=feats, mean=sc.mean_, scale=sc.scale_, coef=f.coef_, intercept=float(f.intercept_))


def predict(fit, df):
    X = df[fit["features"]].to_numpy(float)
    return ((X - fit["mean"]) / fit["scale"]) @ fit["coef"] + fit["intercept"]


def gg_reliability(C, rng, n_splits=200):
    n, k = C.shape
    halves = []
    idx = np.arange(k)
    for _ in range(n_splits):
        p = rng.permutation(idx)
        a, b = p[: k // 2], p[k // 2:]
        halves.append(spear(C[:, a].mean(axis=1), C[:, b].mean(axis=1)))
    r_hh = float(np.mean(halves))
    return sb(r_hh, 2.0), r_hh


out = {"n_boot": NBOOT, "seed": SEED, "models": {}}
rng_global = np.random.default_rng(SEED)

for model in MODELS:
    test = pd.read_csv(DATA / f"test_{model}.csv")
    dev = pd.read_csv(DATA / f"dev_{model}.csv")
    meta = json.load(open(DATA / f"meta_{model}.json"))

    cos_cols = sorted([c for c in test.columns if c.startswith("cos_s")])
    assert len(cos_cols) == 6, cos_cols

    arms = ["U_clean_3", "U_clean_4", "U_steer_3", "U_steer_4", "U_random0_3", "U_random0_4"]
    valid = test[test[arms].notna().all(axis=1)].copy()
    matched = valid[valid[["G", "M", "H", "A0"]].notna().all(axis=1)].copy()
    matched["R3"] = matched["U_steer_3"] - matched["U_clean_3"]
    matched["R4"] = matched["U_steer_4"] - matched["U_clean_4"]
    assert np.allclose(matched[["R3", "R4"]].mean(axis=1), matched["R"]), "per-seed R does not rebuild R"

    G = matched["G"].to_numpy(float)
    M = matched["M"].to_numpy(float)
    R = matched["R"].to_numpy(float)
    R3 = matched["R3"].to_numpy(float)
    R4 = matched["R4"].to_numpy(float)
    C = matched[cos_cols].to_numpy(float)
    n = len(matched)

    fits = {"B": fit_ridge(dev, B_FEATS), "B_text": fit_ridge(dev, TEXT_FEATS), "B_G": fit_ridge(dev, B_FEATS + ["G"])}
    preds = {k: predict(v, matched) for k, v in fits.items()}

    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, n, size=(NBOOT, n))

    rho_G = float(spear(G, R))
    rho_M = float(spear(M, R))
    rho_Btext = float(spear(preds["B_text"], R))
    rho_B = float(spear(preds["B"], R))
    r12 = float(spear(R3, R4))
    r_RR = sb(r12, 2.0)
    r_GG, r_hh = gg_reliability(C, rng_global)
    rho_star = rho_G / np.sqrt(r_RR * r_GG)
    one_seed = 0.5 * (float(spear(R3, R4)) + float(spear(R4, R3)))
    g_vs_seed4 = float(spear(G, R4))
    g_vs_seed3 = float(spear(G, R3))

    Rb, R3b, R4b = R[draw], R3[draw], R4[draw]
    b_rho_G = spear(G[draw], Rb)
    b_rho_M = spear(M[draw], Rb)
    b_rho_Btext = spear(preds["B_text"][draw], Rb)
    b_r12 = spear(R3b, R4b)
    b_rRR = sb(b_r12, 2.0)
    b_rGG = []
    for _ in range(20):
        p = rng_global.permutation(6)
        a, bb = p[:3], p[3:]
        b_rGG.append(sb(spear(C[:, a].mean(axis=1)[draw], C[:, bb].mean(axis=1)[draw]), 2.0))
    b_rGG = np.mean(b_rGG, axis=0)
    with np.errstate(invalid="ignore"):
        b_rho_star = b_rho_G / np.sqrt(b_rRR * b_rGG)
    b_g4 = spear(G[draw], R4b)
    b_one = spear(R3b, R4b)

    mse = {k: float(np.mean((R - v) ** 2)) for k, v in preds.items()}

    out["models"][model] = dict(
        n_valid=int(len(valid)), n_matched=n, null=meta["null"], cos_cols=cos_cols,
        published_check=dict(rho_G=rho_G, rho_M=rho_M,
                             rho_G_ci=ci(b_rho_G), rho_G_minus_M_ci=ci(b_rho_G - b_rho_M),
                             mse_B=mse["B"], mse_B_G=mse["B_G"],
                             mse_reduction_pct=100 * (mse["B"] - mse["B_G"]) / mse["B"]),
        A1=dict(rho_obs=rho_G, r12_seed=r12, r_RR=r_RR, r_hh_G=r_hh, r_GG=r_GG,
                rho_star=float(rho_star), rho_star_ci=ci(b_rho_star),
                r_RR_ci=ci(b_rRR), r_GG_ci=ci(b_rGG),
                sd_R=float(R.std(ddof=1)), sd_R_seeddiff=float((R3 - R4).std(ddof=1))),
        A3a=dict(rho_B_text=rho_Btext, rho_B_text_ci=ci(b_rho_Btext),
                 rho_G_minus_B_text=rho_G - rho_Btext,
                 rho_G_minus_B_text_ci=ci(b_rho_G - b_rho_Btext),
                 mse_B_text=mse["B_text"], rho_B_full=rho_B,
                 coef_B_text=dict(zip(TEXT_FEATS, fits["B_text"]["coef"].round(4).tolist()))),
        A3b=dict(rho_one_seed_vs_other=one_seed, rho_one_seed_ci=ci(b_one),
                 rho_G_vs_seed4=g_vs_seed4, rho_G_vs_seed3=g_vs_seed3,
                 rho_G_minus_one_seed=g_vs_seed4 - one_seed,
                 rho_G_minus_one_seed_ci=ci(b_g4 - b_one)),
    )

json.dump(out, open(OUTDIR / "a1_a3_results.json", "w"), indent=1)