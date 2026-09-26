"""Correlation retained with fewer probe seeds/timesteps, and forward-call budget per configuration."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "per_prompt"
OUTDIR = ROOT / "results" / "text_baselines_and_cost"
OUTDIR.mkdir(parents=True, exist_ok=True)
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS = ["sd3", "flux", "qwen"]
NBOOT = 10000
SEED = 20260912

COST = {
    "sd3": dict(steps=28, traj_forward_per_seed=28, gen_forward_per_image=28, probe_forward_per_cell=4),
    "flux": dict(steps=8, traj_forward_per_seed=8, gen_forward_per_image=8, probe_forward_per_cell=4),
    "qwen": dict(steps=50, traj_forward_per_seed=34, gen_forward_per_image=100, probe_forward_per_cell=4),
}

MODEL_FILES = {m: DATA / f"test_{m}.csv" for m in ["sd3", "flux", "qwen"]}


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


abl_rows = []

for model in MODELS:
    test = pd.read_csv(MODEL_FILES[model])
    arms = ["U_clean_3", "U_clean_4", "U_steer_3", "U_steer_4", "U_random0_3", "U_random0_4"]
    m = test[test[arms].notna().all(axis=1)]
    m = m[m[["G", "M", "H", "A0"]].notna().all(axis=1)].copy()
    cos_cols = sorted([c for c in m.columns if c.startswith("cos_s")])
    seeds = sorted({int(c.split("_")[1][1:]) for c in cos_cols})
    steps = sorted({int(c.split("_")[2][1:]) for c in cos_cols})
    C = {(int(c.split("_")[1][1:]), int(c.split("_")[2][1:])): m[c].to_numpy(float) for c in cos_cols}
    R = m["R"].to_numpy(float)
    n = len(m)
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, n, size=(NBOOT, n))
    Rb = R[draw]
    cst = COST[model]

    full = np.mean([C[k] for k in C], axis=0)
    rho_full = float(spear(full, R))
    b_full = spear(full[draw], Rb)
    for ns in [1, 2, 3]:
        for nt in [1, 2]:
            vals, bdiffs = [], []
            for ss in itertools.combinations(seeds, ns):
                for tt in itertools.combinations(steps, nt):
                    v = np.mean([C[(s, t)] for s in ss for t in tt], axis=0)
                    vals.append(float(spear(v, R)))
                    bdiffs.append(spear(v[draw], Rb) - b_full)
            cells = ns * nt
            fwd = ns * cst["traj_forward_per_seed"] + cells * cst["probe_forward_per_cell"]
            abl_rows.append(dict(model=model, n_seeds=ns, n_steps=nt, cells=cells,
                                 n_subsets=len(vals),
                                 rho_mean=round(float(np.mean(vals)), 4),
                                 rho_min=round(float(np.min(vals)), 4),
                                 rho_max=round(float(np.max(vals)), 4),
                                 retained_pct=round(100 * float(np.mean(vals)) / rho_full, 1),
                                 vs_full_ci_lo=round(ci(np.mean(bdiffs, axis=0))[0], 4),
                                 vs_full_ci_hi=round(ci(np.mean(bdiffs, axis=0))[1], 4),
                                 forwards=fwd))

abl = pd.DataFrame(abl_rows)
abl.to_csv(OUTDIR / "a1_probe_ablation.csv", index=False)
print(abl.to_string(index=False))