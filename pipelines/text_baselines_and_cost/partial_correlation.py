"""Partial Spearman correlation of G with one seed's improvement, controlling for the other seed."""
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


def partial_spear(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank residualisation)."""
    rx, ry, rz = (rankdata(v, axis=-1).astype(float) for v in (x, y, z))

    def resid(a, b):
        a = a - a.mean(axis=-1, keepdims=True)
        b = b - b.mean(axis=-1, keepdims=True)
        beta = (a * b).sum(axis=-1, keepdims=True) / np.maximum((b * b).sum(axis=-1, keepdims=True), 1e-12)
        return a - beta * b

    ex, ey = resid(rx, rz), resid(ry, rz)
    ex = ex - ex.mean(axis=-1, keepdims=True)
    ey = ey - ey.mean(axis=-1, keepdims=True)
    den = np.sqrt((ex * ex).sum(axis=-1) * (ey * ey).sum(axis=-1))
    return np.divide((ex * ey).sum(axis=-1), den, out=np.full_like(den, np.nan), where=den > 1e-12)


data_files = {m: DATA / f"test_{m}.csv" for m in ["sd3", "flux", "qwen"]}

part_rows = []

for model in MODELS:
    test = pd.read_csv(data_files[model])
    arms = ["U_clean_3", "U_clean_4", "U_steer_3", "U_steer_4", "U_random0_3", "U_random0_4"]
    m = test[test[arms].notna().all(axis=1)]
    m = m[m[["G", "M", "H", "A0"]].notna().all(axis=1)].copy()
    R = m["R"].to_numpy(float)
    R3 = (m["U_steer_3"] - m["U_clean_3"]).to_numpy(float)
    R4 = (m["U_steer_4"] - m["U_clean_4"]).to_numpy(float)
    n = len(m)
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, n, size=(NBOOT, n))
    Rb = R[draw]

    # ---- partial correlation: does G add beyond one generated-and-judged seed
    p43 = float(partial_spear(G_ := m["G"].to_numpy(float), R4, R3))
    p34 = float(partial_spear(G_, R3, R4))
    bp43 = partial_spear(G_[draw], R4[draw], R3[draw])
    bp34 = partial_spear(G_[draw], R3[draw], R4[draw])
    part_rows.append(dict(model=model, n=n,
                          partial_G_R4_given_R3=round(p43, 4), ci_lo=round(ci(bp43)[0], 4), ci_hi=round(ci(bp43)[1], 4),
                          partial_G_R3_given_R4=round(p34, 4), ci2_lo=round(ci(bp34)[0], 4), ci2_hi=round(ci(bp34)[1], 4)))

part = pd.DataFrame(part_rows)
part.to_csv(OUTDIR / "a1_partial_corr.csv", index=False)
print(part.to_string(index=False))