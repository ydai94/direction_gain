"""Exp280 B4 analysis: does G still predict benefit when the steering strength changes,
and can G choose a per-prompt strength?

Outcomes use the paper's composite U = 20 * alignment * (5 - bias), averaged over outcome
seeds 3 and 4, minus the clean mean. The deployed strength is not regenerated: its steered
images and all clean images come from Exp271. A secondary outcome uses the eight-seed clean
mean (Exp271 seeds 3,4 plus Exp276 seeds 10-15), which halves the noise on the clean side
for free because those images already exist and are already judged.

G at each strength is the mean cosine over the deployed probe cells at that strength. The
development affine calibration is shared across strengths and monotone, so it changes no
ordering; raw mean cosine is used and per-strength z-scores are reported alongside, because
a score that drifts with strength would make an argmax over strength degenerate.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(os.environ.get("DG_ROOT", "."))
RES = ROOT / 'results'
OUT = RES / 'probe_budget_and_strength'
PLAN = json.loads((ROOT / 'pipelines/probe_budget_and_strength/plan.json').read_text())
MODELS = ['sd3', 'flux']
NBOOT = int(os.environ.get('NBOOT', 5000))
SEED = 20260921
rng = np.random.default_rng(SEED)


def spear(x, y):
    x = rankdata(x, axis=-1).astype(float); y = rankdata(y, axis=-1).astype(float)
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    return np.divide((x * y).sum(-1), den, out=np.full_like(den, np.nan), where=den > 1e-12)


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def U(bias, align):
    return 20.0 * float(align) * (5.0 - float(bias))


curve, contrasts, policy = [], [], []
for m in MODELS:
    dep_d, dep_t = PLAN[m]['deployed_dose'], PLAN[m]['deployed_timesteps']
    doses = sorted(set([dep_d] + PLAN[m]['gen_doses']))

    # --- probe cells -> G per (case, dose) ---
    rows = []
    for f in sorted(glob.glob(str(OUT / 'probe' / m / 'shard_*.json'))):
        d = json.loads(Path(f).read_text())
        for k, v in d['cases'].items():
            if v['status'] != 'PRIMARY_VALID':
                continue
            for cc in v['cells']:
                rows.append(dict(key=k, **cc))
    cells = pd.DataFrame(rows).drop_duplicates(['key', 'seed', 'timestep', 'dose'])
    cells = cells[cells.timestep.isin(dep_t)]
    G = cells.pivot_table(index='key', columns='dose', values='cosine', aggfunc='mean')

    # --- Exp271: clean and deployed-strength steered, seeds 3/4 ---
    a = json.loads((RES / 'heldout_prediction/runtime/analysis.json').read_text())
    base = {c['triplet_key']: c for c in a['models'][m]['cases'] if c['status'] == 'PRIMARY_VALID'}

    # --- Exp280 scoring: new strengths ---
    rec = {}
    for f in sorted(glob.glob(str(OUT / 'scoring' / f'scores_{m}_*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if r['value'] is None:
                continue
            rec.setdefault((r['triplet_key'], r['dose'], r['seed']), {})[r['metric']] = r['value']
    steer = {}
    for (k, dose, s), v in rec.items():
        if len(v) == 2:
            steer.setdefault((k, dose), []).append(U(v['bias'], v['alignment']))
    # --- Exp276: extra clean seeds 10-15 for the eight-seed clean mean ---
    extra = {}
    for f in sorted(glob.glob(str(RES / 'seed_ceiling/scoring' / f'scores_{m}_*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if r['value'] is None or r['arm'] != 'clean':
                continue
            extra.setdefault((r['triplet_key'], r['seed']), {})[r['metric']] = r['value']
    clean8 = {}
    for (k, s), v in extra.items():
        if len(v) == 2:
            clean8.setdefault(k, []).append(U(v['bias'], v['alignment']))

    keys = [k for k in G.index if k in base
            and all((k, d) in steer and len(steer[(k, d)]) == 2 for d in doses if d != dep_d)]
    Rd, R8d, comp = {}, {}, {}
    for d in doses:
        if d == dep_d:
            Rd[d] = np.array([base[k]['S']['steer'] - base[k]['S']['clean'] for k in keys])
            comp[d] = np.array([[base[k]['components']['clean']['bias'] - base[k]['components']['steer']['bias'],
                                 100 * (base[k]['components']['steer']['alignment']
                                        - base[k]['components']['clean']['alignment'])] for k in keys])
            s_mean = np.array([base[k]['S']['steer'] for k in keys])
        else:
            s_mean = np.array([np.mean(steer[(k, d)]) for k in keys])
            Rd[d] = s_mean - np.array([base[k]['S']['clean'] for k in keys])
            cb, ca = [], []
            for k in keys:
                vs = [rec[(k, d, s)] for s in (3, 4) if (k, d, s) in rec]
                cb.append(base[k]['components']['clean']['bias'] - np.mean([x['bias'] for x in vs]))
                ca.append(100 * (np.mean([float(x['alignment']) for x in vs])
                                 - base[k]['components']['clean']['alignment']))
            comp[d] = np.column_stack([cb, ca])
        c8 = np.array([np.mean(clean8[k] + [base[k]['S']['clean']] * 0) if k in clean8 else np.nan for k in keys])
        R8d[d] = s_mean - np.where(np.isfinite(c8), c8, np.array([base[k]['S']['clean'] for k in keys]))

    n = len(keys)
    draws = rng.integers(0, n, size=(NBOOT, n))
    Gd = {d: G.loc[keys, d].to_numpy(float) for d in doses}
    for d in doses:
        r_ci = ci([float(spear(Gd[d][i], Rd[d][i])) for i in draws])
        curve.append(dict(model=m, dose=d, deployed=(d == dep_d), n=n,
                          mean_R=float(Rd[d].mean()), mean_R_ci=ci(Rd[d][draws].mean(1)),
                          mean_R_clean8=float(np.nanmean(R8d[d])),
                          bias_reduction=float(comp[d][:, 0].mean()),
                          alignment_change_pp=float(comp[d][:, 1].mean()),
                          rho_G_R=float(spear(Gd[d], Rd[d])), rho_G_R_ci=r_ci,
                          rho_G_R_clean8=float(spear(Gd[d], R8d[d])),
                          mean_G=float(Gd[d].mean()), sd_G=float(Gd[d].std(ddof=1))))
    for d in doses:
        if d == dep_d:
            continue
        diff = [float(spear(Gd[d][i], Rd[d][i]) - spear(Gd[dep_d][i], Rd[dep_d][i])) for i in draws]
        dR = Rd[d] - Rd[dep_d]
        contrasts.append(dict(model=m, dose=d, vs='deployed',
                              d_rho=float(spear(Gd[d], Rd[d]) - spear(Gd[dep_d], Rd[dep_d])), d_rho_ci=ci(diff),
                              d_mean_R=float(dR.mean()), d_mean_R_ci=ci(dR[draws].mean(1))))

    # --- can G choose a per-prompt strength? ---
    Dm = np.array(doses)
    Rmat = np.column_stack([Rd[d] for d in doses])
    Gmat = np.column_stack([Gd[d] for d in doses])
    Gz = (Gmat - Gmat.mean(0)) / Gmat.std(0, ddof=1)      # per-strength z, removes drift with strength
    pick = {'G argmax (raw)': Gmat.argmax(1), 'G argmax (per-strength z)': Gz.argmax(1),
            'oracle': Rmat.argmax(1), 'random': rng.integers(0, len(doses), size=n)}
    best_fixed = int(np.argmax(Rmat.mean(0)))
    strat = {f'fixed alpha={Dm[j]:g}' + (' (deployed)' if Dm[j] == dep_d else ''): Rmat[:, j] for j in range(len(doses))}
    for name, idx in pick.items():
        strat[name] = Rmat[np.arange(n), idx]
    ref = Rmat[:, best_fixed]
    for name, v in strat.items():
        d = v - ref
        policy.append(dict(model=m, strategy=name, mean_R=float(v.mean()), mean_R_ci=ci(v[draws].mean(1)),
                           vs_best_fixed=float(d.mean()), vs_best_fixed_ci=ci(d[draws].mean(1)),
                           share_of_prompts=None if name not in pick else
                           {f'{Dm[j]:g}': float((pick[name] == j).mean()) for j in range(len(doses))}))
    print(f'[{m}] n={n} doses={doses} best fixed alpha={Dm[best_fixed]:g}', flush=True)

cu = pd.DataFrame(curve); co = pd.DataFrame(contrasts); po = pd.DataFrame(policy)
cu.to_csv(OUT / 'b4_dose_curve.csv', index=False)
co.to_csv(OUT / 'b4_contrasts.csv', index=False)
po.to_csv(OUT / 'b4_policy.csv', index=False)
pd.set_option('display.width', 200)
print(); print(cu.to_string(index=False))
print(); print(co.to_string(index=False))
print(); print(po.to_string(index=False))
