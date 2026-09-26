"""Exp278 (B2): G-gated selective steering -- risk-coverage / AURC and bias-fidelity frontier.
Exp279 (B3-lite): frozen top-20% selection rule evaluated on fresh outcome seeds (Exp276, seeds 10-15).

Zero new generation. Inputs (all existing):
  results/heldout_prediction/runtime/analysis.json                       per-case G, M, R, arm S, components (Qwen3-VL judge)
  results/heldout_prediction/runtime/<m>/scoring/provenance/run_*/predictors_frozen.json   dev-fitted ridge coefficients (B, B+G)
  results/second_judge/scores_<m>.jsonl                          GPT-5.6 rescoring of the same images (second judge)
  results/dev_calibration/runtime/development_g_primary/<m>/calibration.json   dev G, M
  results/composite_outcome/results.json                              dev R, arm S
  results/seed_ceiling/scoring/scores_<m>_*.jsonl             fresh seeds 10-15, 300-case cohort (Qwen3-VL judge)
  pipelines/heldout_prediction/cohort.json   bias_type / head for cluster bootstrap
"""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.environ.get("DG_ROOT", ".")
RES = f"{ROOT}/results/interp_program"
OUT278 = f"{RES}/selective_policy"; OUT279 = f"{RES}/fresh_seed_selection"
os.makedirs(OUT278, exist_ok=True); os.makedirs(OUT279, exist_ok=True)
MODELS = ["sd3", "flux", "qwen"]
OLD_SEEDS = [3, 4]; NEW_SEEDS = [10, 11, 12, 13, 14, 15]
NBOOT = int(os.environ.get("NBOOT", 5000)); SEED = 20260920
FRAC = 0.20
rng = np.random.default_rng(SEED)

def U_of(bias, align): return 20.0 * float(align) * (5.0 - float(bias))
def ci(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
def spear(x, y):
    from scipy.stats import spearmanr; return float(spearmanr(x, y).statistic)

# ----------------------------------------------------------------------------- loaders
analysis = json.load(open(f"{RES}/heldout_prediction/runtime/analysis.json"))
cohort = json.load(open(f"{ROOT}/pipelines/heldout_prediction/cohort.json"))
cmeta = {}
for e in cohort:
    k = e.get("triplet_key") or e.get("key") or e.get("id")
    cmeta[k] = dict(bias_type=e.get("bias_type"), head=e.get("head"))

def frozen_fits(m):
    f = sorted(glob.glob(f"{RES}/heldout_prediction/runtime/{m}/scoring/provenance/run_*/predictors_frozen.json"))[-1]
    return json.load(open(f))["models"][m]["fits"]

def predict(fit, df):
    X = np.column_stack([df[c].to_numpy(float) for c in fit["features"]])
    Z = (X - np.array(fit["mean"])) / np.array(fit["scale"])
    return Z @ np.array(fit["coef"]) + fit["intercept"]

def test_table(m):
    rows = []
    for c in analysis["models"][m]["cases"]:
        if c["status"] != "PRIMARY_VALID": continue
        comp = c["components"]
        rows.append(dict(key=c["triplet_key"], G=c["G"], M=c["M"], R=c["R"],
                         S_clean=c["S"]["clean"], S_steer=c["S"]["steer"], S_rand=c["S"]["random0"],
                         al_clean=comp["clean"]["alignment"], al_steer=comp["steer"]["alignment"],
                         bias_clean=comp["clean"]["bias"], bias_steer=comp["steer"]["bias"],
                         input_direction_norm=c["input_direction_norm"], pair_separability=c["pair_separability"],
                         T_rel=c["T_rel"], H=c["H"], A0=c["A0"]))
    df = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
    fits = frozen_fits(m)
    df["B"] = predict(fits["B"], df); df["BG"] = predict(fits["B_G"], df)
    df["bias_type"] = [cmeta.get(k, {}).get("bias_type") for k in df.key]
    df["head"] = [cmeta.get(k, {}).get("head") for k in df.key]
    assert len(df) == 1295, (m, len(df))
    assert abs(spear(df.G, df.R) - analysis["models"][m]["G_M"]["G"]["rho"]) < 1e-6
    return df

def gpt_table(m):
    d = {}
    for l in open(f"{RES}/second_judge/scores_{m}.jsonl"):
        r = json.loads(l)
        if r["value"] is None or r["seed"] not in OLD_SEEDS or r["arm"] not in ("clean", "steer", "random0"): continue
        d.setdefault((r["triplet_key"], r["arm"], r["seed"]), {})[r["metric"]] = r["value"]
    per = {}
    for (k, arm, s), v in d.items():
        if len(v) == 2: per.setdefault((k, arm), []).append((U_of(v["bias"], v["alignment"]), float(v["bias"]), float(v["alignment"])))
    rows = {}
    for (k, arm), vals in per.items():
        if len(vals) != 2: continue
        a = np.array(vals); rows.setdefault(k, {})[arm] = a.mean(0)
    out = []
    for k, arms in rows.items():
        if not {"clean", "steer"} <= set(arms): continue
        out.append(dict(key=k, S_clean=arms["clean"][0], S_steer=arms["steer"][0], R=arms["steer"][0] - arms["clean"][0],
                        bias_clean=arms["clean"][1], bias_steer=arms["steer"][1], al_clean=arms["clean"][2], al_steer=arms["steer"][2]))
    return pd.DataFrame(out)

def dev_table(m):
    cal = json.load(open(f"{RES}/dev_calibration/runtime/development_g_primary/{m}/calibration.json"))
    g = {r["triplet_key"]: r for r in cal["rows"] if r["status"] == "PRIMARY_VALID"}
    dev = json.load(open(f"{RES}/composite_outcome/results.json"))["models"][m]["cases"]
    rows = []
    for c in dev:
        k = c["triplet_key"]
        if k not in g or c.get("operator_status") not in (None, "WRITABLE"): continue
        rows.append(dict(key=k, G=g[k]["G"], M=g[k]["M"], R=float(c["R"])))
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- policy machinery
def curves(score, R, dbias, dalign, order_key):
    """Rank by score (desc; deterministic tie-break by key). Return per-coverage arrays over k=1..n."""
    n = len(R)
    idx = np.lexsort((order_key, -score))              # primary: -score, secondary: key
    Rs, bs, als = R[idx], dbias[idx], dalign[idx]
    k = np.arange(1, n + 1)
    gain = np.cumsum(Rs) / n                            # benchmark-mean improvement over steer-none
    risk = np.cumsum(Rs < 0) / k                        # selective risk: P(worse | steered)
    harm = np.cumsum(np.maximum(-Rs, 0)) / k            # mean harm magnitude among steered
    bias_red = np.cumsum(bs) / n                        # benchmark-mean stereotype-rating reduction
    align_loss = np.cumsum(als) / n                     # benchmark-mean alignment-rate loss (pp/100)
    return dict(cov=k / n, gain=gain, risk=risk, harm=harm, bias_red=bias_red, align_loss=align_loss)

def aurc(c): return float(np.mean(c["risk"]))
def auharm(c): return float(np.mean(c["harm"]))
def area_gain(c): return float(np.mean(c["gain"]))    # area under gain-coverage; higher = gains front-loaded

GATES = ["G", "M", "B", "BG", "oracle"]

def eval_all(df, Rcol="R", prefix=""):
    R = df[Rcol].to_numpy(float)
    dbias = (df["bias_clean"] - df["bias_steer"]).to_numpy(float)
    dalign = (df["al_clean"] - df["al_steer"]).to_numpy(float)
    key = df["key"].to_numpy()
    scores = {g: df[g].to_numpy(float) for g in ["G", "M", "B", "BG"]}; scores["oracle"] = R.copy()
    return {g: curves(s, R, dbias, dalign, key) for g, s in scores.items()}, R, dbias, dalign

def frontier_table(cv, R, model, judge):
    rows = []; n = len(R); p_worse = float(np.mean(R < 0))
    for g, c in cv.items():
        for q in [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]:
            i = int(np.ceil(q * n)) - 1
            rows.append(dict(model=model, judge=judge, gate=g, coverage=q, n_steered=i + 1,
                             mean_gain_per_steered=c["gain"][i] * n / (i + 1), benchmark_gain=c["gain"][i],
                             selective_risk=c["risk"][i], mean_harm=c["harm"][i],
                             bias_reduction=c["bias_red"][i], alignment_loss_pp=100 * c["align_loss"][i]))
    # random gate expectation at each coverage
    for q in [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]:
        i = int(np.ceil(q * n)) - 1
        rows.append(dict(model=model, judge=judge, gate="random", coverage=q, n_steered=i + 1,
                         mean_gain_per_steered=R.mean(), benchmark_gain=R.mean() * (i + 1) / n,
                         selective_risk=p_worse, mean_harm=float(np.maximum(-R, 0).mean()),
                         bias_reduction=cv["G"]["bias_red"][-1] * (i + 1) / n, alignment_loss_pp=100 * cv["G"]["align_loss"][-1] * (i + 1) / n))
    return pd.DataFrame(rows)

def summary_and_boot(df, Rcol, model, judge):
    cv, R, dbias, dalign = eval_all(df, Rcol)
    n = len(R); p_worse = float(np.mean(R < 0))
    point = {g: dict(AURC=aurc(c), AUHarm=auharm(c), AreaGain=area_gain(c)) for g, c in cv.items()}
    point["random"] = dict(AURC=p_worse, AUHarm=float(np.maximum(-R, 0).mean()), AreaGain=float(R.mean() * (n + 1) / (2 * n)))
    # paired case bootstrap of gate differences
    key = df["key"].to_numpy(); S = {g: df[g].to_numpy(float) for g in ["G", "M", "B", "BG"]}
    draws = rng.integers(0, n, size=(NBOOT, n))
    acc = {k: [] for k in ["G-M", "G-B", "G-BG", "G-random", "BG-B", "M-random"]}
    accg = {k: [] for k in ["G-M", "G-B", "G-random", "BG-B"]}
    for row in draws:
        Rb, bb, ab, kb = R[row], dbias[row], dalign[row], key[row]
        cb = {g: curves(S[g][row], Rb, bb, ab, kb) for g in S}
        a = {g: aurc(cb[g]) for g in cb}; a["random"] = float(np.mean(Rb < 0))
        ag = {g: area_gain(cb[g]) for g in cb}; ag["random"] = float(Rb.mean() * (n + 1) / (2 * n))
        for k in acc: x, y = k.split("-"); acc[k].append(a[x] - a[y])
        for k in accg: x, y = k.split("-"); accg[k].append(ag[x] - ag[y])
    diffs = {k: dict(point=point[k.split("-")[0]]["AURC"] - point[k.split("-")[1]]["AURC"], ci95=ci(v)) for k, v in acc.items()}
    diffs_gain = {k: dict(point=point[k.split("-")[0]]["AreaGain"] - point[k.split("-")[1]]["AreaGain"], ci95=ci(v)) for k, v in accg.items()}
    return cv, point, diffs, diffs_gain, frontier_table(cv, R, model, judge)

# ----------------------------------------------------------------------------- Exp278
res278 = dict(experiment="Exp278", scope="Selective steering policy on existing Exp271 test outcomes; no new generation; thresholds swept (curves) and one threshold frozen on development cases",
              seed=SEED, n_boot=NBOOT, definitions=dict(
                  policy="steer prompt i iff score_i >= tau; otherwise return the clean image",
                  coverage="fraction of prompts steered", selective_risk="P(R<0 | steered) with R the two-seed composite improvement",
                  AURC="mean selective risk over coverage grid k/n, k=1..n (lower is better)",
                  AreaGain="mean over coverage of benchmark-mean improvement (higher = gains concentrated at low coverage)",
                  random_gate="analytic expectation: risk constant at P(R<0), gain linear in coverage"),
              models={})
frontier_rows = []; curve_rows = []
for m in MODELS:
    t = test_table(m); g = gpt_table(m)
    tg = t[["key", "G", "M", "B", "BG"]].merge(g, on="key", how="inner")
    dev = dev_table(m)
    mres = {}
    for judge, df in [("qwen3vl", t), ("gpt56", tg)]:
        cv, point, diffs, diffs_gain, ft = summary_and_boot(df, "R", m, judge)
        frontier_rows.append(ft)
        for gname, c in cv.items():
            for i in range(0, len(c["cov"]), max(1, len(c["cov"]) // 260)):
                curve_rows.append(dict(model=m, judge=judge, gate=gname, cov=c["cov"][i], gain=c["gain"][i], risk=c["risk"][i],
                                       bias_red=c["bias_red"][i], align_loss_pp=100 * c["align_loss"][i]))
        # frozen threshold from development: G quantile at 20% coverage (and M likewise)
        R = df["R"].to_numpy(float)
        frozen = {}
        for gate in ["G", "M"]:
            tau = float(np.quantile(dev[gate], 1 - FRAC))
            sel = df[gate].to_numpy(float) >= tau
            base_rows = df[sel]
            # random gate at same realized coverage: expectation = full-set means
            frozen[gate] = dict(tau=tau, n_steered=int(sel.sum()), coverage=float(sel.mean()),
                                mean_gain_per_steered=float(R[sel].mean()), benchmark_gain=float(R[sel].sum() / len(R)),
                                selective_risk=float((R[sel] < 0).mean()),
                                bias_reduction_steered=float((base_rows.bias_clean - base_rows.bias_steer).mean()),
                                alignment_change_pp_steered=float(100 * (base_rows.al_steer - base_rows.al_clean).mean()))
        # paired bootstrap for frozen G vs frozen M (per-steered gain) and G vs random
        n = len(df); draws = rng.integers(0, n, size=(NBOOT, n))
        Gv, Mv = df["G"].to_numpy(float), df["M"].to_numpy(float); tG, tM = frozen["G"]["tau"], frozen["M"]["tau"]
        dGM, dGr, rGM = [], [], []
        for row in draws:
            Rb = R[row]; sg = Gv[row] >= tG; sm = Mv[row] >= tM
            if sg.sum() == 0 or sm.sum() == 0: continue
            dGM.append(Rb[sg].mean() - Rb[sm].mean()); dGr.append(Rb[sg].mean() - Rb.mean()); rGM.append((Rb[sg] < 0).mean() - (Rb[sm] < 0).mean())
        frozen["G_minus_M_gain_per_steered"] = dict(point=frozen["G"]["mean_gain_per_steered"] - frozen["M"]["mean_gain_per_steered"], ci95=ci(dGM))
        frozen["G_minus_random_gain_per_steered"] = dict(point=frozen["G"]["mean_gain_per_steered"] - float(R.mean()), ci95=ci(dGr))
        frozen["G_minus_M_selective_risk"] = dict(point=frozen["G"]["selective_risk"] - frozen["M"]["selective_risk"], ci95=ci(rGM))
        frozen["steer_all"] = dict(mean_gain=float(R.mean()), selective_risk=float((R < 0).mean()),
                                   bias_reduction=float((df.bias_clean - df.bias_steer).mean()), alignment_change_pp=float(100 * (df.al_steer - df.al_clean).mean()))
        # does any coverage beat steer-all on benchmark-mean composite?
        best = {gname: dict(max_benchmark_gain=float(c["gain"].max()), at_coverage=float(c["cov"][int(np.argmax(c["gain"]))])) for gname, c in cv.items()}
        mres[judge] = dict(n=int(n), P_worse=float((R < 0).mean()), point=point, AURC_differences=diffs, AreaGain_differences=diffs_gain,
                           frozen_dev_threshold=frozen, best_coverage_for_total_gain=best,
                           rho_check=dict(G=spear(df.G, R), M=spear(df.M, R), B=spear(df.B, R), BG=spear(df.BG, R)))
        print(f"[278] {m} {judge}: n={n} P(worse)={mres[judge]['P_worse']:.3f} AURC G={point['G']['AURC']:.3f} M={point['M']['AURC']:.3f} B={point['B']['AURC']:.3f} BG={point['BG']['AURC']:.3f} oracle={point['oracle']['AURC']:.3f} | G-M {diffs['G-M']['point']:+.3f} {diffs['G-M']['ci95']} | frozen20 G gain/steered {frozen['G']['mean_gain_per_steered']:.2f} cov {frozen['G']['coverage']:.2f}", flush=True)
    res278["models"][m] = mres
pd.concat(frontier_rows).to_csv(f"{OUT278}/frontier_points.csv", index=False)
pd.DataFrame(curve_rows).to_csv(f"{OUT278}/curves.csv", index=False)
json.dump(res278, open(f"{OUT278}/results.json", "w"), indent=1)

# ----------------------------------------------------------------------------- Exp279
res279 = dict(experiment="Exp279", scope="Frozen top-20% selection rule (Exp271 test-set G cutoff, fixed before Exp276 generation) evaluated on fresh outcome seeds 10-15 of the 300-case Exp276 cohort; Qwen3-VL judge; no new generation",
              seed=SEED, n_boot=NBOOT, models={})
rows279 = []
for m in MODELS:
    t = test_table(m).set_index("key")
    recs = {}
    for f in sorted(glob.glob(f"{RES}/seed_ceiling/scoring/scores_{m}_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["value"] is None: continue
            recs[(r["triplet_key"], r["arm"], r["seed"], r["metric"])] = r["value"]   # dedupe duplicate shards
    per = {}
    for (k, arm, s, met), v in recs.items(): per.setdefault((k, arm, s), {})[met] = v
    U = {k: (U_of(v["bias"], v["alignment"]), float(v["bias"]), float(v["alignment"])) for k, v in per.items() if len(v) == 2}
    keys = sorted({k for (k, _, _) in U})
    out = []
    for k in keys:
        if k not in t.index: continue
        st = [U.get((k, "steer", s)) for s in NEW_SEEDS]; cl = [U.get((k, "clean", s)) for s in NEW_SEEDS]
        if any(x is None for x in st + cl): continue
        st, cl = np.array(st), np.array(cl)
        out.append(dict(key=k, R_new=st[:, 0].mean() - cl[:, 0].mean(), dbias_new=cl[:, 1].mean() - st[:, 1].mean(),
                        dalign_new_pp=100 * (st[:, 2].mean() - cl[:, 2].mean()), R_old=t.loc[k, "R"], G=t.loc[k, "G"], M=t.loc[k, "M"],
                        B=t.loc[k, "B"], BG=t.loc[k, "BG"], head=t.loc[k, "head"], bias_type=t.loc[k, "bias_type"]))
    d = pd.DataFrame(out); n = len(d)
    # frozen cutoffs = full test-set top-20% values (259th ranked), fixed before Exp276 existed
    cut = {g: float(np.sort(t[g].to_numpy(float))[::-1][int(np.ceil(FRAC * len(t))) - 1]) for g in ["G", "M", "B", "BG"]}
    sel = {g: d[g].to_numpy(float) >= cut[g] for g in cut}
    # within-cohort top-20% (60 cases) as secondary
    kq = int(np.ceil(FRAC * n)); sel_q = {g: np.zeros(n, bool) for g in cut}
    for g in cut: sel_q[g][np.lexsort((d.key.to_numpy(), -d[g].to_numpy(float)))[:kq]] = True
    Rn, Ro = d.R_new.to_numpy(float), d.R_old.to_numpy(float)
    clusters = d["head"].fillna(d["key"]).to_numpy(); uniq = np.unique(clusters); cid = np.searchsorted(uniq, clusters)
    members = [np.where(cid == i)[0] for i in range(len(uniq))]
    def boot_idx(cluster):
        if cluster:
            pick = rng.integers(0, len(uniq), size=len(uniq)); return np.concatenate([members[i] for i in pick])
        return rng.integers(0, n, size=n)
    def contrasts(selmap, Rv):
        outc = {}
        for name, (a, b) in {"G-M": ("G", "M"), "G-all": ("G", None), "G-B": ("G", "B"), "BG-B": ("BG", "B"), "M-all": ("M", None)}.items():
            pt = Rv[selmap[a]].mean() - (Rv[selmap[b]].mean() if b else Rv.mean())
            bc, bi = [], []
            for _ in range(NBOOT):
                for cluster, store in [(True, bc), (False, bi)]:
                    ii = boot_idx(cluster); sa = selmap[a][ii]
                    if sa.sum() == 0: continue
                    ref = Rv[ii][selmap[b][ii]].mean() if b else Rv[ii].mean()
                    if b and selmap[b][ii].sum() == 0: continue
                    store.append(Rv[ii][sa].mean() - ref)
            outc[name] = dict(point=float(pt), ci95_cluster=ci(bc), ci95_case=ci(bi))
        return outc
    mres = dict(n=int(n), n_clusters=int(len(uniq)), cluster_var="head (target concept) from cohort.json",
                frozen_cutoffs=cut, n_selected_frozen={g: int(s.sum()) for g, s in sel.items()},
                rho_new=dict(G=spear(d.G, Rn), M=spear(d.M, Rn), B=spear(d.B, Rn), BG=spear(d.BG, Rn)),
                rho_old_same_cases=dict(G=spear(d.G, Ro), M=spear(d.M, Ro)),
                all_cases=dict(mean_R_new=float(Rn.mean()), mean_R_old=float(Ro.mean()), bias_reduction_new=float(d.dbias_new.mean()), alignment_change_pp_new=float(d.dalign_new_pp.mean())),
                frozen_threshold=dict(groups={g: dict(n=int(s.sum()), mean_R_new=float(Rn[s].mean()), mean_R_old=float(Ro[s].mean()),
                                                      bias_reduction_new=float(d.dbias_new[s].mean()), alignment_change_pp_new=float(d.dalign_new_pp[s].mean())) for g, s in sel.items()},
                                      contrasts_R_new=contrasts(sel, Rn), contrasts_R_old=contrasts(sel, Ro)),
                within_cohort_top20=dict(k=kq, groups={g: dict(mean_R_new=float(Rn[s].mean()), bias_reduction_new=float(d.dbias_new[s].mean()), alignment_change_pp_new=float(d.dalign_new_pp[s].mean())) for g, s in sel_q.items()},
                                         contrasts_R_new=contrasts(sel_q, Rn)))
    res279["models"][m] = mres
    d.to_csv(f"{OUT279}/cases_{m}.csv", index=False)
    f = mres["frozen_threshold"]
    print(f"[279] {m}: n={n} clusters={len(uniq)} sel G={f['groups']['G']['n']} M={f['groups']['M']['n']} | R_new all {Rn.mean():.2f} G {f['groups']['G']['mean_R_new']:.2f} M {f['groups']['M']['mean_R_new']:.2f} | G-M {f['contrasts_R_new']['G-M']['point']:+.2f} cl{f['contrasts_R_new']['G-M']['ci95_cluster']} | G-all {f['contrasts_R_new']['G-all']['point']:+.2f} cl{f['contrasts_R_new']['G-all']['ci95_cluster']} | rho_new G {mres['rho_new']['G']:.3f} M {mres['rho_new']['M']:.3f}", flush=True)
json.dump(res279, open(f"{OUT279}/results.json", "w"), indent=1)
print("DONE")
