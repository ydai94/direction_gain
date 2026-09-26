"""Forward calls needed by G versus generating and judging one clean/steered image pair."""
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

DATA_FILES = {m: DATA / f"test_{m}.csv" for m in ["sd3", "flux", "qwen"]}

cost_rows = []

for model in MODELS:
    test = pd.read_csv(DATA_FILES[model])
    arms = ["U_clean_3", "U_clean_4", "U_steer_3", "U_steer_4", "U_random0_3", "U_random0_4"]
    m = test[test[arms].notna().all(axis=1)]
    m = m[m[["G", "M", "H", "A0"]].notna().all(axis=1)].copy()
    cos_cols = sorted([c for c in m.columns if c.startswith("cos_s")])
    seeds = sorted({int(c.split("_")[1][1:]) for c in cos_cols})
    steps = sorted({int(c.split("_")[2][1:]) for c in cos_cols})
    cst = COST[model]

    g_fwd = len(seeds) * cst["traj_forward_per_seed"] + len(cos_cols) * cst["probe_forward_per_cell"]
    one_seed_fwd = 2 * cst["gen_forward_per_image"]
    cheap_fwd = 1 * cst["traj_forward_per_seed"] + 1 * cst["probe_forward_per_cell"]
    cost_rows.append(dict(model=model, sampler_steps=cst["steps"],
                          G_full_forwards=g_fwd, G_1seed1step_forwards=cheap_fwd,
                          one_seed_baseline_forwards=one_seed_fwd,
                          one_seed_decodes=2, one_seed_judge_calls=4,
                          G_decodes=0, G_judge_calls=0,
                          ratio_Gfull_over_baseline=round(g_fwd / one_seed_fwd, 2),
                          ratio_Gcheap_over_baseline=round(cheap_fwd / one_seed_fwd, 2)))

cost = pd.DataFrame(cost_rows)
cost.to_csv(OUTDIR / "a1_cost_table.csv", index=False)
print(cost.to_string(index=False))