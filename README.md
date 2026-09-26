# Directional Gain: predicting debiasing benefit from denoiser responses

Anonymous code release for the ICLR 2027 submission
*When Steering Works: Predicting Debiasing Benefit from Denoiser Responses*.

Directional gain `G` measures whether an edit at the text-encoder output moves the
diffusion transformer's velocity prediction in the direction the edit is meant to
go, not just by how much. `G` is computed from a few denoiser forward passes at a
shared latent state, before any image is decoded or judged, and is used to predict
which prompts benefit from steering and to select prompts to steer.

The release covers three text-to-image generators (SD3.5-Medium, FLUX.2-klein-9B,
Qwen-Image), a Qwen3-VL-30B-A3B judge, and a GPT-5.6 second judge.

## Repository layout

```
pipelines/     experiment code, one directory per study (see table below)
experiments/   shared modules imported by the pipelines (model loading, encoder
               edits, DiT hooks); kept under their original package paths
analysis/      manuscript analyses that run on bundled inputs (CPU only)
figures/       figure scripts and the numerical inputs they plot
data/          compact per-prompt measurements (G, M, baselines, per-seed outcomes)
results/       compact outputs (JSON/CSV) of each study
configs/       environment template sourced by the SLURM scripts
```

## What reproduces from this repository alone (CPU, minutes)

| Paper element | Command | Output |
|---|---|---|
| Table 1 (two judges) | `python analysis/table1_dual_judge/aggregate.py` | `analysis/table1_dual_judge/results.json` |
| Single-probe selection and retention (§4.6 cost table, App. B.3) | `cd analysis/single_probe_selection && PYTHONPATH=. python reproduce.py --directory . && python verify.py --directory .` | exact replay of all saved estimates and 10,000-draw bootstrap intervals |
| Prediction reliability, text baseline, image-pair baseline (App. A.5, B.1, B.4) | `python pipelines/text_baselines_and_cost/reliability_and_text_baseline.py` | `results/text_baselines_and_cost/a1_a3_results.json` |
| Partial correlations with one-seed improvement (App. B.4) | `python pipelines/text_baselines_and_cost/partial_correlation.py` | `a1_partial_corr.csv` |
| Probe budget: forward calls vs retained ρ (App. B.3) | `python pipelines/text_baselines_and_cost/probe_budget_ablation.py` | `a1_probe_ablation.csv` |
| Probe cost (§4.6) | `python pipelines/text_baselines_and_cost/cost_table.py` | `a1_cost_table.csv` |
| ρ(G, R) vs number of generation seeds (App. B.3) | `python pipelines/seed_ceiling/analyze_reliability.py` | `results/seed_ceiling/seed_curve.csv`, `reliability.csv` |
| Figures | `python figures/build_revision_figures.py`, `build_policy.py`, `build_predicting_cost.py`, `build_figures.py` | PDF/PNG in `figures/` |

All of the above were re-run on the bundled data before release and match the
saved values exactly (NumPy/SciPy/scikit-learn/pandas; see `requirements.txt`).

`data/per_prompt/test_<model>.csv` holds, for each of the 1,314 held-out candidate prompts per
generator (1,295 with complete outcomes): `G`, `M`, the six probe cosines/magnitudes (`cos_s{seed}_t{index}`,
`mag_s…`), the text features (`input_direction_norm`, `pair_separability`,
`T_rel`), clean-image headroom `H` and alignment `A0`, per-arm/per-seed image
scores `U_{clean,steer,random0}_{3,4}`, and the outcome `R`. `dev_<model>.csv`
holds the development cases used to fit the frozen predictors.
`data/seed_ceiling_scores/` holds the Qwen3-VL ratings for the six extra
generation seeds (300 prompts per generator).

## Full pipelines (GPU)

| Directory | Paper section | Contents |
|---|---|---|
| `pipelines/heldout_prediction` | §4.1–4.2, §4.4, App. A, B.1 | Frozen held-out run: image generation (clean / steered / norm-matched random edit), `G`/`M` probes (`g_probe.py`), clean-image headroom (`headroom.py`), Qwen3-VL scoring (`score_engine.py`), frozen ridge predictors (`freeze_predictors.py`, `predictors_frozen.json`), finalisation and bootstrap (`finalize.py`) |
| `pipelines/composite_outcome` | §4.1 (outcome `R`) | Development-set analysis that fixes the combined score `S = 20·aligned·(5 − bias)` |
| `pipelines/selection_qwen`, `pipelines/frozen_selection` | §4.3, App. B.2 | Top-k selection by `G` vs `M`, group means, fractions, stereotype/matching trade-off |
| `pipelines/selective_policy` | §4.3 (Fig. policy frontier), App. B.2 | Risk–coverage curves, AURC, fresh-seed selection |
| `pipelines/second_judge` | §4.1–4.2 (Table 1), App. B.1 | GPT-5.6 re-judging of the same images (needs `OPENAI_API_KEY` in `.env`) |
| `pipelines/seed_ceiling` | §4.6, App. B.3 | Six extra generation seeds on a fixed 300-prompt subset, scoring, reliability analysis |
| `pipelines/probe_budget_and_strength` | §4.5, App. B.5 | Probe design variants (reference, statistic, timesteps) and four steering strengths |
| `pipelines/text_baselines_and_cost` | §4.6, App. B.1, B.3, B.4 | Per-prompt table extraction and CPU analyses listed above |
| `pipelines/layer_routing` | App. C | Choosing the edit layer per prompt (gradient screen + finite-difference check) |

The three generation pipelines (`heldout_prediction`, `seed_ceiling`,
`probe_budget_and_strength`) carry their frozen prompt cohort (`cohort.json`),
protocol/config files and the exact generator adapter used in the paper;
`heldout_prediction` also holds the frozen development null calibration
(`development_calibration.json`) and predictors. The other directories are
analysis or scoring stages that read the outputs of these runs (and the
heldout release files) from `$DG_ROOT/results` and `pipelines/heldout_prediction`. The SLURM scripts are templates: they expect the environment variables
below, and a `logs/` directory in the repository root.

```sh
export DG_ROOT=$PWD            # repository root; all outputs go to $DG_ROOT/results
export MODEL_ROOT=/path/to/models   # local model directories / HF cache
cp configs/env.sh.example configs/env.sh   # edit to activate your environment
mkdir -p logs
sbatch pipelines/heldout_prediction/gpu.slurm "$DG_ROOT/pipelines/heldout_prediction" generate sd3 0
```

Model weights are not included. Expected local paths are
`$MODEL_ROOT/stable-diffusion-3.5-medium`, `$MODEL_ROOT/Qwen-Image`, and the
Hugging Face cache layout for `black-forest-labs/FLUX.2-klein-9B` and
`Qwen/Qwen3-VL-30B-A3B-Instruct`. Generation used 1× A100 80GB (Qwen-Image,
Qwen3-VL) or ≥40GB (FLUX) / ≥23GB (SD3.5) per job.

### Fixed settings (from `pipelines/heldout_prediction/config.json`, `protocol.json`)

- Edit: additive anti-minus-stereotype vector on the first two changed rows of the text-encoder output, at lexical anchors; random control is a norm-matched random direction.
- SD3.5: 28 steps, CFG 4, strength 2, probe indices 6 and 10. FLUX.2-klein: 8 steps, native guidance, strength 1, probe indices 2 and 3. Qwen-Image: 50 steps, normalised CFG 4, strength 2, probe indices 10 and 16.
- Probe seeds 6/7/8; outcome image seeds 3/4; headroom seeds 0/1.
- `G = (C − μ_null)/σ_null`, where `C` is the mean of six cosines between the edited-minus-neutral response and the anti-minus-stereotype reference response; nulls are four norm-matched random edits per development case, calibrated once on development cases and frozen.
- Bootstrap: 10,000 prompt resamples, seed 20260912 (single-probe analysis: 20260913).

## Notes on this snapshot

- The code is a snapshot of the scripts that produced the paper's numbers. Internal
  study identifiers (e.g. `Exp271`) remain in comments, seeds, and record fields;
  they are part of the frozen seeding (`stable_seed`) and are kept unchanged. The
  mapping is: Exp270 composite outcome, Exp271 held-out prediction, Exp273/274
  selection, Exp276 extra seeds, Exp277 second judge, Exp278/279 selective policy
  and fresh-seed selection, Exp280 probe design and strength, Exp261/265 layer routing.
- Site-specific paths were replaced by `DG_ROOT` / `MODEL_ROOT`, and scheduler
  accounts/partitions were removed. Provenance hashes recorded in result files
  (`binding`, `plan_sha256`, `release.json`) refer to the original bytes, so resume
  and integrity checks against old runs will report a mismatch; fresh runs are
  unaffected.
- Generated images are not included (about 60 GB). `figures/qualitative_assets/` contains
  the twelve images shown in the qualitative figure.

## License

MIT (see `LICENSE`).
