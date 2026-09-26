# Table 1: two judges, 2026-09-24

Objective: add GPT-5.6 ratings alongside the existing Qwen3-VL ratings without changing the test cohort or scoring definition. This is an aggregation of archived scores, not a new experiment or judge run; no hypothesis test or Slurm job was added.

Inputs: `scores_{sd3,flux,qwen}.jsonl`, copied from the cluster `${DG_ROOT}/results/second_judge/`. `cohort_keys.json` is the 1,295 valid keys per generator from manuscript `analysis/single_probe_selection/inputs.json`. `qwen_summary.json` preserves the original Table 1 summary from `figures/inputs/plotted_values.json`. SHA256 hashes are in `results.json`.

Reproduce with `python3 aggregate.py`. Only seeds 3 and 4 and clean/random0/steer are included. Every model has 15,540 unique valid ratings, comprising two metrics on 7,770 images. Bias and alignment image hashes must match. No missing records or duplicate outcome keys. Excluded baseline-seed records: SD3 195, FLUX 180, Qwen 252. Each condition has 2,590 images, equally weighted, equivalent to averaging the two seeds and then the 1,295 prompts. Combined score is computed per image as 20*(5-bias) when alignment is true, otherwise zero; never multiply aggregate means.

GPT combined-score steering-minus-clean means: SD3 4.9420849421, FLUX 6.2934362934, Qwen 4.7490347490, reproducing the existing appendix to its stated precision. All original Qwen3-VL table entries are unchanged. Descriptively both judges show stereotype reduction and reduced prompt matching under steering; no new significance claim.

Manuscript Table 1 uses two stacked judge blocks with unchanged metric/condition columns. Compilation: 22 pages, main text ends page 9, final log has no warnings/overfull boxes. Pages 5 and 6 visually inspected. Domain gate found no known patterns before/after; not a certification of scientific validity.

Reusable lesson: cached judge logs can include baseline seeds outside the outcome comparison; assert exact cohort, arm and seed coverage before producing publication means.
