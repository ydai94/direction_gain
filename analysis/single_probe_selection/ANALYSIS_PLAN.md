# Single-probe selection, 2026-09-23

Execute the manuscript revision using cached Exp271 probes and
Qwen3-VL image outcomes. This is a supplementary analysis of existing data.
Freeze this specification before calculating the new selection results.

For each of SD3.5, FLUX and Qwen, retain the same 1,295 PRIMARY_VALID cases as
Exp274. Join records by triplet_key. Require all six unique seed/timestep
cells per case, with seeds 6, 7, 8 and the two original model-specific steps.
Verify shard completion hashes and reproduce the original G/M correlations,
six-probe top-20% identities and mean gains before reporting new estimates.

Rank each of the six single-probe cosines separately, and rank the magnitude
at each matching probe separately. Also retain the original six-probe G and
M rankings. Fixed positive affine calibration does not change these rankings.
Select ceil(0.20*N)=259 cases in descending score order, breaking ties by
ascending triplet_key. Do not tune a probe or select a best-performing cell.

Use the saved R (per-image composite difference, averaged over outcome seeds
3 and 4). Secondary outcomes are clean-minus-steered stereotype rating and
steered-minus-clean prompt matching. ALL is the expectation of random prompt
selection, not random-direction steering. Report six-probe results and each
single-probe result, plus the mean and min/max across the six configurations.
The configuration mean is an average of six separately selected-group means;
it is not selection after averaging scores (which recovers the six-probe arm).

Use 10,000 paired prompt bootstrap draws, resetting seed 20260913 per model,
and reselect the top 259 in each draw for every strategy. Average the six
configuration estimates within each draw before calculating its percentile
interval. Report paired single-probe-mean minus six-probe and single-probe-G
minus matching-single-probe-M contrasts. Use pointwise 95% intervals without
hypothesis tests or claims of family-wise significance. Report the ratio to
six-probe mean R as descriptive only, alongside absolute gains and gain over
ALL. Do not reuse the existing retained-correlation percentage as retained R.

Archive source hashes, a compact per-case input snapshot, selection keys,
configuration identities, results and an independent point-estimate verifier.
No model fitting, image generation, image judging or Slurm submission.
