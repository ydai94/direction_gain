# Exp274 — fixed top-20% selection and content-retention audit

Plan fixed 2026-09-16. Measured reanalysis
of existing Exp271 outputs; no new generation, fitting or human assessment.
The outcomes have been inspected. This registration freezes this reanalysis
before execution; it does not make these historical data confirmatory.

## Inputs and selection

Use the final verified three-model Exp271 analysis.json and its exact frozen
predictors_frozen.json. All three models use the common outcome-complete,
PRIMARY_VALID cases with G, M, H and A0. Report exclusions and case-key hashes.
Retain each model's native units; never pool raw G across models.

Freeze q=0.20, k=ceil(q*N), descending predictor and ascending triplet_key for
ties. Strategies: G, M, H, B, B_G, B_M, B_G_M. B has input_direction_norm,
pair_separability, T_rel, independent headroom H and baseline alignment A0.
Apply saved development scalers/coefficients without refitting. ALL is the
full-cohort mean and the expectation of uniform random case selection at k;
it is not random-direction steering and its interval does not describe a
single realized random subset. Random-direction outcomes remain in the input.

## Endpoints

Preserve per-image S=20*A*(5-bias), averaged over matched seeds 3/4; R is
steer-minus-clean S. Do not reconstruct S by multiplying component means.
Bias reduction D=mean(bias_clean)-mean(bias_steer). Alignment change is
mean(A_steer)-mean(A_clean), not technical image quality. A is binary per image.
Content pass: mean(A_steer)==1, so both fixed seeds pass. Joint success:
content pass AND D>0, denominator all k selected cases. This is stereotype
reduction, not proof of successful anti-attribute writing.

Report clean/steer bias and alignment, R,D,alignment change, content-pass rate,
joint-success rate, and R/D conditional on content pass. The latter condition
on generated outcomes, are descriptive and are never used to refill selection.
Empty retained groups give null estimates, not zero. Report retained counts.

## Uncertainty and comparisons

10,000 paired prompt bootstrap draws, seed20260913 independently reset for each
model, matching Exp273. Resample complete prompt records and reselect top k in
every draw; all strategies share the same draw. Pointwise percentile 95% CIs
only, no multiplicity-adjusted or family-wide significance claim on these
historically exposed data. Contrasts for every endpoint: G-M, G-H, G-ALL,
B_G-B, B_G_M-B_M, B_G-B_M. Conditional differences omit undefined draws and
report their count. The new-axis prospective family is a separate experiment.

## Verification and artifacts

Independently reconstruct all selection identities, point means and frozen
predictions with stdlib arithmetic. Match existing Exp273 SD3/Qwen top20
identities, effects and bootstrap R intervals. Verify full-cohort ranking
invariance at 100%, source/predictor hashes, explicit missingness and stable
ties. Store selected keys, source hashes, registered rules, code hashes,
results, figures and report. Domain/stats gates before and after execution.
No claim of calibration, unseen-axis transfer, human validation or net compute
savings. Human assessment is deferred.
