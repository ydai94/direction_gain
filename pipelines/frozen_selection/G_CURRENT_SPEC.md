# Current fixed-output G and top-20% selection specification

Version: 2026-09-16. Historical G_SPEC.md v1 remains immutable and describes
Exp243. This document describes Exp271/274 and the intended new-axis transfer.

G=(C-mu_null)/sigma_null, where C is the arithmetic mean of six cosines of
edited-minus-neutral denoiser response and anti-minus-stereotype reference at
the same pre-forward latent. Probe seeds6/7/8, two fixed model-specific indices.
Nulls are four norm-matched random writes per development case, each averaged
over its six cells; calibrate on the current development cohort with sample
SD ddof1, freeze and reuse on all subsequent cohorts. Refer to the exact saved
development_calibration.json for parameters and source-calibration hashes;
the original development calibration.json files store n_null. The historical
239-case calibration is not the current development calibration.

G has no response-magnitude multiplier. M=mean(response L2 norm) is separate.
Positive model-wide affine calibration preserves C/G ranks and top20 identities.
Do not claim directional G outperforms cosine ranking. C*M and fitted B_G_M
are different scores. G requires model forward evaluations before decoding;
pre-generation does not mean free, prompt-only or zero generation compute.

Native additive encoder-output write uses the existing lexical anchors and
first-two-changed-row anti-minus-stereotype vector. SD3 uses28steps/CFG4/dose2/
probe6,10; FLUX Klein9B uses8steps/native guidance/dose1/probe2,3; Qwen uses
50steps/normalizedCFG4/dose2/probe10,16. Model-native doses are not physically
equal across architectures. Original checkpoint/precision/config hashes apply.

Current outcome: per-image S=20*A*(5-bias), averaged across seeds3/4; R is
steer-clean. Independent baseline/headroom uses seeds0/1. Current selection
is descending G top ceil(.20*N), key-based ties, per model/cohort. This rank
policy is distinct from historical G>2. Both historical endpoint and G>2
remain labeled by their original experiment. No automatic human-quality or
successful anti-attribute writing interpretation is attached to S or R.
