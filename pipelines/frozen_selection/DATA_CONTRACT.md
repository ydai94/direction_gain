# Exp274 data contract

Mode: measured, reused project experimental outputs. No synthetic scores,
new generated images, new fitted models, or external human ratings.

Authoritative experiment: Exp271 frozen release f044dcfd3f6e4052. Local input is
the final all-model analysis copied from the completed remote run. Each model
has1314 coverage records,1295 primary valid outcome-complete records and19
non-writable records. Exact source and predictor SHA256 are pinned in results.

Case key is the normalized ordered neutral/stereotype/anti triplet hash.
Do not treat the1831 original rows as independent observations or fresh test
data. Split is held out from new development fits but historically exposed.
Outcomes use matched image seeds3/4; headroom and baseline alignment use
independent clean0/1. Probe seeds6/7/8 are separate from these outcomes.

Read stored per-image-aggregated S and R; never multiply averaged alignment by
averaged bias. Component averages suffice for the prespecified two-seed
content-pass indicator because both binary judgments average to1 iff both pass.
Reused B/B_G/B_M/B_G_M parameters are development-frozen; no per-model search.

Remote reproduction reads the exact hash-matched authoritative analysis and a
byte-identical predictor snapshot in the immutable Exp274 release. It requests
CPU only. All source/output/log files produced by that job were fetched into a
new immutable results/remote_runs directory and recorded in Runs. Original
Exp271 PNGs remain remote.
