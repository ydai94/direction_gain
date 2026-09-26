# Single-probe selection analysis

This analysis uses the original six Exp271 denoiser measurements and the
Qwen3-VL image scores. It compares top-20% prompt selection using each probe
with selection using all six probes. No images or ratings were generated.

`ANALYSIS_PLAN.md` fixes the selection and resampling rules. `analyze.py`
reconstructs inputs from the project archives, checks shard hashes, and
reproduces the published six-probe results before calculating the new values.

The manuscript source bundle contains a compact copy of the inputs and
results alongside the scripts. To reproduce all point estimates and intervals
from that folder:

```sh
python3 reproduce.py --directory .
python3 verify.py --directory .
```

The first command uses NumPy and SciPy and runs the recorded 10,000 paired
prompt-bootstrap draws per model. The second independently reconstructs all
selection identities and point estimates using Python's standard library.
Both were run on the supplied snapshot. The bootstrap replay matches every
saved model result exactly; the independent verifier passes 324 checks.

The original execution used NumPy 2.3.5 and SciPy 1.16.3. The random seed is
20260913. `results.json` records the input/source hashes and all selected keys.
`inputs.json` contains the six cosines, six response magnitudes and outcomes
for each prompt. The mean over six single-probe configurations averages six
separate selections. `render_tables.py` formats the saved estimates for LaTeX.

The manuscript reports the selected-group mean gain. Its retention ratio is
single-probe mean R divided by six-probe mean R. The results also retain the
gain above random prompt selection and the corresponding retention ratio.
