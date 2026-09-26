# Exp265A measured inference experiment

Run on the project the cluster checkout with the existing project environment.
No model training, external downloads, or synthetic-result fallback is used.
Dependencies are the existing project Qwen loader, encoder/pole-token utilities,
DiT trajectory adapter, and Qwen3-VL loader. Preserve those source snapshots.

1. Review config.json and the experiment's data_contract.md and experiment_design.md.
2. `python common.py` selects text-only cases; `python freeze.py` binds the reviewed exemptions and code.
3. `python -m unittest discover -s pipelines/layer_routing/finite_n -p test_exp265.py -v` runs numerical unit tests.
4. `sbatch gpu.slurm smoke` runs only the nonbenchmark engineering case.
5. After smoke success: generation array, CPU image verifier, finite-N array,
   scoring array, then `sbatch cpu.slurm finalize`, using afterok dependencies.

GPU stages: `python run.py generate --index 0`, `python run.py probe --index 0`,
and `python score.py --index 0`; valid case indices 0–23. Generation and scoring
resume from per-case validated checkpoints. Approximate budget is 15–25 A100
GPU-hours for 1,296 images plus probes and scoring; actual hardware/time is recorded.

Analysis: `python evaluate.py`; independent check: `python verify.py results`.
Figures: `python plot.py`; report: `python report.py`. Matplotlib PDF exports
must be rasterized and inspected before publication delivery. Frozen inference
stages replace the experiment-suite's generic training loop because no training
is in the hypothesis. The project's experiment/result paths take
precedence over the generic output/experiment-suite directory convention.
