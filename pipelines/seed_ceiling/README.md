# Exp276 generation recovery v1

Goal: extend 300 existing cases with clean/steer outcome seeds10–15,3600PNG/model;
retain Exp271 f044dcfd3f6e4052 adapter/config/protocol/cohort byte-for-byte.
Recovery changes execution and validation only. No scoring/ceiling estimate is implemented.

Entry points (run with DG_ROOT set to the project root):
- recovery.py preflight: source/model-config/asset/selection checks, no model load.
- recovery.py generate --model MODEL --stage smoke|formal --shard N --budget SECONDS.
- recovery.py verify --model MODEL --stage smoke|formal --shard N.
- control.py MODEL --start: submit one smoke array and its CPU afterany controller ONCE.
- control.py MODEL: validate current terminal array; advance or bounded retry. Never run concurrently with its scheduled controller.

Fixed25shards/model,12cases each; smoke uses first case and seed10 only in a separate directory.
Three independent chains:2-image smoke ->144-image pilot(shard0) ->24remaining shards(max4concurrent/model).
GPU8CPU/96GB,CPU verifier2CPU/8GB. Preserved conda project environment.

Runtime:results/seed_ceiling/recovery_v2/{smoke,formal}/MODEL/shard_NNN.
Manifests bind release SHA256; images bind SHA256,case/arm/seed,job/GPU and forward audit.
Completed coverage is verified from actual PNGs. Synthetic contract tests use no GPU/model.
Weights are checked for presence/byte sizes; JSON model configs are hashed. This does not certify model-weight identity.

Directories writer.lock use atomic mkdir; only terminal-owner controller can archive stale locks.
Budget exhaustion exits75. Only timeout/preemption/exit75 retries(max3);2no-progress attempts stop.
Controller journal records commands and jobs; do not blindly rerun --start after any submission error.
If sbatch succeeded but controller submission failed, inspect journal/squeue before scheduling the missing controller.
Job source is frozen; fixes after launch require a new release and documented execution amendment.
FINAL_VERIFICATION.json signifies GENERATION completion only; scoring/analysis remain NOT_RUN.
