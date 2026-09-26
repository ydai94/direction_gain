#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/exp276r_%x_%A_%a.out
#SBATCH --error=logs/exp276r_%x_%A_%a.err
#SBATCH --no-requeue
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export DG_ROOT="$PWD"
export PYTHONHASHSEED=20260908 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
release="$PWD/pipelines/seed_ceiling"
"${PYTHON:-python}" -c 'import sys; print("EXECUTABLE",sys.executable,flush=True)'
export PYTHONPATH="$release:$PWD:$PWD/experiments/interp_program"
"${PYTHON:-python}" "$release/recovery.py" preflight
"${PYTHON:-python}" "$release/recovery.py" generate --model "$1" --stage "$2" --budget "$3" --shard "$SLURM_ARRAY_TASK_ID"
