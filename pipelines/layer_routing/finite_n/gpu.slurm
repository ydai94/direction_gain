#!/bin/bash
#SBATCH --job-name=exp265_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp265_%x_%A_%a.out
#SBATCH --error=logs/exp265_%x_%A_%a.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export PYTHONPATH="$PWD:$PWD/experiments/interp_program:$PWD/pipelines/layer_routing/finite_n"
export DG_ROOT="$PWD"
export OMP_NUM_THREADS=8
if [ "$1" = score ]; then
  python pipelines/layer_routing/finite_n/score.py --index "${SLURM_ARRAY_TASK_ID:-0}"
else
  python pipelines/layer_routing/finite_n/run.py "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
fi
