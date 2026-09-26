#!/bin/bash
# Usage: sbatch [site options] exp280 gen.slurm MODEL BUDGET_SECONDS
#SBATCH --job-name=e280gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/exp280_gen_%x_%j.out
#SBATCH --error=logs/exp280_gen_%x_%j.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export DG_ROOT="$PWD"
export PYTHONHASHSEED=20260908
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
release="$PWD/pipelines/probe_budget_and_strength"
export PYTHONPATH="$release:$PWD:$PWD/experiments/interp_program"
py="${PYTHON:-python}"
"$py" -c 'import torch;print("gpu",torch.cuda.get_device_name(0),round(torch.cuda.get_device_properties(0).total_memory/1024**3),"GB",flush=True)'
"$py" "$release/gen.py" "$1" "$2"
