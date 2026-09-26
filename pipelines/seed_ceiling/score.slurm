#!/bin/bash
# Usage: sbatch [site options] --qos=... --array=0-N score.slurm MODEL NSHARDS
#SBATCH --job-name=e276score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/exp276s_%x_%A_%a.out
#SBATCH --error=logs/exp276s_%x_%A_%a.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
py="${PYTHON:-python}"
"$py" -c 'import torch;print("gpu",torch.cuda.get_device_name(0),round(torch.cuda.get_device_properties(0).total_memory/1024**3),"GB",flush=True)'
"$py" pipelines/seed_ceiling/score.py "$1" "${SLURM_ARRAY_TASK_ID}" "$2"
