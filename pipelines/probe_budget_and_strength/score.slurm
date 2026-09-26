#!/bin/bash
# Usage: sbatch [site options] --array=0-N score.slurm MODEL NSHARDS
#SBATCH --job-name=e280score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=logs/exp280_score_%x_%A_%a.out
#SBATCH --error=logs/exp280_score_%x_%A_%a.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
py="${PYTHON:-python}"
"$py" pipelines/probe_budget_and_strength/score.py "$1" "${SLURM_ARRAY_TASK_ID}" "$2"
