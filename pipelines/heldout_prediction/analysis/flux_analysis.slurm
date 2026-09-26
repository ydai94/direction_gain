#!/bin/bash
#SBATCH --job-name=exp271_flux_analysis
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp271_%x_%j.out
#SBATCH --error=logs/exp271_%x_%j.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export DG_ROOT="$PWD"
export PYTHONHASHSEED=20260908
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
release="$1"
export PYTHONPATH="$release:$PWD:$PWD/experiments/interp_program"
export EXP267_REQUIRE_TORCH=1
exp267_python="${PYTHON:-python}"
"$exp267_python" -c 'import sys, torch; print("Python:", sys.executable, "torch:", torch.__version__, flush=True)'
"$exp267_python" "$2" "$release"
