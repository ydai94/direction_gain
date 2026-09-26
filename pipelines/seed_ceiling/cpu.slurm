#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp276r_%x_%j.out
#SBATCH --error=logs/exp276r_%x_%j.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export DG_ROOT="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
release="$PWD/pipelines/seed_ceiling"
"${PYTHON:-python}" -c 'import sys; print("EXECUTABLE",sys.executable,flush=True)'
export PYTHONPATH="$release:$PWD:$PWD/experiments/interp_program"
"${PYTHON:-python}" "$release/control.py" "$1"
