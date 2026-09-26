#!/bin/bash
#SBATCH --job-name=exp265_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/exp265_%x_%j.out
#SBATCH --error=logs/exp265_%x_%j.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export PYTHONPATH="$PWD:$PWD/experiments/interp_program:$PWD/pipelines/layer_routing/finite_n"
export DG_ROOT="$PWD"
export OMP_NUM_THREADS=4
if [ "$1" = images ]; then
  python pipelines/layer_routing/finite_n/verify.py images
elif [ "$1" = finalize ]; then
  python pipelines/layer_routing/finite_n/evaluate.py
  python pipelines/layer_routing/finite_n/verify.py results
  python pipelines/layer_routing/finite_n/plot.py
  python pipelines/layer_routing/finite_n/report.py
else
  python -m unittest discover -s pipelines/layer_routing/finite_n -p test_exp265.py -v
fi
