#!/bin/bash
#SBATCH --job-name=e261_s_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp261_s_short_%A_%a.out
#SBATCH --error=logs/exp261_s_short_%A_%a.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
mkdir -p logs results/exp261a_full_output_top1_n
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD:$PWD/experiments/interp_program"

pair="${SLURM_ARRAY_TASK_ID:?array task required}"
for offset in 0 1; do
  shard=$((pair * 2 + offset))
  score="results/exp261a_full_output_top1_n/s_screen/cohort/score_shards/scores_shard${shard}.parquet"
  direction="results/exp261a_full_output_top1_n/s_screen/cohort/score_shards/top1_directions_shard${shard}.npz"
  summary="results/exp261a_full_output_top1_n/s_screen/cohort/score_shards/summary_shard${shard}.json"
  (
    if ! flock -n 9; then
      echo "Exp261 S shard ${shard} already active elsewhere"
      exit 0
    fi
    if [[ -s "$score" && -s "$direction" && -s "$summary" ]]; then
      echo "Exp261 S shard ${shard} already complete"
      exit 0
    fi
    python pipelines/layer_routing/exp261_pipeline.py s-screen \
      --mode cohort --shard "$shard" --nshards 20
  ) 9>"results/exp261a_full_output_top1_n/.lock_s-cohort_${shard}"
done
