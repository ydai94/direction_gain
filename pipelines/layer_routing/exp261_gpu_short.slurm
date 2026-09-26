#!/bin/bash
#SBATCH --job-name=e261_gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp261_gpu_short_%A_%a.out
#SBATCH --error=logs/exp261_gpu_short_%A_%a.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
mkdir -p logs results/exp261a_full_output_top1_n
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD:$PWD/experiments/interp_program"

task="$1"
nshards="$2"
shard="${SLURM_ARRAY_TASK_ID:-${3:-0}}"
exec 9>"results/exp261a_full_output_top1_n/.lock_${task}_${shard}"
if ! flock -n 9; then
  echo "Exp261 duplicate task already active: ${task}/${shard}"
  exit 0
fi
if [[ "$task" == "s-smoke" && -s results/exp261a_full_output_top1_n/s_screen/smoke/smoke_summary.json ]]; then
  echo "Exp261 S smoke already complete"
  exit 0
fi
if [[ "$task" == "score-output-smoke" && -s results/exp261a_full_output_top1_n/output_score_smoke.parquet ]]; then
  echo "Exp261 output-score smoke already complete"
  exit 0
fi
if [[ "$task" == "s-cohort" \
      && -s "results/exp261a_full_output_top1_n/s_screen/cohort/score_shards/scores_shard${shard}.parquet" \
      && -s "results/exp261a_full_output_top1_n/s_screen/cohort/score_shards/top1_directions_shard${shard}.npz" \
      && -s "results/exp261a_full_output_top1_n/s_screen/cohort/score_shards/summary_shard${shard}.json" ]]; then
  echo "Exp261 S cohort shard ${shard} already complete"
  exit 0
fi
case "$task" in
  s-smoke) python pipelines/layer_routing/exp261_pipeline.py s-screen --mode smoke --shard 0 --nshards 1 ;;
  s-cohort) python pipelines/layer_routing/exp261_pipeline.py s-screen --mode cohort --shard "$shard" --nshards "$nshards" ;;
  n-smoke) python pipelines/layer_routing/exp261_pipeline.py n-probe --mode smoke --shard 0 --nshards 1 ;;
  n-cohort) python pipelines/layer_routing/exp261_pipeline.py n-probe --mode cohort --shard "$shard" --nshards "$nshards" ;;
  generate) python pipelines/layer_routing/exp261_pipeline.py generate --shard "$shard" --nshards "$nshards" ;;
  score-output-smoke) python pipelines/layer_routing/exp261_pipeline.py score-output --shard 0 --nshards 1 --smoke ;;
  score-output) python pipelines/layer_routing/exp261_pipeline.py score-output --shard "$shard" --nshards "$nshards" ;;
  score-internal) python pipelines/layer_routing/exp261_pipeline.py score-internal --shard "$shard" --nshards "$nshards" ;;
  *) echo "unknown Exp261 GPU task: $task" >&2; exit 2 ;;
esac
