#!/bin/bash
#SBATCH --job-name=e261_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp261_cpu_%j.out
#SBATCH --error=logs/exp261_cpu_%j.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
mkdir -p logs results/exp261a_full_output_top1_n
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export PYTHONPATH="$PWD:$PWD/experiments/interp_program"
task="$1"
if [[ "$task" == "analyze-verify" ]]; then
  python pipelines/layer_routing/exp261_pipeline.py analyze
  python pipelines/layer_routing/exp261_pipeline.py verify
else
  python pipelines/layer_routing/exp261_pipeline.py "$task"
fi

# The QOS counts pending array elements, so register the final one-job stage only
# after the N smoke gate has released the large cohort arrays.
if [[ "$task" == "verify-n-smoke" ]]; then
  source results/exp261a_full_output_top1_n/dag_dependencies.env
  route_job=$(sbatch --parsable --dependency="afterok:${N_ARRAY_JOB}" \
    --partition=interactive --job-name=e261_route_r \
    pipelines/layer_routing/exp261_cpu.slurm merge-route)
  generation_job=$(sbatch --parsable --dependency="afterok:${route_job}" \
    --array=0-23%24 --job-name=e261_gen_r \
    pipelines/layer_routing/exp261_gpu.slurm generate 24)
  echo "Submitted resumed Exp261 route: ${route_job}"
  echo "Submitted resumed Exp261 generation: ${generation_job}"
  internal_score_jobs=()
  for shard in {0..7}; do
    generation_dependencies="${generation_job}_${shard}:${generation_job}_$((shard + 8)):${generation_job}_$((shard + 16))"
    score_job=$(sbatch --parsable --dependency="afterok:${generation_dependencies}" \
      --job-name="e261_intscore_s${shard}" \
      pipelines/layer_routing/exp261_gpu.slurm score-internal 8 "$shard")
    internal_score_jobs+=("$score_job")
    echo "Submitted pipelined Exp261 internal score shard ${shard}: ${score_job}"
  done
  internal_dependency=$(IFS=:; echo "${internal_score_jobs[*]}")
  sbatch --dependency="afterok:${OUTPUT_SCORE_JOB}:${internal_dependency}" \
    --partition=interactive --job-name=e261_finalize \
    pipelines/layer_routing/exp261_cpu.slurm analyze-verify
fi
