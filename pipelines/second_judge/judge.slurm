#!/bin/bash
# Usage: sbatch [site options] gpt_judge.slurm MODEL MAX_USD
#SBATCH --job-name=exp277_gptjudge
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp277_%x_%j.out
#SBATCH --error=logs/exp277_%x_%j.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
release="$PWD/pipelines/second_judge"
set -a; . ./.env; set +a
# fail fast if this node has no outbound network rather than burning the allocation
code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY")
echo "api.openai.com reachable: http $code"
[ "$code" = "200" ] || { echo "NO OUTBOUND NETWORK ON THIS NODE"; exit 3; }
export WORKERS=24
export JUDGE_MODEL=gpt-5.6-terra
python3 "$release/run_judge.py" "$1" "$2"
