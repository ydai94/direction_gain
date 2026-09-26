#!/bin/bash
#SBATCH --job-name=exp274_reproduce
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/exp274_%j.out
#SBATCH --error=logs/exp274_%j.err
set -euo pipefail
cd "${DG_ROOT:?set DG_ROOT to the repository root}"
source "$DG_ROOT/configs/env.sh"   # activate the Python environment; edit for your site
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
release="$1"
audit_result="results/frozen_selection/remote_${SLURM_JOB_ID}"
mkdir -p "$audit_result"
python -c 'import sys,numpy,scipy,platform; print(sys.version); print(numpy.__version__,scipy.__version__); print(platform.platform())' > "$audit_result/environment.txt"
python "$release/analyze.py" --source results/heldout_prediction/runtime/analysis.json --predictors "$release/predictors_frozen.json" --output "$audit_result/results.json"
python - "$audit_result" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'results.json').read_text())
assert set(r['models'])=={'sd3','flux','qwen'}
assert all(v['n_valid']==1295 for v in r['models'].values())
(p/'complete.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],status='COMPLETED_OUTPUT_WRITTEN'))+'\n')
PY
