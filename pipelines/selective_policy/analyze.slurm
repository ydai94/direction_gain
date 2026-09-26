#!/bin/bash
#SBATCH --job-name=exp278_279
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${DG_ROOT}/logs/exp278_279_%j.out
#SBATCH --error=${DG_ROOT}/logs/exp278_279_%j.err
cd ${DG_ROOT}/pipelines/selective_policy
module purge; module load Anaconda3/2024.02-1; export CONDA_ENVS_PATH=${PROJECT_ROOT}/anaconda3/envs; eval "$(conda shell.bash hook)"; conda activate project
export NBOOT=5000
python analyze_b2_b3.py
