#!/bin/bash
#SBATCH --job-name=sdforger-pamap2-formal
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

set -eo pipefail
source /appl/profile/zz-csc-env.sh
set -u
module purge
module load pytorch/2.6

BASE=/scratch/project_2016517/panh/time-series-llm/fms-dgt
ENV=/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312

mkdir -p "$BASE/logs"
source "$ENV/bin/activate"
cd "$BASE"
export DGT_DATA_DIR="$BASE/data"

python - <<"PY"
import torch
import vllm
import fms_dgt
print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("fms_dgt import ok")
PY

python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/pamap2_subject101_multivariate/task.yaml \
  --restart-generation

ls -la output/time_series/pamap2_subject101_multivariate || true
