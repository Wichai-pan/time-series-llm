#!/bin/bash
#SBATCH --job-name=sdforger-formal
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/projappl/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

set -eo pipefail
source /appl/profile/zz-csc-env.sh
set -u
module purge
module load pytorch/2.6

BASE=/projappl/project_2016517/panh/time-series-llm/fms-dgt
ENV=/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312

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
  --task-paths ./tasks/public/time_series/bikesharing_univariate/task.yaml \
  --restart-generation

ls -la output/time_series/bikesharing_univariate || true
