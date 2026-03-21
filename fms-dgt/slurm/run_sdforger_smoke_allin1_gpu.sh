#!/bin/bash
#SBATCH --job-name=sdforger-allin1
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/projappl/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

set -eo pipefail
source /appl/profile/zz-csc-env.sh
set -u
module purge
module load pytorch/2.6

BASE=/projappl/project_2016517/panh/time-series-llm/fms-dgt
ENV=/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312

mkdir -p /projappl/project_2016517/panh/time-series-llm/envs
cd "$BASE"

if [ ! -d "$ENV" ]; then
  python -m venv --system-site-packages "$ENV"
fi

source "$ENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[time_series,vllm]"

python - <<"PY"
import torch
print("torch", torch.__version__)
import vllm
print("vllm", vllm.__version__)
import fms_dgt
print("fms_dgt import ok")
PY

export DGT_DATA_DIR="$BASE/data"
python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/bikesharing_univariate/task_smoke.yaml \
  --restart-generation

ls -la output/time_series/bikesharing_univariate_smoke || true
