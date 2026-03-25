#!/bin/bash
#SBATCH --job-name=sdforger-pamap2-smoke
#SBATCH --account=project_2016517
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

source /appl/profile/zz-csc-env.sh
set -euo pipefail
module purge
module load pytorch/2.6

BASE=/scratch/project_2016517/panh/time-series-llm/fms-dgt
ENV=/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312

mkdir -p "$BASE/logs"
source "$ENV/bin/activate"
cd "$BASE"
export DGT_DATA_DIR="$BASE/data"

python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/pamap2_subject101_multivariate/task_smoke.yaml \
  --restart-generation

ls -la output/time_series/pamap2_subject101_multivariate_smoke || true
