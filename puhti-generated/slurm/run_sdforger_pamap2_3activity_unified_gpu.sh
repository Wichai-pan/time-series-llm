#!/bin/bash
#SBATCH --job-name=sdforger-3activity-unified
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# #3 push-TSG: 3-activity (walking/running/cycling) unified label-conditioned generation,
# multi-subject 101/102/105. New output dir; reads parquet read-only.

set -eo pipefail
source /appl/profile/zz-csc-env.sh >/dev/null 2>&1 || true
set -u
module purge
module load pytorch/2.6

BASE=/scratch/project_2016517/panh/time-series-llm/fms-dgt
ENV=/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312
source "$ENV/bin/activate"
cd "$BASE"
export DGT_DATA_DIR="$BASE/data"
export HF_HOME="/scratch/project_2016517/panh/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

D=data/public/time_series
python scripts/run_unified_multiactivity.py \
  --activity "walking=$D/pamap2_subject101_102_105_walking_hand_acc16_x.parquet" \
  --activity "running=$D/pamap2_subject101_102_105_running_hand_acc16_x.parquet" \
  --activity "cycling=$D/pamap2_subject101_102_105_cycling_hand_acc16_x.parquet" \
  --channel hand_acc16_x \
  --output-dir output/time_series/pamap2_subject101_102_105_walking_running_cycling_unified_20260621 \
  --train-length 15000 --min-per-label 50 --max-per-label 100 \
  --generation-batch-size 64 --seed 42
