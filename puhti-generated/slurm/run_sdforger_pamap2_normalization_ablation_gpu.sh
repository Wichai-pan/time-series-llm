#!/bin/bash
#SBATCH --job-name=sdforger-pamap2-norm
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:45:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%A_%a.out

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

MODES=(
  current_activity_window_zscore
  joint_window_zscore
  global_series_zscore
  activity_series_zscore
)

MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
OUT="output/time_series/pamap2_subject101_norm_ablation_${MODE}_hand_acc16_x"

echo "Running normalization mode: ${MODE}"
echo "Output: ${OUT}"

python scripts/run_unified_label_conditioning_normalization.py \
  --walking-parquet data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet \
  --running-parquet data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet \
  --channel hand_acc16_x \
  --normalization-mode "${MODE}" \
  --output-dir "${OUT}" \
  --min-per-label 50 \
  --max-per-label 100 \
  --generation-batch-size 64
