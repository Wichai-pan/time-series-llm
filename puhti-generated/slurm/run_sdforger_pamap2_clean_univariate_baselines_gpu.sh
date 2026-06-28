#!/bin/bash
#SBATCH --job-name=sdforger-pamap2-clean-u
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

set -eo pipefail
source /appl/profile/zz-csc-env.sh >/dev/null 2>&1 || true
set -u
module purge
module load pytorch/2.6

BASE=/scratch/project_2016517/panh/time-series-llm/fms-dgt
ENV=/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312

mkdir -p "$BASE/logs"
source "$ENV/bin/activate"
cd "$BASE"
export DGT_DATA_DIR="$BASE/data"
export HF_HOME="/scratch/project_2016517/panh/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/pamap2_subject101_walking_hand_acc16_x_univariate_baseline/task.yaml \
  --restart-generation

python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/pamap2_subject101_running_hand_acc16_x_univariate_baseline/task.yaml \
  --restart-generation

python scripts/evaluate_sdforger_paper_metrics.py \
  --real-parquet data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet \
  --synthetic-jsonl output/time_series/pamap2_subject101_walking_hand_acc16_x_univariate_baseline/final_data.jsonl \
  --channel hand_acc16_x \
  --output-json reports/baseline_verification_20260522/pamap2_subject101_walking_hand_acc16_x_univariate_baseline_tsgbench_metrics.json \
  --output-md reports/baseline_verification_20260522/pamap2_subject101_walking_hand_acc16_x_univariate_baseline_tsgbench_metrics.md \
  --output-plot reports/baseline_verification_20260522/pamap2_subject101_walking_hand_acc16_x_univariate_baseline_overlay.pdf \
  --train-length 5000 \
  --train-samples 1 \
  --augmentation-strategy univariate \
  --min-windows-length 300 \
  --min-windows-number 30 \
  --train-splitting minimize-overlap \
  --synthetic-space scaled

python scripts/evaluate_sdforger_paper_metrics.py \
  --real-parquet data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet \
  --synthetic-jsonl output/time_series/pamap2_subject101_running_hand_acc16_x_univariate_baseline/final_data.jsonl \
  --channel hand_acc16_x \
  --output-json reports/baseline_verification_20260522/pamap2_subject101_running_hand_acc16_x_univariate_baseline_tsgbench_metrics.json \
  --output-md reports/baseline_verification_20260522/pamap2_subject101_running_hand_acc16_x_univariate_baseline_tsgbench_metrics.md \
  --output-plot reports/baseline_verification_20260522/pamap2_subject101_running_hand_acc16_x_univariate_baseline_overlay.pdf \
  --train-length 5000 \
  --train-samples 1 \
  --augmentation-strategy univariate \
  --min-windows-length 300 \
  --min-windows-number 30 \
  --train-splitting minimize-overlap \
  --synthetic-space scaled

ls -la output/time_series/pamap2_subject101_walking_hand_acc16_x_univariate_baseline
ls -la output/time_series/pamap2_subject101_running_hand_acc16_x_univariate_baseline
ls -la reports/baseline_verification_20260522/*_univariate_baseline_tsgbench_metrics.*
