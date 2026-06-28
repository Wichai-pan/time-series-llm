#!/bin/bash
#SBATCH --job-name=sdforger-pamap2-unified-stat
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:45:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# T2 stat-prompt: label + quantized per-window mean/std/min/max bins folded into the
# Condition token. New output dir; does not touch existing runs.
# Plan: docs/experiments/experiment_plan_2026-06-15_stat-prompt.md

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

python scripts/run_unified_stat_prompt.py \
  --walking-parquet data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet \
  --running-parquet data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet \
  --channel hand_acc16_x \
  --output-dir output/time_series/pamap2_subject101_unified_stat_prompt_hand_acc16_x \
  --n-bins 4 \
  --min-per-label 50 \
  --max-per-label 100 \
  --generation-batch-size 64 \
  --seed 42
