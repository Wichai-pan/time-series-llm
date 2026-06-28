#!/bin/bash
#SBATCH --job-name=sdforger-synth-amp-ctrl
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# Stage B: synthetic x1/x3 amplitude control. Trains a fresh control model.
# Maximally-separable numeric token; tests H1 (sparsity) vs H3 (capability).
# New output dir; reads parquet read-only.

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

python scripts/run_synthetic_amplitude_control.py \
  --walking-parquet data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet \
  --running-parquet data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet \
  --channel hand_acc16_x \
  --output-dir output/time_series/pamap2_subject101_synthetic_amplitude_control_20260621 \
  --scale 3.0 --per-amp 40 --max-per-amp 80 \
  --generation-batch-size 64 --temperature 1.3 --seed 42
