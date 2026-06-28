#!/bin/bash
#SBATCH --job-name=sdforger-statprompt-temp0
#SBATCH --account=project_2016517
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# A3 probe: greedy (temp=0) generation from the EXISTING stat-prompt checkpoint.
# No fine-tuning. New output dir; reads checkpoint read-only.

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

python scripts/run_stat_prompt_generate_only.py \
  --walking-parquet data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet \
  --running-parquet data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet \
  --model-path output/time_series/pamap2_subject101_unified_stat_prompt_hand_acc16_x/model/model_iter-1/best \
  --output-dir output/time_series/pamap2_subject101_stat_prompt_temp0_probe_20260617 \
  --channel hand_acc16_x --train-length 5000 --window-length 300 \
  --min-windows-number 30 --train-splitting minimize-overlap \
  --n-bins 4 --temperature 0.0 --seed 42
