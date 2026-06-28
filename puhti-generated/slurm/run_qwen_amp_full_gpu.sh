#!/bin/bash
#SBATCH --job-name=qwen-amp-full
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# Stronger-model test: synthetic x1/x3 control on Qwen2.5-1.5B (12x gpt2, modern),
# 3 seeds. Directly tests whether a stronger model can do the numeric conditioning
# gpt2 failed (Stage B). fp16 + small batch to fit V100.

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
for SEED in 42 7 123; do
  echo "########## QWEN SEED $SEED ##########"
  python scripts/run_synthetic_amplitude_control.py \
    --walking-parquet $D/pamap2_subject101_walking_hand_acc16_x.parquet \
    --running-parquet $D/pamap2_subject101_running_hand_acc16_x.parquet \
    --channel hand_acc16_x \
    --output-dir output/time_series/qwen_amp_seed${SEED}_20260621 \
    --model-id-or-path Qwen/Qwen2.5-1.5B --dtype float16 --train-batch-size 2 --epochs 30 \
    --per-amp 40 --max-per-amp 80 --generation-batch-size 32 --seed $SEED
done
echo "QWEN ALL SEEDS DONE"
