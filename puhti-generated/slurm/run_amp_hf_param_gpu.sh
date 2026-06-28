#!/bin/bash
#SBATCH --job-name=sdforger-amp-hf
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# Parametrized x1/x3 amplitude control with HF (transformers) generation.
# Env: MODEL, OUTROOT, SEEDS (space-sep), EPOCHS, PERAMP, DTYPE, BATCH
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

: "${MODEL:?set MODEL}" "${OUTROOT:?set OUTROOT}"
SEEDS="${SEEDS:-42}"; EPOCHS="${EPOCHS:-30}"; PERAMP="${PERAMP:-40}"; DTYPE="${DTYPE:-float16}"; BATCH="${BATCH:-2}"
D=data/public/time_series
for SEED in $SEEDS; do
  echo "########## $OUTROOT SEED $SEED ($MODEL) ##########"
  python scripts/run_synthetic_amplitude_control_hf.py \
    --walking-parquet $D/pamap2_subject101_walking_hand_acc16_x.parquet \
    --running-parquet $D/pamap2_subject101_running_hand_acc16_x.parquet \
    --channel hand_acc16_x \
    --output-dir output/time_series/${OUTROOT}_seed${SEED}_20260621 \
    --model-id-or-path "$MODEL" --dtype $DTYPE --train-batch-size $BATCH --epochs $EPOCHS \
    --per-amp $PERAMP --max-per-amp $((PERAMP * 2)) --generation-batch-size 16 --seed $SEED
done
echo "DONE $OUTROOT"
