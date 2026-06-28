#!/bin/bash
#SBATCH --job-name=sdforger-statprompt-hf
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# Clean stat-prompt adherence (HF gen). Env: MODEL, ENCODING(bins|values), OUTROOT, SEEDS, EPOCHS, NLEVELS, PERLEVEL, DTYPE, BATCH
set -eo pipefail
source /appl/profile/zz-csc-env.sh >/dev/null 2>&1 || true
set -u
module purge; module load pytorch/2.6
BASE=/scratch/project_2016517/panh/time-series-llm/fms-dgt
source /projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312/bin/activate
cd "$BASE"
export DGT_DATA_DIR="$BASE/data" HF_HOME="/scratch/project_2016517/panh/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers"
: "${MODEL:?}" "${OUTROOT:?}" "${ENCODING:?}"
SEEDS="${SEEDS:-42}"; EPOCHS="${EPOCHS:-100}"; NLEVELS="${NLEVELS:-4}"; PERLEVEL="${PERLEVEL:-24}"; DTYPE="${DTYPE:-float16}"; BATCH="${BATCH:-2}"; GENBACKEND="${GENBACKEND:-hf}"
D=data/public/time_series
for SEED in $SEEDS; do
  echo "########## $OUTROOT ($ENCODING) SEED $SEED ($MODEL) ##########"
  python scripts/run_stat_prompt_hf.py \
    --walking-parquet $D/pamap2_subject101_walking_hand_acc16_x.parquet \
    --running-parquet $D/pamap2_subject101_running_hand_acc16_x.parquet \
    --channel hand_acc16_x --output-dir output/time_series/${OUTROOT}_seed${SEED}_20260621 \
    --model-id-or-path "$MODEL" --encoding $ENCODING --n-levels $NLEVELS --per-level $PERLEVEL \
    --epochs $EPOCHS --train-batch-size $BATCH --dtype $DTYPE --gen-backend $GENBACKEND --temperature 1.0 --seed $SEED
done
echo "DONE $OUTROOT"
