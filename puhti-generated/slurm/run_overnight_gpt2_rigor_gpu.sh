#!/bin/bash
#SBATCH --job-name=sdforger-overnight-rigor
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# Overnight rigor: 5-seed combined-best (multi-subject unified + per-activity repair),
# TSG (train+heldout refs) + held-out HAR utility per seed. gpt2, guaranteed.

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
WTR=$D/pamap2_subject101_102_105_walking_hand_acc16_x.parquet
RTR=$D/pamap2_subject101_102_105_running_hand_acc16_x.parquet
WTE=$D/pamap2_subject106_108_walking_hand_acc16_x.parquet
RTE=$D/pamap2_subject106_108_running_hand_acc16_x.parquet
ROOT=output/time_series/overnight_rigor_20260621
REP_ROOT=$ROOT
mkdir -p reports/overnight_rigor_20260621

for SEED in 1 2 3 4 5; do
  echo "########## SEED $SEED ##########"
  GEN=$ROOT/gen_seed${SEED}
  REP=$ROOT/repair_seed${SEED}
  python scripts/run_unified_label_conditioning.py \
    --walking-parquet $WTR --running-parquet $RTR --channel hand_acc16_x \
    --train-length 15000 --min-per-label 50 --max-per-label 100 --generation-batch-size 64 \
    --seed $SEED --output-dir $GEN
  python scripts/apply_smooth_latent_repair.py \
    --walking-parquet $WTR --running-parquet $RTR --generated-dir $GEN --output-dir $REP \
    --channel hand_acc16_x --train-length 15000 --window-length 300 --min-windows-number 30 \
    --train-splitting minimize-overlap --band minmax --reject-frac 0.5 --per-activity-bounds
  python scripts/rerun_combined_best_tsgbench_table.py reports/overnight_rigor_20260621/tsg_seed${SEED} $REP
  python scripts/evaluate_har_utility_heldout.py \
    --walking-train-parquet $WTR --running-train-parquet $RTR \
    --walking-test-parquet $WTE --running-test-parquet $RTE \
    --walking-synthetic-jsonl $REP/clip_p05_p95/walking_final_data.jsonl \
    --running-synthetic-jsonl $REP/clip_p05_p95/running_final_data.jsonl \
    --channel hand_acc16_x --train-length 15000 \
    --output-dir reports/overnight_rigor_20260621/har_seed${SEED}
done
echo "ALL SEEDS DONE"
