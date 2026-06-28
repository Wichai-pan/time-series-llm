#!/bin/bash
#SBATCH --job-name=sdforger-gapfillers
#SBATCH --account=project_2016517
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/%x_%j.out

# Comparative reference experiments (CPU): real-vs-real TSG floor + naive jitter baseline (TSG + HAR).
set -eo pipefail
source /appl/profile/zz-csc-env.sh >/dev/null 2>&1 || true
set -u
module purge; module load pytorch/2.6
BASE=/scratch/project_2016517/panh/time-series-llm/fms-dgt
source /projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312/bin/activate
cd "$BASE"; export DGT_DATA_DIR="$BASE/data"
D=data/public/time_series
R=reports/gap_fillers_20260622
mkdir -p $R/floor $R/jitter

WTR=$D/pamap2_subject101_102_105_walking_hand_acc16_x.parquet
RTR=$D/pamap2_subject101_102_105_running_hand_acc16_x.parquet
WHE=$D/pamap2_subject106_108_walking_hand_acc16_x.parquet
RHE=$D/pamap2_subject106_108_running_hand_acc16_x.parquet

evalit () { # $1 real-parquet  $2 synth-jsonl  $3 train-length  $4 out-json
  python scripts/evaluate_sdforger_paper_metrics.py --real-parquet "$1" --synthetic-jsonl "$2" \
    --channel hand_acc16_x --output-json "$4" --output-md "${4%.json}.md" --output-plot "${4%.json}.png" \
    --train-length "$3" --synthetic-space scaled --subset-mode first >/dev/null 2>&1
}

echo "##### A. real-vs-real 地板 (held-out 106/108 标准化窗口 vs train 101/102/105) #####"
for L in walking running; do
  P=$WHE; TR=$WTR; [ $L = running ] && P=$RHE && TR=$RTR
  python scripts/make_scaled_jsonl.py --parquet $P --label $L --channel hand_acc16_x --output $R/floor/${L}.jsonl --train-length 10000 >/dev/null
  evalit "$TR" "$R/floor/${L}.jsonl" 15000 "$R/floor/${L}.json"
done

echo "##### B. naive jitter 基线 (训练窗口 + 高斯噪声 0.2) — TSG #####"
for L in walking running; do
  TR=$WTR; [ $L = running ] && TR=$RTR
  python scripts/make_scaled_jsonl.py --parquet $TR --label $L --channel hand_acc16_x --output $R/jitter/${L}.jsonl --train-length 15000 --jitter 0.2 >/dev/null
  evalit "$TR" "$R/jitter/${L}.jsonl" 15000 "$R/jitter/${L}.json"
done

echo "##### C. naive jitter HAR (real+jitter vs real-only, held-out 106/108) #####"
python scripts/evaluate_har_utility_heldout.py \
  --walking-train-parquet $WTR --running-train-parquet $RTR \
  --walking-test-parquet $WHE --running-test-parquet $RHE \
  --walking-synthetic-jsonl $R/jitter/walking.jsonl --running-synthetic-jsonl $R/jitter/running.jsonl \
  --channel hand_acc16_x --train-length 15000 --output-dir $R/jitter_har >/dev/null 2>&1

echo "##### 汇总 #####"
python - <<PY
import json, glob, os
R="reports/gap_fillers_20260622"
print("A. real-vs-real 地板 (越低=两批真实数据本就越像; 这是 ACD/DTW 的不可约下限):")
for L in ["walking","running"]:
    f=R+"/floor/%s.json"%L
    if os.path.exists(f):
        r=json.load(open(f)); print("   %-8s MDD=%.3f ACD=%.3f DTW=%.1f"%(L,r["mdd"],r["acd"],r["dtw"]))
print("B. naive jitter TSG (trivial '生成器' 的保真度参考):")
for L in ["walking","running"]:
    f=R+"/jitter/%s.json"%L
    if os.path.exists(f):
        r=json.load(open(f)); print("   %-8s MDD=%.3f ACD=%.3f DTW=%.1f"%(L,r["mdd"],r["acd"],r["dtw"]))
print("C. naive jitter HAR (我们的 synthetic 必须打赢这个):")
f=R+"/jitter_har/har_utility_heldout_results.csv"
if os.path.exists(f):
    import csv
    for row in csv.DictReader(open(f)):
        print("   %-16s acc=%.3f f1=%.3f"%(row["condition"],float(row["accuracy"]),float(row["macro_f1"])))
PY
echo "GAP FILLERS DONE"
