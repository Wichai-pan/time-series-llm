#!/bin/bash
#SBATCH --job-name=chatts-hf
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/projappl/project_2016517/panh/time-series-llm/chatts/ChatTS/logs/%x_%j.out

set -eo pipefail
source /appl/profile/zz-csc-env.sh
set -u
module purge
module load pytorch/2.6

BASE=/projappl/project_2016517/panh/time-series-llm/chatts/ChatTS
SCR=/scratch/project_2016517/panh/chatts

export TMPDIR="$SCR/tmp"
export PIP_CACHE_DIR="$SCR/cache/pip"
export HF_HOME="$SCR/hf"
export HUGGINGFACE_HUB_CACHE="$SCR/hf/hub"
export TRANSFORMERS_CACHE="$SCR/hf/transformers"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SCR/pydeps:${PYTHONPATH:-}"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$SCR/models/ChatTS-8B"
cd "$BASE"

if [ ! -f "$SCR/pydeps/.chatts_hf_ready" ]; then
  rm -rf "$SCR/pydeps"
  mkdir -p "$SCR/pydeps"
  python -m pip install --upgrade --target "$SCR/pydeps" --no-deps \
    "transformers==4.52.4" \
    "tokenizers==0.21.4" \
    "huggingface_hub==0.30.2"
  touch "$SCR/pydeps/.chatts_hf_ready"
fi

python - <<"PY"
import os
from huggingface_hub import snapshot_download
model_dir = "/scratch/project_2016517/panh/chatts/models/ChatTS-8B"
if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id="bytedance-research/ChatTS-8B", local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
print("model ready", model_dir)
PY

export CHATTS_MODEL_PATH="$SCR/models/ChatTS-8B"
export CHATTS_OUT_DIR="$BASE/exp/chatts_hf_demo"
python -m demo.demo_hf_puhti

ls -la "$BASE/exp/chatts_hf_demo"
