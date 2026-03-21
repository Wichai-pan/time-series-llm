#!/bin/bash
#SBATCH --job-name=chatts-demo
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
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

if [ ! -f "$SCR/pydeps/.chatts_pinned_ready" ]; then
  rm -rf "$SCR/pydeps"
  mkdir -p "$SCR/pydeps"
  python -m pip install --upgrade --target "$SCR/pydeps" --no-deps \
    "transformers==4.52.4" \
    "tokenizers==0.21.4" \
    "huggingface_hub==0.30.2" \
    "vllm==0.8.5"
  touch "$SCR/pydeps/.chatts_pinned_ready"
fi

python - <<"PY"
import os
import transformers
import vllm
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
from huggingface_hub import snapshot_download
model_dir = "/scratch/project_2016517/panh/chatts/models/ChatTS-8B"
if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id="bytedance-research/ChatTS-8B", local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    print("Downloaded ChatTS-8B")
else:
    print("ChatTS-8B already present")
PY

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CHATTS_MODEL_PATH="$SCR/models/ChatTS-8B"
export CHATTS_OUT_DIR="$BASE/exp/chatts_puhti_demo"

python -m demo.demo_vllm_puhti

ls -la "$BASE/exp/chatts_puhti_demo"
