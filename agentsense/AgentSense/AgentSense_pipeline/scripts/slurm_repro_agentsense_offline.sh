#!/bin/bash
#SBATCH --job-name=agentsense-offline-repro
#SBATCH --account=project_2016517
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd /projappl/project_2016517/panh/time-series-llm/agentsense/AgentSense/AgentSense_pipeline
python3 scripts/repro_agentsense_offline.py
