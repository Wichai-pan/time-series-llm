# Time Series LLM Project on Puhti

This repository is a cleaned export of the project code used on CSC Puhti for the Aalto `ELEC-E7633` project work on time-series generation with large language models.

The current project compares three different directions for synthetic or simulated time-series generation:

- `fms-dgt/`: SDForger-based synthetic time-series generation for bike-sharing data.
- `chatts/ChatTS/`: LLM-based time-series reasoning and generation demos adapted to Puhti.
- `agentsense/AgentSense/`: LLM agent pipeline and simulation workflow for smart-home sensor generation.

## Project Status

### 1. SDForger / FMS-DGT

Status: formal reproduction completed on Puhti

Completed work:

- Added Puhti-specific Slurm scripts for smoke and formal runs.
- Ran a smoke test for the bike-sharing univariate task.
- Ran the formal `bikesharing_univariate` generation successfully.
- Produced a final synthetic dataset and generation plot on Puhti.

Current result summary:

- Smoke run produced 11 samples.
- Formal run produced 57 synthetic `cnt` time-series samples.
- The formal run generated an overlay plot comparing original and synthetic sequences.

Relevant project files:

- `fms-dgt/slurm/run_sdforger_smoke_gpu.sh`
- `fms-dgt/slurm/run_sdforger_formal_gpu.sh`
- `fms-dgt/tasks/public/time_series/bikesharing_univariate/task_smoke.yaml`
- `fms-dgt/tasks/public/time_series/bikesharing_univariate/task.yaml`

### 2. ChatTS

Status: HuggingFace-based Puhti reproduction completed

Completed work:

- Added Puhti-specific demo scripts and Slurm launch scripts.
- Attempted a `vLLM`-based path and identified hardware/backend instability on V100.
- Switched to the HuggingFace Transformers inference path for a stable run.
- Produced demo outputs for time-series analysis prompts on Puhti.

Current result summary:

- The HF path runs successfully on Puhti and produces structured time-series analysis answers.
- The `vLLM` path was explored but is not currently the stable default in this environment.

Relevant project files:

- `chatts/ChatTS/demo/demo_hf_puhti.py`
- `chatts/ChatTS/demo/demo_vllm_puhti.py`
- `chatts/ChatTS/slurm/run_chatts_hf_demo_gpu.sh`
- `chatts/ChatTS/slurm/run_chatts_demo_gpu.sh`

### 3. AgentSense

Status: offline pipeline reproduction completed, full end-to-end reproduction still incomplete

Completed work:

- Cloned and organized the AgentSense project on Puhti.
- Reproduced the offline steps that do not require OpenAI API access or Unity execution.
- Implemented and ran an offline reproduction script for:
  - Step 4: split weekly routine into daily files
  - Step 5: clean and parse daily routines
  - Step 9: split labeled routines into blocks
  - evaluation: check formatting and grounding quality
- Verified that reproduced offline outputs match the provided reference data.

Current limitations:

- Steps 1, 2, 3, 7, and 8 depend on external OpenAI API access.
- Steps 10 and 11 depend on the VirtualHome / Unity simulation outputs.
- Because of these dependencies, the full end-to-end AgentSense pipeline is not yet fully reproduced on Puhti.

Relevant project files:

- `agentsense/AgentSense/AgentSense_pipeline/scripts/repro_agentsense_offline.py`
- `agentsense/AgentSense/AgentSense_pipeline/scripts/slurm_repro_agentsense_offline.sh`

## Repository Notes

This export intentionally excludes runtime-only artifacts:

- cluster environments and caches
- Slurm logs
- generated outputs and experiment folders
- API keys and local secrets
- nested Git metadata from the original upstream repositories

The contents reflect the project state as used on Puhti, including custom Slurm scripts and local adaptation files.

Upstream repositories originally came from:

- IBM `fms-dgt`
- NetManAIOps `ChatTS`
- Zikang Leng `AgentSense`

This repository is intended for project archival, collaboration, and reporting rather than as a pristine mirror of the upstream repositories.
