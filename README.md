# Time Series LLM Project on Puhti

This repository is a cleaned export of the project code used on CSC Puhti for the Aalto `ELEC-E7633` project work on time-series generation with large language models.

It combines three experiment tracks:

- `fms-dgt/`: SDForger and related configuration used for bike-sharing time-series synthesis on Puhti.
- `chatts/ChatTS/`: ChatTS code plus Puhti-specific demo and Slurm scripts.
- `agentsense/AgentSense/`: AgentSense code plus offline reproduction and Slurm scripts prepared on Puhti.

The export intentionally excludes runtime-only artifacts:

- cluster environments and caches
- Slurm logs
- generated outputs and experiment folders
- API keys and local secrets
- nested Git metadata from the original upstream repositories

## Notes

- The contents reflect the project state as used on Puhti, including custom Slurm scripts and local adaptation files.
- Upstream repositories originally came from IBM `fms-dgt`, NetManAIOps `ChatTS`, and Zikang Leng `AgentSense`.
- This export is intended for project archival and sharing, not as a pristine mirror of the upstream repositories.
