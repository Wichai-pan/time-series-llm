# Normalization Ablation Plan - PAMAP2 subject101 unified label conditioning

日期：2026-06-07

## Goal

诊断 `raw-unified label conditioning` 的 value explosion 是否与进入 FICA 前的窗口尺度/标准化方式有关。

当前不是证明最终方法优于 baseline，而是回答：

> 统一 walking/running 条件生成失败时，主要问题是否来自 FICA input space 的 normalization / scaling contract？

## Fixed Controls

| Item | Value |
|---|---|
| Dataset | PAMAP2 |
| Subject | subject101 |
| Activities | walking + running |
| Channel | `hand_acc16_x` |
| Model | GPT-2 |
| SDForger structure | unchanged |
| Conditioning | `Condition: data is walking/running` |
| Embedding | joint FICA |
| Window length | 300 |
| Train length per activity | 5000 |
| Target generated samples | 50-100 per activity |

## Important Preprocessing Clarification

The prior `raw-unified` run was not raw sensor values directly entering FICA.

In the existing SDForger utility, `preprocess_train_data()` segments windows and then applies `StandardScaler` over the window matrix. The old unified script ran this preprocessing separately for walking and running, then concatenated the resulting scaled windows before joint FICA.

Therefore this ablation does not simply add another z-score before the same preprocessing, because that would mostly be cancelled by the later window scaler. Instead, the variants explicitly change the window space used by FICA.

## Run Matrix

| Array task | Mode | FICA input space | Purpose |
|---:|---|---|---|
| 0 | `current_activity_window_zscore` | Existing activity-specific SDForger timestamp-wise window scaler | Reproduce old raw-unified preprocessing path |
| 1 | `joint_window_zscore` | One timestamp-wise scaler fitted after combining walking+running windows | Test whether separate activity scaling caused mismatch |
| 2 | `global_series_zscore` | Raw segmented windows after one scalar z-score over walking+running training series | Test simple shared pre-embedding normalization without timestamp-wise window scaler |
| 3 | `activity_series_zscore` | Raw segmented windows after activity-specific scalar z-score | Test whether activity-level amplitude scale is the issue |

## Submitted Job

Remote root:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt`

Local scripts:

- `scripts/run_unified_label_conditioning_normalization.py`
- `puhti-generated/slurm/run_sdforger_pamap2_normalization_ablation_gpu.sh`

Remote scripts:

- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/scripts/run_unified_label_conditioning_normalization.py`
- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/slurm/run_sdforger_pamap2_normalization_ablation_gpu.sh`

Slurm job:

`34758239`

Output directories:

- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_norm_ablation_current_activity_window_zscore_hand_acc16_x`
- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_norm_ablation_joint_window_zscore_hand_acc16_x`
- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_norm_ablation_global_series_zscore_hand_acc16_x`
- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_norm_ablation_activity_series_zscore_hand_acc16_x`

Logs:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt/logs/sdforger-pamap2-norm_34758239_<array_id>.out`

## Primary Metrics After Completion

Evaluate each mode on:

- value stability: `synthetic_abs_max`, `synthetic_std_mean`, amplitude ratio
- label controllability: real-trained walking/running classifier on synthetic requested labels
- periodicity: ACF peak lag and PSD peak frequency, using the corresponding model-space real windows saved by the run
- qualitative overlay: real vs generated windows for walking and running
- optional utility smoke after the best stable mode is identified

## Decision Rules

If `joint_window_zscore`, `global_series_zscore`, or `activity_series_zscore` reduces value explosion while preserving label controllability, treat normalization as a promising fix and extend only the best mode to multi-subject.

If all modes still explode, the failure is likely not solved by input normalization alone; next actions should return to latent validity control, FICA component choices, or generation-time rejection.

If a mode is stable but label controllability collapses, normalization fixed amplitude but damaged conditional separation.
