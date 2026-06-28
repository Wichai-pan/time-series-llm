# Code Review: Normalization and Evaluation Scripts

日期：2026-06-07

范围：

- `scripts/run_unified_label_conditioning_normalization.py`
- `scripts/evaluate_normalization_ablation_outputs.py`
- `scripts/evaluate_normalization_tsgbench_style.py`
- `scripts/evaluate_har_utility_smoke.py`
- `scripts/evaluate_unified_raw_like_metrics.py`

## Findings

### High：model-space 评估结果容易被误当成 raw sensor 结果

`scripts/evaluate_normalization_ablation_outputs.py` 读取的是 `*_real_windows_model_space.npy` 和 `*_synthetic_windows_model_space.npy`，输出的 ACF/PSD、label controllability 和 overlay 都在各 setting 自己的 model-space 中。这个评估适合 debugging，但不能和 raw-like HAR utility 或 raw waveform 图直接比较。

处理：已新增 `scripts/evaluate_unified_raw_like_metrics.py`，把 clean/raw unified/clip/global z-score 都放到 raw-like 主评估口径中。

### High：`joint_window_zscore` 没有保存 inverse 所需 scaler 参数

在 `scripts/run_unified_label_conditioning_normalization.py` 中，`joint_window_zscore` 会 fit 一个 `StandardScaler`，但 metadata 只保存了 `joint_scaler_mean_shape`，没有保存 `scaler.mean_` 和 `scaler.scale_`。这导致后续无法严格把 generated model-space windows 转回 raw-like value space。

建议：如果之后还要保留该 setting，必须保存完整 scaler 参数，或保存 raw-like synthetic export。

### Medium：overlay 图有选择偏差

`scripts/evaluate_normalization_ablation_outputs.py` 的 `plot_overlay()` 会选择与 `real_mean` 欧氏距离最近的一条 synthetic sample。这会让图看起来比平均情况更好，适合作为 qualitative representative/best-case 图，但不能代表整体质量。

处理：新脚本生成跨 setting overlay 时仍使用 representative sample，但文件和报告中明确标为 qualitative comparison，不作为误差图。

### Medium：`generated_time_series` 字段没有记录 value space

多个输出 JSONL 都使用 `generated_time_series` 字段，但有的值是 SDForger standardized window space，有的是 normalization-specific model space。字段名本身无法区分 raw / scaled / normalized。

建议：后续每条或每个 run metadata 至少记录：

- `value_space`
- `inverse_normalization`
- `train_length`
- `window_length`
- `channel`
- `activity`

### Medium：global/activity normalization ablation 的 cross-setting 指标不能直接比较

`global_series_zscore` 和 `activity_series_zscore` 绕过了 SDForger 原本的 per-timestamp window scaler，FICA input space 和 old raw-unified 并不等价。因此它们的 model-space TSGBench-style 指标只能说明该 run 内部的诊断结果，不能直接作为“比 baseline 更好”的主结论。

处理：已把 normalization ablation 降级为 diagnostic-only，并补 raw-like 主评估。

## 新增脚本审查

`scripts/evaluate_unified_raw_like_metrics.py` 做了以下约束：

- 对 clean/raw unified/clip，使用 activity train-window per-timestep mean/std 做 inverse transform，与旧 HAR smoke 一致。
- 对 global z-score，使用 `run_metadata.json` 中的 `global_mean/global_std` 做 inverse transform。
- 统一输出 value/rhythm、TSGBench-style、HAR utility、label controllability。
- 生成 setting overlay 图时做 lag alignment，并明确是 representative waveform comparison。

已做检查：

- `python3 -m py_compile scripts/evaluate_unified_raw_like_metrics.py` 通过。
- Puhti 上完成只读评估，没有提交 job，没有覆盖旧实验目录。

## Remaining Risks

- 本次为了速度使用了 `--skip-dtw`，DTW 尚未补。
- HAR utility 仍是 subject101 within-subject smoke，不是 unseen-subject 证据。
- activity-specific inverse transform 适合当前 SDForger scaled-output contract，但不是完全 class-agnostic raw export。
- `clip_p05_p95` 仍是 post-hoc diagnostic，不是 generation-time method。
