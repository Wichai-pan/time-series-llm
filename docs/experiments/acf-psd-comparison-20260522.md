# ACF / PSD Comparison - Clean Univariate Baseline

日期：2026-05-22

## Summary

- Question：SDForger clean univariate baseline 是否保留 walking/running 的主要周期结构？
- Setup：PAMAP2 subject101，activity-specific `hand_acc16_x`，真实窗口和 synthetic 窗口都在 standardized SDForger window space 中比较。
- Headline result：running 的 ACF 主峰和 PSD 主频几乎完全匹配；walking 的 PSD 主频匹配，但 ACF 主峰出现 58 vs 116 的差异。
- Interpretation：running baseline 对主要 motion rhythm 的保留较强；walking baseline 学到主频，但周期结构有 harmonic / 二倍周期差异。
- Next step：把这个作为 baseline verification 图；后续再做 good/bad sample 分层和最小 HAR utility smoke。

## 1. Experiment Motivation

Overlay 图里曲线太多，肉眼很难判断生成样本是否真的保留动作节奏。本实验用 ACF 和 PSD 直接比较真实窗口和 synthetic 窗口：

- ACF 关注时间域重复结构，回答“多少 lag 后形状会重复”。
- PSD 关注频域主频，回答“主要运动节奏频率是否一致”。

这一步只验证 baseline 的周期性保留，不重新训练模型，也不支持最终 HAR utility claim。

## 2. Experiment Setup

| Item | Value |
|---|---|
| Dataset | PAMAP2 subject101 |
| Activities | walking-only, running-only |
| Channel | `hand_acc16_x` |
| Real input | activity-specific parquet from original `subject101.dat` |
| Synthetic input | SDForger clean univariate baseline `final_data.jsonl` |
| Window length | 300 |
| Real windows | 31 |
| Synthetic windows | walking 130, running 98 |
| Value space | standardized SDForger window space |
| Sampling rate assumption | 100 Hz |
| Script | `scripts/compare_sdforger_acf_psd.py` |

Remote outputs:

- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/baseline_verification_20260522/pamap2_subject101_walking_hand_acc16_x_acf_psd_comparison.*`
- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/baseline_verification_20260522/pamap2_subject101_running_hand_acc16_x_acf_psd_comparison.*`

Local outputs:

- `outputs/acf-psd-comparison-20260522/`

## 3. Core Algorithm or Method

Not applicable as a new method. This is a diagnostic evaluation on the already completed SDForger baseline.

The real windows are rebuilt through SDForger's own `preprocess_train_data`, using the same baseline settings:

- `train_length: 5000`
- `train_samples: 1`
- `augmentation_strategy: univariate`
- `min_windows_length: 300`
- `min_windows_number: 30`
- `train_splitting: minimize-overlap`

The synthetic windows are read directly from `generated_time_series["hand_acc16_x"]`.

## 4. Metrics

| Metric | Definition | Direction | Why it matters |
|---|---|---|---|
| Mean ACF peak lag | Peak lag from average ACF curve across windows | closer real/synthetic is better | Measures whether the repeating cycle length is preserved |
| Mean ACF peak score | ACF value at the detected peak | similar magnitude is better | Measures strength of periodic repetition |
| Mean PSD peak Hz | Dominant frequency from average PSD | closer real/synthetic is better | Measures whether the main motion rhythm frequency is preserved |
| Window std mean | Mean per-window standard deviation | similar is better | Checks amplitude/variation scale in standardized space |
| Absolute max | Max absolute value across windows | not too large | Flags extreme generated samples |

## 5. Results

| Activity | Real ACF peak lag | Synthetic ACF peak lag | ACF lag diff | Real PSD peak Hz | Synthetic PSD peak Hz | PSD diff Hz | Real std mean | Synthetic std mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walking | 116 | 58 | 58 | 1.6667 | 1.6667 | 0.0000 | 0.5540 | 0.6541 |
| running | 81 | 82 | 1 | 1.3333 | 1.3333 | 0.0000 | 0.8679 | 0.5991 |

Additional range checks:

| Activity | Real abs max | Synthetic abs max |
|---|---:|---:|
| walking | 4.5153 | 5.3352 |
| running | 4.1307 | 2.4701 |

## 6. How to Read the Figures

Figure files:

- Walking PNG: `outputs/acf-psd-comparison-20260522/pamap2_subject101_walking_hand_acc16_x_acf_psd_comparison.png`
- Running PNG: `outputs/acf-psd-comparison-20260522/pamap2_subject101_running_hand_acc16_x_acf_psd_comparison.png`

Left panel:

- X-axis: lag in samples.
- Y-axis: autocorrelation.
- Blue: real mean ACF.
- Pink: synthetic mean ACF.
- Shaded region: window-level variation.
- Dotted black line: expected SDForger period from preprocessing.

Right panel:

- X-axis: frequency in Hz.
- Y-axis: PSD power.
- Blue: real mean PSD.
- Pink: synthetic mean PSD.

## 7. Interpretation

Running is the stronger baseline case. Real running has a clear ACF peak at lag 81 and synthetic has a peak at lag 82. The PSD peak is identical at 1.3333 Hz. This supports the statement that SDForger preserved the dominant running rhythm in this clean univariate setting.

Walking is more mixed. The PSD peak matches exactly at 1.6667 Hz, so the dominant frequency is preserved. However, the real mean ACF's strongest peak is at 116, while synthetic is at 58. Since 116 is exactly 2 x 58, this looks like a harmonic / two-cycle structure mismatch rather than a completely wrong frequency. The synthetic walking samples capture the shorter base period but underrepresent the stronger two-cycle repetition seen in the real mean ACF.

The amplitude checks also differ by activity. Walking synthetic has a slightly larger absolute max than real, so some generated walking windows may be high-amplitude outliers. Running synthetic has lower average window variance than real, suggesting it may be smoother or less variable than real running despite matching the main rhythm.

## 8. Conclusion and Discussion

This ACF/PSD diagnostic strengthens the baseline verification:

- running-only `hand_acc16_x` is a good clean baseline case because both time-domain and frequency-domain periodicity match.
- walking-only `hand_acc16_x` is usable but less clean because it matches frequency while differing in ACF peak structure.

For a Monday progress report, the safe statement is:

> The clean SDForger univariate baseline preserves dominant periodic structure on running and partially on walking. Running shows strong ACF/PSD agreement; walking matches dominant frequency but differs in the autocorrelation harmonic structure.

## 9. Limitations and Caveats

- This is a one-subject, one-channel diagnostic.
- The comparison uses standardized SDForger window space, not raw sensor units.
- ACF/PSD agreement does not prove generated data is useful for HAR classification.
- The PSD frequency resolution is limited by 300-sample windows.
- Mean curves can hide good and bad individual samples.

## 10. Next Steps

1. Build a small good/bad sample taxonomy using ACF lag distance, PSD peak distance, and amplitude range.
2. Add this ACF/PSD figure to the Monday presentation as baseline verification.
3. Design a minimal HAR utility smoke for walking vs running using real-only, synthetic-only, and real+synthetic training.

## Reproducibility Notes

Commands were run on Puhti under:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt`

Environment:

`/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312`

Local script:

`scripts/compare_sdforger_acf_psd.py`
