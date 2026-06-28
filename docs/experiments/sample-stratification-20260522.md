# Synthetic Sample Stratification - Clean Univariate Baseline

日期：2026-05-22

## Summary

- 目的：把 walking/running clean baseline 生成样本按 ACF lag、PSD peak、方差和幅度范围分成 good / borderline / bad。
- 输入：PAMAP2 subject101 `hand_acc16_x` 的 walking-only 和 running-only SDForger synthetic windows。
- 输出：每个 synthetic sample 的诊断指标、quality label，以及 best/worst 示例图。
- 结论：running 有 42/98 个 good 样本，主周期通常正确，但不少 bad 样本频率偏到 2.33Hz 且幅度过小；walking 有 57/130 个 good 样本，bad 样本主要来自方差过大、幅度异常或周期偏差。

## Setup

| Item | Value |
|---|---|
| Dataset | PAMAP2 subject101 |
| Activities | walking, running |
| Channel | `hand_acc16_x` |
| Real windows | 31 per activity |
| Synthetic windows | walking 130, running 98 |
| Value space | standardized SDForger window space |
| Script | `scripts/stratify_sdforger_samples.py` |
| Local artifacts | `outputs/sample-stratification-20260522/` |

## Rule

每个 synthetic window 单独计算：

- ACF peak lag
- PSD peak Hz
- window std
- max absolute value

参考值来自真实窗口的 median / quantile。样本若同时满足周期、频率、方差和幅度条件，则标为 `good`；部分满足但不够稳定标为 `borderline`；明显偏离标为 `bad`。

## Results

| Activity | Synthetic samples | Good | Borderline | Bad |
|---|---:|---:|---:|---:|
| walking | 130 | 57 | 64 | 9 |
| running | 98 | 42 | 14 | 42 |

Reference values:

| Activity | Real ACF lag median | Real PSD Hz median | Real std median | Real abs max q95 |
|---|---:|---:|---:|---:|
| walking | 115 | 1.6667 | 0.4384 | 3.8356 |
| running | 82 | 1.3333 | 0.8473 | 3.8962 |

## Interpretation

Running has a strong average ACF/PSD match, but individual sample quality is uneven. The best samples have ACF lag 82 and PSD peak 1.3333Hz, matching the real reference well. Many bad running samples still have ACF lag 82, but their PSD peak shifts to 2.3333Hz and their standard deviation is too small, meaning they are periodic but too weak/smooth or frequency-shifted.

Walking has more good/borderline samples and fewer clear bad samples under this diagnostic rule. Its best samples often match PSD 1.6667Hz and either the expected period around 58 or the real median ACF harmonic around 115. The worst samples are mostly caused by excessive variance, amplitude anomalies, or frequency shift.

## Artifacts

- Walking summary: `outputs/sample-stratification-20260522/walking_hand_acc16_x_sample_stratification.md`
- Running summary: `outputs/sample-stratification-20260522/running_hand_acc16_x_sample_stratification.md`
- Walking best examples: `outputs/sample-stratification-20260522/walking_hand_acc16_x_best_samples.png`
- Walking worst examples: `outputs/sample-stratification-20260522/walking_hand_acc16_x_worst_samples.png`
- Running best examples: `outputs/sample-stratification-20260522/running_hand_acc16_x_best_samples.png`
- Running worst examples: `outputs/sample-stratification-20260522/running_hand_acc16_x_worst_samples.png`

## Caveats

- This is a diagnostic quality split, not a training filter yet.
- Thresholds are heuristic and should be presented as sample analysis, not as a final benchmark.
- Good/bad labels are in standardized SDForger window space.
- The result does not yet answer HAR utility.

## Next Step

Use the best/worst images in the Monday presentation to explain that SDForger can generate plausible periodic samples, but quality is uneven. The next scientific step is a minimal HAR utility smoke using real-only, synthetic-only, and real+synthetic training.
