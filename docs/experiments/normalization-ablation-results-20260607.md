# Normalization Ablation Results - PAMAP2 subject101 unified label conditioning

日期：2026-06-07

## Summary

Puhti Slurm array job `34758239` completed successfully for all four normalization modes.

Evaluation status:

> Diagnostic-only. These results are measured in each setting's model space and should not be used as the main raw-sensor quality table.

Main result under the new unified protocol:

> Needs raw-like inverse-normalized re-evaluation.

Diagnostic result:

> Pre-embedding / FICA-input normalization did not solve unified conditional generation instability.

The strongest diagnostic finding is that all tested modes still produce severe synthetic value outliers. `global_series_zscore` is the least bad among the new variants, but its synthetic amplitude is still roughly 150-180x larger than the corresponding real model-space amplitude.

Primary evaluation protocol:

`docs/experiments/unified-evaluation-protocol-20260607.md`

## Runs

| Mode | Status | Walking samples | Running samples | Main outcome |
|---|---:|---:|---:|---|
| `current_activity_window_zscore` | completed | 72 | 79 | Reproduces prior raw-unified failure |
| `joint_window_zscore` | completed | 55 | 99 | Running explodes dramatically |
| `global_series_zscore` | completed | 61 | 62 | Least bad value scale; running rhythm preserved |
| `activity_series_zscore` | completed | 57 | 59 | Walking PSD preserved, running still explodes |

## Diagnostic Value Stability and Rhythm

These metrics are computed in each setting's model space. They are useful for debugging value explosion, but they are not the final raw-like evaluation.

| Mode | Activity | Synthetic abs max | Amplitude ratio | Synthetic std mean | ACF lag diff | PSD Hz diff |
|---|---|---:|---:|---:|---:|---:|
| `current_activity_window_zscore` | walking | 3761.82 | 833.12 | 36.28 | 36 | 0.67 |
| `current_activity_window_zscore` | running | 1572.09 | 380.59 | 15.20 | 1 | 1.00 |
| `joint_window_zscore` | walking | 591.92 | 159.02 | 8.62 | 19 | 0.67 |
| `joint_window_zscore` | running | 187737.53 | 36876.85 | 641.21 | 0 | 0.33 |
| `global_series_zscore` | walking | 277.49 | 152.76 | 2.30 | 24 | 0.67 |
| `global_series_zscore` | running | 811.34 | 177.63 | 5.02 | 1 | 0.00 |
| `activity_series_zscore` | walking | 320.64 | 130.56 | 2.51 | 60 | 0.00 |
| `activity_series_zscore` | running | 6352.74 | 1646.52 | 49.39 | 37 | 0.33 |

Interpretation:

- `global_series_zscore` is the best of the tested normalization-only variants for running rhythm: running ACF lag diff is 1 and PSD diff is 0.
- However, even `global_series_zscore` has very large amplitude ratios, so it is not a usable generator.
- `joint_window_zscore` is actively harmful for running value scale in this smoke.
- `activity_series_zscore` keeps walking PSD but does not stabilize running.

## Diagnostic Label Controllability Check

The quick label controllability check here was run in each mode's saved model-space arrays. It is useful as a diagnostic, but it is not directly comparable to the previous raw-like inverse-transform label controllability number.

| Mode | Model-space label accuracy | Walking requested acc | Running requested acc |
|---|---:|---:|---:|
| `current_activity_window_zscore` | 0.4768 | 1.0000 | 0.0000 |
| `joint_window_zscore` | 0.5260 | 0.4000 | 0.5960 |
| `global_series_zscore` | 0.4553 | 0.3607 | 0.5484 |
| `activity_series_zscore` | 0.4655 | 0.4211 | 0.5085 |

Interpretation:

- In this model-space check, none of the normalization variants shows convincing label controllability.
- This weakens the case that normalization alone fixes unified label conditioning.
- Because the classifier protocol differs from the previous inverse-transformed controllability test, use this only as a diagnostic until a unified evaluation contract is finalized.

## Diagnostic TSGBench-style Metrics

These metrics are computed in each setting's saved model space. They are useful for diagnosing each normalization variant, but they are not a final raw-space benchmark because the model-space scale differs across variants.

| Mode | Activity | MDD | ACD | SD | KD | ED | DTW |
|---|---|---:|---:|---:|---:|---:|---:|
| `current_activity_window_zscore` | walking | 0.256933 | 5.972569 | 3.524391 | 118.647911 | 404.584401 | 6026.478759 |
| `current_activity_window_zscore` | running | 0.298711 | 2.284618 | 2.917641 | 102.877533 | 252.417079 | 3772.691067 |
| `joint_window_zscore` | walking | 0.396232 | 0.677717 | 7.461050 | 79.102913 | 77.631235 | 931.509542 |
| `joint_window_zscore` | running | 0.331390 | 4.064455 | 14.196718 | 215.205322 | 404.404257 | 6000.393147 |
| `global_series_zscore` | walking | 0.573907 | 1.287014 | 9.698354 | 149.908630 | 93.126458 | 1209.319115 |
| `global_series_zscore` | running | 0.490957 | 1.022686 | 10.943453 | 232.544220 | 45.431303 | 536.134086 |
| `activity_series_zscore` | walking | 0.343707 | 2.910986 | 9.114439 | 92.191292 | 18.768758 | 166.706596 |
| `activity_series_zscore` | running | 0.576832 | 2.192114 | 0.287718 | 149.053864 | 1565.446158 | 22343.653335 |

Interpretation:

- Lower is better for all listed TSGBench-style metrics.
- `global_series_zscore` running has the best running `DTW` among these unified normalization variants and also matched ACF/PSD in the rhythm diagnostic.
- `activity_series_zscore` walking has the lowest walking `ED/DTW`, but its rhythm diagnostic still has a large ACF lag mismatch.
- These metrics confirm that no single normalization mode dominates across activity, value stability, rhythm, and label controllability.

## Artifacts

Remote evaluation directory:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_norm_ablation_eval_20260607`

Local evaluation directory:

`outputs/normalization-ablation-20260607/pamap2_subject101_norm_ablation_eval_20260607/`

Main files:

- `normalization_value_rhythm_summary.csv`
- `normalization_label_controllability.csv`
- `normalization_tsgbench_style_metrics.csv`
- `normalization_ablation_evaluation.md`
- `overlays/*.png`

## Conclusion

This ablation suggests the current failure is not simply solved by changing pre-embedding normalization. However, because this report is diagnostic model-space evaluation, it should not be treated as the final comparison of generated PAMAP2 sensor data quality.

Recommended next step:

> First run unified raw-like evaluation under `docs/experiments/unified-evaluation-protocol-20260607.md`; then return to latent validity control / rejection sampling if the raw-like table confirms the same failure mode.

Do not claim:

- normalization fixes unified conditional generation;
- `global_series_zscore` is a stable generator;
- multi-subject expansion should begin before raw-like evaluation and value validity are controlled.
