# ACF / PSD Comparison - walking hand_acc16_x

- Real parquet: `data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet`
- Synthetic JSONL: `output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x/walking_final_data.jsonl`
- Plot PNG: `reports/unified_label_conditioning_20260523/walking_unified_label_conditioned_acf_psd_comparison.png`
- Plot PDF: `reports/unified_label_conditioning_20260523/walking_unified_label_conditioned_acf_psd_comparison.pdf`
- Value space: standardized SDForger window space

| Metric | Value |
|---|---:|
| `activity` | `walking` |
| `channel` | `hand_acc16_x` |
| `real_windows` | `31` |
| `synthetic_windows` | `72` |
| `window_length` | `300` |
| `sampling_rate_hz` | `100.000000` |
| `expected_period` | `58` |
| `real_mean_acf_peak_lag` | `116.000000` |
| `real_mean_acf_peak_score` | `0.499964` |
| `synthetic_mean_acf_peak_lag` | `80.000000` |
| `synthetic_mean_acf_peak_score` | `0.414051` |
| `acf_peak_lag_abs_diff` | `36.000000` |
| `real_mean_psd_peak_hz` | `1.666667` |
| `real_mean_psd_peak_power` | `0.382808` |
| `synthetic_mean_psd_peak_hz` | `2.333333` |
| `synthetic_mean_psd_peak_power` | `55238.421334` |
| `psd_peak_hz_abs_diff` | `0.666667` |
| `real_window_std_mean` | `0.554010` |
| `synthetic_window_std_mean` | `36.276732` |
| `synthetic_abs_max` | `3761.817972` |
| `real_abs_max` | `4.515344` |
