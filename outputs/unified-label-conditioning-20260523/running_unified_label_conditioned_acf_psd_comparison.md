# ACF / PSD Comparison - running hand_acc16_x

- Real parquet: `data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet`
- Synthetic JSONL: `output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x/running_final_data.jsonl`
- Plot PNG: `reports/unified_label_conditioning_20260523/running_unified_label_conditioned_acf_psd_comparison.png`
- Plot PDF: `reports/unified_label_conditioning_20260523/running_unified_label_conditioned_acf_psd_comparison.pdf`
- Value space: standardized SDForger window space

| Metric | Value |
|---|---:|
| `activity` | `running` |
| `channel` | `hand_acc16_x` |
| `real_windows` | `31` |
| `synthetic_windows` | `79` |
| `window_length` | `300` |
| `sampling_rate_hz` | `100.000000` |
| `expected_period` | `82` |
| `real_mean_acf_peak_lag` | `81.000000` |
| `real_mean_acf_peak_score` | `0.762881` |
| `synthetic_mean_acf_peak_lag` | `80.000000` |
| `synthetic_mean_acf_peak_score` | `0.405703` |
| `acf_peak_lag_abs_diff` | `1.000000` |
| `real_mean_psd_peak_hz` | `1.333333` |
| `real_mean_psd_peak_power` | `0.749456` |
| `synthetic_mean_psd_peak_hz` | `2.333333` |
| `synthetic_mean_psd_peak_power` | `8680.467051` |
| `psd_peak_hz_abs_diff` | `1.000000` |
| `real_window_std_mean` | `0.867915` |
| `synthetic_window_std_mean` | `15.197219` |
| `synthetic_abs_max` | `1572.093073` |
| `real_abs_max` | `4.130692` |
