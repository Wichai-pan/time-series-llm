# ACF / PSD Comparison - running hand_acc16_x

- Real parquet: `data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet`
- Synthetic JSONL: `output/time_series/pamap2_subject101_running_hand_acc16_x_label_conditioned_univariate/final_data.jsonl`
- Plot PNG: `reports/label_conditioning_20260523/running_label_conditioned_acf_psd_comparison.png`
- Plot PDF: `reports/label_conditioning_20260523/running_label_conditioned_acf_psd_comparison.pdf`
- Value space: standardized SDForger window space

| Metric | Value |
|---|---:|
| `activity` | `running` |
| `channel` | `hand_acc16_x` |
| `real_windows` | `31` |
| `synthetic_windows` | `90` |
| `window_length` | `300` |
| `sampling_rate_hz` | `100.000000` |
| `expected_period` | `82` |
| `real_mean_acf_peak_lag` | `81.000000` |
| `real_mean_acf_peak_score` | `0.762881` |
| `synthetic_mean_acf_peak_lag` | `82.000000` |
| `synthetic_mean_acf_peak_score` | `0.968343` |
| `acf_peak_lag_abs_diff` | `1.000000` |
| `real_mean_psd_peak_hz` | `1.333333` |
| `real_mean_psd_peak_power` | `0.749456` |
| `synthetic_mean_psd_peak_hz` | `1.333333` |
| `synthetic_mean_psd_peak_power` | `0.555264` |
| `psd_peak_hz_abs_diff` | `0.000000` |
| `real_window_std_mean` | `0.867915` |
| `synthetic_window_std_mean` | `0.604229` |
| `synthetic_abs_max` | `2.431867` |
| `real_abs_max` | `4.130692` |
