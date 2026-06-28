# ACF / PSD Comparison - walking hand_acc16_x

- Real parquet: `data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet`
- Synthetic JSONL: `output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x_constraints/clip_p05_p95/walking_final_data.jsonl`
- Plot PNG: `reports/latent_constraints_20260523/clip_p05_p95/walking_acf_psd_comparison.png`
- Plot PDF: `reports/latent_constraints_20260523/clip_p05_p95/walking_acf_psd_comparison.pdf`
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
| `synthetic_mean_acf_peak_score` | `0.469990` |
| `acf_peak_lag_abs_diff` | `36.000000` |
| `real_mean_psd_peak_hz` | `1.666667` |
| `real_mean_psd_peak_power` | `0.382808` |
| `synthetic_mean_psd_peak_hz` | `1.333333` |
| `synthetic_mean_psd_peak_power` | `0.378818` |
| `psd_peak_hz_abs_diff` | `0.333333` |
| `real_window_std_mean` | `0.554010` |
| `synthetic_window_std_mean` | `0.617855` |
| `synthetic_abs_max` | `3.237178` |
| `real_abs_max` | `4.515344` |
