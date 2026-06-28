# HAR Utility Smoke Results

## Metadata

| Field | Value |
|---|---:|
| `channel` | `hand_acc16_x` |
| `train_length` | `5000` |
| `window_length` | `300` |
| `walking_real_train_windows` | `31` |
| `running_real_train_windows` | `31` |
| `walking_real_test_windows` | `57` |
| `running_real_test_windows` | `54` |
| `walking_synthetic_all` | `72` |
| `running_synthetic_all` | `79` |
| `walking_synthetic_good` | `8` |
| `running_synthetic_good` | `18` |
| `synthetic_space_handling` | `inverse_transformed_from_sdforger_scaled_space_using_activity_train_window_mean_std` |

## Results

| Condition | Train samples | Accuracy | Balanced accuracy | Macro F1 | Confusion matrix [[walk, run], ...] |
|---|---:|---:|---:|---:|---|
| `majority` | 62 | 0.5135 | 0.5000 | 0.3393 | `[[57, 0], [54, 0]]` |
| `real-only` | 62 | 0.6126 | 0.6087 | 0.6022 | `[[43, 14], [29, 25]]` |
| `synthetic-only-all` | 151 | 0.4955 | 0.5010 | 0.4768 | `[[17, 40], [16, 38]]` |
| `real+synthetic-all` | 213 | 0.5315 | 0.5346 | 0.5269 | `[[24, 33], [19, 35]]` |
| `synthetic-only-good` | 26 | 0.6486 | 0.6404 | 0.6073 | `[[54, 3], [36, 18]]` |
| `real+synthetic-good` | 88 | 0.6757 | 0.6735 | 0.6725 | `[[43, 14], [22, 32]]` |
