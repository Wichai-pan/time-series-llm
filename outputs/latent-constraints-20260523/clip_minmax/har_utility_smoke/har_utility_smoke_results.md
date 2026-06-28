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
| `walking_synthetic_good` | `10` |
| `running_synthetic_good` | `31` |
| `synthetic_space_handling` | `inverse_transformed_from_sdforger_scaled_space_using_activity_train_window_mean_std` |

## Results

| Condition | Train samples | Accuracy | Balanced accuracy | Macro F1 | Confusion matrix [[walk, run], ...] |
|---|---:|---:|---:|---:|---|
| `majority` | 62 | 0.5135 | 0.5000 | 0.3393 | `[[57, 0], [54, 0]]` |
| `real-only` | 62 | 0.6126 | 0.6087 | 0.6022 | `[[43, 14], [29, 25]]` |
| `synthetic-only-all` | 151 | 0.5766 | 0.5746 | 0.5731 | `[[37, 20], [27, 27]]` |
| `real+synthetic-all` | 213 | 0.6667 | 0.6608 | 0.6473 | `[[50, 7], [30, 24]]` |
| `synthetic-only-good` | 41 | 0.6306 | 0.6287 | 0.6276 | `[[40, 17], [24, 30]]` |
| `real+synthetic-good` | 103 | 0.6667 | 0.6642 | 0.6627 | `[[43, 14], [23, 31]]` |
