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
| `walking_synthetic_all` | `35` |
| `running_synthetic_all` | `43` |
| `walking_synthetic_good` | `8` |
| `running_synthetic_good` | `17` |
| `synthetic_space_handling` | `inverse_transformed_from_sdforger_scaled_space_using_activity_train_window_mean_std` |

## Results

| Condition | Train samples | Accuracy | Balanced accuracy | Macro F1 | Confusion matrix [[walk, run], ...] |
|---|---:|---:|---:|---:|---|
| `majority` | 62 | 0.5135 | 0.5000 | 0.3393 | `[[57, 0], [54, 0]]` |
| `real-only` | 62 | 0.6126 | 0.6087 | 0.6022 | `[[43, 14], [29, 25]]` |
| `synthetic-only-all` | 78 | 0.6577 | 0.6516 | 0.6361 | `[[50, 7], [31, 23]]` |
| `real+synthetic-all` | 140 | 0.6486 | 0.6472 | 0.6468 | `[[40, 17], [22, 32]]` |
| `synthetic-only-good` | 25 | 0.6486 | 0.6404 | 0.6073 | `[[54, 3], [36, 18]]` |
| `real+synthetic-good` | 87 | 0.6757 | 0.6735 | 0.6725 | `[[43, 14], [22, 32]]` |
