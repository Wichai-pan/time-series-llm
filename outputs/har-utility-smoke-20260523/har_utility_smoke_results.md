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
| `walking_synthetic_all` | `130` |
| `running_synthetic_all` | `98` |
| `walking_synthetic_good` | `57` |
| `running_synthetic_good` | `42` |
| `synthetic_space_handling` | `inverse_transformed_from_sdforger_scaled_space_using_activity_train_window_mean_std` |

## Results

| Condition | Train samples | Accuracy | Balanced accuracy | Macro F1 | Confusion matrix [[walk, run], ...] |
|---|---:|---:|---:|---:|---|
| `majority` | 62 | 0.5135 | 0.5000 | 0.3393 | `[[57, 0], [54, 0]]` |
| `real-only` | 62 | 0.6126 | 0.6087 | 0.6022 | `[[43, 14], [29, 25]]` |
| `synthetic-only-all` | 228 | 0.7027 | 0.6979 | 0.6905 | `[[50, 7], [26, 28]]` |
| `real+synthetic-all` | 290 | 0.7117 | 0.7071 | 0.7010 | `[[50, 7], [25, 29]]` |
| `synthetic-only-good` | 99 | 0.7207 | 0.7139 | 0.6987 | `[[55, 2], [29, 25]]` |
| `real+synthetic-good` | 161 | 0.6396 | 0.6374 | 0.6361 | `[[41, 16], [24, 30]]` |
