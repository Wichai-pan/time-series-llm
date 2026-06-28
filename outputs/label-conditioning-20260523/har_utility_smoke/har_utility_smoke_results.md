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
| `running_synthetic_all` | `90` |
| `walking_synthetic_good` | `33` |
| `running_synthetic_good` | `41` |
| `synthetic_space_handling` | `inverse_transformed_from_sdforger_scaled_space_using_activity_train_window_mean_std` |

## Results

| Condition | Train samples | Accuracy | Balanced accuracy | Macro F1 | Confusion matrix [[walk, run], ...] |
|---|---:|---:|---:|---:|---|
| `majority` | 62 | 0.5135 | 0.5000 | 0.3393 | `[[57, 0], [54, 0]]` |
| `real-only` | 62 | 0.6126 | 0.6087 | 0.6022 | `[[43, 14], [29, 25]]` |
| `synthetic-only-all` | 162 | 0.7387 | 0.7329 | 0.7236 | `[[54, 3], [26, 28]]` |
| `real+synthetic-all` | 224 | 0.7117 | 0.7086 | 0.7063 | `[[47, 10], [22, 32]]` |
| `synthetic-only-good` | 74 | 0.7207 | 0.7139 | 0.6987 | `[[55, 2], [29, 25]]` |
| `real+synthetic-good` | 136 | 0.6396 | 0.6374 | 0.6361 | `[[41, 16], [24, 30]]` |
