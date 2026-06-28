# Unified Raw-like Evaluation

## Metadata

| Field | Value |
|---|---|
| `channel` | `hand_acc16_x` |
| `train_length` | `5000` |
| `window_length` | `300` |
| `space` | `raw_like` |
| `settings` | `clean_unconditioned, raw_unified_label_conditioned, clip_p05_p95, global_series_zscore` |
| `skip_dtw` | `False` |

## Value / rhythm metrics

| setting                       | activity   |   real_windows |   synthetic_windows |   real_abs_max |   synthetic_abs_max |   real_std_mean |   synthetic_std_mean |   amplitude_ratio |   real_acf_lag |   synthetic_acf_lag |   acf_lag_diff |   real_acf_score |   synthetic_acf_score |   real_psd_hz |   synthetic_psd_hz |   psd_hz_diff |   real_psd_peak_power |   synthetic_psd_peak_power |
|:------------------------------|:-----------|---------------:|--------------------:|---------------:|--------------------:|----------------:|---------------------:|------------------:|---------------:|--------------------:|---------------:|-----------------:|----------------------:|--------------:|-------------------:|--------------:|----------------------:|---------------------------:|
| clean_unconditioned           | walking    |             31 |                 130 |        22.0997 |             45.8398 |         4.1155  |              4.17848 |           2.07423 |             58 |                  58 |              0 |         0.652775 |              0.695741 |       1.66667 |            1.66667 |      0        |               24.3153 |               29.8459      |
| clean_unconditioned           | running    |             31 |                  98 |        45.0969 |             46.5146 |         7.80982 |              7.94722 |           1.03144 |             82 |                  82 |              0 |         0.561483 |              0.935584 |       1.33333 |            1.33333 |      0        |               59.3965 |               60.2938      |
| raw_unified_label_conditioned | walking    |             31 |                  72 |        22.0997 |          23900.3    |         4.1155  |            183.389   |        1081.48    |             58 |                 118 |             60 |         0.652775 |              0.44496  |       1.66667 |            2.33333 |      0.666667 |               24.3153 |                1.22696e+06 |
| raw_unified_label_conditioned | running    |             31 |                  79 |        45.0969 |          18399.7    |         7.80982 |            121.211   |         408.003   |             82 |                  82 |              0 |         0.561483 |              0.705027 |       1.33333 |            2.33333 |      1        |               59.3965 |           459892           |
| clip_p05_p95                  | walking    |             31 |                  72 |        22.0997 |             27.0097 |         4.1155  |              4.04427 |           1.22218 |             58 |                 117 |             59 |         0.652775 |              0.462143 |       1.66667 |            1.66667 |      0        |               24.3153 |               18.1447      |
| clip_p05_p95                  | running    |             31 |                  79 |        45.0969 |             55.6976 |         7.80982 |              6.83973 |           1.23506 |             82 |                  82 |              0 |         0.561483 |              0.75425  |       1.33333 |            1.33333 |      0        |               59.3965 |               45.1027      |
| global_series_zscore          | walking    |             31 |                  61 |        22.0997 |           2312.64   |         4.1155  |             19.2206  |         104.646   |             58 |                  82 |             24 |         0.652775 |              0.516084 |       1.66667 |            1       |      0.666667 |               24.3153 |             6640.67        |
| global_series_zscore          | running    |             31 |                  62 |        45.0969 |           6788.99   |         7.80982 |             41.994   |         150.542   |             82 |                  81 |              1 |         0.561483 |              0.413075 |       1.33333 |            1.33333 |      0        |               59.3965 |            51512.6         |

## TSGBench-style metrics

| setting                       | activity   |       MDD |      ACD |       SD |          KD |        ED |       DTW |
|:------------------------------|:-----------|----------:|---------:|---------:|------------:|----------:|----------:|
| clean_unconditioned           | walking    | 0.0682536 | 0.515475 | 0.074572 |   2.46256   |  102.734  |  1038.02  |
| clean_unconditioned           | running    | 0.0605834 | 0.454324 | 0.510041 |   1.06243   |  118.624  |  1130.75  |
| raw_unified_label_conditioned | walking    | 0.0593249 | 1.63803  | 8.04818  |  76.5826    | 2083.06   | 29058.3   |
| raw_unified_label_conditioned | running    | 0.0588493 | 6.26439  | 8.93196  |  70.1688    | 1700.94   | 23350.1   |
| clip_p05_p95                  | walking    | 0.0638483 | 2.06225  | 0.683705 |   1.00947   |   99.0633 |   870.832 |
| clip_p05_p95                  | running    | 0.0632022 | 0.879482 | 0.547516 |   0.0658312 |  156.925  |  1385.84  |
| global_series_zscore          | walking    | 0.0716404 | 1.31315  | 6.79126  |  74.6259    |  778.456  | 10108.8   |
| global_series_zscore          | running    | 0.0621645 | 1.0367   | 7.48928  | 116.247     |  379.766  |  4481.61  |

## HAR utility

| condition          |   train_samples |   accuracy |   balanced_accuracy |   macro_f1 | confusion_matrix     | setting                       |
|:-------------------|----------------:|-----------:|--------------------:|-----------:|:---------------------|:------------------------------|
| majority           |              62 |   0.513514 |            0.5      |   0.339286 | [[57, 0], [54, 0]]   | clean_unconditioned           |
| real-only          |              62 |   0.612613 |            0.608674 |   0.602151 | [[43, 14], [29, 25]] | clean_unconditioned           |
| synthetic-only-all |             228 |   0.702703 |            0.697856 |   0.690547 | [[50, 7], [26, 28]]  | clean_unconditioned           |
| real+synthetic-all |             290 |   0.711712 |            0.707115 |   0.70101  | [[50, 7], [25, 29]]  | clean_unconditioned           |
| majority           |              62 |   0.513514 |            0.5      |   0.339286 | [[57, 0], [54, 0]]   | raw_unified_label_conditioned |
| real-only          |              62 |   0.612613 |            0.608674 |   0.602151 | [[43, 14], [29, 25]] | raw_unified_label_conditioned |
| synthetic-only-all |             151 |   0.495495 |            0.500975 |   0.476768 | [[17, 40], [16, 38]] | raw_unified_label_conditioned |
| real+synthetic-all |             213 |   0.531532 |            0.5346   |   0.526885 | [[24, 33], [19, 35]] | raw_unified_label_conditioned |
| majority           |              62 |   0.513514 |            0.5      |   0.339286 | [[57, 0], [54, 0]]   | clip_p05_p95                  |
| real-only          |              62 |   0.612613 |            0.608674 |   0.602151 | [[43, 14], [29, 25]] | clip_p05_p95                  |
| synthetic-only-all |             151 |   0.693694 |            0.688109 |   0.677326 | [[51, 6], [28, 26]]  | clip_p05_p95                  |
| real+synthetic-all |             213 |   0.594595 |            0.594055 |   0.594067 | [[35, 22], [23, 31]] | clip_p05_p95                  |
| majority           |              62 |   0.513514 |            0.5      |   0.339286 | [[57, 0], [54, 0]]   | global_series_zscore          |
| real-only          |              62 |   0.612613 |            0.608674 |   0.602151 | [[43, 14], [29, 25]] | global_series_zscore          |
| synthetic-only-all |             123 |   0.207207 |            0.211014 |   0.19246  | [[4, 53], [35, 19]]  | global_series_zscore          |
| real+synthetic-all |             185 |   0.279279 |            0.27924  |   0.279221 | [[16, 41], [39, 15]] | global_series_zscore          |

## Label controllability

| setting                       |   label_accuracy | note                  |   label_balanced_accuracy |   label_macro_f1 |   walking_requested_accuracy |   running_requested_accuracy | label_confusion_matrix   |
|:------------------------------|-----------------:|:----------------------|--------------------------:|-----------------:|-----------------------------:|-----------------------------:|:-------------------------|
| clean_unconditioned           |       nan        | not label-conditioned |                nan        |       nan        |                   nan        |                   nan        | nan                      |
| raw_unified_label_conditioned |         0.821192 | nan                   |                  0.820499 |         0.820689 |                     0.805556 |                     0.835443 | [[58, 14], [13, 66]]     |
| clip_p05_p95                  |         0.986755 | nan                   |                  0.987342 |         0.98674  |                     1        |                     0.974684 | [[72, 0], [2, 77]]       |
| global_series_zscore          |         0.455285 | nan                   |                  0.454521 |         0.45005  |                     0.360656 |                     0.548387 | [[22, 39], [28, 34]]     |

Lower is better for MDD/ACD/SD/KD/ED/DTW, ACF lag diff, PSD Hz diff, and amplitude ratio distance from 1. Higher is better for HAR and label accuracies.