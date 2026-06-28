# Normalization Ablation TSGBench-style Metrics

These metrics use each normalization setting's saved model-space real and synthetic windows.
They are diagnostic metrics, not a final cross-setting raw-space benchmark.

| mode                           | activity   | status   |      MDD |      ACD |        SD |       KD |        ED |       DTW |   real_windows |   synthetic_windows |   paired_windows | dtw_status   |
|:-------------------------------|:-----------|:---------|---------:|---------:|----------:|---------:|----------:|----------:|---------------:|--------------------:|-----------------:|:-------------|
| current_activity_window_zscore | walking    | ok       | 0.256933 | 5.97257  |  3.52439  | 118.648  |  404.584  |  6026.48  |             31 |                  72 |               31 | computed     |
| current_activity_window_zscore | running    | ok       | 0.298711 | 2.28462  |  2.91764  | 102.878  |  252.417  |  3772.69  |             31 |                  79 |               31 | computed     |
| joint_window_zscore            | walking    | ok       | 0.396232 | 0.677717 |  7.46105  |  79.1029 |   77.6312 |   931.51  |             31 |                  55 |               31 | computed     |
| joint_window_zscore            | running    | ok       | 0.33139  | 4.06445  | 14.1967   | 215.205  |  404.404  |  6000.39  |             31 |                  99 |               31 | computed     |
| global_series_zscore           | walking    | ok       | 0.573907 | 1.28701  |  9.69835  | 149.909  |   93.1265 |  1209.32  |             31 |                  61 |               31 | computed     |
| global_series_zscore           | running    | ok       | 0.490957 | 1.02269  | 10.9435   | 232.544  |   45.4313 |   536.134 |             31 |                  62 |               31 | computed     |
| activity_series_zscore         | walking    | ok       | 0.343707 | 2.91099  |  9.11444  |  92.1913 |   18.7688 |   166.707 |             31 |                  57 |               31 | computed     |
| activity_series_zscore         | running    | ok       | 0.576832 | 2.19211  |  0.287718 | 149.054  | 1565.45   | 22343.7   |             31 |                  59 |               31 | computed     |
