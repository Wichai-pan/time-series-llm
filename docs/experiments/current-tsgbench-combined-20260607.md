# Current TSGBench-style Tables

日期：2026-06-07

## 先说明

当前有两类 TSG-style 表，不能直接混排成一个统一排名：

1. `scaled SDForger space`：两周前主实验口径，可用于 baseline / label-v1 / raw-unified / clip 的直接比较。
2. `normalization-specific model space`：今天新增 normalization ablation 的诊断口径，只能比较 normalization variants 的 failure mode，不能直接和 baseline 主表比绝对数值。

## A. 主 TSG 表：scaled SDForger space

来源：

- `outputs/scaled-tsgbench-rerun-20260607/scaled_tsgbench_rerun_summary.csv`

| setting | activity | n | MDD | ACD | SD | KD | ED | DTW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | walking | 130 | 0.266287 | 0.165201 | 0.797960 | 1.667799 | 21.279075 | 237.760946 |
| baseline | running | 98 | 0.296503 | 0.497915 | 0.612561 | 0.444503 | 17.078956 | 111.087924 |
| label-v1 | walking | 72 | 0.275518 | 0.549948 | 1.841294 | 1.896688 | 23.914547 | 291.820399 |
| label-v1 | running | 90 | 0.296924 | 0.584861 | 0.655061 | 0.220628 | 18.585106 | 141.005387 |
| raw-unified | walking | 72 | 0.256933 | 5.972569 | 3.524391 | 118.647911 | 404.584401 | 6026.478759 |
| raw-unified | running | 79 | 0.298711 | 2.284618 | 2.917641 | 102.877533 | 252.417079 | 3772.691067 |
| clip_p05_p95 | walking | 72 | 0.273200 | 2.253002 | 0.516341 | 1.143768 | 20.165585 | 198.446299 |
| clip_p05_p95 | running | 79 | 0.315895 | 1.959807 | 0.336292 | 0.422136 | 22.644881 | 139.694653 |

解读：

- `baseline` 是最稳主参考，尤其 ACD 最好。
- `raw-unified` 的 ED/DTW/KD 明显崩。
- `clip_p05_p95` 修复了 raw-unified 的 ED/DTW/KD，但 ACD 仍明显不如 baseline。
- `label-v1` 作为 feasibility check 可以保留，但不能 claim 显著优于 baseline。

## B. 今天新增 normalization ablation：model-space diagnostic

来源：

- `outputs/normalization-ablation-20260607/pamap2_subject101_norm_ablation_eval_20260607/normalization_tsgbench_style_metrics.csv`

| setting | activity | n | MDD | ACD | SD | KD | ED | DTW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_activity_window_zscore | walking | 72 | 0.256933 | 5.972569 | 3.524391 | 118.647911 | 404.584401 | 6026.478759 |
| current_activity_window_zscore | running | 79 | 0.298711 | 2.284618 | 2.917641 | 102.877533 | 252.417079 | 3772.691067 |
| joint_window_zscore | walking | 55 | 0.396232 | 0.677717 | 7.461050 | 79.102913 | 77.631235 | 931.509542 |
| joint_window_zscore | running | 99 | 0.331390 | 4.064455 | 14.196718 | 215.205322 | 404.404257 | 6000.393147 |
| global_series_zscore | walking | 61 | 0.573907 | 1.287014 | 9.698354 | 149.908630 | 93.126458 | 1209.319115 |
| global_series_zscore | running | 62 | 0.490957 | 1.022686 | 10.943453 | 232.544220 | 45.431303 | 536.134086 |
| activity_series_zscore | walking | 57 | 0.343707 | 2.910986 | 9.114439 | 92.191292 | 18.768758 | 166.706596 |
| activity_series_zscore | running | 59 | 0.576832 | 2.192114 | 0.287718 | 149.053864 | 1565.446158 | 22343.653335 |

解读：

- `current_activity_window_zscore` 复现了 raw-unified failure，因此它和主表里的 raw-unified 数值一致。
- `global_series_zscore` 的 running DTW 最低，但 KD/SD 很差，且 raw-like value check 仍有严重 amplitude outlier。
- `activity_series_zscore` 的 walking ED/DTW 很低，但 running DTW 爆炸，不能作为整体改进。
- `joint_window_zscore` 对 running 很差，不建议继续。

## 当前可讲的结论

最准确的说法：

> 主 TSG 表显示，clip_p05_p95 能修复 raw-unified 的 shape-distance 和 kurtosis 爆炸，但 autocorrelation 仍不如 clean baseline。今天新增的 normalization ablation 没有找到一个稳定优于 clip/baseline 的方案；它最多说明 normalization 会改变 failure mode，但不能单独解决 unified conditional generation。

不要说：

> normalization 已经提升了生成质量。

也不要把 B 表和 A 表直接放在一起排名。
