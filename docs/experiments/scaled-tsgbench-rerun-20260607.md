# Scaled-space TSGBench Rerun

日期：2026-06-07

## 目的

复查 2026-05-22 / 2026-05-23 的 TSG 指标是否可复现，并解释为什么 raw-like evaluation 表与旧表数值不一致。

## 结论

旧 TSG 表可以复现。新表数值不一致的原因不是实验变了，而是评估空间变了：

- 旧表：`synthetic_space=scaled`，即 SDForger standardized window space。
- 新 raw-like 表：先 inverse-normalize 到 raw-like sensor units，再计算指标。

因此 slides / advisor update 中的主 TSG 表应使用本报告的 scaled-space rerun。

## 输出

- 本地目录：`outputs/scaled-tsgbench-rerun-20260607/`
- 远程目录：`/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/scaled_tsgbench_rerun_20260607/`
- 汇总表：`outputs/scaled-tsgbench-rerun-20260607/scaled_tsgbench_rerun_summary.csv`
- wrapper：`scripts/rerun_scaled_tsgbench_table.py`

## 复现表

| setting | activity | n | MDD | ACD | SD | KD | ED | DTW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | walking | 130 | 0.266287 | 0.165201 | 0.797960 | 1.667799 | 21.279075 | 237.760946 |
| baseline | running | 98 | 0.296503 | 0.497915 | 0.612561 | 0.444503 | 17.078956 | 111.087924 |
| label-v1 | walking | 72 | 0.275518 | 0.549948 | 1.841294 | 1.896685 | 23.914528 | 291.820 |
| label-v1 | running | 90 | 0.296924 | 0.584861 | 0.655061 | 0.220628 | 18.585121 | 141.005 |
| raw-unified | walking | 72 | 0.256933 | 5.972568 | 3.524391 | 118.647957 | 404.583810 | 6026.479 |
| raw-unified | running | 79 | 0.298711 | 2.284619 | 2.917641 | 102.877609 | 252.416690 | 3772.691 |
| clip_p05_p95 | walking | 72 | 0.273200 | 2.253004 | 0.516341 | 1.143765 | 20.165552 | 198.446 |
| clip_p05_p95 | running | 79 | 0.315895 | 1.959808 | 0.336292 | 0.422136 | 22.644891 | 139.695 |

## 解释

这张表说明：

- `baseline` 是最稳的参考，尤其 ACD 明显最好。
- `raw-unified` 的 ED/DTW/KD 严重爆炸，说明 unified label conditioning 直接生成不稳定。
- `clip_p05_p95` 显著修复 raw-unified 的 ED/DTW/KD，但 ACD 仍不如 baseline，说明自相关/周期结构尚未恢复。
- `label-v1` 没有明显优于 baseline，适合作为 activity label prompt 的 feasibility check，不适合作为最终改进 claim。

## 之后的口径

- 主 TSG 表：使用 scaled-space rerun。
- Value validity / raw amplitude 表：使用 raw-like evaluation。
- HAR utility：单独报告 downstream smoke。
