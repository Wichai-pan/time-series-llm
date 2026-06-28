# Unified Raw-like Evaluation Results

日期：2026-06-07

## 目的

把当前最关键的 4 个 setting 放到同一套主评估口径下比较：

> real raw windows vs inverse-normalized synthetic raw-like windows

这一步用于清理之前混用 model-space / standardized-space / raw-like-space 造成的解释混乱。

## 输入 setting

| Setting | 含义 | 条件控制 |
|---|---|---|
| `clean_unconditioned` | walking 和 running 分开训练的 clean SDForger univariate baseline | 无 |
| `raw_unified_label_conditioned` | walking/running 合在一起训练，prompt 中加 activity label | 有 |
| `clip_p05_p95` | 对 raw unified 的 generated latent 做训练分布 5%-95% clipping 后 decode | 有 |
| `global_series_zscore` | embed 前做 global scalar z-score 的 unified label-conditioned run | 有 |

## 输出位置

- 本地结果目录：`outputs/unified-raw-like-evaluation-20260607/`
- 远程结果目录：`/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_unified_raw_like_eval_20260607/`
- 评估脚本：`scripts/evaluate_unified_raw_like_metrics.py`

## 关键结论

1. `clean_unconditioned` 仍是当前最稳的 baseline。
   - running 的 amplitude ratio 约 1.03，ACF lag diff 0，PSD diff 0。
   - HAR `real+synthetic-all` accuracy 为 0.7117，高于 real-only 0.6126。

2. `raw_unified_label_conditioned` 证明 label signal 存在，但生成数值严重失败。
   - label accuracy 为 0.8212。
   - walking amplitude ratio 约 1081x，running 约 408x。
   - HAR `real+synthetic-all` accuracy 只有 0.5315，低于 real-only。

3. `clip_p05_p95` 是目前最有价值的 method diagnostic。
   - walking amplitude ratio 从 1081x 降到 1.22x。
   - running amplitude ratio 从 408x 降到 1.24x。
   - label accuracy 提升到 0.9868。
   - running ACF/PSD 都匹配；walking PSD 匹配，但 ACF lag 仍不匹配。
   - HAR `synthetic-only-all` accuracy 为 0.6937，但 `real+synthetic-all` 为 0.5946，说明作为 augmentation 还不稳定。

4. `global_series_zscore` 在 raw-like 主评估下不可作为当前改进方向。
   - running rhythm 看似接近，但 raw-like amplitude ratio 仍约 150x。
   - HAR `real+synthetic-all` accuracy 只有 0.2793。

## 主表摘录

| Setting | Activity | Amp ratio | ACF lag diff | PSD diff | HAR real+synthetic acc | Label acc |
|---|---|---:|---:|---:|---:|---:|
| clean unconditioned | walking | 2.07 | 0 | 0.00 | 0.7117 | n/a |
| clean unconditioned | running | 1.03 | 0 | 0.00 | 0.7117 | n/a |
| raw unified label | walking | 1081.48 | 60 | 0.67 | 0.5315 | 0.8212 |
| raw unified label | running | 408.00 | 0 | 1.00 | 0.5315 | 0.8212 |
| clip p05-p95 | walking | 1.22 | 59 | 0.00 | 0.5946 | 0.9868 |
| clip p05-p95 | running | 1.24 | 0 | 0.00 | 0.5946 | 0.9868 |
| global z-score | walking | 104.65 | 24 | 0.67 | 0.2793 | 0.4553 |
| global z-score | running | 150.54 | 1 | 0.00 | 0.2793 | 0.4553 |

## TSGBench-style 指标说明

重要更正：本报告里的 `MDD/ACD/SD/KD/ED/DTW` 是在 raw-like inverse-normalized sensor space 上计算的诊断指标，不能替代两周前 scaled SDForger space 的主 TSG 表，也不能和旧截图中的 TSG 数值直接比较。

用于 slides / advisor update 的主 TSG 表应使用：

- `outputs/scaled-tsgbench-rerun-20260607/scaled_tsgbench_rerun_summary.csv`
- `docs/experiments/scaled-tsgbench-rerun-20260607.md`

当前 raw-like 指标只用于补充说明 value explosion / inverse normalization 后的 raw-value 合理性。

raw-like 诊断指标的结论和 value/rhythm/HAR 一致：

- raw unified 的 `ACD`、`SD`、`KD`、`ED` 明显差，说明不只是图不好看，而是统计结构也失败。
- clip 后 walking/running 的 shape/statistics 明显改善，但 walking 的 autocorrelation 仍有问题。
- global z-score 在 raw-like 下仍存在 value-scale failure。

## 可用于汇报的话

可以说：

> 我们重新统一了评估口径，把所有 synthetic outputs 都映射回 raw-like sensor space 后再比较。结果显示，clean SDForger baseline 在 PAMAP2 subject101 walking/running 单通道上是可用基线；unified label conditioning 可以控制 activity label，但会造成 latent/value outlier；简单的 p05-p95 latent clipping 能修复数值爆炸并保留 label controllability，但还不能稳定提升 HAR augmentation。

不要说：

> normalization 已经改善了生成质量。

也不要说：

> clip_p05_p95 已经是最终方法。

更准确的说法是：

> clip_p05_p95 暴露了一个有价值的方向：generation-time latent validity control / rejection sampling。

## 下一步

1. Slides 中使用 scaled-space TSG 主表，不使用 raw-like TSG 诊断表作为主表。
2. 把 `clip_p05_p95` 从 post-hoc clipping 改成 generation-time validity control。
3. 在 subject 扩展前，先加 roughness / derivative energy 检查 clipping 是否过度平滑。
4. 如果时间允许，再做 subject-level split；不要先扩大到多 subject 再修评估协议。
