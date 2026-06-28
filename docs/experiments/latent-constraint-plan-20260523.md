# Latent Constraint Diagnostic Plan - Unified Label Conditioning

日期：2026-05-23

## Diagnosis

Unified label conditioning 的结果是 mixed：

- positive：requested-label controllability accuracy 0.8212。
- negative：decoded signal amplitude 爆炸，walking abs max 3761.8，running abs max 1572.1。
- negative：PSD peak 都偏到 2.3333Hz，HAR utility all-synthetic 低于 real-only。

这更像 `latent/value outlier` 问题，而不是 label conditioning 完全无效。

## Direction Check

继续修 latent/value constraint 是合理的，但必须明确：

- 这是后处理/约束机制实验，不是重新训练出的完整方法。
- 如果 clipping 后质量恢复，只能说明“outlier 是主要失败源”，不能说明模型天然稳定。
- 如果 clipping 后 controllability 消失，说明原 controllability 可能依赖异常幅度或 artifact。
- 如果 clipping 后质量和 controllability 都保留，才值得把 constraint 变成正式 generation-time rejection / constrained decoding 机制。

## Experiment Question

> Training-latent-distribution constraints 能否修复 unified label-conditioned outputs 的幅度失控，同时保留 label controllability 和 HAR utility？

## Variants

| Variant | Mechanism | Purpose |
|---|---|---|
| raw unified | no constraint | failed reference |
| `clip_minmax` | clip each generated latent dimension to training min/max | strongest safe range constraint |
| `clip_p05_p95` | clip each latent dimension to training 5%-95% quantile | stricter robust constraint |
| `reject_iqr3` | keep only rows inside Q1-3IQR to Q3+3IQR for every dimension | rejection-style quality control |

## Metrics

Same as unified experiment:

- generated count per label
- ACF/PSD
- TSGBench-style metrics
- good/borderline/bad sample split
- HAR utility smoke
- label controllability

## Decision Rule

Promising if a constrained variant has:

- synthetic abs max close to real standardized range, not thousands,
- PSD peak closer to real,
- more good samples than raw unified,
- controllability still clearly above majority,
- HAR utility no worse than real-only for at least real+synthetic-good.

If clipping helps but rejection leaves too few samples, next step should be generation-time rejection sampling rather than fixed clipping.
