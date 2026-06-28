# Current Result Cleanup Summary

日期：2026-06-07

## One-Line Project State

我们已经证明 SDForger 可以迁移到 PAMAP2 subject101 的 walking/running 单通道 smoke；activity conditioning 有信号，但 unified generation 目前不稳定。下一步不是扩实验，而是统一 raw-like evaluation。

## Main Evaluation Rule

主结果只看：

> real raw windows vs inverse-normalized synthetic raw-like windows

Model-space 指标只用于调试，不放在主结论里。

统一协议：

`docs/experiments/unified-evaluation-protocol-20260607.md`

## Keep As Main / Provisional Evidence

| Result | How to use |
|---|---|
| clean univariate baseline | 说明 SDForger 在 PAMAP2 walking/running 单通道上可跑通 |
| HAR utility smoke | 只能说 within-subject smoke 有 task-discriminative signal |
| unified label conditioning | 只能说 label signal 存在，但 generation quality 不稳定 |
| latent constraint | 只能说 latent/value outlier 是重要 failure source，post-hoc constraint 有诊断价值 |

## Downgrade To Diagnostic-Only

| Result | Why |
|---|---|
| normalization ablation model-space metrics | 没有 inverse-normalize 到统一 raw-like space |
| model-space overlay | 可解释形状，但不是 final sensor-quality comparison |
| model-space TSGBench-style table | 不同 normalization setting 的空间不一致 |
| good/bad sample labels | heuristic threshold，不是标准 benchmark |

## Do Not Claim

- SDForger + PAMAP2 unified conditional generation 已经稳定成功。
- normalization 修复了 value explosion。
- `global_series_zscore` 是最佳方法。
- synthetic data 已经稳定提升 HAR augmentation。
- 当前结果支持 unseen-subject generalization。

## Safe Claims For Advisor Update

1. Baseline rebuilt:
   - PAMAP2 subject101 walking/running, `hand_acc16_x`, clean univariate SDForger baseline 已跑通。
2. Conditional direction is meaningful:
   - unified label conditioning 显示 activity label signal，但 raw generation 有 severe value outliers。
3. Failure source narrowed:
   - latent/value outlier 是主要问题之一；post-hoc latent constraints 可以恢复数值稳定性，但还不是最终方法。
4. Evaluation cleanup:
   - 已发现之前 model-space 与 raw-like evaluation 混用的问题，后续主评估将统一到 raw-like sensor space。

## Current Priority

先完成 ACT-034：

> 对 clean baseline、raw unified、`clip_p05_p95`、`global_series_zscore` 做统一 raw-like evaluation。

然后再决定：

- 是否做 generation-time latent validity / rejection sampling；
- 是否进入 multi-subject；
- 是否扩展 multichannel。
