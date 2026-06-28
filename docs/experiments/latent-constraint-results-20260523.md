# Latent / Value Constraint Results - Unified Label Conditioning

日期：2026-05-23

## 结论先行

这个方向是合理的，但现在还不能包装成最终方法。

本实验验证了一个关键判断：unified label conditioning 失败的主要原因之一确实是 generated latent values 出训练分布后，decode 到 time series 时造成 value explosion。对 latent 做训练分布约束后，几千级别的异常幅度被消除，label controllability 不但没有消失，反而明显提高。这说明 label signal 大概率是真实存在的，不只是异常幅度带来的 artifact。

但是，constraint 目前是 post-hoc 修复，不是 generation-time 机制；而且 walking 的节奏仍没有完全恢复。因此它适合作为下一步方法设计依据，不适合作为最终 augmentation claim。

## 实验问题

> 对 unified walking/running label-conditioned SDForger output 加 latent distribution constraint，能否修复幅度失控，同时保留 label controllability 和 HAR utility？

## 输入与设置

- Dataset：PAMAP2 subject101
- Activities：walking / running
- Channel：`hand_acc16_x`
- Window length：300
- Train length：5000
- Base failed run：`pamap2_subject101_unified_label_conditioned_hand_acc16_x`
- Embedding：joint FICA latent space
- Labels：`Condition: data is walking` / `Condition: data is running`

远程输出：

- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x_constraints/`
- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/latent_constraints_20260523/`

本地同步：

- `outputs/latent-constraints-20260523/`

脚本：

- `scripts/apply_unified_latent_constraints.py`

## Variants

| Variant | 机制 | 目的 |
|---|---|---|
| raw unified | 无约束 | 失败参考 |
| `clip_minmax` | 每个 latent 维度裁剪到训练 min/max | 宽松合法范围 |
| `clip_p05_p95` | 每个 latent 维度裁剪到训练 5%-95% 分位 | 更稳健的保守范围 |
| `reject_iqr3` | 只保留落在 Q1-3IQR 到 Q3+3IQR 内的样本 | 模拟 rejection filtering |

## Value Stability

raw unified 的最大问题是 decoded synthetic value 爆炸：

| Variant | Walking abs max | Running abs max | 说明 |
|---|---:|---:|---|
| raw unified | 3761.8 | 1572.1 | 不可用 |
| `clip_minmax` | 7.50 | 4.58 | 基本拉回真实 standardized 范围附近 |
| `clip_p05_p95` | 3.24 | 2.65 | 最稳定，但可能过度压缩幅度 |
| `reject_iqr3` | 2.50 | 3.12 | 稳定，但样本数减少很多 |

判断：latent/value outlier 确实是 unified run 的主要 failure source。尤其 `clip_p05_p95` 把幅度恢复到合理范围，同时不丢样本。

## ACF / PSD

| Variant | Walking ACF lag | Walking PSD Hz | Running ACF lag | Running PSD Hz |
|---|---:|---:|---:|---:|
| real | 116 | 1.6667 | 81 | 1.3333 |
| raw unified | needs fixing | 2.3333 | needs fixing | 2.3333 |
| `clip_minmax` | 80 | 1.3333 | 80 | 1.3333 |
| `clip_p05_p95` | 80 | 1.3333 | 80 | 1.3333 |
| `reject_iqr3` | 80 | 1.3333 | 79 | 1.3333 |

判断：

- running 被修得比较好：ACF lag 接近 81，PSD peak 匹配 1.3333 Hz。
- walking 仍有问题：真实 mean ACF peak 是 116，但 synthetic 约为 80；PSD 也偏到 1.3333 Hz，而不是 clean baseline 里的 1.6667 Hz。
- 所以 constraint 修复了 value scale，但没有完全修复 activity-specific rhythm。

## Label Controllability

| Variant | Overall acc | Walking requested acc | Running requested acc |
|---|---:|---:|---:|
| raw unified | 0.8212 | 0.8056 | 0.8354 |
| `clip_minmax` | 0.9470 | 0.9167 | 0.9747 |
| `clip_p05_p95` | 0.9868 | 1.0000 | 0.9747 |
| `reject_iqr3` | 0.9359 | 1.0000 | 0.8837 |

判断：constraint 后 controllability 没有崩，反而更强。这个结果很重要，因为它说明 unified label conditioning 不是完全错的；问题更像是 unconstrained latent generation 不稳定。

## HAR Utility Smoke

Held-out real test：walking 57 windows，running 54 windows。real-only baseline accuracy 0.6126。

| Variant | Synthetic-only all | Real+synthetic all | Synthetic-only good | Real+synthetic good |
|---|---:|---:|---:|---:|
| raw unified | 0.4955 | 0.5315 | n/a | 0.6757 |
| `clip_minmax` | 0.5766 | 0.6667 | 0.6306 | 0.6667 |
| `clip_p05_p95` | 0.6937 | 0.5946 | 0.7207 | 0.6667 |
| `reject_iqr3` | 0.6577 | 0.6486 | 0.6486 | 0.6757 |

判断：

- `clip_p05_p95` 的 synthetic-only 最强，accuracy 0.6937；good-only synthetic-only 到 0.7207。
- `real+synthetic-all` 不稳定，`clip_p05_p95` 反而低于 real-only，说明“直接把所有 synthetic 加进训练集”还不能作为可靠 augmentation claim。
- `real+synthetic-good` 三个 constraint 都高于 real-only，但提升幅度仍是 smoke-level evidence。

## Sample Quality

| Variant | Walking good / borderline / bad | Running good / borderline / bad |
|---|---|---|
| raw unified | 8 / 10 / 54 | 18 / 22 / 39 |
| `clip_minmax` | 10 / 19 / 43 | 31 / 21 / 27 |
| `clip_p05_p95` | 15 / 15 / 42 | 34 / 20 / 25 |
| `reject_iqr3` | 8 / 7 / 20 | 17 / 13 / 13 |

判断：constraint 明显改善 running，也略微改善 walking，但 walking bad samples 仍然多。这和 ACF/PSD 的观察一致。

## 方向审查

### 这个方向没有问题的地方

1. 它直接针对 unified run 的最大 failure mode：latent/value outliers。
2. 它没有破坏 label controllability，反而增强了 controllability。
3. 它让 synthetic-only HAR utility 从 raw unified 的失败状态恢复到可用 smoke signal。
4. 它和 SDForger 原文的 modular embedding / decoding 结构一致，属于合理 extension。

### 这个方向仍有问题的地方

1. 当前 constraint 是 post-hoc clipping/filtering，不是模型本身学会稳定生成。
2. Walking rhythm 仍没有恢复到 clean baseline 水平。
3. `real+synthetic-all` 仍不稳定，不能说 synthetic data 稳定提升 HAR classifier。
4. 只覆盖 subject101、单通道、两个 activity、一个 classifier。

## 当前最佳判断

Decision：`revise and continue`

最合理的下一步不是继续随意加 label，也不是马上扩到更多数据集，而是把 constraint 从 post-hoc diagnostic 变成正式 generation-time mechanism：

1. 生成 latent 后先做 validity check，再 decode。
2. 对 out-of-range latent 做 rejection 或 resampling。
3. 报告 generated / accepted / rejected counts。
4. 保留 `clip_p05_p95` 作为 conservative diagnostic baseline。
5. 重点看 walking rhythm 是否能恢复；如果不能，需要回到 embedding design，而不是继续调 constraint。

## 可用于汇报的一句话

Unified label conditioning 本身有信号，但 unconstrained latent generation 会产生严重 value outliers；训练分布约束能显著恢复数值稳定性和 label controllability，不过还不能稳定支持 HAR augmentation，因此下一步应转向 generation-time latent validity control。
