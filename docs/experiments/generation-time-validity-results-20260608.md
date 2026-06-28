# Generation-time Latent Validity Results

日期：2026-06-08

## 实验目的

这轮实验测试两个比 post-hoc `clip_p05_p95` 更接近 generation-time control 的规则：

1. `strict_reject_p05_p95`
2. `soft_repair_p05_p95_minmax`

注意：现有 artifact 已经经过 SDForger parser filter，因此本轮是在 parser-accepted latent rows 上做数值 validity 规则。它不能统计原始 malformed text 的完整拒绝率。

## Validity 规则结果

| Variant | Activity | Input rows | Clean accept | Repaired | Rejected | Output rows | Decoded abs max | Decoded std mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| strict_reject_p05_p95 | walking | 51 | 14 | 0 | 37 | 14 | 1.958 | 0.458 |
| strict_reject_p05_p95 | running | 58 | 10 | 0 | 48 | 10 | 1.693 | 0.496 |
| soft_repair_p05_p95_minmax | walking | 51 | 14 | 13 | 24 | 27 | 2.847 | 0.624 |
| soft_repair_p05_p95_minmax | running | 58 | 10 | 24 | 24 | 34 | 2.494 | 0.558 |

解释：

- strict reject 太严格，walking 只保留 14/51，running 只保留 10/58。
- soft repair 比 strict reject 实用，但仍比 post-hoc clip 少很多样本。
- soft repair 的 decoded abs max 仍在合理 standardized range 内。

## TSG-style 结果

### Train-reference：subjects 101/102/105

| Setting | Activity | n | MDD ↓ | ACD ↓ | SD ↓ | KD ↓ | ED ↓ | DTW ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| strict_reject_p05_p95 | walking | 14 | 0.242 | 2.715 | 0.250 | 0.336 | 23.934 | 232.255 |
| strict_reject_p05_p95 | running | 10 | 0.324 | 1.415 | 0.687 | 0.492 | 16.029 | 125.246 |
| soft_repair_p05_p95_minmax | walking | 27 | 0.200 | 3.087 | 0.414 | 0.250 | 23.195 | 205.601 |
| soft_repair_p05_p95_minmax | running | 34 | 0.232 | 2.016 | 0.646 | 0.349 | 19.771 | 141.353 |

### Held-out reference：subjects 106/108

| Setting | Activity | n | MDD ↓ | ACD ↓ | SD ↓ | KD ↓ | ED ↓ | DTW ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| strict_reject_p05_p95 | walking | 14 | 0.259 | 1.648 | 0.194 | 0.765 | 16.100 | 117.918 |
| strict_reject_p05_p95 | running | 10 | 0.325 | 1.562 | 1.022 | 1.360 | 19.009 | 157.000 |
| soft_repair_p05_p95_minmax | walking | 27 | 0.218 | 1.974 | 0.358 | 0.679 | 18.915 | 140.000 |
| soft_repair_p05_p95_minmax | running | 34 | 0.256 | 2.615 | 0.981 | 1.218 | 20.320 | 157.195 |

Source:

- `outputs/generation-time-validity-20260608/generation_time_validity_summary.csv`
- `outputs/generation-time-validity-20260608/validity_variants_tsgbench_20260608/validity_variants_tsgbench_summary.csv`
- Remote report: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/validity_variants_tsgbench_20260608/`

## 与 post-hoc clip 的关系

Post-hoc `clip_p05_p95` 仍然是目前最简单、样本数最多、整体最稳的方案：

| Reference | Activity | post-hoc clip DTW ↓ | soft repair DTW ↓ | Comment |
|---|---|---:|---:|---|
| train 101/102/105 | walking | 166.065 | 205.601 | soft repair 更保守，但略差 |
| train 101/102/105 | running | 147.235 | 141.353 | soft repair 接近/略好 |
| held-out 106/108 | walking | 151.102 | 140.000 | soft repair 接近/略好 |
| held-out 106/108 | running | 146.689 | 157.195 | soft repair 略差 |

## 判断

1. `strict_reject_p05_p95` 不适合作为主方法。它指标有时不错，但样本数太少，容易变成 cherry-picking。
2. `soft_repair_p05_p95_minmax` 是合理候选。它比 post-hoc clip 更保守，并能记录 clean/repaired/rejected counts。
3. 但当前 soft repair 没有明显优于简单 `clip_p05_p95`。它的价值主要是方法解释更干净，而不是指标显著更好。
4. 当前最实际的路线是保留 `clip_p05_p95` 作为 simple strong diagnostic baseline，同时把 `soft_repair` 作为 generation-time validity 的初版方法候选。

## 下一步

1. 如果要继续方法改进，优先优化 soft repair 的 hard bound，而不是使用 strict reject。
2. 加入 generation loop，让 rejected samples 触发 resampling，目标是补回样本数。
3. 报告 accepted clean / repaired / rejected / malformed 四类比例。
4. 在 held-out HAR utility 上比较 post-hoc clip vs soft repair，而不是只看 TSG。
