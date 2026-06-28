# Generation-time Latent Validity Plan

日期：2026-06-08

## 实验问题

`clip_p05_p95` 很简单有效，但它是 post-hoc repair。下一步测试两个更接近 generation-time control 的规则：

1. `strict_reject_p05_p95`
2. `soft_repair_p05_p95_minmax`

目标不是换模型，而是判断 latent validity 规则是否能保留 `clip_p05_p95` 的稳定性，同时减少“事后硬修补”的解释风险。

## 当前 artifact 边界

现有 SDForger 输出已经经过 parser filter：

- malformed text / 维度错误的输出已经被过滤；
- 当前 CSV 中只有 parser-accepted generated latent rows。

因此本轮实验是在 parser-accepted latent 上模拟 generation-time validity：

```text
LLM output -> parser filter -> latent validity rule -> decode
```

它不能统计原始 malformed text 的完整拒绝率，但能统计数值 validity 的 accepted/repaired/rejected count。

## Rules

### strict_reject_p05_p95

训练 latent 每一维计算：

- lower = 5% quantile
- upper = 95% quantile

规则：

- 每一维都在 p05-p95 内：accept
- 任意一维超出：reject
- 不做 clip

### soft_repair_p05_p95_minmax

训练 latent 每一维计算：

- soft bound = p05-p95
- hard bound = train min-max

规则：

- 每一维都在 p05-p95 内：clean accept
- 如果有维度超出 p05-p95，但所有维度仍在 train min-max 内：clip 到 p05-p95 后 accept，记为 repaired
- 任意维度超出 train min-max：reject

这个规则比纯 `clip_p05_p95` 更保守，因为它不会修复超出训练 min-max 的极端 latent。

## Run matrix

| Run | Variant | Real reference | Purpose |
|---|---|---|---|
| V1 | strict reject | train subjects 101/102/105 | 检查严格拒绝是否样本过少 |
| V2 | soft repair | train subjects 101/102/105 | 检查 soft repair 是否接近 post-hoc clip |
| V3 | strict reject | held-out subjects 106/108 | 检查严格规则的 unseen 表现 |
| V4 | soft repair | held-out subjects 106/108 | 检查 soft repair 的 unseen 表现 |

## Metrics

主指标：

- MDD
- ACD
- SD
- KD
- ED
- DTW

机制指标：

- input rows
- clean accept rows
- repaired rows
- rejected rows
- output rows
- decoded abs max
- decoded std mean

## Decision rule

继续 soft repair，如果：

- output rows 不明显低于 `clip_p05_p95`；
- ED/DTW/KD 接近 `clip_p05_p95`；
- ACD 没有明显恶化；
- rejected/repaired counts 可解释。

如果 strict reject 样本数过少，则只作为 diagnostic，不作为主方法。

## Scripts

- `scripts/apply_generation_time_latent_validity.py`
- `scripts/rerun_validity_variants_tsgbench_table.py`

## Outputs

- Remote：`/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/validity_variants_tsgbench_20260608/`
- Local：`outputs/generation-time-validity-20260608/`
