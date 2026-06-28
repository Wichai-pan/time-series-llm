# Unified Evaluation Protocol - PAMAP2 SDForger Experiments

日期：2026-06-07

## Decision

后续只保留一套主评估口径：

> real raw windows vs inverse-normalized synthetic raw-like windows

也就是说，任何 SDForger / conditioning / normalization / latent-constraint setting，最终都要把 generated window 转回同一个 raw-like sensor value space，再用同一套指标比较。

Model-space results 只保留为 debugging / diagnostic，不作为主结果表。

## Why This Is Needed

之前结果混用了几种空间：

| Space | 用途 | 是否主结果 |
|---|---|---|
| SDForger standardized window space | FICA/LLM 内部生成与调试 | 否 |
| normalization-specific model space | normalization ablation 诊断 | 否 |
| raw-like inverse-normalized space | sensor waveform quality / HAR utility | 是 |

因此，旧报告中的 model-space TSGBench、ACF/PSD、overlay 不能直接和 raw-like HAR utility 混在一张主表里。

## Primary Evaluation Suite

每个 setting 统一输出一张主表：

| Category | Metric | Direction | Purpose |
|---|---|---|---|
| Value validity | `amplitude_ratio` | lower, close to 1 | 是否数值爆炸 |
| Temporal rhythm | `acf_lag_diff` | lower | 周期位置是否接近 |
| Frequency rhythm | `psd_hz_diff` | lower | 主频是否接近 |
| TSGBench-style | `MDD` | lower | marginal distribution |
| TSGBench-style | `ACD` | lower | autocorrelation structure |
| TSGBench-style | `SD` | lower | skewness difference |
| TSGBench-style | `KD` | lower | kurtosis difference |
| TSGBench-style | `ED` | lower | paired Euclidean shape distance |
| TSGBench-style | `DTW` | lower | paired time-warped shape distance |
| Task utility | `real+synthetic accuracy` | higher | synthetic 是否帮助 HAR classifier |
| Label control | `requested-label accuracy` | higher | 仅用于 label-conditioning / unified experiments |

如果汇报页数有限，最小展示指标为：

| Metric | Why keep |
|---|---|
| `amplitude_ratio` | 先排除不真实的 value explosion |
| `acf_lag_diff` | 看动作周期 |
| `psd_hz_diff` | 看主频 |
| `MDD` | 标准分布差异 |
| `DTW` | 直观形状差异 |
| `real+synthetic accuracy` | 下游 HAR utility |
| `requested-label accuracy` | 只在 conditional setting 中展示 |

## Evaluation Split

当前最小 smoke 仍然是 subject101 / walking-running / `hand_acc16_x`：

| Evaluation | Real reference | Test target | Status |
|---|---|---|---|
| Generation quality | train-side real windows | synthetic windows | 主质量诊断 |
| HAR utility | held-out same-subject real windows after `train_length` | classifier trained on real/synthetic | provisional utility |
| Subject generalization | unseen subject real windows | not yet done | future |

当前所有主结果必须明确写：

> within-subject smoke, not unseen-subject generalization

## Required Output Files Per Setting

每个 setting 后续应至少产生：

- `raw_like_real_windows.npy`
- `raw_like_synthetic_windows.npy`
- `value_rhythm_metrics.csv`
- `tsgbench_style_metrics.csv`
- `har_utility_metrics.csv`
- `label_controllability.csv` if conditional
- representative overlay figure
- `run_metadata.json` with value-space / inverse-normalization contract

## Status of Existing Results

| Experiment family | Current status under this protocol | Action |
|---|---|---|
| clean univariate baseline | mostly usable, but document value-space clearly | keep as baseline, mark provisional |
| label conditioning v1 | usable as activity-specific conditional smoke | keep as provisional |
| unified label conditioning | useful failure diagnostic | keep, not method success |
| latent constraints | useful diagnostic / candidate mechanism | keep, but post-hoc only |
| normalization ablation | model-space diagnostic only | do not use as main result until raw-like inverse evaluation is added |

## Next Action

Before running multi-subject or more method variants:

1. Implement one evaluation wrapper that takes generated output + run metadata and produces the unified raw-like metrics.
2. Re-evaluate at least these settings:
   - clean unconditioned baseline
   - unified raw failed baseline
   - `clip_p05_p95`
   - `global_series_zscore`
3. Only after this table is coherent, decide whether to continue with latent validity / rejection sampling or multi-subject expansion.
