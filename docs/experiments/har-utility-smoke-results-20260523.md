# Minimal HAR Utility Smoke Results

日期：2026-05-23

## Summary

- 目的：检查 clean SDForger synthetic windows 是否保留 walking vs running 的下游分类信息。
- 数据：PAMAP2 subject101，`hand_acc16_x`，walking/running。
- 分类器：`StandardScaler + LogisticRegression(class_weight="balanced")`。
- 测试集：held-out real windows，walking 57 个，running 54 个。
- 关键结果：`synthetic-only-all` 和 `real+synthetic-all` 都高于 `real-only`。
- 当前判断：synthetic data 有明确 HAR utility signal，但这是 one-subject / one-channel smoke，仍为 provisional。

## 1. Experiment Motivation

前面的 ACF/PSD 结果说明 synthetic samples 保留了一些周期结构，尤其 running 很明显。但周期像不代表对 HAR 分类有用。本实验测试更直接的问题：

> 当前 SDForger synthetic data 是否包含 walking vs running 的 activity-discriminative information？

## 2. Experiment Setup

| Item | Value |
|---|---|
| Dataset | PAMAP2 subject101 |
| Activities | walking vs running |
| Channel | `hand_acc16_x` |
| Real train windows | 31 walking + 31 running |
| Real test windows | 57 walking + 54 running |
| Synthetic all | 130 walking + 98 running |
| Synthetic good | 57 walking + 42 running |
| Window length | 300 |
| Classifier | StandardScaler + LogisticRegression |
| Script | `scripts/evaluate_har_utility_smoke.py` |
| Local artifacts | `outputs/har-utility-smoke-20260523/` |

Important preprocessing note:

SDForger `final_data.jsonl` is in standardized window space. For this utility smoke, synthetic windows were inverse-transformed back to raw-like window space using the corresponding activity's real train-window mean/std. The classifier then applies its own `StandardScaler` fit on each training condition.

## 3. Conditions

| Condition | Train data | Test data | Purpose |
|---|---|---|---|
| majority | dummy majority predictor | held-out real | trivial control |
| real-only | real train windows | held-out real | real-data reference |
| synthetic-only-all | all synthetic windows | held-out real | synthetic task information |
| real+synthetic-all | real train + all synthetic | held-out real | augmentation utility |
| synthetic-only-good | good synthetic only | held-out real | filtered synthetic task information |
| real+synthetic-good | real train + good synthetic | held-out real | filtered augmentation utility |

## 4. Results

| Condition | Train samples | Accuracy | Balanced accuracy | Macro F1 | Confusion matrix |
|---|---:|---:|---:|---:|---|
| majority | 62 | 0.5135 | 0.5000 | 0.3393 | `[[57, 0], [54, 0]]` |
| real-only | 62 | 0.6126 | 0.6087 | 0.6022 | `[[43, 14], [29, 25]]` |
| synthetic-only-all | 228 | 0.7027 | 0.6979 | 0.6905 | `[[50, 7], [26, 28]]` |
| real+synthetic-all | 290 | 0.7117 | 0.7071 | 0.7010 | `[[50, 7], [25, 29]]` |
| synthetic-only-good | 99 | 0.7207 | 0.7139 | 0.6987 | `[[55, 2], [29, 25]]` |
| real+synthetic-good | 161 | 0.6396 | 0.6374 | 0.6361 | `[[41, 16], [24, 30]]` |

## 5. Interpretation

The smoke result is positive but narrow.

`synthetic-only-all` beats `real-only` on held-out real test data. This suggests the generated windows are not just visually periodic; they contain useful walking/running discrimination signal.

`real+synthetic-all` is the best augmentation condition among the mixed-data settings, improving over `real-only` from 0.6126 to 0.7117 accuracy. This is the first direct utility signal for the current SDForger baseline.

`synthetic-only-good` is the highest overall condition at 0.7207 accuracy, suggesting the ACF/PSD/amplitude diagnostic labels are meaningful. However, `real+synthetic-good` is worse than `real+synthetic-all`, so good-only filtering is not yet a reliable augmentation rule. It may remove useful diversity or change class balance/coverage.

## 6. Conclusion

Current clean SDForger baseline can be treated as more than a reconstruction toy baseline: it provides a positive downstream HAR utility smoke on walking vs running for subject101 `hand_acc16_x`.

Safe claim for progress reporting:

> A minimal downstream classifier trained on SDForger-generated walking/running windows transfers above the majority and real-only smoke baselines on held-out real windows, suggesting that the generated samples preserve task-relevant motion information. The evidence is preliminary and limited to subject101, one channel, and a simple classifier.

Do not yet claim:

- General HAR augmentation improvement.
- Robust multi-subject utility.
- Superiority over other generative models.
- That good-sample filtering is validated as a training policy.

## 7. Limitations

- One subject only.
- One channel only.
- Two activity classes only.
- No multiple seeds or classifier family comparison.
- Held-out windows are from the same subject, not cross-subject.
- Synthetic inverse transform uses activity-specific train-window statistics.
- The real-only baseline is very small, so augmentation gains may partly reflect more training samples.

## 8. Next Steps

1. For Monday: include this as a provisional utility smoke, not as final evidence.
2. Next experiment: repeat with `ankle_acc16_x` or multichannel to see if utility is stable.
3. Then decide whether label conditioning is needed to improve class control and sample stability.

## Reproducibility Notes

Remote command was run under:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt`

Output:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/baseline_verification_20260523/har_utility_smoke/`

Local copy:

`outputs/har-utility-smoke-20260523/`
