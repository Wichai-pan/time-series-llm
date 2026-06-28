# Label Conditioning v1 Results - PAMAP2 subject101

日期：2026-05-23

## Summary

这次实验是在 clean activity-specific univariate baseline 之后，做一个最小 `activity label` conditioning 版本。

核心问题不是“最终方法是否已经成功”，而是：

> 在 SDForger 的文本编码阶段加入 activity label 后，pipeline 是否能跑通，生成样本是否仍保留 walking/running 的周期结构和下游分类信息？

当前结论：

- label-conditioned pipeline 已在 Puhti 上成功跑通。
- walking 生成 72 个 synthetic windows，running 生成 90 个 synthetic windows。
- HAR utility smoke 仍然是正向的：`synthetic-only-all` accuracy 0.7387，高于 unconditioned baseline 的 0.7027；`real+synthetic-all` accuracy 0.7117，与 unconditioned baseline 持平。
- 但生成日志中出现 malformed textual outputs，被过滤后有效样本数少于 unconditioned baseline；因此 label conditioning v1 只能算 `provisional method-extension evidence`。

## 1. Motivation

前一个 clean baseline 已经是 activity-specific filtering：

- walking-only model 只看 walking 数据。
- running-only model 只看 running 数据。

这说明 activity 信息已经通过“数据子集”隐式进入模型，但还没有作为 prompt/covariate 进入 SDForger 的文本生成过程。

这次 v1 实验只改一个点：

- 保留 FICA、单通道、window length、train length、classifier protocol。
- 在 SDForger text template 中加入 categorical condition，例如 `Condition: data is walking` 或 `Condition: data is running`。

因此它适合回答“label/covariate conditioning 是否值得继续做”，不适合直接声称这是完整的新方法。

## 2. Implementation

远程代码路径：

`/scratch/project_2016517/panh/time-series-llm/fms-dgt`

修改点：

- `fms_dgt/public/databuilders/time_series/trainer.py`
- `fms_dgt/public/databuilders/time_series/generate.py`

备份：

- `generate.py.pre_label_conditioning_20260523`
- `trainer.py.pre_label_conditioning_20260523`

新增 task：

- `tasks/public/time_series/pamap2_subject101_walking_hand_acc16_x_label_conditioned_univariate/task.yaml`
- `tasks/public/time_series/pamap2_subject101_running_hand_acc16_x_label_conditioned_univariate/task.yaml`

关键配置：

```yaml
text_template: fim_template_textual_encoding
conditioning_label_column: data
conditioning_label_value: walking  # or running
```

注意：这是对远程旧工作区的最小 patch，不是 clean Git-controlled implementation。后续如果继续这条线，应把 patch 迁移到干净 repo，并补 parser / template 测试。

## 3. Run Status

| Activity | Job ID | Output samples | Status |
|---|---:|---:|---|
| walking | 34526946 | 72 | completed |
| running | 34526947 | 90 | completed |

远程输出：

- walking: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_walking_hand_acc16_x_label_conditioned_univariate/`
- running: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_running_hand_acc16_x_label_conditioned_univariate/`

本地结果：

`outputs/label-conditioning-20260523/`

## 4. TSGBench-style Metrics

| Activity | Version | Samples | MDD | ACD | SD | KD | ED | DTW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| walking | unconditioned | 130 | 0.266287 | 0.165201 | 0.797960 | 1.667799 | 21.279075 | 237.760946 |
| walking | label-conditioned v1 | 72 | 0.275518 | 0.549948 | 1.841294 | 1.896688 | 23.914547 | 291.820399 |
| running | unconditioned | 98 | 0.296503 | 0.497915 | 0.612561 | 0.444503 | 17.078956 | 111.087924 |
| running | label-conditioned v1 | 90 | 0.296924 | 0.584861 | 0.655061 | 0.220628 | 18.585106 | 141.005387 |

Interpretation:

- Walking 的 distance / distribution metrics 明显变差，尤其 ACD、SD、DTW。
- Running 的 KD 变好，但 ACD、ED、DTW 变差。
- 单看 TSGBench-style metrics，label conditioning v1 不足以说明生成质量优于 unconditioned baseline。

## 5. ACF / PSD

| Activity | Version | Real ACF lag | Synthetic ACF lag | ACF lag diff | Real PSD Hz | Synthetic PSD Hz | PSD diff |
|---|---|---:|---:|---:|---:|---:|---:|
| walking | unconditioned | 116 | 58 | 58 | 1.6667 | 1.6667 | 0 |
| walking | label-conditioned v1 | 116 | 58 | 58 | 1.6667 | 1.6667 | 0 |
| running | unconditioned | 81 | 82 | 1 | 1.3333 | 1.3333 | 0 |
| running | label-conditioned v1 | 81 | 82 | 1 | 1.3333 | 1.3333 | 0 |

Interpretation:

- Label conditioning v1 没有破坏主要周期结构。
- Running 仍然是最清楚的 baseline case：ACF 和 PSD 都匹配。
- Walking 仍然是 partial case：主频匹配，但 ACF 主峰还是 58 vs 116 的 harmonic / two-cycle mismatch。

## 6. Sample Stratification

| Activity | Version | Total | Good | Borderline | Bad |
|---|---|---:|---:|---:|---:|
| walking | unconditioned | 130 | 57 | 64 | 9 |
| walking | label-conditioned v1 | 72 | 33 | 33 | 6 |
| running | unconditioned | 98 | 42 | 14 | 42 |
| running | label-conditioned v1 | 90 | 41 | 19 | 30 |

Interpretation:

- Walking 的 good ratio 接近，但总样本数下降。
- Running 的 bad sample 数从 42 降到 30，good 数基本持平，说明 label-conditioned v1 对 running 的样本分层可能有一点改善。
- 这些 labels 是 heuristic diagnostics，不是正式 benchmark。

## 7. HAR Utility Smoke

同一套 held-out real test：

- walking real test windows: 57
- running real test windows: 54
- classifier: `StandardScaler + LogisticRegression(class_weight="balanced")`

| Condition | Unconditioned Acc | Label-conditioned Acc | Label-conditioned Bal. Acc | Label-conditioned Macro F1 |
|---|---:|---:|---:|---:|
| majority | 0.5135 | 0.5135 | 0.5000 | 0.3393 |
| real-only | 0.6126 | 0.6126 | 0.6087 | 0.6022 |
| synthetic-only-all | 0.7027 | 0.7387 | 0.7329 | 0.7236 |
| real+synthetic-all | 0.7117 | 0.7117 | 0.7086 | 0.7063 |
| synthetic-only-good | 0.7207 | 0.7207 | 0.7139 | 0.6987 |
| real+synthetic-good | 0.6396 | 0.6396 | 0.6374 | 0.6361 |

Interpretation:

- `synthetic-only-all` 比 unconditioned baseline 更高，说明 label-conditioned synthetic windows 仍然包含 walking/running 区分信息，且可能更利于这个简单 classifier。
- `real+synthetic-all` 没有进一步超过 unconditioned augmentation baseline，因此不能说 label conditioning 已经改善 augmentation。
- `good-only` 结果与上一版相同，不支持“只保留 good samples 就一定更好”的规则。

## 8. Main Takeaway

这次实验支持一个谨慎结论：

> Activity label conditioning v1 is feasible in the recovered SDForger/PAMAP2 pipeline and preserves the downstream walking/running utility signal, but it does not yet consistently improve synthetic quality metrics over the clean unconditioned baseline.

中文汇报可说：

> 我们已经把 activity label 作为文本条件接入 SDForger pipeline，并完成 walking/running 的最小验证。结果显示 label-conditioned 版本能跑通，也能保留分类可用信息；但在 TSGBench-style 质量指标上并没有稳定超过原 baseline，所以当前更适合作为下一步方法探索，而不是最终结论。

## 9. Risks

- 远程旧工作区没有 Git provenance，这次 patch 需要后续迁移到 clean repo。
- Textual condition 增加了 LLM 输出解析风险，日志中已经出现 malformed outputs。
- 当前 label conditioning 是 activity-specific model + activity label，并不是一个 unified multi-class conditional generator。
- 所有 utility 结果仍是 one-subject、one-channel、two-class smoke。

## 10. Next Steps

1. 周一汇报中把 label conditioning 放在“已完成初步尝试 / 不作为主 claim”的位置。
2. 下一步更优先做 multichannel 或 ankle channel 验证，确认 HAR utility signal 是否稳定。
3. 如果继续 label conditioning，应改成 unified walking+running conditional generation，而不是两个 activity-specific task 各自加 label。
4. 把远程 patch 迁移到 Git-controlled clean repo，并为 text template / parser / numerical-column extraction 加测试。
