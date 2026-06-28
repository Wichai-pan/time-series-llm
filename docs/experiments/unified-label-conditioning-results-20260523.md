# Unified Label Conditioning Results - PAMAP2 subject101

日期：2026-05-23

## Summary

- 目的：把 walking/running 放进同一个 SDForger-style conditional model，验证 activity label 是否真的能控制生成。
- 数据：PAMAP2 subject101，`hand_acc16_x`，walking/running，各 31 个训练窗口。
- 方法：分别切 walking/running 窗口，合并后 joint FICA embedding，加入窗口级 `data=walking/running` label，训练一个 GPT-2 conditional generator。
- 结果：生成成功，walking 72 个、running 79 个；label controllability 很强，accuracy 0.8212。
- 但生成质量明显失控：synthetic amplitude 极大，PSD peak 偏到 2.3333Hz，HAR utility 全样本低于 real-only。
- 当前判断：这是一个有用的 diagnostic result。它说明 label signal 被模型学到了，但 joint FICA / unconstrained text generation 还不能直接作为稳定 synthetic HAR generator。

## 1. Experiment Motivation

之前的 label conditioning v1 仍然是 activity-specific：

- walking-only model + `data=walking`
- running-only model + `data=running`

这种设置里 label 的作用很弱，因为模型本来只看一种 activity。为了让实验更有意义，这次改成：

> 一个模型同时学习 walking/running，并在 inference 时用 requested label 控制生成类别。

这更接近真正的 activity-conditioned generation。

## 2. Experiment Setup

| Item | Value |
|---|---|
| Dataset | PAMAP2 subject101 |
| Activities | walking, running |
| Channel | `hand_acc16_x` |
| Real train windows | 31 walking + 31 running |
| Window length | 300 |
| Preprocessing | each activity independently segmented by SDForger |
| Embedding | joint FICA over combined windows |
| FICA dim | 5 |
| Variance retained | 0.7278 |
| Model | GPT-2 |
| Training job | `34530050` |
| Output | `outputs/unified-label-conditioning-20260523/` |

Run script:

- `scripts/run_unified_label_conditioning.py`
- `puhti-generated/slurm/run_sdforger_pamap2_unified_label_conditioned_gpu.sh`

Remote output:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x/`

## 3. Core Method

The important implementation change is window-level label assignment:

1. Load walking-only and running-only parquet files.
2. Run SDForger periodicity-aware preprocessing separately for each activity.
3. Concatenate the scaled windows into one training tensor.
4. Fit one joint FICA embedding.
5. Add a categorical `data` column:
   - walking windows get `data=walking`
   - running windows get `data=running`
6. Fine-tune one text model using `fim_template_textual_encoding`.
7. Generate with fixed requested prompts:
   - `Condition: data is walking`
   - `Condition: data is running`

This avoids the invalid shortcut of simply concatenating raw walking/running time series before segmentation.

## 4. Generation Status

| Requested label | Raw LLM outputs | Parsed rows | Accepted rows |
|---|---:|---:|---:|
| walking | 128 | 128 | 72 |
| running | 128 | 127 | 79 |

Both labels reached the minimum target of 50 valid samples.

However, the logs contain many malformed numeric outputs, and the generated accepted values include extreme latent values. This becomes visible in the amplitude and PSD checks below.

## 5. Label Controllability

Controllability test:

- Train classifier on real walking/running train windows.
- Predict labels of generated windows.
- Compare predicted label to requested generation label.

| Metric | Value |
|---|---:|
| Overall accuracy | 0.8212 |
| Balanced accuracy | 0.8205 |
| Macro F1 | 0.8207 |
| Confusion matrix | `[[58, 14], [13, 66]]` |

Per requested label:

| Requested label | Samples | Predicted walking | Predicted running | Requested-label accuracy |
|---|---:|---:|---:|---:|
| walking | 72 | 58 | 14 | 0.8056 |
| running | 79 | 13 | 66 | 0.8354 |

Interpretation:

- The model did learn a usable label signal.
- This is the strongest positive result from the unified experiment.
- But controllability alone does not imply realistic generated signals.

## 6. TSGBench-style Metrics

| Activity | Version | Samples | MDD | ACD | SD | KD | ED | DTW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| walking | clean unconditioned | 130 | 0.266287 | 0.165201 | 0.797960 | 1.667799 | 21.279075 | 237.760946 |
| walking | label-conditioned v1 | 72 | 0.275518 | 0.549948 | 1.841294 | 1.896688 | 23.914547 | 291.820399 |
| walking | unified label-conditioned | 72 | 0.256933 | 5.972569 | 3.524391 | 118.647911 | 404.584401 | 6026.478759 |
| running | clean unconditioned | 98 | 0.296503 | 0.497915 | 0.612561 | 0.444503 | 17.078956 | 111.087924 |
| running | label-conditioned v1 | 90 | 0.296924 | 0.584861 | 0.655061 | 0.220628 | 18.585106 | 141.005387 |
| running | unified label-conditioned | 79 | 0.298711 | 2.284618 | 2.917641 | 102.877533 | 252.417079 | 3772.691067 |

Interpretation:

- Unified label-conditioned outputs are much worse on most quality/distance metrics.
- The huge KD/ED/DTW values are consistent with generated amplitude outliers.

## 7. ACF / PSD

| Activity | Real ACF lag | Synthetic ACF lag | ACF lag diff | Real PSD Hz | Synthetic PSD Hz | PSD diff | Real std mean | Synthetic std mean | Synthetic abs max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| walking | 116 | 80 | 36 | 1.6667 | 2.3333 | 0.6667 | 0.5540 | 36.2767 | 3761.8180 |
| running | 81 | 80 | 1 | 1.3333 | 2.3333 | 1.0000 | 0.8679 | 15.1972 | 1572.0931 |

Interpretation:

- Running still has a reasonable ACF lag, but PSD is wrong.
- Walking improves ACF lag distance relative to 58 vs 116, but PSD and amplitude are badly wrong.
- The dominant failure is not just periodicity; it is value-scale / latent outlier control.

## 8. Sample Stratification

| Activity | Total | Good | Borderline | Bad |
|---|---:|---:|---:|---:|
| walking | 72 | 8 | 10 | 54 |
| running | 79 | 18 | 22 | 39 |

Compared with previous runs, unified label-conditioned generation creates many more bad samples. This matches the amplitude and PSD failures.

## 9. HAR Utility Smoke

| Condition | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| majority | 0.5135 | 0.5000 | 0.3393 |
| real-only | 0.6126 | 0.6087 | 0.6022 |
| synthetic-only-all | 0.4955 | 0.5010 | 0.4768 |
| real+synthetic-all | 0.5315 | 0.5346 | 0.5269 |
| synthetic-only-good | 0.6486 | 0.6404 | 0.6073 |
| real+synthetic-good | 0.6757 | 0.6735 | 0.6725 |

Interpretation:

- All synthetic samples hurt utility because many are bad/outlier samples.
- Filtering to good samples recovers some utility and beats real-only in `real+synthetic-good`, but the good set is very small: 8 walking + 18 running.
- This supports using sample-quality filtering diagnostically, but not as a final training protocol yet.

## 10. Conclusion

This experiment answers the key question:

> Does unified activity label conditioning create controllable walking/running generations?

Answer:

> Partly yes. The requested label is reflected in the generated samples strongly enough for a real-data classifier to recover it with 0.8212 accuracy. However, the generated signals are not yet high-quality because joint FICA plus unconstrained text generation produces severe amplitude outliers and wrong PSD peaks.

Safe slide wording:

> Unified label conditioning works as a controllability mechanism, but the current joint-embedding implementation is not yet a usable generator. The next fix should target latent/value constraints, not simply more labels.

Do not claim:

- unified label conditioning improves SDForger generation quality,
- generated samples are realistic,
- unified model is better for HAR augmentation.

## 11. Next Steps

1. Add latent norm / percentile clipping before FICA inverse transform and re-evaluate.
2. Try per-class FICA basis with a shared conditional LLM, or normalize generated latent values to training latent ranges.
3. Add shuffled-label and no-label mixed controls after amplitude is stabilized.
4. If reporting Monday, present this as a useful failure analysis: controllability is promising, quality constraints are the next technical bottleneck.

## Reproducibility Notes

Remote root:

`/scratch/project_2016517/panh/time-series-llm/fms-dgt`

Job:

`34530050`

Local artifacts:

`outputs/unified-label-conditioning-20260523/`

Important scripts:

- `scripts/run_unified_label_conditioning.py`
- `scripts/evaluate_label_controllability.py`
- `scripts/evaluate_sdforger_paper_metrics.py`
- `scripts/compare_sdforger_acf_psd.py`
- `scripts/stratify_sdforger_samples.py`
- `scripts/evaluate_har_utility_smoke.py`
