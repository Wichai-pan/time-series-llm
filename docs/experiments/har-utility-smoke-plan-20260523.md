# Minimal HAR Utility Smoke Plan

日期：2026-05-23

## Claim / Question

问题：当前 clean SDForger univariate baseline 生成的 walking/running `hand_acc16_x` synthetic windows，是否保留了足够的 activity-discriminative information，能帮助或至少支持 walking vs running 分类？

这是 internal direction check，不是 paper-level final evidence。

## Hypothesis

Primary hypothesis:

- 如果 SDForger synthetic windows 保留了 task-relevant motion rhythm，那么 `synthetic-only -> held-out real test` 应该高于 majority baseline。
- 如果 synthetic 有 augmentation utility，那么 `real + synthetic -> held-out real test` 应该不低于 `real-only -> held-out real test`，最好略高。

Falsification / weak signal:

- `synthetic-only` 接近 majority baseline。
- `real + synthetic` 明显低于 `real-only`。
- good-only synthetic 不比 all synthetic 更稳。

## Baselines and Conditions

| Condition | Train data | Test data | Purpose |
|---|---|---|---|
| majority | none | held-out real | trivial control |
| real-only | real train windows | held-out real | upper reference for this smoke |
| synthetic-only-all | all synthetic windows | held-out real | checks synthetic task information |
| real+synthetic-all | real train + all synthetic | held-out real | checks augmentation utility |
| synthetic-only-good | good synthetic windows only | held-out real | checks whether diagnostic filtering matters |
| real+synthetic-good | real train + good synthetic | held-out real | checks filtered augmentation utility |

## Data Protocol

Activities:

- walking = label 0
- running = label 1

Channel:

- `hand_acc16_x`

Real train:

- Use SDForger `preprocess_train_data` on the first `train_length=5000` rows per activity.
- This gives 31 real train windows per activity.

Real test:

- Use held-out rows after the first 5000 rows per activity.
- Cut into non-overlapping length-300 windows.
- Test set is real-only and never synthetic.

Synthetic train:

- Read `final_data.jsonl` from the clean walking/running baseline.
- Synthetic values are in standardized SDForger window space.
- Convert synthetic windows back to raw-like window space using the corresponding activity's real train-window mean/std.

Good synthetic:

- Use sample indices labeled `good` from `outputs/sample-stratification-20260522/*_sample_stratification.csv`.

## Classifier

Use a simple sklearn pipeline:

- `StandardScaler`
- `LogisticRegression(max_iter=2000, class_weight="balanced")`

This is intentionally simple. The point is not to tune a strong HAR model; the point is to test whether synthetic windows contain enough class signal.

## Metrics

| Metric | Direction | Why |
|---|---|---|
| accuracy | higher better | simple headline metric |
| balanced_accuracy | higher better | robust if class counts differ |
| macro_f1 | higher better | checks both classes |
| confusion matrix | diagnostic | shows which activity fails |

## Compute Budget

CPU-only, no GPU. Expected runtime: under one minute.

## Stop Condition

Run once. If results are sensible, write report. If classifier crashes or data conversion is inconsistent, stop and debug preprocessing instead of adding new model changes.

## Expected Interpretation

- If `synthetic-only-all` is above majority, current synthetic preserves some activity signal.
- If `real+synthetic-all` improves over `real-only`, current synthetic has augmentation utility.
- If `real+synthetic-good` is better than `real+synthetic-all`, sample quality filtering is useful.
- If synthetic hurts performance, label conditioning or better embeddings become more justified next steps.
