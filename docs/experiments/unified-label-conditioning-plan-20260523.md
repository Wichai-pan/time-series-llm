# Unified Label Conditioning Plan - PAMAP2 subject101

日期：2026-05-23

## Claim

当前要验证的窄 claim：

> 一个 unified SDForger-style conditional model 能否在同一训练集中学习 walking/running 的 activity label，并在生成时按 requested label 产生可区分的 HAR sensor windows？

这不是最终 paper claim，只是周一前可完成的 mechanism check。

## Hypothesis

Primary hypothesis:

- 把 walking/running 窗口级 FICA embeddings 合并，并加入 `data=walking/running` textual condition 后，同一个 LLM fine-tuning run 可以按 label 生成两类 synthetic windows。

Alternative explanations:

- label 没有被模型真正使用，生成结果只是混合分布。
- classifier utility 来自 activity-specific inverse transform 或训练样本数量，而不是 label controllability。
- textual template 增加 malformed output，导致有效样本减少或分布偏移。

Falsification condition:

- generated samples 大量 malformed，无法达到每类 50 个有效样本；或 controllability classifier 无法区分 requested walking/running。

## Baselines and Controls

| Candidate | Role | Requirement | Use in this experiment |
|---|---|---|---|
| clean unconditioned activity-specific baseline | nearest previous method | must-have | compare metrics/utility against previous baseline |
| label-conditioned v1 activity-specific model | ablation baseline | must-have | compare against earlier label attempt |
| real-only classifier | control baseline | must-have | checks downstream real-data reference |
| majority classifier | trivial control | must-have | checks class imbalance |
| shuffled-label conditional model | control baseline | should-have | deferred if compute/time limited |
| unified no-label mixed generator | ablation baseline | should-have | deferred if compute/time limited |

This run prioritizes the minimum complete version: unified correct-label conditional generator.

## Experiment Matrix

| Run ID | Changed variable | Fixed controls | Status |
|---|---|---|---|
| ULC-v1 | unified walking+running conditional training | PAMAP2 subject101, `hand_acc16_x`, FICA, window 300, train length 5000 per activity | to run |

## Data Protocol

- Dataset: PAMAP2 `subject101.dat`.
- Activities: walking (`activity_id=4`) and running (`activity_id=5`).
- Channel: `hand_acc16_x`.
- Windowing: use the same SDForger preprocessing independently per activity.
- Training windows: 31 walking + 31 running.
- Label: window-level `data` column added after embedding, not raw-row-level label.

## Method

1. Load walking-only and running-only parquet files.
2. Preprocess each activity independently using SDForger periodicity-aware segmentation.
3. Concatenate scaled windows into one training tensor.
4. Apply FICA embedding jointly over all windows.
5. Add categorical column `data` with walking/running labels.
6. Fine-tune one GPT-2 SDForger model using `fim_template_textual_encoding`.
7. Generate samples with fixed requested condition:
   - `Condition: data is walking`
   - `Condition: data is running`
8. Decode embeddings back to standardized SDForger window space.

## Metrics

| Metric | Direction | Purpose |
|---|---|---|
| generated count per requested label | >= 50 | checks run viability |
| malformed / dropped count | lower is better | checks text-template stability |
| ACF peak lag distance | lower is better | checks temporal periodicity |
| PSD peak distance | lower is better | checks dominant rhythm |
| good/borderline/bad split | more good, fewer bad | checks individual sample quality |
| HAR utility smoke | higher is better | checks downstream task information |
| requested-label classifier agreement | higher is better | checks controllability |

## Compute Budget

Previous label-conditioned runs finished in about 3 minutes each on 1 V100, with most wall-clock spent in vLLM startup. A unified run should fit under one 30-minute V100 job.

Resource request:

- 1 x V100
- 8 CPU
- 48G memory
- 00:30:00 walltime

## Decision Rule

Proceed with this direction if:

- both labels produce at least 50 valid samples,
- running ACF/PSD remains close to real,
- HAR synthetic-only or real+synthetic remains above real-only,
- and controllability is clearly above majority.

If only feasibility is positive but quality is worse, report it as a partial method-extension result and next prioritize unified no-label / shuffled-label ablations.
