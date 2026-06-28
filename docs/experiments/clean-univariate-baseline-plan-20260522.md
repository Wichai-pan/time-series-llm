# Clean Univariate Baseline Plan

日期：2026-05-22

## Question

在 PAMAP2 subject101 中，SDForger 的单变量 baseline 是否能在周期性明确的单一 activity 上成立？

## Baseline Choice

- baseline：旧 SDForger univariate setting
- dataset：PAMAP2 subject101
- activities：
  - walking，`activity_id=4`
  - running，`activity_id=5`
- channel：`hand_acc16_x`
- embedding：FICA
- train length：5000
- window length：300
- target generated samples：50-100

## Controls

固定：

- subject：101
- channel：`hand_acc16_x`
- model：default GPT-2 baseline
- prompt style：original SDForger FIM embedding prompt
- embedding type：FICA
- train_length：5000
- min_windows_number：30
- min_windows_length：300

变化：

- activity subset：walking vs running

## Compute Budget

旧 mixed-activity run `pamap2_subject101_univariate_paper` 在 V100 上约 68 秒完成，生成 85 samples。两个 activity-specific baseline 预计合计 < 10 分钟 GPU wall time；使用 1x V100 即可。

## Success Criteria

- 每个 task 生成 `final_data.jsonl`
- `task_results.jsonl` 记录 completed
- generated samples 在 50-100 左右
- 可后续用同一 metric script 重算 MDD/ACD/SD/KD/ED/DTW

## Caveats

- 本次 baseline 仍是 original SDForger prompt，不含 activity label。
- 本次生成输出仍按旧 pipeline 保存为 standardized SDForger window space。
- 这是 clean activity-specific baseline，用于后续对比 `+ activity label`。
