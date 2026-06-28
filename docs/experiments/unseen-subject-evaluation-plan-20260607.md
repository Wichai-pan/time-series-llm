# Unseen-subject Evaluation Plan

日期：2026-06-07

## 实验问题

这一步回答导师关心的 unseen 问题：

> 用 subject101/102/105 训练出的 unified conditional generator，生成的 synthetic walking/running windows 是否仍接近未见过 subject 的真实动作窗口？

## 实验类型

Evaluation-only diagnostic。  

不重训模型，不重新生成 synthetic。只改变 evaluation reference：

- Generator train subjects：101/102/105
- Held-out real reference subjects：106/108

这样只改变一个变量：真实 reference 从 train subjects 换成 unseen subjects。

## 数据选择

远程计数检查：

| Subject | walking rows | running rows | 使用 |
|---|---:|---:|---|
| 103 | 29036 | 0 | 不使用，缺 running |
| 104 | 31932 | 1 | 不使用，running 不足 |
| 106 | 25721 | 22825 | 使用 |
| 108 | 31533 | 16532 | 使用 |

因此 held-out subjects 选 `106/108`，每个 subject/activity 取前 5000 rows，合并后每个 activity 10000 rows。

## Run matrix

| Run | Synthetic source | Real reference | Activity | Purpose |
|---|---|---|---|---|
| U1 | `multi_raw_unified` from 101/102/105 | held-out 106/108 | walking | 检查 raw-unified 对 unseen 是否仍爆炸 |
| U2 | `multi_raw_unified` from 101/102/105 | held-out 106/108 | running | 同上 |
| U3 | `multi_clip_p05_p95` from 101/102/105 | held-out 106/108 | walking | 检查 clip 后是否接近 unseen real |
| U4 | `multi_clip_p05_p95` from 101/102/105 | held-out 106/108 | running | 同上 |

## 指标

主指标仍使用 TSG-style：

- MDD
- ACD
- SD
- KD
- ED
- DTW

补充记录：

- synthetic count
- real window count
- paired window count
- train subjects
- held-out reference subjects

## Fairness ledger

| Comparison | 是否公平 | 说明 |
|---|---|---|
| unseen raw vs unseen clip | fair | 同一个 generator、同一 held-out reference，只改变 clip |
| train-subject clip vs unseen-subject clip | fair-with-caveat | synthetic 相同，reference 不同；可看 domain shift，但不能当模型重新训练结果 |
| clean single-activity baseline vs unseen clip | reference-only | baseline 任务更简单，不能直接说谁胜谁负 |

## Stop / decision rule

继续该方向，如果：

- `unseen_clip_p05_p95` 的 ED/DTW/KD 接近 train-subject clip；
- raw-unified 仍差但 clip 稳定；
- ACD 即使偏高，也没有比 train-subject clip 大幅恶化。

如果 unseen clip 明显崩，则说明 current synthetic 更像 train subjects，需要先做 subject conditioning 或更严格 split。

## 输出

- Script：`scripts/rerun_unseen_subject_tsgbench_table.py`
- Remote report：`/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/unseen_subject_tsgbench_20260607/`
- Local output：`outputs/unseen-subject-tsgbench-20260607/`
