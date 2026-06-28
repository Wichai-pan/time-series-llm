# Unseen-subject Evaluation Results

日期：2026-06-07

## 实验目的

这次实验检验：

> 用 subject101/102/105 训练并生成的 synthetic windows，是否也接近未见过 subject106/108 的真实 walking/running 窗口？

这是 evaluation-only diagnostic：不重训模型，不重新生成 synthetic，只把 real reference 从 train subjects 换成 held-out subjects。

## 数据与设置

| Field | Value |
|---|---|
| Generator train subjects | 101/102/105 |
| Held-out reference subjects | 106/108 |
| Activities | walking, running |
| Channel | `hand_acc16_x` |
| Held-out rows | 每个 subject/activity 5000 rows |
| Held-out total rows | 每个 activity 10000 rows |
| Synthetic source | `pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x` |
| Compared variants | `unseen_raw_unified`, `unseen_clip_p05_p95` |

Held-out real preprocessing:

| Activity | Estimated period | Real windows |
|---|---:|---:|
| walking | 110 | 45 |
| running | 76 | 44 |

注意：held-out walking 的 period 110 与 train subjects walking 的 period 56 差异明显，说明 subject/domain shift 确实存在。

## 结果表

| Setting | Activity | n | MDD ↓ | ACD ↓ | SD ↓ | KD ↓ | ED ↓ | DTW ↓ | Real windows | Paired windows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unseen_raw_unified | walking | 51 | 0.188 | 4.212 | 0.072 | 39.805 | 787.558 | 11160.400 | 45 | 45 |
| unseen_raw_unified | running | 58 | 0.232 | 1.296 | 1.684 | 92.044 | 689.475 | 9567.380 | 44 | 44 |
| unseen_clip_p05_p95 | walking | 51 | 0.189 | 1.992 | 0.173 | 1.175 | 21.496 | 151.102 | 45 | 45 |
| unseen_clip_p05_p95 | running | 58 | 0.236 | 2.228 | 0.987 | 1.149 | 20.685 | 146.689 | 44 | 44 |

Source:

- `outputs/unseen-subject-tsgbench-20260607/unseen_subject_tsgbench_20260607/unseen_subject_tsgbench_summary.csv`
- Remote report: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/unseen_subject_tsgbench_20260607/`
- Script: `scripts/rerun_unseen_subject_tsgbench_table.py`

## 与 train-reference multi-subject 结果对比

| Setting | Activity | Reference | ED ↓ | DTW ↓ | ACD ↓ |
|---|---|---|---:|---:|---:|
| multi_clip_p05_p95 | walking | train subjects 101/102/105 | 21.672 | 166.065 | 3.117 |
| unseen_clip_p05_p95 | walking | held-out subjects 106/108 | 21.496 | 151.102 | 1.992 |
| multi_clip_p05_p95 | running | train subjects 101/102/105 | 20.737 | 147.235 | 1.611 |
| unseen_clip_p05_p95 | running | held-out subjects 106/108 | 20.685 | 146.689 | 2.228 |

## 解释

1. `raw_unified` 在 unseen subjects 上仍然失败。ED/DTW/KD 继续很大，说明不加 latent validity control 时，生成仍不稳定。
2. `clip_p05_p95` 在 unseen subjects 上仍然保持可接受的 ED/DTW/KD。walking/running 的 ED 都约 21，DTW 约 151/147，接近 train-reference multi-subject clip。
3. 这说明 `clip_p05_p95` 不是只对 train subjects 有效；至少在 subject106/108 reference 下，它仍能生成与真实动作窗口接近的形状距离。
4. ACD 仍是主要风险。running 的 unseen ACD 2.228，比 train-reference 1.611 更差；walking 虽然变好，但 held-out walking period 与 train walking period 差异很大，需要谨慎解释。

## 当前结论

这次实验支持一个比上一轮更强但仍然保守的结论：

> Multi-subject + `clip_p05_p95` synthetic windows 在 held-out subject reference 上仍保持接近 train-reference 的 ED/DTW/KD，说明该方向具有初步跨 subject 稳定性；但 ACD / rhythm preservation 和 post-hoc clipping 仍是主要限制。

不能说：

- 模型已经完成严格泛化验证；
- synthetic 数据一定能提升 held-out HAR classifier；
- clip 是最终方法。

可以说：

- 当前 synthetic 不是只贴合训练 subject；
- latent validity control 是值得继续的核心方向；
- 下一步可以在 generation-time validity control 后补 held-out HAR utility。

## 下一步

1. 把 `clip_p05_p95` 升级成 generation-time validity control。
2. 用同一 held-out reference 重新比较：
   - raw unified
   - post-hoc clip
   - generation-time validity control
3. 增加 held-out HAR utility：
   - synthetic-only classifier tested on held-out real
   - train-real + synthetic tested on held-out real
4. 记录 rejected latent 的数量和原因，用来证明方法不是简单事后修图。
