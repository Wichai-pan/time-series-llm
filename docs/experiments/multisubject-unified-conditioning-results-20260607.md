# Multi-subject Unified Conditioning Results

日期：2026-06-07

## 实验目的

这次实验检验一个很具体的问题：

> `raw-unified` 失败是不是主要因为 subject101 数据太少？

如果是，那么把训练数据从 subject101 扩展到多个 subject 后，unified activity-conditioned SDForger 应该更稳定，尤其 ED / DTW / KD 不应继续爆炸。

## 执行配置

| Field | Value |
|---|---|
| Dataset | PAMAP2 |
| Subjects | 101, 102, 105 |
| Activities | walking, running |
| Channel | `hand_acc16_x` |
| Rows per subject/activity | 5000 |
| Window length | 300 |
| Model | GPT-2 |
| Embedding | FICA |
| Training windows | walking 54, running 62, combined 116 |
| FICA dim | 8 |
| Output space | standardized SDForger window space |

执行前数据检查发现 subject103 没有 running activity，因此实际从原计划的 101/102/103 改为平衡的 101/102/105。

## 生成结果

| Activity | Raw outputs | Accepted synthetic windows | Attempts |
|---|---:|---:|---:|
| walking | 128 | 51 | 2 |
| running | 192 | 58 | 3 |

有效样本数低于真实窗口数，因此本次 TSG evaluator 做了一个兼容修复：MDD/ACD/SD/KD 使用全部 real 与全部 synthetic 分布；ED/DTW 使用 `min(real_count, synthetic_count)` 个 paired subset。该策略已写入输出 JSON 的 `paired_sample_policy`。

## TSG-style 结果

| Setting | Activity | n | MDD ↓ | ACD ↓ | SD ↓ | KD ↓ | ED ↓ | DTW ↓ | Real windows | Paired windows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_raw_unified | walking | 51 | 0.174 | 5.344 | 0.016 | 40.233 | 700.291 | 9920.940 | 54 | 51 |
| multi_raw_unified | running | 58 | 0.194 | 1.623 | 1.349 | 92.912 | 569.687 | 7943.930 | 62 | 58 |
| multi_clip_p05_p95 | walking | 51 | 0.172 | 3.117 | 0.229 | 0.746 | 21.672 | 166.065 | 54 | 51 |
| multi_clip_p05_p95 | running | 58 | 0.204 | 1.611 | 0.652 | 0.281 | 20.737 | 147.235 | 62 | 58 |

Source:

- `outputs/multisubject-tsgbench-20260607/multisubject_tsgbench_20260607_fixed/multisubject_tsgbench_summary.csv`
- Remote report: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/multisubject_tsgbench_20260607_fixed/`
- Run metadata: `outputs/multisubject-tsgbench-20260607/run_metadata.json`

## Constraint summary

| Variant | Activity | Input rows | Output rows | Latent abs max | Decoded abs max | Decoded std mean |
|---|---|---:|---:|---:|---:|---:|
| clip_p05_p95 | walking | 51 | 51 | 2.369 | 2.847 | 0.705 |
| clip_p05_p95 | running | 58 | 58 | 2.369 | 2.494 | 0.630 |
| reject_iqr3 | walking | 51 | 14 | 2.216 | 1.978 | 0.449 |
| reject_iqr3 | running | 58 | 21 | 2.287 | 2.525 | 0.546 |

`clip_p05_p95` 仍然保留所有 accepted rows，并把 decoded abs max 控制在约 2.5-2.8 的 standardized range。

## 结果判断

1. 增加 subject 没有单独修好 raw-unified。`multi_raw_unified` 的 ED/DTW/KD 仍然很差，说明主要问题不是 subject101 数据太少。
2. `clip_p05_p95` 仍然是关键稳定器。它把 walking DTW 从 9920.940 降到 166.065，把 running DTW 从 7943.930 降到 147.235，KD 也从几十/九十降到小于 1。
3. 多 subject 后的 `clip_p05_p95` 比 subject101 `clip_p05_p95` 在 ED/DTW 上略有改善，但 ACD 仍偏高，尤其 walking ACD 3.117。不能说多 subject 已解决 rhythm/autocorrelation 问题。
4. 生成解析仍不稳定。walking 只有 51 个 accepted windows，running 58 个，说明 textual latent generation 的 malformed output 问题仍在。

## 当前结论

这次实验支持一个更保守的判断：

> 多 subject 增加数据量有帮助，但不是主要解法；unified conditional SDForger 的核心问题仍是 latent validity / decoding stability。下一步应优先做 generation-time validity control，而不是继续单纯加 subject。

因此 multi-subject 方向不 kill，但应作为后续 evaluation/generalization protocol，而不是当前 method fix 的主线。

## 下一步

1. 把 `clip_p05_p95` 从 post-hoc diagnostic 改成 generation-time latent validity / rejection sampling。
2. 记录每个 activity 的 accepted/rejected count 和 reject reason。
3. 如果 generation-time validity 稳定，再做 unseen-subject split：train subjects 101/102/105，test held-out subject。
4. 修正 prompt/parser，减少 malformed numeric latent outputs。
