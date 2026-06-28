# Smooth Latent Repair Results (T1)

日期：2026-06-15

## 背景

会后导师反馈之一：现有 `clip` 修复（把超界 latent 硬夹到 train min/max 或 p05/p95）应改为"更平滑、小幅度 peak 的线性插值"方法。本实验把这个反馈实现成两类新的修复 variant，并与现有 clip 在**同一 FICA basis、同一 canonical TSG 协议**下对比。

- 脚本（修复）：`scripts/apply_smooth_latent_repair.py`
- 脚本（评估）：`scripts/rerun_smooth_repair_tsgbench_table.py`
- 远程输出：`output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x_smooth_repair_20260615/`
- 远程评估：`reports/smooth_repair_tsgbench_20260615/`
- 本地副本：`outputs/smooth-repair-20260615/`
- setting：subject101 unified walking/running、`hand_acc16_x`、train_length 5000、FICA dim 5、generated 72/79。

## Variant 定义

- `clip_p05_p95`（参照，已有方法）：每个 latent 维度独立硬夹到 train p05/p95。
- `shrink_p05_p95` / `shrink_minmax`（**A：latent 全局缩放**）：对整行 latent 乘一个标量，使最越界的维度刚好落到 train 边界（p05/p95 或 min/max）。保持 latent 向量方向（时间形状），只降幅。
- `waveform_interp`（**C：波形域线性插值**）：先 decode raw latent，再把超出"真实窗口幅度带"（real per-activity min/max）的点用 `np.interp` 线性插值填回；单窗超界点 > 50% 则整窗 reject。

## 复现性验证

本次重算的 `clip_p05_p95` 与既有 `combined-experiment-table-20260607` 完全一致：

- walking：ACD 2.253、DTW 198.446
- running：ACD 1.960、DTW 139.695

→ 重拟合 FICA basis（`random_state=0`）精确复现原始 run，新 variant 与 canonical 表直接可比。

## 主结果（standardized SDForger space，越低越好）

参考（来自 canonical 表）：baseline walking ACD 0.165 / DTW 237.8；baseline running ACD 0.498 / DTW 111.1；raw-unified walking ACD 5.973 / DTW 6026；running ACD 2.285 / DTW 3773。

| setting | activity | n | MDD | ACD | SD | KD | ED | DTW |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| clip_p05_p95 | walking | 72 | 0.273 | 2.253 | 0.516 | 1.144 | 20.166 | 198.446 |
| **shrink_p05_p95** | walking | 72 | 0.267 | **2.021** | 0.308 | 1.059 | **17.886** | **179.456** |
| **shrink_minmax** | walking | 72 | 0.263 | **1.395** | **0.071** | **0.107** | 20.755 | 214.654 |
| waveform_interp | walking | 56 | 0.272 | 2.079 | 0.894 | 0.117 | 23.040 | 237.279 |
| clip_p05_p95 | running | 79 | 0.316 | 1.960 | 0.336 | 0.422 | 22.645 | 139.695 |
| shrink_p05_p95 | running | 79 | 0.318 | 2.159 | 0.491 | 0.619 | 20.463 | 148.800 |
| shrink_minmax | running | 79 | 0.313 | 2.846 | 0.631 | 0.265 | 23.265 | 168.414 |
| waveform_interp | running | 67 | 0.310 | 2.737 | 0.505 | 0.879 | 25.776 | 204.736 |

修复计数（clean / repaired / rejected）：
- shrink_p05_p95：walking 20/52/0、running 18/61/0（**不丢样本**）。
- shrink_minmax：walking 29/43/0、running 29/50/0（**不丢样本**）。
- waveform_interp：walking 50/6/16、running 54/13/12（拒掉整窗崩坏样本）。

## 结论（诚实读数）

1. **walking（一直修不好的难例）：latent 缩放有效。** `shrink_minmax` 把 walking ACD 从 clip 的 2.253 降到 **1.395**（向 baseline 0.165 靠近），SD/KD 也大幅改善；`shrink_p05_p95` 在保留全部 72 个样本的同时，ACD/DTW/MDD/ED 都优于 clip。
2. **running（clip 本来就不差）：缩放/插值都没帮助**，clip 仍是 running 最好。
3. **波形域线性插值（C，字面意义的"线性插值"）不是赢家**：丢样本且 ACD/DTW 更差。**latent 空间的全局缩放（A）优于波形域插值（C）。**
4. 净判断：A（shrink）是一个**更平滑、保形、不丢样本**的修复，能改善难例 walking 的 rhythm（ACD），是一个比 clip 更值得作为方法候选的方向；但对 running 无改善，且整体离 baseline ACD 仍远。

## Per-activity 边界（关键发现）

上面的 clip/shrink 边界都来自 walking+running **合并**的训练 latent 分布。改成**每个动作各自的**训练 latent 边界（`--per-activity-bounds`，生成时已知请求的是哪个动作，合法），结果大幅改善——尤其 ACD（rhythm）：

远程：`output/time_series/..._smooth_repair_peractivity_20260615/`、`reports/smooth_repair_peractivity_tsgbench_20260615/`；本地：`outputs/smooth-repair-peractivity-20260615/`。

| setting (per-activity) | activity | n | MDD | ACD | SD | KD | ED | DTW |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| clip_p05_p95 | walking | 72 | 0.265 | 0.810 | 0.212 | 0.936 | 18.193 | 184.868 |
| shrink_p05_p95 | walking | 72 | 0.277 | 0.890 | 0.174 | 0.053 | 16.175 | 192.903 |
| **shrink_minmax** | walking | 72 | 0.264 | **0.362** | 0.041 | 1.162 | 18.195 | 200.182 |
| clip_p05_p95 | running | 79 | 0.314 | **1.370** | 0.399 | 0.352 | 22.410 | 137.845 |
| shrink_p05_p95 | running | 79 | 0.346 | 1.544 | 0.572 | 0.330 | 18.935 | 161.734 |
| shrink_minmax | running | 79 | 0.332 | 1.544 | 0.358 | 0.579 | 20.260 | 158.286 |

对比（walking ACD）：pooled clip 2.253 → per-activity clip **0.810** → per-activity shrink_minmax **0.362**（baseline 0.165）。

关键判断：

1. **最大的杠杆是 per-activity 边界，而不是 clip vs shrink 的选择。** 把 walking+running 合并分布拆成 per-activity 边界，单 clip 就把 walking ACD 从 2.253 砍到 0.810 → unified 的 rhythm 失败很大程度上是"用了混合动作的 latent 分布做约束"造成的。
2. **per-activity + 保形缩放（shrink_minmax）在 walking 上最好**：ACD 0.362，已接近 unconditioned baseline 0.165，SD 0.041 最低，且不丢样本——这是 clip 一直没修好的难例的实质突破。
3. **running**：per-activity clip（ACD 1.370）已是 running 最好，shrink 没再改善。
4. 综合推荐：**per-activity bounds 作为默认**；walking 用 `shrink_minmax`，running 用 `clip`。

## 局限 / 下一步

- 单 subject、单通道、两动作；ACD 仍未完全到 baseline；running 改善幅度小于 walking。
- 下一步候选：① 在 multi-subject run 上重复 per-activity；② held-out HAR utility；③ 进入 T2（stat-prompt）——statistics-in-prompt 与 per-activity 边界是互补的"给模型幅度先验"的两种方式。
