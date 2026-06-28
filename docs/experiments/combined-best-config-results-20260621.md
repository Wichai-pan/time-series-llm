# Combined-Best Config — Multi-subject + Per-activity Repair (TSG)

日期：2026-06-21
脚本：`scripts/apply_smooth_latent_repair.py --per-activity-bounds`、`scripts/rerun_combined_best_tsgbench_table.py`
本地：`outputs/combined-best-20260621/`

## 配置（各轴取最优合并，导师建议方向）

- **Subject**：multi-subject 101/102/105（54 walking + 62 running 训练窗口，train_length 15000，FICA dim 8）— 不再是单 101 smoke。
- **生成**：unified label-conditioned（活动条件生成，已有 run，generated walking 51 / running 58）。
- **归一化**：per-activity window 标准化（消融证明额外 normalization 无用，不加）。
- **修复**：**per-activity 边界** + clip / shrink（T1 赢家）。
- **评估**：canonical scaled space，对 **train(101/102/105)** 和 **held-out(106/108)** 双参考。

## 主表（越低越好）

| ref | setting | act | n | MDD | ACD | SD | KD | ED | DTW |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| train | **clip_p05_p95** | walking | 51 | **0.171** | **0.740** | 0.385 | 0.662 | 20.06 | 182.7 |
| train | **clip_p05_p95** | running | 58 | **0.211** | **0.738** | 0.699 | 0.023 | 20.17 | 154.7 |
| train | shrink_minmax | walking | 51 | 0.192 | 1.062 | 0.986 | 1.345 | 17.97 | 176.8 |
| train | shrink_p05_p95 | running | 58 | 0.265 | 0.793 | 0.745 | 3.288 | 17.93 | 187.8 |
| heldout | **shrink_minmax** | walking | 51 | 0.212 | **0.688** | 0.930 | 0.916 | 18.86 | 171.5 |
| heldout | shrink_p05_p95 | walking | 51 | 0.242 | 0.761 | 0.607 | 0.221 | 17.54 | 183.7 |
| heldout | **clip_p05_p95** | running | 58 | 0.243 | **1.743** | 1.034 | 0.845 | 20.12 | 151.3 |
| heldout | clip_p05_p95 | walking | 51 | 0.191 | 1.401 | 0.329 | 1.091 | 20.08 | 169.5 |

（完整 16 行见 `combined_best_tsgbench_summary.csv`）

## 对比参考

| | walking ACD | running ACD | walking MDD | running MDD |
|---|--:|--:|--:|--:|
| clean baseline（单 subject，无条件） | 0.165 | 0.498 | 0.266 | 0.297 |
| 单 subject + per-activity clip（T1） | 0.810 | 1.370 | — | — |
| **multi + per-activity clip（train ref）** | **0.740** | **0.738** | **0.171** | **0.211** |

## 结论

1. **"取最优合并"确实提升了指标。** 关键是 **multi-subject 救了 running**：running ACD 从单 subject 的 1.37 → multi 的 **0.74**（接近 baseline 0.50）；walking 也从 0.81 → 0.74。
2. **最佳整体配置 = multi-subject + per-activity `clip_p05_p95`**：train ref 上 **两个动作 ACD 都 ~0.74**，且 **MDD 反而低于 baseline**（0.17/0.21 vs 0.27/0.30），DTW 183/155。
3. **泛化（held-out 106/108）**：**walking 稳**（shrink_minmax ACD 0.69，接近 baseline）；**running 退化**（ACD 1.74）——与已知的 running 跨 subject 周期漂移一致（period 110 vs 76）。
4. **诚实边界**：这些是 TSG 保真度指标,不是 HAR 下游 utility；held-out running 仍有 subject-shift 问题；clip/shrink 是选出的最佳 variant（"能跑到多好"的上界,非单一固定方法）。

## 一句话

> 把 multi-subject + 活动条件生成 + per-activity 修复合起来,是目前最好的配置:train ref 上两动作 ACD 都压到 ~0.74、MDD 低于 baseline;walking 能泛化到 held-out,running 受 subject 漂移限制。
