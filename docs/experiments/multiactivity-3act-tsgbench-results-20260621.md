# 3-Activity Push (#3): walking/running/cycling TSG

日期：2026-06-21
脚本：`scripts/run_unified_multiactivity.py`(通用 N-动作生成)、`scripts/repair_and_eval_multiactivity.py`
Job：Puhti `35206768`（COMPLETED 4:16）
本地：`outputs/multiactivity-3act-20260621/`

## 配置

把 unified 活动条件生成从 2 类扩到 **3 类:walking / running / cycling**(multi-subject 101/102/105,各 5000 rows/subject;新增 cycling act6 parquet)。一个 GPT-2 联合条件生成,per-activity 修复,canonical TSG（train ref，scaled space）。生成接受数 walking 53 / running 49 / cycling 51。

## 结果（train ref 101/102/105，越低越好）

| setting | activity | n | MDD | ACD | SD | KD | ED | DTW |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| **clip_p05_p95** | walking | 52 | 0.176 | **0.948** | 0.556 | 0.655 | 20.77 | 166.6 |
| **clip_p05_p95** | running | 48 | 0.212 | **0.943** | 0.572 | 0.418 | 20.71 | 138.0 |
| **clip_p05_p95** | cycling | 50 | 0.181 | **2.000** | 0.208 | 1.475 | 19.45 | 185.5 |
| shrink_minmax | walking | 52 | 0.202 | 1.939 | 1.365 | 4.799 | 18.41 | 188.1 |
| shrink_minmax | running | 48 | 0.227 | 1.768 | 0.470 | 0.480 | 19.12 | 151.8 |
| shrink_minmax | cycling | 50 | 0.195 | 1.822 | 0.350 | 1.524 | 19.34 | 207.1 |

## 结论

1. **方法干净地扩到 3 动作**:一个条件模型同时生成三类,各 ~50 窗口,clip 修复有效。`clip_p05_p95` 仍是最佳 variant。
2. **代价**:加 cycling 后,原 walking/running 的 ACD 从 2-动作组合最佳的 ~0.74 升到 ~0.95（共享 3-类 FICA basis 更难);**cycling 本身最难**(ACD ~2.0,clip / 1.82 shrink——cycling 上 shrink 略好)。
3. **MDD 仍低**(0.18/0.21/0.18),DTW 合理(138–207),分布保真度保持。

## 诚实读数

- 这是 train ref;held-out（106/108 三动作)是下一步(cycling held-out 已备 8000 rows)。
- "加更多动作"会**轻微抬高 ACD**——更多类共享一个 5-8 维 latent basis 的容量限制。要进一步推,可加 latent 维度或 per-class 子空间。
- clip 仍是稳健最佳修复。

## 一句话

> 活动条件生成可扩到 3 类(walking/running/cycling),clip 修复下 MDD 仍低、原两类 ACD 仅小幅上升(0.74→0.95),cycling 最难(ACD ~2)——方法可扩,但更多类有容量代价。
