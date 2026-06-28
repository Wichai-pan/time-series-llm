# Qwen 最优配置 + gpt2 公平对比(2026-06-25,改进上周进度)

上周"最优配置 ACD 0.74"是 **gpt2** 的;导师要求用 **Qwen**。本次把"multi-subject + per-activity 修复"的最优配方做到 Qwen 上,并与 gpt2 做**公平对比(都带修复)**。Qwen 经 `run_unified_multiactivity.py`(已加 HF 后端)生成、`repair_and_eval_multiactivity.py` 修复评估。

## 3-动作(walk/run/cycling,multi-subject 101/102/105,train ref,clip 修复)

| 配置 | walk ACD | run ACD | cycling ACD | 说明 |
|---|--:|--:|--:|---|
| gpt2 + 修复 | 0.95 | 0.94 | 2.00 | 上周方法,gpt2 必须修(raw 5.97 爆炸) |
| Qwen **无修复** | 2.14 | 2.25 | 2.23 | 不爆炸,但 raw ACD 偏高 |
| **Qwen + 修复(clip)** | **1.02** | **1.29** | **1.55** | MDD 0.18/0.22/0.19 |

→ 修复对 Qwen **也有效**(2.2 → 1.0–1.55)。公平对比(都修复)下:
- **walk/run**:gpt2(0.95/0.94)略优于 Qwen(1.02/1.29);
- **cycling(最难)**:**Qwen(1.55)明显优于 gpt2(2.0)**;
- **关键差异**:**gpt2 必须修复**(raw 爆炸,修复=创可贴);**Qwen 不修也合法**(raw 不爆),修复只是锦上添花。

## 结论(改进点)

> **"最优配置"现在落在 Qwen 上**:Qwen + per-activity clip,三动作 ACD 1.0–1.55、MDD 低、**不爆炸**。相比 gpt2:walk/run 相当、**cycling 更好**、且**不依赖修复才能合法**。回应了导师"用 Qwen 当 baseline"。

## 2-动作直接对上上周头条(gpt2 0.74)

| 2-act 配置 | walk ACD | run ACD | walk MDD | run MDD |
|---|--:|--:|--:|--:|
| **Qwen + clip** | **0.728** | 0.893 | 0.184 | 0.226 |
| gpt2 + clip(上周头条) | 0.740 | 0.738 | 0.171 | 0.211 |

→ **走路打平(Qwen 0.73 ≈ gpt2 0.74,Qwen 略好)**;跑步 gpt2 略低(0.74 vs 0.89)。**Qwen 达到了和 gpt2 同级的最优质量,且不依赖修复才合法。**

## 完整对比(汇报用)

| | 2-act walk/run | 3-act +cycling | 爆炸 |
|---|---|---|---|
| gpt2 + clip(上周) | 0.74/0.74 | cycling 2.0 | 💥 必须修复 |
| **Qwen + clip(本次)** | **0.73/0.89** | cycling **1.55** | ✅ 不修也合法 |

**口径**:Qwen+修复 = gpt2 同级质量(走路打平),cycling 更好,且源头不爆炸(修复是锦上添花非必需)。最优配置落在 Qwen,回应导师。

## Held-out(106/108)泛化 — Qwen 最优(2-act clip)

| | train walk/run | held-out walk/run |
|---|---|---|
| **Qwen + clip** | 0.73 / 0.89 | **0.98 / 1.46** |
| gpt2 + clip | 0.74 / 0.74 | 1.40 / **1.74** |
| gpt2 + shrink | — | 0.69 / — |

→ **跑步 held-out:Qwen 1.46 < gpt2 1.74** —— gpt2 最头疼的跑步跨人漂移(0.74→1.74),**Qwen 退化更轻(0.89→1.46)**。走路 held-out 0.98 落在 gpt2 的 0.69-1.40 区间内。**Qwen 最优配置 train+held-out 完整,泛化≥gpt2,难点更强。**

## 组合 normalization(Qwen + norm + clip)— 负结果,trade-off

| 配置 | train walk/run | held-out walk/run |
|---|---|---|
| Qwen + clip(最优) | 0.73 / 0.89 | **0.98** / 1.46 |
| Qwen + norm + clip | 0.74 / **0.683** | 1.389 / 1.53 |

→ norm 帮了 train-跑步(0.89→0.68),但**伤了 held-out-走路(0.98→1.39)**,净效果 trade-off。**结论:不加 norm,最优配置保持 `Qwen + per-activity clip`(held-out 泛化更好,泛化更重要)。** 与"normalization 不一致(修一个坏一个)"的既有发现一致。

## 运维
所有 Qwen run 的 `model/` checkpoint(12GB/个)生成后即删,防 scratch 配额再满(本次曾满 268GB)。
