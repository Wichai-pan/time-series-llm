# Held-out-Subject HAR Utility (#1) — POSITIVE

日期：2026-06-21
脚本：`scripts/evaluate_har_utility_heldout.py`
本地：`outputs/har-utility-heldout-20260621/`

## 设计

下游 walking-vs-running 分类(LogisticRegression,hand_acc16_x,window 300)。
- **train-real**：subjects 101/102/105
- **synthetic**：组合最佳生成(multi-subject unified + per-activity 修复)
- **test**：**held-out subjects 106/108(生成器和分类器都没见过)**

这是有意义的泛化检验——不是同 subject 后半段,而是完全未见的人。

## 结果

| 条件 | accuracy | balanced acc | macro F1 |
|---|--:|--:|--:|
| majority | 0.500 | 0.500 | 0.333 |
| real-only | 0.742 | 0.742 | 0.728 |
| synthetic-only | 0.712 | 0.712 | 0.696 (clip) / 0.704 (shrink) |
| **real+synthetic** | **0.758** | **0.758** | **0.746** |

train n：real 116、synthetic 109、real+synthetic 225；test n=66。clip 与 shrink synthetic 结果几乎一致。

## 结论

- **加 synthetic 在完全未见的 subject 上把分类从 0.742 → 0.758(F1 0.728 → 0.746)** —— 项目最强的应用证据:synthetic 数据真的有下游 utility,且能跨 subject 泛化。
- synthetic-only 单独也有 0.71(接近 real-only),说明生成数据本身携带可判别的活动信号。

## 诚实边界

- 单通道、2 动作、1 个简单 classifier、1 split、test 仅 66 窗口 → provisional 正面信号。
- 强化方向:多 seed/split + CI、多通道、3+ 动作(见 multiactivity 实验)。
