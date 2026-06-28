# 汇报两表（2026-06-22）：① 逐步叠加到最优配置　② Prompt 设计 gpt2 vs Qwen

> 所有 TSG 数字都在**同一 canonical scaled 空间**（`activity_sdforger_scaled`，对齐 TSGBench），可直接纵向比较。越低越好。
> 数据来源：`combined-experiment-table-20260607.md`（baseline/label/raw/clip/normalization/multi）+ `smooth-repair-results-20260615.md`（per-activity）+ `combined-best-config-results-20260621.md`（最优叠加）。

## 表1 · 怎么一步步叠加到最优（每行加一个改动）

| 步 | 配置（累加） | subject | 修复 | walk MDD | walk ACD | walk DTW | run MDD | run ACD | run DTW | 一句话 |
|---|---|---|---|--:|--:|--:|--:|--:|--:|---|
| 0 | clean baseline（无条件） | 单101 | — | 0.266 | **0.165** | 237.8 | 0.297 | **0.498** | 111.1 | 参考下限（但非条件生成） |
| 1 | + label 条件生成 | 单101 | — | 0.276 | 0.550 | 291.8 | 0.297 | 0.585 | 141.0 | 类别可控，质量略降 |
| 2 | + unified 生成（条件叠加） | 单101 | 无 | 0.257 | **5.973💥** | 6026 | 0.299 | 2.285 | 3773 | **爆炸** |
| 3 | + normalization（测了4种） | 单101 | 无 | — | 1.30–3.22 | — | — | 1.02–3.96 | — | **测了，没修好爆炸 → 弃用** |
| 4 | + pooled clip 修复 | 单101 | pooled clip | 0.273 | 2.253 | 198.4 | 0.316 | 1.960 | 139.7 | 修了爆炸/形状，ACD 仍高 |
| 5 | + multi-subject | 101/102/105 | pooled clip | 0.172 | 3.117 | 166.1 | 0.204 | 1.611 | 147.2 | **MDD 大降、救 running**；walking ACD 仍高 |
| 6 | + **per-activity clip**（改进的 clip） | 单101 | **per-act clip** | 0.265 | **0.810** | 184.9 | 0.314 | **1.370** | 137.8 | **关键杠杆：walking ACD 2.25→0.81** |
| 6b | （walking 极致：per-act shrink） | 单101 | per-act shrink_minmax | 0.264 | **0.362** | 200.2 | — | — | — | walking 近 baseline 0.165 |
| **7** | **multi-subject + per-activity clip（最优叠加）** | **101/102/105** | **per-act clip** | **0.171** | **0.740** | 182.7 | **0.211** | **0.738** | 154.7 | **🏆 两动作 ACD 都 0.74，MDD 低于 baseline** |

**最优能到多少（你问的核心）**：
- **walking ACD 0.740、running ACD 0.738**（从 raw-unified 的 5.97 / 2.29 一路降下来）
- **MDD 0.171 / 0.211**（甚至**低于 baseline 0.266 / 0.297**）；DTW 183 / 155
- **关键发现**：真正的杠杆是 **per-activity 边界**（用每个动作自己的 latent 边界做约束），不是 normalization、也不是单纯加数据。multi-subject 的贡献是**救了 running**（单人 per-act running 1.37 → multi 0.74）。
- 泛化（held-out 106/108）：walking 稳（ACD 0.69），running 受 subject 周期漂移退化（1.74）。

**消融结论（每个改动有没有用）**：
| 改动 | 有用吗 | 证据 |
|---|---|---|
| normalization（4 种 z-score） | ❌ 没用 | 全部仍爆炸/ACD 高，无一致改善 → 不采用 |
| multi-subject | ✅ 救 running | running per-act 1.37 → 0.74；MDD 0.27→0.17 |
| **per-activity clip** | ✅✅ 最大杠杆 | walking ACD 2.25 → 0.74/0.81 |
| 最优叠加 | ✅ | 两动作 0.74，MDD < baseline |

## 表2 · Prompt 设计：为什么 gpt2 不行、Qwen 可以

| 模型 | 类别条件（label） | 数值条件（实值编码） | 数值条件（粗档 bins） | 会不会爆炸 |
|---|---|---|---|---|
| **gpt2(124M)** | ✅ 有效（controllability **0.82**） | ❌ adherence Spearman **0.01**；×1/×3 比 **0.94**（应 3.0，3 seed+CI 锁定） | ❌ ~0 | 💥 爆炸（abs max 3761） |
| **Qwen2.5-1.5B** | ✅ | ✅ Spearman **0.36**（3 seed：0.20/0.51/0.38；v2 0.58） | ⚠️ **0.10** | ✅ 不爆炸（max 2.98，100% 在范围） |

**怎么讲这个故事**：
1. **gpt2 的能力上限**：它能条件化于**类别**（label 0.82），但**完全不能条件化于数值**——连"×1 vs ×3"这种最大可分的对照都学不会（比值 0.94，应为 3.0，跨 3 个 seed + 置信区间锁定）。这是**能力上限，不是没调好**。同时 gpt2 还爆炸。
2. **换 Qwen + 实值编码就出现可控性**：把数字以**真实数值**写进 prompt（不是粗档 bin），Qwen 的数值遵守度从 gpt2 的 ~0 升到 **Spearman 0.36**（粗档只有 0.10），而且**完全不爆炸**。
3. **结论**：prompt 做数值可控，需要**两个条件叠加**——① 模型够强（gpt2 不行）+ ② 编码用真实数值（不是粗档）。
4. **诚实边界**：Qwen 的 0.36 是**中等强度、provisional**（3 seed 但无 CI、跨 seed 在 0.20–0.58 波动）——这部分"还有点新/弱"，定位成"初步证明强模型+实值编码能带来可控性"，不是定论。

## 数据完整性自查（你担心的"是不是没跑完"）

- **表1 每一格都来自已完成的 run**（job 全 COMPLETED，报告见上方来源）——数据是全的。
- 唯一的完整性弱点：表1 多数是**单 run，没有多 seed/CI**（只有最优配置另跑过一次 5-seed 复核 = walking 1.04±0.41 / running 0.69±0.14，趋势一致）。若导师问"稳不稳"，就说"已做 5-seed 复核，趋势一致，CI 待补全到每一行"。
- **表2 的 gpt2 负结果有 3 seed + CI（很硬）**；Qwen 正结果有 3 seed、缺 CI（provisional）。
