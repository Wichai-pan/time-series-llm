# T2 Stat-Prompt Results — NEGATIVE (H1 falsified)

日期：2026-06-17
计划：`docs/experiments/experiment_plan_2026-06-15_stat-prompt.md`
Job：Puhti `35003110`（COMPLETED, 4:12, V100×1, gpt2, seed 42, n_bins 4）

## 做了什么

新脚本 `scripts/run_unified_stat_prompt.py`：把每窗口 `mean/std/min/max` 量化成 4 档 bin，复合进现有单条件 token（`Condition: data is walking|m2|s3|n1|x4`），其余与 raw-unified label-only 完全一致（同 FICA basis、gpt2、100 epochs、vLLM 生成）。生成接受数 walking 73 / running 74（与 raw-unified 72/79 相当，stat-prompt 未损害生成量）。

集群 round-trip 自测先行通过（`scripts/test_stat_prompt_roundtrip.py`），确认复合 token 编码/解析/过滤/还原正确。

## 预注册证伪条件（计划 §8）

> Falsify：R2a(raw) ≈ raw-unified（仍爆炸）**或** stat-adherence ≈ 0（模型忽略 stat）。

**两条都触发。**

## R2a — stat-prompt RAW（完全不修复）

| | n | abs_max | std_mean | MDD | ACD | SD | KD | ED | DTW |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| walking | 73 | 62.05 | 2.48 | 0.269 | 3.506 | 0.447 | 25.978 | 62.22 | 813.9 |
| running | 74 | **5341.6** | 92.9 | 0.296 | 1.798 | 5.463 | 50.704 | 2112.4 | **32973** |

对比 raw-unified（label-only，无 stat-prompt）：walking ACD 5.97 / DTW 6026；running ACD 2.285 / DTW 3772、abs_max 1572。

→ **stat-prompt RAW 仍然爆炸**：walking 比 raw-unified 略好（ACD 3.5 vs 5.97），但 running **更差**（abs_max 5341、DTW 32973）。没有从源头消除 value explosion。

## stat-adherence — 模型是否遵守请求的统计量

147 个生成窗口，请求 bin vs 实测值的 Pearson 相关：

| stat | pearson(req_bin, realized) | bin→实测均值 |
|---|--:|---|
| max (x) | **-0.003** | {0:63, 1:100, 2:177, 3:23}（非单调） |
| min (n) | 0.025 | {0:-141, 1:-31, 2:-32, 3:-112} |
| std (s) | 0.040 | {0:62, 1:9, 2:18, 3:115} |
| mean (m) | 0.046 | {0:-2.3, 1:19.7, 2:97.5, 3:0.9} |

→ **所有统计量相关≈0、bin→均值完全非单调** = **GPT-2 实质上忽略了 prompt 里的统计量条件**。这是 H1 机制的直接证伪。

## R2b — stat-prompt + per-activity 轻修复

| setting | act | n | ACD | DTW | SD |
|---|---|--:|--:|--:|--:|
| clip_p05_p95 | walk | 73 | 1.113 | 194.8 | 0.507 |
| shrink_minmax | walk | 73 | 0.851 | 193.3 | 0.958 |
| clip_p05_p95 | run | 74 | 1.718 | 164.8 | 0.603 |
| shrink_minmax | run | 74 | 1.547 | 168.7 | 0.385 |

对比 T1（raw-unified label-only + per-activity 修复）：shrink_minmax walking ACD **0.362** / running 1.544；clip walking 0.810 / running 1.370。

→ **stat-prompt + 修复 并不比 label-only + 修复好，walking 反而更差**（0.851 vs 0.362）。stat-prompt 即使配合修复也没带来增益。

## 结论（H1 FALSIFIED）

1. **prompt-level 统计量条件化（GPT-2 + 此编码）不工作**：模型不遵守数字条件（adherence≈0），raw 仍爆炸（running 更糟），修复后也无增益。
2. **直接回答用户的问题**："改好 prompt 是否就不需要复杂约束？" → **否**。至少在 GPT-2 + 文本编码数字条件下，prompt 改写不能替代 T1 的修复。
3. **真正的天花板被定位得更清楚**：不是 prompt 内容，而是 **GPT-2 把数字当 token、无法真正条件化于数值** 的能力上限。"从源头修"需要更强的条件化机制，而非更丰富的 prompt 文本。

## 局限 / 诚实边界

- 单 subject、单通道、2 动作、1 seed、K=4 bin、GPT-2。stat-adherence≈0 是强信号，但不排除"更粗/更细 bin、把统计量放进数值空间、或更强 base model"会不同。
- running 的 abs_max 5341 比 raw-unified 更大，部分是 OOD latent 的随机性；关键点是 stat-prompt 未能阻止爆炸。

## 下一步候选（按"治本"力度）

1. **更强条件化**（非 prompt 文本）：在 GPT-2 hidden state 加 learned class/stat embedding，或 classifier-free guidance；或换更强 base / instruction-tuned 模型。
2. **constrained decoding**：在 token 级限制生成的数值不出训练范围（把 validity 放进解码器）。
3. **learned latent prior**（GMM/flow）做 generation-time rejection/projection——比 prompt 路线更可能"治本"。
4. 现实落点：**T1 的 per-activity shrink/clip 仍是当前可用方法**；T2 作为一个有价值的负结果写进报告（且对口课程 prompt-design topic：系统说明了 prompt-only 条件化的失败边界）。

## 2026-06-17 对抗审稿 + 指标修正（重要更正）

4 个独立审稿发现上面的分析有方法缺陷，结论需收紧：

- **指标曾太糙**：adherence 上文用 Pearson（档位 vs 原始实测，含 5341 爆炸值）→ 被少数爆炸主导。**已用秩相关复核**：max Spearman=0.010、bootstrap 95% CI [−0.151, 0.166]；高 vs 低档 Mann-Whitney 不显著（walking p=0.79）。→ **负结果（无遵守）在正确指标下成立**，方向无误。
- **但只能收到这么紧**：CI 上界 0.17 只排除"强效应（>0.3）"，排除不了弱效应；且这是在**有缺陷编码**（26 个近乎唯一的复合 token、请求未平衡、目标是间接窗口统计量、1 seed）上得到的。
- **因此不能升级为 H3**（"GPT-2 学不会数字条件"）。第 60 行那句"能力上限"措辞过强，应改为：**"在当前（稀疏、间接、单 seed）编码下，prompt 统计条件化没有产生可检测的遵守；能否在更强编码/更低随机性下生效尚未定论。"**
- 干净判定方案见 `experiment_plan_stat-prompt-v2-hardened.md`：改约束 **latent 量**、平衡请求、秩指标 + CI、加 ceiling 与 synthetic ×1/×3 阳性对照、temp=0 探针、决定性 cell ≥3 seeds。
