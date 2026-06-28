# Stage B: Synthetic ×1/×3 Amplitude Control — H3 LOCKED (numeric conditioning fails)

日期：2026-06-21
计划：`experiment_plan_stat-prompt-v2-hardened.md`
Jobs：Puhti `35204673`(seed42)、`35204682`(seed7)、`35204683`(seed123),全部 COMPLETED

## 目的

T2 stat-prompt 负结果有三个竞争解释。Stage A 已排除两个；Stage B 用一个**最大可分、完美可识别**的数值对照分离最后两个(H1 编码稀疏 vs H3 GPT-2 学不会数值)。

## 设计

每个训练窗口的 latent 系数复制成两份:原始(token `a0`,×1)和 ×3(token `a1`,×3),token 与幅度**完美可分**。生成用**平衡请求**(每个 amp 等量)、temp 1.3(分布级,A3 证明 greedy 无效)、主目标 = **latent L2**(模型直接控制的量)。脚本 `scripts/run_synthetic_amplitude_control.py`。3 seeds。

> 若 GPT-2 能条件化于数值,请求 `a1` 应产出 ≈3× 大的 latent L2。

## 结果（3 seeds,应≈3.0）

| seed | n(a0/a1) | latent_l2 a1/a0 比 | 95%CI | MWU p(a1>a0) | realized_absmax 比 |
|---|---|--:|--:|--:|--:|
| 42 | 125/93 | 1.03 | [0.74,1.28] | 0.82 | 1.03 |
| 7 | 97/103 | 0.89 | [0.68,1.15] | 0.80 | 0.88 |
| 123 | 96/91 | 1.00 | [0.67,1.31] | 0.62 | 0.95 |
| **POOLED** | 318/287 | **0.94** | — | **0.91** | — |

**三个 seed 全部 ≈1.0**(0.89–1.03),CI 全包含 1、上界 ≤1.31(远离 3.0),所有单侧 MWU 不显著(p 0.62–0.91)。生成接受率正常(单 attempt 即够),输出多样(全 distinct),排除 mode collapse。

→ **连"×1 vs ×3"这种最大可分的数值对比都学不会,且跨 seed 可复现。**

## 完整诊断链(T2 → H3)

| 步骤 | 发现 | 排除 |
|---|---|---|
| T2 | stat-prompt 无遵守、仍爆炸 | — |
| A1 | 秩相关复核仍无遵守(Spearman 0.01) | 排除"指标假象" |
| A2 | latent 量→窗口 max 单调 ρ≈0.8 | 排除"目标不可达" |
| A3 | temp=0 贪婪坍缩成单一输出(连 label 都不分) | 排除"H2 太随机"(且证明条件是分布级) |
| **Stage B** | **×1/×3 跨 3 seed 完全不分(比 0.94)** | **排除"H1 编码稀疏"** |

## 结论（锁定）

> **H3 成立**:GPT-2 在这套文本序列化 SDForger 里,**无法按 conditioning token 调节其输出的数值幅度**,即使对比最大化(×3)、完美可分、跨 seed 可复现。与此对照,**类别条件(label)是有效的**(controllability 0.82)。

这是一个干净的**能力分离**:**轻量 LLM 能条件化于类别,不能条件化于数值/统计量**。因此 prompt 路线对"统计/幅度约束"存在能力上限,不是工程没调好。

## 战略含义

1. **停止 prompt 数值条件化方向**:更多 prompt 变体不会有用(已被最大可分对照证伪)。
2. **要从源头控制幅度/有效性,须走架构级**(绕过"GPT-2 从文本 token 吐数字"):
   - learned latent prior(GMM/normalizing flow)做 generation-time rejection / projection;
   - constrained decoding(token 级限制数值范围);
   - hidden-state class/stat embedding 或 classifier-free guidance;
   - 或换更强 / instruction-tuned base model。
3. **T1 的 per-activity shrink/clip 仍是当前可用方法**(把已生成的越界 latent 拉回)。
4. **对课程/论文**:这是一个严谨、可复现、对口 prompt-design topic 的**负结果**——"prompt-only 数值条件化在轻量文本序列化 LLM 时序生成器上有能力上限;类别可条件化、数值不可,且最大可分对照跨 seed 证实"。配合 T1 的正面结果(per-activity 修复把 walking ACD 2.25→0.36),构成"正面方法 + 负面边界"的完整故事。

## 产物
- 远程:`output/time_series/pamap2_subject101_synthetic_amplitude_control{,_seed7,_seed123}_20260621/`
- 本地:`outputs/synthetic-amplitude-control-20260621/`
- 脚本:`scripts/run_synthetic_amplitude_control.py`;A2/A3 脚本 `run_stat_prompt_generate_only.py`
