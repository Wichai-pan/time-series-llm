# 实验计划 2026-06-22 — Qwen 扩充(本周,7/10 draft 前)

> 证据标准:**Internal Direction Check → Report-draft 级**(不是 paper-final)。每个实验:一个假设、一条证伪、一条决策规则、对照、指标。一次只动一个变量。
> 固定对照(所有 run 共享):Qwen2.5-1.5B base、FICA、train_length 15000、epochs 100、multi-subject 101/102/105、hand_acc16_x、HF 后端、temp 1.0。
> 评估规范(经本周教训硬化):TSG MDD/ACD 为主 + **两个地板**做参照(同 subject 2.49/0.93、跨人 1.22/1.02);HAR 是**辅助且 n=66 欠功效 → 必须 Wilson CI + steelman jitter/Gaussian 对照**;adherence/accept 等噪声量 **≥3 seed 给 CI**。

## 目标(导师会上 4 个方向 → 本周可执行)

(a) 更多/更复杂 activity;(b) 去 label、只用数据统计 → unlabeled 泛化;(c) 下游 HAR pre-train+fine-tune;(d) **先把 Qwen 跑成一组有效 baseline**。

---

## Phase 0 · 固化 Qwen baseline(P0,最先做,便宜)

**Claim**:Qwen2.5-1.5B 是一组**有效、稳定、可复现**的 SDForger baseline。

| Run | 变量 | 指标 | seeds | 证伪 | 决策 |
|---|---|---|---|---|---|
| **B1** Qwen multi label-only TSG | — | MDD/ACD(train+held-out) | **3** | 3-seed ACD 方差 > 均值的 50% → baseline 不稳 | 稳→锁为 baseline 数字进 report |
| **B2** Qwen multi stat-prompt adherence | — | pooled+per-label Spearman | **3** | 3-seed 均值 CI 含 0 → 可控性不成立 | 已知单 subj 0.36;给 multi 的 CI |

> 现状:B1 有 1 seed(walk train 2.32/heldout 1.20…),B2 有零散数据。**补到 3 seed + CI** 即可锁 baseline。便宜(已有脚本)。

## Phase 1 · 扩 activity(导师方向 a)

**Claim**:Qwen 把 SDForger 扩到 3+ 动作时,比 gpt2 更稳(不爆炸),保真度不显著退化。
**H1**:Qwen 3-act 的 ACD ≈ gpt2 3-act(walk/run 0.95、cycling 2.0)但**无爆炸**(KD<5)。
**证伪**:Qwen 3-act 出现爆炸(KD>20 / abs max>10)或 ACD 比 2-act 翻倍以上。

| Run | 变量 | 数据 | 指标 | seeds | 对照 |
|---|---|---|---|---|---|
| **A1** Qwen 3-act unified | +cycling | walk/run/cycling | per-act TSG(±clip) | 1(+2 if noisy) | gpt2 3-act(已有)、Qwen 2-act |
| **A2**(stretch) Qwen 4-act | +1 动作(如 ascending/lying) | 4 动作 | per-act TSG | 1 | A1 |

> 复用 `run_unified_multiactivity.py`(gpt2 版),改 `--model Qwen + HF 后端`(同 P1 norm 脚本的 HFLLM 套路)。GPU,中等。

## Phase 2 · 去 label、只用数据统计(导师方向 b,最有新意)

**Claim**:仅凭"从数据本身提取的统计量"(无显式 label)就能条件化生成出**正确动作**的窗口 → 模型可用 unlabeled data。
**H2**:label-free(只给 mean/std/min/max)生成的窗口,其 TSG 接近 label-conditioned,且**能被分类器判回正确动作**(说明动作信息藏在统计量里)。
**证伪**:label-free 生成的窗口分类器判不出动作(≈随机 50%),或 TSG 比 label-conditioned 差一倍以上。

| Run | 变量 | 条件 token | 指标 | 对照 |
|---|---|---|---|---|
| **L0** label-conditioned(对照) | — | `data is walking` | TSG + activity-recoverability(分类器判回准确率) | — |
| **L1** stat-only(无 label) | 去 label | `data is m?\|s?\|n?\|x?`(只统计,不写动作名) | 同上 | L0 |
| **L2**(stretch) cross-activity 取证 | — | 给 running 的统计 → 看能否生成 running | 生成窗口的实测统计 vs 请求 | L1 |

> **关键指标 = activity-recoverability**:在真实数据上训一个 walk/run 分类器,拿它判 L1 生成的窗口属于哪个动作。高准确率 = 统计量确实编码了动作 = 可去 label。需新脚本(改 prompt 去 label + 分类器评估)。GPU+CPU。

## Phase 3 · 下游 HAR(导师方向 c)

**Claim**:Qwen 合成数据对下游 HAR 有用,尤其在"真实数据稀缺 + 合成 pre-train"场景。
**H3a**:real+Qwen-syn ≈ real-only(held-out),且 ≥ steelman jitter/Gaussian(已知这俩追平,所以这里只求"不输")。
**H3b(导师重点)**:Qwen-syn **pre-train + 10–20% 真实数据 fine-tune** ≈ 100% 真实数据训练。
**证伪**:H3b 远低于 all-real(差>2×Wilson CI)→ 合成 pre-train 无下游价值。

| Run | 变量 | 训练数据 | 指标 | 对照 |
|---|---|---|---|---|
| **H1** real-only / syn-only / real+syn | 加不加 syn | 各条件 | acc/F1 + **Wilson CI** | jitter、Gaussian(已有) |
| **H2** pre-train+fine-tune | fine-tune 用 10%/20%/50% 真实 | syn pre-train → 少量 real | acc vs all-real | all-real、scratch-on-little-real |

> 复用 `evaluate_har_utility_heldout.py` + 加 pre-train/fine-tune 分支。**主要 CPU**(分类器),便宜。⚠️ 探针 n=66 弱,务必 Wilson CI + 不夸大。

## Phase 4 · stretch(有余力再做)

| Run | Claim | 指标 | 备注 |
|---|---|---|---|
| **C1** per-subject 归一化 max 条件 | 修复 multi adherence(0.10) | adherence Spearman | 课程 prompt-design 主题,深挖可控性 |
| **S1** Qwen2.5-3B vs 1.5B | 更大模型恢复 multi 可控性 | adherence + TSG | GPU 较重,测"强模型"假设 |
| **M1** 多通道(hand 6-channel) | Qwen 处理多变量 | 多变量 TSG | SDForger 招牌但工程量大,**下一阶段** |

---

## 本周优先级(决策)

1. **P0 固化 baseline**(B1/B2,3-seed)— 0.5 天,便宜,**导师明确要的"有效 baseline"**。
2. **Phase 1 扩 activity**(A1)— 1 天,导师方向 a,扩展 baseline。
3. **Phase 2 去 label**(L0/L1)— 1.5 天,导师方向 b,**最有新意、最对口 prompt-design**。
4. **Phase 3 下游 HAR pre-train/fine-tune**(H1/H2)— 1 天,导师方向 c,便宜。
5. Phase 4 stretch — 有余力。

→ **能填满 report 的 Experiments + Analysis 章节**(每个 phase 一节:setup/结果/好坏原因)。

## Stop conditions / 决策规则

- baseline 3-seed 稳 → 锁数字,不再重复跑。
- A1 不爆炸 → "方法可扩到 3+ 动作"成立,写进 report;爆炸 → 回到 2-act,记为边界。
- L1 activity-recoverability 高 → "可去 label"成立(强卖点);低 → 诚实记为负结果(label 必需)。
- H2 接近 all-real → "合成数据稀缺场景有用"成立;远低 → 诚实记 gap。
- **compute 上限**:本周 ≤ ~15 个 Qwen GPU job;7/10 draft 前停止新实验、转写作。

## Reviewer 风险(已预案)

- HAR 探针 n=66 弱 → 用 Wilson CI、配 jitter/Gaussian steelman、不宣称超越。
- TSG 小样本噪声 → 配两个地板、不做精细排序。
- adherence run-variable → 3 seed CI。
- label-free 的 confound:统计量可能泄漏 label 信息 → 这正是 H2 要测的(是 feature 不是 bug),但要诚实说明"统计量编码了动作"。
- 单通道/2–3 动作/单 split → claim 收窄到"PAMAP2 hand_acc16_x 上的 provisional baseline + 方向探索"。
