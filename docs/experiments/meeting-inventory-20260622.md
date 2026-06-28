# 实验盘点（6/8 开会以来 → 今日汇报）

> 指标规范:**生成质量主指标 = TSG 保真度**(MDD/ACD/SD/KD/ED/DTW,对齐 TSGBench,越低越好);**可控性 = adherence**(请求 vs 实测的 Spearman);**下游 utility = HAR acc/F1**(辅助、TSTR 式,**不是生成质量主指标**)。

## A 线 · 修复/保真度（正面方法,主线)

| # | 实验(日期) | 数据 & 设置 | 主结果 | 指标 | 该用吗 | 全不全 | 今天能讲 |
|---|---|---|---|---|---|---|---|
| A1 | **T1 平滑修复 + per-activity 边界**(6/15) | subject101 单人,hand_acc16_x,train_len 5000,FICA dim5,walking72/running79 | walking ACD **2.25→0.81(per-act clip)→0.36(per-act shrink_minmax)**;running per-act clip **1.37** | TSG 全6项 | ✅正确 | ✅全(但单run无seed) | ✅**核心正面发现** |
| A2 | **组合最优 multi-subject + per-act**(6/21) | 101/102/105,train_len15000,FICA dim8,walking51/running58 | **train ref 两动作 ACD 都 0.74**,MDD 0.17/0.21(低于baseline);held-out walking 0.69/running 1.74 | TSG 全6项 | ✅正确 | ⚠️单run(另有5-seed:1.04±0.41/0.69±0.14) | ✅**最好TSG配置** |
| A3 | **3 动作扩展(+cycling)**(6/21) | 101/102/105各5000行,clip修复,~50窗/类 | ACD walking0.95/running0.94/cycling2.0,MDD 0.18/0.21/0.18 | TSG 全6项 | ✅正确 | ⚠️仅train ref(held-out待) | ✅可扩性(cycling有代价) |

## B 线 · Prompt 设计/条件化（课程主题,正面+负面边界)

| # | 实验(日期) | 数据 & 设置 | 主结果 | 指标 | 该用吗 | 全不全 | 今天能讲 |
|---|---|---|---|---|---|---|---|
| B1 | **T2 stat-prompt → Stage A/B**(6/17–6/21) | gpt2,单subject,stat-prompt/×1×3对照,3 seeds | T2 adherence Spearman **0.01**(无遵守)+仍爆炸;Stage B ×1/×3 比 **0.94**(应3.0),CI含1,跨3seed → **gpt2 能条件化于类别、不能于数值(锁定)** | adherence Spearman+CI+MWU | ✅正确 | ✅**很全(3seed+CI)** | ✅**干净负结果,最对口课程** |
| B2 | **Qwen stat-prompt(强模型+实值编码)**(6/21) | Qwen2.5-1.5B,实值 vs 粗档,3 seeds(42/7/123) | 实值 pooled Spearman **0.36**(0.20/0.51/0.38);粗档 **0.10**;walking per-label 0.41–0.56 vs 0.0003 | adherence Spearman | ✅正确 | ⚠️**无CI、跨seed波动大** | ✅provisional正面(实值>粗档) |

## C 线 · 下游 utility

| # | 实验(日期) | 数据 & 设置 | 主结果 | 指标 | 该用吗 | 全不全 | 今天能讲 |
|---|---|---|---|---|---|---|---|
| C1 | **Held-out HAR utility**(6/21) | train 101/102/105+synthetic,test=未见106/108,LogReg,test n=66 | real-only 0.742 → real+syn **0.758**(单split);**5-seed复核→0.745±0.013(中性)** | accuracy/F1 | ⚠️**辅助指标,非生成主指标** | ⚠️**单split虚高,n=66抽样CI≈±0.11** | ⚠️**谨慎,必带caveat** |

## D 线 · 本次新跑的对照/sanity（6/22,我加的控制,非主结果)

| # | 对照 | 主结果 | 作用 | 今天能讲 |
|---|---|---|---|---|
| D1 | 跨人 + 同subject real-vs-real 地板 | 跨人 ACD 1.22/1.02;同subject 2.49±1.47/0.93 | 校准"ACD多低算好" | 可选(校准用) |
| D2 | naive jitter(真实+噪声) | TSG极低;HAR 调好后 real+jitter 0.742 | 下限基线 | ⚠️暴露"HAR与jitter打平" |
| D3 | NN 记忆化 | 比值 0.73/0.87,距离~9-10 | 排除整窗拷贝 | 可选(防"抄数据"质疑) |
| D4 | 高斯系数生成基线 | TSG walking1.07/running0.60 ≈/优于LLM | 隔离LLM贡献 | ⚠️暴露"无条件生成被高斯追平" |
| D5 | 条件高斯可控性 | pooled Spearman 0.413 ≈/>LLM 0.36 | 隔离可控性贡献 | ⚠️暴露"可控性被条件高斯追平" |

## 指标体检（你问的"指标全不全/对不对"）

- **该用的指标我们都用了**:生成→TSG(对齐 TSGBench),可控性→adherence Spearman,下游→TSTR式 HAR。**没有用错的主指标。**
- **唯一要小心的是 HAR 的 acc/F1**:它是**辅助**,不能当"生成质量"的主卖点(你之前的直觉是对的);且我们的 HAR 探针 **n=66 太小、单split**,精度被高估。
- **最大的完整性缺口**:A 线 TSG 多数是**单 run、没有 seed 方差/CI**(只有 Stage B 和那个 5-seed 复核有 CI)。导师可能问"这些 ACD 数字稳不稳"。

## 今天 2 小时该讲什么（建议)

**主线讲 4 个(都硬、指标对、能站住)**:
1. **A1+A2**:per-activity 修复 + multi-subject → ACD 大幅下降(walking 2.25→0.36;两动作 0.74),MDD 低于 baseline。【正面方法】
2. **B1**:gpt2 能条件化于类别、不能于数值(3 seed + CI 锁定)。【干净负结果,对口课程】
3. **B2**:换 Qwen + 实值编码,可控性出现(0.36 vs 粗档 0.10)。【provisional 正面,prompt-design 主题】
4. **A3**:方法可扩到 3 动作。【可扩性】

**谨慎讲(带 caveat)**:C1 HAR(说"provisional 正面信号,单 split,在补 CI")。

**D 线对照**:看你策略——它们让你**面对"baseline 呢""会不会抄数据"的质疑时有据可答**,但 D4/D5 也说明无条件生成/分箱可控性被 trivial 追平。**建议至少把 D3(不抄数据)和地板讲了**;D4/D5 可作为"我们诚实地做了 baseline,指明下一步往 open-ended 条件化走"。
