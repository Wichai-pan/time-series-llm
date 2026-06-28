# 本周实验总结(6/22 开会后 → 6/28)

> 主线:① 对上周导师建议逐条反馈(换 Qwen / 多动作 / normalization / 画图 / multi 减半);② 更进一步——系统刻画 prompt 可控性边界。所有 TSG 越低越好;adherence 越高越可控。

## 主题一 · 把"最优配置"落到 Qwen(回应导师"用 Qwen")

| 实验 | 配置 | 结果 |
|---|---|---|
| Qwen baseline(3-seed) | 单模型 label-only,multi-subject,**无修复** | walk ACD **2.0±0.25**、run **2.15±0.20**(锁定可信 baseline) |
| Qwen 3-动作 无修复 | walk/run/cycling 联合生成 | ACD 2.14/2.25/2.23,**不爆炸**(KD<1.2) |
| **Qwen 3-动作 + 修复** | + per-activity clip | **walk 1.02 / run 1.29 / cycling 1.55**;cycling 比 gpt2(2.0)好 |
| **Qwen 2-动作 + 修复** | + per-activity clip | **walk 0.728 / run 0.893**(对上 gpt2 头条 0.74/0.74:走路打平) |
| Qwen 最优 held-out | 106/108 未见 subject | **walk 0.98 / run 1.46**(跑步比 gpt2 的 1.74 好,跨人退化更轻) |
| Qwen + norm + 修复(组合最优) | 叠 activity_series 归一化 | train 0.74/0.68、heldout 1.39/1.53 → **trade-off,净不划算,不加 norm** |

**结论**:最优配置现完整落在 Qwen(train+held-out),质量=gpt2 同级、难点更好、且源头不爆炸(修复是锦上添花非必需)。

## 主题二 · 回应具体建议

| 建议 | 实验 | 结果 |
|---|---|---|
| 查 multi 减半 | 加埋点重跑 **4 seed** 的拒绝拆解 | **accept 0.75、零质量拒绝**(解析/标签/去重全 0)→ **不是系统性减半,最早的 0.40 是偶发** |
| Qwen 上加 normalization | activity_series_zscore + Qwen | ACD 全面小幅降(walk heldout **1.20→0.96**),不爆炸 → 有正面效果(但叠到最优上是 trade-off,见上) |
| 一个 sample 一张图 | `plot_samples.py` | 按 NN 距离选好/坏样本,各一格,不叠成一团(fig6) |

## 主题三 · prompt 可控性边界(本周新发现,7 轮自主循环)

| 轮 | 实验 | 结果 |
|---|---|---|
| R1 | max/std/period 三描述符可控性(seed42) | max 0.47 ≫ std 0.09 ≫ period -0.04 |
| R2 | 为什么 period 控不了 | 真实 period 动作内近常数(walk≈60/run≈75,仅 4-6 离散值)、实测不跟请求 |
| R3 | 可控性 ∝ 描述符有效分辨率 | max 16 档→可控,period 4 档→不可控 |
| R4 | seed7 稳健性 | **max 0.40±0.07、std 0.18、period ~0**(排序稳健) |
| R5 | + range 描述符 | 总表 **max 0.40 > std 0.18 > range 0.12 > period -0.02** |
| R6 | **epoch 消融**(验"过训练变死板") | max 随 epoch:20→0.18、**50→0.53(甜点)**、100→0.40、200→0.42;period 全程~0 |
| — | 50 vs 100 生成质量 | TSG 打平(walk 偏100、run 偏50);50 = 保真度不输 + 可控性更好 + 省一半算力 |

**最终机制结论(双层)**:
> prompt 可控性 = **表示层(能不能控)** × **训练层(控多好)**。
> - 幅度(max)落在"可调系数"上 → 能控;频率(period)焊在"固定 FICA 基"里 → 任何训练量都控不了。
> - 对能控的幅度,存在训练甜点(~50 epoch);过训练变死板。
> → "prompt 描述更丰富"不提升控制;决定因素是描述符落在可调系数(幅度)还是固定基(频率)上。

## 主题四 · 诚实 baseline(SDForger 内部消融,界定 LLM 贡献)

| 实验 | 结果 |
|---|---|
| 高斯系数 baseline(LLM→高斯,同解码) | 无条件生成 TSG **≈ 甚至优于 LLM** → LLM 在无条件保真度上无可测增益 |
| naive jitter(steelman) | 下游 HAR 与 SDForger 打平 |
| 条件高斯可控性 | pooled 0.41 ≈ LLM 0.36 → 分箱内可控性也被 trivial 追平 |
| NN 记忆化 | 比值 0.73/0.87,距离~9-10 → 非整窗拷贝 |

→ **诚实定位:LLM 价值不在无条件保真度(被 trivial 追平),而在(有边界的)可控性 + 非拷贝生成。**

## 运维
磁盘配额曾被 Qwen checkpoint 撑满(268GB)→ 已清理(结果保留)+ job 加自动清 model,防复发。

## 产物
- 图表:`outputs/meeting-figures-20260629/`(7 张,会议用)
- 文档:`qwen-best-config-results-20260625.md`、`descriptor-control-results-20260625.md`、`project-summary-and-workshop-readiness-20260628.md`
