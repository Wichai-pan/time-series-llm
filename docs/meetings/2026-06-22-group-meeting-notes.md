# 组会纪要 2026-06-22（转录自录音,whisper 已校正术语）

原始转录:`/Users/huataipan/Wichai/small_demos/whisper/标准录音 10.txt`(490 段)。
形式:三位同学轮流 update + 导师(Si Zuo)反馈 + report/排期安排。**你 = 第三位汇报人(Qwen/GPT-2/stat-prompt 那段)。**

---

## 同学 A — 归一化 + ICA 嵌入维度

- 上周两个问题之一(归一化/反归一化):验证了"画图时反不反归一化结果一样"→ **不是归一化的问题**,是模型本身。很多数据均值本就在 0 附近。
- ICA-FAST(SDForger 那套)嵌入维度:之前用 6 维(试过 3、6)。**这次把 Chest 的 ICA 维度从 6 提到 12 → 准确率大幅提升**;但再往上(18)不再变好 → **12 是当前最优维度**。
- 目前只对 **Chest 陀螺仪**;Hand 是 加速度+陀螺仪 6 通道一起。多通道是**一个网络同时生成 xyz**。
- 导师反馈:① 画图**一个 sample 一张图**(别全叠在一起,看不出东西),好/坏样本分开展示;② 生成的频率似乎比真实数据略低(真实是高频);③ 有时间**试更复杂的 activity**(用最优参数),做 ablation;④ 旧实验结果(好的坏的)都放进 report。

## 同学 B — autoencoder / 生成 coverage / 下游分类

- 在查"为什么生成能抓到 pattern 但不够好"。Step1:自训 autoencoder 编码原始数据有偏差(但可能不是主因);对照 SDForger 的 encoder。Step2:验证 LLM 生成能力 → **生成 coverage 太低**(只覆盖部分原始数据),换 SDForger encoder-decoder 更差。
- 拿生成数据训分类器、真实数据测试(train 5 人 / test 2 人;无人里随机抽 400 条)→ 生成质量不够好但有点意义。
- 导师反馈(重要):把"纯生成训练"拆成两步——① 纯生成数据训 + 真实测(现状);② **生成数据 pre-train + 少量(10–20%/50–100 条)真实数据 fine-tune** → 对应"真实数据稀缺"的真实应用;若能接近全真实数据的结果,就证明生成数据对下游有 contribution。生成数据有 gap 可理解(信息有损失),尽量缩小即可。
- 导师另一建议:**试着去掉 label 信息**,只用从数据本身提取的统计信息生成。若无 label 也能达到接近结果 → 模型可用**大量 unlabeled data** 训练 → 更 generalized(给 running 数据+其统计但不给 label,也能生成 running)。

---

## 同学 C(你)— Qwen vs GPT-2 + stat-prompt + multi-subject

**你汇报的内容**:
- 把上次几个最优设置叠加 + 改 prompt 加统计数据 → 确实好一些。
- 但发现:原论文用 **GPT-2**,加不加 prompt 约束**一样**,还是吐很多离谱大值。
- **前天换 Qwen** → 不管加不加统计 prompt(哪怕只加 max),基本都能**遵守数据范围、不爆炸** → 之前那些优化(修复)显得没太大意义,直接换模型就行。加不加 label 数据差别不大。
- **但 multi-subject 后生成数量变少**:网络输出有"格式检查",输出不完全符合格式就提取不出数字、被丢掉。单人没事,多人变怪,还要看。
- 本周主结论:**Qwen(1.5B)比 GPT-2 好挺多,不需要对输出做筛选/修复**。

**导师对你的反馈(= 你的 action items)**:
1. **保留 GPT-2 结果做对比**:gpt2→Qwen 的 performance 变化放进 report,"不同 task 哪个模型更好"是有价值的结论。GPT-2 太老、能力本就差很多。
2. **查 multi-subject 生成数量减半的问题**:你基本上把有效 example 砍半了,必须搞清。导师猜测:不同 subject 的动作频率/周期不同(有人 1 秒一个 cycle,有人 2–3 秒)→ 数据多样性变复杂 → 模型"一刀切"更 stable → 有效数据变少。
3. **在 Qwen 模型上加 normalization**:之前做的 normalization/processing,现在加到 Qwen 上看效果(光看指标 0.27→0.18 很难知道在真实信号上意味着什么)。
4. **做结构分析 / 改画图**:别全叠在一起;**一个 sample 一张图**,不同轴可叠(YZ),自己写个 plot function。
5. **先用 Qwen 跑出一组有效 baseline**:有了有效 baseline 再看问题在哪、怎么改。Qwen 既然更好就用 Qwen。
6. (可选)下游任务可以变成一个 optimization loss 试试(不确定效果)。

> 你的数据:**手部加速度计(hand accelerometer)**。

---

## 通用 / Report 安排

- **Final report:7 月 19 日**。导师希望 **7 月 10 日左右给一版 draft** → 他周末看 → 给反馈 → 你们还有一周调整。
- **格式不限**(不强制 IEEE/LaTeX/Word),但要有这些模块:Introduction(research question)→ Literature review → Methods → **Experiments(试过的模型、调过的参数、各结果)** → **Result analysis(什么 setup 好/为什么,什么不好/为什么)** → Conclusion/Future work。
- 三人把各自部分**融合成一份连贯的 report**。

## 下次会议

**7 月 29 日(下周一)**(7/6 暑假冲突,故顺延到下周一)。

---

## 我们这次实验 vs 导师反馈(对照)

| 导师 action | 状态 | 结果 |
|---|---|---|
| 保留 gpt2 做对比 | ✅ | 完整 gpt2 vs Qwen 表(可控性/爆炸/TSG),整理进 report |
| 查 multi 数量减半 | ✅ **已解决(修正)** | 加埋点重跑 **4 seed 全部 accept=0.75、零质量拒绝、配额填满**;最早的 0.40 是**一次性不可复现**的坏 draw → **不是系统性减半** |
| **在 Qwen 上加 normalization** | ✅ **已做** | activity_series norm 让 Qwen ACD 全面小幅降(walking heldout 1.20→0.96),不爆炸、有正面效果 |
| 一个 sample 一张图 | ✅ **已做** | `scripts/plot_samples.py`,好/坏样本分开,`outputs/sample-plots-20260622/` |
| 用 Qwen 跑有效 baseline | ✅ | Qwen 单/多 subject TSG 定为正式 baseline |
