# 项目总结 + Workshop 可行性评估(2026-06-28)

> 目的:把三人工作整合,诚实评估"够不够 workshop(IJCAI 2026)",并指出最短补强路径。
> ⚠️ **框架更正(2026-06-28)**:本项目是**在 SDForger 基础上改进/扩展**(SDForger-centric),**不是拿 SDForger 去和 TimeVAE 等外部生成模型比 benchmark**。因此:正确的对照 = 原版 SDForger vs 改进版(已有,消融叠加);我们跑的高斯/jitter 是 **SDForger 内部消融**(LLM 换高斯),属框架内、是对的。**外部 baseline(TimeVAE)不是该补的实验**,下文第五节据此更正。
>
> 一句话结论:**作为课程 report 完全够;作为 workshop"边缘偏可投"——SDForger 框架下故事自洽(改进 + 可控性边界刻画 + 内部消融),主要短板是 scope 偏窄(单通道)。框架内最该补的是「多通道生成」(SDForger 招牌 + HAR 真实场景),不是外部 baseline。**

---

## 一、项目目标

LLM 生成合成传感器时序用于 HAR(SDForger 适配 PAMAP2),课程主题 = **prompt 设计**。交付:report(~7/19)+ workshop poster/oral(8 月)。

## 二、三人工作整合

### 同学 A — 表示/嵌入(ICA 维度 + 归一化)
- 反归一化不是问题(模型本身的事)。
- **ICA-FAST 嵌入维度是关键杠杆**:Chest 从 6 维提到 **12 维,下游准确率大幅提升**(18 不再更好 → 12 是甜点)。
- 目前 Chest 陀螺仪为主;多通道(hand 6ch)是一个网络联合生成。

### 同学 B — 下游 utility / 生成覆盖
- 自训 autoencoder 编码有偏;LLM 生成 **coverage 偏低**(多样性不足)。
- 分类器:生成数据训 + 真实测(train 5 人 / test 2 人),10 类,gap ~10–15%。
- 方向:**pre-train(合成)+ fine-tune(少量真实)**;去 label 用统计量(unlabeled 泛化)。

### 同学 C(你)— 模型 × prompt 设计
- **gpt2 爆炸 + 无视 prompt 数值**(能力上限,3seed+CI 锁定);**换 Qwen2.5-1.5B → 不爆炸 + 部分可控**。
- **最优配置落到 Qwen**(multi + per-act clip):train walk/run 0.73/0.89、held-out 0.98/1.46、3-act +cycling 1.55;**跨人泛化 ≥ gpt2、源头不爆炸**。
- **prompt 可控性边界(本周核心)**:**幅度(max)可控(0.4–0.5)、频率(period)不可控(~0)**;双层机制 = 表示层(频率在固定 FICA 基里,控不了)× 训练层(过训练变死板,~50 epoch 甜点)。
- 诚实 baseline:无条件生成上 **LLM 被 trivial 高斯/jitter 追平** → LLM 价值不在 raw 保真度,而在(有限的)可控性。

## 三、整体的硬结论(可写进论文)

1. **模型容量决定底线**:小模型(gpt2)爆炸且无法数值条件化;1.5B 强模型源头稳定。
2. **prompt 可控性有明确边界**:能控幅度、不能控频率;受表示(基固化)+ 训练(过拟合)双重限制。
3. **诚实定位**:LLM 在无条件保真度/下游 utility 上不优于 trivial baseline(高斯/jitter);其价值在可控、非拷贝生成(隐私/数据共享)。
4. **表示与维度重要**(同学 A):ICA 维度显著影响下游。

## 四、Workshop 可行性 — 诚实评估

### ✅ 优势(workshop 级)
- **方法严谨**:多 seed + CI、对抗审计、steelman baseline、预注册证伪。
- **诚实的负/边界结果**:在 ML workshop 越来越受欢迎(boundary/negative studies)。
- **对口一个真问题**:"prompt 到底能控时序生成的什么"——清晰、可复现。

### ⚠️ 短板(为什么"边缘")
1. **scope 偏窄**:单数据集(PAMAP2)、主要单通道(hand_acc16_x)、2–3 个动作。workshop 审稿人会问泛化性。
2. **缺真正的生成式 baseline**:只对过 trivial(高斯/jitter),**没对 TimeVAE/TimeGAN/DDPM**——没有这个,"LLM 表现如何"缺参照系。
3. **"正面卖点"弱**:LLM 不胜 trivial baseline;可控性中等(0.4–0.5)且有边界。故事更像"严谨的边界研究"而非"我们做出了更好的东西"。
4. **三人工作较散**:ICA 维度 / autoencoder-coverage / prompt 设计三条线,**缺一个统一叙事**。
5. **部分结果 provisional**:需 needs-verification 清理后才进 claim。

### 裁决
- **课程 report**:✅ 远超够格(严谨、完整、诚实)。
- **Workshop**:🟡 **边缘可投**。当前体量更像一篇扎实的 workshop short paper / poster,**前提是把叙事收敛成"边界研究"**;若要 full paper,需补 baseline + scope。

## 五、最短补强路径(SDForger 框架内,已更正)

> 更正:**TimeVAE 等外部 baseline 不是该补的**(本项目是改进 SDForger,不是和外部模型比)。框架内的对照(原版 SDForger + 高斯/jitter 内部消融)已具备。

| 优先级 | 补什么 | 为什么 | 成本 |
|---|---|---|---|
| **1(叙事)** | 统一成一条主线:"我们如何改进 SDForger 适配 HAR + 它的 prompt 能控什么"。把三人工作挂上去 | 解决"散"+给清晰贡献 | 写作,0 计算 |
| **2(框架内最关键实验)** | **多通道生成**(SDForger 招牌:一个模型联合生成 hand 6 轴) | SDForger 核心卖点 + HAR 真实场景 + 直接拓宽 scope(治最大短板) | 中-高(需处理 embedding_dim:auto bug) |
| **3** | 把可控性边界在 multi-subject 上验一遍 + 下游 pre-train+fine-tune(同学 B)做实 | 核心贡献加固 + 应用价值 | 中 |
| 4 | 第二数据集(UCI-HAR)做一点泛化点缀 | 进一步回应泛化(full paper 才需) | 高 |

## 六、推荐的 workshop 叙事(最可投)

> **标题方向**:*"What Can Prompts Actually Control in LLM-Based Time-Series Generation? A Boundary Study on HAR"*
> **主线**:LLM 做时序生成,prompt 能控什么、控不了什么、为什么(表示×训练双层)+ 诚实 baseline(被 trivial 追平 → 价值在可控非拷贝,不在保真度)+ 模型容量的作用。
> 这把三人工作(表示/维度、下游 utility、prompt 可控性)统一成"理解 LLM 时序生成的能力与边界",是一个**诚实、清晰、可复现**的 workshop 贡献——不假装"更好",而是"搞清楚了边界",这类研究 workshop 接受度在上升。

**底线建议**:别再堆零散实验了;**先把叙事收敛(优先级 1)+ 补一个 TimeVAE baseline(优先级 2)**,这两步能把"边缘可投"推到"稳投 poster/short paper"。
