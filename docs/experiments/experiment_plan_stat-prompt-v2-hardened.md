# Experiment Plan v2 — Stat-Prompt (hardened after adversarial review)

日期：2026-06-17
取代：`experiment_plan_2026-06-15_stat-prompt.md`（v1 有方法缺陷）
状态：READY_TO_RUN（分阶段，先零成本探针）

## 为什么要 v2

v1 的 T2 是负结果,但 4 个独立审稿发现 v1 的**分析与设计本身有缺陷**,负结果不能直接升级成"GPT-2 学不会数字条件(H3)":

1. **指标坏**:adherence 用 Pearson 相关"档位 vs 原始实测 max",而实测 max 含 5341 这类爆炸值 → 被少数爆炸主导。**已修正复核**:改 Spearman(秩,对爆炸免疫)后,max Spearman=0.010、CI[-0.15,0.17] → 负结果成立,但 CI 只排除强效应(>0.3),排除不了弱效应。
2. **选择偏倚**:在 `|max|<6` 子集上测 adherence = 在被相关的那个量上做选择,会优先删掉"成功的高档样本",把相关性人为压向 0。
3. **请求未平衡**:生成时 `random.choice` 采复合 token → 档位分布随机、对比变量没铺开 → "无效应"和"没测出来"分不清。
4. **目标是间接的**:模型直接产出 5 个 latent 系数,窗口 max 是它们经 decode 后的**非线性函数**;爆炸本质是 **latent 越界**。约束窗口 max ≠ 约束模型能直接控制的量。
5. **没有 ceiling / 阳性对照不匹配**:没有"上界 run",无法区分"模型学不会"和"这个目标本身不可达";label 对照是 2 类 nominal + 分类器度量,和 ordinal stat + 矩恢复度量不匹配,不能用 label/stat 落差直接论证 H3。
6. **1 seed**:无法区分"不可学"和"这个 seed 没学到";决策阈值 rho>0.3 在 n~73/1seed 下正好在噪声边缘。
7. **决策规则把 adherence(可控性) 和 explosion(有效性) 用 AND/OR 混在一起** → 把析因要分开的两件事又合并了。

## 修正后的设计原则

- **目标改为 latent 量**:条件化 per-window 的 **latent max-abs 系数 / L2 范数**(模型直接吐出、且正是支配爆炸的量)。窗口统计量作为 secondary 附带报告。
- **一次只变一个量**:固定 stat,只变 bin 粒度。
- **平衡请求**:每个 bin 请求等量窗口(controlled prompt sweep),不再 random.choice。
- **指标对爆炸稳健**:主指标 = 秩相关 Spearman + 高档 vs 低档 Mann-Whitney + bootstrap 95% CI,在**全部窗口**上算;explosion rate 作为**正交的第二轴**单独报(不当 adherence 的筛子)。
- **正交 2×2 结果表**:obeys/ignores × safe/explodes,四格各有解释。
- **多样性 gating**:报 duplicate / 最近邻距离,温度降低带来的"少爆炸"必须排除 mode collapse。

## 分阶段执行(最便宜、最决定性的先做)

### Stage A — 零/低成本探针(已全部完成 2026-06-21)
- **A1 ✅**:用秩相关复核现有 T2 → 负结果成立(max Spearman 0.01, CI[-0.15,0.17]),但只排除强效应、run 有 confound。
- **A2 ✅**:oracle 可识别性。Spearman(模型直接产出的 latent maxabs/L2, 解码窗口 max)= **0.78–0.80**(running 0.88–0.90);训练数据里 latent 量 vs max-bin 也 0.82–0.87。→ **窗口 max 目标可达,decode 路径单调**;失败不是"目标不可达",而是模型没按档位调 latent。**且 latent L2 是干净的、可直接控制的目标(同时支配窗口幅度与爆炸)。**
- **A3 ✅(但探针无效)**:temp=0 贪婪。**所有 46 个不同 composite(连 walking/running 都不分)坍缩成完全相同的一个窗口**(全局 1 个取值,realized_max 恒 0.442)。→ **贪婪连已知有效的 label 条件也抹平**;条件在本 pipeline 里靠"移动采样分布"起作用,不靠移动 mode。**因此 temp=0 不是有效的 adherence 探针,也不能作为随机性的一极。**

### Stage A 结论
- H2(太随机)**不是主因**:A3 表明去掉随机性后连 label 都没差异 → 条件本就是分布级的。
- A2 排除"目标不可达"。
- 因此剩下的活假设是 **H1(编码太稀疏/弱,学不动)** vs **H3(GPT-2 学不会数值条件)**,二者只能靠 Stage B 的 **synthetic ×1/×3 最大可分对照** 分离;随机性轴若要测,应在**仍采样的温度间**比(如 1.3 vs 0.7),不能用 0。

### Stage B — 决定性 GPU runs(仅当 A 支持)
- **B1 synthetic ×1/×3 阳性对照**:把窗口人为乘 1 倍 vs 3 倍,作为二值 token——最大可分对比。学得会→证明数字对比**可学**,之前失败是 SNR/稀疏(H1/H2);学不会→H3 强证据。
- **B2 coarse latent-target,平衡请求,temp∈{0,1.3},≥3 seeds(决定性 cell = coarse@temp0)**:条件化 latent L2-范数 2 档,每档等量请求。
- 主指标:高档 vs 低档实测量的单侧 Mann-Whitney + bootstrap CI;判定 = CI 是否排除预注册效应。

## 决策规则(正交)

| | safe(不爆炸) | explodes(爆炸) |
|---|---|---|
| **obeys**(实测随档单调,CI 排除 0) | 理想:prompt 路线可行 | 听话但仍越界 → 配轻修复 |
| **ignores**(CI 紧贴 0) | 没用但也不乱 | 当前 T2 所在格 |

- B1 学得会 + B2 obeys → prompt 路线值得继续(之前败于编码/随机)。
- B1 学得会但 B2(latent target)仍 ignores → 目标/编码问题,换 latent 直接约束。
- B1 也学不会 + label 仍可控 → **H3 强结论**,放弃 prompt 路线,转架构级条件化(class/stat embedding、CFG、constrained decoding、learned latent prior)。

## 严谨性硬要求

- 决定性 cell ≥3 seeds;所有 adherence 报 bootstrap 95% CI;null 表述为"CI 排除 rho>0.3",不是"rho≈0"。
- 锁定分析脚本后再看实测(避免 garden-of-forks);每 cell 预注册**一个**主 endpoint。
- 同 FICA basis、同评估协议,与 T1/canonical 可比。
