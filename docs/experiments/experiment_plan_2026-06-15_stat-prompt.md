# Experiment Design Plan

Date: 2026-06-15
Project: Time-Series LLM Synthetic Sensor Data (SDForger + PAMAP2/HAR)
Owner: agent + user
Mode: diagnostic + single controlled comparison (one independent variable)
Target claim: CLM-002（生成样本保留有用统计/时间结构）的方法侧扩展 — 新证据 EVD-022。
Status: READY_TO_RUN

## 1. Research Question

在 unified label-conditioned SDForger 上，把**每窗口统计量（mean/std/min/max，量化成 bin）加进 prompt 的 `Condition:` 段**，能否从**源头**减少 LLM 生成越界 latent（value explosion），从而**降低甚至免除**下游复杂修复（T1 的 clip/shrink）的必要性？

可证伪化版本：在固定 GPT-2、固定 FICA 预处理、固定训练预算下，label+stat-prompt 相比 label-only(raw-unified)，是否让 **未修复(raw)** 的 decoded abs-max 显著变小、walking ACD 显著更接近 baseline，且模型**确实遵守**所请求的统计量？

## 2. Hypotheses

- **Primary (H1)**：stat-prompt 给了模型幅度先验 → stat-prompt-**raw**（不修复）的 decoded abs-max 远小于 raw-unified-raw，且 walking ACD 明显低于 raw-unified（向 baseline 0.165 靠近）；stat-prompt-raw 与 stat-prompt+轻修复之间的差距很小。
- **Alternative explanations**：
  - A1：GPT-2 对序列化数字条件遵守太弱 → stat token 被忽略，stat-prompt-raw ≈ raw-unified（无改善）。
  - A2：改善并非来自"幅度先验"，而是 bin 条件**限制了输出多样性**（趋向训练样本）造成的 mode collapse。
  - A3：改善来自更长 prompt 改变了训练动态（过拟合 31 窗口），而非条件机制本身。
- **Expected direction**：stat-prompt-raw abs-max ≪ raw-unified；stat-prompt-raw ACD < raw-unified ACD；统计量遵守度（请求 bin vs 实测）相关性 > 0。
- **Falsification condition**：若 stat-prompt-raw 的 abs-max / ACD 与 raw-unified 无显著差异，**或** 统计量遵守相关性 ≈ 0（模型忽略 stat）→ H1 证伪，结论为"prompt-only 不足，仍需修复"。

## 3. Evidence Standard

- **Intended use**：Internal direction check（决定 prompt 路线是否值得继续、是否能替代复杂约束）。不是 paper-claim 级别。
- **Required baselines**：① unconditioned baseline（已存，canonical）② raw-unified label-only，raw + per-activity clip/shrink（已存，T1）。
- **Required datasets/tasks**：subject101、walking/running、`hand_acc16_x`（与 T1 同设置，保证可比）。窄设置 → 窄 claim。
- **Required repeats/seeds**：smoke 阶段 ≥1 seed（GPT-2 SFT 固定 random seed；若 H1 成立再补第 2 seed 看稳定性）。
- **Required ablations**：bin 粒度 K（4 vs 8）做敏感性——**仅在 R2 初步成立后**（staged，不预先 sweep）。
- **Required figures/tables**：统一 TSG 表（同 T1 evaluator）；stat-adherence 散点/相关；raw vs 轻修复对比；abs-max sanity。

## 4. Experimental Unit

- **Dataset/split**：PAMAP2 subject101，walking(act4)/running(act5)，`hand_acc16_x`；per-activity parquet（与 T1 同）。train_length 5000、window 300、min_windows 30、minimize-overlap。
- **Preprocessing**：各动作独立 StandardScaler → 合并 → joint FICA（dim auto→5，random_state=0）。与 T1/canonical 完全一致。
- **Model/method**：GPT-2 文本编码 SFT（fim_template_textual_encoding）。改动：`Condition:` 段除 `data=walking/running` 外，复合加入量化统计 bin。
- **Baselines**：unconditioned baseline、raw-unified label-only（±per-activity clip/shrink）。
- **Metrics**：见 §5。
- **Compute**：1 次 GPU SFT smoke（gpt2，Puhti `gpu` v100，约几十分钟）+ CPU 后处理评估。
- **Environment**：Puhti `/projappl/.../envs/sdforger-py312`，`module load pytorch/2.6`，code root `fms-dgt`。

## 5. Variables and Controls

| Type | Variable | Value(s) | Notes |
|---|---|---|---|
| Independent | `Condition:` prompt 内容 | label-only **vs** label+stat-bins | **唯一**训练侧改变量 |
| Independent (post-hoc, 不增训练) | 修复强度 | raw（不修） / 轻 per-activity shrink | 同一份生成输出上评两次 |
| Controlled | FICA 预处理 / basis | 同 T1（random_state=0, dim 5） | |
| Controlled | base model / 训练预算 | gpt2 / 同 epochs、temperature | |
| Controlled | 生成数量目标、parser、eval 协议 | 同 T1（scaled space, subset-mode first） | |
| Controlled | parquet / 通道 / seed | 同 T1，固定 seed | |
| Nuisance | bin 粒度 K | 默认 8（quartile-ish） | 敏感性留 R3 |
| Nuisance | 多样性塌缩 | duplicate / NN 相似度 | 检验 A2 |
| Nuisance | 生成时统计量采样源 | 从该动作 train bin 采样 | 非 per-target，避免泄漏；需说明 |
| Nuisance | malformed 文本率 | prompt 变长可能升高 | 记录 parsed/malformed |

## 6. Run Matrix

| Run ID | Purpose | Change | Fixed controls | Dataset/split | Metric | Seeds | Expected | Output |
|---|---|---|---|---|---|---|---|---|
| R0 | baseline 参考 | 无条件 | — | s101 walk/run | TSG | 既有 | ACD 0.165/0.498 | canonical（已存） |
| R1 | raw-unified 参考 | label-only | — | 同上 | TSG+abs-max | 既有 | walking ACD 5.97(raw)/0.81(per-act clip) | T1（已存） |
| **R2** | **stat-prompt SFT** | **+stat bins** | 同 R1 | 同上 | TSG, stat-adherence, controllability, counts | 1 | abs-max ≪ R1-raw；ACD↓ | **新 GPU run** |
| R2a | R2 不修复评估 | raw | — | 同上 | TSG+abs-max | — | 若 H1 成立则接近 baseline | 同上 |
| R2b | R2 轻修复评估 | +per-act shrink | — | 同上 | TSG | — | 与 R2a 差距小 | 同上 |
| R3（条件触发） | bin 粒度敏感性 | K=4 vs 8 | 同 R2 | 同上 | TSG | 1 | — | 仅 R2 成立后 |

## 7. Logging Requirements

git commit、命令、task yaml/config、seed、base model(gpt2)、epochs、temperature、K(bin 数)、parquet 版本、env(sdforger-py312, pytorch/2.6)、Slurm job id、输出 dir、generation_stats(raw/parsed/accepted/malformed)、所有指标 CSV/JSON、生成时 stat 采样的具体 bin 组合表。改远程代码前留 `generate.py.pre_stat_prompt_20260615` 备份。

## 8. Stop Conditions and Decision Rule

- **Support（H1 成立）**：R2a(raw) 的 decoded abs-max 落回合理范围（无爆炸）**且** walking ACD ≤ per-activity clip(0.81) **且** stat-adherence 相关 > 0.3 → 结论"prompt 条件化能实质减少修复需求"；T2 成为头牌方法，修复退化为可选轻量步骤。
- **Falsify**：R2a ≈ raw-unified（仍爆炸）**或** stat-adherence ≈ 0 → GPT-2 忽略 stat；诚实记录"prompt-only 不足"，保留 T1 修复，并把"更强条件化（class-embedding / CFG）"列为后续。
- **Continue**：R2a 优于 raw-unified 但仍需修复，且 R2b 明显更好 → 报告 "stat-prompt + 轻修复" 组合方法；考虑 R3 或更强条件化。
- **Stop & report**：无论 support/falsify，结果都可写进下周汇报与报告（负结果也对口课程 prompt-design topic）。
- **Compute ceiling**：R2 先跑 1 次；R3/第二 seed 仅在 R2 成立后。

## 9. Reviewer Risk Check

| Risk | Why it matters | Mitigation |
|---|---|---|
| stat 条件只是限制多样性(A2) | 改善可能是 mode collapse 假象 | 报告 duplicate/NN 多样性 + label controllability |
| GPT-2 太弱、忽略数字条件(A1) | H1 可能直接失败 | stat-adherence 指标直接检验；负结果诚实报告 |
| 生成时采样 train stat = 泄漏 | 评审会质疑 | 说明是 per-activity 先验、非 per-target；可加"随机 bin"对照 |
| 单 subject/通道/2 动作 | 过窄 | 明确只作 internal direction check，claim 收窄 |
| parser 兼容复合条件 token | 改坏会全崩 | 先本地 round-trip 解析测试再上 GPU |

## 10. Next Execution Step

交给 `run-experiment`：

1. 本地改 `puhti-generated/code_patch/.../generate.py`（+trainer/utils 文本编码）：训练侧 stat-bin 注入 + 生成侧 per-activity stat 采样；**本地 prompt round-trip 解析单测**。
2. 远程留备份 → rsync → 写新 task yaml `pamap2_subject101_unified_stat_prompt_*` → `sbatch slurm/run_sdforger_pamap2_unified_stat_prompt_gpu.sh`（基于 `run_sdforger_pamap2_unified_label_conditioned_gpu.sh`）。
3. 生成后用 `apply_smooth_latent_repair.py`（raw + per-activity shrink）+ `rerun_smooth_repair_tsgbench_table.py` 评 R2a/R2b；新增 stat-adherence 评估脚本。
4. **提交 sbatch 前把 diff + 命令贴给用户确认（GPU job，遵守"操作前检查"）。**
