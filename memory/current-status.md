# 当前状态

最后更新：2026-06-25

## 2026-06-25 最优配置落到 Qwen（改进上周进度,`docs/experiments/qwen-best-config-results-20260625.md`)

上周"最优 ACD 0.74"是 gpt2;导师要 Qwen。本次把"multi+per-act 修复"做到 Qwen 并与 gpt2 公平对比(都修复)。**Qwen 3-act+clip:walk 1.02/run 1.29/cycling 1.55(MDD 0.18/0.22/0.19)**;对照 gpt2 3-act+clip 0.95/0.94/2.0、Qwen 无修复 2.14/2.25/2.23。→ **修复对 Qwen 也有效(2.2→1.0-1.55)**;公平对比下 walk/run gpt2 略优、**cycling Qwen 明显更好**;**关键:gpt2 必须修复(raw 爆炸),Qwen 不修也合法(raw 不爆)**。最优配置现落在 Qwen,回应导师。**2-act+clip:Qwen walk 0.728/run 0.893 vs gpt2 0.74/0.74 → 走路打平(Qwen 略好)、跑步 gpt2 略低**。完整口径:Qwen+修复=gpt2 同级质量、cycling 更好、源头不爆炸(修复是锦上添花非必需)。**held-out 106/108:Qwen 2-act clip walk 0.98/run 1.46 vs gpt2 1.40/1.74 → 跑步跨人 Qwen 退化更轻(0.89→1.46 vs gpt2 0.74→1.74),泛化≥gpt2**。Qwen 最优配置 train+held-out 完整。**组合 normalization(Qwen+norm+clip)= 负结果/trade-off**:train-跑步 0.89→0.68 但 held-out-走路 0.98→1.39,净不划算 → **不加 norm,最优保持 Qwen+per-activity clip**(held-out 泛化更好)。`run_unified_label_conditioning_normalization.py` 加了 `--repair clip`(在 norm 基里 clip)。

## 2026-06-25 Pattern 描述符可控性（自主循环 R1–R3,`docs/experiments/descriptor-control-results-20260625.md`)

围绕用户直觉"单 max 太薄、应给 pattern 描述(频率/幅度)"做的自主实验(Qwen 单 subject 101,`run_stat_prompt_hf.py` 已泛化 `--stat max/std/period/range`)。

- **R1**:max adherence **0.467** ≫ std **0.089** ≫ period **-0.042** —— **与假设相反**,单标量 max 最可控,频率完全控不了。
- **R2 机制**:真实数据 period 动作内近乎常数(walking≈60/running≈75,仅 4-6 个离散值),且实测不跟请求 → **频率是动作签名、由 label 隐含、无类内连续谱可控**。
- **R3**:adherence 跟描述符**有效档位数**走(max 16 档→可控,period 4 档→不可控)。
- **R4/R5(2-seed 稳健)总表**:max **0.399±0.067** ≫ std 0.176 > range 0.115 ≫ period **-0.017±0.025**。连幅度家族内部 max 也远比 std/range 可控。
- **最终机制结论(硬)**:**LLM 控的是 FICA 系数;可控性 = 描述符与"系数幅度"对得多直接** —— max 直接 ∝ 系数幅度(可控 0.40);std/range 依赖系数形状(弱);**频率在 FICA 基里、不在系数里 → 不可控(~0)**。"prompt 更丰富"不提升控制;决定因素是描述符落在"可调系数(幅度)"还是"固定基(频率)"上。用户直觉(频率是 pattern 本质)对,但它恰落在不可调那一半 = SDForger+FICA 的结构性边界。要控频率需改**表示/架构**(让基可条件),非 prompt 能解。
- **R6 epoch 消融(验用户"过训练变死板"假设)**:max adherence 随 epoch = 20:0.18(欠训练) / **50:0.53(峰值)** / 100:0.40 / 200:0.42(2-seed)→ **存在训练甜点 ~50,过训练(100-200)可控性下降 = 用户假设成立**;period 所有 epoch ~0 = 结构性。**双层结论**:表示层决定"能不能控"(幅度可/频率不可),训练层决定"控得多好"(50 epoch 甜点)。**实用:epoch 100→50 更好且省算力。**
- **50 vs 100 生成质量对比(只是丰富对比,非换主线)**:TSG label-only 单101 2-seed → walk_ACD 50:2.19/100:1.95、run_ACD 50:2.65/100:2.97、MDD 持平 → **50≈100 打平**(walk偏100、run偏50)。综合 50 = 保真度打平 + 可控性更好(0.53>0.40)+ 省一半算力 = **温和净正面、可作更优默认;但差距小,沿用 100 亦可**。
- **⚠️运维教训(重要)**:Qwen run 的 `model/` 各 12GB,本次累积 268GB **撑爆 scratch 配额 → 训练静默失败、生成 0 输出**。已删全部 qwen checkpoint(结果保留,降到 66GB)。**对策:job 末尾 `rm -rf $OUT/model`(`run_qwen_lo_ep.sh` 已加);以后跑 Qwen 务必生成后清 checkpoint。** 用户 2026-06-25 授权了这次删除(经 AskUserQuestion 确认)。
- **本条线收尾**。后续(非本周必须):multi-subject 重复、可参数化基的架构方向。
- 集群提醒:**gpu 分区严重拥堵(排队到数小时)→ 用 gputest(限并发 2,~6min/job,nohup 编排器串起来,服务器端独立运行抗连接中断)**。

## 2026-06-22 对照参考实验补齐（审计缺口：孤立→对比）

总表：`docs/experiments/meeting-table-20260622.md`（汇报用，诚实分级版）。本轮补的都是**对照参考**，把过去的孤立实验变成可解读的对比。

- **下一阶段 Qwen 扩充计划（`docs/experiments/experiment_plan_2026-06-22_qwen-expansion.md`）**:本周优先级 P0 固化 Qwen baseline(3-seed)→ Phase1 扩 activity(3+动作)→ Phase2 **去 label 只用统计(activity-recoverability,最有新意)**→ Phase3 下游 HAR pre-train+fine-tune → stretch(per-subject 归一化修 multi 可控性 / Qwen-3B / 多通道)。证据标准 report-draft 级,7/10 draft 前停新实验转写作。
- **2026-06-22 组会（`docs/meetings/2026-06-22-group-meeting-notes.md`）+ 会后 action 已做**：导师认可 Qwen 方向。给你(第三汇报人)的 action items 全部完成：① **保留 gpt2 对比**(已有表);② **查 multi 减半 → 修正**：加埋点重跑 **4 seed 全部 accept=0.75、零质量拒绝、配额满**,最早 0.40 是**一次性不可复现**坏 draw,**非系统性减半**(`run_stat_prompt_hf.py` 已加 reject_breakdown);③ **Qwen 上加 normalization → 有用**(`run_unified_label_conditioning_normalization.py` 加了 HFLLM/HF 后端;activity_series norm 让 Qwen ACD 全面小幅降,walking heldout 1.20→0.96,不爆炸,`scripts/eval_norm_clip_dual.py` 评估);④ **plot function**(`scripts/plot_samples.py`,一格一样本/好坏分开,`outputs/sample-plots-20260622/`);⑤ Qwen 单/多 subject TSG 定为正式 baseline。**排期:report draft 7/10 给导师、final 7/19、下次组会 7/29。**

- 🔴 **头号发现(定位重构)·高斯系数生成 baseline**（`scripts/gaussian_coeff_baseline.py`）：把 LLM 换成"对 FICA 系数拟合高斯、采样、走**同样的线性解码**"，TSG **持平甚至略优于** SDForger(高斯 walking ACD 1.07/running 0.60、MDD 0.158/0.191,vs LLM 1.04/0.69、0.169/0.206)，且**同样不爆炸**(absmax 3.4/4.0)；HAR real+syn 0.758 也在 ±0.11 噪声内。→ **无条件逐动作生成上,LLM 相对 trivial 高斯系数先验没有可测增益**；保真度功劳在 **FICA 表示 + 修复**,不在 LLM。**项目定位必须改**：不讲"做了个好 HAR 生成器"(被高斯/jitter 打平)，讲"用正确 baseline 钉死 LLM 到底加了什么"。
- **Round 6·条件高斯可控性基线**（`scripts/conditional_gaussian_adherence.py`，最后一颗钉子）：按 max 分箱、每箱拟合高斯采样 → pooled Spearman **0.413**(walk 0.597/run 0.230) **≈ 甚至 > LLM 0.36**。→ **连可控性(LLM 最后的卖点)也被 trivial 条件基线追平**。**三个轴(保真度→高斯、下游→jitter/高斯、可控性→条件高斯)全部被 trivial 基线追平,LLM 无可测优势**。
- **但这磨锋利了研究问题**：trivial 基线都只能"复现见过的分箱条件";**LLM 唯一未被测到的潜在优势 = open-ended/组合式/文本 prompt 条件化(生成训练里没出现的条件)**——条件高斯/VAE 结构上做不到。**下一步唯一该打的:预注册 held-out-condition 实验**,这才对口 prompt-design 且是 LLM 可能真赢的地方。

- **完整 6 指标 + held-out 三方对比（`full-metrics-table-20260622.md`，汇报用）**：① 逐步叠加全 6 指标(MDD/ACD/SD/KD/ED/DTW)，最优=multi+per-act clip train ACD 0.74/0.74、MDD 低于 baseline。② normalization **局部有用但不全**(activity_series_z walking ACD 5.97→1.30，但 multi 后爆炸)。③ **norm+multi+波形clip 验证"爆炸=没clip"**：16%/14% 点越界，clip 后 DTW 13248→309、KD 33.5→0.98 救回；**norm+clip 的 walking held-out ACD 0.484 全场最好**。④ **Qwen 单/多 subject 都不爆炸**(KD 0.2-1，不必 clip)，但 multi 后 ACD 升到 2.3、可控性退化(pooled 0.36→0.10，walking 0.42 仍可/running 失败)。⑤ **running 在所有方法 held-out 都最难(1.74-2.71)=共性 subject-drift**。Qwen 训练是 unified(两动作一个模型)。脚本：`within_subject_floor/gaussian_coeff_baseline/conditional_gaussian_adherence/eval_norm_clip_dual/clip_waveform_dual_eval.py`。
- **gpt2 stat-prompt（jobs 35210158/35210159，已核实后端）**：retry **是 vLLM 后端**（job 日志证实 V0 engine v0.8.3，非 HF），仍 0/256 可解析（数值离谱如 -94196754、格式错乱）。但 2026-06-17 另一 composite 的 vLLM gpt2 曾出 **73/74 可解析**（只是 adherence≈0）。→ **准确说法**：gpt2 可解析输出**对 composite 格式很脆**，且**任何设置都达不到数值 adherence**（不是"连合法输出都给不出"）。"需强模型"只 Qwen 单边支撑；唯一 backend-matched 的 ×1/×3 显示 **gpt2 0.94 ≈ Qwen 0.96（都不遵守）→ 数值可控性不随模型变强自动出现**，gpt2 实值在公平条件下未测过。
- **两个 real-vs-real TSG 地板**：跨人(106/108 vs 101/102/105) walking ACD 1.22/running 1.02；**同-subject(train 切两半,5 split,`scripts/within_subject_floor.py`) walking ACD 2.49±1.47/running 0.93±0.38、MDD 0.284/0.290**。组合最优 synthetic(同 subject) ACD 1.04/0.69、MDD 0.169/0.206 **低于两个地板** → **不是"接近真实",而是 under-dispersion**(FICA 线性解码生成太规整、缺真实变异,把主导节奏匹配过好);指标 n~27 噪声极大(±1.47),不宜精细排序。比"在合理范围"诚实,比"记忆化"更准(NN 已排除整窗拷贝)。
- **naive jitter 基线（trivial 生成器,已 steelman）**:初看 σ=0.2 时 real+jitter 0.697 输给 SDForger 0.745,但**扫 σ∈{.05,.1,.2}×2seed 取最好 → real+jitter 0.742、jitter-only 0.742,与 SDForger(0.745/0.715)打平甚至更高**。→ **诚实更正:下游 HAR 上 SDForger ≈ 调好的 jitter ≈ real-only,不比 trivial 增强涨点**。SDForger 卖点改为"可控非拷贝生成器"(隐私/prompt 条件化),不是下游精度。教训:naive 基线必须 steelman,否则会假阳性。
- **记忆化 NN 检查**:synth→train 中位 9.26(walk)/10.33(run) vs real-heldout→train 12.64/11.85,比值 0.73/0.87。→ **不是抄训练数据**(距离 ~9-10 远非 ~0),synthetic 合理地像其训练 subject。
- **重要更正**:2026-06-21 的单 split `real+synthetic 0.758`（下方 #1 那条）已被 **5-seed 复核更正为 0.745±0.013 = 中性**（不显著超 real-only 0.742）。诚实表述:**赢 naive jitter=正面;超 real-only=未证到**(2 类探针偏粗)。
- **Qwen values 3-seed 已核实存在**：pooled ρ seed7=0.195/seed42=0.505/seed123=0.383 → **均值 0.36±0.13**(v2 seed42=0.583 accept 0.75);bins seed42=0.101。"values≫bins"坐实(walking per-label 0.41-0.56 vs 0.0003),但**无 CI、跨 seed 0.20-0.58 波动**=provisional。
- **对抗审计(Workflow 7 claim)+ 集群核实**：审计在 C2(后端 vLLM≠HF)、C3(3 seed 确实在)的具体机制上判错并已纠;但 C1/C4/C5/C6/C7 方法学批评成立、已进表:HAR 探针 n=66→Wilson CI≈±0.11 **排不了序**(±0.013 只是生成 seed 方差,虚高);记忆化分母用跨人 + FICA 子空间 confound→只能说"无整窗拷贝";爆炸归因未隔离容量(后端/epoch 混)。
- 新脚本:`scripts/make_scaled_jsonl.py`、`scripts/nn_memorization_check.py`、`scripts/within_subject_floor.py`、`puhti-generated/slurm/run_gap_fillers_cpu.sh`、`scripts/run_stat_prompt_hf.py`(修空 DataFrame 崩溃 + debug)。
- **最大缺口(审计重排为 HIGH)**:**无任何非平凡生成式对照(TimeVAE/TimeGAN/DDPM)**——下游已与 jitter 持平,"可控非拷贝生成器"定位尚未对过标准生成器,**补此前任何"更好"措辞不成立**。其次:多通道/多变量(embedding_dim:auto bug)、LOSO、去 selection-on-test。

## 2026-06-17 T2 stat-prompt（负结果，H1 证伪）

实验报告：`docs/experiments/stat-prompt-results-20260617.md`。Puhti job `35003110` COMPLETED。

- 在 prompt `Condition:` 段复合加入量化窗口统计量（mean/std/min/max bin），其余同 raw-unified。新脚本 `scripts/run_unified_stat_prompt.py`，集群 round-trip 自测先过。
- **预注册的两条证伪条件都触发**：① R2a(raw) 仍爆炸（running abs_max 5341、DTW 32973）；② **stat-adherence ≈ 0**（请求 bin vs 实测相关 -0.003~0.046，非单调）→ GPT-2 实质忽略数字条件。
- R2b（stat-prompt + per-activity 修复）不优于 T1（label-only + 修复），walking 反而更差（shrink_minmax 0.851 vs T1 的 0.362）。
- **结论**：prompt-only 统计条件化（GPT-2 + 文本编码数字）不能替代 T1 修复。天花板是 GPT-2 无法真正条件化于数值，不是 prompt 内容。
- 下一步候选：learned class/stat embedding 或 CFG、constrained decoding、learned latent prior（治本）；T1 per-activity shrink/clip 仍是当前可用方法；T2 作为有价值负结果写报告（对口课程 prompt-design topic）。
- **2026-06-17 对抗审稿更正**：T2 的 adherence 指标曾用 Pearson（被爆炸值主导）。已用秩相关复核 → 负结果成立（max Spearman 0.01, CI[-0.15,0.17]）但**只能排除强效应、且 run 本身有设计 confound**，故**不能升级为"GPT-2 学不会数字(H3)"**。硬化设计见 `docs/experiments/experiment_plan_stat-prompt-v2-hardened.md`（约束 latent 量、平衡请求、秩指标+CI、ceiling、synthetic ×1/×3 对照、temp=0 探针、≥3 seeds）。诚实表述：当前编码下无可检测遵守，更强编码下尚未定论。
- **2026-06-21 Stage A 探针完成**：A1 秩相关复核（负结果成立但只排除强效应）；A2 oracle（latent 量→窗口 max 单调 ρ≈0.8，目标可达、latent L2 是干净可控目标）；A3 temp=0 贪婪**坍缩成单一输出（连 label 都不分）→ temp=0 探针无效，条件是分布级而非 mode 级**。结论：H2（太随机）非主因、目标可达，剩 H1（编码稀疏）vs H3（学不会数值），需 Stage B 的 synthetic ×1/×3 对照分离。
- **2026-06-21 Stage B 完成 → H3 锁定**（`docs/experiments/synthetic-amplitude-control-results-20260621.md`，jobs 35204673/35204682/35204683）：synthetic ×1/×3 最大可分对照,**3 seed 全部 a1/a0≈1.0（pooled 0.94, p=0.91）→ 连最大可分数值对比都学不会,排除 H1**。最终结论：**GPT-2 这套能条件化于类别、不能条件化于数值（能力上限）**。prompt 数值条件化方向**停止**；要源头控幅度须走架构级（learned latent prior / constrained decoding / hidden-state embedding / CFG）；**T1 per-activity 修复仍是当前可用方法**。对课程：T1 正面 + T2/Stage B 负面边界 = 完整故事，对口 prompt-design topic。
- **2026-06-21 组合最佳配置（导师"取最优合并"方向）**：`docs/experiments/combined-best-config-results-20260621.md`。multi-subject 101/102/105 unified 生成 + per-activity 修复（post-hoc CPU，复用已有生成）。**最佳 = multi + per-activity clip**：train ref 上 walking/running ACD 都 ~0.74（multi 救了 running：单 subject 1.37→0.74）、MDD 0.17/0.21 低于 baseline、DTW 183/155。held-out 106/108：walking 稳（shrink_minmax ACD 0.69），running 退化（1.74，subject 周期漂移）。这是目前最好的 TSG 配置；下一步可补 held-out HAR utility。
- **2026-06-21 #1 held-out HAR utility（正面）**：`docs/experiments/har-utility-heldout-results-20260621.md`。train 101/102/105 + 组合最佳 synthetic，test = 未见 subject 106/108：real-only 0.742 → **real+synthetic 0.758**（F1 0.728→0.746），synthetic-only 0.71。→ synthetic 在未见 subject 上有下游 utility（项目最强应用证据，provisional：单通道/2 动作/1 split）。
- **2026-06-21 #3 三动作扩展**：`docs/experiments/multiactivity-3act-tsgbench-results-20260621.md`（job 35206768，新通用脚本 run_unified_multiactivity.py / repair_and_eval_multiactivity.py）。walking/running/cycling 联合条件生成,clip 修复:walking/running ACD ~0.95（加 cycling 后从 0.74 略升）、cycling ACD ~2.0（最难）、MDD 仍低。→ 方法可扩到多动作,有轻微容量代价。

## 2026-06-15 Smooth latent repair + per-activity bounds (T1)

实验报告：`docs/experiments/smooth-repair-results-20260615.md`。集群已恢复，用户 2026-06-15 授权远程操作（每次写操作前先只读检查）。

- 针对会后导师"更平滑/线性插值修复"反馈，新增 `scripts/apply_smooth_latent_repair.py`：A=latent 全局保形缩放（`shrink_*`）、C=波形域线性插值（`waveform_interp`），与 `clip_p05_p95` 同 basis 同协议对比；重算 clip 与 canonical 表精确吻合（可比性已验证）。
- 关键发现：**最大杠杆是 per-activity latent 边界（不是 clip vs shrink）**。把 walking+running 合并分布换成各动作自己的边界，单 clip 就把 walking ACD 从 2.253 → 0.810；再加 `shrink_minmax` → **0.362**（接近 baseline 0.165），不丢样本。running 用 per-activity clip（ACD 1.370）最好。
- 字面"线性插值"(C) 不是赢家（丢样本、ACD/DTW 更差）。
- 产物：`outputs/smooth-repair-20260615/`（pooled）、`outputs/smooth-repair-peractivity-20260615/`、overlay 图 `clip_vs_shrink_overlay.png`。
- 下一步：T2 stat-prompt（已定位 `generate.py` 改动点）；或在 multi-subject 上重复 per-activity + 补 held-out HAR utility。

## 当前阶段

状态：PAMAP2 / SDForger baseline 与 conditional-generation 诊断阶段。

项目已从离线 reset 进入 controlled smoke experiments。当前已经有 clean baseline、label conditioning、unified conditioning、latent constraint、normalization ablation、multi-subject diagnostic、held-out subject diagnostic 和 generation-time-style validity diagnostic 的 provisional results；主要问题不再是“能不能跑”，而是 unified conditional generation 的 latent validity / decoding stability 还不够可靠。

## 工作主题

使用 LLM-enabled 生成方法，为 HAR 生成合成传感器时间序列数据。

当前恢复出的主线是 `SDForger + PAMAP2/HAR`。`ChatTS` 和 `AgentSense` 暂时作为相邻文献与定位边界，而不是立即实现的主线。

## 信任边界

- 远程集群状态：Puhti 可通过 `ssh puhti` 做只读文件检查；未提交 job。
- 旧笔记中的远程 output paths 和 reports：已观察到部分存在，但研究含义仍为 `needs-verification`。
- 旧实验 result counts：只能作为规划线索，不能作为证据。
- `legacy/old-project-files/puhti-time-series-llm/`：本地观察到的残留代码，但不完整，`needs-verification`。
- 本地 PDF 和提取文本：已观察到的 source-library 输入。

## 已恢复材料

- `legacy/old-project-files/TS LLM.md`：旧工作总结，包含 SDForger/PAMAP2、实验名、provisional result counts、局限和下一步想法。
- `legacy/old-project-files/SESSION_HANDOFF.md`：更早的 handoff，强调先复现和集群环境。
- 三篇 seed papers：SDForger、ChatTS、AgentSense。
- 残留代码 repo：`legacy/old-project-files/puhti-time-series-llm/`。

## 当前判断

Decision：`revise and continue`，confidence：medium。

原因：SDForger 可在 PAMAP2 walking/running 单通道上跑通；activity conditioning 有信号，但 unified setting 存在 value / latent outlier。统一评估后，clean baseline 仍最稳，post-hoc `clip_p05_p95` 可修复部分 outlier。multi-subject smoke 显示单纯增加 subject/data 不能单独修好 raw-unified；held-out subject diagnostic 显示 clip 后 ED/DTW/KD 在 106/108 reference 下仍接近 train-reference 范围。最新 validity 诊断显示 strict reject 太苛刻，soft repair 可作为方法候选，但暂未全面优于简单 clip。

## 下一次 session 入口

1. 先读 `docs/experiments/combined-experiment-table-20260607.md`。
2. 再读 `docs/experiments/multisubject-unified-conditioning-results-20260607.md`、`docs/experiments/unseen-subject-evaluation-results-20260607.md` 和 `docs/experiments/generation-time-validity-results-20260608.md`。
3. 下一步选择：要么把 soft repair 接入实际 generation resampling loop，要么先补 held-out HAR utility；不要再继续堆 normalization 或 strict reject。

## 2026-06-07 Multi-subject unified conditioning smoke

实验报告：`docs/experiments/multisubject-unified-conditioning-results-20260607.md`。

- 已完成 PAMAP2 subject101/102/105 walking/running `hand_acc16_x` unified label-conditioned smoke。原计划的 subject103 因缺少 running 未使用。
- 训练 windows：walking 54、running 62、combined 116；FICA dim 8。
- 生成 accepted windows：walking 51、running 58；malformed latent text 仍存在。
- `multi_raw_unified` 仍失败：walking DTW 9920.940、running DTW 7943.930。
- `multi_clip_p05_p95` 明显修复 ED/DTW/KD：walking DTW 166.065、running DTW 147.235；但 ACD 仍偏高。
- 当前判断：数据量增加不是主要解法；下一步应做 generation-time latent validity / rejection sampling，而不是继续堆 subject。

## 2026-06-07 Held-out subject diagnostic

实验报告：`docs/experiments/unseen-subject-evaluation-results-20260607.md`。

- 已完成 evaluation-only unseen diagnostic：generator train subjects 为 101/102/105；held-out real reference subjects 为 106/108。
- Held-out walking/running 每个 subject/activity 取 5000 rows；真实 windows 为 walking 45、running 44。
- Held-out walking period 估计为 110，running 为 76，说明 subject shift 存在。
- `unseen_raw_unified` 仍失败：walking DTW 11160.400、running DTW 9567.380。
- `unseen_clip_p05_p95` 保持可接受：walking ED 21.496 / DTW 151.102；running ED 20.685 / DTW 146.689。
- 当前判断：clip 后 synthetic 不只是贴合训练 subjects；但这仍不是 held-out HAR utility，下一步应在 generation-time validity control 后补 classifier utility。

## 2026-06-08 Generation-time-style latent validity diagnostic

实验报告：`docs/experiments/generation-time-validity-results-20260608.md`。

- 已测试 `strict_reject_p05_p95` 和 `soft_repair_p05_p95_minmax`。
- strict reject 样本数过少：walking 14/51，running 10/58。
- soft repair 保留 walking 27/51、running 34/58，并记录 clean/repaired/rejected counts。
- soft repair 的 ED/DTW 接近 post-hoc clip，但未全面优于 clip。
- 当前判断：post-hoc `clip_p05_p95` 仍是 simple strong diagnostic baseline；soft repair 是更容易解释的方法候选，但需要接入实际 resampling loop 才能补回样本数。

## 2026-05-18 队友 GitHub 审阅

审阅报告：`docs/reviews/2026-05-18-teammate-review.md`。

- `IkMangMok/mmfit-inference`：当前 GitHub API 返回 `404`，无法审阅，状态为 blocked / needs-clarification。
- `SJYANG555/sdforger-pamap2`：已离线下载 `main` snapshot，reviewed commit `90884c109e48bb7a83b4ba722455a59ffa9983fe`。仓库只有 `main`，没有可见 PR 或 teammate branch，因此无法做 branch-vs-main diff。
- 该 repo 包含 PAMAP2 + SDForger-style pipeline、configs、Slurm scripts、docs 和结果摘要；但 raw data、window tensors、checkpoints 和完整 artifact bundle 未包含。
- 关键 review 判断：当前结果应保持 `provisional`。utility evaluation 似乎使用 `prompt_split: test`，不适合作为 clean train-only HAR augmentation 证据；`MDD` / `SD` 实现也像 local proxy metrics，不能直接当作标准 SDForger/TSGBench 指标。
- 建议：对 `sdforger-pamap2` request changes + rerun smoke；对 `mmfit-inference` ask clarification / request access。

## 2026-05-18 本地 teamate/mmfit-inference 补充审阅

用户已把 `mmfit-inference` 放到 `teamate/mmfit-inference`，已完成本地只读审阅并补充到 `docs/reviews/2026-05-18-teammate-review.md`。

- repo 当前为 clean `main`，reviewed commit `009302f Fix workflow dependencies and classifier test metrics`。
- 该队友完成的是 `MM-Fit latent autoencoder + latent SFT + frozen decoder + generated sample evaluation` 路线，不是直接的 SDForger/PAMAP2 复现路线。
- 已有代码、README、Slurm workflow、autoencoder logs 和 latent SFT logs。
- 关键 review 判断：pipeline 方向有探索价值，但 generated-data evaluation 仍为 `needs-verification`。归档 logs 中 generation 阶段出现 latent length/shape 错误，后续 `synthetic_mmfit.jsonl` 缺失；repo 中也没有可核验的 final `eval_report.json` / generated JSONL / checkpoint artifact manifest。
- 建议：对 `mmfit-inference` request changes + rerun smoke；暂不作为主项目 evidence。

## 2026-05-21 Puhti 旧项目只读盘点

盘点报告：`docs/ops/2026-05-21-puhti-time-series-llm-inventory.md`。

- 远程路径 `/scratch/project_2016517/panh/time-series-llm` 存在。
- 该远程目录和 4 层内未发现 `.git`，因此不能确认 branch、commit 或 dirty state。
- 主要代码和实验工作区是 `/scratch/project_2016517/panh/time-series-llm/fms-dgt`。
- 观察到 PAMAP2 task yaml、Slurm scripts、prepared parquet、outputs、logs、reports、plots 和 model checkpoints。
- 旧 logs 支持“某些 SDForger/PAMAP2 jobs 曾经完成并写出 output”的存在性判断；旧 metrics 和 plots 仍不能直接作为 final claim evidence。

## 下一次 session 入口

1. 从 `docs/ops/2026-05-21-puhti-time-series-llm-inventory.md` 选一个 PAMAP2 run 建 manifest。
2. 核验旧 run 的 task yaml、Slurm log、output dir、final_data.jsonl、checkpoint、report、eval script 是否一一对应。
3. 决定是否迁移旧 SDForger/PAMAP2 pipeline 到一个 Git-controlled clean code repo。

## 2026-05-22 Puhti 旧代码审阅

审阅报告：`docs/reviews/2026-05-22-puhti-old-code-review.md`。

- 旧 SDForger/PAMAP2 pipeline 工程方向基本正确，可作为 recovered baseline candidate。
- 不能直接作为 final evidence：远程目录无 Git provenance，旧 PAMAP2 parquet 不含 `activity_id`，`final_data.jsonl` 保存的是 standardized window space。
- 发现一个实现风险：multivariate `embedding_dim: auto` 会被第一个 channel 的维度覆盖，后续 channel 不会独立 auto-select。
- 当前最有用的下一步：重建 label-aware PAMAP2 preprocessing，并比较 original SDForger prompt vs `+ activity label` prompt。

## 2026-05-22 Baseline Verification Step 1-4

实验记录：`docs/experiments/baseline-verification-20260522.md`。

- 找到 PAMAP2 原始数据：`/scratch/project_2016517/panh/datasets/pamap2/PAMAP2_Dataset/Protocol/subject101.dat`。
- 完成 subject101 的 activity-level 周期性检查：walking、running、cycling；通道为 `hand_acc16_x` 和 `ankle_acc16_x`。
- 本地 artifacts：`outputs/baseline-verification-20260522/`，包括 raw/ACF/PSD 图、summary CSV/JSON/MD、旧 baseline task card / task results / metric snapshot。
- 当前选择：优先使用 walking/running + `hand_acc16_x` 作为 clean activity-specific univariate baseline 的下一步；cycling 暂时作为 secondary，因为 hand/ankle 频率表现不一致。
- 旧 `pamap2_subject101_univariate_paper` baseline 工程上可恢复，生成 85 个 synthetic windows，但它是 mixed non-zero-activity parquet，不是 activity-specific final evidence。

## 2026-05-22 Clean activity-specific univariate baseline

实验报告：`docs/experiments/clean-univariate-baseline-results-20260522.md`。

- 已从原始 PAMAP2 `subject101.dat` 重新过滤出 walking-only 和 running-only 的 `hand_acc16_x` 单变量 parquet。
- walking：activity_id 4，22,253 rows；SDForger 选择 period 58，31 个长度 300 的窗口，FICA dim 4，生成 130 个 synthetic windows。
- running：activity_id 5，21,265 rows；SDForger 选择 period 82，31 个长度 300 的窗口，FICA dim 2，生成 98 个 synthetic windows。
- 评估已完成，metrics 存在本地 `outputs/clean-univariate-baseline-20260522/`。
- 当前判断：这是 clean baseline run，可用于周一汇报说明“baseline 已重建并可跑通”；但生成质量和 HAR utility 仍为 provisional，不能直接当 final claim。

## 2026-05-22 ACF/PSD baseline diagnostic

实验报告：`docs/experiments/acf-psd-comparison-20260522.md`。

- 已完成 walking/running clean baseline 的真实窗口 vs synthetic 窗口 ACF/PSD 对比。
- running：真实 ACF peak lag 81，synthetic peak lag 82；真实和 synthetic PSD peak 都是 1.3333 Hz。结论：running baseline 保留主要周期结构较好。
- walking：真实 ACF peak lag 116，synthetic peak lag 58；真实和 synthetic PSD peak 都是 1.6667 Hz。结论：walking 保留主频，但 ACF harmonic / 二倍周期结构不完全一致。
- 当前判断：ACF/PSD 结果支持“baseline 学到主要运动节奏，尤其 running 明显”；仍不能直接支持 HAR utility claim。

## 2026-05-22 Sample-level stratification

实验报告：`docs/experiments/sample-stratification-20260522.md`。

- 已按 ACF lag、PSD peak、std 和 abs max 将 synthetic samples 分为 good / borderline / bad。
- walking：130 个 synthetic windows 中 good 57、borderline 64、bad 9。
- running：98 个 synthetic windows 中 good 42、borderline 14、bad 42。
- 当前判断：sample quality uneven；running 平均周期结构很强，但单样本层面仍有不少频率偏移或过平滑样本。

## 2026-05-23 Minimal HAR utility smoke

实验计划：`docs/experiments/har-utility-smoke-plan-20260523.md`。
实验报告：`docs/experiments/har-utility-smoke-results-20260523.md`。

- 已完成 walking vs running 的最小 HAR utility smoke，测试集为 held-out real windows。
- real-only accuracy 0.6126；synthetic-only-all accuracy 0.7027；real+synthetic-all accuracy 0.7117；synthetic-only-good accuracy 0.7207；real+synthetic-good accuracy 0.6396。
- 当前判断：synthetic windows 有明确 task-discriminative signal，并且 all synthetic augmentation 在该 smoke 中优于 real-only；但结果仍为 provisional，只覆盖 subject101、单通道、两类动作和一个简单 classifier。

## 2026-05-23 Label conditioning v1

实验报告：`docs/experiments/label-conditioning-v1-results-20260523.md`。

- 已在远程 recovered SDForger pipeline 中加入最小 activity textual conditioning：`Condition: data is walking/running`。
- walking label-conditioned run：job `34526946`，生成 72 个 synthetic windows。
- running label-conditioned run：job `34526947`，生成 90 个 synthetic windows。
- HAR utility smoke：synthetic-only-all accuracy 0.7387，real+synthetic-all accuracy 0.7117。
- ACF/PSD：running 仍匹配主周期；walking 仍保持 PSD 主频匹配但 ACF 为 58 vs 116 的 harmonic mismatch。
- 当前判断：label conditioning v1 可以作为 `feasible/provisional method-extension`，但不能说已经稳定优于 unconditioned baseline；TSGBench-style metrics 对 walking 和 running 多数没有改善，且日志中有 malformed textual outputs 被过滤。

## 2026-05-23 Unified label conditioning experiment

实验计划：`docs/experiments/unified-label-conditioning-plan-20260523.md`。
实验报告：`docs/experiments/unified-label-conditioning-results-20260523.md`。

- 已完成一个真正 unified 的 walking/running conditional generator：分别切 activity windows，合并后做 joint FICA embedding，并添加 window-level `data=walking/running` label。
- 成功 job：`34530050`。walking 生成 72 个，running 生成 79 个。
- 正向结果：label controllability accuracy 0.8212；requested walking accuracy 0.8056，requested running accuracy 0.8354。
- 负向结果：生成幅度严重失控，walking synthetic abs max 3761.8、running synthetic abs max 1572.1；PSD peak 都偏到 2.3333Hz。
- HAR utility：all synthetic 明显低于 real-only；good-only filtering 后 real+synthetic-good accuracy 0.6757，高于 real-only 0.6126，但 good samples 很少。
- 当前判断：unified label conditioning 证明 label signal 可被模型使用，但当前 joint embedding / unconstrained text generation 不是可用 generator；下一步应优先做 latent/value constraints。

## 2026-05-23 Latent/value constraint diagnostic

实验计划：`docs/experiments/latent-constraint-plan-20260523.md`。
实验报告：`docs/experiments/latent-constraint-results-20260523.md`。

- 已对 unified label-conditioned outputs 做 post-hoc latent distribution constraints：`clip_minmax`、`clip_p05_p95`、`reject_iqr3`。
- 关键发现：constraint 能把 raw unified 的 decoded abs max 从 walking 3761.8 / running 1572.1 拉回合理 standardized 范围；`clip_p05_p95` 为 walking 3.24、running 2.65。
- Label controllability 没有消失，反而增强；`clip_p05_p95` overall accuracy 0.9868，walking 1.0000，running 0.9747。
- HAR utility mixed：`clip_p05_p95` synthetic-only-all accuracy 0.6937，高于 real-only 0.6126；但 real+synthetic-all accuracy 0.5946，说明直接 augmentation 仍不稳定。
- ACF/PSD mixed：running rhythm 基本恢复，walking 仍偏离真实 ACF/PSD。
- 当前判断：latent/value constraint 是正确的下一步方向，但当前结果只能作为 diagnostic evidence；下一步应实现 generation-time latent validity check / rejection sampling，而不是把 post-hoc clipping 当作最终方法。

## 2026-05-30 Advisor feedback synthesis

反馈记录：`memory/feedback/2026-05-30-advisor-pamap2-conditional-generation.md`。

- 导师反馈 1：不同 channel 可能有幅度尺度不一致；unified/multichannel setting 下应考虑 embed 前 normalization。
- 导师反馈 2：评估需要明确是否是 unseen 数据，尤其是否能跨 subject 泛化；当前 subject101 smoke 不能当 unseen-subject evidence。
- 导师反馈 3：需要解释 `clip_p05_p95` 的具体机制，以及为什么图上看起来更平滑；后续要量化 clipping 是否造成 amplitude/detail 压缩。
- 当前判断：下一步实验不应直接扩大到所有 channel/subjects；应先设计两个受控 ablation：pre-embedding normalization 和 subject-level split。

## 2026-06-07 Pre-embedding normalization ablation submitted

实验计划/提交记录：`docs/experiments/normalization-ablation-plan-20260607.md`。

- 已按导师反馈和当前 failure analysis 启动 normalization ablation，目标是诊断 unified label-conditioned SDForger 的 value explosion 是否来自 FICA input space 的尺度/标准化方式。
- 重要澄清：旧 `raw-unified` 并不是 raw sensor values 直接进入 FICA；旧脚本会对 walking/running 分别做 SDForger window-level StandardScaler 后再合并。因此这次实验显式比较不同 FICA input space，而不是简单在同一 preprocessing 前多加一个 z-score。
- 已提交 Puhti Slurm array job：`34758239`，4 个 array task 均已启动运行。
- 当前 modes：`current_activity_window_zscore`、`joint_window_zscore`、`global_series_zscore`、`activity_series_zscore`。
- 当前判断：这是 diagnostic experiment；不要 claim normalization 有效。任何 normalization variant 进入下一步前，都必须先按统一 raw-like evaluation protocol 重算。

## 2026-06-07 Normalization ablation results

实验报告：`docs/experiments/normalization-ablation-results-20260607.md`。

- Puhti job `34758239` 已完成，4 个 array task 全部 `COMPLETED 0:0`。
- 结果：normalization-only 没有解决 unified conditional generation 的 value outlier 问题。
- `global_series_zscore` 是最不差的新增 variant：running ACF lag diff 为 1，PSD diff 为 0；但 amplitude ratio 仍约 178x，walking/running synthetic abs max 仍分别为 277.49 / 811.34。
- `joint_window_zscore` 对 running 最差，synthetic abs max 达 187737.53，说明 joint timestamp-wise scaling 不是当前方向。
- 已补 TSGBench-style model-space 指标（MDD/ACD/SD/KD/ED/DTW）。`global_series_zscore` running 的 DTW 在 unified normalization variants 中最好，但 value-scale diagnostics 仍显示严重 outlier。
- 当前判断：该报告降级为 diagnostic-only；不要直接进入 multi-subject expansion，也不要把 normalization model-space 指标作为主结果。应先补统一 raw-like evaluation。

## 2026-06-07 Unified evaluation protocol

协议文档：`docs/experiments/unified-evaluation-protocol-20260607.md`。

- 后续主评估统一为：`real raw windows` vs `inverse-normalized synthetic raw-like windows`。
- 主表统一包含：amplitude ratio、ACF lag diff、PSD Hz diff、MDD、ACD、SD、KD、ED、DTW、HAR real+synthetic accuracy；conditional setting 额外报告 requested-label accuracy。
- Model-space metrics 只作为 debugging / diagnostic，不再作为主结果。
- 下一步优先级：先重算 clean baseline、raw unified、`clip_p05_p95` 和 `global_series_zscore` 的 raw-like evaluation；再决定是否进入 latent validity / rejection sampling 或 multi-subject。

## 2026-06-07 Unified raw-like evaluation + code review

实验报告：`docs/experiments/unified-raw-like-evaluation-results-20260607.md`。
代码审查：`docs/reviews/2026-06-07-normalization-evaluation-code-review.md`。

- 已新增并运行 `scripts/evaluate_unified_raw_like_metrics.py`，统一比较 clean baseline、raw unified label-conditioned、`clip_p05_p95` 和 `global_series_zscore`。
- 主评估空间已统一为 raw-like sensor space；model-space 结果继续降级为 diagnostic-only。
- 关键结果：clean unconditioned 仍是当前最稳 baseline；raw unified 有 label controllability 但 value explosion 严重；`clip_p05_p95` 明显修复 value explosion 并保留 label controllability，但 HAR augmentation 不稳定；`global_series_zscore` 在 raw-like 下不可作为当前改进方向。
- 本次快速评估使用 `--skip-dtw`，因此 DTW 尚未补；其余 MDD/ACD/SD/KD/ED、ACF/PSD、HAR utility 和 label controllability 已完成。
- 当前建议：下一步优先做 generation-time latent validity control / rejection sampling，而不是直接加 subject；subject-level split 应放在 value validity 稳定之后。
