# SDForger × PAMAP2 — LLM 生成 HAR 合成时序(ELEC-E7633 Project)

> 课程项目:用 **LLM(SDForger 框架)生成合成传感器时间序列**用于 HAR(人体活动识别),
> 数据集 **PAMAP2**,核心主题 = **prompt 设计**("怎么设计 prompt 让 LLM 更好地完成生成任务")。
> 这份 README 同时面向**人**和 **AI agent** 阅读;"实验结果 + 推理逻辑"都在仓库内,见下方索引。

---

## 0. 30 秒看懂(TL;DR)

- **方法**:真实窗口 → 周期分段 → **FICA** 嵌入(每窗 ~8 个系数)→ 文本序列化 → **LLM 微调(next-token)** → 采样系数 → 线性解码回波形。我们在此基础上**改进 + 系统刻画**,不是和外部模型比 benchmark。
- **最重要的几条结论**(踩坑提醒,你效果烂大概率中了其一):
  1. **别用 gpt2,用 Qwen2.5-1.5B**:gpt2 会爆炸(生成值 abs max 飙到 3761,真实才 ~4.5),Qwen 源头不爆(max 2.98)。
  2. **per-activity clip 修复是最大杠杆**:把生成系数 clip 到"每个动作各自"的 train p05/p95(不是合并),walking ACD 从 2+ → ~0.7。
  3. **multi-subject 救 running**(单人 1.37 → 多人 0.74);**~50 epoch 是甜点**(100 略过训练)。
  4. **磁盘**:每个 Qwen checkpoint ~12GB,跑几十个会撑爆 scratch 配额→训练静默失败。生成完 `rm -rf .../model`。
  5. **GPU 排队**:`gpu` 分区巨堵(排几小时),短 job 用 **`gputest`**(15min 上限,周转快)。
- **最优配置**:Qwen2.5-1.5B + multi-subject(101/102/105)+ per-activity clip → **train walk/run ACD 0.73/0.89,held-out(106/108)0.98/1.46**。
- **核心新发现(prompt 可控性边界)**:prompt 能控**幅度**(max,adherence 0.40)、**控不了频率**(period ~0)——因为频率焊在 FICA 基里、幅度在可调系数里。

---

## 1. 仓库结构(人 + agent 导航)

| 路径 | 内容 |
|---|---|
| `docs/experiments/` | **实验报告(结果 + 推理逻辑)** — 见第 2 节索引 |
| `docs/meetings/` | 组会纪要(`2026-06-22-group-meeting-notes.md`) |
| `memory/` | 项目记忆 boards:`current-status.md`(最新状态,**先读这个**)、`evidence-board.md`、`risk-board.md`、`action-board.md`、`claim-board.md` |
| `scripts/` | 所有实验/评估脚本 — 见第 3 节 |
| `puhti-generated/slurm/` | Puhti(CSC)的 SLURM job 脚本 |
| `outputs/` | 结果产物;**`outputs/meeting-figures-20260629/`** = 7 张会议图表 |
| `reference/sdforger-official/` | SDForger 官方代码参照 |
| `AGENTS.md` / `CLAUDE.md` | **agent 工作指引 + 约束**(远程操作前先只读检查等) |
| `PROJECT.md` | 项目定位 |

> 注:`legacy/`(旧 agentsense/chatts 项目)、队友仓库 tarball、大 PDF、model checkpoint 已 `.gitignore`,不在仓库内。

---

## 2. 实验结果 + 逻辑(docs/experiments 索引)

**先读这几个(最新、最全):**
| 文件 | 讲什么 |
|---|---|
| `this-week-summary-20260628.md` | **本周全部实验 + 结果**(四块:最优配置/回应建议/可控性/诚实 baseline)——总览先看它 |
| `qwen-best-config-results-20260625.md` | 最优配置落到 Qwen(2-act/3-act/held-out)+ 与 gpt2 公平对比 + norm 组合(负结果) |
| `descriptor-control-results-20260625.md` | **prompt 可控性边界**(7 轮):幅度可控/频率不可控 + 双层机制(表示×训练,~50 epoch 甜点) |
| `project-summary-and-workshop-readiness-20260628.md` | 三人工作整合 + workshop 可投性评估 + 补强路径 |

**支撑性 / 历史报告:**
| 文件 | 讲什么 |
|---|---|
| `smooth-repair-results-20260615.md` | per-activity 修复(clip/shrink/线性插值)对比 → **per-activity 边界是最大杠杆** |
| `combined-best-config-results-20260621.md` | multi-subject + per-activity(gpt2 版最优)|
| `har-utility-heldout-results-20260621.md` | 下游 HAR utility(held-out subject)|
| `meeting-table-20260622.md` / `full-metrics-table-20260622.md` | 全指标对比表(经对抗审计的诚实分级版)|
| `experiment_plan_*.md` | 各阶段实验计划(假设/证伪/对照)|

**评估指标**:生成质量用 **TSG(MDD/ACD/SD/KD/ED/DTW,对齐 TSGBench,越低越好)**;可控性用 **adherence(请求 vs 实测的 Spearman)**;下游用 **HAR acc/F1(辅助)**。

---

## 3. 脚本参考(scripts/)

| 脚本 | 作用 |
|---|---|
| `run_stat_prompt_hf.py` | **主生成脚本**:Qwen/gpt2 微调 + 生成,支持 `--encoding {none,values,bins,percentile}`、`--stat {max,std,period,range}`(可控性实验) |
| `run_unified_multiactivity.py` | N 动作联合生成(walk/run/cycling…),已加 HF 后端配 Qwen |
| `run_unified_label_conditioning_normalization.py` | 带 normalization 模式的生成(`--normalization-mode`、`--repair clip`) |
| `apply_smooth_latent_repair.py` / `repair_and_eval_multiactivity.py` | **per-activity 修复**(clip/shrink)+ 评估 |
| `evaluate_sdforger_paper_metrics.py` | TSG 指标(MDD/ACD/…/DTW) |
| `evaluate_har_utility_heldout.py` | 下游 HAR 分类 utility |
| `gaussian_coeff_baseline.py` / `conditional_gaussian_adherence.py` / `within_subject_floor.py` / `nn_memorization_check.py` | **诚实 baseline / 消融**(界定 LLM 到底加了什么) |
| `make_meeting_figures.py` / `plot_samples.py` | 出图(会议柱状图/散点 + 单样本好坏对比) |

---

## 4. 复现 / 运行(Puhti 集群)

```bash
# 环境
module load pytorch/2.6
source /projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312/bin/activate
export DGT_DATA_DIR=$BASE/data    # BASE=fms-dgt 工作目录(在集群 scratch)

# 例:Qwen 最优配置(multi-subject + per-activity 修复)
sbatch puhti-generated/slurm/<job>.sh   # 或参考 scripts/ 里的命令
```
**踩坑(务必看)**:① 用 Qwen 不用 gpt2;② 生成后 `rm -rf $OUT/model` 防配额满;③ 短 job 用 `gputest` 分区;④ vLLM 会崩 Qwen → Qwen 用 HF 后端(`--gen-backend hf`),gpt2 才用 vLLM。

> ⚠️ 代码 + 数据在集群 `fms-dgt` 工作目录;本仓库是**论文/实验记录侧**(报告、脚本、结果、记忆),不含数据集与大 checkpoint。

---

## 5. 给 AI agent 的导航

1. **先读 `memory/current-status.md`** —— 项目最新状态、所有结论、未决项。
2. 再读 `docs/experiments/this-week-summary-20260628.md` —— 实验全景。
3. 遵守 `AGENTS.md` / `CLAUDE.md` 的约束(尤其:**远程/集群写操作前先做只读检查,不做未确认的删除/覆盖**)。
4. 旧结果按 `needs-verification` 对待,进 claim 前需重跑/重算(见 `memory/risk-board.md`)。
5. 指标口径见第 2 节;诚实定位:**LLM 价值在(有边界的)可控性 + 非拷贝生成,不在无条件保真度**(被 trivial baseline 追平,见 baseline 脚本)。

---

## 6. 状态 / 下一步

- **已完成**:Qwen 最优配置(train+held-out)、gpt2-vs-Qwen、prompt 可控性边界、诚实 baseline、多动作扩展。
- **下一步**(框架内,workshop 用):多通道生成(SDForger 招牌 + HAR 真实场景)、可控性 multi-subject 复验、下游 pre-train+fine-tune。
- 详见 `docs/experiments/project-summary-and-workshop-readiness-20260628.md`。
