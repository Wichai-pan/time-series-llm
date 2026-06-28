# Action Board

最后更新：2026-06-15

## ACT-001：补齐用户材料

- Owner：user
- Component：project
- 状态：todo
- 摘要：补充当前 reset 需要的材料。
- 需要材料：
  - 当前一段话版本的 project idea
  - 这个文件夹之外的旧笔记
  - 论文列表或 BibTeX
  - 关键词和目标 community
  - 其他本地残留代码路径
  - 远程路径、job 命名、输出命名规则
  - 导师/课程反馈和评分预期
- 相关风险：RSK-004、RSK-006
- 下一步：用户提供材料后，更新 memory 和 reference index。

## ACT-002：锁定 revised research question

- Owner：agent + user
- Component：project
- 状态：todo
- 摘要：决定近期项目到底是 reproduction/adaptation、empirical evaluation、method extension，还是 HAR augmentation study。
- 相关 claim：CLM-001、CLM-002、CLM-003
- 相关风险：RSK-006
- 下一步：阅读 `docs/designs/idea-reset.md` 后选择一个候选题目。

## ACT-003：定义最小质量评估指标

- Owner：agent
- Component：project
- 状态：done
- 摘要：已统一主评估协议：后续主结果使用 raw-like inverse-normalized synthetic windows 对比 real raw windows；model-space metrics 只作为 diagnostic。
- 相关 claim：CLM-002
- 相关风险：RSK-003
- 下一步：按 `docs/experiments/unified-evaluation-protocol-20260607.md` 重算关键 setting 的 raw-like metrics。

## ACT-004：定义下游 HAR utility protocol

- Owner：agent
- Component：project
- 状态：partial
- 摘要：已完成 subject101 walking/running `hand_acc16_x` 的最小 HAR utility smoke，包含 majority、real-only、synthetic-only、real+synthetic 和 good-only synthetic 条件。
- 相关 claim：CLM-003
- 相关风险：RSK-003
- 下一步：决定是否扩展到 ankle channel、multichannel 或 subject split；当前 smoke 只作为 provisional utility signal。

## ACT-005：基于本地 seed papers 做 literature sprint

- Owner：agent
- Component：reference
- 状态：todo
- 摘要：从 SDForger、ChatTS、AgentSense 和用户提供材料出发，建立 canonical、closest、recent、baseline 和 angle map。
- 相关风险：RSK-004、RSK-005、RSK-006
- 下一步：完善 source cards 和 literature map。

## ACT-006：维护结束后的远程核验

- Owner：agent
- Component：code
- 状态：doing
- 摘要：已开始通过 Puhti 只读检查核验远程旧项目目录、logs、outputs、reports、plots 和 Slurm files；尚未建立 run-level provenance。
- 相关 evidence：EVD-001
- 相关风险：RSK-001、RSK-002、RSK-015
- 下一步：为一个代表性 PAMAP2 run 建 manifest，确认 task/log/output/report/eval script 对应关系。

## ACT-007：重建干净的最小 SDForger/PAMAP2 证据链

- Owner：agent
- Component：code
- 状态：doing
- 摘要：已完成 walking/running + `hand_acc16_x` 的 clean activity-specific univariate baseline rerun，并补充 ACF/PSD 周期诊断、sample-level 分层和最小 HAR utility smoke。
- 相关 claim：CLM-001、CLM-002
- 相关风险：RSK-002、RSK-003
- 下一步：为周一汇报整理 3-4 张 slide；后续实验优先扩展到另一通道或 multichannel。

## ACT-008：baseline selection audit

- Owner：agent
- Component：project
- 状态：todo
- 摘要：确定 time-series generation 和 HAR augmentation 的最低公平 baseline。
- 相关 claim：CLM-002、CLM-003
- 相关风险：RSK-005
- 下一步：依赖 ACT-005 的 literature sprint。

## ACT-009：要求队友澄清 utility generation split

- Owner：user / teammate
- Component：evaluation
- 状态：todo
- 摘要：确认 `sdforger-pamap2` 的 `prompt_split: test` 是否用于 README 中所有 utility results，以及是否有 train-conditioned synthetic run。
- 相关风险：RSK-007
- 下一步：把 `docs/reviews/2026-05-18-teammate-review.md` 的 High finding 发给队友确认。

## ACT-010：重跑 train-conditioned HAR utility smoke

- Owner：teammate / agent after remote available
- Component：experiment
- 状态：blocked
- 摘要：用 train-side prompts 生成 synthetic samples，再评估 real-only、synthetic-only、real+synthetic on held-out test。
- 相关 claim：CLM-003
- 相关风险：RSK-007、RSK-001
- 下一步：等待远程或 artifact bundle 可用。

## ACT-011：修正或重命名队友 metric definitions

- Owner：teammate
- Component：evaluation
- 状态：todo
- 摘要：澄清 `MDD` / `SD` 是 local proxy 还是标准指标；若是标准指标，按文献定义实现并重跑 tables。
- 相关 claim：CLM-002、CLM-003
- 相关风险：RSK-008
- 下一步：要求队友给出 metric definition mapping。

## ACT-012：要求队友提供 artifact manifest 或 smoke bundle

- Owner：teammate
- Component：reproducibility
- 状态：todo
- 摘要：为 `sdforger-pamap2` 提供 raw/tensor/checkpoint/generated artifacts 的 hashes、sizes、paths，或一个可离线验证的最小 smoke bundle。
- 相关 evidence：EVD-004
- 相关风险：RSK-009
- 下一步：列出 README 指标对应的 artifact provenance。

## ACT-013：确认 mmfit-inference 访问权限

- Owner：user / teammate
- Component：collaboration
- 状态：todo
- 摘要：确认 `IkMangMok/mmfit-inference` URL 是否正确，或给当前账号 repo access。
- 相关风险：RSK-010
- 下一步：拿到访问后再做正式 code review。

## ACT-014：要求 mmfit-inference 提供 clean smoke artifacts

- Owner：teammate
- Component：reproducibility
- 状态：todo
- 摘要：提供一个最小 end-to-end smoke run 的 generated latents、decoded JSONL、eval report、raw generations 和 exact commit/config。
- 相关 evidence：EVD-005
- 相关风险：RSK-011
- 下一步：把 review 中 generation/evaluation 未闭环的问题发给队友。

## ACT-015：重跑 mmfit-inference full generated-data evaluation

- Owner：teammate / agent after remote available
- Component：experiment
- 状态：blocked
- 摘要：在 parser/retry/workflow 修正后，重跑 generate -> decode -> evaluate，并归档 final report。
- 相关 claim：CLM-002、CLM-003
- 相关风险：RSK-011、RSK-001
- 下一步：等待远程环境或队友 artifact bundle。

## ACT-016：修正 mmfit-inference workflow script failure handling

- Owner：teammate
- Component：workflow
- 状态：todo
- 摘要：给 `run_generate_llm_data.sh` 等脚本加入 strict shell mode、文件存在性检查和 stage failure messages。
- 相关风险：RSK-012
- 下一步：要求队友先修 workflow，再重跑。

## ACT-017：补充 mmfit-inference autoencoder validation

- Owner：teammate
- Component：evaluation
- 状态：todo
- 摘要：autoencoder checkpoint 不应只按 train loss 选择；需要 train-side validation split 或 held-out subject，并单独报告 reconstruction metrics。
- 相关风险：RSK-013
- 下一步：要求队友说明 decoder selection protocol。

## ACT-018：给 mmfit-inference 添加最小测试

- Owner：teammate
- Component：testing
- 状态：todo
- 摘要：增加 latent JSON parsing、latent dimension validation、decode shape invariant、split isolation、empty/malformed JSONL 和 script failure behavior tests。
- 相关风险：RSK-014
- 下一步：要求队友至少补 parser/decode/eval smoke tests。

## ACT-019：为 Puhti 旧 SDForger/PAMAP2 run 建 manifest

- Owner：agent
- Component：code
- 状态：todo
- 摘要：选择一个代表性 run，例如 `pamap2_subject101_univariate_paper`，记录 task yaml、Slurm log、output dir、final_data.jsonl、checkpoint、report、eval script 和文件时间。
- 相关 evidence：EVD-001、EVD-006
- 相关风险：RSK-002、RSK-015
- 下一步：只读抽查 `task_card.jsonl`、`task_results.jsonl`、log tail 和 report JSON。

## ACT-020：恢复或重建 Git provenance

- Owner：agent / user
- Component：code
- 状态：todo
- 摘要：远程 `/scratch/project_2016517/panh/time-series-llm` 未发现 `.git`，需要找到对应 GitHub/local source commit，或建立新的 clean repo 承接代码。
- 相关 evidence：EVD-006
- 相关风险：RSK-015
- 下一步：对比本地 `legacy/old-project-files/puhti-time-series-llm/` 和远程 `fms-dgt/`，决定 clean repo 来源。

## ACT-021：重新准备 HAR-aware PAMAP2 数据

- Owner：agent
- Component：data
- 状态：todo
- 摘要：当前旧 PAMAP2 parquet 主要保留传感器通道，默认预处理会丢弃 `activity_id`；若目标包含 classification/label conditioning，需要保留 label 并定义 split/window protocol。
- 相关 claim：CLM-003
- 相关风险：RSK-003、RSK-015
- 下一步：写 `docs/experiments/pamap2-data-protocol.md`，明确 subject split、activity labels、channels 和 windowing。

## ACT-022：旧结果进入 evidence 前重跑或重算指标

- Owner：agent
- Component：evaluation
- 状态：todo
- 摘要：旧 reports 可作为线索，但在 claim 使用前必须用固定 eval script、明确 synthetic-space、固定 subset protocol 重算。
- 相关 evidence：EVD-001、EVD-006
- 相关风险：RSK-002、RSK-003、RSK-015
- 下一步：选定 metric suite 后，对一个旧 generated JSONL 做只读重算对照；正式证据仍需 clean rerun。

## ACT-023：重建 label-aware PAMAP2 preprocessing

- Owner：agent
- Component：data
- 状态：partial
- 摘要：已从原始 `subject101.dat` 过滤 walking/running 并生成 activity-specific univariate parquet；但当前 parquet 为复现 SDForger univariate baseline，只保留 sensor channel，activity label 仍未进入 generation prompt 或 classifier protocol。
- 相关 claim：CLM-003
- 相关 evidence：EVD-006、EVD-007
- 相关风险：RSK-016
- 下一步：写 label/covariate conditioning 版本的数据协议，明确是否将 activity label 放入 prompt、metadata 或 downstream classifier。

## ACT-026：重跑 clean activity-specific SDForger univariate baseline

- Owner：agent
- Component：experiment
- 状态：done
- 摘要：已在 walking/running + `hand_acc16_x` 上沿用旧 SDForger univariate + FICA setting 完成 clean baseline。walking 生成 130 个 synthetic windows，running 生成 98 个 synthetic windows，并已重算 metrics。
- 相关 evidence：EVD-008、EVD-009
- 相关风险：RSK-016、RSK-017
- 下一步：人工检查 overlay PDF；将结果用于周一 baseline verification 汇报，但不要作为 HAR utility final claim。

## ACT-024：明确 generated output 的 scaled/raw value-space contract

- Owner：agent
- Component：evaluation
- 状态：done-for-current-scope
- 摘要：已明确后续主评估必须回到 raw-like sensor space，并新增统一 evaluator。normalization ablation 当前 model-space 指标已降级为 diagnostic-only。
- 相关 evidence：EVD-006、EVD-007
- 相关风险：RSK-017
- 下一步：后续新实验必须沿用 `scripts/evaluate_unified_raw_like_metrics.py` 或同等 contract；新增 run metadata 必须记录 value-space / inverse-normalization。

## ACT-025：修正 multivariate auto embedding dimension

- Owner：agent
- Component：code
- 状态：todo
- 摘要：`embedding_dim: auto` 在 channel loop 中会被覆盖成第一个 channel 的整数维度，后续 channel 不能独立选择维度。
- 相关 evidence：EVD-007
- 相关风险：RSK-018
- 下一步：把每个 channel 的 auto dimension 存为局部变量，并添加两通道 synthetic unit test。

## ACT-027：迁移并测试 label conditioning patch

- Owner：agent
- Component：code / method
- 状态：todo
- 摘要：label conditioning v1 已在 Puhti recovered workspace 中通过最小 patch 跑通，但尚未迁移到 Git-controlled clean repo，也没有 parser/template tests。
- 相关 evidence：EVD-013
- 相关风险：RSK-022
- 下一步：把 `generate.py` / `trainer.py` 的 textual conditioning patch 迁移到 clean source tree，添加 template round-trip、malformed output filtering 和 numerical-column extraction tests。

## ACT-028：设计 unified activity-conditioned generator

- Owner：agent + user
- Component：experiment
- 状态：done-for-smoke
- 摘要：已完成一个 unified walking+running conditional generator smoke。结果显示 label controllability positive，但生成质量因 latent/value outliers 明显失败。
- 相关 evidence：EVD-012、EVD-013
- 相关风险：RSK-020、RSK-022
- 下一步：基于 EVD-014 设计 latent/value constraint 版本，而不是继续直接增加 label 或 activity 数。

## ACT-029：修复 unified conditional generation 的 latent/value outliers

- Owner：agent
- Component：method / evaluation
- 状态：done-for-diagnostic
- 摘要：已完成 post-hoc latent constraints 诊断：`clip_minmax`、`clip_p05_p95`、`reject_iqr3`。结果显示训练分布约束可修复 value explosion 并保留 label controllability，但 walking rhythm 和 augmentation stability 仍未完全解决。
- 相关 evidence：EVD-014、EVD-015
- 相关风险：RSK-023
- 下一步：把 post-hoc constraint 升级为 generation-time latent validity check / rejection sampling，并报告 accepted/rejected counts。

## ACT-030：实现 generation-time latent validity control

- Owner：agent
- Component：method / experiment
- 状态：done-for-diagnostic
- 摘要：已完成两个 generation-time-style validity 规则的诊断：`strict_reject_p05_p95` 和 `soft_repair_p05_p95_minmax`。strict reject 样本数太少；soft repair 指标接近 post-hoc clip，但没有明显全面优于 clip，优势主要是能记录 clean/repaired/rejected counts。
- 相关 evidence：EVD-015、EVD-021
- 相关风险：RSK-023、RSK-024
- 下一步：若继续方法化，应把 soft repair 接入实际 generation loop，让 rejected rows 触发 resampling，并补 held-out HAR utility。

## ACT-031：设计 pre-embedding normalization ablation

- Owner：agent
- Component：data / method
- 状态：done-for-diagnostic
- 摘要：根据 2026-05-30 导师反馈，已完成 subject101 walking/running `hand_acc16_x` unified label-conditioning normalization ablation。结果显示 normalization-only 没有解决 value explosion；`global_series_zscore` 最不差但仍有严重 amplitude outliers。
- 相关 evidence：EVD-015
- 相关风险：RSK-025
- 下一步：不要直接扩展 multi-subject；先按 ACT-034 补 raw-like evaluation。`global_series_zscore` 只作为 candidate setting，不作为主结果。

## ACT-032：定义 unseen-subject evaluation protocol

- Owner：agent + user
- Component：evaluation
- 状态：done-for-diagnostic
- 摘要：已完成最小 unseen-subject evaluation-only diagnostic：generator train subjects 为 101/102/105，held-out real reference subjects 为 106/108。结果显示 `clip_p05_p95` 在 unseen reference 下 ED/DTW/KD 仍接近 train-reference clip，但 ACD/rhythm 仍是风险。
- 相关 claim：CLM-003
- 相关 evidence：EVD-020
- 相关风险：RSK-020、RSK-026
- 下一步：在 generation-time latent validity control 完成后，补 held-out HAR utility，而不是只做 TSG reference comparison。

## ACT-034：统一 raw-like evaluation 重算关键 setting

- Owner：agent
- Component：evaluation
- 状态：done-for-current-scope
- 摘要：已按 `docs/experiments/unified-evaluation-protocol-20260607.md`，把关键 generated outputs inverse-normalize 到 raw-like sensor space，并输出统一主指标表。
- 范围：
  - clean unconditioned baseline
  - raw unified label-conditioned failed baseline
  - `clip_p05_p95`
  - `global_series_zscore`
- 指标：amplitude ratio、ACF lag diff、PSD Hz diff、MDD、ACD、SD、KD、ED、DTW、HAR real+synthetic accuracy；conditional setting 额外 requested-label accuracy。
- 相关风险：RSK-003、RSK-017、RSK-028
- 结果：`docs/experiments/unified-raw-like-evaluation-results-20260607.md`；本次快速评估使用 `--skip-dtw`，DTW 待补。
- 下一步：若需要完整 TSGBench-style 表，单独补 DTW；方法上优先 generation-time latent validity control。

## ACT-033：解释并量化 clipping 的影响

- Owner：agent
- Component：method / evaluation
- 状态：partial
- 摘要：已通过统一 raw-like evaluation 量化 `clip_p05_p95` 对 abs max、std、ACF/PSD、HAR 和 label controllability 的影响；仍需补 roughness / derivative energy 与 generation-time acceptance/rejection rate。
- 相关 evidence：EVD-015
- 相关风险：RSK-024、RSK-027
- 下一步：整理一页 method note：clip 作用位置、数学定义、为什么可能变平滑、哪些指标能检验平滑副作用。

## ACT-035：设计并执行 multi-subject unified conditioning smoke

- Owner：agent
- Component：experiment
- 状态：done-for-diagnostic
- 摘要：已完成 multi-subject unified conditioning smoke。执行前发现 subject103 没有 running，因此实际使用 PAMAP2 subject101/102/105、walking/running、`hand_acc16_x`。结果显示增加 subject 没有单独修好 raw-unified；`clip_p05_p95` 仍是关键稳定器。
- 相关 evidence：EVD-018、EVD-019
- 相关风险：RSK-020、RSK-023、RSK-026
- 下一步：不要继续单纯加 subject；优先实现 generation-time latent validity / rejection sampling，并在稳定后再做 held-out subject evaluation。

## ACT-036：执行 T2 stat-prompt 实验

- Owner：agent
- Component：method / experiment
- 状态：done（2026-06-17，H1 证伪 / 负结果）
- 摘要：按 `docs/experiments/experiment_plan_2026-06-15_stat-prompt.md`，在 unified label-conditioned SDForger 的 prompt `Condition:` 段加入量化窗口统计量（mean/std/min/max），从源头减少越界 latent；核心对照 stat-prompt **raw vs 轻 per-activity 修复**，验证"改好 prompt 是否就不需要复杂约束"。
- 结果（`docs/experiments/stat-prompt-results-20260617.md`）：两条预注册证伪条件都触发——stat-adherence≈0（GPT-2 忽略数字条件）、R2a 仍爆炸；prompt-only 不能替代 T1 修复。结论：天花板是 GPT-2 无法条件化于数值。
- 后续 ACT（待建）：更强条件化（class/stat embedding、CFG）、constrained decoding、learned latent prior。
- 前置 T1：smooth latent repair + per-activity 边界已完成（`docs/experiments/smooth-repair-results-20260615.md`），walking ACD 2.25→0.36，作为 T2 对照基线。
- 相关 claim：CLM-002
- 相关 evidence：EVD-022（planned）
- 相关风险：RSK-030（GPT-2 可能忽略数字条件，stat-adherence 直接检验）
- 下一步：本地改 `generate.py` + prompt round-trip 测试 → 远程留 `.pre_stat_prompt_20260615` 备份 → rsync → `sbatch` GPU smoke（提交前贴 diff 给用户确认）。
