# Risk Board

最后更新：2026-06-07

## RSK-001：远程核验被集群维护阻塞

- 类型：execution / reproducibility
- 严重程度：high
- 概率：medium
- 摘要：Puhti 已可通过 SSH 做只读文件检查，但旧输出、报告和 provenance 仍未完成 run-level 核验。
- 威胁：CLM-001、CLM-002、CLM-003
- 缓解：所有旧结果保持 `needs-verification`；先建立远程 artifact manifest，再决定是否重跑。
- 相关 action：ACT-006
- 状态：open
- 确定性：observed

## RSK-002：旧实现和旧结果质量可能不稳

- 类型：reproducibility / evaluation
- 严重程度：high
- 概率：medium
- 摘要：本地代码不完整，旧实验可能存在 config、untracked changes 或评估不足的问题。
- 威胁：CLM-001、CLM-002
- 缓解：先核验 commit/config；在使用任何旧结果前，重新建立一个干净的最小实验。
- 相关 action：ACT-006、ACT-007
- 状态：open
- 确定性：user-stated

## RSK-003：评估指标可能支撑不了目标 claim

- 类型：evaluation
- 严重程度：high
- 概率：medium
- 摘要：样本数和肉眼看图不足以支撑 synthetic HAR data quality 或 utility 的 claim。
- 威胁：CLM-002、CLM-003
- 缓解：明确最小 metric suite，包括 statistics、ACF、PSD、diversity 和 downstream utility。
- 相关 action：ACT-003、ACT-004
- 状态：open
- 确定性：inferred

## RSK-004：closest work 和 novelty 尚未解决

- 类型：novelty / reviewer
- 严重程度：high
- 概率：medium
- 摘要：SDForger、ChatTS、AgentSense、TSGBench、TimeGAN/TimeVAE、diffusion 和 HAR synthetic-data 论文都可能覆盖部分贡献。
- 威胁：CLM-001、CLM-002、CLM-003
- 缓解：先做离线文献 sprint，之后再用 primary sources 核验 recent/concurrent work。
- 相关 action：ACT-005
- 状态：open
- 确定性：inferred

## RSK-005：baseline 风险

- 类型：baseline
- 严重程度：high
- 概率：medium
- 摘要：如果只比较 SDForger 参数变体，项目容易被看作 reproduction 或 engineering adaptation。
- 威胁：CLM-002、CLM-003
- 缓解：实验设计前先确定 must-have baselines。
- 相关 action：ACT-005、ACT-008
- 状态：open
- 确定性：inferred

## RSK-006：generation、understanding、simulation 三条线混在一起

- 类型：positioning
- 严重程度：medium
- 概率：high
- 摘要：SDForger、ChatTS 和 AgentSense 实际问题不同，混在一起会削弱贡献表达。
- 威胁：CLM-001、CLM-003
- 缓解：选择一个明确 paper shape，其余作为 related-work boundary。
- 相关 action：ACT-002、ACT-005
- 状态：open
- 确定性：inferred

## RSK-007：队友结果中的 utility protocol 可能使用 test-conditioned generation

- 类型：evaluation / leakage
- 严重程度：high
- 概率：high
- 摘要：`sdforger-pamap2` 多个 config 使用 `generation.prompt_split: test`，生成 metadata 保留 prompt 的 `activity_id` / `activity_name`，随后 synthetic labels 被用于 HAR utility training。
- 威胁：CLM-003
- 缓解：要求队友澄清 protocol；用 train-side prompts 重跑 synthetic augmentation utility；把 test-conditioned generation 仅作为 diagnostic。
- 相关 action：ACT-009、ACT-010
- 状态：open
- 确定性：observed

## RSK-008：队友结果中的 MDD / SD 指标定义可能不匹配

- 类型：evaluation / claim-evidence
- 严重程度：high
- 概率：high
- 摘要：`sdforger-pamap2` 的 `MDD` 实现为全局均值差，`SD` 实现为 spectral magnitude difference；如果对外称为标准 SDForger/TSGBench 指标，会造成 claim/evidence mismatch。
- 威胁：CLM-002、CLM-003
- 缓解：要求明确这些是 local proxy metrics，或实现标准定义并重跑 similarity tables。
- 相关 action：ACT-011
- 状态：open
- 确定性：observed

## RSK-009：队友 GitHub archive 缺少复现实验所需大文件

- 类型：reproducibility / artifact
- 严重程度：medium
- 概率：high
- 摘要：`sdforger-pamap2` 排除了 raw PAMAP2、train/val/test windows、generated windows、checkpoints 和 optimizer states，公开 checkout 不能独立复现 README 指标。
- 威胁：CLM-001、CLM-002、CLM-003
- 缓解：要求 artifact manifest、hashes、下载路径或最小 smoke artifact bundle。
- 相关 action：ACT-012
- 状态：open
- 确定性：observed

## RSK-010：mmfit-inference 无法访问

- 类型：collaboration / access
- 严重程度：medium
- 概率：high
- 摘要：`IkMangMok/mmfit-inference` 当前返回 GitHub `404`，无法审阅 branch、PR、代码、docs 或结果。
- 威胁：CLM-001、CLM-002
- 缓解：向队友确认 repo URL、可见性和访问权限。
- 相关 action：ACT-013
- 状态：open
- 确定性：observed

## RSK-011：mmfit-inference 生成评估链路未形成干净证据

- 类型：reproducibility / evidence
- 严重程度：high
- 概率：high
- 摘要：本地 `teamate/mmfit-inference` 有完整 pipeline 代码，但 checked-in generation logs 包含 latent shape/length 错误和 `synthetic_mmfit.jsonl` 缺失；repo 未包含 final eval report 或 artifact manifest。
- 威胁：CLM-002、CLM-003
- 缓解：要求队友重跑最小 end-to-end smoke，并归档 generated latents、decoded JSONL、eval report、commit/config 和 artifact hashes。
- 相关 action：ACT-014、ACT-015
- 状态：open
- 确定性：observed

## RSK-012：mmfit-inference Slurm script 失败后可能继续执行后续步骤

- 类型：workflow / reproducibility
- 严重程度：medium
- 概率：medium
- 摘要：`run_generate_llm_data.sh` 未启用 strict shell failure handling，generation 失败后 decode/evaluate 仍可能继续运行并产生二次错误。
- 威胁：CLM-002、CLM-003
- 缓解：添加 `set -euo pipefail`、stage-by-stage artifact checks 和清晰退出信息。
- 相关 action：ACT-016
- 状态：open
- 确定性：observed

## RSK-013：mmfit-inference autoencoder 只按 train loss 选 checkpoint

- 类型：evaluation / overfitting
- 严重程度：medium
- 概率：medium
- 摘要：autoencoder best state 由 training loss 选择，缺少 train-side validation split；latent decoder 质量可能被高估。
- 威胁：CLM-002
- 缓解：加入 autoencoder validation split 或 held-out train-side subject，单独报告 reconstruction quality。
- 相关 action：ACT-017
- 状态：open
- 确定性：observed

## RSK-014：mmfit-inference 缺少自动化测试

- 类型：testing / reproducibility
- 严重程度：medium
- 概率：high
- 摘要：repo 中没有 visible test suite；latent parsing、decode shape、split isolation、metric reporting 和 script failure behavior 都缺少回归保护。
- 威胁：CLM-002、CLM-003
- 缓解：添加最小 unit/integration tests。
- 相关 action：ACT-018
- 状态：open
- 确定性：observed

## RSK-015：Puhti 旧项目缺少 Git provenance

- 类型：reproducibility / provenance
- 严重程度：high
- 概率：high
- 摘要：`/scratch/project_2016517/panh/time-series-llm` 和 4 层内未发现 `.git`，无法从远程旧目录确认 branch、commit、dirty state 或代码版本。
- 威胁：CLM-001、CLM-002、CLM-003
- 缓解：把远程旧目录视为 artifact archive；为旧 run 建 manifest；找到对应 source commit 或迁移到 clean Git-controlled repo 后重跑。
- 相关 action：ACT-019、ACT-020、ACT-022
- 状态：open
- 确定性：observed

## RSK-016：旧 PAMAP2 baseline 不含 activity labels

- 类型：evaluation / data
- 严重程度：high
- 概率：high
- 摘要：远程旧 `pamap2_subject101_*` parquet 只保留 sensor channels，不含 `activity_id`，不能直接支持 activity-conditioned generation 或 HAR classification utility。
- 威胁：CLM-003
- 缓解：重建 label-aware PAMAP2 preprocessing，并定义 window label / split protocol。
- 相关 action：ACT-021、ACT-023
- 状态：open
- 确定性：observed

## RSK-017：旧 generated JSONL 的 value space 容易被误用

- 类型：evaluation / reproducibility
- 严重程度：high
- 概率：medium
- 摘要：旧 `final_data.jsonl` 保存的是 standardized SDForger window space，而 plot 才做 inverse transform；如果直接喂给 raw-scale classifier 会产生错误结论。
- 威胁：CLM-002、CLM-003
- 缓解：在 run manifest 中记录 `value_space`，评估/分类前强制检查 raw vs scaled contract。
- 相关 action：ACT-024
- 状态：open
- 确定性：observed

## RSK-018：multivariate auto embedding dimension 实现风险

- 类型：implementation / mechanism
- 严重程度：medium
- 概率：high
- 摘要：FPC/FICA embedding 中 `embedding_dim: auto` 在 channel loop 内被覆盖，后续 channel 可能复用第一个 channel 的维度。
- 威胁：CLM-001、CLM-002
- 缓解：修正为 per-channel local dimension，并添加 synthetic two-channel test。
- 相关 action：ACT-025
- 状态：open
- 确定性：observed

## RSK-019：多个 SDForger/vLLM run 串行在同一 Slurm job 中可能不稳

- 类型：execution / workflow
- 严重程度：medium
- 概率：medium
- 摘要：2026-05-22 clean baseline 的合并 job 在 walking 生成完成后、running 启动前后出现 `context deadline exceeded` 并以 exit code 127 失败；拆分成单独 Slurm jobs 后成功。
- 威胁：CLM-001、CLM-002
- 缓解：后续每个 activity/method variant 单独提交 Slurm job，或在脚本中加入 stage-level artifact checks 和失败恢复；不要把合并 job 失败误读为方法失败。
- 相关 action：ACT-007、ACT-026
- 状态：open
- 确定性：observed

## RSK-020：HAR utility smoke 可能高估泛化

- 类型：evaluation / generalization
- 严重程度：high
- 概率：medium
- 摘要：2026-05-23 HAR utility smoke 只覆盖 PAMAP2 subject101、`hand_acc16_x`、walking vs running 和一个简单 classifier；held-out test 是同一 subject 的后续窗口，不是 cross-subject 或 multi-activity benchmark。
- 威胁：CLM-003
- 缓解：把结果标为 provisional；下一步至少扩展到另一通道或 multichannel，并最终做 subject split / multi-subject protocol。
- 相关 action：ACT-004、ACT-007、ACT-021
- 状态：open
- 确定性：observed

## RSK-021：Synthetic inverse transform 使用 activity-specific statistics

- 类型：evaluation / preprocessing
- 严重程度：medium
- 概率：medium
- 摘要：HAR utility smoke 将 SDForger scaled synthetic windows 用对应 activity 的真实训练窗口 mean/std inverse-transform 回 raw-like space；这适合生成后处理诊断，但不是完全 class-agnostic preprocessing。
- 威胁：CLM-003
- 缓解：报告中明确 value-space contract；后续可加入完全 standardized classifier protocol 或 raw-scale generation export 作为对照。
- 相关 action：ACT-024、ACT-004
- 状态：open
- 确定性：observed

## RSK-022：Textual label conditioning 增加解析和 provenance 风险

- 类型：implementation / evaluation
- 严重程度：medium
- 概率：high
- 摘要：label conditioning v1 使用 `fim_template_textual_encoding` 后，LLM 输出中出现 malformed strings，被 parser 过滤；有效样本数下降，且远程 patch 尚未进入 clean Git-controlled repo。
- 威胁：CLM-002、CLM-003
- 缓解：把 patch 迁移到干净 repo；添加 text template / numerical-column extraction / malformed output tests；报告 generated count、dropped count 和 parsing examples。
- 相关 action：ACT-027、ACT-028
- 状态：open
- 确定性：observed

## RSK-023：Unified conditional generation has severe latent/value outliers

- 类型：method / evaluation
- 严重程度：high
- 概率：high
- 摘要：2026-05-23 unified label conditioning smoke 中，label controllability positive，但 decoded synthetic windows 的幅度远超真实 standardized space，walking abs max 3761.8、running abs max 1572.1，PSD peak 偏到 2.3333Hz。
- 威胁：CLM-002、CLM-003
- 缓解：post-hoc latent constraints 已证明 value explosion 可被训练分布约束缓解；下一步需要 generation-time validity check / rejection sampling，并报告 accepted/rejected counts。在质量稳定前不要把 unified conditional 输出用于 HAR augmentation final claim。
- 相关 action：ACT-029、ACT-030
- 状态：partially mitigated
- 确定性：observed

## RSK-024：Post-hoc clipping 可能制造人工修复效果

- 类型：method / claim-evidence
- 严重程度：medium
- 概率：medium
- 摘要：当前 latent/value constraint 是对已生成 embeddings 的 post-hoc clipping 或 filtering；它能诊断 failure source，但不能证明模型本身学会了稳定生成。
- 威胁：CLM-002、CLM-003
- 缓解：把 post-hoc clipping 明确标为 diagnostic；已测试 strict reject 和 soft repair。strict reject 样本数过少，soft repair 可作为 generation-time validity 候选，但需要接入实际 resampling loop 并报告 rejection/repair rate。
- 相关 action：ACT-030
- 状态：partially mitigated
- 确定性：inferred from EVD-015

## RSK-025：Channel amplitude mismatch may destabilize unified embedding

- 类型：data / method
- 严重程度：medium
- 概率：medium
- 摘要：导师反馈指出不同 sensor channel 可能存在幅度尺度不一致；在 unified 或 multichannel setting 下，如果 embed 前没有合适 normalization，joint FICA/LLM 可能优先学习尺度差异而不是 activity dynamics。
- 威胁：CLM-002、CLM-003
- 缓解：设计 pre-embedding normalization ablation；所有 normalization statistics 必须只从 train split 估计，避免 test leakage。
- 相关 action：ACT-031
- 状态：open
- 确定性：advisor-stated / inferred

## RSK-026：Current evaluation is not unseen-subject generalization

- 类型：evaluation / generalization
- 严重程度：high
- 概率：high
- 摘要：当前 HAR utility 和 controllability smoke 主要基于 subject101 的 train/test windows；2026-06-07 已补 subject106/108 held-out reference TSG diagnostic，但这仍不是 held-out HAR classifier utility。
- 威胁：CLM-003
- 缓解：已完成 train subjects 101/102/105 -> held-out reference subjects 106/108 的 TSG diagnostic；下一步补 held-out HAR utility。
- 相关 action：ACT-032
- 状态：partially mitigated
- 确定性：observed / advisor-concern

## RSK-027：Latent clipping may over-smooth generated dynamics

- 类型：method / evaluation
- 严重程度：medium
- 概率：medium
- 摘要：`clip_p05_p95` 会把 out-of-range latent values 截断到训练分布分位边界；这能修复 value explosion，但也可能压缩 amplitude、降低高频细节，导致图像看起来更平滑。
- 威胁：CLM-002、CLM-003
- 缓解：除 abs max 外报告 std、PSD power、derivative energy / roughness、ACF peak score 和 downstream utility；不要只凭视觉平滑判断质量提高。
- 相关 action：ACT-033
- 状态：open
- 确定性：inferred

## RSK-028：Model-space 与 raw-like evaluation 混用

- 类型：evaluation / claim-evidence
- 严重程度：high
- 概率：high
- 摘要：已有报告同时包含 SDForger standardized space、normalization-specific model space 和 raw-like inverse-transformed space 的指标；如果不区分，overlay、TSGBench-style metrics、HAR utility 和 label controllability 会被错误比较。
- 威胁：CLM-002、CLM-003
- 缓解：后续主结果只使用 `docs/experiments/unified-evaluation-protocol-20260607.md` 定义的 raw-like evaluation；model-space metrics 只作为 diagnostic，不进入主表。
- 相关 action：ACT-003、ACT-024、ACT-034
- 状态：mitigated-for-current-results
- 确定性：observed

## RSK-029：Representative overlay 可能造成过度乐观解释

- 类型：evaluation / visualization
- 严重程度：medium
- 概率：medium
- 摘要：overlay 图为了可读性会选 representative synthetic sample 并做 lag alignment，不能当作平均误差图或 paired prediction 图。
- 威胁：CLM-002、CLM-003
- 缓解：slides/report 中明确写 qualitative waveform comparison；主判断依赖统一表中的 amplitude、ACF/PSD、TSGBench-style、HAR 和 label metrics。
- 相关 action：ACT-034
- 状态：open
- 确定性：observed
