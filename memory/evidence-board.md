# Evidence Board

最后更新：2026-06-07

## EVD-001：旧 SDForger/PAMAP2 run notes

- 类型：experiment
- 状态：needs-verification
- 摘要：`legacy/old-project-files/TS LLM.md` 记录了旧 runs，包括 `smoke`、`formal`、`var90`、`thr095`、`train120k`、`train120k_win400` 和 `subject102_train120k`，并包含 provisional sample counts 和定性观察。
- Source paths：
  - `legacy/old-project-files/TS LLM.md`
  - `legacy/old-project-files/TS LLM.md` 中列出的远程路径
- 支撑：只有核验后才可能支撑 CLM-001、CLM-002
- 局限：远程 outputs、configs、code commits、logs、plots 和 reports 当前不可访问。
- 确定性：needs-verification

## EVD-002：本地 seed paper set

- 类型：citation
- 状态：planned
- 摘要：本地有 SDForger、ChatTS 和 AgentSense 的 PDF 与提取文本，可作为 literature sprint 的 seed papers。
- Source paths：
  - `legacy/old-project-files/Forging Time Series Synthetic Data.pdf`
  - `legacy/old-project-files/ChatTS Time Series Alignment.pdf`
  - `legacy/old-project-files/AgentSense Virtual Sensor Data.pdf`
  - `legacy/old-project-files/tmp_forging.txt`
  - `legacy/old-project-files/tmp_chatts.txt`
  - `legacy/old-project-files/tmp_agentsense.txt`
- 支撑：literature positioning 和 baseline planning，不支撑 empirical claims。
- 确定性：observed

## EVD-003：未来最小证据包

- 类型：experiment / analysis
- 状态：planned
- 摘要：最小证据包应包括一个已核验的 SDForger/PAMAP2 rerun、一个明确 baseline、per-channel stats、ACF/PSD、diversity 和小规模 downstream HAR utility check。
- 支撑：CLM-001、CLM-002、CLM-003
- 确定性：inferred

## EVD-004：队友 sdforger-pamap2 GitHub archive

- 类型：code / experiment archive
- 状态：provisional
- 摘要：`SJYANG555/sdforger-pamap2` main snapshot 包含 PAMAP2 + SDForger-style pipeline、configs、Slurm scripts、docs、logs、CSV summaries 和 README metric tables。
- Source paths：
  - `reviews/teammate-repos/SJYANG555-sdforger-pamap2-90884c1/`
  - `docs/reviews/2026-05-18-teammate-review.md`
- 支撑：可作为 teammate implementation 和 result-archive 的存在性证据；暂不支撑 final empirical claim。
- 局限：缺少 raw data、window tensors、generated windows、checkpoints；utility protocol 可能 test-conditioned；`MDD` / `SD` metric definitions 需要修正或重命名。
- 确定性：provisional

## EVD-005：队友 mmfit-inference 本地 archive

- 类型：code / experiment archive
- 状态：provisional
- 摘要：`teamate/mmfit-inference` 包含 MM-Fit latent autoencoder、latent SFT、decoder、raw classifier、evaluation pipeline、README、Slurm scripts 和部分 Slurm logs。
- Source paths：
  - `teamate/mmfit-inference/`
  - `docs/reviews/2026-05-18-teammate-review.md`
- 支撑：可作为 teammate method-extension implementation 的存在性证据；暂不支撑 final generated-data quality claim。
- 局限：generation logs 显示 latent shape/length 错误和 decoded JSONL 缺失；repo 未包含 final generated-data eval report、generated JSONL、decoded JSONL、checkpoints 或 artifact manifest。
- 确定性：provisional

## EVD-006：Puhti 旧 SDForger/PAMAP2 artifact archive

- 类型：code / experiment archive
- 状态：provisional
- 摘要：只读观察到 `/scratch/project_2016517/panh/time-series-llm/fms-dgt` 包含 SDForger-style time-series builder、PAMAP2 task yaml、prepared parquet、Slurm scripts、logs、reports、plots、generated JSONL 和 model checkpoints。
- Source paths：
  - `/scratch/project_2016517/panh/time-series-llm`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt`
  - `docs/ops/2026-05-21-puhti-time-series-llm-inventory.md`
- 支撑：可支持“旧 SDForger/PAMAP2 pipeline 和 artifacts 存在”的事实；暂不支撑生成质量或 HAR utility claim。
- 局限：远程旧目录未发现 `.git`；旧 metrics/reports 尚未完成 run-level provenance 核验；PAMAP2 默认预处理可能丢弃 `activity_id`，对 HAR utility 不足。
- 确定性：observed for file existence / provisional for research interpretation

## EVD-007：Puhti 旧代码审阅

- 类型：code review
- 状态：available
- 摘要：只读审阅确认旧 SDForger/PAMAP2 pipeline 工程方向可用，但存在 label 缺失、scaled output contract、multivariate auto embedding dimension 和测试覆盖风险。
- Source paths：
  - `docs/reviews/2026-05-22-puhti-old-code-review.md`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/scripts/prepare_pamap2_subject101.py`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/fms_dgt/public/databuilders/time_series/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/scripts/evaluate_sdforger_paper_metrics.py`
- 支撑：支持“旧代码可作为 recovered baseline candidate”的判断；暂不支持 final empirical claim。
- 局限：未运行新实验；未修改远程代码；未执行完整 test suite。
- 确定性：observed

## EVD-008：PAMAP2 subject101 activity periodicity check

- 类型：analysis
- 状态：available
- 摘要：对 PAMAP2 subject101 的 walking、running、cycling 在 `hand_acc16_x` / `ankle_acc16_x` 上做 raw segment、ACF、PSD/FFT 周期性检查；walking 和 running 更适合作为第一批 SDForger univariate baseline verification activity。
- Source paths：
  - `docs/experiments/baseline-verification-20260522.md`
  - `scripts/pamap2_periodicity_check.py`
  - `outputs/baseline-verification-20260522/periodicity_summary.md`
  - `outputs/baseline-verification-20260522/*.png`
- 支撑：支持选择 walking/running + `hand_acc16_x` 作为 clean activity-specific univariate baseline 的下一步。
- 局限：尚未重新跑 activity-specific SDForger generation；当前旧 baseline 仍是 mixed non-zero-activity。
- 确定性：observed

## EVD-009：PAMAP2 subject101 clean activity-specific SDForger univariate baseline

- 类型：experiment
- 状态：supported for baseline verification / provisional for research claim
- 摘要：从原始 PAMAP2 `subject101.dat` 过滤 walking-only 和 running-only 的 `hand_acc16_x`，用 SDForger univariate + FICA setting 重新生成 synthetic windows 并重算 TSGBench-style metrics。
- Source paths：
  - `docs/experiments/clean-univariate-baseline-results-20260522.md`
  - `outputs/clean-univariate-baseline-20260522/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_walking_hand_acc16_x_univariate_baseline/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_running_hand_acc16_x_univariate_baseline/`
- 关键观察：
  - walking：period 58，31 windows，FICA dim 4，variance retained 0.8306，generated windows 130。
  - running：period 82，31 windows，FICA dim 2，variance retained 0.7010，generated windows 98。
  - walking metrics：MDD 0.266287，ACD 0.165201，SD 0.797960，KD 1.667799，ED 21.279075，DTW 237.760946。
  - running metrics：MDD 0.296503，ACD 0.497915，SD 0.612561，KD 0.444503，ED 17.078956，DTW 111.087924。
- 支撑：支持“clean activity-specific SDForger univariate baseline 可以在 PAMAP2 subject101 walking/running 上跑通”的 baseline claim。
- 局限：不支撑 HAR utility claim；`final_data.jsonl` 是 scaled/standardized window space；尚未人工审核 overlay 图，也未做 classification utility。
- 确定性：observed

## EVD-010：Clean baseline ACF/PSD diagnostic

- 类型：analysis / diagnostic evaluation
- 状态：available
- 摘要：对 EVD-009 的 walking/running clean baseline 做真实窗口 vs synthetic 窗口 ACF/PSD 对比，检查生成样本是否保留主要周期结构。
- Source paths：
  - `docs/experiments/acf-psd-comparison-20260522.md`
  - `scripts/compare_sdforger_acf_psd.py`
  - `outputs/acf-psd-comparison-20260522/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/baseline_verification_20260522/*_acf_psd_comparison.*`
- 关键观察：
  - running：real ACF peak lag 81，synthetic ACF peak lag 82；real/synthetic PSD peak both 1.3333 Hz。
  - walking：real ACF peak lag 116，synthetic ACF peak lag 58；real/synthetic PSD peak both 1.6667 Hz。
- 支撑：支持“clean SDForger univariate baseline 保留 dominant periodic structure，尤其 running 明显”的 diagnostic claim。
- 局限：不支撑 downstream HAR utility；只覆盖 subject101、单通道、两个 activity；mean ACF/PSD 会隐藏 individual sample failures。
- 确定性：observed

## EVD-011：Synthetic sample quality stratification

- 类型：analysis / diagnostic evaluation
- 状态：available
- 摘要：对 walking/running clean baseline 的 synthetic windows 按 ACF lag、PSD peak、std 和 abs max 分为 good / borderline / bad，并生成 best/worst 示例图。
- Source paths：
  - `docs/experiments/sample-stratification-20260522.md`
  - `scripts/stratify_sdforger_samples.py`
  - `outputs/sample-stratification-20260522/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/baseline_verification_20260522/sample_stratification/`
- 关键观察：
  - walking：good 57，borderline 64，bad 9。
  - running：good 42，borderline 14，bad 42。
- 支撑：支持“synthetic samples quality is uneven”的 diagnostic claim，并为 good-only HAR utility smoke 提供 sample indices。
- 局限：thresholds are heuristic；labels are diagnostic, not final benchmark labels。
- 确定性：observed

## EVD-012：Minimal HAR utility smoke on walking vs running

- 类型：experiment / downstream utility
- 状态：provisional
- 摘要：在 PAMAP2 subject101 `hand_acc16_x` 上做 walking vs running 分类 smoke，比较 majority、real-only、synthetic-only、real+synthetic 和 good-only synthetic 条件。
- Source paths：
  - `docs/experiments/har-utility-smoke-plan-20260523.md`
  - `docs/experiments/har-utility-smoke-results-20260523.md`
  - `scripts/evaluate_har_utility_smoke.py`
  - `outputs/har-utility-smoke-20260523/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/baseline_verification_20260523/har_utility_smoke/`
- 关键观察：
  - majority accuracy 0.5135 / balanced accuracy 0.5000。
  - real-only accuracy 0.6126 / balanced accuracy 0.6087。
  - synthetic-only-all accuracy 0.7027 / balanced accuracy 0.6979。
  - real+synthetic-all accuracy 0.7117 / balanced accuracy 0.7071。
  - synthetic-only-good accuracy 0.7207 / balanced accuracy 0.7139。
  - real+synthetic-good accuracy 0.6396 / balanced accuracy 0.6374。
- 支撑：支持“当前 clean SDForger baseline synthetic windows preserve some walking/running task-discriminative information”的 preliminary utility claim。
- 局限：one subject, one channel, two activities, one classifier, no seeds, no cross-subject split；synthetic inverse transform uses activity-specific train-window statistics。
- 确定性：observed / provisional interpretation

## EVD-013：Label conditioning v1 on PAMAP2 subject101 walking/running

- 类型：experiment / method extension
- 状态：provisional
- 摘要：在 recovered SDForger/PAMAP2 pipeline 中把 activity label 作为 textual condition 加入 prompt，分别重跑 walking/running + `hand_acc16_x` univariate generation，并复用 TSGBench-style metrics、ACF/PSD、sample stratification 和 HAR utility smoke。
- Source paths：
  - `docs/experiments/label-conditioning-v1-results-20260523.md`
  - `outputs/label-conditioning-20260523/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/tasks/public/time_series/pamap2_subject101_walking_hand_acc16_x_label_conditioned_univariate/task.yaml`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/tasks/public/time_series/pamap2_subject101_running_hand_acc16_x_label_conditioned_univariate/task.yaml`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/label_conditioning_20260523/`
- 关键观察：
  - walking：生成 72 个 synthetic windows；good 33、borderline 33、bad 6。
  - running：生成 90 个 synthetic windows；good 41、borderline 19、bad 30。
  - ACF/PSD：running synthetic ACF lag 82 vs real 81，PSD both 1.3333 Hz；walking synthetic ACF lag 58 vs real 116，PSD both 1.6667 Hz。
  - HAR utility smoke：synthetic-only-all accuracy 0.7387 / balanced accuracy 0.7329；real+synthetic-all accuracy 0.7117 / balanced accuracy 0.7086。
- 支撑：支持“activity label textual conditioning 在 recovered pipeline 中可运行，并保留 walking/running downstream utility signal”的 preliminary claim。
- 局限：远程 patch 尚未进入 clean Git repo；textual template 产生 malformed outputs；仍是 activity-specific task 加 label，不是 unified multi-class conditional generator；one subject / one channel / two activities。
- 确定性：observed / provisional interpretation

## EVD-014：Unified label conditioning controllability diagnostic

- 类型：experiment / method diagnostic
- 状态：observed / mixed result
- 摘要：训练一个 unified walking/running conditional generator：activity-specific windows 独立切分后合并，joint FICA embedding，并加入 window-level `data` label。模型能按 requested label 产生可分类的样本，但生成质量因 latent/value outliers 明显失败。
- Source paths：
  - `docs/experiments/unified-label-conditioning-plan-20260523.md`
  - `docs/experiments/unified-label-conditioning-results-20260523.md`
  - `scripts/run_unified_label_conditioning.py`
  - `scripts/evaluate_label_controllability.py`
  - `outputs/unified-label-conditioning-20260523/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/unified_label_conditioning_20260523/`
- 关键观察：
  - 生成数：walking 72，running 79。
  - Label controllability：accuracy 0.8212，balanced accuracy 0.8205，macro F1 0.8207。
  - Per-label controllability：walking 0.8056，running 0.8354。
  - Quality failure：walking synthetic abs max 3761.8，running synthetic abs max 1572.1；synthetic PSD peak both 2.3333Hz。
  - HAR utility：synthetic-only-all accuracy 0.4955；real+synthetic-all accuracy 0.5315；real+synthetic-good accuracy 0.6757 with only 26 good synthetic samples.
- 支撑：支持“unified activity label conditioning has controllability signal”的 diagnostic claim。
- 局限：不支持“unified label conditioning improves generation quality or HAR utility”；joint FICA produced unstable latent/value scale; FastICA convergence warning observed.
- 确定性：observed

## EVD-015：Latent/value constraint diagnostic for unified label conditioning

- 类型：experiment / method diagnostic
- 状态：observed / promising diagnostic
- 摘要：对 EVD-014 的 unified label-conditioned generated embeddings 施加训练 latent 分布约束，检查是否能修复 value explosion 并保留 activity label controllability。
- Source paths：
  - `docs/experiments/latent-constraint-plan-20260523.md`
  - `docs/experiments/latent-constraint-results-20260523.md`
  - `scripts/apply_unified_latent_constraints.py`
  - `outputs/latent-constraints-20260523/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x_constraints/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/latent_constraints_20260523/`
- 关键观察：
  - `clip_p05_p95` 将 decoded abs max 控制到 walking 3.2372、running 2.6524；raw unified 分别为 3761.8 和 1572.1。
  - `clip_p05_p95` label controllability accuracy 0.9868；walking requested accuracy 1.0000，running requested accuracy 0.9747。
  - `clip_p05_p95` synthetic-only-all HAR accuracy 0.6937，高于 real-only 0.6126；但 real+synthetic-all accuracy 0.5946，augmentation 仍不稳定。
  - ACF/PSD：running synthetic ACF lag 80 vs real 81、PSD peak both 1.3333 Hz；walking synthetic ACF lag 80 vs real 116、PSD peak 1.3333 vs real 1.6667 Hz。
- 支撑：支持“unified label conditioning 的主要 failure source 包含 latent/value outliers；训练分布约束可恢复数值稳定性并保留 label signal”的 diagnostic claim。
- 局限：post-hoc constraint 不是 generation-time method；只覆盖 subject101、单通道、两个 activities；walking rhythm 尚未恢复。
- 确定性：observed / provisional interpretation

## EVD-016：Normalization ablation diagnostic for unified label conditioning

- 类型：experiment / diagnostic evaluation
- 状态：observed / diagnostic-only
- 摘要：对 unified walking/running conditional generator 测试 4 种 FICA input-space normalization：`current_activity_window_zscore`、`joint_window_zscore`、`global_series_zscore`、`activity_series_zscore`。结果显示 normalization-only 没有形成稳定主方案；`global_series_zscore` 在 running 的 model-space rhythm / DTW 上相对最好，但仍存在严重 value-scale outlier。
- Source paths：
  - `docs/experiments/normalization-ablation-plan-20260607.md`
  - `docs/experiments/normalization-ablation-results-20260607.md`
  - `scripts/run_unified_label_conditioning_normalization.py`
  - `scripts/evaluate_normalization_ablation_outputs.py`
  - `scripts/evaluate_normalization_tsgbench_style.py`
  - `outputs/normalization-ablation-20260607/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_norm_ablation_*`
- 关键观察：
  - `global_series_zscore` running ACF lag diff 为 1、PSD diff 为 0、DTW 为 536.134086，是 normalization variants 中 running 最不差的设置。
  - 但 `global_series_zscore` running amplitude ratio 仍约 177.63，说明 model-space value outlier 未解决。
  - `joint_window_zscore` running synthetic abs max 达 187737.53，明显不可用。
- 支撑：支持“normalization affects the failure mode but does not by itself solve unified conditional generation”的 diagnostic claim。
- 局限：当前 metrics 在 model-space 中计算；不支撑 raw sensor generation quality 主结论，需按 EVD-017 协议补 raw-like evaluation。
- 确定性：observed / diagnostic-only

## EVD-017：Unified evaluation protocol

- 类型：evaluation protocol
- 状态：active
- 摘要：已确定后续主评估统一使用 `real raw windows` vs `inverse-normalized synthetic raw-like windows`；model-space metrics 只作 debugging / diagnostic。
- Source paths：
  - `docs/experiments/unified-evaluation-protocol-20260607.md`
- 主指标：
  - value validity：amplitude ratio
  - temporal/frequency：ACF lag diff、PSD Hz diff
  - TSGBench-style：MDD、ACD、SD、KD、ED、DTW
  - task utility：HAR real+synthetic accuracy
  - label control：requested-label accuracy，仅 conditional settings
- 支撑：支持后续实验报告和 slides 的统一 evaluation contract。
- 局限：关键 setting 已按 raw-like space 重算一版，但 DTW 尚未补；未来新 setting 仍必须沿用该 contract。
- 确定性：decision / active protocol

## EVD-018：Unified raw-like evaluation of key settings

- 类型：experiment / evaluation
- 状态：observed / provisional interpretation
- 摘要：已将 clean unconditioned、raw unified label-conditioned、`clip_p05_p95` 和 `global_series_zscore` 放入统一 raw-like sensor-space 主评估。结果显示 clean baseline 仍最稳；raw unified 有 label signal 但 value explosion；`clip_p05_p95` 修复 value explosion 并保留 label controllability，但 HAR augmentation 仍不稳定；global z-score 在 raw-like 下不可作为当前改进方向。
- Source paths：
  - `docs/experiments/unified-raw-like-evaluation-results-20260607.md`
  - `docs/reviews/2026-06-07-normalization-evaluation-code-review.md`
  - `scripts/evaluate_unified_raw_like_metrics.py`
  - `outputs/unified-raw-like-evaluation-20260607/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_unified_raw_like_eval_20260607/`
- 关键观察：
  - clean unconditioned running：amplitude ratio 1.0314，ACF lag diff 0，PSD diff 0。
  - raw unified label-conditioned：label accuracy 0.8212，但 walking/running amplitude ratio 分别约 1081x / 408x。
  - `clip_p05_p95`：walking/running amplitude ratio 分别约 1.22x / 1.24x，label accuracy 0.9868；running ACF/PSD 匹配，walking ACF 仍不匹配。
  - HAR utility：clean `real+synthetic-all` accuracy 0.7117；`clip_p05_p95` 为 0.5946；global z-score 为 0.2793。
- 支撑：支持“post-hoc latent clipping can repair value-scale failure and preserve requested-label controllability”的 diagnostic claim；支持“normalization-only should not be the next main direction”的 decision。
- 局限：本次使用 `--skip-dtw`，DTW 尚未补；仍是 subject101 within-subject smoke；`clip_p05_p95` 是 post-hoc diagnostic，不是 generation-time method。
- 确定性：observed / provisional interpretation

## EVD-019：Multi-subject unified conditioning smoke

- 类型：experiment / diagnostic evaluation
- 状态：observed / diagnostic-only
- 摘要：已将 unified walking/running conditional SDForger 从 subject101 扩展到 subject101/102/105，检查失败是否主要来自单 subject 数据量不足。结果显示 multi-subject raw-unified 仍然 ED/DTW/KD 爆炸；`clip_p05_p95` 仍能显著修复 ED/DTW/KD，但 ACD 仍偏高。
- Source paths：
  - `docs/experiments/multisubject-unified-conditioning-plan-20260607.md`
  - `docs/experiments/multisubject-unified-conditioning-results-20260607.md`
  - `outputs/multisubject-tsgbench-20260607/`
  - `scripts/prepare_pamap2_multisubject_activity.py`
  - `scripts/rerun_multisubject_tsgbench_table.py`
  - `scripts/evaluate_sdforger_paper_metrics.py`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/multisubject_tsgbench_20260607_fixed/`
- 关键观察：
  - 实际 subjects 为 101/102/105；subject103 因缺少 running 未使用。
  - 训练 windows：walking 54，running 62，combined 116；FICA dim 8。
  - Accepted synthetic windows：walking 51，running 58，说明 malformed latent text 仍导致有效样本数偏低。
  - `multi_raw_unified`：walking DTW 9920.940、running DTW 7943.930，仍然失败。
  - `multi_clip_p05_p95`：walking DTW 166.065、running DTW 147.235，KD 分别为 0.746 / 0.281，数值稳定性明显恢复。
  - ACD 仍偏高：walking 3.117，running 1.611，说明 rhythm/autocorrelation 未完全解决。
- 支撑：支持“单纯增加 subject/data 不是 unified conditional generation 失败的主要解法；latent validity / decoding stability 仍是下一步主线”的 diagnostic claim。
- 局限：评估仍是 train-subject diagnostic，不是 held-out subject generalization；`clip_p05_p95` 仍是 post-hoc constraint；ED/DTW 因 synthetic 数量不足使用 `min(real, synthetic)` paired subset。
- 确定性：observed / provisional interpretation

## EVD-020：Held-out subject TSG diagnostic

- 类型：experiment / diagnostic evaluation
- 状态：observed / provisional interpretation
- 摘要：用 subject101/102/105 训练并生成的 multi-subject unified synthetic windows，对 subject106/108 的真实 walking/running windows 做 held-out reference TSG-style evaluation。结果显示 raw-unified 在 unseen subjects 上仍失败；`clip_p05_p95` 在 ED/DTW/KD 上仍接近 train-reference clip。
- Source paths：
  - `docs/experiments/unseen-subject-evaluation-plan-20260607.md`
  - `docs/experiments/unseen-subject-evaluation-results-20260607.md`
  - `scripts/rerun_unseen_subject_tsgbench_table.py`
  - `outputs/unseen-subject-tsgbench-20260607/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/unseen_subject_tsgbench_20260607/`
- 关键观察：
  - Held-out subjects：106/108；每个 subject/activity 5000 rows。
  - Held-out real windows：walking 45，running 44。
  - Held-out estimated period：walking 110，running 76。
  - `unseen_raw_unified`：walking DTW 11160.400，running DTW 9567.380，仍然失败。
  - `unseen_clip_p05_p95`：walking ED 21.496 / DTW 151.102；running ED 20.685 / DTW 146.689。
  - ACD 仍不稳：walking 1.992，running 2.228。
- 支撑：支持“multi-subject + clipped latent outputs retain preliminary similarity to held-out subject motion windows under ED/DTW/KD”的 diagnostic claim。
- 局限：这不是 held-out HAR utility；synthetic 仍来自 post-hoc clip；reference-space TSG 不是严格 train/test classifier evaluation；ACD/rhythm 仍需改进。
- 确定性：observed / provisional interpretation

## EVD-021：Generation-time-style latent validity diagnostic

- 类型：experiment / method diagnostic
- 状态：observed / diagnostic-only
- 摘要：在 parser-accepted generated latent rows 上测试 `strict_reject_p05_p95` 和 `soft_repair_p05_p95_minmax`。strict reject 过于苛刻，walking/running 只保留 14/10 个样本；soft repair 保留 27/34 个样本，ED/DTW 接近 post-hoc clip，但没有明显全面优于简单 clip。
- Source paths：
  - `docs/experiments/generation-time-validity-plan-20260608.md`
  - `docs/experiments/generation-time-validity-results-20260608.md`
  - `scripts/apply_generation_time_latent_validity.py`
  - `scripts/rerun_validity_variants_tsgbench_table.py`
  - `outputs/generation-time-validity-20260608/`
  - `/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/validity_variants_tsgbench_20260608/`
- 关键观察：
  - strict reject：walking 14/51 accepted，running 10/58 accepted。
  - soft repair：walking 14 clean + 13 repaired + 24 rejected；running 10 clean + 24 repaired + 24 rejected。
  - train-reference soft repair：walking ED 23.195 / DTW 205.601；running ED 19.771 / DTW 141.353。
  - held-out-reference soft repair：walking ED 18.915 / DTW 140.000；running ED 20.320 / DTW 157.195。
  - post-hoc clip remains simpler and retains more samples; soft repair's main advantage is explicit validity accounting.
- 支撑：支持“strict p05-p95 rejection alone is too conservative; soft repair is a plausible generation-time validity candidate but not yet superior to post-hoc clip”的 diagnostic claim。
- 局限：没有重新调用 LLM resampling；raw malformed text outputs 已被 upstream parser 过滤，不能统计完整 malformed rejection rate；样本数仍少，不能作为 final method claim。
- 确定性：observed / provisional interpretation
