# Idea Reset：LLM-Enabled Synthetic Sensor Time Series

最后更新：2026-05-18

## 还需要你提供的材料

有的话请补充：

- 当前粗略 idea，用一段话描述即可
- 这个目录之外的旧笔记
- 论文列表、BibTeX 或必须读的 PDF
- 相关关键词和目标 community
- 除 `legacy/old-project-files/puhti-time-series-llm/` 之外的本地残留代码路径
- 远程路径、job 名、run 命名规则和预期输出
- 导师/课程反馈、评分标准和最终 deliverable 要求

## 当前恢复出的项目形状

工作主题：使用 LLM-enabled 或 LLM-adjacent 的生成方法，为 HAR 生成合成传感器时间序列数据。

当前候选主线：把 `SDForger` 适配到 `PAMAP2` wearable sensor 数据，并用统计、时间结构、多样性和下游 HAR utility 指标评估生成样本。

已观察到的 seed sources：

- `legacy/old-project-files/Forging Time Series Synthetic Data.pdf`
- `legacy/old-project-files/ChatTS Time Series Alignment.pdf`
- `legacy/old-project-files/AgentSense Virtual Sensor Data.pdf`
- `legacy/old-project-files/TS LLM.md`
- `legacy/old-project-files/SESSION_HANDOFF.md`

已观察到的残留代码：

- `legacy/old-project-files/puhti-time-series-llm/`

信任策略：旧结果都是 `needs-verification`；远程输出当前被集群维护阻塞；本地代码不完整，不能直接作为证据。

## 候选研究问题

### RQ-A：适配 / 复现

SDForger 能否从原始示例适配到 PAMAP2 多变量 wearable HAR sensor windows，并提供可复现的 preprocessing、configuration 和 output generation？

适用场景：课程项目或工程复现报告。

最小 claim：

> 一个已核验的 SDForger pipeline 可以生成 PAMAP2 acceleration windows，并在一个小型 metric suite 下与真实窗口进行非平凡相似性比较。

风险：如果评估不够严谨，会被看成 reproduction + domain adaptation。

### RQ-B：经验评估

SDForger 在生成 wearable HAR sensor 数据时，保留了哪些质量维度，又在哪些维度失败？

适用场景：希望项目变成更严谨的 empirical analysis。

最小 claim：

> 在受控 PAMAP2 子集上，SDForger 保留了一部分低阶统计特征，但在 amplitude、diversity、temporal structure 或 downstream utility 上存在可识别局限。

风险：negative/mixed results 可以接受，但分析必须系统，baseline 必须公平。

### RQ-C：HAR augmentation utility

SDForger 生成的数据能否在 low-data HAR classification 中作为 augmentation 提升效果？

适用场景：项目需要 application-facing contribution。

最小 claim：

> 在固定 subject split 和 low-data regime 下，加入 SDForger 生成样本，相比 real-only baseline 和简单 augmentation baseline，可以改善或保持 HAR classifier performance。

风险：需要标签或 label-conditioning 策略；旧笔记显示当前 pipeline 曾丢弃 `activity_id`，因此未必能支撑这个 claim。

### RQ-D：方法扩展

能否扩展 SDForger，使其支持 activity labels 或更多 sensor channels 的条件生成？

适用场景：只有在 RQ-A/RQ-B 核验后才考虑。

最小 claim：

> label-aware 或 channel-aware 扩展，相比已核验 SDForger baseline，改善一个明确测量的 failure mode。

风险：当前 reset 阶段过贵、过早、规格也还不清楚。

## Research-Idea Validation

Decision：`revise`

Confidence：low-to-medium

### FIVE+C 评估

| 维度 | 评级 | 当前依据 | 风险 |
|---|---|---|---|
| Framing | medium | 从旧笔记中可恢复 SDForger/PAMAP2 主线。 | 必须选择 reproduction、empirical analysis、augmentation 或 method extension。 |
| Importance | medium | Synthetic sensor data 与 HAR 数据稀缺和隐私问题相关。 | 重要性取决于目标 audience，以及是否证明 utility。 |
| Validity | medium | SDForger 的机制对 multivariate windows 有一定合理性。 | HAR motion dynamics、label semantics 和跨通道结构可能保留不好。 |
| Evidence | weak/unknown | 旧笔记有 run 名和 provisional counts。 | artifacts 不可访问，评估也不完整。 |
| Execution | medium but blocked | 本地有代码痕迹和远程路径记录。 | 集群不可用；本地代码不可信。 |
| Competition | unknown | seed papers 包括 SDForger、ChatTS、AgentSense。 | closest work 和 baseline expectations 尚未梳理。 |

### 为什么是 revise

方向有潜力，但当前还不能 `pursue`：novelty、baselines、evidence 和 paper shape 都没定。旧结果未来可能有用，但必须先核验。

### 最小可验证 claim

下一阶段先使用这个更保守的 claim：

> 一个干净、可核验的 SDForger pipeline 可以生成 PAMAP2 wearable acceleration windows；这些窗口的基础分布和时间结构可以与真实窗口对比，从而判断该方法是否能作为 HAR synthetic-data baseline。

这个 claim 故意弱于“提升 HAR”或“优于 generative baselines”。等远程访问恢复后，它可以较快被证实或证伪。

## Literature Review Sprint Plan

Sprint 模式：offline-first novelty/baseline/positioning sprint。

由于当前 reset 明确要求离线，这里先是 provisional plan。recent/concurrent work 需要之后用 primary sources 核验，比如 arXiv、OpenReview、proceedings pages、DBLP、Semantic Scholar、PMLR、ACL Anthology、CVF、ACM 或 IEEE。

### Sprint 问题

对于 HAR synthetic sensor time-series generation，判断 SDForger-style 的 LLM/text-encoded generation pipeline 相比 specialized time-series generators、LLM-time-series methods 和 HAR synthetic-data/simulation methods 是否有 novelty 和 utility，并识别必须的 baselines 和证据。

### Query families

| Family | Broad query | Narrow query |
|---|---|---|
| SDForger / LLM generation | LLM synthetic time series generation | SDForger PAMAP2 HAR synthetic data |
| Time-series generators | time series synthetic data generation benchmark | TimeGAN TimeVAE TimeVQVAE diffusion time series generation |
| Evaluation | time series generation evaluation fidelity utility diversity | TSGBench synthetic time series metrics ACF PSD downstream utility |
| HAR synthetic data | synthetic data human activity recognition wearable sensors | PAMAP2 synthetic IMU augmentation HAR |
| LLM-time-series | large language models time series generation forecasting | LLMTime Time-LLM GPT4TS Chronos synthetic data |
| Sensor simulation | virtual sensor data generation HAR smart home | AgentSense VirtualHome ambient sensor data HAR |

### 需要梳理的 paper families

| Family | 核心机制 | 代表 seeds | 对项目的影响 |
|---|---|---|---|
| LLM/text-encoded TS generation | 把时间序列表示编码成 text/embedding，再用 LLM 生成。 | SDForger, LLMTime-like work | 定义 closest method boundary 和 adaptation claim。 |
| Specialized TS generators | GAN/VAE/VQ/diffusion 直接在时间序列上训练。 | TimeGAN, TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4 | 可能是必须 baseline 或 related-work anchors。 |
| TS foundation/forecasting models | 用 pretrained/foundation TS models 做 forecasting/classification。 | Chronos, TimeGPT, Time-LLM, GPT4TS | 避免过度声称 “LLMs for time series” novelty。 |
| HAR synthetic data and augmentation | 生成或模拟 sensor data 改善 HAR。 | AgentSense 和 wearable HAR synthetic-data papers | 应用层面的 closest competition。 |
| Evaluation benchmarks | 标准化 similarity、fidelity、utility、diversity metrics。 | TSGBench 等 | 决定 metric suite 和 reviewer expectations。 |

### Reading priority

| Source | Priority | 要回答的问题 | 影响的决策 |
|---|---|---|---|
| SDForger | read-now | 原文 novelty、datasets、metrics、baselines 是什么？HAR 适配还剩什么空间？ | RQ-A/RQ-B scope |
| TS generation benchmark source | read-now | synthetic time-series quality/utility 应该用哪些 metrics？ | 最小评估计划 |
| AgentSense | read-now | HAR synthetic sensor data 的证据标准是什么？ | 应用定位 |
| TimeGAN / TimeVAE / TimeVQVAE / diffusion family | read-now | 哪些 baseline 是必须的，哪些只能 cite-only？ | baseline policy |
| ChatTS | skim/read-now | 项目到底是 generation 还是 understanding？ | scope control |
| PAMAP2 / HAR classifier papers | read-now | 常用 split、labels、classifiers 和 metrics 是什么？ | downstream utility protocol |

## 可能的新 angle

当前最稳妥的 angle：

> 对 LLM-enabled synthetic generation 在 wearable HAR time series 上做严谨的 adaptation 和 diagnostic evaluation，重点说明 text/embedding generation pipeline 保留了哪些统计、时间和下游性质。

这比“我们在 PAMAP2 上跑了 SDForger”更强，也比“提出完整新方法”更现实。

## 远程恢复后的最小实验计划

当前不要执行。

1. 核验代码状态：
   - local commit/diff
   - remote path
   - SDForger source version
   - task YAMLs 和 Slurm scripts
   - data preprocessing scripts 和 input checksums
2. 重建一个干净的 PAMAP2 pipeline：
   - 一个 subject
   - 固定 6-channel acceleration subset，或清楚记录修订后的 subset
   - 一个 baseline SDForger config
   - short smoke run，然后一个 formal run
3. 产出 metric package：
   - valid sample count
   - per-channel mean/std/min/max/quantiles
   - ACF
   - PSD
   - diversity 或 nearest-neighbor similarity
   - qualitative plots
4. 加一个 comparison：
   - 已核验旧 config comparison，或
   - feasible 的 naive augmentation/resampling baseline
5. 之后才考虑：
   - multi-subject validation
   - label-conditioned generation
   - downstream HAR classifier utility

## Reference workflow

- 原始 PDFs 和 source bundles：`reference/papers/` 或 `reference/sources/`
- source indexes 和 processing state：`reference/.agent/`
- structured source cards：`reference/cards/`
- project implications：`reference/project-use/`
- 临时 extraction logs：`reference/.agent/runs/`

下一步 source cards：

- `sdforger-2025-forging-time-series.md`
- `chatts-2025-aligning-time-series.md`
- `agentsense-2025-virtual-sensor-data.md`
- `project-notes-2026-ts-llm.md`

## 下一步

- ACT-001：收集缺失用户材料。
- ACT-005：完善 seed literature cards 和 map。
- ACT-003：写最小 metric suite。
- ACT-006：维护结束后核验远程 artifacts。

核心目标的idea应该就是完成这个任务：使用 LLM-enabled 或 LLM-adjacent 的生成方法，为 HAR 生成合成传感器时间序列数据。但是怎么完成，使用什么pipline，还不清楚