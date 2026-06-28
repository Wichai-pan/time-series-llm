# Source Card：ChatTS

## Metadata

- Title: ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning
- Authors: Zhe Xie, Zeyan Li, Xiao He, Longlong Xu, Xidao Wen, Tieying Zhang, Jianjun Chen, Rui Shi, Dan Pei
- Year: 2025
- Venue/status: arXiv preprint / conference status 需要后续核验
- DOI/arXiv/URL: arXiv:2412.03104
- Source PDF: `reference/papers/ChatTS Time Series Alignment.pdf`
- Source type: paper-or-pdf
- Citation key: `xie2025chatts`

## Reading Setup

- Reading mode: extract-method + extract-benchmark + extract-baseline + extract-risk
- Model tier: main model
- Reader: Codex
- Date: 2026-05-18
- Confidence: medium
- Source run: `reference/.agent/runs/2026-05-18-source-cards/chatts.txt`

## Role Labels

- method-source: adjacent
- benchmark-source: yes, but for TS understanding/reasoning
- baseline: yes, for LLM/MLLM comparison
- citation-support: yes
- closest-work: no for generation; adjacent for LLM+time-series
- reviewer-risk: medium

## Summary

### Problem

论文关注的是 time series understanding and reasoning。作者认为普通 text-based LLM、vision-based MLLM 和 agent-based tool-use 方法都不能很好地理解多变量时间序列，尤其在全局/局部模式、数值细节、多变量关系上有缺陷。

### Main idea

ChatTS 是一个 time-series multimodal LLM。它把 time series 当作类似图像的 modality 输入给 LLM，用合成 time-series + text 数据训练模型，使模型能够回答关于趋势、周期、噪声、局部波动、多变量相关性和推理的问题。

核心组件：

1. Attribute-Based Time Series Generator：生成带有明确属性描述的 synthetic time series。
2. Time Series Evol-Instruct (TSEvol)：生成更复杂、多样的 time-series Q&A。
3. Context-aware time-series encoder：把 variable-length、multivariate time series 编码进 LLM。
4. 两阶段训练：large-scale alignment training + supervised fine-tuning。

### Main contribution

- 用 synthetic data 训练 time-series MLLM。
- 构造 time-series alignment 和 reasoning evaluation tasks。
- 证明 ChatTS 在理解/推理任务上优于 text-based、vision-based、agent-based baselines。

## Method Details

### Synthetic data

ChatTS 生成的是“带属性和问答的 time series”，不是为传统 ML classifier 直接生成 training samples。它关心的是：

- trend
- seasonality
- noise
- local fluctuation / spike
- multivariate correlation / clustering
- physical meaning 或 causal/reasoning QA

这些 synthetic series 的用途是训练 MLLM 对 time series 与语言进行对齐。

### Training

训练数据包括：

- UTS: univariate time-series attribute tasks
- MTS-Shape: multivariate global trend/correlation tasks
- MTS-Local: multivariate local fluctuation tasks
- TSEvol: reasoning-oriented Q&A
- instruction-following data

基础模型是 Qwen2.5-14B-Instruct，训练方式包括 alignment training 和 SFT。

## Baselines

ChatTS 的 baseline 是不同输入方式的 LLM/MLLM：

- Text-Based: 把 time-series arrays 作为文本 prompt 给 GPT-4o、GPT-4o-mini、GPT-4-Turbo、Qwen2.5-14B。
- Vision-Based: 把 time-series plot 成图，再给 GPT-4o/GPT-4o-mini vision model。
- Agent-Based: ReAct agent 使用 time-series analysis tools，例如 query、STL decomposition、anomaly detection、classification、correlation tools。

这和当前项目的 baseline 不同。当前项目做 synthetic data generation for HAR，baseline 应该是 time-series generators 或 HAR augmentation methods，而不是 ChatTS 的 LLM input baselines。

## Datasets

Evaluation datasets 包括：

- Dataset A: real-world time series，来自 AIOps、weather、NAB、Oracle system metrics 等，525 questions。
- Dataset B: generated time series + templates/LLM-generated QA，1616 questions。
- MCQ2: open-source comparison reasoning tasks。

这些不是 PAMAP2/HAR datasets。

## Evaluation

Evaluation tasks：

- Alignment tasks:
  - univariate: trend, seasonality, noise, local fluctuation
  - multivariate: correlation, clustering
- Reasoning tasks:
  - inductive reasoning
  - deductive reasoning
  - causal reasoning
  - comparison reasoning

Metrics：

- categorical tasks: F1-score
- numerical tasks: relative accuracy
- Q&A inductive reasoning: RAGAS / LLM-based fuzzy matching
- T/F and multiple choice: accuracy
- efficiency: token consumption / cost

结果上，ChatTS 在 alignment 和 reasoning tasks 上明显超过 text、vision、agent baselines。

## Limitations / Risks

- ChatTS 的核心任务不是 synthetic sensor data generation for classifier training。
- 它证明 synthetic time series + text 可以训练 MLLM 理解时间序列，但不能直接证明 synthetic data 对 HAR classifier 有用。
- 它的数据和任务偏 AIOps/weather/system metrics，不是 wearable HAR。
- 对当前项目而言，它是 scope boundary：不要把项目混成“LLMs for time series understanding”。

## 对当前项目选题的启发

ChatTS 最有价值的启发不是 baseline，而是两个思路：

1. **属性描述/条件生成**  
   如果未来要做 activity-conditioned generation，可以借鉴“先定义属性，再生成与属性一致的序列”的思路。

2. **评估不能只看曲线**  
   ChatTS 强调 trend、seasonality、local fluctuation、multivariate correlation 等属性。当前项目可以把这些变成 PAMAP2 的 diagnostic questions：生成数据是否保留活动相关的局部波动、周期性、跨通道相关性？

当前阶段建议：

- ChatTS 作为 related work 和 scope boundary。
- 不把 ChatTS 当作必须复现的 pipeline。
- 不把项目改成 time-series MLLM reasoning，除非后续明确 pivot。

## Claims To Avoid

- 不要说 ChatTS 是 HAR synthetic-data baseline。
- 不要把 ChatTS 的 reasoning performance 当作 synthetic data quality evidence。
- 不要把当前项目描述为 time-series understanding，除非选题正式 pivot。

## Provenance

- Pages/sections inspected: Abstract, Introduction, Methodology, Training datasets, Evaluation tasks, Baselines, Results.
- Extraction method: local `pdftotext -layout` + targeted section reading.
- Reviewed by: agent only.
