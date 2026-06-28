# Source Card：SDForger / Forging Time Series with Language

## Metadata

- Title: Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation
- Authors: Cecile Rousseau, Tobia Boschi, Giandomenico Cornacchia, Dhaval Salwala, Alessandra Pascale, Juan Bernabe Moreno
- Year: 2025
- Venue/status: arXiv preprint, status 需要后续核验
- DOI/arXiv/URL: arXiv:2505.17103
- Source PDF: `reference/papers/Forging Time Series Synthetic Data.pdf`
- Source type: paper-or-pdf
- Citation key: `rousseau2025sdforger`

## Reading Setup

- Reading mode: extract-method + extract-baseline + extract-benchmark + extract-risk
- Model tier: main model
- Reader: Codex
- Date: 2026-05-18
- Confidence: medium
- Source run: `reference/.agent/runs/2026-05-18-source-cards/sdforger.txt`

## Role Labels

- method-source: yes
- benchmark-source: yes
- baseline: yes
- citation-support: yes
- closest-work: yes
- reviewer-risk: high

## Summary

### Problem

现有 time-series synthetic data 方法通常需要为每个数据集从头训练，可能难以处理长序列、多变量依赖和低数据场景。论文尝试把 pretrained language model 的生成能力迁移到 time-series synthetic data generation。

### Main idea

SDForger 的核心 pipeline：

1. 对原始时间序列做 periodicity-aware segmentation，得到多个窗口。
2. 用 FastICA 或 Functional Principal Components (FPC) 将每个窗口投影成低维 tabular embeddings。
3. 把 embedding table 的每一行编码成 structured text prompt。
4. fine-tune 一个 autoregressive LLM，例如 GPT-2。
5. LLM 生成新的 textual embedding values。
6. 从生成 embedding decode 回 synthetic time series。

关键设计点：

- 用 ICA/FPC 让生成发生在 compact embedding space，而不是直接生成很长原始序列。
- 文本编码使用 fill-in-the-middle style prompt。
- 对 feature order 做 random permutation，降低文本位置顺序带来的偏置。
- 生成阶段过滤 NaN、重复样本和明显偏离的 embedding。
- 对 multivariate setting，目标是同时生成多个 interdependent channels。

### Main contribution

- 提出一种把 time series 变成 tabular/textual embeddings 后由 LLM 生成的框架。
- 覆盖 multisample、univariate、multivariate 三类生成设置。
- 与 VAE、VQ-VAE、GAN、SDE、diffusion 类 time-series generators 做比较。
- 使用 similarity metrics 和 downstream utility metrics 评估生成质量。

## Method Details

### 输入和数据形状

论文形式化为：给定 `I` 个 multivariate time-series instances，每个 instance 有 `C` 个 channels、长度 `L`，生成新的 `I_tilde` 个同形状 instances。

如果只有一条长序列，则先切成多个 windows，用这些 windows 估计数据分布。

### Embedding

论文使用两类 basis decomposition：

- FPC: 捕捉最大方差方向，保留相关结构。
- FastICA: 提取统计独立成分，强调 non-Gaussian latent factors。

每个 channel 单独得到 embedding coefficients，再把各 channel embeddings 拼接成 tabular row。

### Text generation

每个 embedding row 被转成文本。模型看到类似 “value_k is [blank]” 的输入和对应数值 target。生成时模型输出新的 embedding values。

### Filtering

生成后先在 embedding space 过滤：

- 缺失或无法解析的值
- 重复实例
- 明显偏离训练 embedding 分布的实例

这对当前项目很重要：如果 PAMAP2 生成样本大量被过滤，可能说明模型没有学到稳定的 embedding distribution，而不只是“样本数少”。

## Baselines

论文比较的 baseline families：

- VAE 类：TimeVAE, TimeVQVAE
- GAN 类：RTSGAN
- SDE/GAN 类：SDEGAN
- Diffusion 类：LS4

对当前项目的 baseline 启发：

- 如果只是课程项目，不一定全部实现，但至少要解释为什么选择 citation-only 或 lightweight baseline。
- 最低限度应有一个 naive baseline，例如 resampling/noise/jitter/window bootstrap。
- 如果要写成更研究化的 report，TimeVAE 或 TimeGAN/TimeVAE family 是比较自然的 baseline anchor。

## Datasets

论文使用 12 个公开 time-series datasets，覆盖 energy、transport、industry、weather、finance 等领域。它不是 HAR/PAMAP2-focused。

重要点：

- 论文包含 multivariate setting，但不等于已经证明 wearable HAR sensor data 有效。
- 如果原文没有覆盖 PAMAP2、IMU 或 activity recognition，本项目的 domain adaptation + diagnostic evaluation 有空间。

## Evaluation

论文把 evaluation 分成两类：

### Similarity metrics

Feature-based:

- Marginal Distribution Difference (MDD)
- Auto-Correlation Difference (ACD)
- Skewness Difference (SD)
- Kurtosis Difference (KD)

Distance-based:

- Euclidean Distance (ED)
- Dynamic Time Warping (DTW)
- SHAP-RE / shapelet-based reconstruction error

### Utility metrics

论文用 Tiny Time Mixers (TTM) 做 downstream forecasting，比较：

1. zero-shot
2. real data only
3. synthetic data only
4. real + synthetic

对当前项目的直接启发：

- 你的 PAMAP2 评估可以借鉴 similarity metrics，但 downstream task 应从 forecasting 改为 HAR classification。
- ACD/ACF、distribution statistics、DTW 很适合先做最小评估。
- 如果不做 downstream HAR utility，项目仍可作为 empirical evaluation，但 claim 要收窄。

## Limitations / Risks

- 原文主要展示 general time-series generation，不直接解决 HAR label-aware generation。
- 原文 utility 是 forecasting，不是 activity classification。
- SDForger 在 PAMAP2 上是否保留 class-discriminative activity patterns 仍未知。
- 过滤机制可能造成“能生成但多样性不足”的问题，需要单独报告 rejection/retention rate。
- 如果只复现 SDForger 并换数据集，novelty 会偏弱。

## 对当前项目选题的启发

最适合的项目路线不是“提出新生成方法”，而是：

> 以 SDForger 为代表，系统诊断 LLM-enabled synthetic time-series generation 在 wearable HAR sensor data 上保留和丢失了什么。

建议最小实验包：

- SDForger on PAMAP2 one-subject / fixed channel subset
- real vs synthetic distribution statistics
- ACF/ACD
- DTW 或 shape-based comparison
- diversity / duplicate / rejection rate
- 如果 labels 可用，再做 HAR classifier real-only vs real+synthetic

## Claims To Avoid

- 不要直接说 SDForger 适合 HAR。
- 不要在没有 downstream HAR classification 前声称 synthetic data 有助于 HAR。
- 不要只用 generated sample count 当质量指标。

## Provenance

- Pages/sections inspected: Abstract, Methodology, Evaluation methodology, Appendix B metrics, Appendix C datasets.
- Extraction method: local `pdftotext -layout` + targeted section reading.
- Reviewed by: agent only.
