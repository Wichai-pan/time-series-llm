# Source Card：TSGBench / Time Series Generation Benchmark

## Metadata

- Title: TSGBench: Time Series Generation Benchmark
- Authors: Yihao Ang, Qiang Huang, Yifan Bao, Anthony K. H. Tung, Zhiyong Huang
- Year: 2023 / VLDB 2024 proceedings metadata in PDF
- Venue/status: PVLDB / benchmark paper
- DOI/URL: https://dl.acm.org/doi/abs/10.14778/3632093.3632097
- Source PDF: `reference/papers/Ang 等 - 2023 - TSGBench Time Series Generation Benchmark.pdf`
- Source type: paper-or-pdf
- Citation key: `ang2023tsgbench`

## Reading Setup

- Reading mode: extract-benchmark + extract-baseline + extract-risk
- Reader: Codex
- Date: 2026-05-18
- Extracted text: `reference/.agent/runs/2026-05-18-eval-papers/tsgbench.txt`
- Confidence: medium-high for evaluation protocol; medium for exact formulas due PDF text extraction limits

## Role Labels

- benchmark-source: yes
- metric-source: yes
- baseline-source: yes
- reviewer-risk: high
- citation-support: yes

## Summary

TSGBench is a benchmark for time-series generation (TSG). It argues that TSG evaluation needs a standard dataset/preprocessing setup, a broad metric suite, and generalization testing rather than one-off visual or downstream metrics.

The benchmark organizes evaluation around three concerns:

- diversity
- fidelity
- usefulness

For the current SDForger/PAMAP2 project, the most important message is that no single metric is enough. Point-wise distance metrics such as ED/RMSE and alignment metrics such as DTW need to be combined with feature-based, distribution-based, model-based, and visualization checks.

## Benchmark Components

TSGBench contains:

1. Real-world datasets with standardized preprocessing.
2. A suite of 12 evaluation measures.
3. Generalization tests based on domain adaptation.
4. Comparisons across multiple TSG methods and datasets.

It includes HAPT, an inertial HAR dataset, which makes it relevant to wearable sensor generation even though it is not directly PAMAP2-focused.

## Evaluation Metrics

### Model-Based Measures

- Discriminative Score (DS): train a model to distinguish real from generated time series.
- Predictive Score (PS): train a prediction model on generated data and test predictive performance on real data.
- Contextual-FID (C-FID): FID-style representation metric for time series context.

Important caution: the paper notes DS and PS can be unstable because they depend on post-hoc neural model training, architecture, random initialization, dataset size, and sequence length.

### Feature-Based Measures

- Marginal Distribution Difference (MDD): compares empirical histograms for original and generated series across dimensions/time steps.
- AutoCorrelation Difference (ACD): compares autocorrelation structure.
- Skewness Difference (SD): compares distribution asymmetry.
- Kurtosis Difference (KD): compares tail behavior.

These are deterministic and easier to interpret than model-based metrics. They are directly useful for the current project's minimum metric suite.

### Distance-Based Measures

- Euclidean Distance (ED): point-wise value similarity.
- Dynamic Time Warping (DTW): alignment-aware temporal similarity.

The paper frames ED/DTW as efficient deterministic alternatives to unstable post-hoc model scores, but they should not be used alone.

### Visualization

- t-SNE plots.
- Distribution plots.

These are not sufficient evidence alone, but useful for diagnosis and reporting.

## Baselines And Method Families

TSGBench evaluates or discusses common TSG method families:

- GAN-based: RGAN, TimeGAN, RTSGAN, COSCI-GAN, GT-GAN, AEC-GAN
- VAE-based: TimeVAE, TimeVQVAE
- Flow-based: Fourier Flow
- Diffusion / SDE-related: LS4, SDEGAN

Project implication: if the current project claims more than adaptation/diagnosis, at least one conventional time-series generator family should be cited or compared. For a course project, a lightweight baseline such as bootstrap/jitter plus citation-only discussion of TimeVAE/TimeGAN may be acceptable if scope is explicit.

## Generalization Test

TSGBench proposes domain-adaptation-style tests:

- Single DA: train on source domain and generate target.
- Cross DA: train on source plus small target subset and generate target.
- Reference DA: train only on small target subset.

For HAR, subject identity can act as domain. This is useful for PAMAP2:

- train subjects as source
- held-out subject as target/test
- evaluate whether synthetic samples preserve target-like motion dynamics

## Recommendations From The Paper

The paper recommends:

- start with VAE-based methods such as TimeVAE and LS4 for initial exploration due performance/efficiency
- use ACD when autocorrelation or forecasting-like temporal dependency matters
- use feature-based measures for statistical attributes
- use distance-based metrics when clustering/similarity is central
- choose metrics based on the application rather than expecting all metrics to agree

## Relevance To Current Project

This paper strongly supports the teacher's advice:

- Do downstream task evaluation, but do not rely on it alone.
- Add distribution-based metrics such as FID/C-FID or MDD.
- Add feature-based metrics such as ACD, SD, KD.
- Add temporal/frequency metrics beyond RMSE/DTW.
- Report metric disagreement as useful diagnosis, not just a problem.

Recommended minimum PAMAP2 evaluation suite inspired by TSGBench:

- retention / parse success / duplicate rate
- MDD with histogram-based implementation
- ACD / ACF difference
- skewness and kurtosis difference
- FFT or PSD comparison
- ED/RMSE and DTW
- real-vs-synthetic discriminative classifier or representation FID if feasible
- HAR downstream classifier: synthetic-only and real+synthetic tested on real held-out subjects

## Risks / Claims To Avoid

- Do not claim generated data is good because RMSE or DTW is low.
- Do not use one metric as final quality.
- Do not call a global mean difference "MDD"; TSGBench defines MDD as histogram-based marginal distribution comparison.
- Do not treat downstream model scores as fully stable unless repeated seeds or confidence intervals are used.

## Provenance

- Pages/sections inspected: abstract, evaluation measures, metric descriptions, generalization test, recommendations.
- Extraction method: local `pdftotext -layout` and targeted section search.
- Reviewed by: agent only.
