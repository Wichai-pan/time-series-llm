# Source Card：Unified Evaluation Framework For Human Motion Generation

## Metadata

- Title: Establishing a unified evaluation framework for human motion generation: A comparative analysis of metrics
- Authors: A. Ismail-Fawaz, M. Devanne, S. Berretti, and coauthors
- Year: 2025
- Venue/status: Computer Vision and Image Understanding
- DOI/URL: https://www.sciencedirect.com/science/article/pii/S1077314225000608
- Source PDF: `reference/papers/Ismail-Fawaz 等 - 2025 - Establishing a unified evaluation framework for human motion generation A comparative analysis of m.pdf`
- Source type: paper-or-pdf
- Citation key: `ismailfawaz2025humanmotionevaluation`

## Reading Setup

- Reading mode: extract-benchmark + extract-risk
- Reader: Codex
- Date: 2026-05-18
- Extracted text: `reference/.agent/runs/2026-05-18-eval-papers/human-motion-eval.txt`
- Confidence: medium-high for metric categories and implications

## Role Labels

- metric-source: yes
- benchmark-source: yes
- human-motion-source: yes
- reviewer-risk: high
- citation-support: yes

## Summary

This paper reviews and standardizes evaluation metrics for human motion generation. It is not specifically about wearable IMU signals, but it is very relevant because PAMAP2/MM-Fit are human motion sensor datasets and the teacher's concern is exactly motion quality: generated samples may have acceptable RMSE/DTW while still failing to preserve motion rhythm, temporal pattern, or class-relevant dynamics.

The paper's central message is:

> Human motion generation must be evaluated along multiple dimensions, mainly fidelity and diversity, and no single metric is sufficient.

## Evaluation Setup

The paper describes a common setup:

1. Train a supervised model on real data for a task, usually classification.
2. Use the model's latent representation to embed real and generated samples.
3. Compute fidelity/diversity metrics in that latent space.

This is highly relevant to HAR: train an activity classifier on real PAMAP2/MM-Fit data, use either its predictions or internal features to evaluate generated samples.

## Metrics Reviewed

### Fidelity Metrics

- FID: distributional distance in latent feature space.
- Accuracy On Generated (AOG): a classifier trained on real data predicts labels on generated samples; high accuracy indicates label-consistent generated samples.
- Density: estimates whether generated samples lie in dense regions of the real-data manifold.
- Precision: discussed but less preferred than density in the paper's table.

### Diversity Metrics

- Average Pair Distance (APD): average pairwise distance among generated samples; should be interpreted relative to real-data APD.
- Average per Class Pair Distance (ACPD): class-aware diversity metric for labeled data.
- Coverage: measures how much of the real-data manifold is covered by generated samples.
- Recall: discussed but less preferred than coverage in the paper's table.
- MMS: novelty/diversity based on nearest-neighbor distances.

### Temporal-Distortion Metric

- Warping Path Diversity (WPD): proposed by the paper to measure diversity in temporal distortion using DTW warping paths.

This is especially relevant to the teacher's point about periodicity, rhythm, time shift, and frequency changes. Latent-space diversity metrics may miss whether generated motion has realistic temporal warping or rhythm variety.

## Important Interpretations

FID:

- Lower is usually better, but perfect FID can also happen if a model copies real samples.
- FID should be interpreted relative to a real-vs-real baseline.

AOG / classifier accuracy:

- Useful for checking whether generated samples preserve class label semantics.
- A high classifier accuracy alone can be misleading if generated samples contain artifacts that do not affect the classifier.

Diversity:

- More diversity is not always better.
- Generated diversity should be close to real-data diversity, not artificially high.

Coverage:

- Useful for detecting mode collapse: whether generated samples cover many real motion modes rather than only a small subset.

WPD:

- Useful when temporal distortion, rhythm, and frequency changes matter.
- Strong candidate for a motion-specific diagnostic, though it may be more complex than needed for the first project milestone.

## Relevance To Current Project

The paper directly supports adding a motion-aware evaluation layer beyond RMSE/DTW:

- classifier-based class consistency
- FID-like distance in HAR classifier feature space
- density / coverage in feature space
- generated diversity compared with real diversity
- temporal distortion / warping diversity

For PAMAP2, a practical adaptation would be:

1. Train a real-data HAR classifier on train subjects.
2. Use classifier accuracy on generated samples as class-consistency check.
3. Extract classifier embeddings for real test and synthetic samples.
4. Compute feature-space FID or MMD.
5. Compute coverage/diversity against real test or real train windows.
6. Compute ACF/PSD/FFT and DTW as raw-signal diagnostics.

For MM-Fit teammate pipeline, this paper is even more directly relevant because the dataset is exercise/human-motion generation. It supports the teammate's current metrics such as classifier accuracy, FID-like distribution metrics, diversity, coverage, and DTW, but also demands clean artifacts and real baselines.

## Relation To Teacher Feedback

The teacher suggested:

- downstream task evaluation: train on generated data and test on real data
- distribution-based metrics such as FID
- feature-based metrics such as autocorrelation difference
- frequency-domain comparison such as FFT / PSD

This paper supports the same direction, especially:

- task classifier as semantic/motion relevance check
- FID-like feature-space distribution comparison
- diversity and coverage metrics
- temporal distortion checks when rhythm or periodicity matters

## Risks / Claims To Avoid

- Do not conclude generated motion is good from RMSE/DTW alone.
- Do not conclude generated motion is good from classifier accuracy alone.
- Do not claim high diversity is good unless compared to real-data diversity.
- Do not compute FID in an arbitrary feature space without describing the feature extractor.
- Do not compare metrics without a real-vs-real baseline.

## Reusable Evaluation Ideas

For current project:

- `AOG`: train HAR classifier on real data, evaluate generated samples against intended activity labels.
- `TSTR`: train HAR classifier on synthetic samples, test on real held-out subjects.
- `TRTS / augmentation`: train real-only vs real+synthetic, test on real held-out subjects.
- Feature FID/MMD: use real-trained HAR classifier penultimate features.
- Coverage: nearest-neighbor coverage in classifier-feature space.
- WPD or simplified DTW-path summary: check whether generated windows have realistic temporal warping/rhythm variation.

## Provenance

- Pages/sections inspected: abstract, introduction, metric definitions, summary table, WPD section, conclusion/references.
- Extraction method: local `pdftotext -layout` and targeted section search.
- Reviewed by: agent only.
