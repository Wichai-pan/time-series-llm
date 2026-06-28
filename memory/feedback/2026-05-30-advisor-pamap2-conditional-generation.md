# Feedback Record

## Context

- Date: 2026-05-30
- Source: advisor
- Artifact reviewed: PAMAP2 / SDForger activity-conditioned generation progress
- Meeting or channel: advisor discussion, user notes

## Triage Table

| # | Item | Target | Tone | Priority | Affects submission |
|---|---|---|---|---|---|
| 1 | Different channels may have inconsistent amplitudes; for unified generation, try normalization before embedding. | data / method | suggestion | should-address | unclear |
| 2 | Evaluation should clarify whether data is unseen; consider training/evaluating across different subjects to increase data and test generalization. | evaluation / data split | concern | must-address | yes |
| 3 | Clarify what `clip_p05_p95` actually does and why generated curves may look smoother. | method / interpretation | question | should-address | unclear |

## Memory Updates

### Claims changed

- Current activity-conditioned generation claim should remain diagnostic. Advisor feedback reinforces that current subject101 single-channel results do not establish unseen-subject generalization or robust multichannel generation.

### Risks added or updated

- RSK-025: Channel amplitude mismatch may destabilize unified / multichannel generation.
- RSK-026: Current evaluation may not be unseen-subject evaluation.
- RSK-027: Clipping can smooth or truncate generated dynamics, so visual improvement may hide loss of amplitude/detail.

### Actions added

- ACT-031: Design pre-embedding normalization ablation for unified/multichannel SDForger.
- ACT-032: Define unseen-subject evaluation protocol.
- ACT-033: Explain and quantify clipping effect, including smoothness/amplitude tradeoff.

### Decisions recorded

- None yet. These notes define follow-up experiments, not final design decisions.

### Evidence gaps identified

- Missing evidence for multichannel amplitude normalization.
- Missing evidence for unseen-subject generalization.
- Missing evidence separating clipping-induced stability from over-smoothing.

## Disagreements Logged

None.

## Follow-Up Actions

| Action | Owner | Due | Linked to |
|---|---|---|---|
| Pre-embedding normalization ablation | agent | next experiment planning | ACT-031 |
| Unseen-subject split design | agent + user | next experiment planning | ACT-032 |
| Clipping explanation and smoothness diagnostics | agent | next result note | ACT-033 |

## Open Questions

- Should the next experiment prioritize single-channel multi-subject generalization or multichannel normalization?
- For multichannel setting, should normalization be per-channel, per-subject-per-channel, per-activity-per-channel, or global train-stat based?
- Should generated samples be inverse-transformed to raw-like space for HAR utility, or should utility be evaluated in a standardized space?
